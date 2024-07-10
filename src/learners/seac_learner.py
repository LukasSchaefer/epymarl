from einops import rearrange
import torch as th

from components.episode_buffer import EpisodeBatch
from learners.actor_critic_learner import ActorCriticLearner


class SEACLearner(ActorCriticLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        assert (
            args.mac == "exp_sharing_mac"
        ), "SEAC only works with the experience sharing MAC (mac='exp_sharing_mac')"
        assert (
            args.critic_type == "ac_critic_es"
        ), "SEAC only works with the experience sharing critic (critic_type='ac_critic_es')"
        assert (
            args.agent == "rnn_ns"
        ), "SEAC assumes non-shared/ individual policy networks (agent='rnn_ns')"

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env)
            )
            return

        # for batch computation have mask for each agent
        mask = mask.repeat(1, 1, self.n_agents)

        # reshape/ repeat from (nstep, eplength, n_agents) to (n_agents, nstep, eplength, n_agents)
        # to match data of all agents being fed through each agent's network in single batch with
        # entry [i, :, :, :, :] being the data of agent i
        rewards = rearrange(
            rewards, "nstep eplength n_agents -> n_agents nstep eplength 1"
        ).repeat(1, 1, 1, self.n_agents)
        actions = rearrange(
            actions, "nstep eplength n_agents 1 -> n_agents nstep eplength 1 1"
        ).repeat(1, 1, 1, self.n_agents, 1)[:, :, :-1]
        mask = rearrange(
            mask, "nstep eplength n_agents -> n_agents nstep eplength 1"
        ).repeat(1, 1, 1, self.n_agents)
        
        critic_mask = mask.clone()

        mac_out = []
        # forward pass for each agent using data of all agents!
        self.mac.init_hidden(batch.batch_size * self.n_agents)
        for t in range(batch.max_seq_length - 1):
            # (n_agents, nstep, n_agents, n_actions)
            agent_outs = self.mac.forward(batch, t=t, all_agents=True)
            mac_out.append(agent_outs)
        # (n_agents, nstep, eplength, n_agents, n_actions) with
        # [i, :, :, j, :] being the output of agent j for the experience of agent i
        mac_out = th.stack(mac_out, dim=2)  # Concat over time

        # mask out terminated timesteps
        pi = mac_out
        pi[mask == 0] = 1.0

        # compute IS weights for SEAC
        probs_taken = th.gather(pi, dim=-1, index=actions).squeeze(-1)
        log_probs_taken = th.log(probs_taken + 1e-10)

        is_weights = []
        for i in range(self.n_agents):
            # log-prob of agent i to take action i in its own experience
            original_log_probs_taken = (
                log_probs_taken[i, :, :, i].unsqueeze(-1).repeat(1, 1, self.n_agents)
            )
            # IS weight to correct from agent i taken its experience to all other agents taking
            # the experience of agent i
            is_weights.append(th.exp(log_probs_taken[i] - original_log_probs_taken))
        is_weights = th.stack(is_weights, dim=0).detach()

        if self.args.seac_retrace_is:
            # use retrace IS weights / clip to avoid large weights
            is_weights_clipped = th.min(is_weights, th.ones_like(is_weights))
        else:
            is_weights_clipped = is_weights

        # matrix with weights for each experience/ agent
        # own experience weights: 1
        lambda_matrix = (
            th.eye(self.n_agents, device=batch.device)
            .reshape(self.n_agents, 1, 1, self.n_agents)
            .repeat(1, batch.batch_size, batch.max_seq_length - 1, 1)
        )
        # others' experience weights: is_weights * seac_lambda
        lambda_matrix += (1 - lambda_matrix) * is_weights_clipped * self.args.seac_lambda

        # sanity check / assert
        # for i in range(self.n_agents):
        #     assert th.equal(
        #         is_weights[i, :, :, i], th.ones_like(is_weights[i, :, :, i])
        #     ), is_weights[i, :, :, i]
        #     for j in range(self.n_agents):
        #         if i != j:
        #             assert not th.allclose(
        #                 is_weights[i, :, :, j], th.ones_like(is_weights[i, :, :, j])
        #             ), is_weights[i, :, :, j]

        # advantages: (n_agents (data shared), nstep, eplength, n_agents
        advantages, critic_train_stats = self.train_critic_sequential(
            self.critic, self.target_critic, batch, rewards, critic_mask, lambda_matrix
        )
        advantages = advantages.detach()

        # Calculate policy grad
        unmasked_pg_loss = -lambda_matrix * advantages * log_probs_taken
        masked_pg_loss = (unmasked_pg_loss * mask).sum() / mask.sum()

        # Compute entropy loss only over own experiences
        entropies = []
        entropy_masks = []
        for i in range(self.n_agents):
            entropies.append(
                -th.sum(pi[i, :, :, i] * th.log(pi[i, :, :, i] + 1e-10), dim=-1)
            )
            entropy_masks.append(mask[i, :, :, i])
        entropy = th.stack(entropies, dim=0)
        entropy_mask = th.stack(entropy_masks, dim=0)
        masked_entropy = (entropy * entropy_mask).sum() / entropy_mask.sum()

        pg_loss = masked_pg_loss - self.args.entropy_coef * masked_entropy

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.grad_norm_clip
        )
        self.agent_optimiser.step()

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("entropy", masked_entropy.item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "is_weight_mean",
                (is_weights * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "is_weight_min",
                is_weights.min().item(),
                t_env,
            )
            self.logger.log_stat(
                "is_weight_max",
                is_weights.max().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def train_critic_sequential(
        self, critic, target_critic, batch, rewards, mask, coefs
    ):
        # Optimise critic
        with th.no_grad():
            # (n_agents (data repetition), batch_size, ep_length, n_agents, 1)
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(-1)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :, :-1].squeeze(-1)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask * coefs
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )
        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        ep_length = rewards.size(2)
        nstep_values = th.zeros_like(values[:, :, :-1])
        for t_start in range(ep_length):
            nstep_return_t = th.zeros_like(values[:, :, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= ep_length:
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma**step * values[:, :, t] * mask[:, :, t]
                    )
                elif t == ep_length - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma**step * rewards[:, :, t] * mask[:, :, t]
                    )
                    nstep_return_t += (
                        self.args.gamma ** (step + 1) * values[:, :, t + 1]
                    )
                else:
                    nstep_return_t += (
                        self.args.gamma**step * rewards[:, :, t] * mask[:, :, t]
                    )
            nstep_values[:, :, t_start, :] = nstep_return_t
        return nstep_values
