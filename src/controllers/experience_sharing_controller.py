from controllers.non_shared_controller import NonSharedMAC
import torch as th


class ExperienceSharingMAC(NonSharedMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # As implemented currently, sharing experiences with agent IDs would lead to wrong indices in the experiences
        # (agent i receiving experience of agent j with agent j's agent ID but using agent i's network)
        assert not self.args.obs_agent_id, "Experience sharing does not support agent id in observations (using individual networks)"

    def select_actions(
        self,
        ep_batch,
        t_ep,
        t_env,
        bs=slice(None),
        test_mode=False,
        all_agents=False,
    ):
        if not all_agents:
            # sample actions for own data only as done in NonSharedMAC
            return super().select_actions(ep_batch, t_ep, t_env, bs, test_mode)

        # Select actions for all agents in the batch
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(
            ep_batch, t_ep, test_mode=test_mode, all_agents=all_agents
        )
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, all_agents=False):
        if not all_agents:
            return super().forward(ep_batch, t, test_mode)

        # forward pass for data of all agents in the batch
        batch_size = ep_batch.batch_size * self.n_agents
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # repeat data for all agents in the batch
        agent_inputs = (
            agent_inputs.unsqueeze(0)
            .repeat(self.n_agents, 1, 1)
            .reshape(batch_size * self.n_agents, -1)
        )

        # for i in range(self.n_agents):
        #     for j in range(self.n_agents):
        #         assert th.equal(
        #             agent_inputs[i * batch_size : (i + 1) * batch_size],
        #             agent_inputs[j * batch_size : (j + 1) * batch_size],
        #         )

        avail_actions = avail_actions.repeat(self.n_agents, 1, 1, 1).reshape(
            self.n_agents, batch_size, -1
        )
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(self.n_agents, ep_batch.batch_size, self.n_agents, -1)
