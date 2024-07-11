import torch as th

from modules.critics.ac_ns import ACCriticNS


class ACCriticExperienceSharing(ACCriticNS):
    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, :, i])
            qs.append(q.view(bs, max_t, self.n_agents, -1))
        return th.stack(qs, dim=-2)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = batch["obs"][:, ts]
        # repeat inputs by agents to feed all agents' experience to all agents' critics
        # in a single batch
        # reshape from (batch_size, ep_length + 1, n_agents, obs_shape)
        # to (batch_size, ep_length + 1, n_agents, n_agents, obs_shape)
        inputs = inputs.unsqueeze(-2).repeat(1, 1, 1, self.n_agents, 1)
        return inputs, bs, max_t
