import torch as th

from modules.critics.ac_ns import ACCriticNS


class ACCriticExperienceSharing(ACCriticNS):
    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, i])
            qs.append(q.view(bs * self.n_agents, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q.reshape(self.n_agents, bs, max_t, self.n_agents, -1)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = batch["obs"][:, ts]
        # inputs: (batch_size, ep_length + 1, n_agents, obs_shape)
        inputs = (
            inputs.unsqueeze(0)
            .repeat(self.n_agents, 1, 1, 1, 1)
            .reshape(bs * self.n_agents, *inputs.shape[1:])
        )
        return inputs, bs, max_t
