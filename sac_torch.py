# sac_torch.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

LOG_2PI = math.log(2.0 * math.pi)

def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
  
    def __init__(self, state_dim, action_dim, action_low, action_high,
                 hidden_sizes=(256, 256), log_std_min=-20.0, log_std_max=2.0, device="cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = torch.device(device)

        self.net = mlp([state_dim, hidden_sizes[0], hidden_sizes[1], 2 * action_dim])

        action_low  = torch.as_tensor(np.asarray(action_low, dtype=np.float32), device=self.device)
        action_high = torch.as_tensor(np.asarray(action_high, dtype=np.float32), device=self.device)
        self.register_buffer("scale", (action_high - action_low) / 2.0)
        self.register_buffer("bias",  (action_high + action_low) / 2.0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def _forward_mu_logstd(self, state: Tensor):
        h = self.net(state)
        mu, log_std = torch.chunk(h, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    @staticmethod
    def _gaussian_log_prob(u, mu, log_std):
        
        # log N(u; mu, std) = -0.5 * [ ((u-mu)/std)^2 + 2*log_std + log(2π) ]
        pre_sum = -0.5 * (((u - mu) * torch.exp(-log_std))**2 + 2.0*log_std + LOG_2PI)
        return pre_sum.sum(dim=-1, keepdim=True)

    @staticmethod
    def _tanh_correction(u):
        # log |det J_tanh(u)| = sum log(1 - tanh(u)^2)
        a = torch.tanh(u)
        return torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    def sample(self, state: Tensor):
        
        mu, log_std = self._forward_mu_logstd(state)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        u = mu + std * eps
        a_tanh = torch.tanh(u)

        base_logp = self._gaussian_log_prob(u, mu, log_std)
        corr = self._tanh_correction(u)
        logp = base_logp - corr

        action = a_tanh * self.scale + self.bias
        return action, logp

    @torch.no_grad()
    def act(self, state: Tensor, deterministic=True):
       
        mu, log_std = self._forward_mu_logstd(state)
        if deterministic:
            a_tanh = torch.tanh(mu)
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            a_tanh = torch.tanh(mu + std * eps)
        return a_tanh * self.scale + self.bias


class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.q = mlp([state_dim + action_dim, hidden[0], hidden[1], 1])

    def forward(self, s: Tensor, a: Tensor):
        x = torch.cat([s, a], dim=-1)
        return self.q(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


class SAC(nn.Module):
    
    def __init__(self, state_dim, action_dim, action_low, action_high, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim, action_low, action_high, device=self.device)
        self.q1 = Critic(state_dim, action_dim).to(self.device)
        self.q2 = Critic(state_dim, action_dim).to(self.device)

        self.q1_tgt = Critic(state_dim, action_dim).to(self.device)
        self.q2_tgt = Critic(state_dim, action_dim).to(self.device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())

        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4)
        self.alpha_opt  = torch.optim.Adam([{'params':[torch.zeros(())]}], lr=3e-4)  # dummy; replaced below

        # temperature (log_alpha as a Parameter so it gets gradients)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(action_dim)

        self.gamma = 0.99
        self.tau   = 0.005
        self.mse   = nn.MSELoss()
        self.actor.to(device)
        self.q1.to(device)
        self.q2.to(device)
        self.q1_tgt.to(device)
        self.q2_tgt.to(device)

        
    def critics_train_step(self, s, a, r, s2, d):
        
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_tp1 = self.q1_tgt(s2, a2)
            q2_tp1 = self.q2_tgt(s2, a2)
            min_q_tp1 = torch.min(q1_tp1, q2_tp1)
            alpha = self.log_alpha.exp()
            v_tgt = min_q_tp1 - alpha * logp2
            y = r + self.gamma * (1.0 - d) * v_tgt

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        loss_q1 = self.mse(q1_pred, y)
        loss_q2 = self.mse(q2_pred, y)
        loss_q = loss_q1 + loss_q2

        self.critic_opt.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=10.0)
        self.critic_opt.step()
        return loss_q.item(), q1_pred.mean().item(), q2_pred.mean().item()

    def actor_train_step(self, s):
        
        alpha = self.log_alpha.exp()
        a_pi, logp_pi = self.actor.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        min_q = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha * logp_pi - min_q).mean()

        self.actor.optimizer.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor.optimizer.step()

        return loss_pi.item(), logp_pi.mean().item(), min_q.mean().item()

    def alpha_train_step(self, s):
        
        with torch.no_grad():
            _, logp = self.actor.sample(s)
        loss_alpha = -(self.log_alpha * (logp + self.target_entropy)).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.alpha_opt.step()
        return loss_alpha.item()

    @torch.no_grad()
    def update_targets(self):
        soft_update(self.q1_tgt, self.q1, self.tau)
        soft_update(self.q2_tgt, self.q2, self.tau)

    def train_on_batch(self, batch):
        
        s, a, r, s2, d = [t.to(self.device) for t in batch]

        loss_q, q1m, q2m = self.critics_train_step(s, a, r, s2, d)
        loss_pi, logp_m, minq_m = self.actor_train_step(s)
        loss_alpha = self.alpha_train_step(s)
        self.update_targets()

        return {
            "loss_q": loss_q,
            "loss_pi": loss_pi,
            "loss_alpha": loss_alpha,
            "alpha": float(self.log_alpha.exp().detach().cpu().item()),
            "q1_mean": q1m,
            "q2_mean": q2m,
            "logp_mean": logp_m,
            "minq_mean": minq_m,
        }
