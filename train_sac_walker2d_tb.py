# train_sac_walker2d_tb.py
import os, time, random
import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from sac_torch import SAC
from collections import deque


class Meter:
    def __init__(self, size=100):
        self.buf = deque(maxlen=size)
    def add(self, x): self.buf.append(float(x))
    @property
    def avg(self): return np.mean(self.buf) if self.buf else 0.0
    @property
    def last(self): return self.buf[-1] if self.buf else 0.0

ret_meter = Meter(20)
len_meter = Meter(20)
lq_meter  = Meter(100)
lpi_meter = Meter(100)
la_meter  = Meter(100)
alpha_meter = Meter(100)
q_meter   = Meter(100)
logp_meter= Meter(100)


SEED               = 42
ENV_ID             = "Walker2d-v5"        
TOTAL_STEPS        = 2_000_000
WARMUP_STEPS       = 10_000
TRAIN_EVERY        = 10
UPDATES_PER_STEP   = 1
BATCH_SIZE         = 512                   
EVAL_EVERY_STEPS   = 10_000
REPLAY_CAPACITY    = 1_000_000             
SAVE_EVERY         = 200_000
SAVE_DIR           = "checkpoints_torch_walker"
RESUME_DIR         = ""                   

LOG_DIR            = "runs/sac_walker2d"   

device = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e6)):
        self.s_buf  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a_buf  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r_buf  = np.zeros((capacity, 1), dtype=np.float32)
        self.s2_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d_buf  = np.zeros((capacity, 1), dtype=np.float32)
        self.cap = capacity
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, d):
        idx = self.ptr % self.cap
        self.s_buf[idx]  = s
        self.a_buf[idx]  = a
        self.r_buf[idx]  = r
        self.s2_buf[idx] = s2
        self.d_buf[idx]  = d
        self.ptr += 1
        self.size = min(self.size + 1, self.cap)

    def sample_torch(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        s   = torch.as_tensor(self.s_buf[idxs],  dtype=torch.float32, device=device)
        a   = torch.as_tensor(self.a_buf[idxs],  dtype=torch.float32, device=device)
        r   = torch.as_tensor(self.r_buf[idxs],  dtype=torch.float32, device=device)
        s2  = torch.as_tensor(self.s2_buf[idxs], dtype=torch.float32, device=device)
        d   = torch.as_tensor(self.d_buf[idxs],  dtype=torch.float32, device=device)
        return s, a, r, s2, d


def set_seed(env, seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass


env = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID)
set_seed(env, SEED); set_seed(eval_env, SEED+1)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
act_low    = env.action_space.low
act_high   = env.action_space.high


sac = SAC(state_dim, action_dim, act_low, act_high, device=device)


if RESUME_DIR and os.path.exists(RESUME_DIR):
    subdirs = [d for d in os.listdir(RESUME_DIR) if os.path.isdir(os.path.join(RESUME_DIR, d))]
    if subdirs:
        latest = sorted(
            subdirs,
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1
        )[-1]
        ckpt_path = os.path.join(RESUME_DIR, latest)
        print(f"🔍 Found checkpoint: {ckpt_path} — loading weights...")
        try:
            sac.actor.load_state_dict(torch.load(os.path.join(ckpt_path, "actor.pt"),   map_location=device))
            sac.q1.load_state_dict(   torch.load(os.path.join(ckpt_path, "q1.pt"),      map_location=device))
            sac.q2.load_state_dict(   torch.load(os.path.join(ckpt_path, "q2.pt"),      map_location=device))
            sac.q1_tgt.load_state_dict(torch.load(os.path.join(ckpt_path, "q1_tgt.pt"), map_location=device))
            sac.q2_tgt.load_state_dict(torch.load(os.path.join(ckpt_path, "q2_tgt.pt"), map_location=device))
            alpha_file = os.path.join(ckpt_path, "alpha.pt")
            if os.path.exists(alpha_file):
                alpha_state = torch.load(alpha_file, map_location="cpu")
                if "log_alpha" in alpha_state:
                    sac.log_alpha.data = alpha_state["log_alpha"].to(device)
            print("✅ Checkpoint loaded successfully.")
        except Exception as e:
            print("⚠️ Failed to load checkpoint:", e)
    else:
        print("No existing checkpoints found — starting fresh.")
else:
    print("No resume directory — starting fresh.")


replay = ReplayBuffer(state_dim, action_dim, capacity=REPLAY_CAPACITY)


os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)
writer.add_text("config", f"""
env: {ENV_ID}
seed: {SEED}
total_steps: {TOTAL_STEPS}
warmup_steps: {WARMUP_STEPS}
train_every: {TRAIN_EVERY}
updates_per_step: {UPDATES_PER_STEP}
batch_size: {BATCH_SIZE}
replay_capacity: {REPLAY_CAPACITY}
save_every: {SAVE_EVERY}
device: {device}
""".strip())


obs, _ = env.reset()
episode_ret, episode_len = 0.0, 0
t0 = time.time()
start_time = t0

for step in range(1, TOTAL_STEPS + 1):
    state = obs.astype(np.float32)

   
    if step <= WARMUP_STEPS:
        action = env.action_space.sample().astype(np.float32)
    else:
        with torch.no_grad():
            s_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
            a_t = sac.actor.act(s_t, deterministic=False)[0].cpu().numpy()
        action = np.clip(a_t, act_low, act_high).astype(np.float32)

  
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = bool(terminated)  # bootstrap only on true terminal
    replay.add(state, action, [reward], next_obs.astype(np.float32), [float(done)])

    obs = next_obs
    episode_ret += reward
    episode_len += 1

    
    if terminated or truncated:
        ret_meter.add(episode_ret)
        len_meter.add(episode_len)

        
        writer.add_scalar("episode/return", episode_ret, global_step=step)
        writer.add_scalar("episode/length", episode_len, global_step=step)

        obs, _ = env.reset()
        episode_ret, episode_len = 0.0, 0

    if step > WARMUP_STEPS and (step % TRAIN_EVERY == 0) and replay.size >= BATCH_SIZE:
        for _ in range(UPDATES_PER_STEP):
            batch = replay.sample_torch(BATCH_SIZE, device)
            logs = sac.train_on_batch(batch)

            
            lq_meter.add(logs["loss_q"])
            lpi_meter.add(logs["loss_pi"])
            la_meter.add(logs["loss_alpha"])
            alpha_meter.add(logs["alpha"])
            q_meter.add(logs["minq_mean"])
            logp_meter.add(logs["logp_mean"])

            
            writer.add_scalar("loss/critic",  logs["loss_q"],   global_step=step)
            writer.add_scalar("loss/actor",   logs["loss_pi"],  global_step=step)
            writer.add_scalar("loss/alpha",   logs["loss_alpha"], global_step=step)
            writer.add_scalar("diagnostics/alpha",    logs["alpha"],    global_step=step)
            writer.add_scalar("diagnostics/minQ_mean", logs["minq_mean"],global_step=step)
            writer.add_scalar("diagnostics/logp_mean", logs["logp_mean"],global_step=step)

       
        if step % 1000 == 0:
            elapsed = time.time() - t0
            print(f"[step {step:7d}] "
                  f"ret(avg){ret_meter.avg:8.1f}  len(avg){len_meter.avg:6.1f}  "
                  f"Lq{lq_meter.avg:8.4f}  Lpi{lpi_meter.avg:8.4f}  "
                  f"alpha{float(alpha_meter.avg):6.3f}  "
                  f"minQ{q_meter.avg:9.3f}  logp{logp_meter.avg:8.3f}  "
                  f"time {elapsed/60:.1f}m")

    
    if step % SAVE_EVERY == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"step_{step//1000}k")
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(sac.actor.state_dict(),    os.path.join(ckpt_path, "actor.pt"))
        torch.save(sac.q1.state_dict(),       os.path.join(ckpt_path, "q1.pt"))
        torch.save(sac.q2.state_dict(),       os.path.join(ckpt_path, "q2.pt"))
        torch.save(sac.q1_tgt.state_dict(),   os.path.join(ckpt_path, "q1_tgt.pt"))
        torch.save(sac.q2_tgt.state_dict(),   os.path.join(ckpt_path, "q2_tgt.pt"))
        torch.save({"log_alpha": sac.log_alpha.detach().cpu()}, os.path.join(ckpt_path, "alpha.pt"))
        print(f"💾 Saved checkpoint: {ckpt_path}")

    
    if step % EVAL_EVERY_STEPS == 0:
        eval_returns = []
        for _ in range(3):
            eo, _ = eval_env.reset()
            done_eval = False
            ep_ret = 0.0
            while not done_eval:
                with torch.no_grad():
                    s_t = torch.as_tensor(eo[None, :], dtype=torch.float32, device=device)
                    a_t = sac.actor.act(s_t, deterministic=True)[0].cpu().numpy()
                eo, rew, term, trunc, _ = eval_env.step(a_t)
                ep_ret += rew
                done_eval = term or trunc
            eval_returns.append(ep_ret)

        avg_ret = float(np.mean(eval_returns))
        elapsed = time.time() - start_time
        alpha_val = float(sac.log_alpha.exp().detach().cpu().item())
        print(f" [avg episode returns {avg_ret:>7.1f}] || [step {step:>7}]  replay {replay.size:>7} | "
              f"alpha {alpha_val:.3f} | time {elapsed/60:.1f}m")

        
        writer.add_scalar("eval/avg_return", avg_ret, global_step=step)
        writer.add_scalar("replay/size", replay.size, global_step=step)


os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(sac.actor.state_dict(),  os.path.join(SAVE_DIR, "actor_final.pt"))
torch.save(sac.q1.state_dict(),     os.path.join(SAVE_DIR, "q1_final.pt"))
torch.save(sac.q2.state_dict(),     os.path.join(SAVE_DIR, "q2_final.pt"))
torch.save(sac.q1_tgt.state_dict(), os.path.join(SAVE_DIR, "q1_tgt_final.pt"))
torch.save(sac.q2_tgt.state_dict(), os.path.join(SAVE_DIR, "q2_tgt_final.pt"))
torch.save({"log_alpha": sac.log_alpha.detach().cpu()}, os.path.join(SAVE_DIR, "alpha_final.pt"))
writer.close()
print("✅ Saved final checkpoints.")
