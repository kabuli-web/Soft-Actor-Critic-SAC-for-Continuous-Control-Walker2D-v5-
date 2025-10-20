# eval_sac_torch_walker.py
import os
import argparse
import numpy as np
import torch
import gymnasium as gym

from sac_torch import SAC  # uses the same Actor architecture as training

def find_latest_checkpoint(save_dir: str):
    if not os.path.isdir(save_dir):
        return None
    subdirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    if not subdirs:
        # fall back to actor_final.pt in root if present
        final_path = os.path.join(save_dir, "actor_final.pt")
        return final_path if os.path.exists(final_path) else None
    # sort by numeric content (e.g., "step_200k" -> 200)
    def numeric_key(name):
        digits = "".join(c for c in name if c.isdigit())
        return int(digits) if digits else -1
    latest = sorted(subdirs, key=numeric_key)[-1]
    return os.path.join(save_dir, latest, "actor.pt")

def make_env(env_id: str, render: str, video_dir: str | None):
    if render == "human":
        return gym.make(env_id, render_mode="human")
    elif render == "record":
        base = gym.make(env_id, render_mode="rgb_array")
        from gymnasium.wrappers import RecordVideo
        os.makedirs(video_dir or "videos", exist_ok=True)
        return RecordVideo(base, video_folder=video_dir or "videos", episode_trigger=lambda ep: True)
    else:
        return gym.make(env_id)

@torch.no_grad()
def evaluate(env_id="Walker2d-v5",
             save_dir="checkpoints_torch_walker",
             actor_path=None,
             episodes=5,
             seed=123,
             render="none",
             video_dir="videos",
             device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build env (fallback to v4 if v5 missing)
    try:
        env = make_env(env_id, render, video_dir)
    except Exception:
        if env_id.endswith("-v5"):
            env_id = env_id.replace("-v5", "-v4")
            env = make_env(env_id, render, video_dir)
        else:
            raise
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    # Dimensions / bounds
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    act_low    = env.action_space.low
    act_high   = env.action_space.high

    # Build agent skeleton (actor only)
    sac = SAC(state_dim, action_dim, act_low, act_high, device=device)

    # Resolve which weights to load
    chosen_path = actor_path or find_latest_checkpoint(save_dir)
    if chosen_path is None:
        raise FileNotFoundError(
            f"No checkpoints found in '{save_dir}'. "
            f"Pass --actor_path to a specific .pt file."
        )

    # Load weights
    state = torch.load(chosen_path, map_location=device)
    sac.actor.load_state_dict(state)
    sac.actor.eval()
    print(f"✅ Loaded actor weights from: {chosen_path} (device: {device})")
    print(f"Env: {env_id}")

    # Rollouts (deterministic policy)
    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            s_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
            a_t = sac.actor.act(s_t, deterministic=True)[0].cpu().numpy()
            obs, rew, term, trunc, _ = env.step(a_t)
            ep_ret += rew
            ep_len += 1
            done = term or trunc

        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"Episode {ep+1:02d} | return = {ep_ret:.1f} | length = {ep_len}")

    print("-" * 60)
    print(f"Avg return: {np.mean(returns):.1f} ± {np.std(returns):.1f} over {episodes} episodes")
    print(f"Avg length: {np.mean(lengths):.1f}")
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Walker2d-v5")
    p.add_argument("--save_dir", type=str, default="checkpoints_torch_walker",
                   help="Directory with step_*k subfolders or actor_final.pt")
    p.add_argument("--actor_path", type=str, default=None,
                   help="Direct path to actor .pt (overrides save_dir auto-detect)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", type=str, choices=["none", "human", "record"], default="none",
                   help="'human' shows a window; 'record' saves MP4s")
    p.add_argument("--video_dir", type=str, default="videos")
    p.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (auto if None)")
    args = p.parse_args()

    evaluate(env_id=args.env,
             save_dir=args.save_dir,
             actor_path=args.actor_path,
             episodes=args.episodes,
             seed=args.seed,
             render=args.render,
             video_dir=args.video_dir,
             device=args.device)
