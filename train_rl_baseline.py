"""
PPO baseline on raw observations for domain-randomized Hopper.

Usage:
    python train_rl_baseline.py [--total-timesteps N] [--obs-norm online|pretrained]

Trains a PPO policy directly on normalized 11-dim Hopper observations
WITHOUT the TEM world model. This serves as a diagnostic baseline to
determine whether the TEM's g_inf representation is the performance
bottleneck or the domain randomization itself.

Compare results against train_rl.py (TEM pipeline) to measure the
representation gap.
"""

import os
import sys
import time
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from config import make_config, make_rl_config
from policy import ActorCritic
from environment import DomainRandomizedHopper, RunningNormalizer


class NormalizedHopperEnv(gym.Env):
    """Domain-randomized Hopper with observation normalization.

    Exposes raw normalized observations (11-dim) for direct RL training.
    Analogous to TEMObservationEnv but without TEM inference.
    """

    def __init__(self, cfg, normalizer, seed=None):
        super().__init__()
        self.hopper = DomainRandomizedHopper(cfg, seed=seed)
        self.normalizer = normalizer
        self._obs_dim = cfg['obs_dim']

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = self.hopper.action_space

    def reset(self, *, seed=None, options=None):
        obs_raw, info = self.hopper.reset()
        return self.normalizer.normalize(obs_raw), info

    def step(self, action):
        obs_raw, reward, terminated, truncated, info = self.hopper.step(action)
        return self.normalizer.normalize(obs_raw), reward, terminated, truncated, info

    def close(self):
        self.hopper.close()


def setup_logging(run_path):
    logger = logging.getLogger('baseline_rl')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(os.path.join(run_path, 'train_rl_baseline.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    return logger


def main():
    parser = argparse.ArgumentParser(description='PPO baseline on raw Hopper observations')
    parser.add_argument('--total-timesteps', type=int, default=None)
    parser.add_argument('--n-envs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obs-norm', type=str, default='online',
                        choices=['online', 'pretrained'],
                        help='online: fresh normalizer updated during training. '
                             'pretrained: load normalizer from TEM training.')
    parser.add_argument('--normalizer-path', type=str, default=None,
                        help='Path to pretrained normalizer .pkl (required if --obs-norm=pretrained)')
    args = parser.parse_args()

    cfg = make_config()
    rl_cfg = make_rl_config()
    if args.total_timesteps is not None:
        rl_cfg['total_timesteps'] = args.total_timesteps
    if args.n_envs is not None:
        rl_cfg['n_envs'] = args.n_envs

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Output directory
    run_path = os.path.join(os.path.dirname(__file__), 'runs_rl_baseline',
                            time.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_path, exist_ok=True)

    logger = setup_logging(run_path)
    writer = SummaryWriter(os.path.join(run_path, 'tensorboard'))

    # Normalizer setup
    if args.obs_norm == 'pretrained':
        if args.normalizer_path is None:
            raise ValueError("--normalizer-path required when --obs-norm=pretrained")
        with open(args.normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        logger.info(f"Loaded pretrained normalizer ({normalizer.count} observations)")
    else:
        normalizer = RunningNormalizer(dim=cfg['obs_dim'])
        # Warm up normalizer with random rollouts
        logger.info("Warming up normalizer with 10000 random observations...")
        warmup_env = DomainRandomizedHopper(cfg, seed=args.seed + 9999)
        warmup_obs = []
        obs, _ = warmup_env.reset()
        for _ in range(10000):
            action = warmup_env.action_space.sample()
            obs, _, terminated, truncated, _ = warmup_env.step(action)
            warmup_obs.append(obs)
            if terminated or truncated:
                obs, _ = warmup_env.reset()
        normalizer.update(np.array(warmup_obs))
        warmup_env.close()
        logger.info(f"Normalizer initialized: mean={normalizer.mean[:3]}..., std={np.sqrt(normalizer.var[:3])}...")

    obs_dim = cfg['obs_dim']
    action_dim = cfg['action_dim']

    # Create environments
    n_envs = rl_cfg['n_envs']
    envs = []
    for i in range(n_envs):
        env_cfg = cfg.copy()
        env_cfg['randomize'] = rl_cfg['randomize']
        env_cfg['terminate_when_unhealthy'] = True
        envs.append(NormalizedHopperEnv(env_cfg, normalizer, seed=args.seed + i))

    logger.info(f"Baseline PPO on raw observations: obs_dim={obs_dim}, device={device}")
    logger.info(f"Created {n_envs} domain-randomized environments")
    logger.info(f"Domain randomization: mass={cfg['mass_range']}, friction={cfg['friction_range']}, "
                f"gravity={cfg['gravity_range']}, damping={cfg['damping_range']}, gear={cfg['gear_range']}")

    # Policy
    policy = ActorCritic(obs_dim, action_dim, rl_cfg['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=rl_cfg['lr'], eps=1e-5)
    total_policy_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Policy parameters: {total_policy_params:,}")

    # Initialize environments
    current_obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        current_obs[i], _ = env.reset()

    # Training state
    n_steps = rl_cfg['n_steps']
    total_timesteps = rl_cfg['total_timesteps']
    batch_size = n_envs * n_steps
    n_updates = total_timesteps // batch_size
    global_step = 0

    # Episode tracking
    ep_rewards = np.zeros(n_envs)
    ep_lengths = np.zeros(n_envs, dtype=int)

    logger.info(f"PPO: {n_envs} envs x {n_steps} steps = {batch_size} per update, "
                f"{n_updates} updates, {total_timesteps} total steps")

    for update in range(n_updates):
        update_start = time.time()

        # -- Rollout collection --
        all_obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        all_actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        all_log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        all_values = np.zeros((n_steps, n_envs), dtype=np.float32)
        all_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        all_dones = np.zeros((n_steps, n_envs), dtype=np.float32)

        completed_rewards = []
        completed_lengths = []

        for step in range(n_steps):
            all_obs[step] = current_obs

            # Policy forward pass
            with torch.no_grad():
                obs_tensor = torch.tensor(current_obs, dtype=torch.float, device=device)
                actions, log_probs, values, _ = policy(obs_tensor)
                actions_np = actions.cpu().numpy()
                all_actions[step] = actions_np
                all_log_probs[step] = log_probs.cpu().numpy()
                all_values[step] = values.cpu().numpy()

            # Step all environments
            for env_i in range(n_envs):
                next_obs, reward, terminated, truncated, info = envs[env_i].step(
                    actions_np[env_i]
                )
                done = terminated or truncated
                all_rewards[step, env_i] = reward
                all_dones[step, env_i] = float(done)
                ep_rewards[env_i] += reward
                ep_lengths[env_i] += 1

                if done:
                    completed_rewards.append(ep_rewards[env_i])
                    completed_lengths.append(ep_lengths[env_i])
                    ep_rewards[env_i] = 0
                    ep_lengths[env_i] = 0
                    next_obs, _ = envs[env_i].reset()

                current_obs[env_i] = next_obs

            global_step += n_envs

            # Update normalizer online
            if args.obs_norm == 'online':
                normalizer.update(current_obs)

        # -- Compute GAE --
        with torch.no_grad():
            obs_tensor = torch.tensor(current_obs, dtype=torch.float, device=device)
            last_values = policy.get_value(obs_tensor).cpu().numpy()

        advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        last_gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(n_steps)):
            next_values = last_values if t == n_steps - 1 else all_values[t + 1]
            non_terminal = 1.0 - all_dones[t]
            delta = all_rewards[t] + rl_cfg['gamma'] * next_values * non_terminal - all_values[t]
            last_gae = delta + rl_cfg['gamma'] * rl_cfg['gae_lambda'] * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + all_values

        # -- Flatten for minibatch sampling --
        b_obs = torch.tensor(all_obs.reshape(batch_size, -1), dtype=torch.float, device=device)
        b_actions = torch.tensor(all_actions.reshape(batch_size, -1), dtype=torch.float, device=device)
        b_log_probs = torch.tensor(all_log_probs.reshape(batch_size), dtype=torch.float, device=device)
        b_advantages = torch.tensor(advantages.reshape(batch_size), dtype=torch.float, device=device)
        b_returns = torch.tensor(returns.reshape(batch_size), dtype=torch.float, device=device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # -- PPO update --
        pg_losses, v_losses, entropy_vals, clip_fracs = [], [], [], []

        for epoch in range(rl_cfg['n_epochs']):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, rl_cfg['minibatch_size']):
                end = start + rl_cfg['minibatch_size']
                mb = indices[start:end]

                new_log_probs, new_values, entropy = policy.evaluate_actions(
                    b_obs[mb], b_actions[mb]
                )

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - b_log_probs[mb]).exp()
                surr1 = ratio * b_advantages[mb]
                surr2 = ratio.clamp(
                    1 - rl_cfg['clip_epsilon'], 1 + rl_cfg['clip_epsilon']
                ) * b_advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, b_returns[mb])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + rl_cfg['value_coeff'] * value_loss
                        + rl_cfg['entropy_coeff'] * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), rl_cfg['max_grad_norm'])
                optimizer.step()

                # Track stats
                pg_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                entropy_vals.append(entropy.mean().item())
                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > rl_cfg['clip_epsilon']).float().mean().item()
                    )

        elapsed = time.time() - update_start

        # -- Logging --
        if update % rl_cfg['log_interval'] == 0:
            mean_pg = np.mean(pg_losses)
            mean_vl = np.mean(v_losses)
            mean_ent = np.mean(entropy_vals)
            mean_clip = np.mean(clip_fracs)

            logger.info(
                f"Update {update:4d} | step {global_step:8d} | "
                f"pg_loss {mean_pg:.4f} | v_loss {mean_vl:.4f} | "
                f"entropy {mean_ent:.3f} | clip {mean_clip:.3f} | "
                f"{elapsed:.1f}s"
            )

            if completed_rewards:
                mean_rew = np.mean(completed_rewards)
                mean_len = np.mean(completed_lengths)
                logger.info(
                    f"  Episodes: {len(completed_rewards)} completed | "
                    f"reward {mean_rew:.1f} +/- {np.std(completed_rewards):.1f} | "
                    f"length {mean_len:.0f}"
                )
                writer.add_scalar('Rollout/ep_reward_mean', mean_rew, global_step)
                writer.add_scalar('Rollout/ep_length_mean', mean_len, global_step)
                writer.add_scalar('Rollout/ep_count', len(completed_rewards), global_step)

            writer.add_scalar('Train/policy_loss', mean_pg, global_step)
            writer.add_scalar('Train/value_loss', mean_vl, global_step)
            writer.add_scalar('Train/entropy', mean_ent, global_step)
            writer.add_scalar('Train/clip_fraction', mean_clip, global_step)
            writer.add_scalar('Train/update_time', elapsed, global_step)

        # -- Save checkpoint --
        if update % rl_cfg['save_interval'] == 0 and update > 0:
            ckpt_path = os.path.join(run_path, f'policy_{update}.pt')
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'global_step': global_step,
                'rl_cfg': rl_cfg,
                'obs_dim': obs_dim,
            }, ckpt_path)
            # Save normalizer with each checkpoint so evaluation works even if training is stopped early
            with open(os.path.join(run_path, 'normalizer.pkl'), 'wb') as f:
                pickle.dump(normalizer, f)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

            # Sync to Google Drive if mounted (Colab persistence)
            drive_dir = os.environ.get('TEM_DRIVE_DIR')
            if drive_dir and os.path.isdir(drive_dir):
                import shutil
                drive_rl_path = os.path.join(drive_dir, 'rl_baseline')
                os.makedirs(drive_rl_path, exist_ok=True)
                shutil.copy2(ckpt_path, drive_rl_path)
                logger.info(f"  Synced baseline checkpoint to Google Drive")

    # Final save
    final_path = os.path.join(run_path, 'policy_final.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': n_updates,
        'global_step': global_step,
        'rl_cfg': rl_cfg,
        'obs_dim': obs_dim,
    }, final_path)
    # Save normalizer for evaluation
    with open(os.path.join(run_path, 'normalizer.pkl'), 'wb') as f:
        pickle.dump(normalizer, f)
    logger.info(f"Training complete. Final policy saved to {final_path}")

    # Final sync to Google Drive
    drive_dir = os.environ.get('TEM_DRIVE_DIR')
    if drive_dir and os.path.isdir(drive_dir):
        import shutil
        drive_rl_path = os.path.join(drive_dir, 'rl_baseline')
        os.makedirs(drive_rl_path, exist_ok=True)
        shutil.copy2(final_path, drive_rl_path)
        shutil.copy2(os.path.join(run_path, 'normalizer.pkl'), drive_rl_path)
        log_file = os.path.join(run_path, 'train_rl_baseline.log')
        if os.path.exists(log_file):
            shutil.copy2(log_file, drive_rl_path)
        logger.info("Final baseline policy synced to Google Drive.")

    # Cleanup
    writer.close()
    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
