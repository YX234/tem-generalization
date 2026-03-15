"""
PPO training with TEM representations for domain-randomized Hopper.

Usage:
    python train_rl.py --model-dir runs/<run>/models/

The policy receives g_inf (108-dim abstract state) from the frozen TEM
and learns to produce Hopper actions via PPO.
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
from torch.utils.tensorboard import SummaryWriter

from config import make_rl_config
from tem_model import TEMModel
from tem_wrapper import TEMObservationEnv
from policy import ActorCritic
from environment import RunningNormalizer


def setup_logging(run_path):
    logger = logging.getLogger('tem_rl')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(os.path.join(run_path, 'train_rl.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    return logger


def load_tem(model_dir, device):
    """Load frozen TEM model and normalizer from a training checkpoint."""
    # Find checkpoint files
    cfg_path = os.path.join(model_dir, 'cfg_final.pt')
    model_path = os.path.join(model_dir, 'tem_final.pt')
    norm_path = os.path.join(model_dir, 'normalizer_final.pkl')

    # Fall back to latest numbered checkpoint
    if not os.path.exists(model_path):
        import glob
        ckpts = sorted(glob.glob(os.path.join(model_dir, 'tem_*.pt')))
        if not ckpts:
            raise FileNotFoundError(f"No TEM checkpoints in {model_dir}")
        model_path = ckpts[-1]
        idx = os.path.basename(model_path).replace('tem_', '').replace('.pt', '')
        cfg_path = os.path.join(model_dir, f'cfg_{idx}.pt')
        norm_path = os.path.join(model_dir, f'normalizer_{idx}.pkl')

    # Load config and model
    cfg = torch.load(cfg_path, weights_only=False)
    tem = TEMModel(cfg, device=device).to(device)
    tem.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    tem.eval()
    for p in tem.parameters():
        p.requires_grad_(False)

    # Load normalizer
    with open(norm_path, 'rb') as f:
        normalizer = pickle.load(f)

    return tem, cfg, normalizer


def main():
    parser = argparse.ArgumentParser(description='PPO on TEM representations')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to models/ directory from world model training')
    parser.add_argument('--total-timesteps', type=int, default=None)
    parser.add_argument('--n-envs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

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
    run_path = os.path.join(os.path.dirname(__file__), 'runs_rl',
                            time.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_path, exist_ok=True)

    logger = setup_logging(run_path)
    writer = SummaryWriter(os.path.join(run_path, 'tensorboard'))

    # Load frozen TEM
    logger.info(f"Loading TEM from {args.model_dir}")
    tem, tem_cfg, normalizer = load_tem(args.model_dir, device)
    g_dim = sum(tem_cfg['n_g'])
    action_dim = tem_cfg['action_dim']
    logger.info(f"TEM loaded: g_dim={g_dim}, device={device}")
    logger.info(f"Normalizer: {normalizer.count} observations seen during training")

    # Create wrapped environments
    n_envs = rl_cfg['n_envs']
    envs = []
    for i in range(n_envs):
        env_cfg = tem_cfg.copy()
        env_cfg['randomize'] = rl_cfg['randomize']
        env_cfg['terminate_when_unhealthy'] = True  # RL needs survival signal
        envs.append(TEMObservationEnv(env_cfg, tem, normalizer, device, seed=args.seed + i))
    logger.info(f"Created {n_envs} domain-randomized environments")

    # Policy
    policy = ActorCritic(g_dim, action_dim, rl_cfg['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=rl_cfg['lr'], eps=1e-5)
    total_policy_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Policy parameters: {total_policy_params:,}")

    # Initialize environments
    current_obs = np.zeros((n_envs, g_dim), dtype=np.float32)
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
        all_obs = np.zeros((n_steps, n_envs, g_dim), dtype=np.float32)
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
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(run_path, 'policy_final.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': n_updates,
        'global_step': global_step,
        'rl_cfg': rl_cfg,
    }, final_path)
    logger.info(f"Training complete. Final policy saved to {final_path}")

    # Cleanup
    writer.close()
    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
