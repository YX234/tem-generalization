"""
Evaluate TEM adaptation to novel environments.

Measures how quickly the TEM's Hebbian memory and precision weighting
adapt to held-out physics configurations, using observation prediction
MSE as the core metric.

Usage:
    python evaluate_adaptation.py --model-dir runs/<run>/models/
"""

import os
import argparse
import pickle
import numpy as np
import torch
import json

from config import make_config
from tem_model import TEMModel
from environment import DomainRandomizedHopper, RunningNormalizer


def load_model(model_dir, device):
    """Load trained TEM model and normalizer."""
    cfg_path = os.path.join(model_dir, 'cfg_final.pt')
    model_path = os.path.join(model_dir, 'tem_final.pt')
    norm_path = os.path.join(model_dir, 'normalizer_final.pkl')

    if not os.path.exists(model_path):
        import glob
        ckpts = sorted(glob.glob(os.path.join(model_dir, 'tem_*.pt')))
        if not ckpts:
            raise FileNotFoundError(f"No TEM checkpoints in {model_dir}")
        model_path = ckpts[-1]
        idx = os.path.basename(model_path).replace('tem_', '').replace('.pt', '')
        cfg_path = os.path.join(model_dir, f'cfg_{idx}.pt')
        norm_path = os.path.join(model_dir, f'normalizer_{idx}.pkl')

    cfg = torch.load(cfg_path, weights_only=False)
    model = TEMModel(cfg, device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with open(norm_path, 'rb') as f:
        normalizer = pickle.load(f)

    return model, cfg, normalizer


def make_env_configs(base_cfg):
    """Create test environment configurations."""
    configs = {}

    # Default Hopper (no randomization)
    default_cfg = base_cfg.copy()
    default_cfg['randomize'] = False
    default_cfg['gravity_range'] = None
    configs['default'] = default_cfg

    # In-distribution: within training ranges
    id_cfg = base_cfg.copy()
    id_cfg['randomize'] = True
    configs['in_distribution'] = id_cfg

    # OOD: extrapolation beyond training ranges
    ood_cfg = base_cfg.copy()
    ood_cfg['randomize'] = True
    ood_cfg['gravity_range'] = base_cfg.get('eval_gravity_range', (2.5, 4.0))
    ood_cfg['friction_range'] = base_cfg.get('eval_friction_range', (0.02, 0.1))
    ood_cfg['mass_range'] = base_cfg.get('eval_mass_range', (4.0, 8.0))
    configs['ood_extrapolation'] = ood_cfg

    # Extreme single-param: gravity only
    extreme_grav = base_cfg.copy()
    extreme_grav['randomize'] = True
    extreme_grav['mass_range'] = (1.0, 1.0)
    extreme_grav['damping_range'] = (1.0, 1.0)
    extreme_grav['friction_range'] = (1.0, 1.0)
    extreme_grav['gear_range'] = (1.0, 1.0)
    extreme_grav['gravity_range'] = (3.0, 4.0)
    configs['extreme_gravity'] = extreme_grav

    # Extreme single-param: friction only
    extreme_fric = base_cfg.copy()
    extreme_fric['randomize'] = True
    extreme_fric['mass_range'] = (1.0, 1.0)
    extreme_fric['damping_range'] = (1.0, 1.0)
    extreme_fric['friction_range'] = (0.02, 0.1)
    extreme_fric['gear_range'] = (1.0, 1.0)
    extreme_fric['gravity_range'] = (1.0, 1.0)
    configs['extreme_friction'] = extreme_fric

    return configs


def run_episode(model, env, normalizer, device, max_steps=200):
    """Run a single episode, collecting per-step adaptation metrics.

    Returns dict with per-step arrays:
        mse: observation prediction MSE at each step
        transition_err_ema: mean EMA across frequency modules
        w_path, w_mem, w_obs: effective precision weights (from sigma values)
    """
    cfg = model.cfg
    n_f = cfg['n_f']

    obs_raw, _ = env.reset()
    obs_normed = normalizer.normalize(obs_raw)
    obs_tensor = torch.tensor(obs_normed, dtype=torch.float, device=device).unsqueeze(0)

    # Initialize TEM state
    n_p_total = sum(cfg['n_p'])
    M = [
        torch.zeros(1, n_p_total, n_p_total, device=device),
        torch.zeros(1, n_p_total, n_p_total, device=device),
    ]
    g_inf = [model.g_init[f].detach().unsqueeze(0).clone() for f in range(n_f)]
    x_inf = [torch.zeros(1, cfg['n_x_f'][f], device=device) for f in range(n_f)]
    transition_err_ema = None
    a_prev = None

    mses = []
    ema_vals = []

    for step in range(max_steps):
        with torch.no_grad():
            # Run full iteration to get observation predictions
            g_gen, g_gen_sigma = model._gen_g(a_prev, g_inf, transition_err_ema)
            x_f, g_new, p_inf_x, p_inf = model._inference(
                obs_tensor, M, x_inf, (g_gen, g_gen_sigma)
            )
            x_gen, p_gen = model._generative(M, p_inf, g_new, g_gen)

            # Observation prediction MSE (from inferred p pathway)
            mu_pred = x_gen[0][0]
            mse = torch.mean((mu_pred - obs_tensor) ** 2).cpu().item()
            mses.append(mse)

            # Update Hebbian memory
            M_new = [model._hebbian(M[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]
            M_new.append(model._hebbian(
                M[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1),
                do_hierarchical=False
            ))

            # Update transition error EMA
            terr = [torch.abs(g_new[f] - g_gen[f]) for f in range(n_f)]
            if transition_err_ema is not None:
                decay = cfg['transition_err_ema_decay']
                new_ema = [decay * transition_err_ema[f] + (1 - decay) * terr[f]
                           for f in range(n_f)]
            else:
                new_ema = terr

            ema_mean = np.mean([new_ema[f].mean().cpu().item() for f in range(n_f)])
            ema_vals.append(ema_mean)

            # Take random action and step environment
            action = env.env.action_space.sample()
            a_prev = torch.tensor(action, dtype=torch.float, device=device).unsqueeze(0)

            obs_raw, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

            obs_normed = normalizer.normalize(obs_raw)
            obs_tensor = torch.tensor(obs_normed, dtype=torch.float, device=device).unsqueeze(0)

            # Carry state forward
            M = M_new
            g_inf = g_new
            x_inf = x_f
            transition_err_ema = new_ema

    return {
        'mse': np.array(mses),
        'transition_err_ema': np.array(ema_vals),
    }


def compute_metrics(mse_array):
    """Compute summary metrics from an adaptation curve."""
    metrics = {}
    metrics['mse_step_1'] = float(mse_array[0]) if len(mse_array) > 0 else float('nan')
    metrics['mse_step_10'] = float(mse_array[min(9, len(mse_array) - 1)])
    metrics['mse_step_50'] = float(mse_array[min(49, len(mse_array) - 1)])
    if len(mse_array) >= 150:
        metrics['mse_asymptotic'] = float(np.mean(mse_array[149:]))
    else:
        metrics['mse_asymptotic'] = float(np.mean(mse_array[max(0, len(mse_array) - 50):]))
    return metrics


def run_ablation(model, cfg, env, normalizer, device, ablation_type, max_steps=200):
    """Run episode with specific ablation applied."""
    original_eta = cfg.get('eta', 0.5)
    original_lambda = cfg.get('lambda_', 0.9999)
    original_decay = cfg.get('transition_err_ema_decay', 0.95)

    if ablation_type == 'no_memory':
        cfg['eta'] = 0.0
        cfg['lambda_'] = 0.0
        model.cfg['eta'] = 0.0
        model.cfg['lambda_'] = 0.0
    elif ablation_type == 'no_ema':
        cfg['transition_err_ema_decay'] = None
        model.cfg['transition_err_ema_decay'] = None

    result = run_episode(model, env, normalizer, device, max_steps)

    # Restore
    cfg['eta'] = original_eta
    cfg['lambda_'] = original_lambda
    cfg['transition_err_ema_decay'] = original_decay
    model.cfg['eta'] = original_eta
    model.cfg['lambda_'] = original_lambda
    model.cfg['transition_err_ema_decay'] = original_decay

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate TEM adaptation')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to models/ directory')
    parser.add_argument('--n-episodes', type=int, default=20,
                        help='Episodes per environment config')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: model-dir/../eval)')
    parser.add_argument('--ablations', action='store_true',
                        help='Run ablation studies')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Loading model from {args.model_dir}")
    model, cfg, normalizer = load_model(args.model_dir, device)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.model_dir.rstrip('/')), 'eval')
    os.makedirs(output_dir, exist_ok=True)

    env_configs = make_env_configs(cfg)
    all_results = {}

    for env_name, env_cfg in env_configs.items():
        print(f"\n--- {env_name} ---")
        env_cfg['terminate_when_unhealthy'] = False

        episode_results = []
        for ep in range(args.n_episodes):
            env = DomainRandomizedHopper(env_cfg, seed=args.seed + ep)
            result = run_episode(model, env, normalizer, device, args.max_steps)
            episode_results.append(result)
            env.close()

        # Aggregate adaptation curves (pad shorter episodes)
        max_len = max(len(r['mse']) for r in episode_results)
        mse_matrix = np.full((args.n_episodes, max_len), np.nan)
        ema_matrix = np.full((args.n_episodes, max_len), np.nan)
        for i, r in enumerate(episode_results):
            mse_matrix[i, :len(r['mse'])] = r['mse']
            ema_matrix[i, :len(r['transition_err_ema'])] = r['transition_err_ema']

        mean_mse = np.nanmean(mse_matrix, axis=0)
        mean_ema = np.nanmean(ema_matrix, axis=0)

        metrics = compute_metrics(mean_mse)
        print(f"  mse_step_1:  {metrics['mse_step_1']:.4f}")
        print(f"  mse_step_10: {metrics['mse_step_10']:.4f}")
        print(f"  mse_step_50: {metrics['mse_step_50']:.4f}")
        print(f"  mse_asymptotic: {metrics['mse_asymptotic']:.4f}")

        all_results[env_name] = {
            'metrics': metrics,
            'mean_mse_curve': mean_mse.tolist(),
            'mean_ema_curve': mean_ema.tolist(),
        }

    # Ablation studies
    if args.ablations:
        print("\n=== ABLATION STUDIES ===")
        ablation_types = ['no_ema', 'no_memory']

        for ablation in ablation_types:
            print(f"\n--- Ablation: {ablation} ---")
            ablation_results = {}

            for env_name, env_cfg in env_configs.items():
                env_cfg['terminate_when_unhealthy'] = False
                episode_results = []
                for ep in range(args.n_episodes):
                    env = DomainRandomizedHopper(env_cfg, seed=args.seed + ep)
                    result = run_ablation(
                        model, cfg, env, normalizer, device, ablation, args.max_steps)
                    episode_results.append(result)
                    env.close()

                max_len = max(len(r['mse']) for r in episode_results)
                mse_matrix = np.full((args.n_episodes, max_len), np.nan)
                for i, r in enumerate(episode_results):
                    mse_matrix[i, :len(r['mse'])] = r['mse']

                mean_mse = np.nanmean(mse_matrix, axis=0)
                metrics = compute_metrics(mean_mse)
                print(f"  {env_name}: mse_50={metrics['mse_step_50']:.4f} "
                      f"mse_asymp={metrics['mse_asymptotic']:.4f}")

                ablation_results[env_name] = {
                    'metrics': metrics,
                    'mean_mse_curve': mean_mse.tolist(),
                }

            all_results[f'ablation_{ablation}'] = ablation_results

    # Save results
    results_path = os.path.join(output_dir, 'adaptation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
