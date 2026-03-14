"""
Evaluation and visualization for TEM world model.
Tests whether g-space captures meaningful abstract dynamical structure.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import make_config
from tem_model import TEMModel
from environment import DomainRandomizedHopper, collect_trajectories


def load_model(model_path, cfg_path=None):
    """Load a trained TEM model from checkpoint."""
    if cfg_path is not None:
        cfg = torch.load(cfg_path, weights_only=False)
    else:
        cfg = make_config()
    model = TEMModel(cfg)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, cfg


def collect_representations(model, cfg, n_episodes=5, steps_per_episode=200, randomize=True):
    """Run model on episodes and collect g representations + metadata."""
    cfg_eval = cfg.copy()
    cfg_eval['randomize'] = randomize

    all_g = []          # abstract state vectors
    all_obs = []        # raw observations
    all_pred_obs = []   # predicted observations
    all_episode = []    # episode index
    all_step = []       # step within episode
    all_params = []     # body parameters

    for ep in range(n_episodes):
        env = DomainRandomizedHopper(cfg_eval, seed=1000 + ep)
        obs, _ = env.reset()
        body_params = env.body_params

        # Collect trajectory
        chunk = []
        observations = [obs]
        for t in range(steps_per_episode):
            action = env.action_space.sample()
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float).unsqueeze(0)
            chunk.append({'obs': obs_tensor, 'action': action_tensor})

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            observations.append(obs)

        # Run through model
        with torch.no_grad():
            steps = model(chunk)

        # Extract representations
        for t, step in enumerate(steps):
            g_concat = torch.cat(step.g_inf, dim=1).squeeze(0).numpy()
            all_g.append(g_concat)
            all_obs.append(step.obs.squeeze(0).numpy())
            all_pred_obs.append(step.x_gen[0][0].squeeze(0).numpy())  # from p_inf
            all_episode.append(ep)
            all_step.append(t)
            all_params.append(body_params)

        env.close()

    return {
        'g': np.stack(all_g),
        'obs': np.stack(all_obs),
        'pred_obs': np.stack(all_pred_obs),
        'episode': np.array(all_episode),
        'step': np.array(all_step),
        'params': all_params,
    }


def evaluate_prediction_accuracy(model, cfg, n_episodes=10, steps_per_episode=200):
    """Evaluate observation prediction MSE across domain-randomized episodes."""
    data = collect_representations(model, cfg, n_episodes, steps_per_episode, randomize=True)

    mse_per_dim = np.mean((data['obs'] - data['pred_obs']) ** 2, axis=0)
    mse_total = np.mean(mse_per_dim)

    dim_names = [
        'z_pos', 'angle', 'thigh_ang', 'leg_ang', 'foot_ang',
        'x_vel', 'z_vel', 'ang_vel', 'thigh_vel', 'leg_vel', 'foot_vel'
    ]

    print(f"\nObservation Prediction MSE (total): {mse_total:.6f}")
    print(f"{'Dimension':<12} {'MSE':>10}")
    print("-" * 24)
    for name, mse in zip(dim_names, mse_per_dim):
        print(f"{name:<12} {mse:>10.6f}")

    return mse_per_dim, mse_total


def visualize_g_space(model, cfg, n_episodes=5, steps_per_episode=200,
                      save_path=None, method='tsne'):
    """Visualize abstract state space g using dimensionality reduction."""
    data = collect_representations(model, cfg, n_episodes, steps_per_episode, randomize=True)
    g = data['g']

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=0)
        g_2d = reducer.fit_transform(g)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        g_2d = reducer.fit_transform(g)
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color by episode (should separate if different body params)
    ax = axes[0]
    scatter = ax.scatter(g_2d[:, 0], g_2d[:, 1], c=data['episode'],
                          cmap='tab10', s=5, alpha=0.5)
    ax.set_title('g-space colored by episode\n(different body params)')
    plt.colorbar(scatter, ax=ax, label='Episode')

    # Color by time step (should show trajectory structure)
    ax = axes[1]
    scatter = ax.scatter(g_2d[:, 0], g_2d[:, 1], c=data['step'],
                          cmap='viridis', s=5, alpha=0.5)
    ax.set_title('g-space colored by time step')
    plt.colorbar(scatter, ax=ax, label='Step')

    # Color by a kinematic variable (e.g., z-position = body height)
    ax = axes[2]
    scatter = ax.scatter(g_2d[:, 0], g_2d[:, 1], c=data['obs'][:, 0],
                          cmap='coolwarm', s=5, alpha=0.5)
    ax.set_title('g-space colored by body height (z)')
    plt.colorbar(scatter, ax=ax, label='z_pos')

    for ax in axes:
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_frequency_modules(model, cfg, n_episodes=3, steps_per_episode=300,
                                  save_path=None):
    """Analyze temporal dynamics of each frequency module."""
    cfg_eval = cfg.copy()
    cfg_eval['randomize'] = False  # fixed body for clean analysis

    env = DomainRandomizedHopper(cfg_eval, seed=42)
    obs, _ = env.reset()

    chunk = []
    for t in range(steps_per_episode):
        action = env.action_space.sample()
        obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float).unsqueeze(0)
        chunk.append({'obs': obs_tensor, 'action': action_tensor})
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    with torch.no_grad():
        steps = model(chunk)

    n_f = cfg['n_f']
    fig, axes = plt.subplots(n_f, 2, figsize=(14, 3 * n_f))

    for f in range(n_f):
        # Extract g activations for this frequency module
        g_f = np.stack([step.g_inf[f].squeeze(0).numpy() for step in steps])

        # Plot activations over time (first 5 neurons)
        ax = axes[f, 0]
        n_show = min(5, g_f.shape[1])
        for neuron in range(n_show):
            ax.plot(g_f[:, neuron], alpha=0.7, label=f'g[{neuron}]')
        ax.set_title(f'Freq module {f} (alpha_init={cfg["f_initial"][f]:.2f}): activations')
        ax.set_xlabel('Step')
        ax.legend(fontsize=7)

        # Autocorrelation of activations
        ax = axes[f, 1]
        max_lag = min(100, len(g_f) // 2)
        mean_autocorr = np.zeros(max_lag)
        for neuron in range(g_f.shape[1]):
            signal = g_f[:, neuron] - np.mean(g_f[:, neuron])
            norm = np.sum(signal ** 2)
            if norm > 1e-10:
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(autocorr) // 2:len(autocorr) // 2 + max_lag]
                mean_autocorr += autocorr / norm
        mean_autocorr /= g_f.shape[1]
        ax.plot(mean_autocorr)
        ax.set_title(f'Freq module {f}: mean autocorrelation')
        ax.set_xlabel('Lag (steps)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def compare_domains(model, cfg, n_configs=5, steps_per_episode=200, save_path=None):
    """Compare g representations across different body configurations.

    If the model learns structural abstraction, similar dynamical states
    should map to similar g vectors regardless of body parameters.
    """
    all_g = []
    all_height = []
    all_config = []

    for config_i in range(n_configs):
        cfg_eval = cfg.copy()
        cfg_eval['randomize'] = True

        env = DomainRandomizedHopper(cfg_eval, seed=2000 + config_i)
        obs, _ = env.reset()

        chunk = []
        for t in range(steps_per_episode):
            action = env.action_space.sample()
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float).unsqueeze(0)
            chunk.append({'obs': obs_tensor, 'action': action_tensor})
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

        with torch.no_grad():
            steps = model(chunk)

        for step in steps:
            g_concat = torch.cat(step.g_inf, dim=1).squeeze(0).numpy()
            all_g.append(g_concat)
            all_height.append(step.obs[0, 0].item())  # z_pos
            all_config.append(config_i)

    all_g = np.stack(all_g)
    all_height = np.array(all_height)
    all_config = np.array(all_config)

    # PCA for visualization
    pca = PCA(n_components=2)
    g_2d = pca.fit_transform(all_g)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for c in range(n_configs):
        mask = all_config == c
        ax.scatter(g_2d[mask, 0], g_2d[mask, 1], s=5, alpha=0.4, label=f'Config {c}')
    ax.set_title('g-space: different body configurations')
    ax.legend()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    ax = axes[1]
    scatter = ax.scatter(g_2d[:, 0], g_2d[:, 1], c=all_height, cmap='coolwarm', s=5, alpha=0.4)
    ax.set_title('g-space: colored by body height')
    plt.colorbar(scatter, ax=ax, label='z_pos')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save figures')
    args = parser.parse_args()

    model, cfg = load_model(args.model, args.config)

    save_dir = args.save_dir or os.path.dirname(args.model)

    print("=" * 60)
    print("Evaluating observation prediction accuracy...")
    print("=" * 60)
    evaluate_prediction_accuracy(model, cfg)

    print("\n" + "=" * 60)
    print("Visualizing g-space...")
    print("=" * 60)
    visualize_g_space(model, cfg, save_path=os.path.join(save_dir, 'g_space.png'))

    print("\n" + "=" * 60)
    print("Analyzing frequency modules...")
    print("=" * 60)
    visualize_frequency_modules(model, cfg, save_path=os.path.join(save_dir, 'freq_modules.png'))

    print("\n" + "=" * 60)
    print("Comparing across body configurations...")
    print("=" * 60)
    compare_domains(model, cfg, save_path=os.path.join(save_dir, 'domain_compare.png'))
