"""
Hyperparameters and precomputed matrices for TEM-inspired continuous control.
Adapted from torch_tem-main/parameters.py for MuJoCo Hopper.
"""

import numpy as np
import torch


def make_config():
    cfg = {}

    # -- Environment dimensions
    cfg['obs_dim'] = 11       # Hopper-v5 observation
    cfg['action_dim'] = 3     # Hopper-v5 action

    # -- Frequency modules (timescale hierarchy for locomotion)
    # High freq first (fast balance), low freq last (slow trajectory)
    cfg['n_f'] = 4
    cfg['n_g_subsampled'] = [6, 5, 4, 3]         # subsampled entorhinal cells per module
    cfg['n_g'] = [3 * g for g in cfg['n_g_subsampled']]  # full entorhinal cells: [18, 15, 12, 9]
    cfg['f_initial'] = [0.99, 0.3, 0.09, 0.03]   # temporal filtering rates

    # -- Sensory encoding
    cfg['n_x_c'] = 16         # compressed observation dim (replaces two-hot)
    cfg['n_x_f'] = [cfg['n_x_c'] for _ in range(cfg['n_f'])]  # filtered obs per freq

    # -- Grounded location (hippocampal place cells): outer product of g_sub x x_f
    cfg['n_p'] = [g * x for g, x in zip(cfg['n_g_subsampled'], cfg['n_x_f'])]  # [96, 80, 64, 48]

    # -- Transition model
    cfg['d_hidden_dim'] = 128  # direct delta prediction allows larger hidden layer

    # -- Hebbian memory
    cfg['eta'] = 0.5          # rate of remembering
    cfg['lambda_'] = 0.9999   # rate of forgetting
    cfg['kappa'] = 0.8        # attractor decay
    cfg['i_attractor'] = cfg['n_f']  # attractor iterations = number of freq modules

    # -- Hierarchical connectivity: low freq -> high freq
    # g_connections[f_to][f_from] = True if f_from can influence f_to
    cfg['g_connections'] = [
        [cfg['f_initial'][f_from] <= cfg['f_initial'][f_to] for f_from in range(cfg['n_f'])]
        for f_to in range(cfg['n_f'])
    ]

    # -- Memory masks for hierarchical retrieval
    n_p = np.cumsum(np.concatenate(([0], cfg['n_p'])))
    # Hebbian update mask: connections from low to high frequency
    cfg['p_update_mask'] = torch.zeros((sum(cfg['n_p']), sum(cfg['n_p'])), dtype=torch.float)
    for f_from in range(cfg['n_f']):
        for f_to in range(cfg['n_f']):
            if cfg['f_initial'][f_from] <= cfg['f_initial'][f_to]:
                cfg['p_update_mask'][n_p[f_from]:n_p[f_from+1], n_p[f_to]:n_p[f_to+1]] = 1.0

    # Hierarchical retrieval masks: early-stop low-frequency modules
    cfg['i_attractor_max_freq_inf'] = [cfg['i_attractor'] for _ in range(cfg['n_f'])]
    cfg['i_attractor_max_freq_gen'] = [cfg['i_attractor'] - f for f in range(cfg['n_f'])]

    cfg['p_retrieve_mask_inf'] = [torch.zeros(sum(cfg['n_p'])) for _ in range(cfg['i_attractor'])]
    cfg['p_retrieve_mask_gen'] = [torch.zeros(sum(cfg['n_p'])) for _ in range(cfg['i_attractor'])]
    for mask, max_iters in zip(
        [cfg['p_retrieve_mask_inf'], cfg['p_retrieve_mask_gen']],
        [cfg['i_attractor_max_freq_inf'], cfg['i_attractor_max_freq_gen']]
    ):
        for f, max_i in enumerate(max_iters):
            for i in range(max_i):
                mask[i][n_p[f]:n_p[f+1]] = 1.0

    # -- Static matrices for outer-product binding (same as original TEM)
    cfg['W_repeat'] = [
        torch.tensor(np.kron(np.eye(cfg['n_g_subsampled'][f]),
                             np.ones((1, cfg['n_x_f'][f]))), dtype=torch.float)
        for f in range(cfg['n_f'])
    ]
    cfg['W_tile'] = [
        torch.tensor(np.kron(np.ones((1, cfg['n_g_subsampled'][f])),
                             np.eye(cfg['n_x_f'][f])), dtype=torch.float)
        for f in range(cfg['n_f'])
    ]
    cfg['g_downsample'] = [
        torch.cat([torch.eye(dim_out, dtype=torch.float),
                   torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float)])
        for dim_in, dim_out in zip(cfg['n_g'], cfg['n_g_subsampled'])
    ]

    # -- Training parameters
    cfg['train_iterations'] = 30000
    cfg['n_rollout'] = 50         # steps per BPTT chunk
    cfg['batch_size'] = 32
    cfg['grad_clip'] = 1.0

    # -- Learning rate schedule
    cfg['lr_max'] = 3e-4
    cfg['lr_min'] = 5e-6
    cfg['lr_decay_rate'] = 0.7
    cfg['lr_decay_steps'] = 12000

    # -- Loss weights: [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p, L_x_mse, L_g_inv]
    cfg['loss_weights_p'] = 0.02
    cfg['loss_weights_x'] = 1.0
    cfg['loss_weights_g'] = 0.3
    cfg['loss_weights_reg_g'] = 0.01
    cfg['loss_weights_reg_p'] = 0.02
    cfg['loss_weights_x_mse'] = 0.5
    cfg['loss_weights_g_inv'] = 0.01  # cross-environment g invariance penalty
    cfg['loss_weights'] = torch.tensor([
        cfg['loss_weights_p'], cfg['loss_weights_p'],
        cfg['loss_weights_x'], cfg['loss_weights_x'], cfg['loss_weights_x'],
        cfg['loss_weights_g'],
        cfg['loss_weights_reg_g'], cfg['loss_weights_reg_p'],
        cfg['loss_weights_x_mse'], cfg['loss_weights_g_inv']
    ], dtype=torch.float)

    # -- Loss ramp-up iterations
    cfg['loss_weights_p_g_it'] = 2000
    cfg['loss_weights_g_it'] = 1500  # transition model ramp (faster than p_g, but not so fast it locks in early)
    cfg['loss_weights_reg_p_it'] = 4000
    cfg['loss_weights_reg_g_it'] = float('inf')  # never ramp down g regularization
    cfg['eta_it'] = 3000          # engage memory earlier — adaptive precision handles novelty
    cfg['lambda_it'] = 200

    # -- Inference parameters
    cfg['p2g_sig_val'] = 10000
    cfg['p2g_sig_half_it'] = 3000    # engage memory trust earlier — adaptive precision handles novelty
    cfg['p2g_sig_scale_it'] = 500    # sharper transition
    cfg['p2g_scale_offset'] = 0

    # -- Model architecture
    cfg['g_init_std'] = 0.5
    cfg['g_mem_std'] = 0.1
    cfg['do_sample'] = False

    # -- Domain randomization ranges (multipliers on default values)
    cfg['randomize'] = True
    cfg['terminate_when_unhealthy'] = True   # keep training on active locomotion, not fallen states
    cfg['mass_range'] = (0.25, 4.0)
    cfg['damping_range'] = (0.2, 5.0)
    cfg['friction_range'] = (0.1, 5.0)
    cfg['gear_range'] = (0.25, 4.0)
    cfg['gravity_range'] = (0.4, 2.5)

    # -- Held-out evaluation ranges (beyond training distribution)
    cfg['eval_gravity_range'] = (2.5, 4.0)
    cfg['eval_friction_range'] = (0.02, 0.1)
    cfg['eval_mass_range'] = (4.0, 8.0)

    # -- Transition error EMA for adaptive precision weighting
    cfg['transition_err_ema_decay'] = 0.95  # ~20-step time constant

    # -- Logging
    cfg['log_interval'] = 10
    cfg['save_interval'] = 1000

    return cfg


def iteration_params(iteration, cfg):
    """Compute iteration-dependent hyperparameters (same schedule as original TEM)."""
    eta_target = cfg.get('eta_target', cfg['eta'])
    lambda_target = cfg.get('lambda_target', cfg['lambda_'])
    eta = min((iteration + 1) / cfg['eta_it'], 1) * eta_target
    lamb = min((iteration + 1) / cfg['lambda_it'], 1) * lambda_target

    p2g_scale_offset = 1 / (1 + np.exp(
        (iteration - cfg['p2g_sig_half_it']) / cfg['p2g_sig_scale_it']
    ))

    lr = max(
        cfg['lr_min'] + (cfg['lr_max'] - cfg['lr_min']) *
        (cfg['lr_decay_rate'] ** (iteration / cfg['lr_decay_steps'])),
        cfg['lr_min']
    )

    # Loss weights ramp-up
    ramp_pg = min((iteration + 1) / cfg['loss_weights_p_g_it'], 1)
    ramp_g = min((iteration + 1) / cfg['loss_weights_g_it'], 1)
    L_p_g = ramp_pg * cfg['loss_weights_p']
    L_p_x = ramp_pg * cfg['loss_weights_p'] * (1 - p2g_scale_offset)
    L_x_gen = cfg['loss_weights_x']
    L_x_g = cfg['loss_weights_x']
    L_x_p = cfg['loss_weights_x']
    L_g = ramp_g * cfg['loss_weights_g']
    L_reg_g = (1 - min((iteration + 1) / cfg['loss_weights_reg_g_it'], 1)) * cfg['loss_weights_reg_g']
    L_reg_p = (1 - min((iteration + 1) / cfg['loss_weights_reg_p_it'], 1)) * cfg['loss_weights_reg_p']

    L_x_mse = cfg['loss_weights_x_mse']
    L_g_inv = cfg['loss_weights_g_inv']
    loss_weights = torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p, L_x_mse, L_g_inv])

    return eta, lamb, p2g_scale_offset, lr, loss_weights


def make_rl_config():
    """PPO hyperparameters for RL training on TEM representations."""
    cfg = {}

    # PPO
    cfg['n_envs'] = 8
    cfg['n_steps'] = 2048
    cfg['n_epochs'] = 10
    cfg['minibatch_size'] = 64
    cfg['gamma'] = 0.99
    cfg['gae_lambda'] = 0.95
    cfg['clip_epsilon'] = 0.2
    cfg['lr'] = 3e-4
    cfg['entropy_coeff'] = 0.0
    cfg['value_coeff'] = 0.5
    cfg['max_grad_norm'] = 0.5
    cfg['total_timesteps'] = 1_000_000

    # Policy architecture
    cfg['hidden_dim'] = 256

    # Environment
    cfg['randomize'] = True

    # Logging
    cfg['log_interval'] = 1
    cfg['save_interval'] = 10

    return cfg
