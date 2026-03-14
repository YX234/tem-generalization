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
    cfg['n_g_subsampled'] = [12, 10, 8, 6]       # subsampled entorhinal cells per module
    cfg['n_g'] = [3 * g for g in cfg['n_g_subsampled']]  # full entorhinal cells: [36, 30, 24, 18]
    cfg['f_initial'] = [0.99, 0.3, 0.09, 0.03]   # temporal filtering rates

    # -- Sensory encoding
    cfg['n_x_c'] = 16         # compressed observation dim (replaces two-hot)
    cfg['n_x_f'] = [cfg['n_x_c'] for _ in range(cfg['n_f'])]  # filtered obs per freq

    # -- Grounded location (hippocampal place cells): outer product of g_sub x x_f
    cfg['n_p'] = [g * x for g, x in zip(cfg['n_g_subsampled'], cfg['n_x_f'])]  # [192, 160, 128, 96]

    # -- Transition model
    cfg['d_hidden_dim'] = 64  # increased from 20 for continuous dynamics

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
    cfg['train_iterations'] = 20000
    cfg['n_rollout'] = 50         # steps per BPTT chunk
    cfg['batch_size'] = 16
    cfg['episode_length'] = 500   # max steps per episode before reset
    cfg['grad_clip'] = 1.0

    # -- Learning rate schedule
    cfg['lr_max'] = 3e-4
    cfg['lr_min'] = 1e-5
    cfg['lr_decay_rate'] = 0.5
    cfg['lr_decay_steps'] = 4000

    # -- Loss weights: [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]
    cfg['loss_weights_p'] = 1.0
    cfg['loss_weights_x'] = 1.0
    cfg['loss_weights_g'] = 1.0
    cfg['loss_weights_reg_g'] = 0.01
    cfg['loss_weights_reg_p'] = 0.02
    cfg['loss_weights'] = torch.tensor([
        cfg['loss_weights_p'], cfg['loss_weights_p'],
        cfg['loss_weights_x'], cfg['loss_weights_x'], cfg['loss_weights_x'],
        cfg['loss_weights_g'],
        cfg['loss_weights_reg_g'], cfg['loss_weights_reg_p']
    ], dtype=torch.float)

    # -- Loss ramp-up iterations
    cfg['loss_weights_p_g_it'] = 2000
    cfg['loss_weights_reg_p_it'] = 4000
    cfg['loss_weights_reg_g_it'] = float('inf')  # never ramp down g regularization
    cfg['eta_it'] = 8000          # faster ramp than original (less data diversity)
    cfg['lambda_it'] = 200

    # -- Inference parameters
    cfg['p2g_sig_val'] = 10000
    cfg['p2g_sig_half_it'] = 400
    cfg['p2g_sig_scale_it'] = 200
    cfg['p2g_scale_offset'] = 0

    # -- Model architecture
    cfg['g_init_std'] = 0.5
    cfg['g_mem_std'] = 0.1
    cfg['do_sample'] = False

    # -- Domain randomization ranges (multipliers on default values)
    cfg['randomize'] = True
    cfg['mass_range'] = (0.5, 2.0)
    cfg['damping_range'] = (0.5, 2.0)
    cfg['friction_range'] = (0.5, 2.0)
    cfg['gear_range'] = (0.8, 1.2)

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
    L_p_g = ramp_pg * cfg['loss_weights_p']
    L_p_x = ramp_pg * cfg['loss_weights_p'] * (1 - p2g_scale_offset)
    L_x_gen = cfg['loss_weights_x']
    L_x_g = cfg['loss_weights_x']
    L_x_p = cfg['loss_weights_x']
    L_g = ramp_pg * cfg['loss_weights_g']
    L_reg_g = (1 - min((iteration + 1) / cfg['loss_weights_reg_g_it'], 1)) * cfg['loss_weights_reg_g']
    L_reg_p = (1 - min((iteration + 1) / cfg['loss_weights_reg_p_it'], 1)) * cfg['loss_weights_reg_p']

    loss_weights = torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p])

    return eta, lamb, p2g_scale_offset, lr, loss_weights
