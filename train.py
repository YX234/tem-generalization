"""
Training loop for TEM world model on domain-randomized Hopper.
No RL policy — trains observation prediction only.
Adapted from torch_tem-main/run.py.
"""

import os
import time
import logging
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import make_config, iteration_params
from tem_model import TEMModel
from environment import DomainRandomizedHopper, RunningNormalizer, collect_trajectories


def setup_logging(run_path):
    logger = logging.getLogger('tem_train')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(os.path.join(run_path, 'train.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(handler)
    # Also log to console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    return logger


def main():
    # Setup
    cfg = make_config()
    np.random.seed(0)
    torch.manual_seed(0)

    # Device selection: GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Create output directories
    run_path = os.path.join(os.path.dirname(__file__), 'runs',
                            time.strftime('%Y-%m-%d_%H-%M-%S'))
    model_path = os.path.join(run_path, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = setup_logging(run_path)
    writer = SummaryWriter(os.path.join(run_path, 'tensorboard'))

    logger.info(f"Device: {device}")
    logger.info(f"Config: n_f={cfg['n_f']}, n_g={cfg['n_g']}, n_p={cfg['n_p']}")
    logger.info(f"Training for {cfg['train_iterations']} iterations, "
                f"batch_size={cfg['batch_size']}, n_rollout={cfg['n_rollout']}")

    # Create model and optimizer
    model = TEMModel(cfg, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr_max'])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Create environments with domain randomization
    envs = [DomainRandomizedHopper(cfg, seed=i) for i in range(cfg['batch_size'])]
    normalizer = RunningNormalizer(cfg['obs_dim'])

    # Collect initial observations
    prev_obs = []
    for env in envs:
        obs, _ = env.reset()
        prev_obs.append(obs)

    prev_iter = None
    carry_resets = None

    # Save original values that iteration_params needs as targets
    # (the loop overwrites cfg['eta'] and cfg['lambda_'] each iteration)
    cfg['eta_target'] = cfg['eta']
    cfg['lambda_target'] = cfg['lambda_']

    # Training loop
    for iteration in range(cfg['train_iterations']):
        start_time = time.time()

        # Get iteration-dependent parameters
        eta, lamb, p2g_offset, lr, loss_weights = iteration_params(iteration, cfg)

        # Update model hyperparameters
        cfg['eta'] = eta
        cfg['lambda_'] = lamb
        cfg['p2g_scale_offset'] = p2g_offset
        model.cfg['eta'] = eta
        model.cfg['lambda_'] = lamb
        model.cfg['p2g_scale_offset'] = p2g_offset

        # Update learning rate
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Collect trajectory chunk (on CPU from envs, then move to device)
        chunk, prev_obs, resets, carry_resets = collect_trajectories(
            envs, cfg['n_rollout'], prev_obs, normalizer, carry_resets)
        for step_data in chunk:
            step_data['obs'] = step_data['obs'].to(device)
            step_data['obs_raw'] = step_data['obs_raw'].to(device)
            step_data['action'] = step_data['action'].to(device)
        loss_weights = loss_weights.to(device)

        # Forward pass (mid-chunk resets are handled inside model.forward)
        steps = model(chunk, prev_iter)

        # Accumulate loss
        loss = torch.tensor(0.0, device=device)
        plot_loss = np.zeros(8)

        for step in steps:
            # step.L is list of 8 tensors each of shape (batch,)
            # loss_weights is (8,) — need to weight each component then sum over batch
            stacked = torch.stack(step.L)  # (8, batch)
            weighted = loss_weights.unsqueeze(1) * stacked  # (8, batch)
            plot_loss += torch.sum(weighted, dim=1).detach().cpu().numpy()  # sum over batch
            loss = loss + torch.sum(weighted)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

        optimizer.step()

        # Save previous iteration for next chunk (detached)
        prev_iter = [steps[-1].detach()]

        # Compute observation prediction accuracy (move to CPU for numpy)
        with torch.no_grad():
            obs_mses = []
            for step in steps:
                mu_pred = step.x_gen[2][0]
                obs_mses.append(torch.mean((mu_pred - step.obs) ** 2).cpu().item())
            avg_obs_mse = np.mean(obs_mses)

            mse_from_p = np.mean([torch.mean((s.x_gen[0][0] - s.obs) ** 2).cpu().item() for s in steps])
            mse_from_g_inf = np.mean([torch.mean((s.x_gen[1][0] - s.obs) ** 2).cpu().item() for s in steps])
            mse_from_g_gen = np.mean([torch.mean((s.x_gen[2][0] - s.obs) ** 2).cpu().item() for s in steps])

        elapsed = time.time() - start_time

        # Logging
        if iteration % cfg['log_interval'] == 0:
            logger.info(
                f"Iter {iteration:5d} | loss {loss.item():.2f} | "
                f"obs_mse {avg_obs_mse:.4f} | "
                f"mse(p) {mse_from_p:.4f} mse(g_inf) {mse_from_g_inf:.4f} mse(g_gen) {mse_from_g_gen:.4f} | "
                f"eta {eta:.4f} lambda {lamb:.6f} lr {lr:.6f} | "
                f"{elapsed:.2f}s"
            )
            logger.info(
                f"  Losses: p_g={plot_loss[0]:.2f} p_x={plot_loss[1]:.2f} "
                f"x_gen={plot_loss[2]:.2f} x_g={plot_loss[3]:.2f} x_p={plot_loss[4]:.2f} "
                f"g={plot_loss[5]:.2f} reg_g={plot_loss[6]:.2f} reg_p={plot_loss[7]:.2f}"
            )

            # TensorBoard
            writer.add_scalar('Loss/total', loss.cpu().item(), iteration)
            for name, val in zip(
                ['p_g', 'p_x', 'x_gen', 'x_g', 'x_p', 'g', 'reg_g', 'reg_p'],
                plot_loss
            ):
                writer.add_scalar(f'Loss/{name}', val, iteration)
            writer.add_scalar('Obs_MSE/from_p', mse_from_p, iteration)
            writer.add_scalar('Obs_MSE/from_g_inf', mse_from_g_inf, iteration)
            writer.add_scalar('Obs_MSE/from_g_gen', mse_from_g_gen, iteration)
            writer.add_scalar('Obs_MSE/average', avg_obs_mse, iteration)
            writer.add_scalar('Params/eta', eta, iteration)
            writer.add_scalar('Params/lambda', lamb, iteration)
            writer.add_scalar('Params/lr', lr, iteration)

        # Save checkpoint
        if iteration % cfg['save_interval'] == 0:
            torch.save(model.state_dict(), os.path.join(model_path, f'tem_{iteration}.pt'))
            torch.save(cfg, os.path.join(model_path, f'cfg_{iteration}.pt'))
            with open(os.path.join(model_path, f'normalizer_{iteration}.pkl'), 'wb') as f:
                pickle.dump(normalizer, f)
            logger.info(f"  Saved checkpoint at iteration {iteration}")

    # Final save
    torch.save(model.state_dict(), os.path.join(model_path, f'tem_final.pt'))
    torch.save(cfg, os.path.join(model_path, f'cfg_final.pt'))
    with open(os.path.join(model_path, 'normalizer_final.pkl'), 'wb') as f:
        pickle.dump(normalizer, f)
    logger.info("Training complete.")

    # Cleanup
    writer.close()
    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
