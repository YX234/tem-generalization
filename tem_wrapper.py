"""
Gymnasium environment wrapper that runs frozen TEM inference.
Exposes abstract state g_inf as observations for RL policy training.
"""

import dataclasses
import numpy as np
import torch
import gymnasium as gym

from environment import DomainRandomizedHopper


@dataclasses.dataclass
class TEMState:
    """TEM internal state carried between RL steps."""
    M: list       # [M_gen, M_inf] episodic buffer dicts
    x_inf: list   # n_f filtered observation tensors
    g_inf: list   # n_f abstract state tensors
    transition_err_ema: list  # n_f transition error EMA tensors (or None)

    @staticmethod
    def initial(model):
        """Create fresh TEM state: empty buffers, prior g, zero x_inf."""
        cfg = model.cfg
        n_f = cfg['n_f']
        return TEMState(
            M=[model._init_episodic_buffer(1), model._init_episodic_buffer(1)],
            g_inf=[model.g_init[f].detach().unsqueeze(0).clone() for f in range(n_f)],
            x_inf=[torch.zeros(1, cfg['n_x_f'][f], device=model.device) for f in range(n_f)],
            transition_err_ema=None,
        )


class TEMObservationEnv(gym.Env):
    """Gymnasium env wrapping Hopper + frozen TEM.

    Returns g_inf (abstract state, 54-dim) as observations.
    Manages TEM internal state (episodic memory, temporal filtering)
    across steps and episode boundaries.
    """

    def __init__(self, cfg, tem_model, normalizer, device, seed=None):
        super().__init__()
        self.hopper = DomainRandomizedHopper(cfg, seed=seed)
        self.tem = tem_model
        self.normalizer = normalizer
        self.device = device
        self._g_dim = sum(tem_model.cfg['n_g'])

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._g_dim,), dtype=np.float32
        )
        self.action_space = self.hopper.action_space
        self.state = None

    def reset(self, *, seed=None, options=None):
        obs_raw, info = self.hopper.reset()
        self.state = TEMState.initial(self.tem)

        obs_tensor = self._normalize_to_tensor(obs_raw)
        with torch.no_grad():
            g_inf, x_inf, M, new_ema = self.tem.step_inference(
                obs_tensor, None, self.state.M, self.state.x_inf, self.state.g_inf,
                self.state.transition_err_ema
            )
        self.state = TEMState(M=M, x_inf=x_inf, g_inf=g_inf,
                              transition_err_ema=new_ema)
        return self._g_to_numpy(g_inf), info

    def step(self, action):
        obs_raw, reward, terminated, truncated, info = self.hopper.step(action)

        obs_tensor = self._normalize_to_tensor(obs_raw)
        a_tensor = torch.tensor(
            action, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            g_inf, x_inf, M, new_ema = self.tem.step_inference(
                obs_tensor, a_tensor,
                self.state.M, self.state.x_inf, self.state.g_inf,
                self.state.transition_err_ema
            )
        self.state = TEMState(M=M, x_inf=x_inf, g_inf=g_inf,
                              transition_err_ema=new_ema)
        return self._g_to_numpy(g_inf), reward, terminated, truncated, info

    def _normalize_to_tensor(self, obs_raw):
        obs_normed = self.normalizer.normalize(obs_raw)
        return torch.tensor(
            obs_normed, dtype=torch.float, device=self.device
        ).unsqueeze(0)

    def _g_to_numpy(self, g_inf):
        return torch.cat(g_inf, dim=1).squeeze(0).cpu().numpy()

    def close(self):
        self.hopper.close()
