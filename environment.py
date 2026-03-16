"""
Domain-randomized MuJoCo Hopper environment.
Replaces TEM's world.py (graph-based navigation) with physics-based locomotion.
"""

import numpy as np
import gymnasium as gym
import torch
import copy


class RunningNormalizer:
    """Online mean/std tracker for observation normalization."""

    def __init__(self, dim, clip=10.0):
        self.dim = dim
        self.clip = clip
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self._M2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch):
        """Update running stats with a batch of observations (N, dim)."""
        batch = np.asarray(batch, dtype=np.float64)
        for x in batch:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self._M2 += delta * delta2
        if self.count > 1:
            self.var = self._M2 / (self.count - 1)

    def normalize(self, obs):
        """Normalize observation array using current stats."""
        std = np.sqrt(self.var + 1e-8)
        return np.clip((obs - self.mean) / std, -self.clip, self.clip).astype(np.float32)


class DomainRandomizedHopper:
    """Wrapper around Hopper-v5 that randomizes body parameters on each reset."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.seed = seed
        self.env = gym.make(
            'Hopper-v5',
            terminate_when_unhealthy=cfg.get('terminate_when_unhealthy', True),
        )
        self.rng = np.random.RandomState(seed)

        # Store default model parameters for randomization reference
        model = self.env.unwrapped.model
        self.default_body_mass = model.body_mass.copy()
        self.default_dof_damping = model.dof_damping.copy()
        self.default_geom_friction = model.geom_friction.copy()
        self.default_actuator_gear = model.actuator_gear.copy()
        self.default_gravity = model.opt.gravity.copy()

        self.body_params = {}

    def reset(self):
        """Reset environment with new randomized body parameters."""
        obs, info = self.env.reset()

        if self.cfg['randomize']:
            self._randomize_body()

        self.body_params = self._get_body_params()
        return obs, info

    def step(self, action):
        """Standard gymnasium step."""
        return self.env.step(action)

    def change_physics(self, override_cfg=None):
        """Re-randomize physics mid-episode without resetting simulation state.

        Unlike reset(), this preserves the Hopper's current pose and velocity.
        MuJoCo applies the new parameters on the next step() call.

        Args:
            override_cfg: optional dict with range overrides (e.g. {'gravity_range': (3.0, 4.0)}).
                          If None, uses self.cfg ranges.
        """
        if override_cfg is not None:
            old_cfg = self.cfg
            self.cfg = {**self.cfg, **override_cfg}
        self._randomize_body()
        self.body_params = self._get_body_params()
        if override_cfg is not None:
            self.cfg = old_cfg

    def _randomize_body(self):
        """Randomize physics parameters within configured ranges."""
        model = self.env.unwrapped.model
        mass_lo, mass_hi = self.cfg['mass_range']
        damp_lo, damp_hi = self.cfg['damping_range']
        fric_lo, fric_hi = self.cfg['friction_range']
        gear_lo, gear_hi = self.cfg['gear_range']

        # Randomize body masses (skip worldbody at index 0)
        for i in range(1, len(self.default_body_mass)):
            model.body_mass[i] = self.default_body_mass[i] * self.rng.uniform(mass_lo, mass_hi)

        # Randomize joint damping
        for i in range(len(self.default_dof_damping)):
            model.dof_damping[i] = self.default_dof_damping[i] * self.rng.uniform(damp_lo, damp_hi)

        # Randomize friction (first column is sliding friction)
        for i in range(len(self.default_geom_friction)):
            model.geom_friction[i, 0] = self.default_geom_friction[i, 0] * self.rng.uniform(fric_lo, fric_hi)

        # Randomize actuator gear ratios
        for i in range(len(self.default_actuator_gear)):
            model.actuator_gear[i, 0] = self.default_actuator_gear[i, 0] * self.rng.uniform(gear_lo, gear_hi)

        # Randomize gravity
        if self.cfg.get('gravity_range') is not None:
            lo, hi = self.cfg['gravity_range']
            model.opt.gravity[2] = self.default_gravity[2] * self.rng.uniform(lo, hi)

    def _get_body_params(self):
        """Return current body parameters as a dict for logging."""
        model = self.env.unwrapped.model
        return {
            'body_mass': model.body_mass.copy(),
            'dof_damping': model.dof_damping.copy(),
            'geom_friction': model.geom_friction[:, 0].copy(),
            'actuator_gear': model.actuator_gear[:, 0].copy(),
            'gravity_z': model.opt.gravity[2],
        }

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


def collect_trajectories(envs, n_rollout, prev_obs=None, normalizer=None,
                         carry_resets=None):
    """Collect n_rollout steps from each environment using random actions.

    Args:
        envs: list of DomainRandomizedHopper
        n_rollout: number of steps to collect
        prev_obs: list of previous observations (None if starting fresh)
        normalizer: optional RunningNormalizer for obs standardization
        carry_resets: resets from last step of previous chunk (None if first call)

    Returns:
        chunk: list of n_rollout dicts, each with:
            'obs': (batch, obs_dim) normalized observation tensor
            'action': (batch, action_dim) action tensor
        new_obs: list of current raw observations after collection
        resets: list of booleans indicating which envs were reset during collection
        last_step_resets: resets from the last step (pass to next call as carry_resets)
    """
    batch_size = len(envs)
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    # Initialize observations if needed
    if prev_obs is None:
        prev_obs = []
        for env in envs:
            obs, _ = env.reset()
            prev_obs.append(obs)

    chunk = []
    resets = [False] * batch_size
    # Track resets from the PREVIOUS step so chunk[t].resets means
    # "a reset occurred before this observation" (not after).
    # carry_resets propagates resets from the last step of the previous chunk.
    prev_step_resets = carry_resets if carry_resets is not None else [False] * batch_size

    for step in range(n_rollout):
        # Random actions for world model training (no policy yet)
        actions = [env.action_space.sample() for env in envs]

        # Current observations as tensor
        raw = np.stack(prev_obs)
        if normalizer is not None:
            normalizer.update(raw)
            normed = np.stack([normalizer.normalize(o) for o in prev_obs])
        else:
            normed = raw
        obs_tensor = torch.tensor(normed, dtype=torch.float)
        action_tensor = torch.tensor(np.stack(actions), dtype=torch.float)

        chunk.append({
            'obs': obs_tensor,
            'action': action_tensor,
            'resets': prev_step_resets,
        })

        # Step all environments
        step_resets = [False] * batch_size
        new_obs = []
        for env_i, (env, action) in enumerate(zip(envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                resets[env_i] = True
                step_resets[env_i] = True
            new_obs.append(obs)

        prev_step_resets = step_resets
        prev_obs = new_obs

    return chunk, prev_obs, resets, prev_step_resets
