"""
Domain-randomized MuJoCo Hopper environment.
Replaces TEM's world.py (graph-based navigation) with physics-based locomotion.
"""

import numpy as np
import gymnasium as gym
import torch
import copy


class DomainRandomizedHopper:
    """Wrapper around Hopper-v5 that randomizes body parameters on each reset."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.seed = seed
        self.env = gym.make('Hopper-v5')
        self.rng = np.random.RandomState(seed)

        # Store default model parameters for randomization reference
        model = self.env.unwrapped.model
        self.default_body_mass = model.body_mass.copy()
        self.default_dof_damping = model.dof_damping.copy()
        self.default_geom_friction = model.geom_friction.copy()
        self.default_actuator_gear = model.actuator_gear.copy()

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

    def _get_body_params(self):
        """Return current body parameters as a dict for logging."""
        model = self.env.unwrapped.model
        return {
            'body_mass': model.body_mass.copy(),
            'dof_damping': model.dof_damping.copy(),
            'geom_friction': model.geom_friction[:, 0].copy(),
            'actuator_gear': model.actuator_gear[:, 0].copy(),
        }

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


def collect_trajectories(envs, n_rollout, prev_obs=None):
    """Collect n_rollout steps from each environment using random actions.

    Args:
        envs: list of DomainRandomizedHopper
        n_rollout: number of steps to collect
        prev_obs: list of previous observations (None if starting fresh)

    Returns:
        chunk: list of n_rollout dicts, each with:
            'obs': (batch, obs_dim) tensor
            'action': list of actions (one per env)
        new_obs: list of current observations after collection
        resets: list of booleans indicating which envs were reset during collection
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

    for step in range(n_rollout):
        # Random actions for world model training (no policy yet)
        actions = [env.action_space.sample() for env in envs]

        # Current observations as tensor
        obs_tensor = torch.tensor(np.stack(prev_obs), dtype=torch.float)
        action_tensor = torch.tensor(np.stack(actions), dtype=torch.float)

        # Track which envs reset at this step (for mid-chunk state resets)
        step_resets = [False] * batch_size

        # Step all environments
        new_obs = []
        for env_i, (env, action) in enumerate(zip(envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                resets[env_i] = True
                step_resets[env_i] = True
            new_obs.append(obs)

        chunk.append({
            'obs': obs_tensor,
            'action': action_tensor,
            'resets': step_resets,
        })

        prev_obs = new_obs

    return chunk, prev_obs, resets
