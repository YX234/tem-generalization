"""
TEM-inspired world model for continuous control.
Adapted from torch_tem-main/model.py.

Core components preserved from original TEM:
- Multi-frequency modules with temporal filtering
- Episodic buffer memory with attention-based retrieval
- Factored representation: g (abstract state) x (observation) -> p (grounded state)
- Inference/generative dual pathway

Key adaptations for continuous control:
- Learned encoder/decoder replaces two-hot compression
- State-dependent transition D(a, g) replaces action-only D(a)
- Gaussian observation likelihood replaces categorical
- No shiny/OVC modules
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.stats import truncnorm

from encoder import ObservationEncoder, ObservationDecoder


# ============================================================================
# Utility functions (from original TEM utils.py)
# ============================================================================

def inv_var_weight(mus, sigmas):
    """Inverse-variance weighted mean of Gaussians."""
    mus = torch.stack(mus, dim=0)
    sigmas = torch.clamp(torch.stack(sigmas, dim=0), min=1e-4)
    inv_var_var = 1.0 / torch.sum(1.0 / (sigmas ** 2), dim=0)
    inv_var_avg = torch.sum(mus / (sigmas ** 2), dim=0) * inv_var_var
    inv_var_sigma = torch.sqrt(inv_var_var)
    return inv_var_avg, inv_var_sigma


def normalise(x):
    return F.normalize(x, p=2, dim=-1)


def leaky_relu(x):
    return F.leaky_relu(x)


def squared_error(value, target):
    """Per-batch squared error, summed over feature dim."""
    if isinstance(value, list):
        return [0.5 * torch.sum(F.mse_loss(v, t, reduction='none'), dim=-1) for v, t in zip(value, target)]
    return 0.5 * torch.sum(F.mse_loss(value, target, reduction='none'), dim=-1)


# ============================================================================
# MLP module (same as original TEM)
# ============================================================================

class MLP(nn.Module):
    """Multi-module MLP: one 2-layer sub-network per frequency module."""

    def __init__(self, in_dim, out_dim, activation=(F.elu, None), hidden_dim=None, bias=(True, True)):
        super().__init__()
        if isinstance(in_dim, list):
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False

        self.N = len(in_dim)
        self.w = nn.ModuleList()
        for n in range(self.N):
            hidden = hidden_dim[n] if self.is_list and isinstance(hidden_dim, list) else (
                hidden_dim if hidden_dim is not None else int(np.mean([in_dim[n], out_dim[n]]))
            )
            self.w.append(nn.ModuleList([
                nn.Linear(in_dim[n], hidden, bias=bias[0]),
                nn.Linear(hidden, out_dim[n], bias=bias[1]),
            ]))
        self.activation = activation

        with torch.no_grad():
            for layer in range(2):
                for n in range(self.N):
                    nn.init.xavier_normal_(self.w[n][layer].weight)
                    if bias[layer] and self.w[n][layer].bias is not None:
                        self.w[n][layer].bias.fill_(0.0)

    def set_weights(self, layer, value):
        if not isinstance(value, list):
            value = [value for _ in range(self.N)]
        with torch.no_grad():
            for n in range(self.N):
                if isinstance(value[n], torch.Tensor):
                    self.w[n][layer].weight.copy_(value[n])
                else:
                    self.w[n][layer].weight.fill_(value[n])

    def forward(self, data):
        input_data = data if self.is_list else [data]
        output = []
        for n in range(self.N):
            x = self.w[n][0](input_data[n])
            if self.activation[0] is not None:
                x = self.activation[0](x)
            x = self.w[n][1](x)
            if self.activation[1] is not None:
                x = self.activation[1](x)
            output.append(x)
        return output if self.is_list else output[0]


# ============================================================================
# Iteration data container
# ============================================================================

class Iteration:
    """Stores all variables for one TEM step."""

    def __init__(self, obs=None, action=None, L=None, M=None,
                 g_gen=None, p_gen=None, x_gen=None,
                 x_inf=None, g_inf=None, p_inf=None,
                 transition_err_ema=None):
        self.obs = obs
        self.action = action
        self.L = L
        self.M = M
        self.g_gen = g_gen
        self.p_gen = p_gen
        self.x_gen = x_gen      # (mu, sigma) tuples
        self.x_inf = x_inf      # temporally filtered obs per freq
        self.g_inf = g_inf       # inferred abstract state per freq
        self.p_inf = p_inf       # inferred grounded state per freq
        self.transition_err_ema = transition_err_ema  # EMA of obs prediction error per freq

    @staticmethod
    def _detach_buf(buf):
        """Detach an episodic buffer dict (or a plain tensor for backward compat)."""
        if isinstance(buf, dict):
            return {
                'keys': buf['keys'].detach(),
                'values': buf['values'].detach(),
                'count': buf['count'],        # long tensor, no grad
                'write_idx': buf['write_idx'],  # long tensor, no grad
            }
        return buf.detach()

    def detach(self):
        if self.L is not None:
            self.L = [t.detach() for t in self.L]
        if self.M is not None:
            self.M = [self._detach_buf(b) for b in self.M]
        if self.g_gen is not None:
            self.g_gen = [t.detach() for t in self.g_gen]
        if self.p_gen is not None:
            self.p_gen = [t.detach() for t in self.p_gen]
        if self.x_gen is not None:
            self.x_gen = tuple(
                tuple(t.detach() for t in pair) for pair in self.x_gen
            )
        if self.x_inf is not None:
            self.x_inf = [t.detach() for t in self.x_inf]
        if self.g_inf is not None:
            self.g_inf = [t.detach() for t in self.g_inf]
        if self.p_inf is not None:
            self.p_inf = [t.detach() for t in self.p_inf]
        if self.transition_err_ema is not None:
            self.transition_err_ema = [t.detach() for t in self.transition_err_ema]
        return self


# ============================================================================
# Core TEM Model
# ============================================================================

class TEMModel(nn.Module):
    """TEM-inspired world model for continuous control."""

    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.device = device or torch.device('cpu')
        self._init_trainable()
        # Move static config tensors to device
        self._move_config_to_device()

    def _move_config_to_device(self):
        """Move all static tensors in config to the model's device."""
        cfg = self.cfg
        for key in ['W_repeat', 'W_tile', 'g_downsample',
                     'p_retrieve_mask_inf', 'p_retrieve_mask_gen']:
            if isinstance(cfg[key], list):
                cfg[key] = [t.to(self.device) for t in cfg[key]]
        cfg['p_update_mask'] = cfg['p_update_mask'].to(self.device)

    def _init_trainable(self):
        cfg = self.cfg
        n_f = cfg['n_f']

        # -- Observation encoder/decoder
        self.encoder = ObservationEncoder(cfg)
        self.decoder = ObservationDecoder(cfg)

        # -- Temporal filtering: learned alpha per frequency module
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(
                np.log(cfg['f_initial'][f] / (1 - cfg['f_initial'][f])),
                dtype=torch.float
            )) for f in range(n_f)
        ])

        # -- Entorhinal preference weights (for p -> x decoding)
        self.w_x = nn.Parameter(torch.tensor(1.0))
        self.b_x = nn.Parameter(torch.zeros(cfg['n_x_c'] * cfg['n_f']))

        # -- Per-frequency sensory scaling
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(n_f)])

        # -- Initial abstract state prior
        self.g_init = nn.ParameterList([
            nn.Parameter(torch.tensor(
                truncnorm.rvs(-2, 2, size=cfg['n_g'][f], loc=0, scale=cfg['g_init_std']),
                dtype=torch.float
            )) for f in range(n_f)
        ])
        self.logsig_g_init = nn.ParameterList([
            nn.Parameter(torch.tensor(
                truncnorm.rvs(-2, 2, size=cfg['n_g'][f], loc=0, scale=cfg['g_init_std']),
                dtype=torch.float
            )) for f in range(n_f)
        ])

        # -- STATE-DEPENDENT transition model: (action, g_connected) -> delta_g
        # Direct delta prediction replaces matrix formulation for better
        # gradient flow (output dims drop from e.g. 3888 to 36 for module 0).
        transition_in_dims = []
        for f_to in range(n_f):
            g_in_dim = sum(cfg['n_g'][f_from] for f_from in range(n_f) if cfg['g_connections'][f_to][f_from])
            transition_in_dims.append(cfg['action_dim'] + g_in_dim + cfg['n_g'][f_to])

        self.MLP_D_a = MLP(
            transition_in_dims, cfg['n_g'],
            activation=[torch.tanh, None],
            hidden_dim=[cfg['d_hidden_dim'] for _ in range(n_f)],
            bias=[True, True]
        )
        # Initialize output layer near zero so initial delta ≈ 0 (identity transition)
        self.MLP_D_a.set_weights(1, 0.0)

        # -- Transition uncertainty (input = g_prev concatenated with transition_err_ema)
        self.MLP_sigma_g_path = MLP(
            [2 * g for g in cfg['n_g']], cfg['n_g'],
            activation=[torch.tanh, torch.exp],
            hidden_dim=[3 * g for g in cfg['n_g']]
        )

        # -- g inference from memory-retrieved p
        self.MLP_mu_g_mem = MLP(
            cfg['n_g_subsampled'], cfg['n_g'],
            hidden_dim=[2 * g for g in cfg['n_g']]
        )
        # Initialize last layer with small weights
        self.MLP_mu_g_mem.set_weights(-1, [
            torch.tensor(truncnorm.rvs(
                -2, 2,
                size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape),
                loc=0, scale=cfg['g_mem_std']
            ), dtype=torch.float)
            for f in range(n_f)
        ])

        # -- g uncertainty from memory quality measures
        self.MLP_sigma_g_mem = MLP(
            [2 for _ in cfg['n_g_subsampled']], cfg['n_g'],
            activation=[torch.tanh, torch.exp],
            hidden_dim=[2 * g for g in cfg['n_g']]
        )

        # -- Episodic buffer attention temperature (learnable)
        self.log_attn_temp = nn.Parameter(torch.tensor(
            np.log(cfg.get('episodic_attn_init_temp', 1.0)), dtype=torch.float
        ))

        # -- Direct observation -> g pathway (breaks memory bootstrap deadlock)
        # Maps encoded observation directly to abstract state, bypassing memory.
        # Provides observation-specific g from step 1, so p_inf is diverse
        # and episodic memory receives meaningful write signals.
        self.MLP_g_obs = MLP(
            [cfg['n_x_c'] for _ in range(n_f)], cfg['n_g'],
            hidden_dim=[2 * g for g in cfg['n_g']]
        )
        # Learned uncertainty for direct observation path (initialized at sigma ≈ 1.0)
        self.logsig_g_obs = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg['n_g'][f])) for f in range(n_f)
        ])

    # ====================================================================
    # Forward pass
    # ====================================================================

    def forward(self, chunk, prev_iter=None):
        """Process a sequence of observations and actions.

        Args:
            chunk: list of dicts with 'obs' (batch, obs_dim) and 'action' (batch, action_dim)
            prev_iter: previous Iteration from last chunk, or None

        Returns:
            steps: list of Iteration objects
        """
        steps = self._init_walks(prev_iter)

        for step_data in chunk:
            obs = step_data['obs']
            action = step_data['action']
            step_resets = step_data.get('resets', None)

            if steps is None:
                # First step ever: initialize
                batch_size = obs.shape[0]
                steps = [self._init_iteration(obs, batch_size)]

            # Get previous iteration
            prev = steps[-1]

            # Handle mid-chunk episode resets: zero memory and state
            # for environments that terminated on the previous step
            # Use out-of-place ops to avoid breaking autograd
            if step_resets is not None and any(step_resets):
                reset_mask = torch.tensor(step_resets, dtype=torch.bool, device=self.device)
                keep = (~reset_mask).float()
                # Reset episodic buffers for terminated environments
                prev.M = [
                    {
                        'keys': buf['keys'] * keep.view(-1, 1, 1),
                        'values': buf['values'] * keep.view(-1, 1, 1),
                        'count': buf['count'] * (~reset_mask).long(),
                        'write_idx': buf['write_idx'] * (~reset_mask).long(),
                    }
                    for buf in prev.M
                ]
                # Reset g_inf to prior for reset envs, keep others
                prev.g_inf = [
                    keep.unsqueeze(1) * prev.g_inf[f] +
                    reset_mask.float().unsqueeze(1) * self.g_init[f].detach().unsqueeze(0)
                    for f in range(self.cfg['n_f'])
                ]
                # Zero out x_inf for reset envs
                prev.x_inf = [
                    prev.x_inf[f] * keep.unsqueeze(1)
                    for f in range(self.cfg['n_f'])
                ]
                # Zero transition error EMA for reset envs
                if prev.transition_err_ema is not None:
                    prev.transition_err_ema = [
                        keep.unsqueeze(1) * prev.transition_err_ema[f]
                        for f in range(self.cfg['n_f'])
                    ]

            # Run one TEM iteration
            L, M, g_gen, p_gen, x_gen, g_inf, p_inf, x_inf, new_ema = self._iteration(
                obs, prev.action, prev.M, prev.x_inf, prev.g_inf,
                prev.transition_err_ema
            )

            steps.append(Iteration(
                obs=obs, action=action, L=L, M=M,
                g_gen=g_gen, p_gen=p_gen, x_gen=x_gen,
                x_inf=x_inf, g_inf=g_inf, p_inf=p_inf,
                transition_err_ema=new_ema
            ))

        # Remove initialization step
        steps = steps[1:]
        return steps

    def step_inference(self, obs, a_prev, M_prev, x_prev, g_prev,
                       transition_err_ema=None):
        """Single-step inference for RL: returns g_inf without loss or decoding.

        Runs only the inference path + episodic memory update, skipping the
        observation decoder and loss computation for efficiency.

        Args:
            obs: (batch, obs_dim) normalized observation
            a_prev: (batch, action_dim) or None (episode start)
            M_prev: [M_gen, M_inf] episodic memory buffers
            x_prev: list of n_f filtered observation tensors
            g_prev: list of n_f abstract state tensors
            transition_err_ema: list of n_f EMA tensors or None

        Returns:
            g_inf: list of n_f inferred abstract state tensors
            x_inf: list of n_f updated filtered observation tensors
            M: [M_gen, M_inf] updated episodic memory buffers
            new_ema: list of n_f updated transition error EMA tensors
        """
        n_f = self.cfg['n_f']

        # Transition prediction (needed for precision-weighted g inference)
        g_gen, g_gen_sigma = self._gen_g(a_prev, g_prev, transition_err_ema)

        # Inference path: encode, filter, retrieve, infer g and p
        x_inf, g_inf, p_inf_x, p_inf = self._inference(
            obs, M_prev, x_prev, (g_gen, g_gen_sigma)
        )

        # Memory retrieval for update (skip full _generative / decoder)
        p_gen = self._gen_p(g_inf, M_prev[0])

        # Update episodic buffers
        M = [self._episodic_store(M_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]
        M.append(self._episodic_store(
            M_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1),
        ))

        # Update transition error EMA: per-neuron g-space prediction error
        with torch.no_grad():
            terr = [torch.abs(g_gen[f] - g_inf[f]) for f in range(n_f)]
            if transition_err_ema is not None:
                decay = self.cfg['transition_err_ema_decay']
                new_ema = [decay * transition_err_ema[f] + (1 - decay) * terr[f]
                           for f in range(n_f)]
            else:
                new_ema = terr

        return g_inf, x_inf, M, new_ema

    def _iteration(self, obs, a_prev, M_prev, x_prev, g_prev, transition_err_ema=None):
        """Single TEM iteration. Mirrors original model.iteration()."""
        n_f = self.cfg['n_f']

        # Transition: predict next abstract state from action
        g_gen, g_gen_sigma = self._gen_g(a_prev, g_prev, transition_err_ema)

        # Inference: infer abstract state from observation + memory
        x_inf, g_inf, p_inf_x, p_inf = self._inference(obs, M_prev, x_prev, (g_gen, g_gen_sigma))

        # Generative: predict observation from inferred state
        x_gen, p_gen = self._generative(M_prev, p_inf, g_inf, g_gen)

        # Update episodic buffers
        M = [self._episodic_store(M_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]
        M.append(self._episodic_store(
            M_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1),
        ))

        # Compute losses
        L = self._loss(g_gen, g_gen_sigma, p_gen, x_gen, obs, g_inf, p_inf, p_inf_x)

        # Update transition error EMA: per-neuron g-space prediction error.
        # g_gen (transition prediction, no current obs) vs g_inf (observation-informed).
        # In-distribution: g_gen ≈ g_inf → low EMA. OOD: g_gen ≠ g_inf → high EMA.
        with torch.no_grad():
            terr = [torch.abs(g_gen[f] - g_inf[f]) for f in range(n_f)]
            if transition_err_ema is not None:
                decay = self.cfg['transition_err_ema_decay']
                new_ema = [decay * transition_err_ema[f] + (1 - decay) * terr[f]
                           for f in range(n_f)]
            else:
                new_ema = terr

        return L, M, g_gen, p_gen, x_gen, g_inf, p_inf, x_inf, new_ema

    # ====================================================================
    # Inference path (bottom-up)
    # ====================================================================

    def _inference(self, obs, M_prev, x_prev, g_gen_tuple):
        """Infer abstract and grounded state from observation."""
        cfg = self.cfg
        n_f = cfg['n_f']

        # Encode observation (replaces two-hot compression)
        x_c = self.encoder(obs)

        # Temporal filtering per frequency module
        x_f = self._temporal_filter(x_prev, x_c)

        # Prepare for memory: normalize and reshape
        x_ = self._x2x_(x_f)

        # Retrieve grounded location from inference memory
        p_x = self._episodic_retrieve(x_, M_prev[1], cfg['p_retrieve_mask_inf'])

        # Infer abstract state: precision-weighted combination of
        # path integration (g_gen), memory retrieval (g_mem), and direct observation (g_obs)
        g = self._inf_g(p_x, g_gen_tuple, obs, x_f)

        # Prepare abstract state for memory
        g_ = self._g2g_(g)

        # Infer grounded state: element-wise product of g_ and x_
        p = self._inf_p(x_, g_)

        return x_f, g, p_x, p

    # ====================================================================
    # Generative path (top-down)
    # ====================================================================

    def _generative(self, M_prev, p_inf, g_inf, g_gen):
        """Generate observation predictions from inferred/generated states."""
        # From inferred grounded location -> observation
        x_p = self._gen_x(p_inf)

        # From inferred abstract location -> memory -> grounded -> observation
        p_g_inf = self._gen_p(g_inf, M_prev[0])
        x_g = self._gen_x(p_g_inf)

        # From generated abstract location -> memory -> grounded -> observation
        p_g_gen = self._gen_p(g_gen, M_prev[0])
        x_gt = self._gen_x(p_g_gen)

        # x_gen is 3 (mu, sigma) tuples; p_gen is from inferred g
        return (x_p, x_g, x_gt), p_g_inf

    # ====================================================================
    # Component functions
    # ====================================================================

    def _gen_g(self, a_prev, g_prev, transition_err_ema=None):
        """State-dependent transition: g_new = tanh(g_old + delta(a, g_connected, ema_context)).

        When transition_err_ema is provided, it conditions both the delta prediction
        and the uncertainty estimate, enabling online adaptation to physics changes."""
        cfg = self.cfg
        n_f = cfg['n_f']

        # Handle None actions (new episodes)
        if a_prev is None:
            # All new: return prior
            batch_size = g_prev[0].shape[0]
            g_init = [self.g_init[f].unsqueeze(0).expand(batch_size, -1) for f in range(n_f)]
            sigma_init = [torch.exp(self.logsig_g_init[f]).unsqueeze(0).expand(batch_size, -1) for f in range(n_f)]
            return g_init, sigma_init

        # Concatenate connected g modules for each target frequency
        g_in = [
            torch.cat([g_prev[f_from] for f_from in range(n_f) if cfg['g_connections'][f_to][f_from]], dim=1)
            for f_to in range(n_f)
        ]

        # Direct delta prediction: MLP maps (action, g_connected, ema_context) -> delta_g
        if transition_err_ema is not None:
            d_input = [torch.cat([a_prev, g_in[f], transition_err_ema[f]], dim=1)
                       for f in range(n_f)]
        else:
            d_input = [torch.cat([a_prev, g_in[f],
                        torch.zeros(a_prev.shape[0], cfg['n_g'][f], device=a_prev.device)], dim=1)
                       for f in range(n_f)]
        delta = self.MLP_D_a(d_input)

        g_step = [torch.tanh(g_prev[f] + delta[f]) for f in range(n_f)]

        # Transition uncertainty — conditioned on recent prediction accuracy
        if transition_err_ema is not None:
            sigma_input = [torch.cat([g_prev[f], transition_err_ema[f]], dim=1)
                           for f in range(n_f)]
        else:
            sigma_input = [torch.cat([g_prev[f], torch.zeros_like(g_prev[f])], dim=1)
                           for f in range(n_f)]
        sigma_g = self.MLP_sigma_g_path(sigma_input)
        sigma_g = [s + cfg.get('sigma_g_floor', 0.3) for s in sigma_g]

        return g_step, sigma_g

    def _gen_p(self, g, M_prev):
        """Retrieve grounded location from memory using abstract state as cue."""
        g_ = self._g2g_(g)
        mu_p = self._episodic_retrieve(g_, M_prev, self.cfg['p_retrieve_mask_gen'])
        return mu_p

    def _gen_x(self, p_list):
        """Generate observation prediction from all frequency modules."""
        x_parts = [torch.matmul(p_list[f], self.cfg['W_tile'][f].t()) for f in range(self.cfg['n_f'])]
        x_compressed = self.w_x * torch.cat(x_parts, dim=-1) + self.b_x
        mu, sigma = self.decoder(x_compressed)
        return mu, sigma

    def _inf_g(self, p_x, g_gen_tuple, obs, x_f):
        """Infer abstract state from path integration + memory + direct observation."""
        cfg = self.cfg
        n_f = cfg['n_f']
        g_gen, sigma_g_path = g_gen_tuple

        # Source 1: memory-retrieved g
        # From memory-retrieved p, extract g by summing over sensory prefs
        g_downsampled = [
            torch.matmul(p_x[f], cfg['W_repeat'][f].t())
            for f in range(n_f)
        ]
        mu_g_mem = self.MLP_mu_g_mem(g_downsampled)

        # Memory quality measures for uncertainty
        with torch.no_grad():
            x_hat_mu, _ = self._gen_x(p_x)
            err = squared_error(obs, x_hat_mu)

        sigma_g_input = [
            torch.cat([torch.sum(g ** 2, dim=1, keepdim=True),
                        err.unsqueeze(1)], dim=1)
            for g in mu_g_mem
        ]

        mu_g_mem = [torch.tanh(g) for g in mu_g_mem]
        sigma_g_mem_raw = self.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [
            sigma_g_mem_raw[f] + self.cfg['p2g_scale_offset'] * self.cfg['p2g_sig_val']
            + cfg.get('sigma_g_mem_floor', 0.35)
            for f in range(n_f)
        ]

        # Source 2: direct observation -> g (memory-independent)
        # Each frequency module receives its own temporally-filtered x_f[f],
        # preserving the timescale hierarchy (fast modules see raw obs,
        # slow modules see smoothed obs).
        mu_g_obs = self.MLP_g_obs(x_f)
        mu_g_obs = [torch.tanh(g) for g in mu_g_obs]
        sigma_g_obs = [
            torch.exp(self.logsig_g_obs[f]).unsqueeze(0).expand_as(g_gen[f])
            + cfg.get('sigma_g_obs_floor', 0.5)
            for f in range(n_f)
        ]

        # 3-way precision-weighted combination: path integration + memory + observation
        g = []
        for f in range(n_f):
            mu, sigma = inv_var_weight(
                [g_gen[f], mu_g_mem[f], mu_g_obs[f]],
                [sigma_g_path[f], sigma_g_mem[f], sigma_g_obs[f]]
            )
            g.append(mu)

        return g

    def _inf_p(self, x_, g_):
        """Infer grounded location from element-wise product of g_ and x_."""
        p = []
        for f in range(self.cfg['n_f']):
            mu_p = leaky_relu(torch.tanh(g_[f] * x_[f]))
            p.append(mu_p)
        return p

    def _temporal_filter(self, x_prev, x_c):
        """Exponential temporal filtering per frequency module."""
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.cfg['n_f'])]
        return [(1 - alpha[f]) * x_prev[f] + alpha[f] * x_c for f in range(self.cfg['n_f'])]

    def _x2x_(self, x):
        """Prepare sensory input for memory: normalize, reshape via W_tile."""
        cfg = self.cfg
        centered = [x[f] - torch.mean(x[f], dim=-1, keepdim=True) for f in range(cfg['n_f'])]
        normalised = [normalise(torch.tanh(centered[f])) for f in range(cfg['n_f'])]
        return [
            torch.sigmoid(self.w_p[f]) * torch.matmul(normalised[f], cfg['W_tile'][f])
            for f in range(cfg['n_f'])
        ]

    def _g2g_(self, g):
        """Prepare abstract state for memory: downsample, reshape via W_repeat."""
        cfg = self.cfg
        downsampled = [torch.matmul(g[f], cfg['g_downsample'][f]) for f in range(cfg['n_f'])]
        return [torch.matmul(downsampled[f], cfg['W_repeat'][f]) for f in range(cfg['n_f'])]

    def _f_p(self, p):
        """Activation for grounded location: tanh + leaky relu."""
        if isinstance(p, list):
            return [leaky_relu(torch.tanh(p_f)) for p_f in p]
        return leaky_relu(torch.tanh(p))

    # ====================================================================
    # Episodic memory and retrieval
    # ====================================================================

    def _init_episodic_buffer(self, batch_size):
        """Create an empty episodic buffer."""
        K = self.cfg.get('episodic_capacity', 50)
        n_p_total = sum(self.cfg['n_p'])
        return {
            'keys':      torch.zeros(batch_size, K, n_p_total, device=self.device),
            'values':    torch.zeros(batch_size, K, n_p_total, device=self.device),
            'count':     torch.zeros(batch_size, dtype=torch.long, device=self.device),
            'write_idx': torch.zeros(batch_size, dtype=torch.long, device=self.device),
        }

    def _episodic_store(self, M_buf, p_key, p_value, do_hierarchical=True):
        """Store (key, value) pair in episodic buffer. Replaces _hebbian.

        Novelty gate: only writes when the new key is sufficiently different
        from all existing entries (cosine distance > threshold). This prevents
        the buffer from filling with near-identical entries from consecutive
        timesteps, maintaining diverse keys for meaningful retrieval.
        """
        # Ablation: eta=0 means no memory writes
        if self.cfg['eta'] == 0:
            return M_buf

        batch_size = p_key.shape[0]
        K = self.cfg.get('episodic_capacity', 50)

        # Clone to avoid in-place modification (autograd safety)
        new_keys = M_buf['keys'].clone()
        new_values = M_buf['values'].clone()
        new_count = M_buf['count'].clone()
        new_write_idx = M_buf['write_idx'].clone()

        # Novelty gate: skip write if key is too similar to existing entries
        novelty_thresh = self.cfg.get('episodic_novelty_threshold', 0.0)
        if novelty_thresh > 0:
            keys_n = F.normalize(new_keys, dim=2)           # [batch, K, D]
            p_key_n = F.normalize(p_key, dim=1)             # [batch, D]
            sims = torch.bmm(keys_n, p_key_n.unsqueeze(2)).squeeze(2)  # [batch, K]
            # Mask empty slots so they don't count as similar
            slot_idx = torch.arange(K, device=p_key.device).unsqueeze(0)
            valid = slot_idx < new_count.unsqueeze(1)
            sims = sims.masked_fill(~valid, -1.0)
            max_sim = sims.max(dim=1).values                # [batch]
            should_write = (max_sim < (1.0 - novelty_thresh)) | (new_count == 0)
        else:
            should_write = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # Vectorized circular write (only for novel entries)
        b_idx = torch.arange(batch_size, device=self.device)
        new_keys[b_idx, new_write_idx] = torch.where(
            should_write.unsqueeze(1), p_key, new_keys[b_idx, new_write_idx])
        new_values[b_idx, new_write_idx] = torch.where(
            should_write.unsqueeze(1), p_value, new_values[b_idx, new_write_idx])

        new_write_idx = torch.where(should_write, (new_write_idx + 1) % K, new_write_idx)
        new_count = torch.where(should_write, torch.clamp(new_count + 1, max=K), new_count)

        return {
            'keys': new_keys,
            'values': new_values,
            'count': new_count,
            'write_idx': new_write_idx,
        }

    def _episodic_retrieve(self, p_query, M_buf, retrieve_mask):
        """Attention-based retrieval from episodic buffer. Replaces _attractor."""
        cfg = self.cfg

        # Concatenate query across frequency modules
        q = torch.cat(p_query, dim=1)  # [batch, n_p_total]
        q = self._f_p(q)

        keys = M_buf['keys']       # [batch, K, n_p_total]
        values = M_buf['values']   # [batch, K, n_p_total]
        count = M_buf['count']     # [batch]

        n_p = np.cumsum(np.concatenate(([0], cfg['n_p'])))
        K = keys.shape[1]
        batch_size = q.shape[0]

        # Compute per-module blend weights from retrieve mask structure
        blend_weights = []
        for f in range(cfg['n_f']):
            active_iters = sum(1 for mask in retrieve_mask if mask[n_p[f]].item() > 0)
            blend_weights.append(active_iters / cfg['i_attractor'])

        # Handle fully empty buffers (all batch elements)
        if count.max().item() == 0:
            return [q[:, n_p[f]:n_p[f+1]] for f in range(cfg['n_f'])]

        # Temperature-scaled dot-product attention
        temp = torch.exp(self.log_attn_temp)
        d = q.shape[1]
        scale = (d ** 0.5) * temp

        scores = torch.bmm(keys, q.unsqueeze(2)).squeeze(2) / scale  # [batch, K]

        # Mask unused slots
        slot_indices = torch.arange(K, device=q.device).unsqueeze(0)  # [1, K]
        valid_mask = slot_indices < count.unsqueeze(1)                  # [batch, K]
        scores = scores.masked_fill(~valid_mask, float('-inf'))

        # Per-batch has_entries mask for consistent empty-buffer fallback
        has_entries = (count > 0).float().unsqueeze(1)  # [batch, 1]

        attn = F.softmax(scores, dim=1)  # [batch, K]
        # Replace NaN from all-inf rows (empty buffers) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)

        retrieved = torch.bmm(attn.unsqueeze(1), values).squeeze(1)  # [batch, n_p_total]
        retrieved = self._f_p(retrieved)

        # Hierarchical blending with query-memory interpolation
        # For inference: blend_weights = [1.0, 1.0, 1.0, 1.0] → alpha capped at 0.5
        # so the query always contributes at least 50% (prevents discarding sensory signal)
        # For generative: blend_weights = [1.0, 0.75, 0.5, 0.25] → alpha scaled proportionally
        max_memory_weight = 0.5  # cap memory influence to preserve query signal
        result = []
        for f in range(cfg['n_f']):
            q_f = q[:, n_p[f]:n_p[f+1]]
            r_f = retrieved[:, n_p[f]:n_p[f+1]]
            alpha = blend_weights[f] * max_memory_weight
            # For empty-buffer batch elements, use pure query (has_entries=0 → blended=q_f)
            blended = (1 - alpha) * q_f + alpha * r_f
            result.append(has_entries * blended + (1 - has_entries) * q_f)

        return result

    # ====================================================================
    # Loss computation
    # ====================================================================

    def _loss(self, g_gen, g_gen_sigma, p_gen, x_gen, obs, g_inf, p_inf, p_inf_x):
        """Compute all 10 loss components."""
        # L_p_g: inferred p vs generated p (from inferred g)
        L_p_g = torch.sum(torch.stack(squared_error(p_inf, p_gen), dim=0), dim=0)

        # L_p_x: inferred p vs memory-retrieved p (from observation)
        L_p_x = torch.sum(torch.stack(squared_error(p_inf, p_inf_x), dim=0), dim=0)

        # L_g: Gaussian NLL — penalizes both overconfidence (small sigma, large error)
        # and underconfidence (large sigma), naturally calibrating transition uncertainty.
        # Detach g_inf so gradients only update the transition model, not inference.
        n_f = self.cfg['n_f']
        L_g = torch.sum(torch.stack([
            torch.sum(0.5 * (
                torch.log(g_gen_sigma[f] ** 2 + 1e-6) +
                (g_inf[f].detach() - g_gen[f]) ** 2 / (g_gen_sigma[f] ** 2 + 1e-6)
            ), dim=-1)
            for f in range(n_f)
        ], dim=0), dim=0)

        # L_x: observation prediction losses (Gaussian NLL)
        # x_gen is ((mu_p, sig_p), (mu_g, sig_g), (mu_gt, sig_gt))
        L_x_p = self._gaussian_nll(obs, x_gen[0][0], x_gen[0][1])
        L_x_g = self._gaussian_nll(obs, x_gen[1][0], x_gen[1][1])
        L_x_gen = self._gaussian_nll(obs, x_gen[2][0], x_gen[2][1])

        # Regularization
        L_reg_g = torch.sum(torch.stack([torch.sum(g ** 2, dim=1) for g in g_inf], dim=0), dim=0)
        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0)

        # L_x_mse: direct MSE on mu (bypasses sigma to prevent mean-collapse on velocities)
        L_x_mse = torch.sum(F.mse_loss(x_gen[0][0], obs, reduction='none'), dim=-1)

        # L_g_inv: cross-environment g invariance penalty
        # Penalizes each batch element's g from deviating from the batch mean,
        # providing pressure against encoding physics params into g.
        L_g_inv = torch.sum(torch.stack([
            torch.sum((g - g.mean(dim=0, keepdim=True)) ** 2, dim=-1)
            for g in g_inf
        ], dim=0), dim=0)

        return [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p, L_x_mse, L_g_inv]

    def _gaussian_nll(self, target, mu, sigma):
        """Gaussian negative log-likelihood, summed over obs dims, per batch."""
        log2pi = 1.8378770664093453  # log(2*pi)
        return torch.sum(
            0.5 * (log2pi + torch.log(sigma ** 2 + 1e-6) + (target - mu) ** 2 / (sigma ** 2 + 1e-6)),
            dim=-1
        )

    # ====================================================================
    # Initialization helpers
    # ====================================================================

    def _init_iteration(self, obs, batch_size):
        """Create initial Iteration with prior g and empty memory."""
        cfg = self.cfg
        n_f = cfg['n_f']

        # Empty episodic buffers
        M = [self._init_episodic_buffer(batch_size), self._init_episodic_buffer(batch_size)]

        # Prior on abstract state
        g_inf = [self.g_init[f].unsqueeze(0).expand(batch_size, -1).clone() for f in range(n_f)]

        # Zero initial filtered observation
        x_inf = [torch.zeros(batch_size, cfg['n_x_f'][f], device=self.device) for f in range(n_f)]

        return Iteration(obs=obs, action=None, M=M, x_inf=x_inf, g_inf=g_inf)

    def _init_walks(self, prev_iter):
        """Return previous iteration state, or None to trigger initialization."""
        return prev_iter
