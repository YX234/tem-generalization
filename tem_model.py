"""
TEM-inspired world model for continuous control.
Adapted from torch_tem-main/model.py.

Core components preserved from original TEM:
- Multi-frequency modules with temporal filtering
- Hebbian associative memory with attractor dynamics
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
    sigmas = torch.stack(sigmas, dim=0)
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
                 x_inf=None, g_inf=None, p_inf=None):
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

    def detach(self):
        if self.L is not None:
            self.L = [t.detach() for t in self.L]
        if self.M is not None:
            self.M = [t.detach() for t in self.M]
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
        self.b_x = nn.Parameter(torch.zeros(cfg['n_x_c']))

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

        # -- STATE-DEPENDENT transition model D(a, g) -> delta_g
        # Key adaptation: input includes continuous action AND connected g modules
        transition_in_dims = []
        transition_out_dims = []
        for f_to in range(n_f):
            g_in_dim = sum(cfg['n_g'][f_from] for f_from in range(n_f) if cfg['g_connections'][f_to][f_from])
            transition_in_dims.append(cfg['action_dim'] + g_in_dim)  # action + state
            transition_out_dims.append(g_in_dim * cfg['n_g'][f_to])  # transition matrix

        self.MLP_D_a = MLP(
            transition_in_dims, transition_out_dims,
            activation=[torch.tanh, None],
            hidden_dim=[cfg['d_hidden_dim'] for _ in range(n_f)],
            bias=[True, False]
        )
        # Initialize output layer near zero so initial transition is identity
        self.MLP_D_a.set_weights(1, 0.0)

        # -- Transition uncertainty
        self.MLP_sigma_g_path = MLP(
            cfg['n_g'], cfg['n_g'],
            activation=[torch.tanh, torch.exp],
            hidden_dim=[2 * g for g in cfg['n_g']]
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
                # Mask memories: zero out rows for reset envs
                prev.M = [
                    M * keep.view(-1, 1, 1)
                    for M in prev.M
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

            # Run one TEM iteration
            L, M, g_gen, p_gen, x_gen, g_inf, p_inf, x_inf = self._iteration(
                obs, prev.action, prev.M, prev.x_inf, prev.g_inf
            )

            steps.append(Iteration(
                obs=obs, action=action, L=L, M=M,
                g_gen=g_gen, p_gen=p_gen, x_gen=x_gen,
                x_inf=x_inf, g_inf=g_inf, p_inf=p_inf
            ))

        # Remove initialization step
        steps = steps[1:]
        return steps

    def _iteration(self, obs, a_prev, M_prev, x_prev, g_prev):
        """Single TEM iteration. Mirrors original model.iteration()."""
        # Transition: predict next abstract state from action
        g_gen, g_gen_sigma = self._gen_g(a_prev, g_prev)

        # Inference: infer abstract state from observation + memory
        x_inf, g_inf, p_inf_x, p_inf = self._inference(obs, M_prev, x_prev, (g_gen, g_gen_sigma))

        # Generative: predict observation from inferred state
        x_gen, p_gen = self._generative(M_prev, p_inf, g_inf, g_gen)

        # Update Hebbian memory
        M = [self._hebbian(M_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]
        # Inference memory (separate from generative)
        M.append(self._hebbian(
            M_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1),
            do_hierarchical=False
        ))

        # Compute losses
        L = self._loss(g_gen, p_gen, x_gen, obs, g_inf, p_inf, p_inf_x)

        return L, M, g_gen, p_gen, x_gen, g_inf, p_inf, x_inf

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
        p_x = self._attractor(x_, M_prev[1], cfg['p_retrieve_mask_inf'])

        # Infer abstract state: precision-weighted combination of
        # path integration (g_gen) and memory retrieval (g_mem)
        g = self._inf_g(p_x, g_gen_tuple, obs)

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
        x_p = self._gen_x(p_inf[0])

        # From inferred abstract location -> memory -> grounded -> observation
        p_g_inf = self._gen_p(g_inf, M_prev[0])
        x_g = self._gen_x(p_g_inf[0])

        # From generated abstract location -> memory -> grounded -> observation
        p_g_gen = self._gen_p(g_gen, M_prev[0])
        x_gt = self._gen_x(p_g_gen[0])

        # x_gen is 3 (mu, sigma) tuples; p_gen is from inferred g
        return (x_p, x_g, x_gt), p_g_inf

    # ====================================================================
    # Component functions
    # ====================================================================

    def _gen_g(self, a_prev, g_prev):
        """State-dependent transition: g_new = g_old + D(a, g) * g_old."""
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

        # State-dependent transition: input is [action, g_connected]
        d_input = [torch.cat([a_prev, g_in[f]], dim=1) for f in range(n_f)]
        D_a = self.MLP_D_a(d_input)

        # Reshape to transition matrices and apply
        g_step = []
        for f_to in range(n_f):
            g_in_dim = g_in[f_to].shape[1]
            D_mat = D_a[f_to].reshape(-1, g_in_dim, cfg['n_g'][f_to])
            delta = torch.squeeze(torch.matmul(g_in[f_to].unsqueeze(1), D_mat), 1)
            g_new = torch.tanh(g_prev[f_to] + delta)
            g_step.append(g_new)

        # Transition uncertainty
        sigma_g = self.MLP_sigma_g_path(g_prev)

        return g_step, sigma_g

    def _gen_p(self, g, M_prev):
        """Retrieve grounded location from memory using abstract state as cue."""
        g_ = self._g2g_(g)
        mu_p = self._attractor(g_, M_prev, self.cfg['p_retrieve_mask_gen'])
        return mu_p

    def _gen_x(self, p):
        """Generate observation prediction from highest-freq grounded location."""
        # Sum over entorhinal preferences: p @ W_tile[0]^T
        x_compressed = self.w_x * torch.matmul(p, self.cfg['W_tile'][0].t()) + self.b_x
        # Decode to observation space
        mu, sigma = self.decoder(x_compressed)
        return mu, sigma

    def _inf_g(self, p_x, g_gen_tuple, obs):
        """Infer abstract state from memory + path integration."""
        cfg = self.cfg
        n_f = cfg['n_f']
        g_gen, sigma_g_path = g_gen_tuple

        # From memory-retrieved p, extract g by summing over sensory prefs
        g_downsampled = [
            torch.matmul(p_x[f], cfg['W_repeat'][f].t())
            for f in range(n_f)
        ]
        mu_g_mem = self.MLP_mu_g_mem(g_downsampled)

        # Memory quality measures for uncertainty
        with torch.no_grad():
            x_hat_mu, _ = self._gen_x(p_x[0])
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
            for f in range(n_f)
        ]

        # Precision-weighted combination
        g = []
        for f in range(n_f):
            mu, sigma = inv_var_weight(
                [g_gen[f], mu_g_mem[f]],
                [sigma_g_path[f], sigma_g_mem[f]]
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
        normalised = [normalise(F.relu(x[f] - torch.mean(x[f], dim=-1, keepdim=True))) for f in range(cfg['n_f'])]
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
    # Hebbian memory and attractor dynamics
    # ====================================================================

    def _attractor(self, p_query, M, retrieve_mask):
        """Pattern completion via attractor dynamics in Hebbian memory."""
        cfg = self.cfg
        h_t = torch.cat(p_query, dim=1)
        h_t = self._f_p(h_t)

        for tau in range(cfg['i_attractor']):
            mask = retrieve_mask[tau]
            h_t = ((1 - mask) * h_t +
                    mask * self._f_p(
                        cfg['kappa'] * h_t +
                        torch.squeeze(torch.matmul(h_t.unsqueeze(1), M), 1)
                    ))

        # Split back into frequency modules
        n_p = np.cumsum(np.concatenate(([0], cfg['n_p'])))
        return [h_t[:, n_p[f]:n_p[f+1]] for f in range(cfg['n_f'])]

    def _hebbian(self, M_prev, p_inferred, p_generated, do_hierarchical=True):
        """Hebbian memory update: M = lambda * M + eta * (p+p^)(p-p^)^T."""
        cfg = self.cfg
        M_new = torch.bmm(
            (p_inferred + p_generated).unsqueeze(2),
            (p_inferred - p_generated).unsqueeze(1)
        )
        if do_hierarchical:
            M_new = M_new * cfg['p_update_mask']
        M = torch.clamp(
            cfg['lambda_'] * M_prev + cfg['eta'] * M_new,
            -1, 1
        )
        return M

    # ====================================================================
    # Loss computation
    # ====================================================================

    def _loss(self, g_gen, p_gen, x_gen, obs, g_inf, p_inf, p_inf_x):
        """Compute all 8 loss components."""
        # L_p_g: inferred p vs generated p (from inferred g)
        L_p_g = torch.sum(torch.stack(squared_error(p_inf, p_gen), dim=0), dim=0)

        # L_p_x: inferred p vs memory-retrieved p (from observation)
        L_p_x = torch.sum(torch.stack(squared_error(p_inf, p_inf_x), dim=0), dim=0)

        # L_g: generated g vs inferred g
        L_g = torch.sum(torch.stack(squared_error(g_inf, g_gen), dim=0), dim=0)

        # L_x: observation prediction losses (Gaussian NLL)
        # x_gen is ((mu_p, sig_p), (mu_g, sig_g), (mu_gt, sig_gt))
        L_x_p = self._gaussian_nll(obs, x_gen[0][0], x_gen[0][1])
        L_x_g = self._gaussian_nll(obs, x_gen[1][0], x_gen[1][1])
        L_x_gen = self._gaussian_nll(obs, x_gen[2][0], x_gen[2][1])

        # Regularization
        L_reg_g = torch.sum(torch.stack([torch.sum(g ** 2, dim=1) for g in g_inf], dim=0), dim=0)
        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0)

        return [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]

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
        self.cfg['batch_size'] = batch_size

        # Empty Hebbian memories
        n_p_total = sum(cfg['n_p'])
        M = [
            torch.zeros(batch_size, n_p_total, n_p_total, device=self.device),
            torch.zeros(batch_size, n_p_total, n_p_total, device=self.device),
        ]

        # Prior on abstract state
        g_inf = [self.g_init[f].unsqueeze(0).expand(batch_size, -1).clone() for f in range(n_f)]

        # Zero initial filtered observation
        x_inf = [torch.zeros(batch_size, cfg['n_x_f'][f], device=self.device) for f in range(n_f)]

        return Iteration(obs=obs, action=None, M=M, x_inf=x_inf, g_inf=g_inf)

    def _init_walks(self, prev_iter):
        """Handle episode boundaries: reset memory and state for new episodes."""
        if prev_iter is None:
            return None

        # Check for None actions indicating new episodes
        prev = prev_iter[0]
        if prev.action is None:
            # All new: already initialized correctly
            return prev_iter

        return prev_iter
