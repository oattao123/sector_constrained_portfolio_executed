import math
import logging
import numpy as np
import torch
from .covariance import ledoit_wolf_covariance_gpu_dynamic, to_tensor, get_best_device

logger = logging.getLogger(__name__)

def optimize_weights_ebgwo_monte_carlo_entropy(train_returns_gpu, max_weight=0.1, lambda_ent=0.05,
                                               num_wolves=500, iterations=1000, return_convergence=False,
                                               **kwargs):
    """
    EBGWO Optimization + Monte Carlo + Entropy Regularization.
    Uses Ledoit-Wolf Shrinkage Covariance to compute volatility.
    """
    num_days, num_assets = train_returns_gpu.shape
    device = train_returns_gpu.device
    RISK_FREE_RATE = 0.0434

    # log_n prevents division by zero
    log_n = math.log(num_assets) if num_assets > 1 else 1.0
    ST = 0.4
    eps = 1e-10

    # 1. Initialize Wolves (Population)
    wolves = torch.rand((num_wolves, num_assets), dtype=torch.float32, device=device)
    wolves = wolves / wolves.sum(dim=1, keepdim=True)

    best_weights_global = None
    best_fitness_global = -float('inf')

    convergence_best = []
    convergence_avg = []

    for iteration in range(iterations):
        # MONTE CARLO SIMULATION (Bootstrapping)
        mc_indices = torch.randint(0, num_days, (num_days,), device=device)
        mc_returns = train_returns_gpu[mc_indices]

        # 1. Portfolio Return
        mc_mean_returns = mc_returns.mean(dim=0)
        port_ann_return = torch.matmul(wolves, mc_mean_returns) * 252

        # 2. Portfolio Volatility using Ledoit-Wolf
        shrunk_cov, _, _ = ledoit_wolf_covariance_gpu_dynamic(mc_returns)
        port_variance = torch.sum(wolves * torch.matmul(wolves, shrunk_cov), dim=1)
        port_ann_vol = torch.sqrt(port_variance * 252)

        # 3. Sharpe Ratio
        sharpe_ratios = torch.where(port_ann_vol > 0,
                                    (port_ann_return - RISK_FREE_RATE) / port_ann_vol,
                                    torch.zeros_like(port_ann_return))

        # 4. Normalised Entropy
        entropy = -torch.sum(wolves * torch.log(wolves + eps), dim=1)
        norm_entropy = entropy / log_n

        # 5. Penalty (limits max_weight)
        weight_penalties = torch.sum(torch.relu(wolves - max_weight), dim=1) * 100.0

        # TOTAL FITNESS
        fitnesses = sharpe_ratios + (lambda_ent * norm_entropy) - weight_penalties

        # Update convergence
        avg_fitness = fitnesses.mean().item()

        # 6. Select Alpha, Beta, Delta
        sorted_indices = torch.argsort(fitnesses, descending=True)
        alpha_pos = wolves[sorted_indices[0]].clone()
        beta_pos = wolves[sorted_indices[1]].clone()
        delta_pos = wolves[sorted_indices[2]].clone()
        alpha_fitness = fitnesses[sorted_indices[0]].item()

        if alpha_fitness > best_fitness_global:
            best_fitness_global = alpha_fitness
            best_weights_global = alpha_pos.cpu().numpy()

        convergence_best.append(best_fitness_global)
        convergence_avg.append(avg_fitness)

        # 7. Update wolf positions (EBGWO Core)
        a = 2.0 - iteration * (2.0 / iterations)

        r1, r2 = torch.rand_like(wolves), torch.rand_like(wolves)
        A1, C1 = 2 * a * r1 - a, 2 * r2
        X1 = alpha_pos - A1 * torch.abs(C1 * alpha_pos - wolves)

        r1, r2 = torch.rand_like(wolves), torch.rand_like(wolves)
        A2, C2 = 2 * a * r1 - a, 2 * r2
        X2 = beta_pos - A2 * torch.abs(C2 * beta_pos - wolves)

        r1, r2 = torch.rand_like(wolves), torch.rand_like(wolves)
        A3, C3 = 2 * a * r1 - a, 2 * r2

        exploration_mask = torch.rand((num_wolves, 1), device=device) < ST
        random_wolves = wolves[torch.randint(0, num_wolves, (num_wolves,))]
        delta_or_random_pos = torch.where(exploration_mask, random_wolves, delta_pos)
        X3 = delta_or_random_pos - A3 * torch.abs(C3 * delta_or_random_pos - wolves)

        new_wolves = (X1 + X2 + X3) / 3.0
        new_wolves = torch.clamp(new_wolves, 0.0, 1.0)
        sums = new_wolves.sum(dim=1, keepdim=True)
        wolves = torch.where(sums > 0, new_wolves / sums, wolves)

    if return_convergence:
        return best_weights_global, convergence_best, convergence_avg
    return best_weights_global

def optimize_weights_aco_ebgwo(train_returns_gpu, target_assets, heuristic_tensor=None, sector_labels=None,
                               num_iterations=2000, num_agents=500,
                               alpha_aco=1.0, beta_aco=2.0, evaporation_rate=0.1, Q=1.0,
                               PHEROMONE_MIN=0.1, PHEROMONE_MAX=10.0, ST=0.3, patience=300,
                               **kwargs):
    """
    Co-Evolutionary Optimization: ACO (Asset Selection) + EBGWO (Weight Allocation)
    Returns: (weights_numpy, convergence_best_list, convergence_avg_list)
    """
    device = train_returns_gpu.device
    num_days, num_assets = train_returns_gpu.shape
    RISK_FREE_RATE = 0.0434

    # 1. Prepare Correlation Matrix
    mean_returns = train_returns_gpu.mean(dim=0, keepdim=True)
    centered_returns = train_returns_gpu - mean_returns
    cov_matrix = torch.matmul(centered_returns.T, centered_returns) / (num_days - 1)
    std_dev = torch.sqrt(torch.diag(cov_matrix))
    # Avoid division by zero
    std_dev = torch.clamp(std_dev, min=1e-8)
    full_corr_tensor = cov_matrix / torch.outer(std_dev, std_dev)
    full_corr_tensor = torch.clamp(full_corr_tensor, -1.0, 1.0)

    # 2. Handle Heuristics & Sectors
    if heuristic_tensor is None:
        train_ann_ret = train_returns_gpu.mean(dim=0) * 252
        train_ann_vol = train_returns_gpu.std(dim=0) * math.sqrt(252)
        train_ann_vol = torch.clamp(train_ann_vol, min=1e-8)
        heuristic_tensor = torch.clamp(train_ann_ret / train_ann_vol, min=0.01)

    if sector_labels is None:
        sector_labels = torch.zeros(num_assets, dtype=torch.int64, device=device)
        unique_sectors = [0]
        min_per_sector = 0
    else:
        unique_sectors = torch.unique(sector_labels).tolist()
        min_per_sector = 1

    # Pheromones Initialization
    pheromones = torch.ones(num_assets, dtype=torch.float32, device=device)

    # EBGWO Initialization
    wolves = torch.rand((num_agents, target_assets), dtype=torch.float32, device=device)
    wolves = wolves / wolves.sum(dim=1, keepdim=True)

    stagnation_counter = 0

    sqrt_252 = torch.tensor(math.sqrt(252), dtype=torch.float32, device=device)
    global_best_fitness = -float('inf')
    global_best_portfolio = None
    global_best_weights = None

    convergence_curve = []
    avg_convergence_curve = []

    for iteration in range(num_iterations):
        # 1. ACO: Asset Selection Phase
        probabilities = (pheromones ** alpha_aco) * (heuristic_tensor ** beta_aco)
        probabilities = probabilities / probabilities.sum()
        probs_batch = probabilities.unsqueeze(0).expand(num_agents, -1)

        probs_remaining = probs_batch.clone()
        mandatory_selections = []

        for sector_idx in unique_sectors:
            sector_mask = (sector_labels == sector_idx).float()
            probs_sector = probs_remaining * sector_mask.unsqueeze(0)

            epsilon = 1e-8
            probs_sector = probs_sector + (sector_mask.unsqueeze(0) * epsilon)

            available_in_sector = int(sector_mask.sum().item())
            actual_req = min(min_per_sector, available_in_sector)

            if actual_req > 0:
                selected_s = torch.multinomial(probs_sector, actual_req, replacement=False)
                mandatory_selections.append(selected_s)
                probs_remaining.scatter_(1, selected_s, 0.0)

        if mandatory_selections:
            selected_mandatory = torch.cat(mandatory_selections, dim=1)
        else:
            selected_mandatory = torch.empty((num_agents, 0), dtype=torch.int64, device=device)

        num_mandatory = selected_mandatory.shape[1]
        num_remaining = target_assets - num_mandatory

        if num_remaining > 0:
            selected_rem = torch.multinomial(probs_remaining, num_remaining, replacement=False)
            selected_indices = torch.cat([selected_mandatory, selected_rem], dim=1)
        else:
            selected_indices = selected_mandatory[:, :target_assets]

        # 2. Unified Fitness Calculation
        idx_row = selected_indices.unsqueeze(2)
        idx_col = selected_indices.unsqueeze(1)
        batch_corr_matrices = full_corr_tensor[idx_row, idx_col]
        sum_corrs = batch_corr_matrices.sum(dim=(1, 2))

        if target_assets > 1:
            avg_pairwise_corr = (sum_corrs - target_assets) / (target_assets * (target_assets - 1))
        else:
            avg_pairwise_corr = torch.ones_like(sum_corrs)

        decorrelation_fitness = 1.0 - avg_pairwise_corr

        mask = torch.zeros((num_agents, num_assets), device=device)
        mask.scatter_(1, selected_indices, wolves)

        portfolio_returns = torch.matmul(train_returns_gpu, mask.T)
        port_ann_return = portfolio_returns.mean(dim=0) * 252
        port_ann_vol = portfolio_returns.std(dim=0, unbiased=True) * sqrt_252

        portfolio_sharpe = torch.where(port_ann_vol > 0,
                                      (port_ann_return - RISK_FREE_RATE) / port_ann_vol,
                                      torch.zeros_like(port_ann_return))

        unified_fitness = portfolio_sharpe * decorrelation_fitness

        # 3. Global Best Tracking & Stagnation Check
        sorted_indices = torch.argsort(unified_fitness, descending=True)
        best_agent_idx = sorted_indices[0]
        iter_best_fitness = unified_fitness[best_agent_idx].item()

        avg_fitness = unified_fitness.mean().item()

        if iter_best_fitness > global_best_fitness:
            global_best_fitness = iter_best_fitness
            global_best_portfolio = selected_indices[best_agent_idx].clone()
            global_best_weights = wolves[best_agent_idx].clone()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        convergence_curve.append(global_best_fitness)
        avg_convergence_curve.append(avg_fitness)

        # 4. Escape Mechanism
        if stagnation_counter > patience:
            mean_pheromone = pheromones.mean().item()
            pheromones = torch.ones_like(pheromones) * mean_pheromone

            num_reset = num_agents // 2
            reset_indices = sorted_indices[-num_reset:]
            new_random_wolves = torch.rand((num_reset, target_assets), dtype=torch.float32, device=device)
            new_random_wolves = new_random_wolves / new_random_wolves.sum(dim=1, keepdim=True)
            wolves[reset_indices] = new_random_wolves

            stagnation_counter = 0

        # 5. Updates (ACO & EBGWO)
        pheromones *= (1 - evaporation_rate)
        pheromones[selected_indices[best_agent_idx]] += Q * iter_best_fitness
        pheromones = torch.clamp(pheromones, PHEROMONE_MIN, PHEROMONE_MAX)

        alpha_pos = wolves[sorted_indices[0]].clone()
        beta_pos = wolves[sorted_indices[1]].clone()
        delta_pos = wolves[sorted_indices[2]].clone()

        a = 2.0 * (1.0 - (iteration % (num_iterations // 5)) / (num_iterations // 5))

        r1, r2 = torch.rand_like(wolves), torch.rand_like(wolves)
        A1, C1 = 2 * a * r1 - a, 2 * r2
        D_alpha = torch.abs(C1 * alpha_pos - wolves)
        X1 = alpha_pos - A1 * D_alpha

        r1, r2 = torch.rand_like(wolves), torch.rand_like(wolves)
        A2, C2 = 2 * a * r1 - a, 2 * r2
        D_beta = torch.abs(C2 * beta_pos - wolves)
        X2 = beta_pos - A2 * D_beta

        r1, r2 = torch.rand_like(wolves), torch.rand_like(wolves)
        A3, C3 = 2 * a * r1 - a, 2 * r2

        exploration_mask = torch.rand((num_agents, 1), device=device) < ST
        random_wolves = wolves[torch.randint(0, num_agents, (num_agents,))]
        delta_or_random_pos = torch.where(exploration_mask, random_wolves, delta_pos)

        D_delta = torch.abs(C3 * delta_or_random_pos - wolves)
        X3 = delta_or_random_pos - A3 * D_delta

        new_wolves = (X1 + X2 + X3) / 3.0
        new_wolves = torch.clamp(new_wolves, 0.0, 1.0)
        sums = new_wolves.sum(dim=1, keepdim=True)
        wolves = torch.where(sums > 0, new_wolves / sums, wolves)

    final_full_weights = torch.zeros(num_assets, device=device)
    if global_best_portfolio is not None and global_best_weights is not None:
        final_full_weights[global_best_portfolio] = global_best_weights

    return final_full_weights.cpu().numpy(), convergence_curve, avg_convergence_curve


def optimize_weights_pso(train_returns_gpu, max_weight=0.1, lambda_ent=0.05,
                         num_particles=500, iterations=1000,
                         w_start=0.9, w_end=0.4,
                         c1=2.0, c2=2.0,
                         return_convergence=False, **kwargs):
    """
    Particle Swarm Optimization (PSO) for portfolio weight allocation.

    Maximizes entropy-regularized Sharpe Ratio using Ledoit-Wolf shrinkage
    covariance. Inertia weight decays linearly from w_start → w_end to
    balance exploration (early) vs exploitation (late).

    Args:
        train_returns_gpu : torch.Tensor  shape (T, N), daily returns on GPU
        max_weight        : float         per-asset weight cap (soft penalty)
        lambda_ent        : float         entropy regularization strength
        num_particles     : int           swarm size
        iterations        : int           number of PSO iterations
        w_start           : float         initial inertia weight
        w_end             : float         final inertia weight
        c1                : float         cognitive coefficient (personal best)
        c2                : float         social coefficient   (global best)
        return_convergence: bool          if True also return convergence lists

    Returns:
        best_weights      : np.ndarray    shape (N,)
        convergence_best  : list[float]   (only when return_convergence=True)
        convergence_avg   : list[float]   (only when return_convergence=True)
    """
    num_days, num_assets = train_returns_gpu.shape
    device = train_returns_gpu.device
    RISK_FREE_RATE = 0.0434

    log_n = math.log(num_assets) if num_assets > 1 else 1.0
    eps = 1e-10

    # ── Initialize positions (weights) & velocities ──────────────────────────
    positions = torch.rand((num_particles, num_assets), dtype=torch.float32, device=device)
    positions = positions / positions.sum(dim=1, keepdim=True)

    velocities = torch.zeros_like(positions)

    # Personal bests
    pbest_positions = positions.clone()
    pbest_fitness = torch.full((num_particles,), -float('inf'), dtype=torch.float32, device=device)

    # Global best
    gbest_position = positions[0].clone()
    gbest_fitness = -float('inf')

    convergence_best: list[float] = []
    convergence_avg: list[float] = []

    for iteration in range(iterations):
        # ── Ledoit-Wolf shrinkage covariance ─────────────────────────────────
        shrunk_cov, _, _ = ledoit_wolf_covariance_gpu_dynamic(train_returns_gpu)

        # ── Fitness evaluation ────────────────────────────────────────────────
        mean_returns = train_returns_gpu.mean(dim=0)                        # (N,)
        port_ann_return = torch.matmul(positions, mean_returns) * 252       # (P,)

        port_variance = torch.sum(positions * torch.matmul(positions, shrunk_cov), dim=1)
        port_ann_vol = torch.sqrt(port_variance * 252).clamp(min=eps)      # (P,)

        sharpe = (port_ann_return - RISK_FREE_RATE) / port_ann_vol         # (P,)

        entropy = -torch.sum(positions * torch.log(positions + eps), dim=1)
        norm_entropy = entropy / log_n                                      # (P,)

        weight_penalty = torch.sum(torch.relu(positions - max_weight), dim=1) * 100.0

        fitness = sharpe + lambda_ent * norm_entropy - weight_penalty      # (P,)

        # ── Update personal & global bests ───────────────────────────────────
        improved = fitness > pbest_fitness
        pbest_fitness = torch.where(improved, fitness, pbest_fitness)
        pbest_positions = torch.where(improved.unsqueeze(1), positions, pbest_positions)

        iter_best_val, iter_best_idx = fitness.max(dim=0)
        if iter_best_val.item() > gbest_fitness:
            gbest_fitness = iter_best_val.item()
            gbest_position = positions[iter_best_idx].clone()

        convergence_best.append(gbest_fitness)
        convergence_avg.append(fitness.mean().item())

        # ── Inertia decay (linear) ────────────────────────────────────────────
        w = w_start - (w_start - w_end) * (iteration / max(iterations - 1, 1))

        # ── Velocity update ───────────────────────────────────────────────────
        r1 = torch.rand_like(velocities)
        r2 = torch.rand_like(velocities)

        cognitive = c1 * r1 * (pbest_positions - positions)
        social    = c2 * r2 * (gbest_position  - positions)
        velocities = w * velocities + cognitive + social

        # ── Position update & simplex projection ─────────────────────────────
        positions = positions + velocities
        positions = positions.clamp(0.0, 1.0)
        sums = positions.sum(dim=1, keepdim=True)
        positions = torch.where(sums > 0, positions / sums, positions)

    best_weights = gbest_position.cpu().numpy()

    if return_convergence:
        return best_weights, convergence_best, convergence_avg
    return best_weights
