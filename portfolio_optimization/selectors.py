import math
import logging
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from .covariance import to_tensor, compute_correlation_matrix, get_best_device

logger = logging.getLogger(__name__)

def select_by_diversification(returns_df, all_sector_map, risk_free_rate, top_n=30, min_per_sector=1, min_sharpe_threshold=0.2):
    """
    1. Diversification Selected Strategy (using standard log returns and correlation).
    """
    annual_returns = returns_df.mean() * 252
    annual_volatility = returns_df.std() * np.sqrt(252)
    sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility
    
    corr_all = returns_df.corr()
    portfolio_cols = returns_df.columns
    
    results = []
    for ticker in portfolio_cols:
        if ticker not in corr_all.columns or ticker not in sharpe_ratios.index:
            continue
            
        avg_corr = corr_all.loc[ticker, portfolio_cols].mean()
        max_corr = corr_all.loc[ticker, portfolio_cols].max()
        min_corr = corr_all.loc[ticker, portfolio_cols].min()
        
        sharpe = float(sharpe_ratios.loc[ticker])
        annual_ret = float(annual_returns.loc[ticker]) * 100
        annual_vol = float(annual_volatility.loc[ticker]) * 100
        sector = all_sector_map.get(ticker, 'Unknown')
        
        if pd.isna(sharpe) or pd.isna(avg_corr):
            continue
            
        results.append({
            'Ticker': ticker,
            'Sector': sector,
            'Annual Return (%)': round(annual_ret, 2),
            'Annual Vol (%)': round(annual_vol, 2),
            'Sharpe': round(sharpe, 3),
            'Avg Corr (vs Portfolio)': round(avg_corr, 3),
            'Max Corr': round(max_corr, 3),
            'Min Corr': round(min_corr, 3),
        })
        
    if not results:
        return pd.DataFrame()
        
    candidates_df = pd.DataFrame(results)
    good_candidates = candidates_df[candidates_df['Sharpe'] > min_sharpe_threshold].copy()
    if good_candidates.empty:
        good_candidates = candidates_df.copy() # fallback
        
    good_candidates['Diversification Score'] = (
        good_candidates['Sharpe'] * (1 - good_candidates['Avg Corr (vs Portfolio)'])
    ).round(3)
    good_candidates = good_candidates.sort_values('Diversification Score', ascending=False)
    
    # Stratified selection
    target_sectors = list(set(all_sector_map.values()))
    selected = pd.DataFrame()
    remaining = good_candidates.copy()
    
    for sector in target_sectors:
        sector_stocks = remaining[remaining['Sector'] == sector]
        if not sector_stocks.empty:
            take_n = min(min_per_sector, len(sector_stocks))
            top_sector = sector_stocks.head(take_n)
            selected = pd.concat([selected, top_sector])
            remaining = remaining[~remaining['Ticker'].isin(top_sector['Ticker'])]
            
    slots_left = top_n - len(selected)
    if slots_left > 0 and not remaining.empty:
        fill = remaining.head(slots_left)
        selected = pd.concat([selected, fill])
        
    return selected.sort_values('Diversification Score', ascending=False)

def select_by_lw_diversification(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, top_n=30, min_per_sector=1, min_sharpe_threshold=0.2):
    """
    2. Ledoit-Wolf Diversification Selected Strategy.
    """
    device = shrunk_cov.device
    valid_tickers = list(returns_df.columns)
    
    mean_daily_returns_gpu = torch.mean(to_tensor(returns_df, device), dim=0)
    annual_returns_gpu = mean_daily_returns_gpu * 252
    daily_vols_gpu = torch.sqrt(torch.diag(shrunk_cov))
    annual_volatility_gpu = daily_vols_gpu * math.sqrt(252)
    sharpe_ratios_gpu = (annual_returns_gpu - risk_free_rate) / annual_volatility_gpu
    
    lw_annual_returns = pd.Series(annual_returns_gpu.cpu().numpy(), index=valid_tickers)
    lw_annual_volatility = pd.Series(annual_volatility_gpu.cpu().numpy(), index=valid_tickers)
    lw_sharpe_ratios = pd.Series(sharpe_ratios_gpu.cpu().numpy(), index=valid_tickers)
    lw_corr_all = pd.DataFrame(corr_matrix.cpu().numpy(), index=valid_tickers, columns=valid_tickers)
    
    results = []
    for ticker in valid_tickers:
        if ticker not in lw_corr_all.columns or ticker not in lw_sharpe_ratios.index:
            continue
            
        avg_corr = lw_corr_all.loc[ticker, valid_tickers].mean()
        max_corr = lw_corr_all.loc[ticker, valid_tickers].max()
        min_corr = lw_corr_all.loc[ticker, valid_tickers].min()
        
        sharpe = float(lw_sharpe_ratios.loc[ticker])
        annual_ret = float(lw_annual_returns.loc[ticker]) * 100
        annual_vol = float(lw_annual_volatility.loc[ticker]) * 100
        sector = all_sector_map.get(ticker, 'Unknown')
        
        if pd.isna(sharpe) or pd.isna(avg_corr):
            continue
            
        results.append({
            'Ticker': ticker,
            'Sector': sector,
            'LW Annual Return (%)': round(annual_ret, 2),
            'LW Annual Vol (%)': round(annual_vol, 2),
            'LW Sharpe': round(sharpe, 3),
            'Avg Corr (vs Portfolio)': round(avg_corr, 3),
            'Max Corr': round(max_corr, 3),
            'Min Corr': round(min_corr, 3),
        })
        
    if not results:
        return pd.DataFrame()
        
    candidates_df = pd.DataFrame(results)
    good_candidates = candidates_df[candidates_df['LW Sharpe'] > min_sharpe_threshold].copy()
    if good_candidates.empty:
        good_candidates = candidates_df.copy()
        
    good_candidates['Diversification Score'] = (
        good_candidates['LW Sharpe'] * (1 - good_candidates['Avg Corr (vs Portfolio)'])
    ).round(3)
    good_candidates = good_candidates.sort_values('Diversification Score', ascending=False)
    
    # Stratified selection
    target_sectors = list(set(all_sector_map.values()))
    selected = pd.DataFrame()
    remaining = good_candidates.copy()
    
    for sector in target_sectors:
        sector_stocks = remaining[remaining['Sector'] == sector]
        if not sector_stocks.empty:
            take_n = min(min_per_sector, len(sector_stocks))
            top_sector = sector_stocks.head(take_n)
            selected = pd.concat([selected, top_sector])
            remaining = remaining[~remaining['Ticker'].isin(top_sector['Ticker'])]
            
    slots_left = top_n - len(selected)
    if slots_left > 0 and not remaining.empty:
        fill = remaining.head(slots_left)
        selected = pd.concat([selected, fill])
        
    return selected.sort_values('Diversification Score', ascending=False)

def select_by_hrp_sharpe(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters=30):
    """
    3. LW Sharpe Cluster Strategy (HRP + Max Sharpe per cluster).
    """
    device = shrunk_cov.device
    valid_tickers = list(returns_df.columns)
    
    mean_daily_returns_gpu = torch.mean(to_tensor(returns_df, device), dim=0)
    annual_returns_gpu = mean_daily_returns_gpu * 252
    daily_vols_gpu = torch.sqrt(torch.diag(shrunk_cov))
    annual_volatility_gpu = daily_vols_gpu * math.sqrt(252)
    sharpe_ratios_gpu = (annual_returns_gpu - risk_free_rate) / annual_volatility_gpu
    
    sharpe_ratios_cpu = sharpe_ratios_gpu.cpu().numpy()
    annual_returns_cpu = annual_returns_gpu.cpu().numpy()
    annual_vols_cpu = annual_volatility_gpu.cpu().numpy()
    
    # Hierarchical Clustering Linkage
    dist_matrix_gpu = torch.sqrt(torch.clamp(0.5 * (1.0 - corr_matrix), min=0.0))
    dist_matrix_cpu = dist_matrix_gpu.cpu().numpy()
    condensed_dist = squareform(dist_matrix_cpu, checks=False)
    Z = linkage(condensed_dist, method='ward')
    
    n_cl = min(num_clusters, len(valid_tickers))
    cluster_labels = fcluster(Z, n_cl, criterion='maxclust')
    
    selected_tickers = []
    selected_indices = []
    
    for i in range(1, n_cl + 1):
        cluster_indices = (cluster_labels == i).nonzero()[0]
        if len(cluster_indices) == 0:
            continue
            
        max_sharpe_idx = cluster_indices[sharpe_ratios_cpu[cluster_indices].argmax()]
        selected_indices.append(max_sharpe_idx)
        selected_tickers.append(valid_tickers[max_sharpe_idx])
        
    # Build dataframe summary
    results = []
    corr_matrix_cpu = corr_matrix.cpu().numpy()
    
    for ticker in selected_tickers:
        idx = valid_tickers.index(ticker)
        sharpe = sharpe_ratios_cpu[idx]
        annual_ret = annual_returns_cpu[idx] * 100
        annual_vol = annual_vols_cpu[idx] * 100
        sector = all_sector_map.get(ticker, 'Unknown')
        
        # Corr vs others in selected group
        sub_corr = corr_matrix_cpu[idx, selected_indices]
        avg_corr = np.mean(sub_corr)
        
        mask = np.ones(len(sub_corr), dtype=bool)
        mask[selected_tickers.index(ticker)] = False
        max_corr = np.max(sub_corr[mask]) if len(sub_corr[mask]) > 0 else 1.0
        min_corr = np.min(sub_corr[mask]) if len(sub_corr[mask]) > 0 else 1.0
        
        results.append({
            'Ticker': ticker,
            'Sector': sector,
            'LW Annual Return (%)': round(annual_ret, 2),
            'LW Annual Vol (%)': round(annual_vol, 2),
            'LW Sharpe': round(sharpe, 3),
            'Avg Corr (vs Portfolio)': round(avg_corr, 3),
            'Max Corr': round(max_corr, 3),
            'Min Corr': round(min_corr, 3),
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        df['Diversification Score'] = (df['LW Sharpe'] * (1 - df['Avg Corr (vs Portfolio)'])).round(3)
        return df.sort_values('Diversification Score', ascending=False)
    return df

def select_by_hrp_div(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters=30):
    """
    4. LW Div Cluster Strategy (HRP + Max Diversification Score per cluster).
    """
    device = shrunk_cov.device
    valid_tickers = list(returns_df.columns)
    
    mean_daily_returns_gpu = torch.mean(to_tensor(returns_df, device), dim=0)
    annual_returns_gpu = mean_daily_returns_gpu * 252
    daily_vols_gpu = torch.sqrt(torch.diag(shrunk_cov))
    annual_volatility_gpu = daily_vols_gpu * math.sqrt(252)
    sharpe_ratios_gpu = (annual_returns_gpu - risk_free_rate) / annual_volatility_gpu
    
    # Diversification Score for all assets
    diversification_scores_gpu = sharpe_ratios_gpu * (1 - torch.mean(corr_matrix, dim=0))
    
    sharpe_ratios_cpu = sharpe_ratios_gpu.cpu().numpy()
    annual_returns_cpu = annual_returns_gpu.cpu().numpy()
    annual_vols_cpu = annual_volatility_gpu.cpu().numpy()
    div_scores_cpu = diversification_scores_gpu.cpu().numpy()
    
    # Hierarchical Clustering
    dist_matrix_gpu = torch.sqrt(torch.clamp(0.5 * (1.0 - corr_matrix), min=0.0))
    dist_matrix_cpu = dist_matrix_gpu.cpu().numpy()
    condensed_dist = squareform(dist_matrix_cpu, checks=False)
    Z = linkage(condensed_dist, method='ward')
    
    n_cl = min(num_clusters, len(valid_tickers))
    cluster_labels = fcluster(Z, n_cl, criterion='maxclust')
    
    selected_tickers = []
    selected_indices = []
    
    for i in range(1, n_cl + 1):
        cluster_indices = (cluster_labels == i).nonzero()[0]
        if len(cluster_indices) == 0:
            continue
            
        max_div_idx = cluster_indices[div_scores_cpu[cluster_indices].argmax()]
        selected_indices.append(max_div_idx)
        selected_tickers.append(valid_tickers[max_div_idx])
        
    # Build dataframe summary
    results = []
    corr_matrix_cpu = corr_matrix.cpu().numpy()
    
    for ticker in selected_tickers:
        idx = valid_tickers.index(ticker)
        sharpe = sharpe_ratios_cpu[idx]
        annual_ret = annual_returns_cpu[idx] * 100
        annual_vol = annual_vols_cpu[idx] * 100
        sector = all_sector_map.get(ticker, 'Unknown')
        
        sub_corr = corr_matrix_cpu[idx, selected_indices]
        avg_corr = np.mean(sub_corr)
        
        mask = np.ones(len(sub_corr), dtype=bool)
        mask[selected_tickers.index(ticker)] = False
        max_corr = np.max(sub_corr[mask]) if len(sub_corr[mask]) > 0 else 1.0
        min_corr = np.min(sub_corr[mask]) if len(sub_corr[mask]) > 0 else 1.0
        
        results.append({
            'Ticker': ticker,
            'Sector': sector,
            'LW Annual Return (%)': round(annual_ret, 2),
            'LW Annual Vol (%)': round(annual_vol, 2),
            'LW Sharpe': round(sharpe, 3),
            'Avg Corr (vs Portfolio)': round(avg_corr, 3),
            'Max Corr': round(max_corr, 3),
            'Min Corr': round(min_corr, 3),
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        df['Diversification Score'] = (df['LW Sharpe'] * (1 - df['Avg Corr (vs Portfolio)'])).round(3)
        return df.sort_values('Diversification Score', ascending=False)
    return df

def select_by_aco(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, target_assets=30, num_ants=500, num_iterations=1000):
    """
    5. ACO Selected Strategy (2D-ACO on all tickers directly).
    """
    device = shrunk_cov.device
    tickers = list(returns_df.columns)
    num_assets = len(tickers)
    target_assets = min(target_assets, num_assets)
    
    returns_gpu = to_tensor(returns_df, device)
    mean_returns_gpu = torch.mean(returns_gpu, dim=0)
    vols_gpu = torch.sqrt(torch.diag(shrunk_cov))
    
    ind_ann_return_gpu = mean_returns_gpu * 252
    ind_ann_vol_gpu = vols_gpu * math.sqrt(252)
    ind_sharpe_gpu = (ind_ann_return_gpu - risk_free_rate) / ind_ann_vol_gpu
    
    # heuristic adjustment
    min_sharpe = torch.min(ind_sharpe_gpu)
    heuristic_gpu = ind_sharpe_gpu - min_sharpe + 1e-4
    
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1
    Q = 1.0
    
    pheromones_2d = torch.ones((num_assets, num_assets), dtype=torch.float32, device=device)
    pheromones_start = torch.ones(num_assets, dtype=torch.float32, device=device)
    
    global_best_portfolio = None
    global_best_fitness = -np.inf
    
    for iteration in range(num_iterations):
        pheromones_start = torch.clamp(pheromones_start, min=0.01, max=10.0)
        pheromones_2d = torch.clamp(pheromones_2d, min=0.01, max=10.0)
        
        available_mask = torch.ones((num_ants, num_assets), dtype=torch.bool, device=device)
        paths = torch.zeros((num_ants, target_assets), dtype=torch.long, device=device)
        W = torch.zeros((num_ants, num_assets), dtype=torch.float32, device=device)
        
        for step in range(target_assets):
            if step == 0:
                p_matrix = pheromones_start.unsqueeze(0).expand(num_ants, -1)
            else:
                last_picked = paths[:, step - 1]
                p_matrix = pheromones_2d[last_picked, :]
                
            h_matrix = heuristic_gpu.unsqueeze(0).expand(num_ants, -1)
            
            probs = (p_matrix ** alpha) * (h_matrix ** beta)
            probs = probs * available_mask
            probs = probs + (1e-8 * available_mask)
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            chosen = torch.multinomial(probs, num_samples=1).squeeze(1)
            paths[:, step] = chosen
            W[torch.arange(num_ants, device=device), chosen] = 1.0 / target_assets
            available_mask[torch.arange(num_ants, device=device), chosen] = False
            
        port_returns = torch.matmul(W, mean_returns_gpu) * 252
        port_variances = torch.sum(torch.matmul(W, shrunk_cov) * W, dim=1) * 252
        port_vols = torch.sqrt(port_variances)
        
        fitnesses = (port_returns - risk_free_rate) / port_vols
        
        iter_best_idx = torch.argmax(fitnesses)
        iter_best_fitness = fitnesses[iter_best_idx].item()
        iter_best_path = paths[iter_best_idx]
        
        if iter_best_fitness > global_best_fitness:
            global_best_fitness = iter_best_fitness
            global_best_portfolio = iter_best_path.cpu().numpy()
            
        # Update pheromones
        pheromones_start *= (1 - evaporation_rate)
        pheromones_2d *= (1 - evaporation_rate)
        
        pheromones_start[iter_best_path[0]] += Q * iter_best_fitness
        for i in range(target_assets - 1):
            u = iter_best_path[i]
            v = iter_best_path[i + 1]
            pheromones_2d[u, v] += Q * iter_best_fitness
            pheromones_2d[v, u] += Q * iter_best_fitness
            
    # Compile dataframe summary
    best_tickers = [tickers[i] for i in global_best_portfolio]
    ind_ann_return_cpu = ind_ann_return_gpu.cpu().numpy()
    ind_ann_vol_cpu = ind_ann_vol_gpu.cpu().numpy()
    ind_sharpe_cpu = ind_sharpe_gpu.cpu().numpy()
    corr_matrix_cpu = corr_matrix.cpu().numpy()
    
    results = []
    for idx in global_best_portfolio:
        ticker = tickers[idx]
        sharpe = ind_sharpe_cpu[idx]
        annual_ret = ind_ann_return_cpu[idx] * 100
        annual_vol = ind_ann_vol_cpu[idx] * 100
        sector = all_sector_map.get(ticker, 'Unknown')
        
        sub_corr = corr_matrix_cpu[idx, global_best_portfolio]
        avg_corr = np.mean(sub_corr)
        
        mask = np.ones(len(sub_corr), dtype=bool)
        mask[np.where(global_best_portfolio == idx)[0][0]] = False
        max_corr = np.max(sub_corr[mask]) if len(sub_corr[mask]) > 0 else 1.0
        min_corr = np.min(sub_corr[mask]) if len(sub_corr[mask]) > 0 else 1.0
        
        results.append({
            'Ticker': ticker,
            'Sector': sector,
            'LW Annual Return (%)': round(annual_ret, 2),
            'LW Annual Vol (%)': round(annual_vol, 2),
            'LW Sharpe': round(sharpe, 3),
            'Avg Corr (vs Portfolio)': round(avg_corr, 3),
            'Max Corr': round(max_corr, 3),
            'Min Corr': round(min_corr, 3),
            'Diversification Score': round(sharpe * (1 - avg_corr), 3)
        })
        
    return pd.DataFrame(results).sort_values('Diversification Score', ascending=False)

def select_by_aco_cluster(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters=30, num_ants=500, num_epochs=1000):
    """
    6. ACO Cluster Selected Strategy (Selects 1 representative asset from each HRP cluster using 2D-ACO).
    """
    device = shrunk_cov.device
    valid_tickers = list(returns_df.columns)
    n_features = len(valid_tickers)
    
    returns_gpu = to_tensor(returns_df, device)
    mean_returns_gpu = torch.mean(returns_gpu, dim=0)
    vols_gpu = torch.sqrt(torch.diag(shrunk_cov))
    sharpe_ratios_gpu = math.sqrt(252) * (mean_returns_gpu / vols_gpu)
    
    min_sharpe = torch.min(sharpe_ratios_gpu)
    heuristic_gpu = sharpe_ratios_gpu - min_sharpe + 1e-4
    
    alpha = 1.0
    beta = 2.0
    evaporation = 0.1
    Q = 1.0
    
    # Hierarchical Clustering
    dist_matrix_gpu = torch.sqrt(torch.clamp(0.5 * (1.0 - corr_matrix), min=0.0))
    dist_matrix_cpu = dist_matrix_gpu.cpu().numpy()
    condensed_dist = squareform(dist_matrix_cpu, checks=False)
    Z = linkage(condensed_dist, method='ward')
    
    n_cl = min(num_clusters, len(valid_tickers))
    cluster_labels = fcluster(Z, n_cl, criterion='maxclust')
    
    cluster_indices_list = []
    for i in range(1, n_cl + 1):
        indices = torch.tensor((cluster_labels == i).nonzero()[0], device=device)
        cluster_indices_list.append(indices)
        
    pheromones_2d = torch.ones((n_features, n_features), device=device)
    pheromones_start = torch.ones(n_features, device=device)
    
    best_global_sharpe = -float('inf')
    best_global_portfolio = None
    
    for epoch in range(num_epochs):
        W = torch.zeros((num_ants, n_features), device=device)
        paths = torch.zeros((num_ants, n_cl), dtype=torch.long, device=device)
        last_picked = None
        
        for step, cluster_indices in enumerate(cluster_indices_list):
            h_cluster = heuristic_gpu[cluster_indices]
            
            if step == 0:
                p_cluster = pheromones_start[cluster_indices]
                probs = (p_cluster ** alpha) * (h_cluster ** beta)
                probs = probs / torch.sum(probs)
                
                chosen_local_idx = torch.multinomial(probs, num_samples=num_ants, replacement=True)
                chosen_global_idx = cluster_indices[chosen_local_idx]
            else:
                p_matrix = pheromones_2d[last_picked.unsqueeze(1), cluster_indices.unsqueeze(0)]
                h_matrix = h_cluster.unsqueeze(0).expand(num_ants, -1)
                
                probs = (p_matrix ** alpha) * (h_matrix ** beta)
                probs = probs / torch.sum(probs, dim=1, keepdim=True)
                
                chosen_local_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                chosen_global_idx = cluster_indices[chosen_local_idx]
                
            W[torch.arange(num_ants, device=device), chosen_global_idx] = 1.0 / n_cl
            paths[:, step] = chosen_global_idx
            last_picked = chosen_global_idx
            
        port_returns = torch.matmul(W, mean_returns_gpu) * 252
        port_variances = torch.sum(torch.matmul(W, shrunk_cov) * W, dim=1) * 252
        port_vols = torch.sqrt(port_variances)
        port_sharpes = (port_returns - risk_free_rate) / port_vols
        
        max_sharpe_epoch, best_ant_idx_epoch = torch.max(port_sharpes, dim=0)
        
        if max_sharpe_epoch > best_global_sharpe:
            best_global_sharpe = max_sharpe_epoch.item()
            best_global_portfolio = W[best_ant_idx_epoch].clone()
            
        # Update pheromones
        pheromones_start = pheromones_start * (1 - evaporation)
        pheromones_2d = pheromones_2d * (1 - evaporation)
        
        best_path_epoch = paths[best_ant_idx_epoch]
        pheromones_start[best_path_epoch[0]] += Q * max_sharpe_epoch
        for step in range(n_cl - 1):
            u = best_path_epoch[step]
            v = best_path_epoch[step + 1]
            pheromones_2d[u, v] += Q * max_sharpe_epoch
            
    best_portfolio_indices = (best_global_portfolio > 0).nonzero(as_tuple=True)[0].cpu().numpy()
    
    # Build dataframe summary
    corr_matrix_cpu = corr_matrix.cpu().numpy()
    results = []
    for idx in best_portfolio_indices:
        ticker = valid_tickers[idx]
        sharpe = sharpe_ratios_gpu[idx].item()
        annual_ret = mean_returns_gpu[idx].item() * 252 * 100
        annual_vol = vols_gpu[idx].item() * math.sqrt(252) * 100
        sector = all_sector_map.get(ticker, 'Unknown')
        
        sub_corr = corr_matrix_cpu[idx, best_portfolio_indices]
        avg_corr = sub_corr.mean().item()
        
        mask = torch.ones(len(sub_corr), dtype=torch.bool)
        mask[np.where(best_portfolio_indices == idx)[0][0]] = False
        max_corr = sub_corr[mask].max().item() if len(sub_corr[mask]) > 0 else 1.0
        min_corr = sub_corr[mask].min().item() if len(sub_corr[mask]) > 0 else 1.0
        
        diversification_score = sharpe * (1 - avg_corr)
        
        results.append({
            'Ticker': ticker,
            'Sector': sector,
            'Annual Return (%)': round(annual_ret, 2),
            'Annual Vol (%)': round(annual_vol, 2),
            'Sharpe': round(sharpe, 3),
            'Avg Corr (vs Portfolio)': round(avg_corr, 3),
            'Max Corr': round(max_corr, 3),
            'Min Corr': round(min_corr, 3),
            'Diversification Score': round(diversification_score, 3)
        })
        
    if results:
        return pd.DataFrame(results).sort_values('Diversification Score', ascending=False)
    return pd.DataFrame()
