#!/usr/bin/env python
# coding: utf-8

# # 📊 Multi-Asset Sector-Constrained Portfolio Optimization
# 
# ## เป้าหมาย
# - ดาวน์โหลดข้อมูลสินทรัพย์ 30 ตัว จากหลากหลาย Asset Class & ภูมิภาค
# - รวมหุ้น US, หุ้นไทย, หุ้นจีน, พันธบัตร (US/จีน/ไทย-EM), ทองคำ, Silver
# - จำกัดน้ำหนัก (Weight) ในแต่ละกลุ่ม (Sector/Asset Class) เพื่อป้องกันการกระจุกตัว
# - เปรียบเทียบพอร์ตที่มี Sector Constraint vs ไม่มี

# ## 1. Install & Import Libraries

# In[46]:


#get_ipython().system('pip install yfinance pandas numpy matplotlib seaborn scipy pyswarm -q')


# In[47]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12


# ## 2. กำหนดสินทรัพย์ 30 ตัว (Multi-Asset, Multi-Region)
# 
# ครอบคลุม 9 กลุ่ม: US Tech, Healthcare, Financial, Energy, Consumer, Thai, Chinese, Crypto, Bonds, Commodities

# In[48]:


# กำหนด Asset 30 ตัว แยกตาม Sector / Asset Class
# คัดเลือกเฉพาะตัวที่มี Sharpe Ratio > 0.5 (ย้อนหลัง 4 ปี)
stock_sector_map = {
    # US Technology (5 ตัว)
    'AAPL':      'US Technology',
    'NVDA':      'US Technology',
    'META':      'US Technology',
    'AVGO':      'US Technology',      # Sharpe 1.219
    'GOOGL':     'US Technology',     # Sharpe 0.854

    # US Healthcare (3 ตัว)
    'JNJ':       'US Healthcare',
    'ABBV':      'US Healthcare',
    'LLY':       'US Healthcare',

    # US Financial (3 ตัว)
    'JPM':       'US Financial',
    'GS':        'US Financial',
    'V':         'US Financial',       # Sharpe 0.750

    # US Energy (2 ตัว)
    'XOM':       'US Energy',
    'MPC':       'US Energy',

    # US Consumer (4 ตัว)
    'CAT':       'US Consumer',
    'WMT':       'US Consumer',
    'COST':      'US Consumer',        # Sharpe 0.869
    'MCD':       'US Consumer',        # Sharpe 0.779

    # Thai Stocks (3 ตัว)
    'DELTA.BK':  'Thai Stocks',
    'ADVANC.BK': 'Thai Stocks',
    'KBANK.BK':  'Thai Stocks',

    # Chinese Stocks (2 ตัว)
    'PDD':       'Chinese Stocks',
    'FUTU':      'Chinese Stocks',

    # Crypto (2 ตัว)
    'BTC-USD':   'Crypto',
    'ETH-USD':   'Crypto',             # Sharpe 0.200 but important for diversity

    # Bonds (3 ตัว)
    'HYG':       'Bonds',
    'VCSH':      'Bonds',
    'EMLC':      'Bonds',

    # Commodities (3 ตัว)
    'GLD':       'Commodities',
    'SLV':       'Commodities',
    'COPX':      'Commodities',        # Copper miners, Sharpe 0.743
}

stock_list = list(stock_sector_map.keys())
tickers = stock_list

# Summary
print(f'จำนวน Asset ทั้งหมด: {len(stock_list)} ตัว')
sectors = sorted(set(stock_sector_map.values()))
print(f'จำนวนกลุ่ม: {len(sectors)} กลุ่ม')
print()
for s in sectors:
    members = [t for t in stock_list if stock_sector_map[t] == s]
    print(f'  {s:20s} ({len(members)} ตัว): {", ".join(members)}')


# ## 3. ดาวน์โหลดข้อมูลราคา (4 ปี)

# In[49]:


# กำหนดช่วงเวลา
end_date = datetime.now()
start_date = end_date - timedelta(days=365*4)  # 4 ปี

print(f"ช่วงเวลา: {start_date.strftime('%Y-%m-%d')} ถึง {end_date.strftime('%Y-%m-%d')}")
print(f"กำลังดาวน์โหลดข้อมูล {len(tickers)} ตัว...")
print()

# ดาวน์โหลดราคาปิด (Adjusted Close)
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

# ลบแถวที่มี NaN
data = data.dropna()

print(f"\nได้ข้อมูล {data.shape[0]} Daysทำการ, {data.shape[1]} Asset")
print(f"\nAssetที่ได้: {list(data.columns)}")
data.tail()


# ### Currency Conversion (THB Base)
# Convert all prices to THB for fair comparison:
# - US stocks/ETFs: multiply by USD/THB exchange rate
# - Thai stocks (.BK): already in THB
# - Crypto: multiply by USD/THB

# In[50]:


# ==========================================================
# Currency Conversion: All Prices to THB
# ==========================================================

# Download USD/THB exchange rate
print("Downloading USD/THB exchange rate...")
try:
    usdthb = yf.download("USDTHB=X", start=start_date, end=end_date, auto_adjust=True)["Close"]
    usdthb = usdthb.reindex(data.index).ffill().bfill()
    print(f"  USD/THB range: {float(usdthb.min()):.2f} - {float(usdthb.max()):.2f}")
    print(f"  USD/THB latest: {float(usdthb.iloc[-1]):.2f}")
except Exception as e:
    print(f"  Warning: Could not download USD/THB ({e})")
    print("  Using fixed rate: 35.00 THB/USD")
    usdthb = pd.Series(35.0, index=data.index)

# Convert to THB
data_thb = data.copy()
thai_tickers = [t for t in data.columns if t.endswith(".BK")]
usd_tickers = [t for t in data.columns if not t.endswith(".BK")]

for ticker in usd_tickers:
    data_thb[ticker] = data[ticker] * usdthb.values

print(f"\nConverted {len(usd_tickers)} USD assets to THB")
print(f"Thai assets ({len(thai_tickers)}): already in THB")

# Show latest prices in THB
print("\nLatest Prices (THB):")
print("-" * 40)
latest_thb = data_thb.iloc[-1]
for ticker in sorted(data_thb.columns):
    currency = "THB" if ticker.endswith(".BK") else "USD->THB"
    print(f"  {ticker:12s} {latest_thb[ticker]:>12,.2f} THB  ({currency})")

# IMPORTANT: Use THB-converted data for returns going forward
data = data_thb
print("\n>> All prices converted to THB base for fair comparison")


# ## 4. คำนวณ Returns และ Statistics

# In[51]:


# Daily returns
returns = data.pct_change().dropna()

# Annualized statistics
annual_returns = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)
sharpe_ratios = annual_returns / annual_volatility

# สร้างตาราง summary
summary = pd.DataFrame({
    'Sector': [stock_sector_map[t] for t in returns.columns],
    'Annual Return (%)': (annual_returns * 100).round(2),
    'Annual Vol (%)': (annual_volatility * 100).round(2),
    'Sharpe Ratio': sharpe_ratios.round(3)
})

print("=" * 70)
print("สรุปผลตอบแทนรายปี (Annualized)")
print("=" * 70)
summary.sort_values('Sharpe Ratio', ascending=False)


# In[52]:


# Correlation Heatmap
fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = returns.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 7})
ax.set_title('Correlation Matrix: Multi-Asset Portfolio (US, Thai, China, Bonds, Gold, Silver)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ## 5. Mean-Variance (M-V) Model — ทฤษฎี Markowitz (1952)
# 
# ### ตัวแปรหลัก:
# - **Mean (μ)**: ผลตอบแทนที่คาดหวัง — คำนวณจากค่าเฉลี่ยผลตอบแทนในอดีต
# - **Variance (σ²)**: ความเสี่ยง — วัดการกระจายตัวของผลตอบแทน
# 
# ### 2 เป้าหมายของ M-V Model:
# 1. **Maximize Return** ณ ระดับความเสี่ยงที่ยอมรับได้
# 2. **Minimize Risk** ณ ระดับผลตอบแทนที่ต้องการ
# 
# ### หัวใจสำคัญ:
# > ไม่ใช่แค่เลือกหุ้นกำไรดี แต่คือการดู **Correlation** ระหว่างสินทรัพย์ — เลือกสินทรัพย์ที่ "ไม่ไปด้วยกัน" จะทำให้พอร์ตเสี่ยงน้อยลง

# In[53]:


# =============================================
# Mean-Variance (M-V) Model — Markowitz (1952)
# =============================================

# --- 1) Mean: Expected Return Vector (mu) ---
mu = returns.mean() * 252  # Annualized
print("=" * 60)
print("Mean-Variance Model Components")
print("=" * 60)
print()
print("1) Expected Return Vector (mu) - Annualized:")
print("-" * 40)
for ticker, ret in mu.sort_values(ascending=False).items():
    bar = "#" * int(max(ret * 100, 0) / 3)
    print(f"  {ticker:10s} {ret*100:>8.2f}%  {bar}")

# --- 2) Covariance Matrix (Sigma) ---
cov = returns.cov() * 252  # Annualized
print()
print("2) Covariance Matrix (Sigma) - shape:", cov.shape)
print("-" * 40)
print("  Diagonal = Variance of each asset:")
for ticker in mu.sort_values(ascending=False).index:
    var_val = cov.loc[ticker, ticker]
    vol_val = np.sqrt(var_val) * 100
    print(f"    {ticker:10s}  Var={var_val:.4f}  Vol={vol_val:.1f}%")

# --- 3) Portfolio Formulas ---
print()
print("3) M-V Model Formulas:")
print("-" * 40)
print("  Portfolio Return:  E(Rp) = sum(wi * mui)")
print("  Portfolio Risk:    Vp    = w.T @ Sigma @ w")
print("  Sharpe Ratio:      SR    = E(Rp) / sqrt(Vp)")

# --- 4) Equal-Weight Portfolio Example ---
n = len(mu)
w_eq = np.array([1/n] * n)
port_ret = np.dot(w_eq, mu)
port_var = np.dot(w_eq.T, np.dot(cov, w_eq))
port_vol = np.sqrt(port_var)
port_sr = port_ret / port_vol

print()
print("4) Example: Equal-Weight Portfolio (w = 1/N):")
print("-" * 40)
print(f"  E(Rp) = sum(wi * mui)      = {port_ret*100:.2f}%")
print(f"  Vp    = w.T @ Sigma @ w    = {port_var:.6f}")
print(f"  Vol   = sqrt(Vp)           = {port_vol*100:.2f}%")
print(f"  Sharpe = E(Rp) / Vol       = {port_sr:.3f}")

# --- 5) Optimization with Constraints ---
print()


# In[54]:


# ==========================================================
# หัวใจ M-V Model: Correlation/Covariance ระหว่างAsset
# ==========================================================
# ไม่ใช่แค่เลือกหุ้นกำไรดี แต่คือการดู "ความสัมพันธ์" ระหว่างAsset
# - หากลงหุ้นที่ขึ้นลงพร้อมกัน -> พอร์ตเสี่ยงสูง
# - หากเลือกAssetที่ "ไม่ไปด้วยกัน" -> พอร์ตเสี่ยงน้อยลง ผลตอบแทนเท่าเดิม
# ==========================================================

mu = returns.mean() * 252
cov = returns.cov() * 252
corr = returns.corr()

# --- 1) แสดงตัวอย่าง Correlation สูง vs ต่ำ ---
print("=" * 65)
print("หัวใจ M-V Model: ทำไม Correlation สำคัญ?")
print("=" * 65)

# หา pairs ที่ Corr สูงสุด และ ต่ำสุด
corr_pairs = []
tickers_list = list(corr.columns)
for i in range(len(tickers_list)):
    for j in range(i+1, len(tickers_list)):
        corr_pairs.append((tickers_list[i], tickers_list[j], corr.iloc[i,j]))

corr_pairs.sort(key=lambda x: x[2])

print()
print("Correlation ต่ำสุด (diversify ได้ดี):")
for a, b, c in corr_pairs[:5]:
    print(f"  {a:10s} vs {b:10s}  corr = {c:+.3f}  {"<-- negative!" if c < 0 else ""}")

print()
print("Correlation สูงสุด (เสี่ยงกระจุก):")
for a, b, c in corr_pairs[-5:]:
    print(f"  {a:10s} vs {b:10s}  corr = {c:+.3f}")

# --- 2) พิสูจน์ด้วยตัวเลข: 2-asset portfolio ---
print()
print("=" * 65)
print("พิสูจน์: เปรียบเทียบ 2-asset portfolio")
print("=" * 65)

# หา high-corr pair & low-corr pair
high_a, high_b, high_c = corr_pairs[-1]
low_a, low_b, low_c = corr_pairs[0]

def two_asset_portfolio(t1, t2, w1=0.5):
    w2 = 1 - w1
    ret = w1 * mu[t1] + w2 * mu[t2]
    var = (w1**2 * cov.loc[t1,t1] + w2**2 * cov.loc[t2,t2]
           + 2 * w1 * w2 * cov.loc[t1,t2])
    vol = np.sqrt(var)
    avg_vol = w1 * np.sqrt(cov.loc[t1,t1]) + w2 * np.sqrt(cov.loc[t2,t2])
    diversification = (1 - vol/avg_vol) * 100
    return ret, vol, avg_vol, diversification

ret_h, vol_h, avg_vol_h, div_h = two_asset_portfolio(high_a, high_b)
ret_l, vol_l, avg_vol_l, div_l = two_asset_portfolio(low_a, low_b)

print(f"\nCase 1: HIGH Corr ({high_a} + {high_b}, corr={high_c:+.3f})")
print(f"  Average Vol (ไม่มี diversification) = {avg_vol_h*100:.2f}%")
print(f"  Portfolio Vol (50/50)               = {vol_h*100:.2f}%")
print(f"  Diversification Benefit             = {div_h:.1f}%  <-- น้อย!")

print(f"\nCase 2: LOW Corr ({low_a} + {low_b}, corr={low_c:+.3f})")
print(f"  Average Vol (ไม่มี diversification) = {avg_vol_l*100:.2f}%")
print(f"  Portfolio Vol (50/50)               = {vol_l*100:.2f}%")
print(f"  Diversification Benefit             = {div_l:.1f}%  <-- สูงกว่ามาก!")

print(f"\n>> สรุป: Corr ต่ำ -> ลดความเสี่ยงได้ {div_l:.1f}% vs Corr สูง -> ลดได้แค่ {div_h:.1f}%")

# --- 3) Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap: Asset Class avg correlation
sector_list = sorted(set(stock_sector_map.values()))
sector_corr = pd.DataFrame(index=sector_list, columns=sector_list, dtype=float)
for s1 in sector_list:
    t1s = [t for t in corr.columns if stock_sector_map.get(t) == s1]
    for s2 in sector_list:
        t2s = [t for t in corr.columns if stock_sector_map.get(t) == s2]
        vals = [corr.loc[a,b] for a in t1s for b in t2s if a != b]
        sector_corr.loc[s1, s2] = np.mean(vals) if vals else (1.0 if s1==s2 else 0.0)

sns.heatmap(sector_corr.astype(float), annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, vmin=-0.5, vmax=1, ax=axes[0], linewidths=1)
axes[0].set_title("Avg Correlation Between Asset Groups\n(Lower = Better Diversification)", fontsize=12, fontweight="bold")

# Bar chart: diversification benefit
weights_range = np.linspace(0, 1, 50)
vols_high = [two_asset_portfolio(high_a, high_b, w)[1]*100 for w in weights_range]
vols_low = [two_asset_portfolio(low_a, low_b, w)[1]*100 for w in weights_range]

axes[1].plot(weights_range*100, vols_high, "r-", linewidth=2,
             label=f"High Corr: {high_a}+{high_b} ({high_c:+.2f})")
axes[1].plot(weights_range*100, vols_low, "g-", linewidth=2,
             label=f"Low Corr: {low_a}+{low_b} ({low_c:+.2f})")
axes[1].set_xlabel("Weight of Asset 1 (%)", fontsize=12)
axes[1].set_ylabel("Portfolio Volatility (%)", fontsize=12)
axes[1].set_title("Diversification Effect\n(Low Corr = More Risk Reduction)", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ## 6. กำหนด Sector Weight Limits (ข้อจำกัดน้ำหนัก)
# 
# ### กฎ:
# - **น้ำหนักสูงสุดต่อกลุ่ม**: ไม่เกิน 25%
# - **น้ำหนักต่ำสุดต่อกลุ่ม**: อย่างน้อย 2%
# - **น้ำหนักสูงสุดต่อตัว**: ไม่เกิน 15%
# - **น้ำหนักต่ำสุดต่อตัว**: อย่างน้อย 1%

# In[55]:


# === Sector Weight Constraints Configuration ===

# น้ำหนักรวมของแต่ละกลุ่ม
SECTOR_MAX_WEIGHT = 0.25   # สูงสุด 25% ต่อกลุ่ม
SECTOR_MIN_WEIGHT = 0.02   # ต่ำสุด 2% ต่อกลุ่ม

# น้ำหนักของAssetแต่ละตัว
STOCK_MAX_WEIGHT = 0.15    # สูงสุด 15% ต่อตัว
STOCK_MIN_WEIGHT = 0.01    # ต่ำสุด 1% ต่อตัว

print("=" * 55)
print("📋 Sector / Asset Class Weight Constraints")
print("=" * 55)
print(f"  Sector Max Weight : {SECTOR_MAX_WEIGHT*100:.0f}%")
print(f"  Sector Min Weight : {SECTOR_MIN_WEIGHT*100:.0f}%")
print(f"  Stock Max Weight  : {STOCK_MAX_WEIGHT*100:.0f}%")
print(f"  Stock Min Weight  : {STOCK_MIN_WEIGHT*100:.0f}%")
print()

# สร้าง Sector mapping
unique_sectors = sorted(set(stock_sector_map.values()))
stock_list = list(returns.columns)
n_stocks = len(stock_list)

# สร้าง sector index mapping
sector_stock_indices = {}
for sector in unique_sectors:
    indices = [i for i, s in enumerate(stock_list) if stock_sector_map[s] == sector]
    sector_stock_indices[sector] = indices
    print(f"  {sector:20s}: {[stock_list[i] for i in indices]}")


# In[56]:


# ==========================================================
# Risk-Free Rate (Rf) for Sharpe Ratio
# ==========================================================
# Sharpe Ratio = (E(Rp) - Rf) / sigma_p
# Using US 10-Year Treasury Rate as proxy

import yfinance as yf

try:
    # Download US 10Y Treasury yield
    tnx = yf.download("^TNX", period="5d", auto_adjust=True)["Close"]
    RISK_FREE_RATE = float(tnx.iloc[-1]) / 100  # convert from % to decimal
except:
    RISK_FREE_RATE = 0.045  # fallback: 4.5%

print("=" * 60)
print("Risk-Free Rate Configuration")
print("=" * 60)
print(f"  Rf (annual) = {RISK_FREE_RATE*100:.2f}%  (US 10Y Treasury)")
print(f"  Rf (daily)  = {RISK_FREE_RATE/252*100:.4f}%")
print()
print("  Sharpe Ratio formula:")
print("    SR = (E(Rp) - Rf) / sigma_p")
print(f"    (previously used Rf=0, now Rf={RISK_FREE_RATE*100:.2f}%)")


# ## 7. Portfolio Optimization Functions

# In[57]:


# === Optimization Helper Functions ===

def portfolio_performance(weights, mean_returns, cov_matrix):
    """คำนวณผลตอบแทนและความเสี่ยงของพอร์ต"""
    port_return = np.sum(mean_returns * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - RISK_FREE_RATE) / port_volatility
    return port_return, port_volatility, sharpe


def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    """Objective function: minimize negative Sharpe Ratio"""
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return -sharpe


def minimize_volatility(weights, mean_returns, cov_matrix):
    """Objective function: minimize volatility"""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def get_base_constraints():
    """Constraints ทั่วไป: น้ำหนักรวม = 100%"""
    return [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]


def get_sector_constraints(sector_stock_indices, sector_max, sector_min):
    """สร้าง Constraints จำกัดน้ำหนักรายกลุ่ม (Sector/Asset Class)"""
    constraints = []
    for sector, indices in sector_stock_indices.items():
        # น้ำหนักรวม Sector <= sector_max
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=indices: sector_max - np.sum(w[idx])
        })
        # น้ำหนักรวม Sector >= sector_min
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=indices: np.sum(w[idx]) - sector_min
        })
    return constraints


def optimize_portfolio(mean_returns, cov_matrix, objective_func,
                       sector_constraint=False, n_trials=50):
    """Optimize portfolio with or without sector constraints"""
    n = len(mean_returns)
    
    # Bounds สำหรับAssetแต่ละตัว
    if sector_constraint:
        bounds = tuple((STOCK_MIN_WEIGHT, STOCK_MAX_WEIGHT) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))
    
    # Constraints
    constraints = get_base_constraints()
    if sector_constraint:
        constraints += get_sector_constraints(
            sector_stock_indices, SECTOR_MAX_WEIGHT, SECTOR_MIN_WEIGHT
        )
    
    # Multi-start optimization
    best_result = None
    best_obj = np.inf
    
    for _ in range(n_trials):
        # Random initial weights (normalized)
        init_weights = np.random.dirichlet(np.ones(n))
        
        try:
            result = minimize(
                objective_func,
                init_weights,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except:
            continue
    
    return best_result

print("✅ Functions defined successfully")


# ## 8. Optimize: Maximum Sharpe Ratio Portfolio

# In[58]:


mean_returns = returns.mean()
cov_matrix = returns.cov()

# === 1) ไม่มี Sector Constraint (Unconstrained) ===
print("⏳ Optimizing: Max Sharpe (Unconstrained)...")
result_unconstrained = optimize_portfolio(
    mean_returns, cov_matrix, neg_sharpe_ratio, sector_constraint=False
)
w_unconstrained = result_unconstrained.x
ret_uc, vol_uc, sharpe_uc = portfolio_performance(w_unconstrained, mean_returns, cov_matrix)
print(f"  Return: {ret_uc*100:.2f}% | Vol: {vol_uc*100:.2f}% | Sharpe: {sharpe_uc:.3f}")

# === 2) มี Sector Constraint ===
print("\n⏳ Optimizing: Max Sharpe (Sector Constrained)...")
t0_slsqp = time.time()
result_constrained = optimize_portfolio(
    mean_returns, cov_matrix, neg_sharpe_ratio, sector_constraint=True
)
w_constrained = result_constrained.x
ret_c, vol_c, sharpe_c = portfolio_performance(w_constrained, mean_returns, cov_matrix)
print(f"  Return: {ret_c*100:.2f}% | Vol: {vol_c*100:.2f}% | Sharpe: {sharpe_c:.3f}")

print("\n✅ Optimization complete!")


# ### PSO (Particle Swarm Optimization)
# 
# PSO is a **metaheuristic** optimization algorithm inspired by bird flocking behavior:
# - **Particles** = candidate portfolios (each particle = set of weights)
# - **Swarm** = population of particles exploring the solution space
# - Each particle updates its position based on:
#   1. **Personal best** (pbest) — best solution this particle has found
#   2. **Global best** (gbest) — best solution any particle has found
# 
# **Advantages over SLSQP:**
# - Does not require gradient computation
# - Better at escaping local optima
# - Can handle non-convex objective functions
# 
# **Disadvantage:**
# - Slower (more function evaluations)
# - Cannot enforce equality constraints directly (uses penalty method)

# In[59]:


# ==========================================================
# PSO (Particle Swarm Optimization) — Improved Version
# ==========================================================
# Improvements:
#   1) Swarm size: 100 -> 200 particles
#   2) Max iterations: 200 -> 500
#   3) Hybrid: PSO -> SLSQP refinement
#   4) Better constraint handling (stronger penalty)
#   5) Multiple restarts (best of 3 runs)
# ==========================================================
from pyswarm import pso
from scipy.optimize import minimize as sp_minimize

mu_annual = returns.mean() * 252
cov_annual = returns.cov() * 252
n_assets = len(mu_annual)

def pso_objective(w):
    """Negative Sharpe Ratio with penalty for constraint violations"""
    w_norm = w / np.sum(w) if np.sum(w) > 0 else np.ones(n_assets) / n_assets
    
    port_ret = np.dot(w_norm, mu_annual)
    port_vol = np.sqrt(np.dot(w_norm.T, np.dot(cov_annual, w_norm)))
    
    if port_vol < 1e-10:
        return 999
    
    sharpe = (port_ret - RISK_FREE_RATE) / port_vol
    
    # Stronger penalty for constraint violations
    penalty = 0
    for sector in set(stock_sector_map.values()):
        idx = [i for i, t in enumerate(stock_list) if stock_sector_map[t] == sector]
        sector_w = np.sum(w_norm[idx])
        if sector_w > SECTOR_MAX_WEIGHT:
            penalty += 100 * (sector_w - SECTOR_MAX_WEIGHT)**2
        if sector_w < SECTOR_MIN_WEIGHT:
            penalty += 100 * (SECTOR_MIN_WEIGHT - sector_w)**2
    
    # Per-asset bounds penalty
    for wi in w_norm:
        if wi > STOCK_MAX_WEIGHT:
            penalty += 50 * (wi - STOCK_MAX_WEIGHT)**2
        if wi < STOCK_MIN_WEIGHT:
            penalty += 50 * (STOCK_MIN_WEIGHT - wi)**2
    
    return -sharpe + penalty

def refine_with_slsqp(w_init):
    """Use SLSQP to refine PSO result (Hybrid approach)"""
    def neg_sharpe_rf(w):
        ret = np.dot(w, mu_annual)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_annual, w)))
        return -((ret - RISK_FREE_RATE) / vol) if vol > 0 else 0
    
    bounds = tuple((STOCK_MIN_WEIGHT, STOCK_MAX_WEIGHT) for _ in range(n_assets))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for sector in set(stock_sector_map.values()):
        idx = [i for i, t in enumerate(stock_list) if stock_sector_map[t] == sector]
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: np.sum(w[idx]) - SECTOR_MIN_WEIGHT})
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: SECTOR_MAX_WEIGHT - np.sum(w[idx])})
    
    try:
        res = sp_minimize(neg_sharpe_rf, w_init, method="SLSQP",
                         bounds=bounds, constraints=cons,
                         options={"maxiter": 1000, "ftol": 1e-12})
        if res.success:
            return res.x, -res.fun
    except:
        pass
    return w_init, 0

# PSO bounds
lb = np.ones(n_assets) * STOCK_MIN_WEIGHT
ub = np.ones(n_assets) * STOCK_MAX_WEIGHT

print("=" * 70)
print("PSO (Improved) — Particle Swarm Optimization")
print("=" * 70)
print(f"  Config: 200 particles x 500 iterations x 3 restarts")
print(f"  Hybrid: PSO -> SLSQP refinement")
print(f"  Assets: {n_assets} | Rf: {RISK_FREE_RATE*100:.2f}%")
print()

start_time = time.time()
best_sharpe = -np.inf
best_w = None
run_results = []

for run in range(3):
    # Phase 1: PSO
    w_raw, fopt = pso(pso_objective, lb, ub,
                      swarmsize=200, maxiter=500,
                      minstep=1e-12, minfunc=1e-12,
                      debug=False)
    w_pso_norm = w_raw / np.sum(w_raw)
    pso_ret = np.dot(w_pso_norm, mu_annual)
    pso_vol = np.sqrt(np.dot(w_pso_norm.T, np.dot(cov_annual, w_pso_norm)))
    pso_sr = (pso_ret - RISK_FREE_RATE) / pso_vol
    
    # Phase 2: SLSQP refinement
    w_hybrid, hybrid_sr = refine_with_slsqp(w_pso_norm)

    run_results.append({"run": run+1, "pso_sr": pso_sr, "hybrid_sr": hybrid_sr})
    print(f"  Run {run+1}: PSO Sharpe={pso_sr:.4f} -> Hybrid Sharpe={hybrid_sr:.4f}")
    
    if hybrid_sr > best_sharpe:
        best_sharpe = hybrid_sr
        best_w = w_hybrid

pso_time = time.time() - start_time
w_pso = best_w

pso_ret = np.dot(w_pso, mu_annual)
pso_vol = np.sqrt(np.dot(w_pso.T, np.dot(cov_annual, w_pso)))
pso_sharpe = (pso_ret - RISK_FREE_RATE) / pso_vol

# SLSQP results
slsqp_ret = np.dot(w_constrained, mu_annual)
slsqp_vol = np.sqrt(np.dot(w_constrained.T, np.dot(cov_annual, w_constrained)))
slsqp_sharpe = (slsqp_ret - RISK_FREE_RATE) / slsqp_vol

print(f"\n  Best Hybrid Sharpe: {best_sharpe:.4f} (in {pso_time:.1f} sec)")
print()

# --- Comparison Table ---
print("=" * 70)
print("SLSQP vs PSO (Improved Hybrid): Comparison")
print("=" * 70)
comp_data = {
    "Metric": ["Annual Return (%)", "Annual Volatility (%)", "Sharpe Ratio (Rf-adj)", "Optimization Time"],
    "SLSQP": [f"{slsqp_ret*100:.2f}", f"{slsqp_vol*100:.2f}", f"{slsqp_sharpe:.3f}", "< 1 sec"],
    "PSO Hybrid": [f"{pso_ret*100:.2f}", f"{pso_vol*100:.2f}", f"{pso_sharpe:.3f}", f"{pso_time:.1f} sec"],
}
comp_df = pd.DataFrame(comp_data)
print(comp_df.to_string(index=False))

winner = "PSO Hybrid" if pso_sharpe > slsqp_sharpe else "SLSQP"
diff = abs(pso_sharpe - slsqp_sharpe)
print(f"\n  >> Winner: {winner} (Sharpe diff = {diff:.4f})")
if pso_sharpe >= slsqp_sharpe:
    print("  >> PSO Hybrid found a BETTER solution than SLSQP alone!")
    print("  >> This means SLSQP was stuck in a local optimum.")


# In[60]:


# ==========================================================
# Differential Evolution (DE) — Hybrid Approach
# ==========================================================
from scipy.optimize import differential_evolution

bounds_de = [(STOCK_MIN_WEIGHT, STOCK_MAX_WEIGHT) for _ in range(n_assets)]

print("=" * 70)
print("DE (Differential Evolution) + SLSQP Hybrid")
print("=" * 70)
print(f"  Config: popsize=15 x maxiter=100 x 3 restarts")
print()

start_time = time.time()
best_de_sharpe = -np.inf
best_de_w = None

for run in range(3):
    res_de = differential_evolution(
        pso_objective, 
        bounds_de, 
        maxiter=100, 
        popsize=15, 
        mutation=(0.5, 1.0), 
        recombination=0.7,
        disp=False,
        polish=False  # We do SLSQP polish ourselves
    )
    
    w_de_raw = res_de.x
    w_de_norm = w_de_raw / np.sum(w_de_raw) if np.sum(w_de_raw) > 0 else np.ones(n_assets) / n_assets
    
    # Phase 2: SLSQP refinement
    w_hybrid, hybrid_sr = refine_with_slsqp(w_de_norm)
    
    print(f"  Run {run+1}: DE Sharpe={-res_de.fun:.4f} -> Hybrid Sharpe={hybrid_sr:.4f}")
    
    if hybrid_sr > best_de_sharpe:
        best_de_sharpe = hybrid_sr
        best_de_w = w_hybrid

de_time = time.time() - start_time
w_de = best_de_w

de_ret = np.dot(w_de, mu_annual)
de_vol = np.sqrt(np.dot(w_de.T, np.dot(cov_annual, w_de)))
de_sharpe = (de_ret - RISK_FREE_RATE) / de_vol

print(f"\n  Best DE Hybrid Sharpe: {best_de_sharpe:.4f} (in {de_time:.1f} sec)")
print()

# --- Ultimate Comparison Table ---
print("=" * 80)
print("ULTIMATE COMPARISON: SLSQP vs PSO Hybrid vs DE Hybrid")
print("=" * 80)
comp_data = {
    "Metric": ["Annual Return (%)", "Annual Volatility (%)", "Sharpe Ratio (Rf-adj)", "Time"],
    "SLSQP": [f"{slsqp_ret*100:.2f}", f"{slsqp_vol*100:.2f}", f"{slsqp_sharpe:.3f}", "< 1s"],
    "PSO Hybrid": [f"{pso_ret*100:.2f}", f"{pso_vol*100:.2f}", f"{pso_sharpe:.3f}", f"{pso_time:.1f}s"],
    "DE Hybrid": [f"{de_ret*100:.2f}", f"{de_vol*100:.2f}", f"{de_sharpe:.3f}", f"{de_time:.1f}s"],
}
comp_df = pd.DataFrame(comp_data)
print(comp_df.to_string(index=False))

scores = {"SLSQP": slsqp_sharpe, "PSO": pso_sharpe, "DE": de_sharpe}
winner = max(scores, key=scores.get)
print(f"\n  🏆 ULTIMATE WINNER: {winner} (Sharpe = {scores[winner]:.4f})")


# In[61]:


# === SLSQP vs PSO vs DE: Visualization ===

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# 1) Weight comparison bar chart
x = np.arange(n_assets)
bw = 0.25
axes[0].barh(x - bw, w_constrained * 100, bw, label="SLSQP", color="#2196F3", alpha=0.9)
axes[0].barh(x, w_pso * 100, bw, label="PSO", color="#FF9800", alpha=0.9)
axes[0].barh(x + bw, w_de * 100, bw, label="DE", color="#4CAF50", alpha=0.9)
axes[0].set_yticks(x)
axes[0].set_yticklabels(stock_list, fontsize=7)
axes[0].set_xlabel("Weight (%)")
axes[0].set_title("SLSQP vs PSO vs DE: Asset Weights", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="x")

# 2) Sector weights comparison
sectors = sorted(set(stock_sector_map.values()))
slsqp_sec = [sum(w_constrained[i] for i, t in enumerate(stock_list) if stock_sector_map[t] == s) * 100 for s in sectors]
pso_sec = [sum(w_pso[i] for i, t in enumerate(stock_list) if stock_sector_map[t] == s) * 100 for s in sectors]
de_sec = [sum(w_de[i] for i, t in enumerate(stock_list) if stock_sector_map[t] == s) * 100 for s in sectors]
x_sec = np.arange(len(sectors))
axes[1].barh(x_sec - bw, slsqp_sec, bw, label="SLSQP", color="#2196F3", alpha=0.9)
axes[1].barh(x_sec, pso_sec, bw, label="PSO", color="#FF9800", alpha=0.9)
axes[1].barh(x_sec + bw, de_sec, bw, label="DE", color="#4CAF50", alpha=0.9)
axes[1].axvline(x=SECTOR_MAX_WEIGHT*100, color="red", linestyle="--", alpha=0.5, label=f"Max {SECTOR_MAX_WEIGHT*100:.0f}%")
axes[1].axvline(x=SECTOR_MIN_WEIGHT*100, color="green", linestyle="--", alpha=0.5, label=f"Min {SECTOR_MIN_WEIGHT*100:.0f}%")
axes[1].set_yticks(x_sec)
axes[1].set_yticklabels(sectors, fontsize=9)
axes[1].set_xlabel("Weight (%)")
axes[1].set_title("SLSQP vs PSO vs DE: Sector Weights", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="x")

# 3) Risk-Return scatter
axes[2].scatter(slsqp_vol*100, slsqp_ret*100, s=300, c="#2196F3", edgecolors="black",
               linewidth=2, zorder=5, label=f"SLSQP (SR={slsqp_sharpe:.3f})")
axes[2].scatter(pso_vol*100, pso_ret*100, s=300, c="#FF9800", edgecolors="black",
               linewidth=2, zorder=5, marker="D", label=f"PSO (SR={pso_sharpe:.3f})")
axes[2].scatter(de_vol*100, de_ret*100, s=400, c="#4CAF50", edgecolors="black",
               linewidth=2, zorder=6, marker="*", label=f"DE (SR={de_sharpe:.3f})")
# Individual assets
for i, t in enumerate(stock_list):
    axes[2].scatter(np.sqrt(cov_annual.iloc[i,i])*100, mu_annual.iloc[i]*100,
                   s=30, c="gray", alpha=0.4)
    axes[2].annotate(t, (np.sqrt(cov_annual.iloc[i,i])*100, mu_annual.iloc[i]*100),
                    fontsize=5, alpha=0.5)
axes[2].set_xlabel("Volatility (%)", fontsize=12)
axes[2].set_ylabel("Return (%)", fontsize=12)
axes[2].set_title("Risk-Return: Ultimate Battle", fontsize=12, fontweight="bold")
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle("Optimizer Battle: SLSQP vs PSO vs DE",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()

# ==================================================================
# PROPAGATE DE HYBRID WINS:
# Set w_constrained = w_de so all downstream analysis uses the best!
# ==================================================================
w_constrained_old = w_constrained.copy() # Backup SLSQP
w_constrained = w_de.copy()
print("\n\n[SYSTEM] Assigned w_constrained = w_de for all downstream analysis!")


# ## 9. เปรียบเทียบ Sector/Asset Class Weights

# In[62]:


def get_sector_weights(weights, stock_list, stock_sector_map):
    """คำนวณน้ำหนักรวมแต่ละกลุ่ม"""
    sector_w = {}
    for i, stock in enumerate(stock_list):
        sector = stock_sector_map[stock]
        sector_w[sector] = sector_w.get(sector, 0) + weights[i]
    return pd.Series(sector_w)

sector_w_uc = get_sector_weights(w_unconstrained, stock_list, stock_sector_map)
sector_w_slsqp = get_sector_weights(w_constrained_old, stock_list, stock_sector_map)
sector_w_pso = get_sector_weights(w_pso, stock_list, stock_sector_map)
sector_w_de = get_sector_weights(w_de, stock_list, stock_sector_map)

# สร้าง DataFrame เปรียบเทียบ
sector_comp = pd.DataFrame({
    "Unconstrained (%)": (sector_w_uc * 100).round(2),
    "SLSQP (%)": (sector_w_slsqp * 100).round(2),
    "PSO (%)": (sector_w_pso * 100).round(2),
    "DE (%)": (sector_w_de * 100).round(2),
    "Limit (%)": SECTOR_MAX_WEIGHT * 100
}).fillna(0)

print("=" * 80)
print("📊 Sector Weights: Unconstrained vs SLSQP vs PSO vs DE")
print("=" * 80)
print(sector_comp.to_string())


# In[63]:


# === Visualization: Sector Weights Comparison ===

fig, ax = plt.subplots(figsize=(16, 8))

sectors = sector_comp.index
x = np.arange(len(sectors))
bw = 0.2  # bar width

ax.bar(x - 1.5*bw, sector_comp["Unconstrained (%)"], bw, label="Unconstrained", color="#9E9E9E", alpha=0.7)
ax.bar(x - 0.5*bw, sector_comp["SLSQP (%)"], bw, label="SLSQP", color="#2196F3", alpha=0.9)
ax.bar(x + 0.5*bw, sector_comp["PSO (%)"], bw, label="PSO", color="#FF9800", alpha=0.9)
ax.bar(x + 1.5*bw, sector_comp["DE (%)"], bw, label="DE (Winner)", color="#4CAF50", alpha=0.9)

# เส้น Limit
ax.axhline(y=SECTOR_MAX_WEIGHT*100, color="red", linestyle="--", linewidth=2, label=f"Max Limit ({SECTOR_MAX_WEIGHT*100:.0f}%)")
ax.axhline(y=SECTOR_MIN_WEIGHT*100, color="green", linestyle="--", linewidth=2, label=f"Min Limit ({SECTOR_MIN_WEIGHT*100:.0f}%)")

ax.set_xticks(x)
ax.set_xticklabels(sectors, rotation=45, ha="right")
ax.set_ylabel("Weight (%)", fontsize=12)
ax.set_title("Sector Weights Comparison (4 Optimizers)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()


# ## 10. แสดงน้ำหนักสินทรัพย์แต่ละตัว

# In[64]:


# สร้างตาราง Weights เปรียบเทียบสินทรัพย์รายตัว
weights_df = pd.DataFrame({
    "Ticker": stock_list,
    "Sector": [stock_sector_map[s] for s in stock_list],
    "Unconstrained (%)": (w_unconstrained * 100).round(2),
    "SLSQP (%)": (w_constrained_old * 100).round(2),
    "PSO (%)": (w_pso * 100).round(2),
    "DE (%)": (w_de * 100).round(2),
})

print("=" * 80)
print("📊 Individual Asset Weights (Top 10 in DE Portfolio)")
print("=" * 80)
top10 = weights_df.sort_values(by="DE (%)", ascending=False).head(10)
print(top10.to_string(index=False))


# In[65]:


# === Bar Chart: Individual Asset Weights ===

# เรียงตาม Sector เพื่อให้ดูง่าย
weights_df_sorted = weights_df.sort_values(by=["Sector", "Ticker"])

fig, ax = plt.subplots(figsize=(18, 8))
x = np.arange(len(weights_df_sorted))
bw = 0.2

ax.bar(x - 1.5*bw, weights_df_sorted["Unconstrained (%)"], bw, label="Unconstrained", color="#9E9E9E", alpha=0.7)
ax.bar(x - 0.5*bw, weights_df_sorted["SLSQP (%)"], bw, label="SLSQP", color="#2196F3", alpha=0.9)
ax.bar(x + 0.5*bw, weights_df_sorted["PSO (%)"], bw, label="PSO", color="#FF9800", alpha=0.9)
ax.bar(x + 1.5*bw, weights_df_sorted["DE (%)"], bw, label="DE (Winner)", color="#4CAF50", alpha=0.9)

ax.axhline(y=STOCK_MAX_WEIGHT*100, color="red", linestyle="--", alpha=0.5, label=f"Max Limit ({STOCK_MAX_WEIGHT*100:.0f}%)")
ax.axhline(y=STOCK_MIN_WEIGHT*100, color="green", linestyle="--", alpha=0.5, label=f"Min Limit ({STOCK_MIN_WEIGHT*100:.0f}%)")

ax.set_xticks(x)
ax.set_xticklabels(weights_df_sorted["Ticker"], rotation=90, fontsize=9)
ax.set_ylabel("Weight (%)", fontsize=12)
ax.set_title("Individual Asset Weights Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# เพิ่มสีพื้นหลังแบ่ง Sector
current_sector = ""
start_idx = 0
colors = plt.cm.Pastel1(np.linspace(0, 1, 10))
color_idx = 0

for i, row in enumerate(weights_df_sorted.itertuples()):
    if row.Sector != current_sector:
        if i > 0:
            ax.axvspan(start_idx - 0.5, i - 0.5, color=colors[color_idx % 10], alpha=0.2)
            ax.text((start_idx + i - 1) / 2, ax.get_ylim()[1] * 0.95, current_sector, 
                    ha="center", va="top", rotation=90, fontsize=8, alpha=0.7)
            color_idx += 1
        current_sector = row.Sector
        start_idx = i

# ล็อตสุดท้าย
ax.axvspan(start_idx - 0.5, len(weights_df_sorted) - 0.5, color=colors[color_idx % 10], alpha=0.2)
ax.text((start_idx + len(weights_df_sorted) - 1) / 2, ax.get_ylim()[1] * 0.95, current_sector, 
        ha="center", va="top", rotation=90, fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()


# ## 11. Efficient Frontier (เปรียบเทียบ)

# In[66]:


def generate_efficient_frontier(mean_returns, cov_matrix, sector_constraint, n_points=30):
    """สร้าง Efficient Frontier"""
    n = len(mean_returns)
    
    # หา Min/Max Return ที่เป็นไปได้
    target_returns = np.linspace(
        mean_returns.min() * 252,
        mean_returns.max() * 252,
        n_points
    )
    
    frontier_vol = []
    frontier_ret = []
    
    if sector_constraint:
        bounds = tuple((STOCK_MIN_WEIGHT, STOCK_MAX_WEIGHT) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))
    
    for target in target_returns:
        constraints = get_base_constraints()
        constraints.append({
            'type': 'eq',
            'fun': lambda w, t=target: np.sum(w * mean_returns) * 252 - t
        })
        if sector_constraint:
            constraints += get_sector_constraints(
                sector_stock_indices, SECTOR_MAX_WEIGHT, SECTOR_MIN_WEIGHT
            )
        
        init = np.array(n * [1./n])
        try:
            result = minimize(
                minimize_volatility, init,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            if result.success:
                r, v, _ = portfolio_performance(result.x, mean_returns, cov_matrix)
                frontier_vol.append(v)
                frontier_ret.append(r)
        except:
            continue
    
    return frontier_vol, frontier_ret


print("⏳ Generating Efficient Frontiers...")
ef_vol_uc, ef_ret_uc = generate_efficient_frontier(mean_returns, cov_matrix, False)
ef_vol_c, ef_ret_c = generate_efficient_frontier(mean_returns, cov_matrix, True)
print("✅ Done!")


# In[67]:


# === Plot Efficient Frontier with 3 Optimizers ===

fig, ax = plt.subplots(figsize=(16, 9))

# Plot Unconstrained Frontier
ax.plot([v*100 for v in ef_vol_uc], [r*100 for r in ef_ret_uc],
        "gray", linestyle="--", linewidth=2, label="Unconstrained Frontier")

# Plot Unconstrained Max Sharpe
ax.scatter(vol_uc*100, ret_uc*100, marker="*", color="black", s=300, zorder=5, 
           label=f"Unconstrained Obj (SR={sharpe_uc:.3f})")

# Plot the 3 constrained optimizers
ax.scatter(slsqp_vol*100, slsqp_ret*100, marker="o", color="#2196F3", s=200, zorder=6, 
           label=f"SLSQP (SR={slsqp_sharpe:.3f})")
ax.scatter(pso_vol*100, pso_ret*100, marker="D", color="#FF9800", s=200, zorder=6, 
           label=f"PSO Hybrid (SR={pso_sharpe:.3f})")
ax.scatter(de_vol*100, de_ret*100, marker="*", color="#4CAF50", s=400, zorder=7, 
           label=f"DE Hybrid (SR={de_sharpe:.3f})")

# Individual assets
for i, t in enumerate(stock_list):
    ax.scatter(np.sqrt(cov_annual.iloc[i,i])*100, mu_annual.iloc[i]*100,
               s=30, c="gray", alpha=0.4)
    ax.annotate(t, (np.sqrt(cov_annual.iloc[i,i])*100, mu_annual.iloc[i]*100),
                fontsize=7, alpha=0.6)

ax.set_title("Efficient Frontier & Optimizer Positions", fontsize=15, fontweight="bold")
ax.set_xlabel("Annual Volatility (%)", fontsize=12)
ax.set_ylabel("Annual Return (%)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ## 12. Backtest: เปรียบเทียบ Portfolio Performance

# In[68]:


# คำนวณ portfolio returns
port_returns_uc = (returns * w_unconstrained).sum(axis=1)
port_returns_slsqp = (returns * w_constrained_old).sum(axis=1)
port_returns_pso = (returns * w_pso).sum(axis=1)
port_returns_de = (returns * w_de).sum(axis=1)

# Equal-weight benchmark
w_equal = np.array([1/n_stocks] * n_stocks)
port_returns_eq = (returns * w_equal).sum(axis=1)


# In[69]:


# === สรุปผล Backtest (5 Portfolios) ===

def portfolio_metrics(daily_returns, name):
    total_return = (1 + daily_returns).prod() - 1
    annual_ret = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol
    
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    
    return [name, total_return, annual_ret, annual_vol, sharpe, max_dd, calmar]

results = [
    portfolio_metrics(port_returns_uc, "Unconstrained"),
    portfolio_metrics(port_returns_slsqp, "SLSQP Constrained"),
    portfolio_metrics(port_returns_pso, "PSO Constrained"),
    portfolio_metrics(port_returns_de, "DE Constrained (Winner)"),
    portfolio_metrics(port_returns_eq, "Equal Weight")
]

columns = ["Portfolio", "Total Return (%)", "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio", "Max Drawdown (%)", "Calmar Ratio"]
results_df = pd.DataFrame(results, columns=columns)

# Format as percentage
for col in ["Total Return (%)", "Annual Return (%)", "Annual Vol (%)", "Max Drawdown (%)"]:
    results_df[col] = (results_df[col] * 100).round(2)

results_df["Sharpe Ratio"] = results_df["Sharpe Ratio"].round(3)
results_df["Calmar Ratio"] = results_df["Calmar Ratio"].round(3)

print("=" * 105)
print("📊 สรุปผล Backtest Performance (All Optimizers)")
print("=" * 105)
print(results_df.to_string(index=False))

# Visualize Cumulative Returns
plt.figure(figsize=(16, 7))
plt.plot((1 + port_returns_uc).cumprod() * 100, label="Unconstrained", color="#9E9E9E", alpha=0.7)
plt.plot((1 + port_returns_slsqp).cumprod() * 100, label="SLSQP Constrained", color="#2196F3", linewidth=2)
plt.plot((1 + port_returns_pso).cumprod() * 100, label="PSO Constrained", color="#FF9800", linewidth=2)
plt.plot((1 + port_returns_de).cumprod() * 100, label="DE Constrained (Winner)", color="#4CAF50", linewidth=3)
plt.plot((1 + port_returns_eq).cumprod() * 100, label="Equal Weight", color="#9C27B0", linestyle="--")

plt.title("Cumulative Returns (Base = 100 THB)", fontsize=14, fontweight="bold")
plt.ylabel("Value (THB)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ## 13. Transaction Costs (TC) — ต้นทุนในการทำธุรกรรม
# 
# ### ประเภทต้นทุน:
# | ประเภท | รายละเอียด | ค่าเริ่มต้น |
# |--------|-----------|------------|
# | **Broker Fee** | ค่าธรรมเนียมโบรกเกอร์ | 0.10% |
# | **Tax** | ภาษีหัก ณ ที่จ่าย | 0.05% |
# | **Bid-Ask Spread** | ���่วนต่างราคา (Liquidity Cost) | 0.05% |
# | **รวม (Total TC)** | ต้นทุนต่อการซื้อขาย 1 ครั้ง | **0.20%** |
# 
# > ⚠️ หากไม่คำนวณ TC โมเดลจะ **ดูดีเกินจริง (Optimistic)** เพราะทุกการปรับพอร์ตมีต้นทุนที่ต้องหักจากกำไรเสมอ

# In[70]:


# ==========================================================
# Transaction Costs (TC) Analysis
# ==========================================================

# --- TC Parameters ---
BROKER_FEE = 0.0010    # 0.10% ค่าธรรมเนียมโบรกเกอร์
TAX = 0.0005           # 0.05% ภาษี
BID_ASK_SPREAD = 0.0005  # 0.05% ส่วนต่างราคา
TOTAL_TC = BROKER_FEE + TAX + BID_ASK_SPREAD  # 0.20% per trade
REBALANCE_FREQ = 21    # ปรับพอร์ตทุก 21 Days (รายเดือน)

print("=" * 65)
print("Transaction Costs (TC) Configuration")
print("=" * 65)
print(f"  Broker Fee     : {BROKER_FEE*100:.2f}%")
print(f"  Tax            : {TAX*100:.2f}%")
print(f"  Bid-Ask Spread : {BID_ASK_SPREAD*100:.2f}%")
print(f"  Total TC/trade : {TOTAL_TC*100:.2f}%")
print(f"  Rebalance Freq : Every {REBALANCE_FREQ} trading days")

# --- Backtest with TC ---
def backtest_with_tc(returns, weights, tc_rate, rebal_days):
    """Backtest with transaction costs on rebalancing days"""
    n_days = len(returns)
    port_values = [1.0]  # start with 1.0
    current_weights = weights.copy()
    total_tc_paid = 0
    n_rebalances = 0
    daily_tc = []
    
    for day in range(n_days):
        daily_ret = returns.iloc[day].values
        
        # Portfolio return for the day
        port_ret = np.sum(current_weights * daily_ret)
        
        # Update weights based on returns (drift)
        new_vals = current_weights * (1 + daily_ret)
        current_weights = new_vals / new_vals.sum()
        
        # Rebalance check
        tc_cost = 0
        if (day + 1) % rebal_days == 0:
            # Calculate turnover (how much weights changed)
            turnover = np.sum(np.abs(current_weights - weights))
            tc_cost = turnover * tc_rate
            total_tc_paid += tc_cost
            current_weights = weights.copy()
            n_rebalances += 1
        
        new_value = port_values[-1] * (1 + port_ret - tc_cost)
        port_values.append(new_value)
        daily_tc.append(tc_cost)
    
    return np.array(port_values[1:]), total_tc_paid, n_rebalances, daily_tc

# Run backtests
print("\nRunning backtests...")

# Without TC
cum_no_tc_uc = (1 + (returns * w_unconstrained).sum(axis=1)).cumprod()
cum_no_tc_c = (1 + (returns * w_constrained).sum(axis=1)).cumprod()

# With TC
cum_tc_uc, tc_paid_uc, n_reb_uc, _ = backtest_with_tc(returns, w_unconstrained, TOTAL_TC, REBALANCE_FREQ)
cum_tc_c, tc_paid_c, n_reb_c, _ = backtest_with_tc(returns, w_constrained, TOTAL_TC, REBALANCE_FREQ)

print(f"  Rebalances: {n_reb_uc} times over {len(returns)} days")
print(f"  Total TC paid (Unconstrained): {tc_paid_uc*100:.2f}%")
print(f"  Total TC paid (Constrained):   {tc_paid_c*100:.2f}%")


# In[71]:


# === TC Impact: Comparison Table & Charts ===

def calc_metrics(cum_values, label):
    total_ret = cum_values[-1] / cum_values[0] - 1
    days = len(cum_values)
    ann_ret = (1 + total_ret)**(252/days) - 1
    
    daily_rets = pd.Series(cum_values).pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    
    peak = pd.Series(cum_values).cummax()
    dd = (pd.Series(cum_values) - peak) / peak
    max_dd = dd.min()
    
    return [label, total_ret*100, ann_ret*100, ann_vol*100, sharpe, max_dd*100]

tc_results = []

# 1. Unconstrained
cum_uc_notc, _, _, _ = backtest_with_tc(returns, w_unconstrained, 0.0, REBALANCE_FREQ)
cum_uc_tc, _, _, _ = backtest_with_tc(returns, w_unconstrained, TOTAL_TC, REBALANCE_FREQ)
tc_results.append(calc_metrics(cum_uc_notc, "Unconstrained (NO TC)"))
tc_results.append(calc_metrics(cum_uc_tc, "Unconstrained (WITH TC)"))

# 2. SLSQP
cum_slsqp_notc, _, _, _ = backtest_with_tc(returns, w_constrained_old, 0.0, REBALANCE_FREQ)
cum_slsqp_tc, _, _, _ = backtest_with_tc(returns, w_constrained_old, TOTAL_TC, REBALANCE_FREQ)
tc_results.append(calc_metrics(cum_slsqp_notc, "SLSQP (NO TC)"))
tc_results.append(calc_metrics(cum_slsqp_tc, "SLSQP (WITH TC)"))

# 3. PSO
cum_pso_notc, _, _, _ = backtest_with_tc(returns, w_pso, 0.0, REBALANCE_FREQ)
cum_pso_tc, _, _, _ = backtest_with_tc(returns, w_pso, TOTAL_TC, REBALANCE_FREQ)
tc_results.append(calc_metrics(cum_pso_notc, "PSO (NO TC)"))
tc_results.append(calc_metrics(cum_pso_tc, "PSO (WITH TC)"))

# 4. DE
cum_de_notc, _, _, _ = backtest_with_tc(returns, w_de, 0.0, REBALANCE_FREQ)
cum_de_tc, _, _, _ = backtest_with_tc(returns, w_de, TOTAL_TC, REBALANCE_FREQ)
tc_results.append(calc_metrics(cum_de_notc, "DE (NO TC)"))
tc_results.append(calc_metrics(cum_de_tc, "DE (WITH TC)"))

tc_df = pd.DataFrame(tc_results, columns=["Portfolio", "Total Return (%)", "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio", "Max DD (%)"])
tc_df = tc_df.round(3)

print("=" * 85)
print("Transaction Costs Impact: Before vs After TC")
print("=" * 85)
print(tc_df.to_string(index=False))

drag_uc = tc_df.iloc[0]["Total Return (%)"] - tc_df.iloc[1]["Total Return (%)"]
drag_slsqp = tc_df.iloc[2]["Total Return (%)"] - tc_df.iloc[3]["Total Return (%)"]
drag_pso = tc_df.iloc[4]["Total Return (%)"] - tc_df.iloc[5]["Total Return (%)"]
drag_de = tc_df.iloc[6]["Total Return (%)"] - tc_df.iloc[7]["Total Return (%)"]

print("\nTC Drag (กำไรที่หายไปจาก TC สำหรับระยะเวลา 4 ปี):")
print(f"  Unconstrained: {drag_uc:.2f}% of total return lost")
print(f"  SLSQP:         {drag_slsqp:.2f}% of total return lost")
print(f"  PSO:           {drag_pso:.2f}% of total return lost")
print(f"  DE:            {drag_de:.2f}% of total return lost")

# Bar Chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
drags = [drag_uc, drag_slsqp, drag_pso, drag_de]
labels = ["Unconstrained", "SLSQP", "PSO", "DE"]
colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50"]

bars = ax.bar(labels, drags, color=colors, alpha=0.8)
ax.set_ylabel("Total Return Lost to TC (%)")
ax.set_title("Transaction Cost Drag by Optimizer", fontweight="bold")

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f"{height:.2f}%", ha="center", va="bottom", fontweight="bold")

plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# ## 14. Transaction Lots (TL) — หน่วยการซื้อขาย
# 
# สินทรัพย์บางอย่างไม่สามารถซื้อเป็นทศนิยมได้ ต้องซื้อเป็น **จำนวนเต็ม** หรือ **Lot**
# 
# | ตลาด | Lot Size | ตัวอย่าง |
# |------|----------|--------|
# | 🇹🇭 SET (หุ้นไทย) | **100 หุ้น/Lot** | ซื้อ DELTA.BK ต้องซื้อ 100, 200, 300... |
# | 🇺🇸 US Stocks | **1 หุ้น** | ซื้อ AAPL ได้ 1, 2, 3... หุ้น |
# | 🇺🇸 US ETFs | **1 หน่วย** | ซื้อ GLD ได้ 1, 2, 3... หน่วย |
# | ₿ Crypto | **ทศนิยมได้** | ซื้อ BTC 0.001 ได้ |
# 

# In[72]:


# ==========================================================
# Transaction Lots (TL) - ปรับน้ำหนักให้ซื้อได้จริง
# ==========================================================

PORTFOLIO_VALUE = 1_000_000  # เงินลงทุน 1,000,000 บาท

# กำหนด Lot Size ตามจริงของแต่ละตลาด
lot_sizes = {}
for ticker in stock_list:
    if ticker.endswith(".BK"):        # หุ้นไทย = 100 หุ้น/Lot
        lot_sizes[ticker] = 100
    elif ticker in ["BTC-USD", "ETH-USD"]:  # Crypto = ซื้อทศนิยมได้
        lot_sizes[ticker] = 0.0001
    else:                             # US Stocks & ETFs = 1 หุ้น
        lot_sizes[ticker] = 1

# ราคาล่าสุดของแต่ละตัว
latest_prices = data.iloc[-1]

print("=" * 80)
print(f"Transaction Lots Analysis (Portfolio = {PORTFOLIO_VALUE:,.0f} THB)")
print("=" * 80)

def adjust_to_lots(weights, prices, port_value, lot_sizes):
    """ปรับน้ำหนักให้ซื้อได้จริงตาม Lot Size"""
    results = []
    total_invested = 0
    
    for i, ticker in enumerate(prices.index):
        target_value = weights[i] * port_value
        price = prices[ticker]
        lot = lot_sizes[ticker]
        
        # คำนวณจำนวนที่ซื้อได้จริง (ปัดลง)
        if lot < 1:  # Crypto
            shares = target_value / price
            shares = round(shares, 4)
        else:
            raw_shares = target_value / price
            shares = int(raw_shares / lot) * lot  # ปัดลงเป็น Lot
        
        actual_value = shares * price
        total_invested += actual_value
        
        results.append({
            "Ticker": ticker,
            "Sector": stock_sector_map[ticker],
            "Lot Size": lot,
            "Price": round(price, 2),
            "Target Wt (%)": round(weights[i] * 100, 2),
            "Target Value": round(target_value, 0),
            "Shares": shares,
            "Actual Value": round(actual_value, 0),
            "Actual Wt (%)": 0,  # calculate after
        })
    
    # Recalculate actual weights
    for r in results:
        r["Actual Wt (%)"] = round(r["Actual Value"] / total_invested * 100, 2)
    
    return pd.DataFrame(results), total_invested

# ปรับ Constrained portfolio
lots_df, total_inv = adjust_to_lots(w_constrained, latest_prices, PORTFOLIO_VALUE, lot_sizes)

print(f"\nSector-Constrained (DE) Portfolio (Lot-Adjusted):")
print("-" * 80)
print(lots_df.to_string(index=False))
print(f"\nTotal Invested: {total_inv:>15,.0f} THB")
print(f"Cash Remaining: {PORTFOLIO_VALUE - total_inv:>15,.0f} THB")
print(f"Cash %:         {(PORTFOLIO_VALUE - total_inv)/PORTFOLIO_VALUE*100:>14.2f}%")


# In[73]:


# === Lot Adjustment Impact: ความแตกต่าง Target vs Actual ===

# Weight deviation
lots_df["Wt Diff (%)"] = lots_df["Actual Wt (%)"] - lots_df["Target Wt (%)"]

tracking_error = np.sqrt(np.sum(lots_df["Wt Diff (%)"].values**2))
max_dev = lots_df["Wt Diff (%)"].abs().max()
avg_dev = lots_df["Wt Diff (%)"].abs().mean()

print("=" * 60)
print("Lot Adjustment Impact")
print("=" * 60)
print(f"  Tracking Error (RMSE):    {tracking_error:.3f}%")
print(f"  Max Weight Deviation:     {max_dev:.3f}%")
print(f"  Avg Weight Deviation:     {avg_dev:.3f}%")

# Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1) Target vs Actual weights
x = np.arange(len(lots_df))
w_bar = 0.35
axes[0].barh(x - w_bar/2, lots_df["Target Wt (%)"], w_bar, label="Target (Optimal)", color="#3498db", alpha=0.8)
axes[0].barh(x + w_bar/2, lots_df["Actual Wt (%)"], w_bar, label="Actual (Lot-Adjusted)", color="#e74c3c", alpha=0.8)
axes[0].set_yticks(x)
axes[0].set_yticklabels(lots_df["Ticker"], fontsize=9)
axes[0].set_xlabel("Weight (%)")
axes[0].set_title("Target vs Actual Weights (Lot-Adjusted)", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="x")

# 2) Weight deviation
colors_dev = ["#27ae60" if v >= 0 else "#e74c3c" for v in lots_df["Wt Diff (%)"]]
axes[1].barh(lots_df["Ticker"], lots_df["Wt Diff (%)"], color=colors_dev, alpha=0.8)
axes[1].axvline(x=0, color="black", linewidth=1)
axes[1].set_xlabel("Weight Deviation (%)")
axes[1].set_title(f"Weight Deviation from Optimal\n(RMSE={tracking_error:.3f}%)", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.show()

print("\n>> สรุป: การปรับ Lot ทำให้น้ำหนักเบี่ยงเบนจาก Optimal เล็กน้อย")
print(f"   แต่ยังอยู่ในเกณฑ์ที่ยอมรับได้ (RMSE = {tracking_error:.3f}%)")


# ## 15. Short Sell with Leverage Constraint
# 
# ### แนวคิด:
# - **Long-Only**: ซื้ออย่างเดียว (w ≥ 0), ∑w = 1
# - **Long-Short**: ซื้อ (Long) + ขายชอร์ต (Short) ได้, ∑w = 1 แต่ ∑|w| > 1
# - **Leverage**: Gross Exposure = ∑|w| → ถ้า > 1 แปลว่าขยายสถานะเกินทุนจริง
# 
# ### ตัวอย่าง:
# | สถานะ | Long | Short | Net (∑w) | Gross (∑\|w\|) | Leverage |
# |--------|------|-------|----------|---------------|----------|
# | Long-Only | 100% | 0% | 100% | 100% | 1.0x |
# | Long-Short (free) | 120% | -20% | 100% | 140% | 1.4x |
# | Long-Short (limited) | 110% | -10% | 100% | **≤ 130%** | ≤ 1.3x |
# 
# > ⚠️ ถ้าไม่จำกัด Leverage → optimizer จะทำ long-short มากเกินไป ทำให้เสี่ยงสูงและไม่สามารถใช้ได้จริง

# In[74]:


# ==========================================================
# Short Sell with Leverage Constraint
# ==========================================================

from scipy.optimize import minimize as sp_min

mu = returns.mean() * 252
cov_mat = returns.cov() * 252
n = len(mu)

# --- Leverage Levels to Compare ---
LEVERAGE_LEVELS = {
    "Long-Only (1.0x)":        {"bounds": (0, 1),       "max_gross": 1.0},
    "Long-Short (1.3x)":       {"bounds": (-0.15, 0.3), "max_gross": 1.3},
    "Long-Short (1.6x)":       {"bounds": (-0.20, 0.4), "max_gross": 1.6},
    "Long-Short (2.0x)":       {"bounds": (-0.30, 0.5), "max_gross": 2.0},
    "Long-Short (No Limit)":   {"bounds": (-0.50, 1.0), "max_gross": 99},
}

def optimize_with_leverage(mu, cov_mat, bounds_range, max_gross, n_trials=30):
    """Optimize Max Sharpe with leverage constraint"""
    n = len(mu)
    bounds = tuple((bounds_range[0], bounds_range[1]) for _ in range(n))
    
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # net weight = 1
    ]
    if max_gross < 50:  # add leverage constraint
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_gross - np.sum(np.abs(w))  # gross <= max
        })
    
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        return -(ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    
    best = None
    best_val = np.inf
    for _ in range(n_trials):
        init = np.random.dirichlet(np.ones(n))
        try:
            res = sp_min(neg_sharpe, init, method="SLSQP",
                        bounds=bounds, constraints=constraints,
                        options={"maxiter": 1000, "ftol": 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best = res
        except:
            continue
    return best

# --- Run All Leverage Levels ---
print("=" * 80)
print("Short Sell with Leverage Constraint: Comparison")
print("=" * 80)
print()

results_lev = []
weights_lev = {}

for name, params in LEVERAGE_LEVELS.items():
    print(f"Optimizing: {name}...")
    res = optimize_with_leverage(mu, cov_mat, params["bounds"], params["max_gross"])
    w = res.x
    
    ret = np.dot(w, mu)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    sharpe = (ret - RISK_FREE_RATE) / vol
    gross = np.sum(np.abs(w))
    long_sum = np.sum(w[w > 0])
    short_sum = np.sum(w[w < 0])
    n_short = np.sum(w < -0.005)
    
    results_lev.append({
        "Strategy": name,
        "Return (%)": round(ret * 100, 2),
        "Vol (%)": round(vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Long (%)": round(long_sum * 100, 1),
        "Short (%)": round(short_sum * 100, 1),
        "Gross (%)": round(gross * 100, 1),
        "N Short": int(n_short),
    })
    weights_lev[name] = w

lev_df = pd.DataFrame(results_lev)
print()
print(lev_df.to_string(index=False))


# In[75]:


# === Short Sell & Leverage: Visualization ===

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1) Sharpe vs Leverage
axes[0,0].plot(lev_df["Gross (%)"], lev_df["Sharpe"], "bo-", linewidth=2, markersize=10)
for _, row in lev_df.iterrows():
    axes[0,0].annotate(row["Strategy"].split("(")[1].rstrip(")"),
                      (row["Gross (%)"], row["Sharpe"]),
                      fontsize=9, ha="left", va="bottom")
axes[0,0].set_xlabel("Gross Exposure (%)", fontsize=12)
axes[0,0].set_ylabel("Sharpe Ratio", fontsize=12)
axes[0,0].set_title("Sharpe Ratio vs Leverage Level", fontsize=13, fontweight="bold")
axes[0,0].grid(True, alpha=0.3)

# 2) Return vs Vol scatter
colors_lev = plt.cm.viridis(np.linspace(0, 0.9, len(results_lev)))
for j, (name, row) in enumerate(zip(lev_df["Strategy"], results_lev)):
    axes[0,1].scatter(row["Vol (%)"], row["Return (%)"], s=200,
                     c=[colors_lev[j]], edgecolors="black", linewidth=1.5, zorder=5)
    axes[0,1].annotate(name.split("(")[1].rstrip(")"),
                      (row["Vol (%)"], row["Return (%)"]),
                      fontsize=9, ha="left", va="bottom")
axes[0,1].set_xlabel("Volatility (%)", fontsize=12)
axes[0,1].set_ylabel("Return (%)", fontsize=12)
axes[0,1].set_title("Risk-Return: Different Leverage Levels", fontsize=13, fontweight="bold")
axes[0,1].grid(True, alpha=0.3)

# 3) Weights comparison: Long-Only vs Long-Short 1.3x
if "Long-Only (1.0x)" in weights_lev and "Long-Short (1.3x)" in weights_lev:
    w_lo = weights_lev["Long-Only (1.0x)"]
    w_ls = weights_lev["Long-Short (1.3x)"]
    x_idx = np.arange(n)
    bw = 0.35
    axes[1,0].bar(x_idx - bw/2, w_lo * 100, bw, label="Long-Only", color="#3498db", alpha=0.8)
    axes[1,0].bar(x_idx + bw/2, w_ls * 100, bw, label="Long-Short (1.3x)", color="#e74c3c", alpha=0.8)
    axes[1,0].axhline(y=0, color="black", linewidth=1)
    axes[1,0].set_xticks(x_idx)
    axes[1,0].set_xticklabels(mu.index, rotation=45, ha="right", fontsize=8)
    axes[1,0].set_ylabel("Weight (%)")
    axes[1,0].set_title("Long-Only vs Long-Short Weights", fontsize=13, fontweight="bold")
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3, axis="y")

# 4) Stacked bar: Long vs Short composition
strategies = [r["Strategy"].split("(")[1].rstrip(")") for r in results_lev]
longs = [r["Long (%)"] for r in results_lev]
shorts = [-r["Short (%)"] for r in results_lev]  # make positive for display
x_s = np.arange(len(strategies))
axes[1,1].bar(x_s, longs, 0.6, label="Long (%)", color="#27ae60", alpha=0.8)
axes[1,1].bar(x_s, shorts, 0.6, bottom=longs, label="Short (%)", color="#e74c3c", alpha=0.8)
axes[1,1].set_xticks(x_s)
axes[1,1].set_xticklabels(strategies, fontsize=10)
axes[1,1].set_ylabel("Exposure (%)")
axes[1,1].set_title("Gross Exposure Breakdown", fontsize=13, fontweight="bold")
axes[1,1].legend(fontsize=10)
for i, (l, s) in enumerate(zip(longs, shorts)):
    axes[1,1].text(i, l + s + 2, f"{l+s:.0f}%", ha="center", fontsize=10, fontweight="bold")
axes[1,1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Short Sell with Leverage Constraint Analysis",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("\nสรุป:")
print("  - Leverage สูงขึ้น -> Sharpe อาจดีขึ้น แต่ Gross Exposure สูง (เสี่ยงมาก)")
print("  - Leverage 1.3x เป็น sweet spot: ได้ประโยชน์จาก short sell แต่ไม่เสี่ยงเกินไป")
print("  - No Limit -> optimizer ใช้ leverage มากเกินไป ไม่เหมาะใช้จริง")


# ## Out-of-Sample Testing (Train/Test Split)
# 
# Split data into **Train (3 years)** and **Test (1 year)** to check for overfitting:
# - Optimize weights on training data
# - Test performance on unseen data
# - Compare In-Sample vs Out-of-Sample results

# In[76]:


# ==========================================================
# Out-of-Sample Testing: Train (3yr) / Test (1yr)
# ==========================================================

split_idx = int(len(returns) * 0.75)  # 75% train, 25% test
train_returns = returns.iloc[:split_idx]
test_returns = returns.iloc[split_idx:]

print("=" * 70)
print("Out-of-Sample Testing: Train/Test Split")
print("=" * 70)
print(f"  Train period: {train_returns.index[0].strftime("%Y-%m-%d")} to {train_returns.index[-1].strftime("%Y-%m-%d")} ({len(train_returns)} days)")
print(f"  Test period:  {test_returns.index[0].strftime("%Y-%m-%d")} to {test_returns.index[-1].strftime("%Y-%m-%d")} ({len(test_returns)} days)")

# --- Optimize on TRAIN data ---
train_mu = train_returns.mean() * 252
train_cov = train_returns.cov() * 252

# Define Optimization Functions for Train set
def slsqp_train():
    def neg_sharpe(w):
        ret = np.dot(w, train_mu)
        vol = np.sqrt(np.dot(w.T, np.dot(train_cov, w)))
        return -((ret - RISK_FREE_RATE) / vol) if vol > 0 else 0
    
    n = len(train_mu)
    bounds = tuple((STOCK_MIN_WEIGHT, STOCK_MAX_WEIGHT) for _ in range(n))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for sector in set(stock_sector_map.values()):
        idx = [i for i, t in enumerate(stock_list) if stock_sector_map[t] == sector]
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: np.sum(w[idx]) - SECTOR_MIN_WEIGHT})
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: SECTOR_MAX_WEIGHT - np.sum(w[idx])})
        
    best_w = None
    best_val = np.inf
    for _ in range(15):
        init = np.random.dirichlet(np.ones(n))
        try:
            res = sp_minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=cons)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_w = res.x
        except:
            pass
    return best_w, -best_val

def pso_train():
    def objective(w):
        w_norm = w / np.sum(w) if np.sum(w) > 0 else np.ones(n_assets)/n_assets
        ret = np.dot(w_norm, train_mu)
        vol = np.sqrt(np.dot(w_norm.T, np.dot(train_cov, w_norm)))
        if vol < 1e-10: return 999
        sharpe = (ret - RISK_FREE_RATE) / vol
        
        penalty = 0
        for sector in set(stock_sector_map.values()):
            idx = [i for i, t in enumerate(stock_list) if stock_sector_map[t] == sector]
            sw = np.sum(w_norm[idx])
            if sw > SECTOR_MAX_WEIGHT: penalty += 100*(sw - SECTOR_MAX_WEIGHT)**2
            if sw < SECTOR_MIN_WEIGHT: penalty += 100*(SECTOR_MIN_WEIGHT - sw)**2
        for wi in w_norm:
            if wi > STOCK_MAX_WEIGHT: penalty += 50*(wi - STOCK_MAX_WEIGHT)**2
            if wi < STOCK_MIN_WEIGHT: penalty += 50*(STOCK_MIN_WEIGHT - wi)**2
        return -sharpe + penalty

    lb = np.ones(n_assets) * STOCK_MIN_WEIGHT
    ub = np.ones(n_assets) * STOCK_MAX_WEIGHT
    
    w_raw, fopt = pso(objective, lb, ub, swarmsize=100, maxiter=200, debug=False)
    w_norm = w_raw / np.sum(w_raw)
    
    # Calculate raw Sharpe
    ret = np.dot(w_norm, train_mu)
    vol = np.sqrt(np.dot(w_norm.T, np.dot(train_cov, w_norm)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return w_norm, sharpe

def de_train():
    def objective(w):
        w_norm = w / np.sum(w) if np.sum(w) > 0 else np.ones(n_assets)/n_assets
        ret = np.dot(w_norm, train_mu)
        vol = np.sqrt(np.dot(w_norm.T, np.dot(train_cov, w_norm)))
        if vol < 1e-10: return 999
        sharpe = (ret - RISK_FREE_RATE) / vol
        
        penalty = 0
        for sector in set(stock_sector_map.values()):
            idx = [i for i, t in enumerate(stock_list) if stock_sector_map[t] == sector]
            sw = np.sum(w_norm[idx])
            if sw > SECTOR_MAX_WEIGHT: penalty += 100*(sw - SECTOR_MAX_WEIGHT)**2
            if sw < SECTOR_MIN_WEIGHT: penalty += 100*(SECTOR_MIN_WEIGHT - sw)**2
        for wi in w_norm:
            if wi > STOCK_MAX_WEIGHT: penalty += 50*(wi - STOCK_MAX_WEIGHT)**2
            if wi < STOCK_MIN_WEIGHT: penalty += 50*(STOCK_MIN_WEIGHT - wi)**2
        return -sharpe + penalty
        
    bounds_de = [(STOCK_MIN_WEIGHT, STOCK_MAX_WEIGHT) for _ in range(n_assets)]
    res_de = differential_evolution(objective, bounds_de, maxiter=50, popsize=10, disp=False, polish=False)
    
    w_raw = res_de.x
    w_norm = w_raw / np.sum(w_raw)
    
    # Calculate raw Shape
    ret = np.dot(w_norm, train_mu)
    vol = np.sqrt(np.dot(w_norm.T, np.dot(train_cov, w_norm)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return w_norm, sharpe

print("\nRunning Optimizers on TRAIN data (3 years)...")
t0 = time.time()
w_slsqp_train, sr_slsqp_train = slsqp_train()
time_slsqp = time.time() - t0
print(f"  SLSQP Train Sharpe: {sr_slsqp_train:.4f}  |  Time: {time_slsqp:.2f}s")

t0 = time.time()
w_pso_train, sr_pso_train = pso_train()
time_pso = time.time() - t0
print(f"  PSO   Train Sharpe: {sr_pso_train:.4f}  |  Time: {time_pso:.2f}s")

t0 = time.time()
w_de_train, sr_de_train = de_train()
time_de = time.time() - t0
print(f"  DE    Train Sharpe: {sr_de_train:.4f}  |  Time: {time_de:.2f}s")

# --- Evaluate on TRAIN vs TEST ---
def eval_portfolio(w, ret_data, label, optimizer):
    port_ret = (ret_data * w).sum(axis=1)
    cum = (1 + port_ret).cumprod()
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return {
        "Optimizer": optimizer,
        "Period": label,
        "Ann Return": round(ann_ret * 100, 2),
        "Ann Vol": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Max DD": round(max_dd * 100, 2),
        "Cum Return": round((cum.iloc[-1] - 1) * 100, 2),
    }

oos_results = []
# SLSQP
oos_results.append(eval_portfolio(w_slsqp_train, train_returns, "Train", "SLSQP"))
oos_results.append(eval_portfolio(w_slsqp_train, test_returns, "Test", "SLSQP"))
# PSO
oos_results.append(eval_portfolio(w_pso_train, train_returns, "Train", "PSO"))
oos_results.append(eval_portfolio(w_pso_train, test_returns, "Test", "PSO"))
# DE
oos_results.append(eval_portfolio(w_de_train, train_returns, "Train", "DE"))
oos_results.append(eval_portfolio(w_de_train, test_returns, "Test", "DE"))

oos_df = pd.DataFrame(oos_results)
print("\n" + "=" * 90)
print("In-Sample (Train) vs Out-of-Sample (Test) Performance by Optimizer")
print("=" * 90)
print(oos_df.to_string(index=False))

# Overfitting Check (Sharpe Degradation)
print("\nSharpe Degradation (Train -> Test):")
for opt in ["SLSQP", "PSO", "DE"]:
    opt_data = oos_df[oos_df["Optimizer"] == opt]
    train_sr = opt_data[opt_data["Period"] == "Train"]["Sharpe"].values[0]
    test_sr = opt_data[opt_data["Period"] == "Test"]["Sharpe"].values[0]
    degradation = (train_sr - test_sr) / train_sr * 100 if train_sr > 0 else 0
    status = "GOOD" if degradation < 30 else ("MODERATE" if degradation < 50 else "OVERFITTING")
    print(f"  {opt:5s}: {degradation:6.1f}%  -> {status}")


# In[77]:


# === Out-of-Sample: Visualization (3 Optimizers) ===

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 1) Cumulative returns: Train vs Test
train_cum_slsqp = (1 + (train_returns * w_slsqp_train).sum(axis=1)).cumprod()
test_cum_slsqp = (1 + (test_returns * w_slsqp_train).sum(axis=1)).cumprod()
test_cum_slsqp_norm = test_cum_slsqp / test_cum_slsqp.iloc[0] * train_cum_slsqp.iloc[-1]

train_cum_pso = (1 + (train_returns * w_pso_train).sum(axis=1)).cumprod()
test_cum_pso = (1 + (test_returns * w_pso_train).sum(axis=1)).cumprod()
test_cum_pso_norm = test_cum_pso / test_cum_pso.iloc[0] * train_cum_pso.iloc[-1]

train_cum_de = (1 + (train_returns * w_de_train).sum(axis=1)).cumprod()
test_cum_de = (1 + (test_returns * w_de_train).sum(axis=1)).cumprod()
test_cum_de_norm = test_cum_de / test_cum_de.iloc[0] * train_cum_de.iloc[-1]

axes[0].plot(train_cum_slsqp.index, train_cum_slsqp.values, color="#2196F3", linestyle="-", alpha=0.9, label="SLSQP (Train)")
axes[0].plot(test_cum_slsqp_norm.index, test_cum_slsqp_norm.values, color="#2196F3", linestyle="--", alpha=0.9, label="SLSQP (Test)")

axes[0].plot(train_cum_pso.index, train_cum_pso.values, color="#FF9800", linestyle="-", alpha=0.9, label="PSO (Train)")
axes[0].plot(test_cum_pso_norm.index, test_cum_pso_norm.values, color="#FF9800", linestyle="--", alpha=0.9, label="PSO (Test)")

axes[0].plot(train_cum_de.index, train_cum_de.values, color="#4CAF50", linestyle="-", linewidth=2.5, alpha=0.9, label="DE (Train)")
axes[0].plot(test_cum_de_norm.index, test_cum_de_norm.values, color="#4CAF50", linestyle="--", linewidth=2.5, alpha=0.9, label="DE (Test)")

axes[0].axvline(x=train_cum_slsqp.index[-1], color="black", linestyle="-.", linewidth=2, alpha=0.7, label="TRAIN / TEST SPLIT")

axes[0].set_title("Out-of-Sample Test: Cumulative Returns (Base=100)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cumulative Return")
axes[0].legend(fontsize=9, loc="upper left")
axes[0].grid(True, alpha=0.3)

# 2) Bar chart: Sharpe degradation
degradations = []
for opt in ["SLSQP", "PSO", "DE"]:
    opt_data = oos_df[oos_df["Optimizer"] == opt]
    train_sr = opt_data[opt_data["Period"] == "Train"]["Sharpe"].values[0]
    test_sr = opt_data[opt_data["Period"] == "Test"]["Sharpe"].values[0]
    deg = (train_sr - test_sr) / train_sr * 100 if train_sr > 0 else 0
    degradations.append(deg)

x = np.arange(3)
colors = ["#2196F3", "#FF9800", "#4CAF50"]
bars = axes[1].bar(x, degradations, color=colors, alpha=0.8)
axes[1].axhline(y=0, color="black", linewidth=1)
axes[1].axhline(y=30, color="red", linestyle="--", alpha=0.5, label="Warning Threshold (30%)")

axes[1].set_xticks(x)
axes[1].set_xticklabels(["SLSQP", "PSO", "DE"])
axes[1].set_title("Sharpe Degradation (Train to Test)\nLower is better (Negative means Test > Train)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Degradation (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

for i, bar in enumerate(bars):
    height = bar.get_height()
    y_pos = height + 1 if height >= 0 else height - 4
    axes[1].text(bar.get_x() + bar.get_width()/2., y_pos, f"{degradations[i]:.1f}%", 
                 ha="center", va="bottom", fontweight="bold")

plt.suptitle("Model Evaluation: Generalization on Unseen Data (1 Year Test Set)", fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.show()


# ## 16. สรุป
# 
# ### Flow การทำงาน:
# ```
# 1. Import & Data        → ดาวน์โหลดราคา 30 สินทรัพย์ (4 ปี)
# 2. Returns & Stats      → คำนวณผลตอบแทน, Correlation
# 3. M-V Model Theory     → Mean, Variance, Covariance, Diversification
# 4. Constraints          → Sector weights, Stock weights
# 5. Optimization         → Max Sharpe (Unconstrained vs Constrained)
# 6. Analysis             → Weights, Efficient Frontier, Backtest
# 7. Real-World Costs     → Transaction Costs, Transaction Lots
# 8. Advanced             → Short Sell with Leverage Constraint
# ```
# 
# ### Parameters ที่ปรับได้:
# - `SECTOR_MAX_WEIGHT`: น้ำหนักสูงสุดต่อกลุ่ม (default: 25%)
# - `SECTOR_MIN_WEIGHT`: น้ำหนักต่ำสุดต่อกลุ่ม (default: 2%)
# - `STOCK_MAX_WEIGHT`: น้ำหนักสูงสุดต่อสินทรัพย์ (default: 15%)
# - `STOCK_MIN_WEIGHT`: น้ำหนักต่ำสุดต่อสินทรัพย์ (default: 1%)
# - `TOTAL_TC`: Transaction Cost ต่อครั้ง (default: 0.20%)
# - `REBALANCE_FREQ`: ความถี่ปรับพอร์ต (default: 21 วัน)
# - `PORTFOLIO_VALUE`: เงินลงทุน (default: 1,000,000 บาท)
# - `LEVERAGE_LEVELS`: ระดับ Leverage ที่ทดสอบ
