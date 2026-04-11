# 📊 สรุปผลการวิเคราะห์: Ledoit-Wolf Portfolio Optimization
## โฟลเดอร์ `sector_constrained_portfolio_executed_LedoitWolf`

> **Notebooks ในโฟลเดอร์นี้**: 3 ไฟล์  
> **เทคนิคหลัก**: Ledoit-Wolf Shrinkage Covariance + EBGWO + 2D-ACO (บน GPU/CUDA)  
> **Benchmark**: SPY (S&P 500 ETF)  
> **ภาษา**: ไทย

---

## สารบัญ

1. [ภาพรวมโฟลเดอร์](#1-ภาพรวมโฟลเดอร์)
2. [Ledoit-Wolf.ipynb — พื้นฐานและการทดสอบ](#2-ledoit-wolfipynb--พื้นฐานและการทดสอบ)
3. [Ledoit-Wolf-Optimze_.ipynb — Multi-Asset Walk-Forward](#3-ledoit-wolf-optimze_ipynb--multi-asset-walk-forward)
4. [Ledoit-Wolf-Optimze_ACO_Selection.ipynb — ACO + EBGWO](#4-ledoit-wolf-optimze_aco_selectionipynb--aco--ebgwo)
5. [สรุปเปรียบเทียบทั้ง 3 Notebooks](#5-สรุปเปรียบเทียบทั้ง-3-notebooks)

---

## 1. ภาพรวมโฟลเดอร์

โฟลเดอร์นี้รวม Notebooks ที่พัฒนาขึ้นเพื่อแก้ปัญหาหลักของ Mean-Variance Optimization แบบดั้งเดิม: **Sample Covariance Matrix ที่ไม่เสถียรเมื่อจำนวนสินทรัพย์มาก** โดยใช้ **Ledoit-Wolf Shrinkage** ประมวลผลบน PyTorch GPU ร่วมกับอัลกอริทึม Metaheuristic ในการหาน้ำหนักที่เหมาะสม

| Notebook | จุดเด่น | Universe | ช่วงข้อมูล |
| :--- | :--- | :--- | :--- |
| `Ledoit-Wolf.ipynb` | พื้นฐาน LW + EBGWO, S&P 500 | S&P 500 (503 ตัว) | 2010–2024 |
| `Ledoit-Wolf-Optimze_.ipynb` | Multi-Asset Walk-Forward | S&P 500 + Thai + China + Crypto + ETFs (~573 ตัว) | 2017–2025 |
| `Ledoit-Wolf-Optimze_ACO_Selection.ipynb` | 2D-ACO คัดหุ้น + EBGWO หาน้ำหนัก | Multi-Asset 573 ตัว | 2017–2025 |

---

## 2. Ledoit-Wolf.ipynb — พื้นฐานและการทดสอบ

### 2.1 วัตถุประสงค์

Notebook นี้เป็นจุดเริ่มต้น สาธิตการคำนวณ **Ledoit-Wolf Shrinkage Covariance Matrix บน GPU** โดยเริ่มจากตัวอย่างเล็ก 20 หุ้น แล้วขยายไปสู่ S&P 500 เต็มรูปแบบ 503 ตัว

### 2.2 กระบวนการทำงาน

```
[1] ดาวน์โหลดข้อมูล S&P 500 (2010–2024, 503 tickers)
       ↓
[2] คำนวณ Ledoit-Wolf Shrinkage Covariance บน GPU (PyTorch)
    Σ_LW = (1 - δ) × Σ_sample + δ × (μ_var × I)
    δ = Dynamic: คำนวณจาก b²/d²  หรือ Fixed: δ = 0.20
       ↓
[3] สร้าง Correlation Matrix → Hierarchical Clustering (Ward Linkage)
    Distance = √(0.5 × (1 - Corr))
       ↓
[4] คัดเลือกหุ้นตัวแทน (Lowest Risk หรือ Highest Sharpe) ต่อ Cluster
       ↓
[5] EBGWO / Sortino / Entropy Optimization สำหรับน้ำหนักสินทรัพย์
       ↓
[6] Walk-Forward Backtest (Quarterly Rebalance, 2013–2025)
```

### 2.3 ผลการคัดเลือกหุ้น (S&P 500 เท่านั้น)

**ชุดที่ 1 — 10 หุ้น (ความเสี่ยงต่ำ, จาก Cluster)**

| อันดับ | Ticker | กลุ่ม |
| :---: | :--- | :--- |
| 1 | PG | Consumer Staples |
| 2 | JNJ | Health Care |
| 3 | DUK | Utilities |
| 4 | PSA | Real Estate |
| 5 | IBM | Information Technology |
| 6 | BDX | Health Care |
| 7 | XOM | Energy |
| 8 | BRK-B | Financials |
| 9 | LIN | Materials |
| 10 | MCD | Consumer Discretionary |

**ชุดที่ 2 — 30 หุ้น (Sharpe สูงสุดต่อ Cluster)**

| อันดับ | Ticker | อันดับ | Ticker | อันดับ | Ticker |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | AWK | 11 | CHTR | 21 | CPRT |
| 2 | SBAC | 12 | CTAS | 22 | KDP |
| 3 | EXR | 13 | NVR | 23 | CHD |
| 4 | TPL | 14 | ORLY | 24 | NOC |
| 5 | BRK-B | 15 | COST | 25 | NDAQ |
| 6 | BKNG | 16 | DPZ | 26 | AJG |
| 7 | ODFL | 17 | SW | 27 | LLY |
| 8 | ROP | 18 | AVGO | 28 | UNH |
| 9 | LYB | 19 | WST | 29 | COR |
| 10 | APH | 20 | CDNS | 30 | HCA |

### 2.4 Top Sharpe (Log Return) ของ 30 หุ้นที่คัดเลือก

| Ticker | Annual Return | Annual Vol | Sharpe |
| :--- | :---: | :---: | :---: |
| **DPZ** | 29.04% | 29.83% | **0.973** |
| **COST** | 19.61% | 20.44% | **0.959** |
| **CTAS** | 23.86% | 25.03% | **0.954** |
| **LLY** | 22.89% | 24.35% | **0.940** |
| **AJG** | 19.26% | 20.74% | **0.929** |
| **ODFL** | 27.70% | 30.24% | **0.916** |
| **CDNS** | 27.17% | 30.10% | **0.903** |
| **AVGO** | 31.55% | 35.21% | **0.896** |

### 2.5 ผลการ Optimize น้ำหนัก (Static, ไม่ Walk-Forward)

**EBGWO — Sharpe Objective (ไม่จำกัด Max Weight)**

| Metric | ผลลัพธ์ |
| :--- | :---: |
| Annual Return | 25.00% |
| Annual Volatility | 16.59% |
| **Sharpe Ratio** | **1.2456** |

น้ำหนักสำคัญ: LLY 21.81%, DPZ 19.87%, TPL 11.26%, ORLY 9.49%, COST 8.47%

**EBGWO — Sortino Objective (จำกัด Max 15%)**

| Metric | ผลลัพธ์ |
| :--- | :---: |
| Annual Return | 24.82% |
| Downside Vol | 10.46% |
| **Sortino Ratio** | **1.9583** |

น้ำหนักสำคัญ: LLY 15.00%, DPZ 15.00%, COST 14.97%, ORLY 13.46%

### 2.6 Walk-Forward Optimization Results (2013–2025)

**Rebalance ทุก ~3 เดือน เปรียบเทียบกับ SPY**

| กลยุทธ์ | Cum Return | Ann Return | Ann Vol | Sharpe | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EBGWO + Monte Carlo (Sharpe)** | 640.15% | 16.84% | 16.66% | 1.0107 | -33.62% |
| **Entropy-Regularised GA** | 670.96% | 17.43% | 18.20% | 0.9574 | -37.63% |
| **🏆 EBGWO + MC + Entropy** | **1,209.87%** | **21.25%** | **16.65%** | **1.2760** | -35.46% |
| **EBGWO + MC (Sortino)** | 575.68% | 16.10% | 16.44% | 0.9796 | -35.79% |
| **SPY (Benchmark)** | 487.91% | 15.11% | 16.93% | 0.8925 | -33.72% |

> [!IMPORTANT]
> **ผลลัพธ์ดีที่สุด**: EBGWO + Monte Carlo + Entropy Regularization ให้ผลตอบแทนสะสม **1,209.87%** เทียบกับ SPY **487.91%** — สูงกว่าถึง **2.5 เท่า** โดยมี Sharpe สูงถึง **1.2760**

### 2.7 ผลเฉพาะปี 2025 (Out-of-Sample)

| กลยุทธ์ | Cum Return | Ann Return | Ann Vol | Sharpe | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🏆 EBGWO + MC + Entropy** | **28.31%** | **26.75%** | 17.40% | **1.5374** | **-11.30%** |
| **Entropy-Regularised GA** | 19.77% | 19.84% | 17.75% | 1.1176 | -12.26% |
| **Static Backtest (no WF)** | 10.74% | 12.14% | 18.87% | 0.6434 | -16.68% |
| **SPY (Benchmark)** | 18.60% | 19.47% | 19.55% | 0.9961 | -18.76% |

**น้ำหนักล่าสุด (2025, EBGWO+MC+Entropy)**: HCA 14.09%, COR 14.05%, AVGO 14.01%, LLY 13.97%, ORLY 13.04%, NOC 12.41%, BKNG 11.36%, COST 5.70%

---

## 3. Ledoit-Wolf-Optimze_.ipynb — Multi-Asset Walk-Forward

### 3.1 วัตถุประสงค์

ขยาย Universe จาก S&P 500 เท่านั้น ไปสู่ **Multi-Asset Portfolio** ครอบคลุมหุ้นไทย, หุ้นจีน, Crypto และ ETFs เพื่อทดสอบประสิทธิภาพของ Ledoit-Wolf ในสภาพแวดล้อมสินทรัพย์ที่หลากหลายและมีความสัมพันธ์ต่ำกว่า

### 3.2 Universe ที่ใช้

| กลุ่มสินทรัพย์ | จำนวน |
| :--- | :---: |
| S&P 500 | ~503 ตัว |
| หุ้นไทย (SET) | ~60 ตัว |
| หุ้นจีน (ADRs) | ~20 ตัว |
| ETFs (Bond, Commodity, Crypto) | ~37 ตัว |
| **รวมหลังกรอง** | **~573 ตัว** |

### 3.3 หุ้น 10 ตัวที่คัดเลือก (Highest Sharpe per Cluster, LW Metric)

| Ticker | กลุ่ม | Annual Return | Annual Vol | Sharpe | Max Drawdown |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NVDA** | Information Technology | 39.09% | 40.70% | **0.960** | -61.87% |
| **PGR** | Financials | 17.04% | 20.61% | **0.827** | -22.87% |
| **FIX** | Industrials | 23.00% | 32.57% | **0.706** | -50.37% |
| **BTC-USD** | Crypto | 39.56% | 58.44% | **0.677** | -83.27% |
| **COM7.BK** | Thai Stocks | 19.05% | 31.86% | **0.598** | -59.24% |
| **EQIX** | Real Estate | 11.72% | 22.81% | **0.514** | -32.36% |
| **IBKR** | Financials | 13.25% | 26.58% | **0.498** | -54.14% |
| **ATO** | Utilities | 8.24% | 18.72% | **0.440** | -29.85% |
| **NTES** | Chinese Stocks | 11.78% | 36.02% | **0.327** | -48.83% |
| **VCSH** | Bonds | 1.85% | 6.59% | **0.280** | -12.35% |

### 3.4 Walk-Forward Results (Quarterly Rebalance, 2017–2025)

| ช่วงเวลา | Cum Return | Ann Return | Ann Vol | Sharpe | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2017–2025 (เต็มช่วง)** | **599.40%** | **16.34%** | 16.11% | **1.0147** | -33.07% |
| **2022–2025 (Backtest)** | **87.03%** | **12.07%** | 15.89% | **0.7597** | -31.17% |
| **2025 เท่านั้น** | 15.51% | 11.26% | 15.99% | 0.7042 | -19.06% |
| **SPY (2017–2025)** | 196.39% | 9.60% | 15.42% | 0.6222 | -35.75% |
| **SPY (2022–2025)** | 43.41% | 7.33% | 14.89% | 0.4922 | -26.22% |
| **SPY (2025 เท่านั้น)** | 16.44% | 11.81% | 15.98% | 0.7391 | -19.21% |

**น้ำหนักล่าสุด (2025, Quarterly Rebalance)**:
BTC-USD 12.46%, NVDA 12.29%, NTES 10.54%, IBKR 10.37%, EQIX 10.01%, COM7.BK 9.65%, FIX 9.49%, VCSH 8.57%, ATO 8.41%, PGR 8.21%

> [!NOTE]
> Multi-Asset Portfolio ให้ผลสะสม **599.40%** เทียบ SPY **196.39%** (สูงกว่า ~3 เท่า) แต่ในปี 2025 เพียงปีเดียวเกือบเทียบเท่า SPY (0.7042 vs 0.7391)

---

## 4. Ledoit-Wolf-Optimze_ACO_Selection.ipynb — ACO + EBGWO

### 4.1 วัตถุประสงค์

รวม **2D-Ant Colony Optimization (2D-ACO)** ในขั้นตอนคัดเลือกสินทรัพย์ เข้ากับ **EBGWO + Ledoit-Wolf** ในขั้นตอนหาน้ำหนัก เพื่อทดสอบว่า ACO ช่วยให้ได้ชุดสินทรัพย์ที่ดีกว่า Clustering ธรรมดาหรือไม่

### 4.2 กระบวนการ 2D-ACO

```
[1] คำนวณ LW Covariance Matrix (573 สินทรัพย์, GPU)
[2] 2D-ACO เรียนรู้ Pheromone Matrix (N×N)
    - มดเดินเลือกหุ้นทีละตัว (ขั้นแรก: pheromones_start)
    - ขั้นถัดไป: pheromones_2d[last_picked, :] — เรียนรู้ Synergy
    - Fitness = Annualised Sharpe Ratio (LW Covariance)
    - Pheromone Update: δ += Q × Sharpe ของเส้นทางที่ดีที่สุด
[3] ได้หุ้น 30 ตัวที่ ACO เลือก → Expected Portfolio Sharpe = 1.1273
[4] EBGWO + Monte Carlo + Entropy หาน้ำหนักในแต่ละ Rebalance
[5] Walk-Forward Backtest
```

### 4.3 หุ้น 30 ตัวที่ 2D-ACO คัดเลือก

**Expected Portfolio Annualised Sharpe Ratio: 1.1273**

| อันดับ | Ticker | อันดับ | Ticker | อันดับ | Ticker |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | AMZN | 11 | DBA | 21 | LLY |
| 2 | ANET | 12 | DHR | 22 | NTES |
| 3 | ARES | 13 | EQIX | 23 | NVDA |
| 4 | ATO | 14 | EXR | 24 | PGR |
| 5 | AZO | 15 | FICO | 25 | STLD |
| 6 | BCP.BK | 16 | FIX | 26 | TISCO.BK |
| 7 | BTC-USD | 17 | GLD | 27 | TMUS |
| 8 | CHD | 18 | HLT | 28 | TPL |
| 9 | COM7.BK | 19 | IBKR | 29 | UNH |
| 10 | COST | 20 | IEF | 30 | WRB |

### 4.4 Walk-Forward Results (ACO + EBGWO)

| ช่วงเวลา | Cum Return | Ann Return | Ann Vol | Sharpe | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2017–2025 (Quarterly)** | **394.16%** | **13.19%** | **12.92%** | **1.0211** | -32.27% |
| **2022–2025 (Monthly)** | **82.11%** | **11.09%** | **12.14%** | **0.9130** | **-16.92%** |
| **2025 เท่านั้น** | 4.36% | 3.75% | 12.59% | 0.2979 | **-15.23%** |
| **SPY (2017–2025)** | 196.39% | 9.60% | 15.42% | 0.6222 | -35.75% |
| **SPY (2022–2025)** | 43.41% | 7.33% | 14.89% | 0.4922 | -26.22% |
| **SPY (2025 เท่านั้น)** | 16.44% | 11.81% | 15.98% | 0.7391 | -19.21% |

**น้ำหนักล่าสุด (2025, 30 สินทรัพย์)**:

| Ticker | น้ำหนัก | Ticker | น้ำหนัก | Ticker | น้ำหนัก |
| :--- | :---: | :--- | :---: | :--- | :---: |
| ANET | 4.98% | WRB | 4.82% | HLT | 4.41% |
| IBKR | 4.89% | NTES | 4.79% | DBA | 4.39% |
| EQIX | 4.87% | IEF | 4.69% | FIX | 4.33% |
| BTC-USD | 4.86% | COST | 4.69% | AXON | 4.15% |
| TISCO.BK | 4.54% | ... | ... | NVDA | 0.02% |

> [!TIP]
> ACO ให้ **Volatility ต่ำที่สุด** ในบรรดาทุก Notebook ที่ทดสอบ: **12.14%** (2022–2025) และ **12.92%** (2017–2025) เทียบ SPY ~15% แสดงว่า ACO ช่วยสร้างพอร์ตที่ "นิ่ง" กว่า
>
> อย่างไรก็ตาม ในปี 2025 เพียงปีเดียว ACO ตามหลัง SPY (3.75% vs 11.81%) แต่ Max Drawdown ต่ำกว่า SPY มาก (-15.23% vs -19.21%)

### 4.5 Dynamic ACO (Monthly Rebalance — คัดหุ้นใหม่ทุกเดือน)

| Metric | ผลลัพธ์ | SPY |
| :--- | :---: | :---: |
| Cumulative Return | 274.73% | 196.39% |
| Annual Return | 11.05% | 9.60% |
| Annual Volatility | 12.88% | 15.42% |
| Sharpe Ratio | 0.8576 | 0.6222 |
| Max Drawdown | -31.72% | -35.75% |

---

## 5. สรุปเปรียบเทียบทั้ง 3 Notebooks

### 5.1 ตารางสรุป Walk-Forward ทุกกลยุทธ์ (เรียงตาม Sharpe)

| Notebook | กลยุทธ์ | ช่วงเวลา | Cum Return | Ann Return | Ann Vol | Sharpe | Max DD |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LW.ipynb** | EBGWO+MC+Entropy | 2013–2025 | **1,209.87%** | **21.25%** | 16.65% | **1.2760** | -35.46% |
| **LW.ipynb** | EBGWO+MC+Entropy (2025) | 2025 only | 28.31% | 26.75% | 17.40% | **1.5374** | **-11.30%** |
| **LW.ipynb** | EBGWO+MC (Sharpe) | 2013–2025 | 640.15% | 16.84% | 16.66% | 1.0107 | -33.62% |
| **LW-ACO.ipynb** | 2D-ACO + EBGWO | 2017–2025 | 394.16% | 13.19% | **12.92%** | 1.0211 | -32.27% |
| **LW-Opt.ipynb** | Multi-Asset EBGWO | 2017–2025 | 599.40% | 16.34% | 16.11% | 1.0147 | -33.07% |
| **LW-ACO.ipynb** | ACO+EBGWO (2022–2025) | 2022–2025 | 82.11% | 11.09% | **12.14%** | 0.9130 | **-16.92%** |
| **LW.ipynb** | ER-GA | 2013–2025 | 670.96% | 17.43% | 18.20% | 0.9574 | -37.63% |
| — | **SPY (Benchmark)** | 2017–2025 | 196.39% | 9.60% | 15.42% | 0.6222 | -35.75% |

### 5.2 การค้นพบสำคัญ

1. **EBGWO + Monte Carlo + Entropy Regularization คือสุดยอด**: ให้ผลสะสม **1,209.87%** และ Sharpe **1.2760** ซึ่งดีที่สุดในทุก Notebook บน S&P 500 และยังให้ผลในปี 2025 สูงกว่า SPY (28.31% vs 18.60%) พร้อม Max Drawdown ต่ำเพียง **-11.30%**

2. **Ledoit-Wolf Shrinkage บน GPU คือหัวใจ**: Covariance Matrix ที่เสถียรกว่า Sample Cov ธรรมดา ช่วยให้ EBGWO หาน้ำหนักที่ Robust ต่อ Regime Change ได้ดีขึ้น

3. **2D-ACO ลด Volatility ได้จริง**: ACO Selection ให้ Vol เพียง 12.14–12.92% เทียบ SPY 15.42% แสดงว่าการเรียนรู้ Synergy ระหว่างคู่หุ้น (Pheromone Matrix 2D) ช่วยสร้างพอร์ตที่ "นิ่ง" ได้จริง

4. **Multi-Asset ชนะ SPY อย่างชัดเจน**: ทั้ง 3 Notebook ให้ Sharpe > 0.9 เทียบ SPY 0.6222 ในช่วง 2017–2025 ยืนยันว่าการกระจายความเสี่ยงข้ามตลาด (ไทย + จีน + Crypto + US) สร้างมูลค่าเพิ่มอย่างมีนัยสำคัญ

5. **ปี 2025 เป็นบทพิสูจน์**: EBGWO+MC+Entropy ยังชนะ SPY ในปี 2025 (Sharpe 1.5374 vs 0.9961) ขณะที่ ACO แพ้ SPY ในปีนี้ แต่ยังมี Max Drawdown ต่ำกว่า SPY

### 5.3 คำแนะนำตามวัตถุประสงค์

| วัตถุประสงค์ | Notebook/กลยุทธ์ที่แนะนำ | เหตุผล |
| :--- | :--- | :--- |
| **ผลตอบแทนสูงสุดระยะยาว** | `LW.ipynb` → EBGWO+MC+Entropy | Cum 1,209.87%, Sharpe 1.2760 |
| **ความผันผวนต่ำที่สุด** | `LW-ACO.ipynb` → ACO+EBGWO | Vol 12.14%, Max DD -16.92% |
| **สมดุล Multi-Asset** | `LW-Optimze_.ipynb` → Multi-Asset WF | 599.40%, Sharpe 1.0147, กระจายทั่วโลก |
| **พร้อม Rebalance อัตโนมัติ** | `LW-ACO.ipynb` → Dynamic ACO | เลือกหุ้นใหม่ทุกเดือนอัตโนมัติ |

---

*สรุปนี้จัดทำจากผลลัพธ์จริงใน Notebooks ทั้ง 3 ไฟล์ในโฟลเดอร์นี้*  
*⚠️ คำเตือน: เนื้อหานี้เพื่อวัตถุประสงค์ทางการศึกษาเท่านั้น ไม่ถือเป็นคำแนะนำในการลงทุน*