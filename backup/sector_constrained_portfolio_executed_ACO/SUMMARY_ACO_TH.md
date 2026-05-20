# 📋 สรุปผลการวิเคราะห์: ACO Asset Selection (Unconstrained)

สรุปเนื้อหาจาก Notebooks ในโฟลเดอร์นี้ ซึ่งมุ่งเน้นการคัดเลือกสินทรัพย์ด้วยอัลกอริทึม **Ant Colony Optimization (ACO)** แบบไม่จำกัดกลุ่มอุตสาหกรรม (Unconstrained) และการหาน้ำหนักที่เหมาะสมด้วย **EBGWO (Enhanced Grey Wolf Optimizer)**

---

## 📂 Notebooks ในโฟลเดอร์นี้

| ไฟล์ | คำอธิบาย |
| :--- | :--- |
| `ACO_Asset_selection_uncontrain.ipynb` | ACO คัดสินทรัพย์จาก Universe 629 ตัว ด้วย Simple Returns |
| `ACO_Asset_selection_uncontrain_log_retrun.ipynb` | เหมือน Notebook แรก แต่ใช้ **Log Returns** ในการคำนวณ Metrics |

---

## 🎯 วัตถุประสงค์หลัก

- คัดเลือกสินทรัพย์ที่ดีที่สุดจาก Universe ขนาดใหญ่ **629 ตัว** โดยไม่ถูกจำกัดว่าต้องมีหุ้นจากกลุ่มอุตสาหกรรมใดบ้าง
- ให้ ACO เรียนรู้ **Synergy ระหว่างคู่หุ้น** ผ่าน Pheromone Matrix แบบ 2 มิติ
- หาน้ำหนักพอร์ตที่ดีที่สุดด้วย EBGWO บน GPU เพื่อ **Maximize Sharpe Ratio**
- เปรียบเทียบหลายการตั้งค่า (Hyperparameter Configurations) เพื่อหาจุดที่เหมาะสมที่สุด

---

## 📊 Universe และข้อมูล

| รายการ | รายละเอียด |
| :--- | :--- |
| **พอร์ตเดิม** | 30 สินทรัพย์ใน 10 กลุ่ม |
| **S&P 500** | 503 ตัว (จาก Wikipedia) |
| **หุ้นไทย SET** | 69 ตัว |
| **หุ้นจีน ADRs** | 20 ตัว |
| **ETFs** | 37 ตัว |
| **รวม Universe** | **629 ตัว** (หลังลบ Duplicates) |
| **ผ่านเกณฑ์ข้อมูล ≥ 750 วัน** | **621 ตัว** |
| **ช่วงเวลาข้อมูล** | 2022-04-08 ถึง 2026-04-07 (4 ปี, 1,461 วัน) |
| **สกุลเงินฐาน** | THB (แปลงจาก USD/THB Real-time) |

---

## 🐜 กลไก ACO (Ant Colony Optimization)

### 1D ACO (การคัดเลือกอิสระ)
มดแต่ละตัวเลือกสินทรัพย์ที่จะนำเข้าพอร์ต โดยอ้างอิงจาก:
```
P(i) ∝ τ(i)^α × η(i)^β
```
- `τ(i)` = Pheromone ของสินทรัพย์ i (เรียนรู้จากประวัติการเลือก)
- `η(i)` = Heuristic = LW Sharpe Ratio ของสินทรัพย์ i (ปรับให้เป็นบวกเสมอ)
- `α` = น้ำหนัก Pheromone (ค่าทดสอบ: 1.0)
- `β` = น้ำหนัก Heuristic (ค่าทดสอบ: 2.0)

### 2D ACO (เรียนรู้ Synergy ระหว่างคู่หุ้น)
```
ก้าวแรก:  P(i) ∝ τ_start(i)^α × η(i)^β
ก้าวถัดไป: P(j|i) ∝ τ_2d(i,j)^α × η(j)^β
```
- `τ_2d(i,j)` = Pheromone ระหว่างหุ้น i กับ j
- มดเรียนรู้ว่า **หุ้นคู่ไหนจัดพอร์ตร่วมกันแล้วลด Volatility ได้ดี**

**Pheromone Update:**
```
τ ← (1 - ρ) × τ + Q × Fitness_best
```
- `ρ` = Evaporation Rate (0.1)
- `Q` = ปริมาณ Pheromone ที่ปล่อย (1.0)

---

## ⚙️ EBGWO (Enhanced Grey Wolf Optimizer) — การหาน้ำหนัก

หลังจาก ACO คัดสินทรัพย์แล้ว EBGWO หาน้ำหนักที่เหมาะสมที่สุด:

**Fitness Function:**
```
Fitness = Sharpe Ratio + λ × Normalized Entropy - Weight Penalty
         = (Rp - Rf) / σp  +  λ × H(w)/log(n)  -  100 × Σmax(w_i - max_w, 0)
```

- **Sharpe**: ผลตอบแทนต่อความเสี่ยง (ใช้ Ledoit-Wolf Covariance บน GPU)
- **Entropy Regularization** (`λ = 0.05`): ป้องกันน้ำหนักกระจุกตัวในหุ้นตัวเดียว
- **Weight Penalty**: บังคับให้น้ำหนักแต่ละตัวไม่เกิน `max_weight`
- **Monte Carlo Bootstrapping**: Resample ข้อมูลทุก Iteration เพื่อความทนทาน

**Hierarchy ของหมาป่า:**
- α (Alpha) = พอร์ตที่ให้ Fitness สูงสุด → นำทิศทาง
- β (Beta) = อันดับ 2
- δ (Delta) = อันดับ 3
- ω (Omega) = สมาชิกที่เหลือ → อัปเดตตำแหน่งตาม α, β, δ

---

## 📈 ผลลัพธ์การทดสอบ (ทุก Configuration)

### ผลลัพธ์รวมเรียงตาม Sharpe Ratio

| Run | Algorithm | Target Assets | Sharpe | Return/ปี | Vol/ปี | หมายเหตุ |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **EBGWO-B** | EBGWO (ไม่มี Rf) | 30 | **2.7999** | 27.79% | **9.93%** | 🏆 Best Overall |
| **EBGWO-C** | EBGWO (มี Rf) | 100 | **2.4121** | 36.96% | 13.53% | Pool ใหญ่ 100 ตัว |
| **ACO-B** | ACO Weight | 30 | **2.3365** | — | — | Best ACO Weight |
| **EBGWO-A** | EBGWO (มี Rf) | 30 | **2.3070** | 36.39% | 13.90% | Run มาตรฐาน |
| **Co-Evo Run 1** | ACO + EBGWO | 30 | **2.2838** | 41.52% | 16.29% | Sector-Constrained |
| **Co-Evo Run 3** | ACO + EBGWO + Escape | 30 | **2.1639** | 41.74% | 17.30% | 20,000 Iterations |
| **Co-Evo Run 6** | ANTS+EBGWO float16 | 30 | **1.8506** | 33.18% | 17.93% | min 3/sector |
| **Co-Evo Run 2** | Unified Fitness | 100 | **1.8624** | 76.29% | 38.65% | Return สูงแต่ Vol สูง |
| **Co-Evo Run 4** | ANTS+EBGWO | 30 | **1.7131** | 50.59% | 27.00% | 1,000,000 Iterations |
| **Co-Evo Run 5** | ANTS+EBGWO Wide | 30 | **1.6380** | 55.20% | 31.05% | evap=0.01 |
| **Co-Evo Run 8** | ANTS+EBGWO float16 | 30 | **1.5186** | 53.32% | 32.25% | min 1/sector |
| **Co-Evo Run 7** | ANTS+EBGWO float16 | 100 | **0.7275** | 19.57% | 20.91% | Target 100 อ่อนแอ |

---

## 🏆 ผลลัพธ์ที่ดีที่สุด: EBGWO-B (Sharpe 2.7999)

**Configuration**: 100 หมาป่า × 1,000 Iterations, ST=0.2, **ไม่มี Risk-Free Rate**

| Ticker | กลุ่ม | น้ำหนัก |
| :--- | :--- | :---: |
| **TISCO.BK** | Thai Stocks | 24.00% |
| **KTB.BK** | Thai Stocks | 17.42% |
| **CBOE** | Financials | 13.09% |
| **ADVANC.BK** | Thai Stocks | 10.47% |
| **FIX** | Industrials | 7.65% |
| **CAH** | Health Care | 6.29% |
| **DELTA.BK** | Thai Stocks | 5.08% |
| **EDU** | Chinese Stocks | 3.72% |
| **LLY** | Health Care | 2.41% |
| **SATS** | Communication Services | 1.53% |

**ผลลัพธ์:**
- Annual Return: **27.79%**
- Annual Volatility: **9.93%** ← ต่ำมาก
- **Sharpe Ratio: 2.7999** 🏆

---

## 🔄 Co-Evolutionary ACO + EBGWO (ผลลัพธ์ที่น่าสนใจ)

### Run 1 (3,000 Iterations, 500 Agents, Sector-Constrained)
- **ACO Fitness (Decorrelation):** 1.3055
- **EBGWO Sharpe: 2.2838**
- Annual Return: **41.52%** | Volatility: **16.29%**

| Ticker | กลุ่ม | น้ำหนัก |
| :--- | :--- | :---: |
| KTB.BK | Thai Stocks | 34.30% |
| FIX | Industrials | 16.55% |
| T | Communication Services | 14.25% |
| DELTA.BK | Thai Stocks | 9.77% |
| EDU | Chinese Stocks | 6.18% |
| TRUE.BK | Thai Stocks | 5.24% |

### Run 3 (20,000 Iterations + Escape Mechanism)
- **Sharpe: 2.1639** | Return: **41.74%** | Vol: **17.30%**
- มีกลไก Escape ป้องกัน Premature Convergence

---

## 🌟 Core Holdings — หุ้นที่ถูกเลือกซ้ำในทุก Run

| Ticker | กลุ่ม | เหตุผลที่ถูกเลือกบ่อย |
| :--- | :--- | :--- |
| **FIX** | Industrials | Sharpe สูง, Correlation ต่ำ |
| **KTB.BK** | Thai Stocks | Low Vol, High Sharpe ในหน่วย THB |
| **DELTA.BK** | Thai Stocks | Growth Driver ของตลาดไทย |
| **CAH** | Health Care | Defensive, Correlation ต่ำกับ Tech |
| **CBOE** | Financials | Market Structure ที่เสถียร |
| **ADVANC.BK** | Thai Stocks | Dividend Yield + Stability |
| **TISCO.BK** | Thai Stocks | Vol ต่ำที่สุดในกลุ่มไทย |
| **TTB.BK** | Thai Stocks | Consistent Returns |
| **TIPS** | Bonds | Inflation Hedge |
| **SATS** | Comm. Services | Low Correlation กับทุกกลุ่ม |
| **PLTR** | IT | Growth + Low Corr กับ Old Economy |
| **VRT** | Industrials | AI Infrastructure Play |
| **LLY** | Health Care | Best Sharpe ใน Healthcare Global |
| **EDU** | Chinese Stocks | High Return / High Risk |
| **GLD** | Commodities | Safe Haven Asset |

---

## 🔬 ความแตกต่างระหว่าง 2 Notebooks

| ประเด็น | Simple Returns | Log Returns |
| :--- | :--- | :--- |
| **สูตรผลตอบแทน** | `(P_t - P_{t-1}) / P_{t-1}` | `ln(P_t / P_{t-1})` |
| **ข้อดี** | เข้าใจง่าย ใช้งานทั่วไป | Robust ต่อ Outliers มากกว่า |
| **Sharpe ที่ได้** | สูงกว่าเล็กน้อย | ต่ำกว่าเล็กน้อย (Conservative) |
| **เหมาะกับ** | ตลาดปกติ, สินทรัพย์ที่ราคาไม่กระโดด | Crypto, หุ้น Volatile สูง |
| **ผลลัพธ์รวม** | ใกล้เคียงกันมาก | Correlation Matrix เสถียรกว่า |

---

## 💡 ข้อสรุปและคำแนะนำ

1. **ACO แบบไม่จำกัดกลุ่ม** สามารถค้นพบสินทรัพย์ที่ "ซ่อนอยู่" ซึ่งกลยุทธ์ Sector-Constrained แบบดั้งเดิมอาจมองข้าม เช่น TISCO.BK และ KTB.BK ที่ให้ Volatility ต่ำมากในหน่วย THB

2. **EBGWO ไม่มี Risk-Free Rate** ให้ Sharpe สูงกว่า เพราะไม่ถูก Penalize ด้วยค่า Rf ในการคำนวณ Fitness — เหมาะสำหรับช่วงที่ดอกเบี้ยสูง (Rf > 4%)

3. **Co-Evolutionary ACO + EBGWO** ให้ผลตอบแทนสูงกว่า (41-55%/ปี) แต่แลกด้วย Volatility ที่สูงขึ้น — เหมาะสำหรับนักลงทุนที่รับความเสี่ยงได้สูง

4. **หุ้นไทยครองน้ำหนักสูง** ใน Best Run (TISCO.BK 24%, KTB.BK 17%, ADVANC.BK 10%) เพราะในหน่วย THB หุ้นไทยมี Risk-Adjusted Return ที่น่าสนใจโดยไม่มี Currency Risk

5. **Log Returns แนะนำสำหรับ Universe ที่มี Crypto** เพราะลด Impact ของ BTC/ETH ที่มีการเคลื่อนไหวรุนแรง ทำให้ Covariance Matrix มีเสถียรภาพมากขึ้น

---

*สรุปจาก `ACO_Asset_selection_uncontrain.ipynb` และ `ACO_Asset_selection_uncontrain_log_retrun.ipynb`*