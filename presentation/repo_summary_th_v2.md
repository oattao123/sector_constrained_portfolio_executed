# รายงานสรุป Repository: Sector-Constrained Portfolio Optimization

เอกสารนี้สรุป repo ตามหัวข้อที่กำหนด โดยอ้างอิงจากโครงสร้างโปรเจกต์, `manual.md`, `config/settings.json`, `config/assets.csv`, `portfolio_optimization/selectors.py`, `portfolio_optimization/optimizers.py` และ `portfolio_optimization/backtest.py`

---

## 1. Data Collection

ระบบนี้ใช้ไฟล์ `config/assets.csv` เป็นตัวกำหนด universe ของสินทรัพย์ และใช้ `SPY` เป็น benchmark สำหรับเปรียบเทียบกับ S&P 500

| กลุ่มสินทรัพย์ | สถานะใน repo |
|---|---|
| S&P 500 | มีหุ้น US หลายตัว และใช้ `SPY` เป็น benchmark |
| Thai | มีหุ้นไทย เช่น `KTB.BK`, `TISCO.BK`, `TTB.BK`, `TCAP.BK`, `SCB.BK` |
| Chinese | ยังไม่มีใน asset universe ปัจจุบัน |
| ETC / ETF / Bond | มี `TIPS` ซึ่งอยู่ในกลุ่ม Bonds |
| Crypto | ยังไม่มีใน asset universe ปัจจุบัน |

สรุป: repo ปัจจุบันรองรับข้อมูลหลักคือ **หุ้น US, หุ้นไทย, Bonds/ETF-like asset และ SPY benchmark** แต่ยังไม่ครอบคลุม Chinese และ Crypto ตามหัวข้อที่ระบุ

---

## 2. Train Data และ Test Data

ช่วงเวลาที่ต้องการตามโจทย์:

```text
Train:
start_date = 2015-01-01
end_date   = 2024-01-01

Test:
start_date = 2025-01-01
end_date   = 2026-01-01
```

ค่าปัจจุบันใน `config/settings.json`:

```json
{
  "start_date": "2015-01-01",
  "end_date": "2026-06-01",
  "insample_end_date": "2022-06-01",
  "outsample_end_date": "2026-06-01"
}
```

ข้อสังเกต:

- `start_date` ตรงกับ requirement คือ `2015-01-01`
- `insample_end_date` ปัจจุบันคือ `2022-06-01` ยังไม่ใช่ `2024-01-01`
- `outsample_end_date` ปัจจุบันคือ `2026-06-01` ยังไม่ใช่ `2026-01-01`
- ถ้าต้องการ train/test ตามโจทย์ ต้องแก้ config ให้ตรงกับช่วงเวลานั้น

---

## 3. Data Preprocess

Pipeline มีขั้นตอน preprocessing สำคัญดังนี้:

| ขั้นตอน | คำอธิบาย |
|---|---|
| Missing data | ตรวจและจัดการข้อมูลราคาที่ขาดหาย |
| Price transform | แปลงข้อมูลราคาเป็น return |
| Log return | ใช้ log return สำหรับวิเคราะห์ผลตอบแทน |
| Currency conversion | แปลงสินทรัพย์ USD เป็น THB ผ่าน `USDTHB=X` |
| Quality check | ตรวจ daily return และ annual return ที่ผิดปกติ |
| Ledoit-Wolf Shrinkage Covariance | ใช้ covariance แบบ shrinkage เพื่อลด noise และเพิ่มเสถียรภาพ |

Ledoit-Wolf covariance เป็นส่วนสำคัญ เพราะถูกใช้ทั้งใน asset selection และ weight optimization เพื่อคำนวณ volatility และ risk ของพอร์ต

---

## 4. Modeling: Asset Selection

Asset selection อยู่ในไฟล์ `portfolio_optimization/selectors.py`

### 4.1 Top Sharpe Ratio

ใน repo ยังไม่มี function แยกชื่อ Top 10 Sharpe Ratio โดยตรง แต่ Sharpe Ratio ถูกใช้เป็นตัวแปรหลักในหลาย selector เช่น HRP Max Sharpe และ ACO heuristic

### 4.2 Diversification Score

คำนวณคะแนนจาก:

```text
Diversification Score = Sharpe Ratio x (1 - Average Correlation)
```

แนวคิดคือเลือกสินทรัพย์ที่มี Sharpe ดีและไม่เคลื่อนไหวเหมือนสินทรัพย์อื่นมากเกินไป

### 4.3 Ledoit-Wolf + Diversification Score

เหมือน Diversification Score ปกติ แต่ใช้ covariance และ correlation ที่ผ่าน Ledoit-Wolf Shrinkage แล้ว ทำให้ค่าความเสี่ยงและ correlation เสถียรกว่า sample covariance

### 4.4 HRP Clustering + Max Sharpe per Cluster

ขั้นตอน:

1. แปลง correlation เป็น distance
2. ทำ hierarchical clustering
3. เลือก asset ที่ Sharpe สูงที่สุดจากแต่ละ cluster

เหมาะกับกรณีที่ต้องการลดการเลือก asset ที่ซ้ำกลุ่มความเสี่ยงเดียวกัน

### 4.5 HRP Clustering + Max Diversification Score per Cluster

คล้ายวิธี HRP Max Sharpe แต่แทนที่จะเลือก Sharpe สูงสุด จะเลือก asset ที่มี Diversification Score สูงสุดในแต่ละ cluster

### 4.6 2D-ACO แบบอิสระ

ใช้ Ant Colony Optimization เพื่อเลือกชุดสินทรัพย์โดยตรง:

- ant แต่ละตัวสร้าง path ของ asset
- ใช้ pheromone และ heuristic จาก Sharpe
- ประเมิน portfolio ด้วย equal weight
- path ที่ fitness ดีจะได้รับ pheromone เพิ่ม

### 4.7 2D-ACO + Cluster Constraint

ใช้ ACO ร่วมกับ HRP clustering:

- แบ่ง asset เป็น cluster ก่อน
- ant ต้องเลือกตัวแทนจากแต่ละ cluster
- ช่วยลดการกระจุกตัวในกลุ่มสินทรัพย์ที่ correlation สูง

---

## 5. Asset Weight Optimization

หลังจากเลือก asset แล้ว ระบบใช้ optimizer เพื่อหา weight ของแต่ละสินทรัพย์

ไฟล์หลักคือ `portfolio_optimization/optimizers.py`

| Algorithm | รายละเอียด |
|---|---|
| PSO | Particle Swarm Optimization สำหรับหา weight vector |
| ACO / ACO_R | Ant Colony Optimization สำหรับ continuous weight หรือ asset selection |
| EBGWO | Enhanced Binary Grey Wolf Optimizer |
| CLPSO | Comprehensive Learning PSO |
| APSO | Adaptive PSO |
| LAPSO | Landscape Adaptive PSO |
| CIAC | Continuous Interacting Ant Colony |

ทุก algorithm ใช้ weight vector ที่เป็นค่า float และ normalize ให้ผลรวมของ weight เท่ากับ 1

---

## 6. Objective Function

Objective หลักของระบบคือ:

```text
Maximize Sharpe Ratio
```

รูปแบบ fitness โดยรวม:

```text
fitness = Sharpe Ratio + Entropy Regularization - Weight Penalty
```

ส่วนประกอบ:

- Sharpe Ratio ใช้วัดผลตอบแทนต่อความเสี่ยง
- Entropy ช่วยให้พอร์ตกระจายตัวมากขึ้น
- Weight penalty ลงโทษกรณี weight เกิน constraint

---

## 7. Constraints

Requirement:

```text
min weight = 0%
max weight = 10%
```

สถานะใน repo:

| Constraint | สถานะ |
|---|---|
| Minimum weight | 0% |
| Maximum weight | 10% ผ่าน `max_weight=0.1` |
| Sum of weights | normalize ให้รวมเป็น 1 |
| Short sell | ไม่รองรับ เพราะ weight ถูก clamp ระหว่าง 0 และ 1 |

ดังนั้น constraint หลักเรื่อง min 0% และ max 10% มีอยู่ใน repo แล้ว

---

## 8. Training: Representation and Encoding

| Component | Representation |
|---|---|
| ACO asset selection | integer index ของ asset |
| ACO cluster selection | integer index ภายใน cluster |
| PSO / GWO / SI / DE-like optimizers | float vector ของ portfolio weights |
| Weight vector | continuous value และ normalize ให้ sum = 1 |

สรุป:

- `int` ใช้กับการเลือก asset เช่น ACO
- `float` ใช้กับ weight optimization เช่น PSO, GWO-style, CIAC, LAPSO, APSO

---

## 9. Algorithm Mechanism

### PSO

PSO ใช้ population ของ particles โดยแต่ละ particle คือ weight vector

กลไก:

- เริ่มจาก random weights
- คำนวณ fitness จาก Sharpe Ratio
- เก็บ personal best และ global best
- ปรับ velocity และ position
- normalize weight ใหม่ทุก iteration

### ACO

ACO ใช้ pheromone เพื่อเรียนรู้ asset หรือ solution ที่ดี

กลไก:

- ant สุ่มสร้าง solution ตาม probability
- probability มาจาก pheromone และ heuristic
- solution ที่ fitness ดีจะเพิ่ม pheromone
- pheromone เก่าระเหยด้วย evaporation rate

### GWO / EBGWO

GWO ใช้แนวคิด alpha, beta, delta wolves เป็นตัวนำทาง

กลไก:

- wolf แต่ละตัวคือ candidate weight
- alpha, beta, delta คือ solution ที่ดีที่สุด
- wolf อื่นปรับตำแหน่งตามผู้นำ
- EBGWO เพิ่ม exploration mechanism เพื่อช่วยหนี local optimum

### Other SI Algorithms

Repo ยังมี variation เพิ่มเติม เช่น CLPSO, APSO, LAPSO, ACO_R และ CIAC ซึ่งเป็นวิธีปรับปรุงการค้นหา weight ให้เสถียรขึ้นหรือ adaptive มากขึ้น

---

## 10. Convergence and Stability

ระบบบันทึก convergence ระหว่าง training:

```text
Avg_Best_Convergence
Avg_Avg_Convergence
```

และสร้างกราฟ:

```text
convergence.png
```

Convergence handling ที่มี:

- บันทึก best fitness ทุก iteration
- บันทึก average fitness ทุก iteration
- ใช้ pheromone evaporation ใน ACO
- ใช้ pheromone clamp เพื่อไม่ให้ค่า pheromone สูงหรือต่ำเกินไป
- ใช้ stagnation counter ใน ACO + EBGWO
- reset บางส่วนเมื่อ optimizer ไม่พัฒนานานเกินไป
- ใช้ multiple trials เพื่อวัด mean และ standard deviation

---

## 11. Metrics

Metric ที่ repo คำนวณใน `compute_metrics()`:

| Metric | สถานะ |
|---|---|
| Sharpe Ratio | มี |
| Cumulative Return | มี |
| Max Drawdown | มี |
| Annual Return | มี |
| Annual Volatility | มี |
| Sortino Ratio | ยังไม่มี |

ข้อสังเกต:

- หัวข้อระบุ Sortino Ratio แต่ repo ปัจจุบันยังไม่ได้ implement
- ถ้าต้องการให้ครบ ต้องเพิ่ม downside deviation และ Sortino Ratio ใน `backtest.py`

---

## 12. Test: Walk-Forward Optimization

Walk-Forward Optimization อยู่ใน `portfolio_optimization/backtest.py`

กลไก:

1. ใช้ข้อมูลย้อนหลังเป็น training window
2. optimize portfolio weight
3. ทดสอบกับข้อมูล out-of-sample window ถัดไป
4. รวมผลตอบแทน out-of-sample
5. เปรียบเทียบกับ `SPY`

Requirement:

```text
training window = 4 years
testing / rebalance step = 1 year
```

ค่า default ใน repo:

```python
lookback_window = 252 * 3
step_size = 21 * 3
```

แปลว่า repo ปัจจุบันใช้:

- training window ประมาณ 3 ปี
- rebalance step ประมาณ 3 เดือน

ถ้าต้องการตรงตาม requirement ควรเปลี่ยนเป็น:

```python
lookback_window = 252 * 4
step_size = 252
```

---

## 13. สรุป Gap ระหว่าง Requirement กับ Repo ปัจจุบัน

| หัวข้อ | สถานะ |
|---|---|
| S&P 500 / US stocks | มีบางส่วน และมี SPY benchmark |
| Thai stocks | มี |
| Chinese assets | ยังไม่มี |
| Crypto | ยังไม่มี |
| Train 2015-2024 | ยังไม่ตรง config ปัจจุบัน |
| Test 2025-2026 | ยังไม่ตรง config ปัจจุบัน |
| Missing data handling | มี |
| Log return | มี |
| Ledoit-Wolf covariance | มี |
| Asset selection หลายวิธี | มี |
| PSO / ACO / GWO-style | มี |
| Max Sharpe objective | มี |
| Weight min 0%, max 10% | มี |
| Convergence graph | มี |
| Sharpe / Cum Return / Max DD | มี |
| Sortino Ratio | ยังไม่มี |
| Walk-forward 4 ปี / 1 ปี | ยังไม่ใช่ default |

---

## 14. สรุปสุดท้าย

Repo นี้มีโครงสร้างครบสำหรับงาน portfolio optimization ตั้งแต่ data collection, preprocessing, covariance estimation, asset selection, weight optimization, walk-forward backtest, convergence tracking และ performance report

ส่วนที่ควรปรับเพิ่มเพื่อให้ตรงกับหัวข้อทั้งหมดคือ:

1. เพิ่ม Chinese assets และ Crypto assets ใน `config/assets.csv`
2. ปรับ train/test date ใน `config/settings.json`
3. เพิ่ม Top 10 Sharpe Ratio selector ถ้าต้องการ method แยก
4. เพิ่ม Sortino Ratio ใน `compute_metrics()`
5. ปรับ Walk-Forward Optimization เป็น 4-year lookback และ 1-year step
