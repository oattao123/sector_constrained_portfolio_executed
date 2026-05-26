# สรุป Repo ตามหัวข้อ Portfolio Optimization

Repo นี้เป็นระบบ **Portfolio Optimization Pipeline** สำหรับคัดเลือกสินทรัพย์และหา weight ของพอร์ต โดยใช้ข้อมูลราคา, log return, Ledoit-Wolf covariance, asset selector หลายแบบ และ optimizer แบบ Swarm Intelligence เพื่อ maximize Sharpe Ratio

---

## 1. Data Collection

ข้อมูลสินทรัพย์ถูกกำหนดใน `config/assets.csv`

สถานะใน repo ปัจจุบัน:

| กลุ่มข้อมูล | สถานะ |
|---|---|
| S&P 500 | มีบางหุ้น US และใช้ `SPY` เป็น benchmark |
| Thai | มี เช่น `KTB.BK`, `TISCO.BK`, `TTB.BK`, `TCAP.BK`, `SCB.BK` |
| Chinese | ยังไม่มีใน `assets.csv` |
| ETC / ETF / Bond | มี `TIPS` เป็น bond/ETF-like asset |
| Crypto | ยังไม่มีใน `assets.csv` |

ดังนั้น repo ตอนนี้รองรับหลัก ๆ คือ **Thai stocks + US stocks + TIPS + SPY benchmark** แต่ยังไม่ได้ใส่ Chinese และ Crypto จริง

---

## 2. Train / Test Data

Requirement ที่ระบุ:

```python
train_start = "2015-01-01"
train_end   = "2024-01-01"

test_start  = "2025-01-01"
test_end    = "2026-01-01"
```

แต่ config ปัจจุบันใน repo คือ:

```json
"start_date": "2015-01-01",
"end_date": "2026-06-01",
"insample_end_date": "2022-06-01",
"outsample_end_date": "2026-06-01"
```

แปลว่า repo ปัจจุบันยังไม่ได้ตั้งค่า train/test ให้ตรงกับ requirement ล่าสุด ต้องแก้ `config/settings.json` ถ้าต้องการช่วงตามโจทย์

---

## 3. Data Preprocess

Pipeline มีขั้นตอน preprocessing หลักดังนี้:

- ดาวน์โหลดหรือโหลดข้อมูลราคาจาก cache
- ตรวจ missing data และจัดการข้อมูลที่ไม่สมบูรณ์
- แปลงข้อมูลราคาเป็น return
- ใช้ **log return** สำหรับการคำนวณ portfolio
- ตรวจ outlier เช่น daily return ที่สูงผิดปกติ
- แปลงค่าเงิน USD เป็น THB ผ่าน FX pair เช่น `USDTHB=X`
- คำนวณ **Ledoit-Wolf Shrinkage Covariance** เพื่อให้ covariance matrix เสถียรกว่า sample covariance ปกติ

---

## 4. Modeling: Asset Selection

ขั้นตอน **Asset Selection** คือขั้นตอนคัดเลือกสินทรัพย์จาก universe ทั้งหมดให้เหลือเป็นชุด candidate ก่อนส่งต่อไปยังขั้นตอน **Asset Weight Optimization** เช่น PSO, ACO, EBGWO, CLPSO, APSO หรือ LAPSO

เหตุผลที่ต้องมี asset selection:

- ลดจำนวนสินทรัพย์ก่อน optimize weight ทำให้ optimization เร็วขึ้น
- ลด noise จากสินทรัพย์ที่คุณภาพต่ำหรือมีข้อมูลไม่นิ่ง
- ลดการเลือกสินทรัพย์ที่เคลื่อนไหวคล้ายกันมากเกินไป
- ช่วยให้พอร์ตมี diversification ดีขึ้น
- ทำให้ optimizer โฟกัสกับสินทรัพย์ที่มี risk-adjusted return ดี

ใน repo นี้ asset selection อยู่ในไฟล์:

```text
portfolio_optimization/selectors.py
```

Input หลักของ selector คือ:

| Input | ความหมาย |
|---|---|
| `returns_df` | ตาราง daily returns ของสินทรัพย์ |
| `shrunk_cov` | Ledoit-Wolf shrinkage covariance matrix |
| `corr_matrix` | correlation matrix |
| `all_sector_map` | mapping ระหว่าง ticker กับ sector |
| `risk_free_rate` | risk-free rate สำหรับคำนวณ Sharpe Ratio |
| `top_n` / `num_clusters` / `target_assets` | จำนวนสินทรัพย์ที่ต้องการคัดเลือก |

Output ของ selector คือ `DataFrame` ที่มีรายการสินทรัพย์ที่ถูกเลือก พร้อมข้อมูลประกอบ เช่น sector, annual return, annual volatility, Sharpe Ratio, average correlation และ diversification score

---

### 4.1 Top Sharpe Ratio

หัวข้อ requirement ระบุว่าใช้ **10 Sharpe Ratio** หรือการเลือกสินทรัพย์จาก Sharpe Ratio สูงสุด

ใน repo ปัจจุบันยังไม่มี function แยกชื่อ `select_top_sharpe()` โดยตรง แต่ Sharpe Ratio ถูกใช้เป็นแกนหลักในหลาย selector เช่น:

- `select_by_diversification()`
- `select_by_lw_diversification()`
- `select_by_hrp_sharpe()`
- `select_by_aco()`
- `select_by_aco_cluster()`

แนวคิดของ Top Sharpe Ratio คือคำนวณ:

```text
Sharpe Ratio = (Annual Return - Risk Free Rate) / Annual Volatility
```

แล้วเลือกสินทรัพย์ที่มี Sharpe Ratio สูงที่สุด เช่น top 10 ตัว

ข้อดี:

- เข้าใจง่าย
- คำนวณเร็ว
- เน้นสินทรัพย์ที่ให้ผลตอบแทนต่อความเสี่ยงดีที่สุดในอดีต

ข้อจำกัด:

- อาจเลือกสินทรัพย์ที่อยู่ใน sector เดียวกันมากเกินไป
- ไม่ได้พิจารณา correlation ระหว่างสินทรัพย์
- ถ้าเลือกจาก Sharpe อย่างเดียว พอร์ตอาจกระจุกตัวและ diversification ต่ำ

ดังนั้น repo นี้จึงใช้ Sharpe ร่วมกับ correlation, Ledoit-Wolf covariance, HRP clustering และ ACO เพื่อให้ selection แข็งแรงขึ้น

---

### 4.2 Diversification Score

Function:

```python
select_by_diversification()
```

วิธีนี้คัดเลือกสินทรัพย์จากสองมิติ:

1. สินทรัพย์มี Sharpe Ratio ดีหรือไม่
2. สินทรัพย์มี correlation เฉลี่ยกับสินทรัพย์อื่นต่ำหรือไม่

สูตรหลัก:

```text
Diversification Score = Sharpe Ratio x (1 - Average Correlation)
```

ความหมาย:

- ถ้า Sharpe สูง คะแนนจะสูง
- ถ้า average correlation ต่ำ คะแนนจะสูง
- ถ้าสินทรัพย์มี Sharpe สูงแต่ correlation กับทั้งพอร์ตสูงมาก คะแนนจะถูกลดลง

ขั้นตอนทำงาน:

1. คำนวณ annual return จาก daily returns
2. คำนวณ annual volatility
3. คำนวณ Sharpe Ratio
4. คำนวณ correlation matrix จาก returns
5. หาค่า average correlation ของแต่ละสินทรัพย์เทียบกับ universe
6. กรองสินทรัพย์ที่ Sharpe ต่ำกว่า threshold ถ้ามี candidate เพียงพอ
7. คำนวณ Diversification Score
8. เลือกสินทรัพย์แบบ stratified selection โดยพยายามให้มีอย่างน้อยบางตัวจากแต่ละ sector
9. เติม slot ที่เหลือด้วยสินทรัพย์ที่ score สูงสุด

ข้อดี:

- ง่ายและอธิบายได้ชัดเจน
- พิจารณาทั้ง return/risk และ diversification
- มี sector-aware selection เบื้องต้น

ข้อจำกัด:

- ใช้ sample correlation ปกติ ซึ่งอาจ noisy
- ยังไม่ได้ optimize พอร์ตโดยตรง เป็นแค่การจัดอันดับสินทรัพย์
- ถ้าข้อมูลย้อนหลังผิดปกติ Sharpe อาจหลอกได้

---

### 4.3 Ledoit-Wolf + Diversification Score

Function:

```python
select_by_lw_diversification()
```

วิธีนี้เหมือน Diversification Score แต่เปลี่ยนจากการใช้ sample covariance/correlation ปกติ มาใช้ **Ledoit-Wolf Shrinkage Covariance**

เหตุผลที่ใช้ Ledoit-Wolf:

- covariance matrix ปกติมัก noisy โดยเฉพาะเมื่อจำนวนสินทรัพย์เยอะ
- shrinkage ช่วยดึง covariance ให้เสถียรกว่า
- volatility และ correlation ที่ได้มีความ robust มากขึ้น

สูตรโดยรวม:

```text
LW Sharpe = (LW Annual Return - Risk Free Rate) / LW Annual Volatility
Diversification Score = LW Sharpe x (1 - Average LW Correlation)
```

ขั้นตอนทำงาน:

1. แปลง returns เป็น tensor
2. ใช้ diagonal ของ `shrunk_cov` เพื่อคำนวณ volatility
3. คำนวณ Ledoit-Wolf Sharpe Ratio
4. ใช้ `corr_matrix` ที่มาจาก Ledoit-Wolf covariance
5. คำนวณ average correlation
6. คำนวณ Diversification Score
7. เลือกสินทรัพย์แบบ sector-stratified เหมือนวิธีก่อนหน้า

ข้อดี:

- เสถียรกว่า diversification score แบบ sample covariance
- เหมาะกับข้อมูลการเงินที่ covariance เปลี่ยนแปลงและมี noise
- ใช้ GPU ผ่าน PyTorch ได้เมื่อระบบรองรับ

ข้อจำกัด:

- shrinkage อาจทำให้ความสัมพันธ์เฉพาะช่วงเวลาถูก smooth
- ยังเป็น asset-level ranking ไม่ใช่ portfolio-level optimization เต็มรูปแบบ

---

### 4.4 HRP Clustering + Max Sharpe per Cluster

Function:

```python
select_by_hrp_sharpe()
```

วิธีนี้ใช้แนวคิดจาก **Hierarchical Risk Parity (HRP)** เพื่อแบ่งสินทรัพย์เป็น cluster ตาม correlation แล้วเลือกตัวแทนที่ Sharpe สูงสุดจากแต่ละ cluster

แนวคิดหลัก:

- สินทรัพย์ที่ correlation สูงมักเคลื่อนไหวคล้ายกัน
- ถ้าเลือกหลายตัวจาก cluster เดียวกัน พอร์ตอาจไม่ได้ diversify จริง
- การเลือกตัวแทนจากแต่ละ cluster ช่วยลดการซ้ำซ้อนของ risk exposure

สูตรแปลง correlation เป็น distance:

```text
Distance_ij = sqrt(0.5 x (1 - Corr_ij))
```

ขั้นตอนทำงาน:

1. คำนวณ Ledoit-Wolf Sharpe Ratio ของทุกสินทรัพย์
2. แปลง correlation matrix เป็น distance matrix
3. ใช้ hierarchical clustering ด้วย Ward linkage
4. แบ่ง asset เป็นจำนวน cluster ที่กำหนด
5. ในแต่ละ cluster เลือกสินทรัพย์ที่ Sharpe Ratio สูงสุด
6. สร้างตารางผลลัพธ์และคำนวณ diversification score เพื่อใช้เรียงลำดับ

ข้อดี:

- ลดการเลือกสินทรัพย์ที่คล้ายกันเกินไป
- ได้ตัวแทนจากหลายกลุ่ม correlation
- ยังรักษาคุณภาพด้วยการเลือก Sharpe สูงสุดใน cluster

ข้อจำกัด:

- ผลลัพธ์ขึ้นกับจำนวน cluster
- ถ้า cluster แบ่งไม่ดี selection ก็อาจไม่ดี
- Max Sharpe ใน cluster อาจไม่ได้เป็นตัวที่ diversify ดีที่สุด

---

### 4.5 HRP Clustering + Max Diversification Score per Cluster

Function:

```python
select_by_hrp_div()
```

วิธีนี้คล้าย `select_by_hrp_sharpe()` แต่แทนที่จะเลือก asset ที่ Sharpe สูงสุดในแต่ละ cluster จะเลือก asset ที่มี **Diversification Score สูงสุด**

สูตร:

```text
Diversification Score = LW Sharpe x (1 - Mean Correlation)
```

ขั้นตอนทำงาน:

1. คำนวณ Ledoit-Wolf Sharpe Ratio ของทุกสินทรัพย์
2. คำนวณ mean correlation ของแต่ละสินทรัพย์
3. คำนวณ diversification score
4. สร้าง HRP cluster จาก correlation distance
5. เลือก asset ที่ diversification score สูงสุดในแต่ละ cluster
6. ส่งออก selected assets พร้อม metrics

ข้อดี:

- ได้ทั้ง cluster diversification และ asset-level diversification
- ลดโอกาสเลือกสินทรัพย์ที่ Sharpe สูงแต่ correlation สูงมาก
- เหมาะกับเป้าหมายสร้างพอร์ตที่กระจายตัวกว่า Max Sharpe per Cluster

ข้อจำกัด:

- อาจไม่เลือกสินทรัพย์ที่ผลตอบแทนดีที่สุดใน cluster
- average correlation อาจซ่อนความเสี่ยงแบบ pairwise บางคู่
- ยังเป็น selection stage ไม่ใช่ final portfolio allocation

---

### 4.6 2D-ACO แบบอิสระ

Function:

```python
select_by_aco()
```

วิธีนี้ใช้ **2D Ant Colony Optimization (ACO)** เพื่อค้นหาชุดสินทรัพย์โดยตรง ไม่ได้อาศัยแค่การ sort ranking แบบง่าย

แนวคิด:

- ant แต่ละตัวสร้าง path ของสินทรัพย์
- path คือชุด asset ที่ถูกเลือก
- pheromone บอกว่าสินทรัพย์หรือคู่สินทรัพย์ใดเคยอยู่ใน solution ที่ดี
- heuristic ใช้ Sharpe Ratio เพื่อชี้นำการเลือก

องค์ประกอบหลัก:

| ตัวแปร | ความหมาย |
|---|---|
| `pheromones_start` | pheromone สำหรับการเลือก asset ตัวแรก |
| `pheromones_2d` | pheromone ระหว่าง asset คู่ต่าง ๆ |
| `heuristic` | signal จาก Sharpe Ratio |
| `alpha` | น้ำหนักของ pheromone |
| `beta` | น้ำหนักของ heuristic |
| `evaporation_rate` | อัตราการระเหยของ pheromone |
| `Q` | ขนาด reward สำหรับ path ที่ดี |

probability ในการเลือก asset:

```text
Probability ∝ pheromone^alpha x heuristic^beta
```

fitness ที่ใช้ประเมิน:

```text
Portfolio Sharpe Ratio จาก equal-weight portfolio
```

ขั้นตอนทำงาน:

1. คำนวณ Sharpe ของสินทรัพย์แต่ละตัวจาก Ledoit-Wolf covariance
2. สร้าง heuristic จาก Sharpe
3. ant แต่ละตัวสุ่มเลือก asset ตาม probability
4. ห้ามเลือก asset ซ้ำใน path เดียวกัน
5. สร้าง equal-weight portfolio จาก selected assets
6. คำนวณ portfolio Sharpe
7. เลือก path ที่ดีที่สุดใน iteration
8. update global best ถ้าดีกว่าเดิม
9. pheromone ระเหย
10. reinforce pheromone ให้ path ที่ดีที่สุด

ข้อดี:

- ค้นหาชุดสินทรัพย์แบบ combinatorial ได้ดีกว่าการ sort อย่างเดียว
- พิจารณาความเข้ากันของ asset เป็นชุด
- มี memory ผ่าน pheromone

ข้อจำกัด:

- stochastic ต้องใช้หลายรอบหรือหลาย trials เพื่อความเสถียร
- ใช้เวลามากกว่า rank-based selector
- fitness ใช้ equal weight ซึ่งอาจต่างจาก final weights ที่ optimizer จะหาในขั้นถัดไป

---

### 4.7 2D-ACO + Cluster Constraint

Function:

```python
select_by_aco_cluster()
```

วิธีนี้ผสมระหว่าง **HRP clustering** และ **ACO** โดยบังคับให้ ant เลือกตัวแทนจากแต่ละ cluster

แนวคิด:

- HRP cluster ช่วยจัดกลุ่มสินทรัพย์ที่ correlation คล้ายกัน
- ACO ช่วยค้นหาว่าควรเลือกตัวแทนตัวไหนจากแต่ละ cluster
- constraint นี้ช่วยลดการเลือก asset หลายตัวจากกลุ่มความเสี่ยงเดียวกัน

ขั้นตอนทำงาน:

1. คำนวณ Sharpe Ratio ของทุกสินทรัพย์
2. สร้าง heuristic จาก Sharpe
3. แปลง correlation matrix เป็น distance matrix
4. ใช้ hierarchical clustering เพื่อสร้าง cluster
5. ant เลือก asset หนึ่งตัวจาก cluster แรก
6. ant เลือก asset จาก cluster ถัดไปโดยอาศัย pheromone ระหว่าง asset และ heuristic
7. ทำจนครบทุก cluster
8. สร้าง equal-weight portfolio จาก path ที่เลือก
9. คำนวณ portfolio Sharpe
10. update pheromone ตาม path ที่ดีที่สุด

ข้อดี:

- คุม diversification ผ่าน cluster constraint ชัดเจน
- ยังใช้ความสามารถของ ACO ในการค้นหาชุดสินทรัพย์
- ลดโอกาสเกิดพอร์ตที่กระจุกในสินทรัพย์ correlation สูง

ข้อจำกัด:

- ถ้าจำนวน cluster มากเกินไป จำนวน selected assets อาจมากเกินจำเป็น
- ผลลัพธ์ขึ้นกับคุณภาพของ clustering
- ยังมี stochastic behavior จึงควรรันหลาย trials

---

### 4.8 เปรียบเทียบ Asset Selection Methods

| Selector | ใช้ Sharpe | ใช้ Correlation | ใช้ Ledoit-Wolf | ใช้ Clustering | ใช้ ACO | จุดเด่น |
|---|---:|---:|---:|---:|---:|---|
| Top Sharpe Ratio | ใช่ | ไม่ | ไม่จำเป็น | ไม่ | ไม่ | ง่ายและเร็ว |
| Diversification Score | ใช่ | ใช่ | ไม่ | ไม่ | ไม่ | balance Sharpe กับ diversification |
| Ledoit-Wolf + Diversification | ใช่ | ใช่ | ใช่ | ไม่ | ไม่ | covariance เสถียรกว่า |
| HRP + Max Sharpe | ใช่ | ใช่ | ใช่ | ใช่ | ไม่ | เลือกตัวแทน Sharpe สูงจากแต่ละ cluster |
| HRP + Max Diversification | ใช่ | ใช่ | ใช่ | ใช่ | ไม่ | กระจาย risk exposure ดีขึ้น |
| 2D-ACO | ใช่ | ใช่ | ใช่ | ไม่ | ใช่ | search asset combination โดยตรง |
| 2D-ACO + Cluster Constraint | ใช่ | ใช่ | ใช่ | ใช่ | ใช่ | search พร้อมบังคับ diversification ตาม cluster |

สรุปคือ asset selection ใน repo นี้ไม่ได้เลือกสินทรัพย์จากผลตอบแทนอย่างเดียว แต่พยายามรวม 3 แนวคิดเข้าด้วยกัน:

1. **Performance quality** ผ่าน Sharpe Ratio
2. **Risk stability** ผ่าน Ledoit-Wolf covariance
3. **Diversification** ผ่าน correlation, sector และ clustering

---

## 5. Asset Weight Optimization

ใน `portfolio_optimization/optimizers.py` มี optimizer สำหรับหา weight หลายแบบ เช่น:

| Optimizer | บทบาท |
|---|---|
| PSO | Particle Swarm Optimization สำหรับหา portfolio weights |
| ACO / ACO_R | Ant Colony Optimization สำหรับ continuous weight หรือ selection + weight |
| EBGWO / GWO-style | Enhanced Binary Grey Wolf Optimizer |
| CLPSO | Comprehensive Learning PSO |
| APSO | Adaptive PSO |
| LAPSO | Landscape Adaptive PSO |
| CIAC | Continuous Interacting Ant Colony |

ทุกวิธีใช้แนวคิดเดียวกันคือหา weight ที่ให้ผลตอบแทนต่อความเสี่ยงดีที่สุด

---

## 6. Objective Function

Objective หลักคือ:

```text
maximize Sharpe Ratio
```

โดย fitness ในหลาย optimizer มีรูปแบบประมาณ:

```text
fitness = Sharpe Ratio + Entropy Bonus - Weight Penalty
```

Ledoit-Wolf covariance ถูกใช้ในการคำนวณ volatility ของพอร์ต

---

## 7. Constraints

Requirement ที่ระบุ:

```text
min weight = 0%
max weight = 10%
```

ใน repo ปัจจุบัน:

- min weight = `0.0`
- max weight = `0.1` หรือ 10%
- weight ถูก normalize ให้ผลรวมเท่ากับ 1
- ถ้า weight เกิน 10% จะโดน penalty
- ไม่มี short sell เพราะ weight ถูก clamp ให้อยู่ระหว่าง `0` ถึง `1`

ดังนั้น constraint นี้ตรงกับ requirement โดยรวม

---

## 8. Representation and Encoding

| ส่วน | Encoding |
|---|---|
| ACO asset selection | ใช้ integer index ของ asset |
| PSO / GWO / DE-like / SI optimizers | ใช้ float vector เป็น portfolio weights |
| Weight vector | normalize ให้ sum = 1 |
| ACO cluster | เลือก integer asset index จากแต่ละ cluster |

---

## 9. Algorithm Mechanism

### PSO

- แต่ละ particle คือ weight vector
- อัปเดตจาก personal best และ global best
- ใช้ inertia weight ลดลงเพื่อเปลี่ยนจาก exploration ไป exploitation

### ACO

- ใช้ pheromone และ heuristic
- heuristic มักอ้างอิง Sharpe
- ant สร้าง candidate portfolio
- portfolio ที่ fitness ดีจะเพิ่ม pheromone

### GWO / EBGWO

- ใช้ alpha, beta, delta wolves เป็นผู้นำ
- ปรับตำแหน่ง weight ตามผู้นำ
- มี exploration mechanism เพื่อลดการติด local optimum

### ACO + EBGWO

- ACO เลือก asset
- EBGWO ปรับ weight
- มี stagnation handling ถ้า fitness ไม่ดีขึ้นนานเกินไป

---

## 10. Convergence and Stability

Repo มีการเก็บ convergence curve:

- `Avg_Best_Convergence`
- `Avg_Avg_Convergence`

และสร้างกราฟ:

```text
convergence.png
```

Convergence handling ที่มี:

- เก็บ best fitness ทุก iteration
- เก็บ average fitness ทุก iteration
- ใช้ evaporation ใน ACO
- ใช้ pheromone clamp
- ใช้ stagnation counter และ reset บางส่วนใน ACO + EBGWO
- ใช้หลาย trials เพื่อลดผลจาก randomness

---

## 11. Metrics

ใน `backtest.py` metric ที่คำนวณจริงคือ:

| Metric | สถานะ |
|---|---|
| Sharpe Ratio | มี |
| Cum Return | มี |
| Max Drawdown | มี |
| Annual Return | มี |
| Annual Volatility | มี |
| Sortino Ratio | ยังไม่มีใน `compute_metrics()` |

ดังนั้นถ้าต้องการตามหัวข้อเต็ม ต้องเพิ่ม Sortino Ratio เข้าไปใน `compute_metrics()`

---

## 12. Test: Walk-Forward Optimization

Repo ใช้ Walk-Forward Optimization ใน `WalkForwardBacktester`

กลไก:

- ใช้ rolling training window
- optimize weight ในช่วง train
- test ใน out-of-sample window ถัดไป
- รวมผลลัพธ์เป็น OOS returns
- เปรียบเทียบกับ `SPY`

Requirement ที่ระบุ:

```text
lookback = 4 years
test step = 1 year
```

แต่ค่า default ใน repo ตอนนี้คือ:

```python
lookback_window = 252 * 3
step_size = 21 * 3
```

แปลว่า default ปัจจุบันประมาณ:

- train window = 3 ปี
- rebalance/test step = ประมาณ 3 เดือน

ถ้าต้องการให้ตรง requirement ต้องเปลี่ยนเป็น:

```python
lookback_window = 252 * 4
step_size = 252
```

---

## 13. สรุปภาพรวม

Repo นี้ทำระบบ portfolio optimization ได้ครบหลายส่วนแล้ว ได้แก่ data loading, preprocessing, Ledoit-Wolf covariance, asset selection 6 วิธี, weight optimization หลาย algorithm, walk-forward test, transaction cost, convergence graph และ performance report

ส่วนที่ยังไม่ตรงกับ requirement ล่าสุดคือ:

- ยังไม่มี Chinese assets
- ยังไม่มี Crypto assets
- ETC/ETF มีเพียง `TIPS`
- วันที่ train/test ใน config ปัจจุบันยังไม่ตรง
- Walk-forward ยังเป็น 3 ปี / 3 เดือน ไม่ใช่ 4 ปี / 1 ปี
- ยังไม่มี Sortino Ratio ใน metric
- ไม่มี selector “Top 10 Sharpe Ratio” แยกเป็น method โดยตรง
