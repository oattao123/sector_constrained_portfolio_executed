# คู่มือผู้ใช้งาน — ระบบจำลองการจัดการพอร์ตโฟลิโอแบบออปติไมซ์ (Portfolio Optimization Pipeline)

ระบบท่อส่งข้อมูลการจัดการพอร์ตแบบโมดูลาร์สำหรับการใช้งานจริง ซึ่งรันการประมาณค่าความแปรปรวนร่วมด้วยวิธี Ledoit-Wolf, การวิเคราะห์ความเสี่ยงแบบลำดับขั้น (Hierarchical Risk Parity: HRP), การเลือกสินทรัพย์ด้วยฝูงมดแบบสองมิติ (2D-ACO) และการจัดสัดส่วนการลงทุนด้วยฝูงหมาป่าสีเทาแบบไบนารีที่ปรับปรุงแล้ว (EBGWO) สำหรับจำลองการเดินหน้าหาผลตอบแทนพอร์ตโฟลิโอนอกกลุ่มตัวอย่าง (Walk-Forward Portfolio Optimization)

---

## 1. สิ่งที่ต้องมีก่อนการใช้งาน (Prerequisites)

ติดตั้งแพ็กเกจที่จำเป็นผ่าน `uv`:

```bash
uv sync
```

แพ็กเกจหลักที่จำเป็น: `torch`, `yfinance`, `pandas`, `numpy`, `scipy`, `rich`, `matplotlib`, `scikit-learn`

ระบบจะตรวจจับและเรียกใช้งาน GPU (CUDA) ให้โดยอัตโนมัติหากมีระบบติดตั้งอยู่; หากไม่มีจะสลับไปใช้งาน CPU แทนโดยอัติโนมัติ

---

## 2. การรันระบบเบื้องต้น (Basic Execution)

```bash
uv run python run_pipeline.py
```

### 2.5 การแยกสคริปต์สำหรับแต่ละออปติไมเซอร์ (Separate Optimizer Pipelines)

หากต้องการรันการจำลอง walk-forward OOS โดยเน้นใช้เฉพาะออปติไมเซอร์ใดออปติไมเซอร์หนึ่งโดยเฉพาะ สามารถเรียกใช้งานสคริปต์แยกต่อไปนี้:

**1. รันเฉพาะตัวออปติไมเซอร์ ACO + EBGWO:**
```bash
uv run python run_pipeline_aco_ebgwo.py
```

**2. รันเฉพาะตัวออปติไมเซอร์ PSO:**
```bash
uv run python run_pipeline_pso.py
```

**3. รันเฉพาะตัวออปติไมเซอร์ CLPSO:**
```bash
uv run python run_pipeline_clpso.py
```

**4. รันเฉพาะตัวออปติไมเซอร์ APSO:**
```bash
uv run python run_pipeline_apso.py
```

**5. รันเฉพาะตัวออปติไมเซอร์ LAPSO:**
```bash
uv run python run_pipeline_lapso.py
```

**6. รันเฉพาะตัวออปติไมเซอร์ ACOR:**
```bash
uv run python run_pipeline_acor.py
```

**7. รันเฉพาะตัวออปติไมเซอร์ CIAC:**
```bash
uv run python run_pipeline_ciac.py
```

---

## 3. พารามิเตอร์ของคำสั่ง (Command Line Arguments)

| พารามิเตอร์ (Parameter) | ประเภท | ค่าเริ่มต้น (Default) | คำอธิบาย (Description) |
| :--- | :---: | :---: | :--- |
| `--settings` | `str` | `config/settings.json` | พาธไปยังไฟล์คอนฟิกูเรชัน JSON |
| `--assets` | `str` | `config/assets.csv` | พาธไปยังไฟล์รายการและกลุ่มสินทรัพย์หลักทรัพย์เป้าหมาย |
| `--cache` | `str` | `data/all_data.csv` | ชื่อไฟล์แคชสำหรับจัดเก็บข้อมูลราคาที่ดาวน์โหลดมา |
| `--wolves` | `int` | `500` | จำนวนประชากรหมาป่าสำหรับตัวออปติไมเซอร์ EBGWO |
| `--iterations` | `int` | `1000` | จำนวนรอบของการออปติไมซ์ (EBGWO, ACO, PSO, CLPSO, APSO, LAPSO, ACOR และ CIAC) |
| `--agents` | `int` | `500` | จำนวนมดสำหรับการเลือกสินทรัพย์ตัวแบบ ACO |
| `--particles` | `int` | `500` | จำนวนอนุภาค/ประชากร/มดต่อเนื่องสำหรับออปติไมเซอร์ PSO, CLPSO, APSO, LAPSO, ACOR และ CIAC |
| `--trials` | `int` | `5` | จำนวนการรันซ้ำต่อหนึ่งกลยุทธ์เพื่อหาค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน (แสดงค่าเป็น Mean +- Std) |

### ตัวอย่างการทดสอบด่วน (Fast Test Run)
```bash
uv run python run_pipeline.py --wolves 5 --iterations 3 --agents 5 --trials 2
```

### การรันจำลองการผลิตจริงเต็มรูปแบบ (Full Production Run)
```bash
uv run python run_pipeline.py --wolves 500 --iterations 1000 --agents 500 --trials 5
```

---

## 4. คอนฟิกูเรชัน: `config/settings.json`

```json
{
  "base_currency": "THB",
  "data": {
    "start_date":          "2015-01-01",
    "end_date":            "2026-03-20",
    "insample_end_date":   "2024-01-01",
    "outsample_end_date":  "2026-03-20"
  },
  "risk_free_rate": {
    "ticker":   "^TNX",
    "fallback": 0.045
  },
  "portfolio": {
    "value": 1000000,
    "transaction_cost_rate": 0.001
  },
  "currency":  { "fx_pairs": { "USD": "USDTHB=X" } },
  "quality_checks": {
    "max_abs_daily_return": 0.25,
    "max_annual_return":    1.5
  }
}
```

### คอนฟิกูเรชันพอร์ตโฟลิโอและต้นทุนการทำธุรกรรม (Portfolio & Transaction Cost Configuration)

| คีย์ (Key) | ประเภท | จุดประสงค์ |
| :--- | :---: | :--- |
| `value` | `float` | มูลค่าเริ่มต้นของพอร์ตโฟลิโอสำหรับการจัดสรรสัดส่วนสกุลเงินเป้าหมาย |
| `transaction_cost_rate` | `float` | อัตราต้นทุนการทำธุรกรรม (Transaction Cost Rate) ที่จะถูกเรียกเก็บจากการปรับสัดส่วนน้ำหนักการลงทุน (เช่น `0.001` หมายถึง 0.1% หรือ 10 bps) |

### คอนฟิกูเรชันเกี่ยวกับวันที่ (Date Configuration)

| คีย์ (Key) | จุดประสงค์ |
| :--- | :--- |
| `start_date` | วันที่เริ่มต้นสำหรับการเริ่มดาวน์โหลดข้อมูลราคาย้อนหลัง |
| `end_date` | วันที่สิ้นสุดสำหรับการสิ้นสุดการดาวน์โหลดข้อมูลราคาย้อนหลัง |
| `insample_end_date` | จุดแบ่งวันสำหรับการประมาณค่าความแปรปรวนร่วมและตัวเลือกสินทรัพย์ ข้อมูล**ก่อน**วันนี้จะนำไปใช้ในการฟิตโมเดล (In-Sample Training) |
| `outsample_end_date` | วันที่สิ้นสุดของการประเมินเพื่อวัดผลนอกกลุ่มตัวอย่าง (Out-of-Sample) |

> **ตัวอย่าง**: `insample_end_date = 2024-01-01` หมายความว่าการเลือกหลักทรัพย์ทั้ง 6 ตัว และการประเมินค่า Ledoit-Wolf covariance จะคำนวณจากข้อมูลตั้งแต่ `start_date` จนถึง `2024-01-01` เท่านั้น จากนั้นพอร์ตโฟลิโอจะประเมินแบบเดินหน้าย้อนหลังจำลองเชิงปฏิบัติการ (Walk-forward backtester) ตั้งแต่ `2024-01-01` ไปจนถึง `outsample_end_date` เพื่อทดสอบผลตอบแทนจริงนอกกลุ่มตัวอย่าง

---

## 4.5 แบบจำลองต้นทุนการทำธุรกรรม (Transaction Cost modeling)
ต้นทุนการทำธุรกรรมจะถูกคำนวณในทุกๆ ขั้นตอนของการหมุนหน้าต่างทดสอบ OOS (Walk-Forward Step) โดยคิดจากอัตราการปรับเปลี่ยนพอร์ตโฟลิโอแบบสองทาง (Two-Way Turnover):
$$\text{Turnover} = \sum_i |w_{\text{new}, i} - w_{\text{prev}, i}|$$
$$\text{Cost} = \text{Turnover} \times \text{Transaction Cost Rate}$$

ต้นทุนนี้จะถูกหักออกเชิงเรขาคณิต (Geometrically Deducted) จากผลตอบแทนของวันนอกกลุ่มตัวอย่างวันแรกในหน้าต่างจำลองรอบใหม่:
$$R_{\text{adj}, 0} = (1 + R_0) \times (1 - \text{Cost}) - 1$$

---

## 5. ผลลัพธ์จากการรันระบบ (Pipeline Outputs)

ทุกการรันจะสร้างโฟลเดอร์ระบุเวลาขึ้นใน `output/run_{DD_MM_HH_MM}/` ซึ่งประกอบไปด้วย:

| ไฟล์ | คำอธิบาย |
| :--- | :--- |
| `pipeline.log` | ประวัติล็อกการรันระบบโดยละเอียดพร้อมระบุประทับเวลา |
| `convergence.png` | กราฟแนวโน้มแสดงการลู่เข้าของความฟิตเฉลี่ยสำหรับออปติไมเซอร์ EBGWO |
| `performance.png` | กราฟผลตอบแทนสะสมเปรียบเทียบในทุกกลยุทธ์ (ตลอดช่วงประเมินทั้งหมด) |
| `performance_2025.png` | กราฟผลตอบแทนสะสมเปรียบเทียบเฉพาะในช่วงปี ค.ศ. 2025 |
| `performance_report.md` | ตารางรายงานประสิทธิภาพการทดสอบ OOS เปรียบเทียบแบบข้างเคียงระหว่าง **With Cost** และ **No Cost** โดยแสดงค่าเป็น `Mean +- Standard Deviation` จากการทดสอบซ้ำหลายรอบ |
| `candles/` | โฟลเดอร์ย่อยเก็บภาพแท่งเทียนประเมินแนวโน้มและปริมาณการเทรดแยกตามแต่ละกลยุทธ์การเลือกสินทรัพย์ (6 แฟ้มภาพ) |
| `selections/` | โฟลเดอร์ย่อยเก็บไฟล์ CSV รายการสินทรัพย์ที่ผ่านการคัดเลือกแยกตามแต่ละกลยุทธ์การเลือกสินทรัพย์ (เช่น `aco_cluster_selected.csv` เป็นต้น) |

หมายเหตุ: การรันแยกตามออปติไมเซอร์เฉพาะตัว (`run_pipeline_aco_ebgwo.py` หรือ `run_pipeline_pso.py`) จะบันทึกข้อมูลผลลัพธ์ใน `output/aco_ebgwo_{DD_MM_HH_MM}/` หรือ `output/pso_{DD_MM_HH_MM}/` ตามลำดับ โดยมีโครงสร้างไฟล์ผลลัพธ์ที่เหมือนกันทุกประการ

---

## 5.5 ระบบจูนพารามิเตอร์และวิเคราะห์ความอ่อนไหว (Parameter Tuning & Sensitivity Analysis Pipeline)

ระบบมีสคริปต์ `tune_pipeline.py` เพื่อใช้จำลองการค้นหาแบบกริด (Grid Search) และวิเคราะห์ความอ่อนไหวของพารามิเตอร์ต่อผลลัพธ์นอกกลุ่มตัวอย่าง (OOS)

### พารามิเตอร์ที่รองรับการจูน (Tunable Parameters)

- **พารามิเตอร์ EBGWO**: `ST` (เกณฑ์การสำรวจ/Exploration threshold), `eps` (ค่า regularizer ของเอนโทรปี), `wolves` (จำนวนประชากรหมาป่า), `iterations` (จำนวนรอบการทำงาน)
- **พารามิเตอร์ ACO**: `Q` (ความเข้มข้นของฟีโรโมน), `evaporation_rate` (อัตราการระเหยของฟีโรโมน), `alpha_aco`, `beta_aco`, `patience`, `agents` (จำนวนมด)
- **พารามิเตอร์สำหรับวัตถุประสงค์พอร์ตโฟลิโอ**: `lambda_ent` (สัมประสิทธิ์เอนโทรปี regularizer), `max_weight` (ขีดจำกัดสัดส่วนการลงทุนสูงสุดของแต่ละหลักทรัพย์)

### พารามิเตอร์ของคำสั่ง (Command-Line Arguments)

| อาร์กิวเมนต์ | ประเภท | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `--settings` | `str` | `"config/settings.json"` | พาธไปยังไฟล์คอนฟิก JSON |
| `--assets` | `str` | `"config/assets.csv"` | พาธไปยังไฟล์นิยามรายการหลักทรัพย์เป้าหมาย |
| `--cache` | `str` | `"data/all_data.csv"` | พาธจัดเก็บไฟล์ราคารายวันย้อนหลัง |
| `--wolves` | `int` | `100` | จำนวนหมาป่า EBGWO เริ่มต้นสำหรับการค้นหาพารามิเตอร์ |
| `--iterations` | `int` | `150` | จำนวนรอบการปรับปรุงสัดส่วนการลงทุน |
| `--agents` | `int` | `100` | จำนวนมดสำหรับตัวจำลองการเลือกสินทรัพย์ ACO |
| `--trials` | `int` | `10` | จำนวนรอบทดสอบการรันจำลองของการตั้งค่าหนึ่งๆ เพื่อนำมาคำนวณค่าเฉลี่ยแบบสุ่ม |
| `--param` | `str` | `"ST"` | ชื่อพารามิเตอร์หลักที่ต้องการทำการจูน |
| `--values` | `str` | `"0.1,0.2,0.3,0.4,0.5"` | ค่าต่างๆ ของพารามิเตอร์หลักคั่นด้วยเครื่องหมายจุลภาค (Comma) |
| `--param2` | `str` | `None` | พารามิเตอร์ตัวรองเพื่อจูนแบบ 2 มิติร่วมกัน |
| `--values2` | `str` | `None` | ค่าต่างๆ ของพารามิเตอร์ตัวรองคั่นด้วยเครื่องหมายจุลภาค |
| `--strategy` | `str` | `"ACO Cluster Selected"` | กลยุทธ์การเลือกสินทรัพย์หลักที่ต้องการใช้รันการจูนและประเมินผล |

### ตัวอย่างการใช้งาน

**1. วิเคราะห์ความอ่อนไหวแบบ 1 มิติ (ผลลัพธ์เป็นกราฟเส้นแสดงทิศทางผลตอบแทนและ Sharpe)**
```bash
uv run python tune_pipeline.py --param ST --values 0.1,0.2,0.3,0.4,0.5
```

**2. ค้นหาความสัมพันธ์ร่วมกันแบบ 2 มิติ (ผลลัพธ์เป็นแผนภูมิความร้อน Heatmap)**
```bash
uv run python tune_pipeline.py --param ST --values 0.2,0.3,0.4 --param2 lambda_ent --values2 0.01,0.05,0.1
```

### ผลลัพธ์จากการรัน

ผลการทำงานจะจัดเก็บลงในโฟลเดอร์ระบุเวลาเฉพาะใน `output/tuning_{DD_MM_HH_MM}/`:
1. `tuning.log`: ล็อกแสดงขั้นตอนการจำลองโดยละเอียด
2. `tuning_report.md`: รายงานผลการทดสอบ เรียงลำดับตาม Sharpe Ratio ที่ดีที่สุด ผลลัพธ์แต่ละฟิลด์จะรายงานในโครงสร้างแสดงค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน `mean +- std` (เช่น `20.20% +- 1.15%` / `0.6499 +- 0.0315`) ตลอดจานวนครั้งของการทดสอบที่กำหนดผ่าน `--trials` เพื่อวัดความทนทานและความน่าเชื่อถือของกลยุทธ์
3. `sensitivity_analysis.png`:
   - รูปภาพกราฟแกนคู่ (Dual-axis line plot) แสดงสถิติกำไรและ Sharpe สำหรับการวิเคราะห์แบบ 1 มิติ
   - แผนภูมิความร้อน (2D Heatmaps) แสดง Sharpe และผลตอบแทนสะสมของการจับคู่ตัวแปรแบบ 2 มิติ

---

## 6. เอกสารโค้ดเชิงลึก: แพ็กเกจ `portfolio_optimization`

---

### `data_manager.py` — `DataManager`

**คลาส: `DataManager(settings_path, assets_path)`**

รับผิดชอบการดึงราคาย้อนหลังย้อนหลัง, ล้างความผิดปกติของข้อมูล และการแปลงสกุลเงินต่างประเทศ

| เมธอด (Method) | โครงสร้าง (Signature) | คำอธิบาย (Description) |
| :--- | :--- | :--- |
| `load_config` | `()` | โหลดและพาร์สข้อมูลคอนฟิกพร้อมจับคู่สินทรัพย์ Sectors และสกุลเงินสำหรับรันกระบวนการทั้งหมด |
| `download_data` | `(start_date, end_date, cache_path, use_cache)` | ดาวน์โหลดราคาสินสุดวัน (Close) จาก Yahoo Finance ทีละ 50 คอลัมน์ หากข้อมูลในไฟล์แคชตรงกันและครบถ้วนจะทำการอ่านข้อมูลจากแคชโดยอัติโนมัติ พร้อมผูกดัชนี SPY สำหรับใช้เปรียบเทียบในภายหลัง |
| `clean_data` | `(df, max_gap_days=365)` | จัดการและตัดส่วนหลักทรัพย์ที่ไม่มีราคาย้อนหลังขาดหายยาวติดต่อกันนานกว่า `max_gap_days` วัน จากนั้นทำ Forward-fills และ Backward-fills สำหรับส่วนที่เหลือ |
| `convert_to_base_currency` | `(df, start_date, end_date)` | ดึงอัตราแลกเปลี่ยน `USDTHB=X` และแปลงราคาสินทรัพย์สกุลดอลลาร์สหรัฐให้เป็นเงินบาทไทย (THB) หากระบบล้มเหลวจะคำนวณด้วยค่าคงที่สำรองที่ `35.0` บาทต่อดอลลาร์ |
| `get_log_returns` | `(df)` | คำนวณหาผลตอบแทนแบบล็อกรายวัน `log(P_t / P_{t-1})` และทำความสะอาดแถวแรกซึ่งจะเป็น NaN |

---

### `covariance.py` — การประมาณค่าตัวแปรความแปรปรวนร่วมที่มีความทนทาน (Robust Covariance)

**`get_best_device() → torch.device`**
ตรวจสอบการรองรับความเร่งบน GPU ของ CUDA หากรองรับจะส่งกลับเป็นอุปกรณ์ `cuda` หากไม่พบลำดับการประมวลผลจะสลับไปประมวลผลบนหน่วยประมวลผลกลาง `cpu` โดยธรรมชาติ

---

**`to_tensor(data, device, dtype) → torch.Tensor`**
ช่วยจัดรูปแบบโครงสร้างข้อมูลต้นน้ำหลากหลายประเภท เช่น `np.ndarray`, `pd.DataFrame` หรือ `list` ให้เป็นเทนเซอร์ PyTorch และย้ายข้อมูลไปยังอุปกรณ์ประมวลผลที่เหมาะสม

---

**`ledoit_wolf_covariance_gpu_dynamic(X) → (shrunk_cov, sample_cov, delta)`**

หาค่า Ledoit-Wolf Shrinkage Covariance โดยตรงบน GPU ด้วยเทคโนโลยีของ PyTorch

| พารามิเตอร์นำเข้า | ประเภทข้อมูล | คำอธิบายรายละเอียด |
| :--- | :--- | :--- |
| `X` | `Tensor (n_samples, n_features)` | เมทริกซ์อัตราผลตอบแทนแบบลอการิทึมรายวันของหลักทรัพย์ทั้งหมด |

**ผลลัพธ์ส่งออก:**

| ข้อมูลที่ได้รับคืน | คำอธิบาย |
| :--- | :--- |
| `shrunk_cov` | `(n_features, n_features)` — ข้อมูล covariance หลังการลดทอนความปั่นป่วนด้วยสูตร `(1-δ)·S + δ·T` |
| `sample_cov` | `(n_features, n_features)` — ข้อมูล sample covariance รายวันดั้งเดิม |
| `delta` | `float` — สัดส่วนความหนาแน่นและน้ำหนักของการหดตัวโมเดลที่ดีที่สุด `δ ∈ [0, 1]` |

**ขั้นตอนวิธี:** ประยุกต์วิธีเชิงลึก Oracle Approximating Shrinkage (OAS) ซึ่งคำนวณหาค่า `δ` ทางสถิติจากอัตราส่วน `b²/d²` เพื่อลดสัญญาณรบกวนของตัวแปรและป้องกันปัญหา Overfitting

---

**`compute_correlation_matrix(cov_matrix) → torch.Tensor`**

แปลงค่า covariance เมทริกซ์ให้เป็นเมทริกซ์ค่าความสัมพันธ์ (Correlation Matrix) โดยการหารด้วยผลคูณของส่วนเบี่ยงเบนมาตรฐาน (Standard Deviations) พร้อมครอบข้อมูลให้อยู่ในขอบข่ายเพื่อไม่ให้อุปกรณ์มีปัญหาจากการปัดเศษทางทศนิยม

---

### `selectors.py` — กลยุทธ์การคัดกรองสินทรัพย์ (Asset Selection Strategies)

ทุกตัวเลือกสินทรัพย์จะคืนตารางข้อมูลสรุปประเภท `pd.DataFrame` ซึ่งจัดเรียงจากคะแนนความหลากหลาย (Diversification Score) จากสูงไปต่ำ และบรรจุข้อมูลสรุป: `Ticker`, `Sector`, `Ann Return (%)`, `Ann Vol (%)`, `Sharpe`, `Avg Corr`, `Diversification Score`

---

**`select_by_diversification(returns_df, all_sector_map, risk_free_rate, top_n, min_per_sector, min_sharpe_threshold) → DataFrame`**

**กลยุทธ์ที่ 1 — การเลือกด้วยความหลากหลายรูปแบบดั้งเดิม (Standard Diversification Selection)**

หาค่าผลตอบแทนเฉลี่ย ความแปรปรวน และ Sharpe ratio ในรูปแบบปกติ จากนั้นประเมินคะแนนตัวเลือกจากสูตร `Sharpe × (1 − avg_corr)` กรองสัดส่วนสินทรัพย์ที่มีผลตอบแทนเป็นลบหรือต่ำกว่าความเสี่ยง รวมถึงบังคับการคัดเลือกสัดส่วนจากแต่ละอุตสาหกรรม (Sectors) เพื่อให้มีความกระจายตัวเพียงพอในระดับกว้าง

---

**`select_by_lw_diversification(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, top_n, min_per_sector, min_sharpe_threshold) → DataFrame`**

**กลยุทธ์ที่ 2 — การเลือกด้วยความหลากหลายแบบ Ledoit-Wolf (LW Diversification Selection)**

ทำงานเช่นเดียวกับกลยุทธ์ที่ 1 แต่จะเปลี่ยนไปใช้งานประมวลผลจาก covariance และ correlation ที่ประเมินด้วย Ledoit-Wolf เพื่อเพิ่มความทนทานต่อข้อมูลที่คลาดเคลื่อนหรือกรณีศึกษาของกลุ่มหลักทรัพย์จำกัด

---

**`select_by_hrp_sharpe(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters) → DataFrame`**

**กลยุทธ์ที่ 3 — กลยุทธ์ประเมินความเสี่ยงและพฤติกรรมความสัมพันธ์เป็นกลุ่ม (HRP + Max Sharpe Cluster Selection)**

สร้างโครงสร้างความสัมพันธ์ความเสี่ยงแบบลำดับขั้น (Ward Linkage จากระยะห่าง LW) จากนั้นจัดกลุ่มสินทรัพย์ออกเป็น `num_clusters` กลุ่มย่อย และทำการดึงเฉพาะสินทรัพย์ที่มีค่า **Sharpe Ratio ดีที่สุด**ในแต่ละกลุ่มขึ้นมาเป็นตัวแทนสัดส่วนการลงทุน

---

**`select_by_hrp_div(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters) → DataFrame`**

**กลยุทธ์ที่ 4 — กลยุทธ์ความหลากหลายเป็นกลุ่ม (HRP + Max Diversification Cluster Selection)**

ประยุกต์กลุ่มสินทรัพย์แบบเดียวกับกลยุทธ์ที่ 3 แต่จะเลือกสินทรัพย์ที่มี**คะแนนความหลากหลายดีที่สุด (Max Diversification Score)** ในแต่ละกลุ่มเข้ามาทดแทน

---

**`select_by_aco(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, target_assets, num_ants, num_iterations) → DataFrame`**

**กลยุทธ์ที่ 5 — กลยุทธ์เดินขบวนฝูงมดแบบตรง (2D-ACO Direct Selection)**

ใช้ขั้นตอนวิธี Ant Colony Optimization แบบสองมิติเพื่อค้นหาตัวแทนพอร์ตที่ดีที่สุด ตัวมดในแต่ละรอบจะประเมินความร่วมมือในการเลือกสินทรัพย์ร่วมกัน เพื่อสร้างระดับฟีโรโมนบนเส้นทางคู่ความสัมพันธ์ที่มีค่าประสิทธิภาพ Sharpe พอร์ตโดยรวมโดดเด่น

---

**`select_by_aco_cluster(returns_df, shrunk_cov, corr_matrix, all_sector_map, risk_free_rate, num_clusters, num_ants, num_epochs) → DataFrame`**

**กลยุทธ์ที่ 6 — กลยุทธ์ผสมมดและกลุ่มสินทรัพย์ (2D-ACO Cluster Selection)**

ผสานกระบวนการคัดกรองระหว่าง HRP Clustering และ 2D-ACO เข้าด้วยกัน สินทรัพย์จะโดนจัดกลุ่มก่อน จากนั้นตัวมดจะทำการเรียนรู้พฤติกรรมการดึงข้อมูลสินทรัพย์จากตัวแทนกลุ่มหลักทรัพย์ต่างๆ เพื่อสร้างการจัดสรรหลักทรัพย์โดยพิจารณาความสัมพันธ์ระหว่างกลุ่ม (Inter-cluster synergies)

---

### `optimizers.py` — การจัดสรรน้ำหนัก (Weight Allocation)

**`optimize_weights_ebgwo_monte_carlo_entropy(train_returns_gpu, max_weight, lambda_ent, num_wolves, iterations) → np.ndarray`**

จัดหาสัดส่วนการลงทุนด้วยโมเดล Enhanced Binary Grey Wolf Optimizer (EBGWO) ร่วมกับแบบจำลองมอนเตการ์โลและการบวกลดหย่อนด้วยเอนโทรปี (Entropy Regularization) เพื่อป้องกันความกระจุกตัวในกลุ่มสินทรัพย์น้อยตัว

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ (ค่าสูงจะเอื้อต่อการกระจายสัดส่วนเท่าๆ กัน) |
| `num_wolves` | `500` | จำนวนประชากรของกลุ่มหมาป่าในการปรับปรุง |
| `iterations` | `1000` | จำนวนรอบฝึกฝนการค้นหา |

**ฟังก์ชันประเมินความเหมาะสม (Fitness):** `Sharpe + λ·H(w)/log(n) − penalty` โดยที่ `H(w)` คือเอนโทรปีของสัดส่วนเพื่อวัดความสม่ำเสมอในการลงน้ำหนัก และ `penalty` คือฟังก์ชันการทำโทษสัดส่วนเกินกำหนดสูงสุด

**Returns:** `np.ndarray (n_assets,)` — น้ำหนักการลงทุนของแต่ละสินทรัพย์รวมกันเท่ากับ 1

---

**`optimize_weights_aco_ebgwo(train_returns_gpu, target_assets, heuristic_tensor, sector_labels, num_iterations, num_agents) → (weights, best_conv, avg_conv)`**

โมเดลทำงานร่วมกันแบบ Co-evolutionary ACO + EBGWO โดยให้ ACO คัดกรองตัวสินทรัพย์เข้าในพอร์ตโฟลิโอ และให้ EBGWO จัดสัดส่วนน้ำหนักการลงทุนของกลุ่มสินทรัพย์ที่เลือกเหล่านั้นอย่างสอดคล้องตามข้อจำกัดด้านอุตสาหกรรม (`sector_labels`)

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (กราฟการลู่เข้า)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของประชากรในแต่ละรอบการวนลูป

---

**`optimize_weights_pso(train_returns_gpu, max_weight, lambda_ent, num_particles, iterations) → (weights, best_conv, avg_conv)`**

การจัดหาสัดส่วนด้วยวิธี Particle Swarm Optimization (PSO) โดยมีเป้าหมายเพื่อหาจุดสูงสุดของ Sharpe Ratio ร่วมกับการควบคุมเอนโทรปี (Entropy Regularization) น้ำหนักความเฉื่อย (Inertia Weight) จะลดลงแบบเชิงเส้นเพื่อรักษาสมดุลระหว่างการสำรวจพื้นที่ใหม่ (exploration) และการขยายผลจุดที่ดีที่สุดเดิม (exploitation)

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ (Entropy regularization strength) |
| `num_particles` | `500` | จำนวนอนุภาค (swarm size) ในระบบ |
| `iterations` | `1000` | จำนวนรอบของการทำงาน |

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของกลุ่มในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)

---

**`optimize_weights_clpso(train_returns_gpu, max_weight, lambda_ent, num_particles, iterations) → (weights, best_conv, avg_conv)`**

การจัดหาสัดส่วนด้วยวิธี Comprehensive Learning Particle Swarm Optimization (CLPSO) เพื่อให้มิติ (dimensions) ของแต่ละอนุภาคเรียนรู้จากตัวแทน (exemplars) ที่สร้างจากจุดส่วนตัวที่ดีที่สุดของอนุภาคอื่นๆ ในฝูง ช่วยรักษาความหลากหลายของประชากรและป้องกันการติดอยู่ในค่าที่เหมาะสมเฉพาะถิ่น (local optima)

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ |
| `num_particles` | `500` | จำนวนอนุภาค (swarm size) ในระบบ |
| `iterations` | `1000` | จำนวนรอบของการทำงาน |

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของกลุ่มในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)

---

**`optimize_weights_apso(train_returns_gpu, max_weight, lambda_ent, num_particles, iterations) → (weights, best_conv, avg_conv)`**

การจัดหาสัดส่วนด้วยวิธี Adaptive Particle Swarm Optimization (APSO) โดยอ้างอิงหลักการประมาณสถานะวิวัฒนาการ (Evolutionary State Estimation: ESE) ปรับเปลี่ยนน้ำหนักความเฉื่อย $w$ และค่าสัมประสิทธิ์การเรียนรู้ $c_1, c_2$ ตามระยะห่างและการกระจายตัวของประชากรในแต่ละรอบการวนลูปโดยอัตโนมัติ

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ |
| `num_particles` | `500` | จำนวนอนุภาค (swarm size) ในระบบ |
| `iterations` | `1000` | จำนวนรอบของการทำงาน |

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของกลุ่มในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)

---

**`optimize_weights_lapso(train_returns_gpu, max_weight, lambda_ent, num_particles, iterations) → (weights, best_conv, avg_conv)`**

การจัดหาสัดส่วนด้วยวิธี Landscape-Aware Adaptive Particle Swarm Optimization (LAPSO) (2022) ปรับเปลี่ยนน้ำหนักความเฉื่อย $w$ และค่าสัมประสิทธิ์การเรียนรู้ $c_1, c_2$ ในแต่ละรอบการวนลูปโดยใช้สัมประสิทธิ์ความสอดคล้องระหว่างค่าความเหมาะสมและระยะห่าง (Fitness Distance Correlation: FDC) เพื่อวิเคราะห์ลักษณะภูมิประเทศ (landscape modality) ของฟังก์ชันเป้าหมาย พร้อมระบบจัดการขอบเขตแบบสะท้อนกลับ (Mirrored Boundary Handling)

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ |
| `num_particles` | `500` | จำนวนอนุภาค (swarm size) ในระบบ |
| `iterations` | `1000` | จำนวนรอบของการทำงาน |

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของกลุ่มในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)

---

**`optimize_weights_acor(train_returns_gpu, max_weight, lambda_ent, num_particles, iterations) → (weights, best_conv, avg_conv)`**

การหาค่าน้ำหนักพอร์ตด้วยวิธี Ant Colony Optimization for Continuous Domains ($ACO_{\mathbb{R}}$ หรือ ACOR) ซึ่งทำการบันทึกและสืบทอดประชากรมดที่ดีที่สุดผ่านคลังเก็บชุดคำตอบ (Solution Archive) ขนาด $k$ เพื่อทำหน้าที่เป็นความทรงจำฟีโรโมน จากนั้นสร้างมดตัวใหม่ผ่านฟังก์ชันความน่าจะเป็นโดยใช้การแจกแจงแบบเกาส์เซียน (Gaussian Kernel) อิงลำดับคำตอบ และกระจายค้นหาตัวแปรด้วยส่วนเบี่ยงเบนมาตรฐานที่อัปเดตแบบไดนามิก

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ |
| `num_particles` | `500` | จำนวนมด/คำตอบที่สร้างในแต่ละรอบการวนลูป |
| `iterations` | `1000` | จำนวนรอบของการทำงาน |

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของกลุ่มในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)

---

**`optimize_weights_ciac(train_returns_gpu, max_weight, lambda_ent, num_particles, iterations) → (weights, best_conv, avg_conv)`**

การหาค่าน้ำหนักพอร์ตด้วยวิธี Continuous Interacting Ant Colony (CIAC) [Dréo and Siarry, 2002] ที่ใช้แรงขับเคลื่อนปฏิสัมพันธ์สามรูปแบบในการสำรวจ Simplex ของน้ำหนักสินทรัพย์ ได้แก่ แรงดึงดูดฟีโรโมนสะสม (Stigmergic Attraction) ไปยังตำแหน่งที่ดีที่สุดในอดีต แรงดึงดูดโดยตรงระหว่างมด (Direct Interaction) ที่ดึงดูดมดที่ประสิทธิภาพต่ำเข้าหามดที่ประสิทธิภาพสูงกว่า และการสุ่มก้าวสำรวจ (Random Walk) ที่มีรัศมีการขยายตัวที่ลดลงตามกาลเวลา

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `max_weight` | `0.1` | เพดานควบคุมสัดส่วนสูงสุดของการซื้อสินทรัพย์ตัวใดตัวหนึ่งในพอร์ต |
| `lambda_ent` | `0.05` | สัมประสิทธิ์การควบคุมความสม่ำเสมอในสัดส่วนพอร์ตโฟลิโอ |
| `num_particles` | `500` | จำนวนประชากรมดในรัง |
| `iterations` | `1000` | จำนวนรอบของการทำงาน |

**Returns:**
- `weights` — `np.ndarray (n_assets,)` น้ำหนักการลงทุนสุดท้ายของแต่ละหลักทรัพย์
- `best_conv` — `list[float]` ค่า fitness ที่ดีที่สุดในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)
- `avg_conv` — `list[float]` ค่า fitness เฉลี่ยของกลุ่มในแต่ละรอบการวนลูป (ส่งคืนเฉพาะเมื่อ `return_convergence=True`)

---

### `backtest.py` — ระบบจำลองทดสอบเดินหน้า (Walk-Forward Engine)

**`compute_metrics(returns, risk_free_rate) → dict`**

ประเมินและคัดแยกตัวชี้วัดความน่าจะเป็นทางการเงินจากชุดข้อมูลอัตราการเจริญเติบโตรายวันของพอร์ต

| ตัวชี้วัดที่รายงาน | คำอธิบายความหมาย |
| :--- | :--- |
| `cum_return` | อัตราผลตอบแทนสะสมสุทธิตลอดช่วงระยะเวลาจำลองพอร์ต |
| `ann_return` | อัตราผลตอบแทนเฉลี่ยปรับค่าปี (Annualised Return) |
| `ann_vol` | ความผันผวนปรับค่าปี (Annualised Volatility) |
| `sharpe` | อัตราส่วนความคุ้มค่าคุ้มความเสี่ยง (Sharpe Ratio) |
| `max_dd` | ข้อมูลเปอร์เซ็นต์ขาดทุนสะสมสูงสุดจากจุดยอดเดิม (Maximum Drawdown) |
| `cum_returns_arr` | ชุดข้อมูลผลตอบแทนสะสมรายวันตลอดช่วงเวลา |

---

**คลาส: `WalkForwardBacktester(full_returns, spy_full_returns, sector_map, risk_free_rate)`**

ทำหน้าที่จัดการการทดสอบประสิทธิภาพพอร์ตโดยแบ่งช่วงเวลาและเดินหน้าทดสอบไปเรื่อยๆ เพื่อวัดผลลัพธ์ของการปรับพอร์ตอย่างต่อเนื่องตามสภาวะตลาดจริง

**เมธอด: `.run(portfolio_name, selected_stocks, lookback_window, step_size, num_iterations, num_agents, use_sector_constraints, optimizer, cost_rate) → dict`**

| พารามิเตอร์การตั้งค่า | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :--- |
| `lookback_window` | `252×3` | ขนาดหน้าต่างข้อมูลย้อนหลังสำหรับฟิตพอร์ตโฟลิโอ (วันทำการเทรด) |
| `step_size` | `21×3` | ขนาดช่วงการเดินหน้ารันผลพอร์ตเพื่อปรับพอร์ตใหม่เป็นรอบไตรมาส |
| `num_iterations` | `1000` | จำนวนรอบของการหาค่าพอร์ตน้ำหนักที่เหมาะสมต่อหน้าต่างจำลอง |
| `num_agents` | `500` | จำนวนประชากรหมาป่า/มด/อนุภาค ในการประมวลผลต่อหน้าต่างจำลอง |
| `use_sector_constraints` | `False` | การเปิดใช้ข้อจำกัดควบคุมสัดส่วนการลงทุนตามกลุ่มอุตสาหกรรมในโมเดล ACO+EBGWO |
| `optimizer` | `'aco_ebgwo'` | ตัวระบุออปติไมเซอร์ที่เลือกใช้งาน (`'aco_ebgwo'` หรือ `'pso'`) |
| `cost_rate` | `0.0` | อัตราค่าธรรมเนียม/ต้นทุนธุรกรรมที่ถูกคิดคำนวณจาก Turnover แบบสองทาง |

**ผลลัพธ์ส่งออก (Keys ใน Dictionary):**
- `Strategy`: ชื่อกลยุทธ์/พอร์ตโฟลิโอเป้าหมาย
- `Cum Return` / `Ann Return` / `Ann Volatility` / `Sharpe Ratio` / `Max Drawdown` (แบบคิดค่าธรรมเนียมธุรกรรม)
- `OOS_Returns_Array` / `OOS_Cum_Returns_Array` (อาเรย์ผลตอบแทนแบบคิดค่าธรรมเนียม OOS)
- `Cum Return (No Cost)` / `Ann Return (No Cost)` / `Ann Volatility (No Cost)` / `Sharpe Ratio (No Cost)` / `Max Drawdown (No Cost)` (แบบไม่คิดค่าธรรมเนียมธุรกรรม)
- `OOS_Returns_Array_No_Cost` / `OOS_Cum_Returns_Array_No_Cost` (อาเรย์ผลตอบแทนแบบไม่คิดค่าธรรมเนียม OOS)
- `Avg_Best_Convergence` / `Avg_Avg_Convergence` (ประวัติการลู่เข้าของออปติไมเซอร์เฉลี่ย)
- `Dates`: อาเรย์วันที่ (Datetime) ทั้งหมดในหน้าต่าง Walk-Forward ของช่วง OOS

### `cli.py` — การรายงานผ่าน Terminal ที่สวยงามด้วย Rich

| หน้าที่ฟังก์ชัน | คำอธิบายการแสดงผล |
| :--- | :--- |
| `print_welcome()` | แสดงผลแผงต้อนรับระบบด้วยสีสันและข้อมูลสกุลเงินบาทที่พอร์ตจำลอง |
| `print_asset_summary(assets_df)` | จัดทำตารางแสดงข้อมูลหลักทรัพย์ อุตสาหกรรม และประเภทค่าเงินที่โหลด |
| `print_selection_summary(strategy_name, df_selected)` | แสดงผลตารางหลักทรัพย์ที่แต่ละกลยุทธ์เลือกพร้อมระบุ Sharpe และคะแนนกระจายความเสี่ยง |
| `print_backtest_table(results, title)` | จัดทำตารางเปรียบเทียบประสิทธิภาพในการรันแบบ Walk-forward |

ทุกตารางควบคุมด้วยความกว้างคงที่ 120 ตัวอักษร เพื่อแก้ปัญหามุมมองบิดเบี้ยวบน Terminal ของ Windows

---

### `charts.py` — การสร้างกราฟแท่งเทียนเปรียบเทียบ

**`save_candlestick_grid(strategy_name, selected_df, start_date, end_date, output_path, max_cols, candle_period, lookback_bars, sector_map) → None`**

ดึงราคาย้อนหลังหลักทรัพย์ทั้งหมดในผลลัพธ์ และประมวลภาพโครงข่ายกราฟแท่งเทียนสีสันธีมเข้ม พร้อมเน้นเส้นตารางขอบพอร์ตตามจำแนกกลุ่มอุตสาหกรรมและบันทึกเป็นแฟ้มภาพ PNG

**รายละเอียดโครงร่างภาพ:**
- ธีมสีมืดแบบ `#0d1117` (สไตล์ GitHub Dark Mode)
- กำหนดสีแท่งเทียนขาขึ้นเป็นสีเขียวสด (`#00e676`) และขาลงเป็นสีแดงสด (`#ff1744`)
- แสดงเปอร์เซ็นต์อัตราการเปลี่ยนแปลงมูลค่าของหลักทรัพย์แต่ละตัวบนหัวรูปภาพย่อย
- บันทึกภาพที่ความละเอียดเหมาะสม 150 DPI
- เรียกใช้งานโดยอัติโนมัติผ่าน `run_pipeline.py` เพื่อเก็บภาพผลลัพธ์สินทรัพย์ของทุกกลยุทธ์ในโฟลเดอร์ `candles/` ใต้โฟลเดอร์ของรอบการทำงานนั้นๆ
