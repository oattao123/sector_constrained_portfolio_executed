# 📊 สรุปผลการวิเคราะห์: Summary.ipynb
## การเปรียบเทียบวิธีคัดเลือกสินทรัพย์ด้วย Walk-Forward Optimization

> **Notebook**: `Summary.ipynb`
> **สกุลเงินฐาน**: THB (บาทไทย)
> **ช่วงข้อมูล**: 2015-01-01 ถึง 2024-01-01 (3,287 วันซื้อขาย)
> **Benchmark**: SPY (S&P 500 ETF)

---

## 🎯 วัตถุประสงค์

Notebook นี้เป็น **ศูนย์กลางการเปรียบเทียบ** วิธีคัดเลือกสินทรัพย์ทั้งหมด 6 แบบ โดยใช้
**Walk-Forward Optimization** (Rebalance ทุก ~2 เดือน) เพื่อประเมินผลที่สมจริงที่สุด
สำหรับนักลงทุนไทย โดยแปลงราคาทุกสินทรัพย์เป็น **THB** ก่อนคำนวณ

---

## 📦 Universe และการเตรียมข้อมูล

| รายการ | รายละเอียด |
| :--- | :--- |
| **Universe เริ่มต้น** | 629 ตัว (S&P 500: 503, Thai SET: 69, Chinese ADRs: 20, ETFs: 37) |
| **หลังกรองคุณภาพ** | 571 ตัว (ตัดสินทรัพย์ที่ขาดข้อมูล ≥ 252 วัน) |
| **อัตราแลกเปลี่ยน** | USD/THB: 29.68 – 38.30 บาท/ดอลลาร์ |
| **สินทรัพย์ USD** | 511 ตัว (แปลงเป็น THB รายวัน) |
| **หุ้นไทย** | 60 ตัว (ในหน่วย THB อยู่แล้ว) |
| **Risk-Free Rate** | US 10Y Treasury (^TNX) ≈ 4.34% |
| **เงินทุนเริ่มต้น** | 1,000,000 บาท |

---

## 🔬 วิธีคัดเลือกสินทรัพย์ทั้ง 6 แบบ

### แนวคิด Diversification Score
```
Diversification Score = Sharpe Ratio × (1 - Avg Correlation กับพอร์ต)
```
คะแนนสูง = ให้ผลตอบแทนดี **และ** ไม่เคลื่อนไหวตามพอร์ตอื่น (กระจายความเสี่ยงได้จริง)

| # | ชื่อวิธี | แนวคิดหลัก | เกณฑ์ Filter | จำนวนที่ได้ |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Diversification Selected** | Simple Returns + Avg Correlation | Sharpe > 0.2 (193 ตัว) | 30 |
| 2 | **LW Diversification Selected** | Ledoit-Wolf GPU + LW Correlation | LW Sharpe > 0.2 (187 ตัว) | 30 |
| 3 | **LW Sharpe Cluster** | HRP Clustering + Max Sharpe/Cluster | Ward Linkage 30 Clusters | 30 |
| 4 | **LW Div Cluster** | HRP Clustering + Max Div Score/Cluster | Ward Linkage 30 Clusters | 30 |
| 5 | **ACO Selected** | 2D-ACO อิสระ (Pheromone 2D N×N) | LW Sharpe Heuristic | 30 |
| 6 | **ACO Cluster Selected** | 2D-ACO + บังคับ 1 ตัว/Cluster | 30 Clusters | 30 |

---

## ⚖️ ผลลัพธ์ Equal-Weight เบื้องต้น (ก่อน Walk-Forward)

> วัดคุณภาพของ "ชุดสินทรัพย์" ล้วนๆ โดยไม่ใช้การหาน้ำหนักที่เหมาะสม

| วิธีคัดเลือก | Return/ปี | Vol/ปี | Sharpe (EW) | Avg Corr | Avg Div Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Diversification Selected | 19.88% | 17.61% | **1.1287** | 0.3040 | 0.3449 |
| LW Diversification Selected | 19.91% | 17.54% | 0.8890 | 0.2864 | 0.3497 |
| LW Sharpe Cluster | 14.67% | 14.18% | 0.7299 | 0.2735 | 0.2410 |
| LW Div Cluster | 14.71% | 14.09% | 0.7377 | **0.2674** | 0.2417 |
| ACO Selected | 17.56% | 15.54% | 0.8527 | 0.2760 | 0.2952 |
| ACO Cluster Selected | 14.44% | 13.83% | 0.7314 | **0.2526** | **0.3731** |

---

## 📈 ผลลัพธ์ Walk-Forward (หน่วย THB, ช่วง 2015–2024)

> ผลลัพธ์นี้สำคัญที่สุดสำหรับนักลงทุนไทย: รวมผลกระทบค่าเงิน + Rebalance จริง

| กลยุทธ์ | Cum Return | Return/ปี | Vol/ปี | Sharpe | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LW Diversification** | **839.96%** | **18.78%** | 17.03% | **1.1029** | -33.65% |
| 🥈 **Diversification** | 706.05% | 17.61% | 17.12% | 1.0286 | -37.69% |
| 🥉 **ACO Selected** | 523.77% | 15.47% | 16.20% | 0.9550 | -34.99% |
| LW Div Cluster | 317.91% | 12.00% | 13.70% | 0.8761 | -32.65% |
| LW Sharpe Cluster | 288.65% | 11.43% | 13.65% | 0.8374 | -35.31% |
| ACO Cluster | 269.59% | 10.97% | **13.08%** | 0.8385 | **-31.57%** |
| **SPY (Benchmark)** | 196.39% | 9.60% | 15.42% | 0.6222 | -35.75% |

> [!IMPORTANT]
> **ทุก 6 กลยุทธ์ชนะ SPY** ในช่วงเวลาเต็ม ทั้งด้านผลตอบแทนและ Sharpe Ratio
> **LW Diversification** ให้ผลตอบแทนสะสมสูงกว่า SPY ถึง **4.3 เท่า** (839% vs 196%)

---

## 📅 ผลลัพธ์ Walk-Forward เฉพาะปี 2025 (หน่วย THB)

> ทดสอบความสามารถรับมือกับสภาวะตลาดล่าสุด

| กลยุทธ์ | Cum Return | Return/ปี | Vol/ปี | Sharpe | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **LW Sharpe Cluster** | **12.25%** | **8.81%** | **12.70%** | **0.3516** | **-15.06%** |
| ACO Cluster | 7.22% | 5.59% | 12.32% | 0.1013 | -15.12% |
| LW Div Cluster | 8.49% | 6.68% | 14.36% | 0.1627 | -19.27% |
| ACO Selected | 6.90% | 6.12% | 17.29% | 0.1028 | -23.81% |
| LW Diversification | 5.67% | 5.30% | 17.18% | 0.0557 | -23.21% |
| Diversification | 1.83% | 2.77% | 17.37% | -0.0904 | -25.06% |
| **SPY (Benchmark)** | **16.44%** | **11.81%** | 15.98% | **0.4674** | -19.21% |

> [!WARNING]
> **ทุกกลยุทธ์ยังตามหลัง SPY ในปี 2025** เพราะตลาดสหรัฐฯ ขาขึ้นแรงจาก Mega-Cap
> อย่างไรก็ตาม **LW Sharpe Cluster จำกัด Max DD ได้ดีที่สุด (-15.06%)** ต่ำกว่า SPY (-19.21%)

---

## 🤖 ทดสอบ Co-Evolutionary ACO + EBGWO

Walk-Forward ด้วยอัลกอริทึมขั้นสูงสุด (ACO คัดสินทรัพย์ + EBGWO หาน้ำหนัก
+ Monte Carlo Bootstrapping + Entropy Regularization):

| ช่วงเวลา | Cum Return | Return/ปี | Vol/ปี | Sharpe | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **เต็มช่วง** | 188.40% | 9.66% | 17.09% | 0.5648 | -37.52% |
| **ปี 2025** | 10.26% | 7.60% | **12.98%** | 0.2516 | **-11.48%** |
| **SPY (เต็มช่วง)** | 196.39% | 9.60% | 15.42% | 0.6222 | -35.75% |
| **SPY (ปี 2025)** | 16.44% | 11.81% | 15.98% | 0.4674 | -19.21% |

> **ข้อค้นพบน่าสนใจ**: อัลกอริทึมที่ซับซ้อนที่สุดนี้ไม่ได้ให้ผลดีที่สุดในระยะยาว
> แต่ **Max Drawdown ปี 2025 ต่ำที่สุดในทุกวิธี (-11.48%)** จาก Entropy Regularization

---

## 🏆 สินทรัพย์ที่ถูกเลือกในทุกวิธี (Core Holdings)

| Ticker | กลุ่ม | LW Return/ปี | LW Vol/ปี | LW Sharpe | จุดเด่น |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **NVDA** | IT | 35.83% | 40.35% | 0.781 | อันดับ 1 ทุกวิธี |
| **BTC-USD** | Crypto | 37.92% | 59.36% | 0.566 | Return สูงสุด |
| **COM7.BK** | Thai | 20.14% | 32.47% | 0.487 | Avg Corr เพียง 0.06 |
| **LLY** | Health Care | 17.99% | 24.12% | 0.567 | Defensive + Growth |
| **CPRT** | Industrials | 18.52% | 24.00% | 0.605 | Vol ต่ำ Sharpe ดี |
| **DELTA.BK** | Thai | 21.30% | 46.03% | 0.368 | ตัวขับเคลื่อนหลักไทย |

---

## 📊 Top 10 Sharpe Ratio ในช่วง 2015–2024 (Simple Returns, THB)

| อันดับ | Ticker | กลุ่ม | Return/ปี | Vol/ปี | Sharpe |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | **NVDA** | Information Technology | 44.03% | 40.70% | 0.976 |
| 2 | **BTC-USD** | Crypto | 55.67% | 59.27% | 0.866 |
| 3 | **AMD** | Information Technology | 43.24% | 49.91% | 0.780 |
| 4 | **CDNS** | Information Technology | 24.41% | 26.85% | 0.748 |
| 5 | **FICO** | Information Technology | 26.01% | 29.70% | 0.731 |
| 6 | **MSCI** | Financials | 23.49% | 26.25% | 0.730 |
| 7 | **SNPS** | Information Technology | 22.45% | 24.90% | 0.728 |
| 8 | **CPRT** | Industrials | 21.27% | 23.47% | 0.722 |
| 9 | **AVGO** | Information Technology | 25.33% | 30.07% | 0.699 |
| 10 | **LLY** | Health Care | 20.79% | 23.79% | 0.692 |

---

## 🔑 สรุปข้อค้นพบหลัก 5 ข้อ

1. **Ledoit-Wolf บน GPU ให้ผลดีที่สุดระยะยาว**: LW Diversification ชนะด้วย Sharpe 1.1029
   และผลตอบแทนสะสม 839.96% (สูงกว่า SPY 4.3 เท่า) เพราะ Covariance Matrix เสถียรกว่า

2. **Cluster-Based ลด Vol แต่ลด Return ด้วย**: HRP Clustering ให้ Vol 13-14%
   (ต่ำกว่า SPY 15.42%) แต่ Return ลดลงเหลือ 11-12% ต่อปี เหมาะกับนักลงทุนอนุรักษ์นิยม

3. **2D-ACO เรียนรู้ Synergy ได้จริง**: ACO Cluster ให้ Avg Corr ต่ำสุด (0.2526)
   แสดงว่าการเรียนรู้คู่หุ้นผ่าน Pheromone 2D ช่วยกระจายความเสี่ยงได้ดีกว่า

4. **ความซับซ้อนไม่ได้หมายความว่าดีกว่า**: Co-Evolutionary ACO+EBGWO ที่ซับซ้อนที่สุด
   ให้ผลสะสมต่ำที่สุด (188%) แต่ควบคุม Max Drawdown ในปี 2025 ได้ดีที่สุด (-11.48%)

5. **ผลกระทบค่าเงิน THB สำคัญมาก**: การแปลงเป็น THB ทำให้ LW Diversification
   ขึ้นมาเป็นอันดับ 1 แสดงว่าพอร์ตนี้ได้ประโยชน์จากค่าเงินบาทอ่อนช่วง 2015-2024

---

## 💡 คำแนะนำตามประเภทนักลงทุน

| ประเภทนักลงทุน | วิธีที่แนะนำ | เหตุผล |
| :--- | :--- | :--- |
| เน้นผลตอบแทนสูงสุดระยะยาว | LW Diversification | Sharpe 1.1029, Cum 839.96% |
| สมดุล Return/Risk | ACO Selected | Sharpe 0.9550, Return 15.47%/ปี |
| เน้น Drawdown ต่ำปี 2025 | LW Sharpe Cluster | Max DD -15.06%, Sharpe 0.3516 |
| ความปลอดภัยสูงสุดระยะสั้น | ACO+EBGWO Co-Evo | Max DD 2025 ต่ำสุด -11.48% |

---

*สรุปจากผลลัพธ์จริงใน `Summary.ipynb` | Walk-Forward Rebalance ทุก ~2 เดือน*
*⚠️ เพื่อวัตถุประสงค์ทางการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน*