# 📊 รายงานฉบับสมบูรณ์: การเพิ่มประสิทธิภาพพอร์ตโฟลิโอหลายสินทรัพย์ทั่วโลก
# 📊 Comprehensive Report: Global Multi-Asset Portfolio Optimization

---

## 1. การออกแบบโมเดลและกรอบทฤษฎี / Model Design & Theoretical Framework

### [TH] 1.1 แนวคิดการกระจายความเสี่ยงระดับสากล
โปรเจกต์นี้ขยายขอบเขตสภาวะการลงทุนไปสู่สินทรัพย์ **50 รายการ** ครอบคลุม **10 กลุ่มอุตสาหกรรม** ทั่วโลก เพื่อลดความเสี่ยงเชิงระบบ (Systemic Risk) ผ่านความสัมพันธ์ที่ต่ำระหว่างประเภทสินทรัพย์ (Low Correlation):
*   **เป้าหมาย**: การสร้างพอร์ตโฟลิโอที่ไม่กระจุกตัวในภูมิภาคหรืออุตสาหกรรมใดอุตสาหกรรมหนึ่งมากเกินไป

### [EN] 1.1 Global Diversification Concept
This project expands the investment universe to **50 assets** across **10 global industrial sectors** to mitigate systemic risk through low correlation between asset classes:
*   **Objective**: To construct a portfolio that is not over-concentrated in any single region or industry.

---

### [TH] 1.2 ฟังก์ชันเป้าหมายและความคุ้มค่า
เราใช้ 2 แนวทางหลักในการวัดประสิทธิภาพพอร์ตโฟลิโอ:
1.  **Sharpe Ratio (MVO)**: เน้นการหาผลตอบแทนส่วนเกินต่อความผันผวน (Volatility) เหมาะสำหรับการบริหารจัดการในสภาวะตลาดปกติ
2.  **STARR Ratio (CVaR-Based)**: เน้นการบริหารจัดการความเสี่ยงในช่วงเลวร้ายที่สุด (Tail Risk) โดยใช้ **Conditional Value at Risk (CVaR)** ที่ระดับความเชื่อมั่น 95% เพื่อป้องกันเหตุการณ์ที่ไม่คาดฝัน (Black Swan Events)

### [EN] 1.2 Objective Functions & Risk-Adjusted Returns
We utilize two primary approaches for measuring portfolio efficiency:
1.  **Sharpe Ratio (MVO)**: Focuses on maximizing excess returns relative to overall volatility, ideal for managing portfolios under normal market conditions.
2.  **STARR Ratio (CVaR-Based)**: Focuses on managing extreme downside risk (Tail Risk) using **Conditional Value at Risk (CVaR)** at a 95% confidence level to protect against "Black Swan Events."

---

## 2. อัลกอริทึมและการหาค่าเหมาะสมขั้นสูง / Advanced Optimization Algorithms

### [TH] 2.1 เทคนิค Metaheuristics
เพื่อเอาชนะปัญหาจุดต่ำสุดเฉพาะที่ (Local Optima) ในพื้นที่คำตอบที่มีข้อจำกัดซับซ้อน เราได้นำเทคนิคระดับสูงมาใช้:
*   **PSO (Particle Swarm Optimization)**: จำลองการหาอาหารของฝูงนกเพื่อหาคำตอบที่สูงที่สุดในภาพรวม
*   **DE (Differential Evolution)**: ใช้หลักการวิวัฒนาการเพื่อค้นหาความทนทานของน้ำหนักสินทรัพย์
*   **ACO (Ant Colony Optimization)**: ใช้ตรรกะฝูงมดในการค้นหาเส้นทางพอร์ตที่ให้ค่า **STARR Ratio** สูงที่สุด (ให้ผลดีเยี่ยมในโมเดลความเสี่ยงส่วนปลาย)
*   **EBGWO (Enhanced Grey Wolf Optimizer)**: จำลองการล่าของหมาป่าเพื่อให้เข้าใกล้จุดสมดุลได้แม่นยำและเสถียรยิ่งขึ้น

### [EN] 2.1 Metaheuristic Techniques
To overcome the "Local Optima" problem in complex, constrained solution spaces, we have implemented high-level techniques:
*   **PSO (Particle Swarm Optimization)**: Simulates the foraging behavior of bird flocks to discover global optima.
*   **DE (Differential Evolution)**: Utilizes evolutionary principles to identify robust asset weights.
*   **ACO (Ant Colony Optimization)**: Uses ant foraging logic to identify paths leading to the highest **STARR Ratio** (exhibiting superior performance in tail-risk models).
*   **EBGWO (Enhanced Grey Wolf Optimizer)**: Models grey wolf pack hunting to converge on equilibrium with higher precision and stability.

---

## 3. สรุปผลการทดลองและการเปรียบเทียบ / Experiment Results & Comparison

### [TH] 3.1 ตารางเปรียบเทียบประสิทธิภาพ
ผลการทดสอบเปรียบเทียบระหว่าง 4 แนวทางหลัก พบข้อมูลที่น่าสนใจดังนี้:

| กลยุทธ์ (Strategy) | ตัวชี้วัดหลัก | ผลตอบแทนรายปี (%) | ความเสี่ยง (%) | ประสิทธิภาพ (Ratio) |
| :--- | :--- | :--- | :--- | :--- |
| **Simple MVO** | Sharpe Ratio | 35.67% | 18.19% (Vol) | 1.738 |
| **CVaR (ACO)** | STARR Ratio | **35.80%** | **2.22% (CVaR)** | **12.1169** |
| **Alternative (Hybrid)** | Max Sharpe | 37.17% | 16.03% (Vol) | 2.066 |
| **Long-Short (1.3x)** | Adjusted Sharpe | 33.99% | 11.90% (Vol) | 2.857 |

### [EN] 3.1 Performance Comparison Table
Testing across four primary tracks revealed key insights:

| Strategy | Key Metric | Annual Return (%) | Risk Metric (%) | Efficiency Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Simple MVO** | Sharpe Ratio | 35.67% | 18.19% (Vol) | 1.738 |
| **CVaR (ACO)** | STARR Ratio | **35.80%** | **2.22% (CVaR)** | **12.1169** |
| **Alternative (Hybrid)** | Max Sharpe | 37.17% | 16.03% (Vol) | 2.066 |
| **Long-Short (1.3x)** | Adjusted Sharpe | 33.99% | 11.90% (Vol) | 2.857 |

> [!IMPORTANT]
> **Key Insight**: The **CVaR model using ACO** achieved a STARR Ratio of **12.1169**, providing the best safety-to-return profile during extreme market stress.
>
> **ข้อสรุปสำคัญ**: โมเดล **CVaR ที่ใช้ ACO** สามารถทำค่า STARR Ratio ได้สูงถึง **12.1169** ซึ่งให้สมดุลระหว่างความปลอดภัยและผลตอบแทนได้ดีที่สุดในช่วงตลาดวิกฤต

---

## 4. การนำไปใช้งานจริงและข้อจำกัด / Practical Implementation & Constraints

### [TH] 4.1 การปรับพอร์ตและ Leverage
*   **Lot Size Adjustment**: โมเดลคำนวณจำนวนหุ้นจริงตามกฎตลาด (เช่น หุ้นไทย 100 หุ้น) ทำให้มีค่าความคลาดเคลื่อน (RMSE) อยู่ที่ 5.749% ซึ่งอยู่ในระดับที่ต่ำและยอมรับได้
*   **The Sweet Spot (1.3x)**: การใช้ Leverage ที่ 1.3 เท่า เป็นจุดที่มีประสิทธิภาพสูงสุด เพราะช่วยให้การขายชอร์ต (Short Selling) ป้องกันความเสี่ยงได้โดยไม่เพิ่มความเสี่ยงจากการกู้ยืมจนเกินไป

### [EN] 4.1 Rebalancing & Leverage
*   **Lot Size Adjustment**: The model calculates actual share counts based on market rules (e.g., 100 shares for Thai stocks), resulting in a low Tracking Error (RMSE) of 5.749%, which is well within acceptable limits.
*   **The Sweet Spot (1.3x)**: Utilizing 1.3x leverage was identified as the most efficient point, as it enables short-selling for hedging purposes without excessive borrowing risk.

---

## 5. ความเสถียรของโมเดล / Model Stability

### [TH] 5.1 การทดสอบ Out-of-Sample
เราทดสอบความแม่นยำด้วยการแบ่งข้อมูลเป็น 75/25 (Train/Test) พบว่าประสิทธิภาพในกลุ่มข้อมูลที่ไม่เคยเห็นลดลงเพียง **11-15%** ยืนยันว่าโมเดลไม่มีปัญหา Overfitting

### [EN] 5.1 Out-of-Sample Validation
We validated accuracy using a 75/25 (Train/Test) split, finding that performance on unseen data degraded by only **11-15%**, confirming that the model avoids overfitting.

---

## 6. บทสรุป / Conclusion

**[TH]** โปรเจกต์นี้แสดงให้เห็นว่าการใช้ตัวหาค่าเหมาะสมขั้นสูงร่วมกับการจัดการความเสี่ยงช่วงวิกฤต (CVaR) สามารถสร้างพอร์ตโฟลิโอที่ทั้งทำกำไรได้สูงและมีความปลอดภัยระดับสากล เหมาะสำหรับการลงทุนในโลกจริงที่เต็มไปด้วยความผันผวน

**[EN]** This project demonstrates that combining advanced optimizers with tail-risk management (CVaR) can produce portfolios that are both highly profitable and internationally secure, suitable for real-world investments in a volatile market.
