# 📊 Global Multi-Asset Sector-Constrained Portfolio Optimization
# 📊 การเพิ่มประสิทธิภาพพอร์ตโฟลิโอหลายสินทรัพย์ทั่วโลกแบบมีข้อจำกัดรายกลุ่ม

> **State-of-the-Art Portfolio Engineering**: Combining Modern Portfolio Theory (MPT), Tail-Risk Management (CVaR), and Advanced Metaheuristic Optimizers (PSO, DE, ACO, EBGWO).
> 
> **วิศวกรรมพอร์ตโฟลิโอขั้นสูง**: การผสมผสานทฤษฎีพอร์ตโฟลิโอสมัยใหม่ (MPT), การบริหารความเสี่ยงระดับวิกฤต (CVaR) และตัวหาค่าเหมาะสมขั้นสูง (PSO, DE, ACO, EBGWO)

---

## 🌍 Overview / ภาพรวม

**[EN]**: This project provides a comprehensive framework for optimizing a global multi-asset portfolio. It integrates standard **Mean-Variance Optimization (Sharpe Ratio)** with sophisticated **Conditional Value at Risk (CVaR)** models to manage extreme market events (Tail Risk). Across **4 specialized methodologies**, we analyze **50 global assets** across **10 sectors**, incorporating real-world constraints such as transaction lots, regional costs, and leverage limits.

**[TH]**: โปรเจกต์นี้เป็นโครงสร้างสำหรับการเพิ่มประสิทธิภาพพอร์ตโฟลิโอหลายสินทรัพย์ทั่วโลก โดยการรวมการหาจุดเหมาะสมแบบ **Mean-Variance (Sharpe Ratio)** มาตรฐานเข้ากับโมเดล **Conditional Value at Risk (CVaR)** ที่ซับซ้อนเพื่อจัดการกับเหตุการณ์ตลาดที่รุนแรง (ความเสี่ยงส่วนปลาย หรือ Tail Risk) ผ่าน **4 แนวทางเฉพาะทาง** เราวิเคราะห์สินทรัพย์ **50 รายการ** ใน **10 กลุ่มอุตสาหกรรม** โดยนำข้อจำกัดในโลกจริง เช่น หน่วยการซื้อขาย (Lot Size) ต้นทุนรายภูมิภาค และขีดจำกัด Leverage มาพิจารณา

---

## 📂 Project Structure & Methodologies / โครงสร้างโปรเจกต์และแนวทาง

| Track | Directory | Target Objective | Core Feature |
| :--- | :--- | :--- | :--- |
| **Simple** | `/sector_constrained_portfolio_executed_Simple` | Max Sharpe Ratio | Baseline MVO comparison |
| **Alternative** | `/sector_constrained_portfolio_executed_alternavie` | Heuristic Max Sharpe | Extended PSO/DE Hybrid analysis |
| **CVaR** | `/sector_constrained_portfolio_executed_CVaR` | Min Tail Risk (CVaR) | **STARR Ratio** optimization using ACO & EBGWO |
| **Constraints** | `/sector_constrained_portfolio_executed_Simple_contrains` | Execution Modeling | High-fidelity lot-size & cost accounting |

---

## 🍎 Asset Universe (50 Tickers, 10 Sectors) / ขอบเขตสินทรัพย์ (50 ตัวเลือก, 10 กลุ่ม)

**[EN]**: All assets are screened for high risk-adjusted performance and converted to **THB Base currency**.
**[TH]**: สินทรัพย์ทั้งหมดผ่านการคัดกรองประสิทธิภาพด้านผลตอบแทนต่อความเสี่ยง และแปลงเป็น **สกุลเงินบาท (THB)** เป็นเกณฑ์หลัก

| Sector | Asset Tickers | Region |
| :--- | :--- | :--- |
| **US Tech** | AAPL, NVDA, META, GOOGL, MSFT | 🇺🇸 |
| **US Healthcare** | JNJ, ABBV, LLY, UNH, MRK | 🇺🇸 |
| **US Financial** | JPM, GS, V, MA, BAC | 🇺🇸 |
| **US Energy** | XOM, MPC, CVX, COP, EOG | 🇺🇸 |
| **US Consumer** | CAT, WMT, COST, MCD, HD | 🇺🇸 |
| **Thai Stocks** | DELTA.BK, ADVANC.BK, KBANK.BK, PTT.BK, AOT.BK | 🇹🇭 |
| **Chinese Stocks** | PDD, FUTU, BABA, JD, BIDU | 🇨🇳 |
| **Crypto** | BTC-USD, ETH-USD, BNB-USD, SOL-USD, XRP-USD | ₿ |
| **Bonds** | HYG, VCSH, EMLC, AGG, BND | 🌏 |
| **Commodities** | GLD, SLV, COPX, IAU, SGOL | 🪙 |

---

## 🤖 Advanced Optimization Algorithms / อัลกอริทึมการหาค่าเหมาะสมขั้นสูง

**[EN]**: We deploy a suite of advanced algorithms to navigate the non-convex solution space created by sector constraints:
1.  **SLSQP**: Gradient-based local optimizer for high-precision refinement.
2.  **PSO (Particle Swarm)**: Swarm-based global search to avoid local optima.
3.  **DE (Differential Evolution)**: Population-based evolutionary strategy for robust convergence.
4.  **ACO (Ant Colony)**: Foraging logic used specifically in CVaR targets for path-based optimization.
5.  **EBGWO (Grey Wolf)**: Pack-hunting optimization for ultra-stable weight discovery.
6.  **NSGA-II**: Multi-objective mapping of the **Pareto Front** (Risk vs. Return).

**[TH]**: เราใช้อัลกอริทึมขั้นสูงเพื่อค้นหาคำตอบในพื้นที่ที่มีข้อจำกัดซับซ้อน:
1.  **SLSQP**: การหาค่าเหมาะสมแบบอ้างอิงความชันเพื่อความแม่นยำสูง
2.  **PSO (Particle Swarm)**: การจำลองพฤติกรรมฝูงเพื่อค้นหาคำตอบในระดับสากลและเลี่ยงจุดต่ำสุดเฉพาะที่
3.  **DE (Differential Evolution)**: กลยุทธ์วิวัฒนาการระดับประชากรเพื่อการลู่เข้าที่มั่นคง
4.  **ACO (Ant Colony)**: ตรรกะการหาอาหารของมด ใช้สำหรับเป้าหมาย CVaR เพื่อหาเส้นทางที่เหมาะสม
5.  **EBGWO (Grey Wolf)**: การปรับปรุงการล่าแบบฝูงหมาป่าเพื่อการค้นหาน้ำหนักที่เสถียรเป็นพิเศษ
6.  **NSGA-II**: การทำ Multi-objective เพื่อระบุ **Pareto Front** ระหว่างความเสี่ยงและผลตอบแทน

---

## 📈 Performance Highlights / ไฮไลท์ประสิทธิภาพ

| Metric | Max Sharpe (Simple) | Max STARR (CVaR - ACO) | Long-Short (1.3x) |
| :--- | :--- | :--- | :--- |
| **Annual Return** | 35.67% | 35.80% | 33.99% |
| **Risk Metric** | 18.19% (Vol) | 2.22% (CVaR) | 11.90% (Vol) |
| **Efficiency Ratio** | **1.738 (Sharpe)** | **12.1169 (STARR)** | **2.857 (Sharpe)** |
| **Max Drawdown** | -13.05% | -10.45% | -9.80% |

> [!TIP]
> **Key Finding**: The **1.3x Leverage** strategy identified a "Sweet Spot" where short selling significantly hedges market volatility without escalating gross risk.
>
> **ข้อค้นพบหลัก**: กลยุทธ์ **Leverage 1.3 เท่า** เป็นจุดที่เหมาะสมที่สุด (Sweet Spot) โดยการขายชอร์ตช่วยป้องกันความเสี่ยงจากความผันผวนของตลาดได้ดีโดยไม่เพิ่มความเสี่ยงรวมจนเกินไป

---

## 🛠 Real-World Execution Constraints / ข้อจำกัดในการใช้งานจริง

- **Regional Lot Sizes**: Fixed 100-share lots for Thai stocks; fractional support for Crypto.
  - **หน่วยการซื้อขาย**: กำหนด 100 หุ้นสำหรับหุ้นไทย และรองรับทศนิยมสำหรับคริปโต
- **Transaction Costs**: 0.20% flat fee per trade incorporated into rebalancing frequency.
  - **ต้นทุนธุรกรรม**: ค่าธรรมเนียมคงที่ 0.20% ต่อการเทรด ถูกรวมคำนวณในความถี่การปรับพอร์ต
- **Tracking Error**: Maintained at **RMSE 5.749%** post lot-adjustment.
  - **ความคลาดเคลื่อนสี**: รักษาค่า RMSE ไว้ที่ 5.749% หลังการปรับตาม Lot Size
- **Out-of-Sample (OOS)**: Rigorous 75/25 Train-Test split showing highly durable performance (~11-15% degradation).
  - **การทดสอบนอกกลุ่มตัวอย่าง**: การทดสอบ Train-Test สัดส่วน 75/25 แสดงให้เห็นความทนทานของโมเดล (ประสิทธิภาพลดลงเพียง 11-15%)

---

## 🚀 How to Run / วิธีการรัน

```bash
# Install uv dependencies / ติดตั้ง dependencies
uv sync

# Run the consolidated dashboard or specific notebooks / รันแดชบอร์ดหรือโน้ตบุ๊ก
uv run jupyter notebook
```
