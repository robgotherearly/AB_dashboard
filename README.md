# ⚖️ A/B Testing Dashboard

A professional **Streamlit application** for analyzing A/B tests using statistical significance testing, confidence intervals, and sample size calculations.

Designed for **data-driven decision making** with an intuitive, modern UI.

---

## 🚀 Features

### 📊 Statistical Analysis
- Chi-Square Test (conversion data)
- T-Test (continuous metrics)
- P-value calculation with configurable significance level (α)
- Automatic winner determination
- Lift percentage calculation
- 95% confidence intervals

### 🎯 Results Visualization
- Conversion rate comparison charts
- Confidence interval display
- Summary statistics tables
- Clear winner / loser indicators

### 🔬 Sample Size Calculator
- Required sample size estimation
- Adjustable statistical power (1−β)
- One-tailed and two-tailed tests
- Minimum Detectable Effect (MDE) configuration

### 📈 Data Input
- CSV upload support
- Manual data entry
- Data preview before analysis
- Flexible input formats

### 🎨 Modern UI
- Dark theme with gradient design
- Responsive layout
- Smooth animations
- Color-coded results

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas & NumPy
- SciPy
- Matplotlib & Seaborn

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip

## 🧪 How It Works
### Tab 1: Upload & Analyze

Upload CSV (group, conversions, total_visitors)

Or enter data manually

Choose test type and significance level

Click Analyze A/B Test

### Tab 2: Results

Conversion rates

Lift percentage

P-value & significance

Winner declaration

Visualizations & confidence intervals

### Tab 3: Sample Size Calculator

Baseline conversion rate

Minimum Detectable Effect (MDE)

Power & significance level

Required sample size per group

### Tab 4: Guide

A/B testing concepts

Statistical significance explained

Best practices & common mistakes

## 📁 Sample CSV Format
group,conversions,total_visitors
Control,100,1000
Variant,120,1000

## 🧠 Statistical Interpretation

P-value < 0.05 → Statistically significant

P-value ≥ 0.05 → Inconclusive (collect more data)

Winner Logic

Significant + positive lift → Variant wins

Significant + negative lift → Control wins

Not significant → No clear winner

## 📂 Project Structure
ab-testing-dashboard/
├── AB_dashboard.py
├── requirements.txt
├── README.md
└── .gitignore

## 👤 Author

Robert Marsh Deku
Aspiring Data Scientist & AI Engineer

Interests:

Artificial Intelligence

Data Engineering

Applied Machine Learning