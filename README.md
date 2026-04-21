 🏥 Healthcare Patient Satisfaction Analysis & Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-0073E6?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-2E7D32?style=for-the-badge)

**End-to-end machine learning and deep learning project analyzing 1.5M+ U.S. hospital records (2016–2020) to predict patient satisfaction ratings using CMS Medicare & Medicaid data.**

[Problem Statement](#-problem-statement) • [Dataset](#-dataset) • [Project Pipeline](#-project-pipeline) • [Models](#-models--results) • [Key Insights](#-key-insights) • [Setup](#-setup--installation) • [Results](#-results-summary)

</div>

---

## 📌 Problem Statement

Every U.S. hospital receiving Medicare and Medicaid payments is mandated by **The Centers for Medicare & Medicaid Services (CMS)** to submit annual quality and patient satisfaction data. The **CMS Star Rating Program** evaluates hospital performance on a **1–5 star scale**, measuring patient experience, clinical outcomes, care safety, and operational efficiency.

### Business Questions Addressed

| # | Question |
|---|----------|
| 1 | Which hospital types and states deliver the highest patient satisfaction? |
| 2 | Do hospitals offering emergency services achieve significantly higher satisfaction ratings? |
| 3 | Has patient satisfaction improved significantly from 2016 to 2020? |
| 4 | Can we predict patient satisfaction classification from hospital features with high accuracy? |
| 5 | Which clinical and operational factors are the strongest predictors of satisfaction? |

---

## 📂 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [CMS Provider Data — Archived Hospitals](https://data.cms.gov/provider-data/archived-data/hospitals) |
| **Coverage** | United States — All 50 States |
| **Time Period** | 2016 – 2020 (5 years) |
| **Records** | 1.5M+ rows across multiple annual CSV files |
| **Target Variable** | Patient Rating Classification (derived from Patient Survey Star Rating) |
| **Format** | Multiple CSVs merged via `glob` |

### Key Features Used

```
State                                     Hospital Type
Hospital Ownership                        Emergency Services
Hospital overall rating                   Mortality national comparison
Safety of care national comparison        Readmission national comparison
Patient experience national comparison    Effectiveness of care national comparison
Timeliness of care national comparison    Efficient use of medical imaging national comparison
Number of Completed Surveys               Survey Response Rate Percent
Year
```

---

## 🔁 Project Pipeline

```
Raw CSV Files (2016–2020)
        │
        ▼
  Data Ingestion          ← glob multi-file merge, pd.concat
        │
        ▼
  Data Exploration        ← shape, dtypes, null check, duplicates
        │
        ▼
  Data Cleaning           ← case normalization, type casting, null handling
        │
        ▼
  EDA & Visualization     ← countplots, pivot tables, stacked bar charts
        │
        ▼
  Statistical Testing     ← T-Test, ANOVA, Paired T-Test (SciPy)
        │
        ▼
  Feature Engineering     ← LabelEncoder, SMOTE, RFE (6 features), StandardScaler
        │
        ▼
  Model Training          ← Logistic Regression | XGBoost | Random Forest + DNN
        │
        ▼
  Evaluation              ← Accuracy, Confusion Matrix, Classification Report
        │
        ▼
  Insights & Conclusions
```

---

## 📊 Exploratory Data Analysis

### 1. Patient Survey Star Rating Distribution
- A significant proportion of patients **did not submit ratings**, highlighting a critical gap in feedback collection.
- Hospitals should actively invest in post-discharge feedback systems to capture more meaningful satisfaction data.

### 2. Hospital Type Distribution
- **Acute Care Hospitals** dominate the dataset across all years.
- **Acute Care — Department of Defense** hospitals first appear in **2020**, indicating new operational activation.
- **Children's Hospitals** maintain a consistent count year-over-year.
- Acute Care hospitals saw approximately a **50% increase** in count in **2019**.

### 3. Year-over-Year Trends (2016–2020)
- State-wise patient satisfaction ratings show a **positive trajectory** over the 5-year period.
- Stacked bar charts confirm the growing volume of hospital types over time.

### 4. National Comparison Metrics
Analysis across 6 national comparison dimensions by hospital type:

| Metric | Observation |
|--------|------------|
| Mortality | Acute Care hospitals dominate all categories |
| Safety of Care | Consistent distribution across types |
| Readmission | Significant variation between hospital types |
| Patient Experience | Strong indicator of overall satisfaction |
| Effectiveness of Care | Uniform across most hospital types |
| Timeliness of Care | Higher variability — area for improvement |

---

## 🔬 Statistical Hypothesis Testing

### Test 1 — Independent Samples T-Test
**Question:** Do hospitals with emergency services have significantly higher patient satisfaction ratings than those without?

```
H₀: No significant difference between emergency and non-emergency hospitals
H₁: Emergency hospitals have significantly higher satisfaction ratings
Confidence Level: 95% (α = 0.05)
Result: Reject H₀ — Significant difference confirmed
```

### Test 2 — One-Way ANOVA
**Question:** Is there a significant difference in patient satisfaction across different hospital types?

```
H₀: No significant difference across hospital types
H₁: At least one hospital type differs significantly
Groups: Acute Care | Critical Access | Children's | Acute Care — DoD
Result: Reject H₀ — Significant difference exists across hospital types
```

### Test 3 — Paired T-Test
**Question:** Did patient satisfaction ratings significantly improve from 2016 to 2020?

```
H₀: No significant improvement from 2016 to 2020
H₁: Ratings improved significantly over the 5-year period
Matched on: Facility ID (same hospitals compared across years)
Result: Reject H₀ — Statistically significant improvement confirmed
```

---

## 🤖 Models & Results

### Data Preprocessing for ML

| Step | Method | Detail |
|------|--------|--------|
| Encoding | `LabelEncoder` | All categorical columns encoded |
| Class Imbalance | `SMOTE` | Synthetic minority oversampling on training set |
| Feature Selection | `RFE` with Random Forest | Top 6 most important features selected |
| Scaling | `StandardScaler` | Applied for Deep Neural Network |
| Split | `train_test_split` | 70/30 and 75/25 depending on model |

### Model 1 — Logistic Regression

```python
LogisticRegression(max_iter=1000, class_weight='balanced')
Train/Test Split : 70 / 30
Class Balancing  : SMOTE applied
Accuracy         : ~83%
```

### Model 2 — XGBoost Classifier

```python
XGBClassifier(n_estimators=200, learning_rate=0.1)
Train/Test Split : 75 / 25
Class Balancing  : SMOTE applied
Accuracy         : ~85%
```

### Model 3 — Random Forest + RFE Feature Selection

```python
RandomForestClassifier(n_estimators=150, random_state=42)
Feature Selection : RFE → Top 6 features
Accuracy          : ~83%
Selected Features : 6 most predictive hospital attributes
```

### Model 4 — Deep Neural Network (TensorFlow / Keras)

```python
Architecture:
  Input  → Dense(128, relu) → Dropout(0.2)
         → Dense(64,  relu)
         → Dense(32,  relu) → Dropout(0.5)
         → Dense(1,   sigmoid)

Optimizer  : Adam
Loss       : Binary Crossentropy
Epochs     : 150
Batch Size : 5,000
Input Shape: 6 features (RFE selected)
```

---

## 📈 Results Summary

| Model | Accuracy | Technique |
|-------|----------|-----------|
| Logistic Regression | ~83% | SMOTE + class_weight='balanced' |
| XGBoost | ~85% | SMOTE + 200 estimators |
| Random Forest (RFE) | ~83% | 6-feature RFE selection |
| Deep Neural Network | ~83% | 4-layer DNN, 150 epochs |

> **Best Performing Model:** XGBoost with ~85% accuracy — recommended for production deployment due to speed, interpretability, and robustness to class imbalance.

---

## 💡 Key Insights

1. **Rating Gap** — The majority of patients do not submit satisfaction ratings. Hospitals must prioritize structured post-visit feedback collection to get actionable data.

2. **Emergency Services Matter** — Hospitals offering 24/7 emergency services score **significantly higher** on patient satisfaction (confirmed at 95% confidence level).

3. **Hospital Type Drives Satisfaction** — Acute Care Hospitals and Critical Access Hospitals show meaningfully different satisfaction distributions — one-size-fits-all policy is insufficient.

4. **Positive National Trend** — Patient satisfaction across U.S. hospitals improved statistically significantly between 2016 and 2020 — indicating that CMS quality mandates are having a measurable positive effect.

5. **Top Features for Prediction** — The 6 RFE-selected features (from Random Forest) are the strongest predictors of patient satisfaction — focusing hospital improvement efforts on these variables yields the highest ROI.

6. **XGBoost Outperforms Deep Learning** — For tabular healthcare data, gradient boosting (XGBoost) matched or exceeded the deep neural network — a common and important finding in structured data ML projects.

---

## 🗂️ Project Structure

```
healthcare-patient-satisfaction/
│
├── Healthcare_Patient_Satisfaction_DL.ipynb   ← Main analysis notebook
│
├── data/
│   └── patient_satisfaction/
│       ├── 2016_hospital_data.csv
│       ├── 2017_hospital_data.csv
│       ├── 2018_hospital_data.csv
│       ├── 2019_hospital_data.csv
│       └── 2020_hospital_data.csv
│
├── outputs/
│   └── Patient Survey Star Rating.png         ← EDA visualizations
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/givishalt/healthcare-patient-satisfaction.git
cd healthcare-patient-satisfaction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the archived hospital data from CMS:
```
https://data.cms.gov/provider-data/archived-data/hospitals
```
Place all CSV files in: `data/patient_satisfaction/`

### 4. Run the Notebook

```bash
jupyter notebook Healthcare_Patient_Satisfaction_DL.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
xgboost
tensorflow
scipy
glob2
jupyter
```

Create `requirements.txt`:
```bash
pip freeze > requirements.txt
```

---

## 🔮 Future Scope

- [ ] Deploy best model (XGBoost) as a REST API using **FastAPI** or **Flask**
- [ ] Build an interactive **Streamlit dashboard** for real-time hospital satisfaction lookup by state and hospital type
- [ ] Integrate **SHAP values** for model explainability — understand which features drive individual predictions
- [ ] Apply **NLP sentiment analysis** on free-text patient feedback fields
- [ ] Extend dataset to include **post-COVID years (2021–2023)** to measure pandemic impact on satisfaction
- [ ] Implement **time-series forecasting** to predict future satisfaction trends by state

---

## 👤 Author

**Vishal**
- 📧 vishal9681032@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/vishal-793170396/)
- 🐙 [GitHub](https://github.com/givishalt)

---

## 📄 Data Source & Citation

```
Centers for Medicare & Medicaid Services (CMS)
Hospital Compare — Patient Satisfaction Data (2016–2020)
https://data.cms.gov/provider-data/archived-data/hospitals
```

---

## 📃 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If this project helped you, please give it a star on GitHub!** ⭐

*Built with ❤️ using Python, TensorFlow, XGBoost, and CMS Open Data*

</div>

