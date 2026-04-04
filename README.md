# 🏥 Healthcare Patient Satisfaction Prediction

## 📌 Overview

This project analyzes healthcare patient survey data to understand and predict satisfaction levels. It demonstrates a complete machine learning workflow — from raw data ingestion to model evaluation — using real-world structured data.

---

## 🎯 Problem Statement

Patient satisfaction is a critical metric in healthcare quality assessment. However, deriving insights from large-scale survey data is complex.

This project aims to:

* Analyze patient satisfaction patterns
* Identify key influencing factors
* Build predictive models for satisfaction classification

---

## 📊 Dataset

* Healthcare patient survey dataset (Kaggle)
* Includes:

  * Facility details (ID, name, location)
  * Survey responses and descriptions
  * Patient satisfaction ratings
  * Number of completed surveys

---

## 🧠 Approach

### 🔹 Data Ingestion

* Loaded multiple CSV files using `glob`
* Merged into a single structured dataset

### 🔹 Data Preprocessing

* Handled missing values
* Converted data types
* Encoded categorical features
* Prepared clean input for models

### 🔹 Exploratory Data Analysis (EDA)

* Studied distribution of satisfaction ratings
* Identified trends across facilities and responses
* Visualized relationships using plots

### 🔹 Model Building

Implemented and tested:

* Logistic Regression
* Support Vector Machine (SVM)

### 🔹 Evaluation

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score

---

## 📈 Key Insights

* Model performance is highly dependent on preprocessing
* Feature scaling significantly impacts SVM results
* Dataset characteristics (categorical variables, imbalance) affect prediction quality

---

## ⚠️ Challenges

* Combining multiple data sources
* Handling mixed data types (categorical + numerical)
* Debugging identical model outputs
* Ensuring correct preprocessing pipeline

---

## 📁 Project Structure

```id="g9z0d8"
├── data/                                # Raw dataset files
├── Healthcare_Patient_Satisfaction.ipynb
└── README.md
```

---

## 🔮 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Add advanced models (XGBoost, Random Forest)
* Handle class imbalance using SMOTE
* Convert project into a deployable web application

---

## 💡 What This Project Demonstrates

* Strong understanding of data preprocessing
* Practical application of machine learning models
* Ability to work with real-world datasets
* End-to-end problem-solving approach

---

## 👨‍💻 Author

Vishal

---

