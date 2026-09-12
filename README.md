# Explainable Machine Learning for Student Performance Prediction and Early Risk Detection

## 📌 Overview
Educational institutions often struggle to identify academically at-risk students early enough to provide timely intervention. Traditional monitoring methods rely heavily on mid-term or final grades, which leaves little time for corrective action. 

This project implements a machine learning pipeline using early academic metrics and behavioral indicators (such as past performance, study time, failures, and absences) to accurately predict student risk levels early in the academic cycle.

---

## 🎯 Project Objectives
- Build a binary classification pipeline to identify at-risk students (`1` = Safe/Pass, `0` = At-Risk).
- Compare different Machine Learning classifiers (Logistic Regression, Decision Tree, Random Forest).
- Provide interpretable insights into key factors influencing academic success.

---

## 📊 Dataset & Features
The model utilizes student academic and behavioral indicators:
- **`studytime`**: Weekly study hours
- **`failures`**: Number of past class failures
- **`absences`**: Number of school absences
- **`G1`, `G2`**: First and second period grades
- **`famrel`, `freetime`, `goout`**: Behavioral and social factors

---

## ⚡ Machine Learning Workflow
1. **Data Preprocessing**: Feature selection and binary target transformation (`G3 >= 10` for Pass).
2. **Train/Test Split**: 80% training set and 20% test set.
3. **Model Evaluation**: Metrics include Accuracy Score, Precision, Recall, and F1-score across evaluated algorithms.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Hafiza-Ayesha7/student-performance-risk-ml.git](https://github.com/Hafiza-Ayesha7/student-performance-risk-ml.git)