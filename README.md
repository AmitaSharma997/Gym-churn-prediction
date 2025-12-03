# 🏋️‍♂️ Gym Customer Churn Prediction  
A complete end-to-end machine learning project predicting customer churn for a fitness center using Python, cross-validation, and model comparison.

---

## 📌 Project Overview
This project aims to predict the likelihood of a customer churning (canceling their gym membership).  
It includes data cleaning, exploratory data analysis, modeling, threshold tuning, and final business recommendations.

Churn prediction helps gyms:
- Identify at-risk customers early  
- Focus retention efforts efficiently  
- Improve customer satisfaction  
- Reduce revenue loss  

---

## 📂 Dataset
**Source**: Local CSV file `gym_churn_us.csv`  
**Rows**: (approx.) 4000  
**Target variable**: `Churn` (0 = stay, 1 = churn)

### Key Features:
- `Age`
- `Contract_period`
- `Avg_class_frequency_total`
- `Avg_class_frequency_current_month`
- `Month_to_end_contract`
- `Lifetime`
- `Avg_additional_charges_total`
- Categorical variables: Gender, Partner, Near_Location, etc.

---

## 🧹 1. Data Preparation
- Loaded the data into pandas  
- Checked missing values  
- Converted types where needed  
- Visualized distributions & outliers  
- Explored correlations & feature interactions  
- Used **Stratified Train-Test Split** to preserve class balance

---

## 📊 2. Exploratory Data Analysis (EDA)
Highlights:
- Customers with **low class frequency** are more likely to churn.  
- Short **contract periods** strongly correlate with churn.  
- Higher **additional charges** appear linked to churn.  
- Demographic features like **gender** have minimal effect.

Visualizations included:
- Boxplots  
- Histograms  
- Correlation heatmap  
- Count plots for categorical features  

---

## 🤖 3. Modeling Approach
### Models Trained:
- **Random Forest**  
- **Logistic Regression**  
- **Gradient Boosting Classifier**  

Each model was built using a **Pipeline** (scaling + estimator) and evaluated using:
- **5-fold Stratified Cross-Validation**
- Metrics:
  - ROC-AUC  
  - Average Precision  
  - F1-score  
  - Precision  
  - Recall  

---

## ⚙️ 4. Threshold Tuning
After training the models, probability outputs were evaluated using:
- Precision-Recall Curve  
- Custom threshold tuning to maximize **F1-score** or meet specific business constraints.

Example:
- A lower threshold improves recall (catch more churners).
- A higher threshold increases precision (reduce false alarms).

---

## 🏆 5. Model Comparison
Using cross-validation:

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| Random Forest | ~0.96 | High | High | High |
| Gradient Boosting | Competitive | Moderate | Moderate | Moderate |
| Logistic Regression | Baseline | Lower | Lower | Moderate |

**Best performing model:**  
✔ **Random Forest** (after threshold tuning)

---

## 🎯 6. Final Evaluation (Test Set)
Using the selected Random Forest model:

- **ROC-AUC:** ~0.96  
- **Precision (churn):** High  
- **Recall (churn):** High  
- **Confusion Matrix:** Low false positives + low false negatives  

---

## 🧠 7. Business Recommendations
From the model and feature insights:

- Target customers with declining class frequency  
- Offer incentives to users with short contract periods  
- Encourage longer-term memberships  
- Create programs to increase customer engagement  
- Monitor customers nearing contract end  

---

## 🛠 8. Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📁 9. Project Structure
gym-churn-prediction/
│
├── data/
│ └── gym_churn_us.csv
│
├── notebooks/
│ └── churn_modeling.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ └── evaluation.py
│
├── models/
│ └── best_pipeline_churn.pkl
│
└── README.md

## 10. Final Model Export
The trained best-performing model is saved as:


Amita Sharma  
Machine Learning / Data Analysis Enthusiast  
