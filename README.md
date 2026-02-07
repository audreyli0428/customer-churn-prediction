# Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn in subscription-based businesses using machine learning techniques.
Customer churn is formulated as a binary classification problem, where the goal is to identify customers who are likely
to discontinue their subscription based on demographic, service usage, and contract-related features.

The project follows a complete end-to-end data science workflow, including exploratory data analysis (EDA),
data preprocessing, class imbalance handling, model training, and evaluation.

---

## Dataset

The project uses the **Telco Customer Churn** dataset, which contains customer-level information such as:
- Demographics (e.g. gender, senior citizen, dependents)
- Service usage (e.g. internet service, streaming services)
- Contract and billing information (e.g. contract type, payment method, monthly charges)
- Churn label (Yes / No)

The dataset exhibits a clear class imbalance, with churned customers representing a minority of the population.

> **Note**: Raw data files are excluded from version control and stored locally under `data/raw/`.

---

## Exploratory Data Analysis (EDA)

The exploratory analysis highlights several important patterns related to customer churn:
\* Customers with shorter tenure are significantly more likely to churn.
\* Month-to-month contracts exhibit the highest churn rates compared to one-year and two-year contracts.
\* Churned customers tend to have higher monthly charges, suggesting price sensitivity.
These findings indicate that churn behavior is influenced by a combination of tenure, contract commitment, and pricing factors, motivating the use of supervised machine learning models.


---


## Methodology

The churn prediction task is addressed using the following workflow:
1. Data cleaning and type correction
2. Categorical feature encoding and numerical feature scaling
3. Train-test split with stratification
4. Handling class imbalance using SMOTE (applied only on the training set)
5. Model training and comparison
6. Evaluation using metrics suitable for imbalanced classification

---

## Models
Three classification models are evaluated:
\* Logistic Regression – interpretable linear baseline
\* Decision Tree – non-linear model capturing feature interactions
\* Random Forest – ensemble method reducing variance and improving generalization

---


## Model Performance

The performance of three classification models was evaluated using ROC-AUC on a held-out test set.

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | 0.833 |
| Random Forest | 0.811 |
| Decision Tree | 0.683 |

Logistic Regression achieved the highest ROC-AUC, indicating strong discriminatory
power despite its linear formulation. This suggests that customer churn in this
dataset is largely driven by additive effects of tenure, contract type, and pricing
features. Random Forest provided competitive performance by capturing non-linear
patterns, while Decision Trees showed weaker generalization due to high variance.

---

## Tools \\\& Libraries
\* Python
\* Pandas, NumPy
\* Scikit-learn
\* Imbalanced-learn (SMOTE)
\* Matplotlib, Seaborn


---

## Project Structure

```text
customer-churn-prediction/
├─ data/
│  ├─ raw/              # raw dataset (ignored by git)
│  └─ processed/
├─ notebooks/
│  ├─ 01_eda.ipynb      # exploratory data analysis
│  └─ 02_modeling.ipynb # preprocessing, modeling, evaluation
├─ src/
│  ├─ data_prep.py
│  ├─ train.py
│  └─ evaluate.py
├─ reports/
│  └─ figures/          
├─ README.md
└─ requirements.txt
```
---

## Author
Hao Ju Li (Audrey)
MSc in Data Science \\\& Business Analytics
ESSEC Business School \\\& CentraleSupélec

---


## Acknowledgements
The dataset used in this project is the **Telco Customer Churn** dataset, publicly available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
We thank the original data contributors and Kaggle community for making this dataset accessible for research and educational purposes.

