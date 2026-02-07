# Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn in subscription-based businesses
using machine learning techniques. Customer churn is formulated as a binary
classification problem, where the objective is to identify customers who are likely
to discontinue their subscription based on demographic, service usage, and
contract-related features.

Rather than treating churn prediction as a purely modeling task, this project adopts
an evidence-based data science workflow, explicitly linking exploratory data analysis
(EDA) to modeling decisions and evaluation strategy.

---

## Dataset

The project uses the **Telco Customer Churn** dataset, which contains customer-level
information such as:
- Demographics (e.g., gender, senior citizen, dependents)
- Service usage (e.g., internet service, streaming services)
- Contract and billing information (e.g., contract type, payment method, monthly charges)
- Churn label (Yes / No)

The dataset exhibits a clear class imbalance, with churned customers representing a
minority of the population.

> **Note**: Raw data files are excluded from version control and stored locally under
> `data/raw/`.

---

## Exploratory Data Analysis (EDA)

EDA is conducted using a structured, evidence-driven approach rather than ad-hoc
visualization.

The analysis begins with a global overview of numerical features to examine scale,
distribution, and correlations. Univariate statistical feature screening is then
performed to assess the association between each feature and churn:
- Numerical features are evaluated using the Mann–Whitney U test.
- Categorical features are evaluated using the chi-square test.

Based on this screening, only features with strong statistical evidence are selected
for deeper exploratory analysis. Key findings include:
- Customers with shorter tenure are significantly more likely to churn.
- Month-to-month contracts exhibit substantially higher churn rates compared to
  one-year and two-year contracts.
- Churned customers tend to have higher monthly charges, indicating price sensitivity.
- Customers without value-added internet services (e.g., online security or technical
  support) show increased churn risk.

A detailed EDA summary and supporting visualizations are provided in `01_eda.ipynb`.

---

## Methodology

The churn prediction task is addressed using the following workflow:
1. Data cleaning and type correction
2. Categorical feature encoding and numerical feature scaling
3. Train–test split with stratification
4. Handling class imbalance using SMOTE (applied only on the training set)
5. Model training and comparison
6. Evaluation using metrics suitable for imbalanced classification

---

## Models

Three classification models are evaluated:
- **Logistic Regression** – interpretable linear baseline
- **Decision Tree** – non-linear model capturing feature interactions
- **Random Forest** – ensemble method reducing variance and improving generalization

---
## Model Performance and Selection

Model performance is evaluated using multiple metrics, including precision, recall,
F1-score, accuracy, and ROC-AUC, to provide a comprehensive assessment under class
imbalance conditions. In the context of churn prediction, recall and ROC-AUC are
particularly emphasized, as failing to identify churned customers is typically more
costly than false positives.

### Model Comparison

| Model               | Precision | Recall | F1-score | Accuracy | ROC-AUC |
|---------------------|-----------|--------|----------|----------|---------|
| Logistic Regression | 0.50      | 0.78   | 0.61     | 0.73     | 0.83    |
| Decision Tree       | 0.50      | 0.57   | 0.53     | 0.73     | 0.68    |
| Random Forest       | 0.58      | 0.57   | 0.57     | 0.77     | 0.81    |

Logistic Regression achieves the highest recall and ROC-AUC, indicating strong
overall ranking ability and robustness in identifying churned customers. While
Random Forest attains slightly higher accuracy and precision by capturing non-linear
patterns, it underperforms Logistic Regression in recall and F1-score. Decision Trees
show weaker generalization, likely due to high variance.

Given the business objective of churn prevention—where missing a churned customer
(false negative) is typically more costly than contacting a non-churned customer
(false positive)—Logistic Regression is selected as the final model due to its
favorable trade-off between predictive performance and interpretability.


---

## Business Insights

Beyond predictive performance, the analysis provides several actionable insights
relevant to business decision-making:

- **Early-stage customers are at higher risk**: Customers with shorter tenure are
  significantly more likely to churn, suggesting that retention efforts should focus
  on the early stages of the customer lifecycle.

- **Contract structure strongly influences retention**: Month-to-month contracts
  exhibit substantially higher churn rates compared to longer-term contracts. This
  indicates that incentivizing customers to switch to longer-term contracts may
  effectively reduce churn.

- **Pricing and perceived value matter**: Churned customers tend to have higher
  monthly charges, and customers without value-added services (e.g., online security
  or technical support) show higher churn risk. This suggests that bundling services
  or offering targeted discounts could improve customer retention.

- **Model output can support targeted interventions**: The selected Logistic
  Regression model prioritizes recall, enabling the business to identify a larger
  proportion of at-risk customers. This allows marketing or customer success teams
  to design proactive retention campaigns while accepting a manageable number of
  false positives.

These insights demonstrate how predictive modeling can be translated into concrete
retention strategies, aligning analytical results with business objectives.


---

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn (modeling and evaluation)
- Imbalanced-learn (SMOTE for class imbalance handling)
- Matplotlib, Seaborn (visualization)

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



