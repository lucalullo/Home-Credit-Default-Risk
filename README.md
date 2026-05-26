# Home Credit Default Risk

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-6C63FF?style=for-the-badge&logo=optuna&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

This repository contains a complete data science workflow for predicting credit default risk using machine learning.

🔗 **Original Notebook on Kaggle:** [View here](https://www.kaggle.com/code/lucalullo/home-credit-default-risk)  
📓 **Notebook on GitHub:** [View here](https://github.com/lucalullo/Home-Credit-Default-Risk/blob/main/home-credit-default-risk.ipynb)

## 📈 Performance
- **Model:** XGBoost Classifier  
- **Approach:** Gradient Boosting with feature engineering, hyperparameter tuning and threshold optimization  
- **Optimization:** Optuna — 50 trials maximizing AUC-ROC  
- **Threshold:** Tuned on Precision-Recall curve with minimum Precision constraint of 20%

## 🛠️ Key Features

- **Data Cleaning:**  
  - Removal of irrelevant columns and columns with excessive missing values  
  - Handling of `DAYS_EMPLOYED = 365243` placeholder (retirees/unemployed → `NaN`)  
  - Median imputation for numerical features, mode imputation for categorical features  
  - One-Hot Encoding with `drop_first=True` to avoid multicollinearity  
  - StandardScaler fitted on train set only to prevent data leakage  

- **Class Imbalance Handling:**  
  - Dataset heavily imbalanced: ~91.9% non-default vs ~8.1% default  
  - Comparison of Class Weight, SMOTE and Undersampling on Logistic Regression  
  - Class Weight selected for best AUC-ROC without altering the data  

- **Model Comparison:**  
  - Logistic Regression (linear baseline), Extra Trees, XGBoost  
  - Extra Trees discarded due to poor probability calibration  
  - XGBoost selected as final model for its non-linear capacity after feature engineering  

- **Feature Engineering from 6 External Sources:**  
  - `bureau.csv` — credit history with other institutions  
  - `previous_application.csv` — past loan applications with Home Credit  
  - `installments_payments.csv` — payment punctuality on installments  
  - `POS_CASH_balance.csv` — active and completed POS and cash loans  
  - `credit_card_balance.csv` — credit card usage and delays  
  - `bureau_balance.csv` — monthly behavior on third-party credits  
  - Each source validated with incremental AUC-ROC before proceeding  

- **Hyperparameter Tuning:**  
  - Optuna with 50 trials on a separate validation set (test set untouched)  
  - Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`  

- **Interpretability with SHAP:**  
  - `TreeExplainer` — exact, non-approximated for XGBoost  
  - Bar chart for global feature importance  
  - Beeswarm plot for feature impact and direction  
  - Waterfall plot for individual client explanation  
  - Business report with risk factors, protective factors and operational recommendation  

## 🚀 How to use

1. Clone the repo: `git clone https://github.com/lucalullo/Home-Credit-Default-Risk.git`  
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost imbalanced-learn optuna shap matplotlib seaborn`  
3. Download the dataset from the Kaggle competition: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)  
4. Run the `home-credit-default-risk.ipynb` notebook  

---
**Author:** [Luca Lullo](https://github.com/lucalullo)  
*Data Scientist | Machine Learning Applied*
