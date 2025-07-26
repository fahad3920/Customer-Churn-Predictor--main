# Customer Churn Predictor

A machine learning project designed to predict whether a customer is likely to churn based on various demographic, account, and usage-related features. The solution uses data preprocessing, feature engineering, and ensemble modeling techniques to achieve high accuracy.

## Project Overview

Customer churn is when existing customers stop doing business with a company. Predicting churn helps businesses take proactive steps to retain valuable customers. In this project, we build a predictive model using historical customer data.

## Dataset

The dataset contains various features:
- 'RowNumber',
- 'CustomerId',
- 'Surname',
- 'CreditScore',
- 'Geography',
- 'Gender',
- 'Age',
- 'Tenure',
- 'Balance',
- 'NumOfProducts',
- 'HasCrCard',
- 'IsActiveMember',
- 'EstimatedSalary',
- 'Exited'(Target Binary 1(Will Churn) 0(NO))

## 🔧 Project Workflow

1. **EDA Using Visulaization**

2. **Feature Engineering**
   - one-hot encoding
   - Feature scaling using StandardScaler

3. **Model Building**
   - Models used:
     - Logistic Regression
     - SVM
     - Extra Trees Classifier
     - Decision Tree
     - Random Forest
     - XGBoost
     - Gradient Boosting
     - ADABoost
     - **Voting Classifier** (final model)
   - Hyperparameter tuning with GridSearchCV

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

5. **Model Serialization**
   - Saving the trained model and scaler using `pickle` for later use

## Final Model

The best performance was achieved using a **Voting Classifier** combining:
- Random Forest
- Gradient Boosting
- Extra Trees Classifier
- DecisionTreeClassifier
- XGBClassifier


## Technical Stack  
- **Python Libraries**: Pandas, Scikit-learn, XGBoost, Imbalanced-learn  
- **Visualization**: Seaborn , Matplolib

