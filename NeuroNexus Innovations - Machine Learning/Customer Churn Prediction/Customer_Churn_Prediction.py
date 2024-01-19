# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Step 1: Data Exploration
# Load the dataset
# Dataset: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
data = pd.read_csv('Churn_Modelling.csv')

# Exploring the data
print('\n', data.head())
print('\n', data.info())
print('\n', data.describe())

# Step 2: Feature Engineering (if required)
# Identify important features that could potentially influence customer churn
# Add feature engineering steps if necessary based on the dataset and problem at hand

# Step 3: Data Preprocessing
X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])  # Features
y = data['Exited']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Building
# Using Random Forest, Logistic Regression, and Gradient Boosting classifiers
model_rf = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']),
        ('cat', OneHotEncoder(), ['Geography', 'Gender'])
    ])),
    ('classifier', RandomForestClassifier())
])

model_lr = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']),
        ('cat', OneHotEncoder(), ['Geography', 'Gender'])
    ])),
    ('classifier', LogisticRegression())
])

model_gb = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']),
        ('cat', OneHotEncoder(), ['Geography', 'Gender'])
    ])),
    ('classifier', GradientBoostingClassifier())
])


# Step 5: Hyperparameter Tuning for Random Forest Classifier
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None]
}
grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='accuracy').fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
print("\nBest Hyperparameters for Random Forest Classifier:", best_params_rf)

# Step 6: Model Evaluation for Random Forest Classifier
best_model_rf = grid_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)
metrics_rf = {
    'Accuracy RF': accuracy_score,
    'Precision RF': precision_score,
    'Recall RF': recall_score,
    'F1 Score RF': f1_score,
    'ROC AUC RF': roc_auc_score
}
results_rf = {metric: score(y_test, y_pred_rf) for metric, score in metrics_rf.items()}
print("\n", results_rf)

# Step 5: Hyperparameter Tuning for Logistic Regression
param_grid_lr = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=5, scoring='accuracy').fit(X_train, y_train)
best_params_lr = grid_search_lr.best_params_
print("\nBest Hyperparameters for Logistic Regression:", best_params_lr)

# Step 6: Model Evaluation for Logistic Regression
best_model_lr = grid_search_lr.best_estimator_
y_pred_lr = best_model_lr.predict(X_test)
metrics_lr = {
    'Accuracy LR': accuracy_score,
    'Precision LR': precision_score,
    'Recall LR': recall_score,
    'F1 Score LR': f1_score,
    'ROC AUC LR': roc_auc_score
}
results_lr = {metric: score(y_test, y_pred_lr) for metric, score in metrics_lr.items()}
print("\n", results_lr)

# Step 5: Hyperparameter Tuning for Gradient Boosting Classifier
param_grid_gb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5]
}
grid_search_gb = GridSearchCV(model_gb, param_grid_gb, cv=5, scoring='accuracy').fit(X_train, y_train)
best_params_gb = grid_search_gb.best_params_
print("\nBest Hyperparameters for Gradient Boosting Classifier:", best_params_gb)

# Step 6: Model Evaluation for Gradient Boosting Classifier
best_model_gb = grid_search_gb.best_estimator_
y_pred_gb = best_model_gb.predict(X_test)
metrics_gb = {
    'Accuracy GB': accuracy_score,
    'Precision GB': precision_score,
    'Recall GB': recall_score,
    'F1 Score GB': f1_score,
    'ROC AUC GB': roc_auc_score
}
results_gb = {metric: score(y_test, y_pred_gb) for metric, score in metrics_gb.items()}
print("\n", results_gb)

# Step 7: Deployment
# Deploy the best model to make predictions on new customer data
# Include deployment steps as per your specific deployment requirements
