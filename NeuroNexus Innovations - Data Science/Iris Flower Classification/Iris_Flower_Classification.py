# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Function to load the dataset from a file
def load_dataset(file_path):
    try:
        iris_data = pd.read_csv(file_path)
        return iris_data
    except FileNotFoundError:
        print("File not found. Please provide the correct file path.")
        exit()

# Function to preprocess the data
def preprocess_data(X, y):
    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split the data into training and testing sets, and perform feature scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, label_encoder

# Function to train and tune the model using SVM with grid search
def train_and_tune_model(X_train, y_train):
    # Define the parameter grid for grid search
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
    
    # Perform grid search with 5-fold cross-validation
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    grid.fit(X_train, y_train)

    print("Best hyperparameters:", grid.best_params_)
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    return best_model, grid.best_params_

# Function to evaluate different classifiers
def evaluate_classifiers(classifiers, X_test, y_test, label_encoder):
    for name, clf in classifiers.items():
        # Skip Multinomial Naive Bayes due to non-negative feature requirement
        if name == "Multinomial Naive Bayes":
            print("Skipping Multinomial Naive Bayes due to non-negative feature requirement")
            continue
        
        # Fit the classifier, make predictions, and calculate evaluation metrics
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        # Print accuracy and classification report
        print(f"\nClassifier: {name}")
        print(f"Accuracy: {accuracy}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        # Plot confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

# Main code
# Dataset: https://www.kaggle.com/datasets/arshid/iris-flower-dataset
file_path = 'IRIS.csv'
iris_data = load_dataset(file_path)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Preprocess the data
X_train, X_test, y_train, y_test, label_encoder = preprocess_data(X, y)

# Train and tune the model using SVM with grid search
best_model, best_params = train_and_tune_model(X_train, y_train)

# Define classifiers for evaluation
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree (Gini)": DecisionTreeClassifier(criterion="gini"),
    "Decision Tree (Entropy)": DecisionTreeClassifier(criterion="entropy"),
    "KNN": KNeighborsClassifier(),
    "Support Vector Machine": SVC(**best_params)  # Use best hyperparameters for SVM
}

# Evaluate the classifiers
evaluate_classifiers(classifiers, X_test, y_test, label_encoder)
