# Import required libraries
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc

# Load the training dataset and drop duplicate entries
def load_and_process_data(file_path):
    df = pd.read_csv(file_path).drop_duplicates()
    return df

# Feature Engineering
def add_time_related_features(df):
    # Calculate age based on date of birth
    df['age'] = datetime.now().year - pd.to_datetime(df['dob']).dt.year
    # Extract hour, day of the week, and month from transaction date and time
    df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
    df['day'] = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['trans_date_trans_time']).dt.month
    return df

# Model Training and Testing
def prepare_train_test_data(df, target_column):
    # Prepare features and target variable for model training
    train = df[['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'age', 'hour', 'day', 'month', target_column]]
    # Convert categorical 'category' column to dummy variables
    train = pd.get_dummies(train, columns=['category'], drop_first=True)
    # Split the data into training and testing sets
    y_train = train[target_column].values
    X_train = train.drop(target_column, axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function for model training and testing
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model and make predictions
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # Print classification report and confusion matrix
    print(model_name + ' - Classification Report:\n', classification_report(y_test, predicted))
    conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
    print(model_name + ' - Confusion Matrix:\n', conf_mat)
    print('\nShare of Non-Fraud in Test Data:', 1 - round(y_test.sum() / len(y_test), 4))

    # Perform cross-validation and calculate precision, recall, and F1-score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-Validation Scores for " + model_name + ":", cv_scores)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted, average='binary')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", fscore)

    # Plot ROC curve and calculate AUC
    fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = model_name + ' (AUC = %0.2f)' % roc_auc)

# Create models and evaluate them
def evaluate_models(models, X_train, X_test, y_train, y_test):
    # Initialize a plot for ROC curve
    plt.figure(figsize=(10, 8))
    # Iterate through each model, train and evaluate it
    for model_name, model in models.items():
        train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test)
    
    # Plot the ROC curve and display the graph
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Main code execution
file_path = 'fraudTrain.csv'
df = load_and_process_data(file_path)
df = add_time_related_features(df)

target_column = 'is_fraud'
X_train, X_test, y_train, y_test = prepare_train_test_data(df, target_column)

# Define models and evaluate them using the prepared data
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=5),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree (Gini)': DecisionTreeClassifier(criterion='gini', random_state=5),
    'Decision Tree (Entropy)': DecisionTreeClassifier(criterion='entropy', random_state=5),
    'Support Vector Machine': SVC(probability=True, random_state=5)
}

evaluate_models(models, X_train, X_test, y_train, y_test)
