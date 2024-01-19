import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import classification_report, confusion_matrix#, RocCurveDisplay
#import scikitplot as skplt
# import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def normalize_data(data):
    scaler = StandardScaler()
    data['normalized_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    return data

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, name):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Classifier: {name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Use the probability estimates for plotting precision-recall curve
    # y_probas = classifier.predict_proba(X_test)
    # skplt.metrics.plot_precision_recall(y_test, y_probas, title='Precision-Recall curve for ' + name)

    # plt.show()
    return classifier

"""
def plot_roc_for_all(classifiers, X_test, y_test):
    plt.figure()
    for name, classifier in classifiers.items():
        skplt.metrics.plot_roc(classifier, X_test, y_test, title=f'ROC curve for {name}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.show()
"""

def main():
    # Load the dataset
    # Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    file_path = 'creditcard.csv'
    data = load_data(file_path)

    # Normalization
    data = normalize_data(data)

    # Dealing with Class Imbalance
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers with hyperparameters
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree (Gini)': DecisionTreeClassifier(criterion="gini", max_depth=5),
        'Decision Tree (Entropy)': DecisionTreeClassifier(criterion="entropy", max_depth=5),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True)
    }

    # Iterate over each classifier and evaluate
    trained_classifiers = {}
    for name, classifier in classifiers.items():
        trained_classifiers[name] = evaluate_classifier(classifier, X_train, y_train, X_test, y_test, name)

    # Plotting ROC curve for all classifiers
    # plot_roc_for_all(trained_classifiers, X_test, y_test)

if __name__ == "__main__":
    main()
