import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Step 1: Data Preprocessing
    # Load the dataset
    train_data = pd.read_csv('train_data.txt', sep=' ::: ', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python').dropna(subset=['DESCRIPTION']).drop_duplicates(subset=['DESCRIPTION']).reset_index(drop=True)
    test_data = pd.read_csv('test_data.txt', sep=' ::: ', names=['ID', 'TITLE', 'DESCRIPTION'], engine='python').reset_index(drop=True)

    # Print summary information about the datasets
    for data, name in zip([train_data, test_data], ['Train Data', 'Test Data']):
        print(f"\n{name} Head:\n{data.head()}")
        print(f"\nDescription of {name}:\n{data.describe()}")
        print(f"\nInfo of {name}:\n{data.info()}")
        print(f"\nNull Values of {name}:\n{data.isnull().sum()}")

    # Visualize genre distribution in the training data
    fig, axes = plt.subplots(2, 1, figsize=(14, 14))
    sns.countplot(data=train_data, y='GENRE', order=train_data['GENRE'].value_counts().index, palette='viridis', ax=axes[0])
    axes[0].set_xlabel('COUNT', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('GENRE', fontsize=14, fontweight='bold')

    counts = train_data['GENRE'].value_counts()
    sns.barplot(x=counts.index, y=counts, palette='viridis', ax=axes[1])
    axes[1].set_xlabel('GENRE', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('COUNT', fontsize=14, fontweight='bold')
    axes[1].set_title('Distribution of Genres', fontsize=16, fontweight='bold')
    axes[1].tick_params(axis='x', labelrotation=90)
    plt.show()

    # Add Length column to DataFrames
    for data, name in zip([train_data, test_data], ['Train Data', 'Test Data']):
        data['Length'] = data['DESCRIPTION'].apply(len)
        print(f"\nLength of {name}'s Description:\n{data.head()}")

    # Visualize length distribution in the training data
    plt.figure(figsize=(8, 7))
    sns.histplot(data=train_data, x='Length', bins=20, kde=True, color='blue')
    plt.xlabel('Length', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title('Distribution of Lengths', fontsize=16, fontweight='bold')
    plt.show()

    # Tokenize descriptions and remove stopwords using TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(train_data['DESCRIPTION'])
    y = train_data['GENRE']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Model Building with Hyperparameter Tuning
    # Initialize and train the classifiers with hyperparameter tuning
    clf = GridSearchCV(MultinomialNB(), param_grid={'alpha': [0.01, 0.1, 1.0, 10.0]}, scoring='accuracy', cv=5)
    logistic_clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid={'C': [0.001, 0.01, 0.1, 1.0, 10.0]}, scoring='accuracy', cv=5)

    classifiers = {
        "Multinomial Naive Bayes": clf,
        "Logistic Regression": logistic_clf
    }

    # Step 3: Model Training and Evaluation with Ensemble
    # Fit the tuned classifiers and evaluate with proper metrics
    for name, model in classifiers.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=1)  # Add zero_division parameter
            logging.info("%s - Accuracy: %.4f", name, accuracy)
            logging.info("%s - Classification Report:\n%s", name, report)
        except Exception as e:
            logging.error("Training and evaluation of %s model failed: %s", name, str(e))

    # Create a voting classifier for ensembling
    voting_clf = VotingClassifier(estimators=[('nb', clf), ('lr', logistic_clf)], voting='hard')
    voting_clf.fit(X_train, y_train)

    # Evaluate the voting classifier
    try:
        y_pred_voting = voting_clf.predict(X_test)
        accuracy_voting = accuracy_score(y_test, y_pred_voting)
        report_voting = classification_report(y_test, y_pred_voting, zero_division=1)  # Add zero_division parameter
        logging.info("Voting Classifier - Accuracy: %.4f", accuracy_voting)
        logging.info("Voting Classifier - Classification Report:\n%s", report_voting)
    except Exception as e:
        logging.error("Evaluation of Voting Classifier model failed: %s", str(e))

    # Step 4: Prediction and Output
    try:
        # Make predictions using all classifiers on test data
        X_test_data = tfidf_vectorizer.transform(test_data['DESCRIPTION'])
        predictions = {
            "Multinomial Naive Bayes": clf.predict(X_test_data),
            "Logistic Regression": logistic_clf.predict(X_test_data),
            "Voting Classifier": voting_clf.predict(X_test_data)
        }

        # Output the predicted genres along with IDs and titles for all classifiers
        output_df = pd.DataFrame({
            'ID': test_data['ID'],
            'TITLE': test_data['TITLE'],
            **{f'PREDICTED_GENRE_{name}': prediction for name, prediction in predictions.items()}
        })
        # Instead of directly printing, you can write to a file or return from a function
        print(output_df)
    except Exception as e:
        logging.error("Prediction and output step failed: %s", str(e))

except FileNotFoundError as e:
    logging.error("File not found: %s", str(e))

except pd.errors.ParserError as e:
    logging.error("Error parsing data: %s", str(e))

except Exception as e:
    logging.error("Error processing data: %s", str(e))
