import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to evaluate and visualize the model
def evaluate_and_visualize_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    print(f"{name} Mean Squared Error: {mse}")
    print(f"{name} R-squared: {r_squared}")

    # Visualize the regression results
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"{name} - True Values vs. Predicted Values")
    plt.show()

    return predictions

def load_and_preprocess_data(file_path):
    # Read the dataset with specified encoding
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Display the first few rows of the dataset
    print(data.head())

    # Drop unnecessary columns
    data.drop(columns=['customer name', 'customer e-mail', 'country'], inplace=True)

    # Display information about the dataset
    print(data.info())
    print(data.describe())
    print(data.isna().sum())

    # Visualize the correlation between 'annual Salary' and 'credit card debt'
    plt.scatter(data['annual Salary'], data['credit card debt'])
    plt.xlabel('Annual Salary')
    plt.ylabel('Credit Card Debt')
    plt.title('Correlation B/W Credit Card Debt & Annual Salary')
    plt.show()

    # Visualize pairplot for the dataset
    sns.pairplot(data)
    plt.show()

    return data

def prepare_and_split_data(data, feature_cols, target_col, test_size=0.2, random_state=42):
    # Separate features and target label
    X = data[feature_cols]
    y = data[target_col]

    # Normalize the feature data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def fit_and_evaluate_models(regressors, X_train, X_test, y_train, y_test):
    predictions_dict = {}
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        predictions_dict[name] = evaluate_and_visualize_model(name, model, X_test, y_test)
    
    return predictions_dict

if __name__ == "__main__":
    # Dataset: https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction
    file_path = "car_purchasing.csv"
    feature_cols = ['age', 'annual Salary', 'credit card debt', 'net worth']
    target_col = 'car purchase amount'

    # Load and preprocess the data
    data = load_and_preprocess_data(file_path)

    # Prepare and split the data
    X_train, X_test, y_train, y_test =  prepare_and_split_data(data, feature_cols, target_col)

    # Define the regressors
    regressors = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(),
        'SVR': SVR()
    }

    # Fit and evaluate the models
    predictions = fit_and_evaluate_models(regressors, X_train, X_test, y_train, y_test)
