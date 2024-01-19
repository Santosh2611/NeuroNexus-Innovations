import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Step 1: Data Preprocessing
# Dataset: https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies
file_path = 'IMDb Movies India.csv'

# Load the data and handle non-numeric columns during data loading
movie_data = pd.read_csv(file_path, encoding='latin1', na_values=['N/A', 'NA', 'NaN'])

# Extract numeric values from 'Year' and 'Duration' columns
movie_data['Year'] = movie_data['Year'].str.extract('(\d+)').astype(float)
movie_data['Duration'] = movie_data['Duration'].str.extract('(\d+)').astype(float)

# Handle missing values using SimpleImputer for numerical columns
numerical_cols = ['Year', 'Duration']
imputer = SimpleImputer(strategy='mean')
movie_data[numerical_cols] = imputer.fit_transform(movie_data[numerical_cols])

# Define the feature matrix (X) and the target variable (y)
X = movie_data[['Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = movie_data['Rating'].astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use ColumnTransformer to apply OneHotEncoder to the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])
    ],
    remainder='passthrough'
)

# Fit and transform the preprocessor on the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Transform the testing data using the preprocessor fitted on the training data
X_test_preprocessed = preprocessor.transform(X_test)

# Step 2: Modeling
# Initialize and fit the models
models = {
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor()
}

# Results dictionary to store model evaluation metrics
results = {}

# Handle missing values in the target variable using SimpleImputer
y_imputer = SimpleImputer(strategy='mean')
y_train_imputed = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_imputed = y_imputer.transform(y_test.values.reshape(-1, 1)).flatten()

# Use the imputed target variable for modeling
for model_name, model in models.items():
    model.fit(X_train_preprocessed, y_train_imputed)
    y_pred = model.predict(X_test_preprocessed)
    
    mse = mean_squared_error(y_test_imputed, y_pred)

    # Store predictions and other relevant information for further analysis
    results[model_name] = {
        'mean_squared_error': mse,
        'predictions': y_pred
        # Additional information can be added here for comprehensive analysis
    }

# Print results for each model
for model_name, result in results.items():
    print(f"Results for {model_name}:")
    print("Mean Squared Error:", result['mean_squared_error'])
    print("\n")
