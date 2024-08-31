# Run this cell to install mlflow
!pip install mlflow


# Run this cell to import the modules you require
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
weather = pd.read_csv("dataset\london_weather.csv")

# Display the first few rows of the dataset
print(weather.head())

# Get an overview of the dataset (columns, non-null counts, data types)
print(weather.info())

# Check for missing values
print(weather.isnull().sum())
# Drop rows where the target variable 'mean_temp' is NaN
weather = weather.dropna(subset=['mean_temp'])

# Check again for missing values in the target variable
print(weather['mean_temp'].isnull().sum())


# Convert the 'date' column to datetime format
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

# Extract year and month from the 'date' column
weather['year'] = weather['date'].dt.year
weather['month'] = weather['date'].dt.month

# Drop the original 'date' column as it's no longer needed
weather = weather.drop(columns=['date'])

# Display the updated dataset structure
print(weather.head())

# Visualize the mean temperature over time
sns.lineplot(x='year', y='mean_temp', data=weather)
plt.title('Mean Temperature Over Years')
plt.show()

# Correlation heatmap
corr_matrix = weather.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Selecting features with high correlation with 'mean_temp'
features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth', 'year', 'month']
X = weather[features]
y = weather['mean_temp']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize MLflow
mlflow.set_experiment("London Weather Prediction")

# List of models to train
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10)
}

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Calculate RMSE
        rmse = mean_squared_error(y_test, predictions, squared=False)
        
        # Log the model and RMSE to MLflow
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_metric("rmse", rmse)
        
        print(f"{model_name} RMSE: {rmse}")
        
        # Query MLflow for the experiment results
experiment_results = mlflow.search_runs()

# Display the experiment results
print(experiment_results)
