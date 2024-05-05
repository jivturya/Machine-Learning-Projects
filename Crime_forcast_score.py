from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from app import crime_scores
import numpy as np

DATABASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(DATABASE_DIR, "chicago_crime.db")

def fetch_crime_data():
    conn = sqlite3.connect(DATABASE_PATH)
    query = "SELECT DATE(Date) as Date, Primary_Type, Postcode, Population FROM Crime WHERE Population>0"
    data = pd.read_sql_query(query, conn, parse_dates=['Date'])
    conn.close()
    return data

def calculate_crime_scores(data):
    data['Crime_Score'] = data['Primary_Type'].map(crime_scores).fillna(0)
    monthly_scores = data.groupby(['Postcode', pd.Grouper(key='Date', freq='M')]).agg({'Crime_Score': 'sum', 'Population': 'mean'})
    monthly_scores['Score_Per_Capita'] = monthly_scores['Crime_Score'] / monthly_scores['Population']
    return monthly_scores.reset_index()

# Adding Lag Features
def add_lag_features(data, num_lags=1):
    for lag in range(1, num_lags + 1):
        data[f'lag_{lag}'] = data.groupby('Postcode')['Score_Per_Capita'].shift(lag)
    return data.dropna()

# Fetch and prepare data
data = fetch_crime_data()
prepared_data = calculate_crime_scores(data)

# Feature Engineering
prepared_data['Month'] = prepared_data['Date'].dt.month
prepared_data['Year'] = prepared_data['Date'].dt.year

# Adding lag features
prepared_data = add_lag_features(prepared_data, num_lags=1)

# Splitting Data for Training and Testing
# Use the last part of the data as the test set
train_size = int(len(prepared_data) * 0.8)
train_data = prepared_data.iloc[:train_size]
test_data = prepared_data.iloc[train_size:]

X_train = train_data[['Postcode', 'Month', 'Year', 'lag_1']]
y_train = train_data['Score_Per_Capita']
X_test = test_data[['Postcode', 'Month', 'Year', 'lag_1']]
y_test = test_data['Score_Per_Capita']

# Model Training
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", mean_squared_error(y_test, predictions, squared=False))
print("R-squared:", r2_score(y_test, predictions))

# Hyperparameter Tuning (optional, can be time-consuming)
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10]
# }
# grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# best_rf_model = grid_search.best_estimator_

# Save the model
joblib.dump(model, 'crime_forecast_model.pkl')

# Example for making a prediction
sample_postcode = 60601
sample_month = 12
sample_year = 2024
sample_input = np.array([[sample_postcode, sample_month, sample_year, 0.488831664834211]]) # Include lag values here
predict_point = model.predict(sample_input)
print(f"Predicted Crime Score for Postcode {sample_postcode}: {predict_point[0]}")
