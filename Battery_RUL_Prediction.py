# -*- coding: utf-8 -*-
"""
@author: Chinmay Pathare
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from PyEMD import EMD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, Input
import seaborn as sns

def read_data(file_path):
    data = pd.read_csv(file_path)
    return pd.DataFrame(data)

def clean_data(df):
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
    rows_with_NaT = df[df['date'].isnull()]
    row_indices_with_NaT = rows_with_NaT.index
    df = df.drop(row_indices_with_NaT)
    df.reset_index(drop=True, inplace=True)
    
    # Create a new column 'cycle' and initialize it with NaN values
    df['cycle'] = pd.Series(dtype='float64')

# Iterate over the rows of the DataFrame
    prev_cycle = 0
    for index, row in df.iterrows():
        # Check if the value in the 'type' column is 'C'
        if row['type'] == 'C':
            # Update the 'cycle' column with the previous value plus 1
            prev_cycle += 1
            df.at[index, 'cycle'] = prev_cycle
    
    # Fill in the NaN values in the 'cycle' column with intermediate values using linear interpolation
    df['cycle'] = df['cycle'].interpolate(method='linear')
        
    return df

def calculate_capacity(df):
    df['capacity'] = (df['relativeTime'] * df['current']) / 3600
    return df

def calculate_cumulative_charge(df):
    df['Cumulative_charge'] = 0
    
    for index, row in df.iterrows():
        if index == 0:
            df.at[index, 'Cumulative_charge'] = abs(row['relativeTime'] * row['current'] / 3600)
        else:
            if row['type'] == 'C':
                df.at[index, 'Cumulative_charge'] = df.at[index - 1, 'Cumulative_charge'] + abs(row['relativeTime'] * row['current'] / 3600)
            else:
                df.at[index, 'Cumulative_charge'] = df.at[index - 1, 'Cumulative_charge']
    
    return df

def filter_rows(df):
    df_filtered = df[(df['type'] == 'D') & (df['comment'] != 'reference discharge')]
    return df_filtered

def plot_graph(df, x_column, y_column, title):
    # Plot capacity against date
    plt.scatter(df[x_column], df[y_column], marker='o', color='blue')

    # Add labels and title
    plt.xlabel('Cycles')
    plt.ylabel(y_column)
    plt.title(title)

    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45)

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def corr(df):# Check correlation between variables
    
    correlation_matrix = df.corr()
    
    # Print correlation matrix
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Plot correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()


def emd_application(df):
    # Set 'date' column as index
    df.set_index('date', inplace=True)

    # Format the index to YYYY-MM-DD HH
    df.index = df.index.strftime('%Y-%m-%d %H:00:00')

    # Convert index back to datetime format
    df.index = pd.to_datetime(df.index)

    # Reindex with the specified frequency
    new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    # if duplicates, take the last index from the group
    df_2 = df.groupby(level=0).last().reindex(new_index)

    # Remove rows with NaN values
    df_2 = df_2.dropna()

    # Drop columns type and comment
    df_2.drop(['type', 'comment'], axis=1, inplace=True)

    # EMD Analysis on capacity
    emd_train = df_2['capacity']
    emd = EMD()
    emd_training = emd(emd_train.values)

    # Pull the residue from the EMD output
    residue = emd_training[-1]
    df_2['emdcapacity']=residue
    
    for column in ['voltage','temperature']:
        # EMD Analysis on capacity
        emd_train = df_2[column]
        emd = EMD()
        emd_training = emd(emd_train.values)
    
        # Pull the residue from the EMD output
        residue = emd_training[-1]
        df_2[column]=residue

    return df_2

def plot_original_vs_residue(original_data,new_data,original_column,new_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(original_data['cycle'], original_data[original_column], label='Original Data', color='blue')
    plt.plot(new_data['cycle'], new_data[new_column], label='Residue', color='red')
    plt.xlabel('Cycles')
    plt.title(new_column)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_capacity_vs_date_w_threshold(df):
    # Calculate 50% threshold
    initial_capacity = df['emdcapacity'].iloc[0]
    threshold = 0.45 * initial_capacity

    # Plot Capacity Vs Date
    plt.figure(figsize=(10, 6))

    # Plot capacity against date
    plt.scatter(df.index, df['emdcapacity'], marker='o', color='blue', label='Capacity')

    # Plot 50% threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label='50% Threshold')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Capacity')
    plt.title('Capacity vs. Date')

    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45)

    # Add legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(df_column):
    
    
    # Plot ACF
    plt.figure(figsize=(10, 6))
    plot_acf(df_column)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

    # Plot PACF
    plt.figure(figsize=(10, 6))
    plot_pacf(df_column)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.show()

### Define ARMAX Model and evaluation

def train_arimax(df, split_factor=0.4, p_range=range(1, 5), d_range=range(1, 5), q_range=range(1, 5)):
    # Define response and variables
    response = df[['emdcapacity']]
    variables = df[['cycle','voltage', 'current', 'temperature', 'Cumulative_charge']]
    
    # Convert index to Timestamp
    #variables.index = variables.index.to_timestamp()
    variables.index = pd.DatetimeIndex(variables.index).to_period('H')
    
    # Split the data into training and testing sets
    train_size = int(len(variables) * split_factor)
    variables_train, variables_test = variables[:train_size], variables[train_size:]
    response_train, response_test = response[:train_size], response[train_size:]
    response_train.index = variables_train.index
    
    
    #Perform parameter tuning on p,d,q
    best_aic = float('inf')
    best_order = None

    for p in (p_range):
        for d in (d_range):
            for q in (q_range):
                order = (p, d, q)
                model = sm.tsa.ARIMA(endog=response_train, exog=variables_train, order=order,dates=response_train.index,freq='H')
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order


    # Use the best p, d, q values found during grid search
    order = best_order

    # Fit ARIMA model with the best parameters
    model = sm.tsa.ARIMA(endog=response_train, exog=variables_train, order=order,dates=variables_train.index,freq='H')
    results = model.fit()

    return model, results, variables_test, response_test

def implement_arimax(model, results, df,split_factor=0.5):
    
    # Define response and variables
    response = df[['emdcapacity']]
    variables = df[['cycle','voltage', 'current', 'temperature', 'Cumulative_charge']]

    # Split the data into training and testing sets
    train_size = int(len(variables) * split_factor)
    variables_train, variables_test = variables[:train_size], variables[train_size:]
    response_train, response_test = response[:train_size], response[train_size:]
    
    # create new test dataset (predicted based on historical values)
    test_df=create_test_dataset(df,len(response_test),split_factor)
    
   # Forecast
    forecast_steps = len(test_df)
    forecast = results.forecast(steps=forecast_steps, exog=test_df)

    # Calculate evaluation metrics
    mae_arimax = mean_absolute_error(response_test[:len(forecast)], forecast)
    mse_arimax = mean_squared_error(response_test[:len(forecast)], forecast)
    rmse_arimax = np.sqrt(mse_arimax)
    
    # Calculate State of Health (SOH) in percentage
    max_capacity = df['emdcapacity'].max()
    df['SOH'] = (df['emdcapacity'] / max_capacity) * 100
    
    # Calculate forecasted SOH based on the same highest number
    forecasted_soh = (forecast / max_capacity) * 100
    
    try:
        # Filter the DataFrame where SOH is equal to or lower than 70%
        filtered_df = df[df['SOH'] <= 70]
        # Extract values from the 'cycle' column
        cycles = filtered_df['cycle'].values 
        #cycle where battery goes below capacity
        min_cycle = min(cycles)
        
        #Forecsated df
        forecasted_df = pd.DataFrame(columns=['cycle', 'SOH'])
        forecasted_df['SOH']=forecasted_soh.values
        cycles_append=df.iloc[train_size:train_size+len(forecast), df.columns.get_loc('cycle')]
        forecasted_df['cycle']=cycles_append.values
        
        filtered_df_RUL = forecasted_df[forecasted_df['SOH'] <= 70]
        # Extract values from the 'cycle' column
        forecasted_cycles=filtered_df_RUL['cycle'].values
        #forecasted cycle where battery goes below capacity
        min_cycle_forecasted = min(forecasted_cycles)
    except:
        min_cycle_forecasted = "Forecast did not reach 70% Capacity"
    # Print a message indicating that the system did not converge to 70% capacity
        print("Forecast did not reach 70% Capacity")    
    
    
    # Plot full dataset + predictions
    plt.figure(figsize=(10, 6))
    plt.plot(df['cycle'], df['SOH'], label='Original Data', color='blue')
    plt.plot(df.iloc[train_size:train_size+len(forecast), df.columns.get_loc('cycle')], forecasted_soh, label='Predicted Capacity (ARIMAX)', color='red')
    
    # Plot 50% threshold line
    threshold = 70  # 60% threshold
    plt.axhline(y=threshold, color='red', linestyle='--', label='70% Threshold')
    
    plt.xlabel('Cycles')
    plt.ylabel('SOH ( % Capacity)')
    plt.title('Original Data vs. Predictions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print evaluation metrics
    print("Mean Absolute Error (MAE) ARIMAX:", mae_arimax)
    print("Mean Squared Error (MSE) ARIMAX:", mse_arimax)
    print("Root Mean Squared Error (RMSE) ARIMAX:", rmse_arimax)
    print("RUL Cycle (Actual):", min_cycle)
    print("RUL Cycle (Forecasted):",min_cycle_forecasted)

def implement_arimax_orginaldata(model, results, df,split_factor=0.5):
    
    # Define response and variables
    response = df[['emdcapacity']]
    variables = df[['cycle','voltage', 'current', 'temperature', 'Cumulative_charge']]

    # Split the data into training and testing sets
    train_size = int(len(variables) * split_factor)
    variables_train, variables_test = variables[:train_size], variables[train_size:]
    response_train, response_test = response[:train_size], response[train_size:]
    
    
   # Forecast
    forecast_steps = len(response_test)
    forecast = results.forecast(steps=forecast_steps, exog=variables_test)


    # Calculate evaluation metrics
    mae_arimax = mean_absolute_error(response_test[:len(forecast)], forecast)
    mse_arimax = mean_squared_error(response_test[:len(forecast)], forecast)
    rmse_arimax = np.sqrt(mse_arimax)
    
    # Calculate State of Health (SOH) in percentage
    max_capacity = df['emdcapacity'].max()
    df['SOH'] = (df['emdcapacity'] / max_capacity) * 100
    
    # Calculate forecasted SOH based on the same highest number
    forecasted_soh = (forecast / max_capacity) * 100
    
    try:
        # Filter the DataFrame where SOH is equal to or lower than 70%
        filtered_df = df[df['SOH'] <= 70]
        # Extract values from the 'cycle' column
        cycles = filtered_df['cycle'].values 
        #cycle where battery goes below capacity
        min_cycle = min(cycles)
        
        #Forecsated df
        forecasted_df = pd.DataFrame(columns=['cycle', 'SOH'])
        forecasted_df['SOH']=forecasted_soh.values
        cycles_append=df.iloc[train_size:train_size+len(forecast), df.columns.get_loc('cycle')]
        forecasted_df['cycle']=cycles_append.values
        
        filtered_df_RUL = forecasted_df[forecasted_df['SOH'] <= 70]
        # Extract values from the 'cycle' column
        forecasted_cycles=filtered_df_RUL['cycle'].values
        #forecasted cycle where battery goes below capacity
        min_cycle_forecasted = min(forecasted_cycles)
        
    except:
        min_cycle_forecasted = "Forecast did not reach 70% Capacity"
     # Print a message indicating that the system did not converge to 70% capacity
        print("Forecast did not reach 70% Capacity")     
    
    
    # Plot full dataset + predictions
    plt.figure(figsize=(10, 6))
    plt.plot(df['cycle'], df['SOH'], label='Original Data', color='blue')
    plt.plot(df.iloc[train_size:train_size+len(forecast), df.columns.get_loc('cycle')], forecasted_soh, label='Predicted Capacity (ARIMAX)', color='red')
    
    # Plot 50% threshold line
    threshold = 70  # 50% threshold
    plt.axhline(y=threshold, color='red', linestyle='--', label='70% Threshold')
    
    plt.xlabel('Cycles')
    plt.ylabel('SOH ( % Capacity)')
    plt.title('Original Data vs. Predictions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print evaluation metrics
    print("Mean Absolute Error (MAE) ARIMAX:", mae_arimax)
    print("Mean Squared Error (MSE) ARIMAX:", mse_arimax)
    print("Root Mean Squared Error (RMSE) ARIMAX:", rmse_arimax)
    print("RUL Cycle (Actual):", min_cycle)
    print("RUL Cycle (Forecasted):", min_cycle_forecasted)

###Define CNN model and evaluation

def train_cnn(df, split_factor=0.4, filters_values=[20, 30, 100], kernel_size_values=[4, 6, 10], lstm_units_values=[100, 200, 300]):
    # Extract features and target variable
    X = df[['cycle','voltage', 'current', 'temperature', 'Cumulative_charge']]
    y = df['emdcapacity']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape features for CNN
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X_reshaped) * split_factor)
    X_train, X_test = X_reshaped[:train_size], X_reshaped[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    input_shape=(X_train.shape[1], X_train.shape[2])
    best_model = None
    best_mse = np.inf

    for filters in filters_values:
        for kernel_size in kernel_size_values:
            for lstm_units in lstm_units_values:
                # Define the function to create the CNN+LSTM model
                model = Sequential([
                    Input(shape=input_shape),
                    Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                           input_shape=input_shape, padding='same'),
                    MaxPooling1D(pool_size=1),
                    Dropout(0.2),
                    Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
                    MaxPooling1D(pool_size=1),
                    Dropout(0.2),
                    LSTM(lstm_units, activation='relu', return_sequences=True),
                    Dense(1),
                    Flatten(),
                    Dense(100, activation='relu'),
                    Dense(1)
                ])
                
    
                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(X_train, y_train, epochs=10, verbose=0)

                # Forecast
                # Predict RUL for a subset of training data matching the length of X_test
                predicted_capacity = model.predict(X_test)

                # MSE calculation
                mse = mean_squared_error(y_test, predicted_capacity)

                # Update the best model if the current model is better
                if mse < best_mse:
                    best_mse = mse
                    best_model = model

    return best_model

def evaluate_and_plot_cnn(model, df, split_factor=0.4):
    # Extract features and target variable
    X = df[['cycle','voltage', 'current', 'temperature', 'Cumulative_charge']]
    y = df['emdcapacity']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape features for CNN
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X_reshaped) * split_factor)
    X_train, X_test = X_reshaped[:train_size], X_reshaped[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


    # create new test dataset (predicted based on historical values)
    test_df=create_test_dataset(df,len(y_test),split_factor)
    
    # Standardize features
    tdf_scaled = scaler.fit_transform(test_df)

    # Reshape features for CNN
    tdf_reshaped = tdf_scaled.reshape(tdf_scaled.shape[0], tdf_scaled.shape[1], 1)

    # Evaluate model fit
    predicted_capacity = model.predict(tdf_reshaped)
    
    # Calculate State of Health (SOH) in percentage
    max_capacity = df['emdcapacity'].max()
    df['SOH'] = (df['emdcapacity'] / max_capacity) * 100
    
    # Calculate forecasted SOH based on the same highest number
    forecasted_soh = (predicted_capacity.flatten() / max_capacity) * 100
    
    # Filter the DataFrame where SOH is equal to or lower than 70%
    filtered_df = df[df['SOH'] <= 70]
    # Extract values from the 'cycle' column
    cycles = filtered_df['cycle'].values 
    #cycle where battery goes below capacity
    min_cycle = min(cycles)
    
    #Forecsated df
    forecasted_df = pd.DataFrame(columns=['cycle', 'SOH'])
    forecasted_df['SOH']=forecasted_soh
    cycles_append=df.iloc[train_size:train_size+len(predicted_capacity), df.columns.get_loc('cycle')]
    forecasted_df['cycle']=cycles_append.values
    
    filtered_df_RUL = forecasted_df[forecasted_df['SOH'] <= 70]
    # Extract values from the 'cycle' column
    forecasted_cycles=filtered_df_RUL['cycle'].values
    #forecasted cycle where battery goes below capacity
    min_cycle_forecasted = min(forecasted_cycles)
    
    # Plot predicted RUL vs. actual RUL
    plt.figure(figsize=(10, 6))
    plt.plot(df['cycle'], df['SOH'], color='blue', label='Original Data')
    plt.plot(df.iloc[train_size:train_size+len(predicted_capacity), df.columns.get_loc('cycle')], forecasted_soh, color='red', label='Predicted Capacity(CNN)')
    plt.title('Predicted RUL vs. Actual RUL')
    plt.xlabel('Cycles')
    plt.ylabel('SOH (% Capacity)')
    plt.grid(True)
    
    
    # Plot 50% threshold line
    initial_capacity = df['emdcapacity'].iloc[0]
    threshold = 70
    plt.axhline(y=threshold, color='red', linestyle='--', label='70% Threshold')
    plt.legend()

    # Print evaluation metrics for ARIMAX and CNN Models
    mae_cnn = mean_absolute_error(y_test[:len(forecasted_soh)], predicted_capacity)
    mse_cnn = mean_squared_error(y_test[:len(forecasted_soh)], predicted_capacity)
    rmse_cnn = np.sqrt(mse_cnn)
    print("Mean Absolute Error (MAE) CNN:", mae_cnn)
    print("Mean Squared Error (MSE) CNN:", mse_cnn)
    print("Root Mean Squared Error (RMSE) CNN:", rmse_cnn)
    print("RUL Cycle (Actual):", min_cycle)
    print("RUL Cycle (Forecasted):", min_cycle_forecasted)
    plt.show()

def evaluate_and_plot_cnn_originaldata(model, df, split_factor=0.4):
    # Extract features and target variable
    X = df[['cycle','voltage', 'current', 'temperature', 'Cumulative_charge']]
    y = df['emdcapacity']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape features for CNN
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X_reshaped) * split_factor)
    X_train, X_test = X_reshaped[:train_size], X_reshaped[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Evaluate model fit
    predicted_capacity = model.predict(X_test)
    
    # Calculate State of Health (SOH) in percentage
    max_capacity = df['emdcapacity'].max()
    df['SOH'] = (df['emdcapacity'] / max_capacity) * 100
    
    # Calculate forecasted SOH based on the same highest number
    forecasted_soh = (predicted_capacity.flatten() / max_capacity) * 100
    
    # Filter the DataFrame where SOH is equal to or lower than 70%
    filtered_df = df[df['SOH'] <= 70]
    # Extract values from the 'cycle' column
    cycles = filtered_df['cycle'].values 
    #cycle where battery goes below capacity
    min_cycle = min(cycles)
    
    #Forecsated df
    forecasted_df = pd.DataFrame(columns=['cycle', 'SOH'])
    forecasted_df['SOH']=forecasted_soh
    cycles_append=df.iloc[train_size:, df.columns.get_loc('cycle')]
    forecasted_df['cycle']=cycles_append.values
    
    filtered_df_RUL = forecasted_df[forecasted_df['SOH'] <= 70]
    # Extract values from the 'cycle' column
    forecasted_cycles=filtered_df_RUL['cycle'].values
    #forecasted cycle where battery goes below capacity
    min_cycle_forecasted = min(forecasted_cycles)
    
    # Plot predicted RUL vs. actual RUL
    plt.figure(figsize=(10, 6))
    plt.plot(df['cycle'], df['SOH'], color='blue', label='Original Data')
    plt.plot(df.iloc[train_size:, df.columns.get_loc('cycle')], forecasted_soh, color='red', label='Predicted Capacity(CNN)')
    plt.title('Predicted RUL vs. Actual RUL')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.grid(True)
    
    # Plot 70% threshold line
    initial_capacity = df['emdcapacity'].iloc[0]
    threshold = 70
    plt.axhline(y=threshold, color='red', linestyle='--', label='70% Threshold')
    plt.legend()
    
    # Print evaluation metrics for ARIMAX and CNN Models
    mae_cnn = mean_absolute_error(y_test, predicted_capacity)
    mse_cnn = mean_squared_error(y_test, predicted_capacity)
    rmse_cnn = np.sqrt(mse_cnn)
    print("Mean Absolute Error (MAE) CNN:", mae_cnn)
    print("Mean Squared Error (MSE) CNN:", mse_cnn)
    print("Root Mean Squared Error (RMSE) CNN:", rmse_cnn)
    print("RUL Cycle (Actual):", min_cycle)
    print("RUL Cycle (Forecasted):", min_cycle_forecasted)
    plt.show()

def emd_column(df,column):
    # EMD Analysis on capacity
    emd_train = df[column]
    emd = EMD()
    emd_training = emd(emd_train.values)

    # Pull the residue from the EMD output
    residue = emd_training[-1]

    return residue

def create_test_dataset(df, test_size,split_factor):
    
    # Create an empty DataFrame with the same columns as df
    test_df = pd.DataFrame(columns=df.columns)
    test_df_2 = pd.DataFrame(columns=df.columns) 
    
    #Filter test_df
    test_df=test_df[['cycle','voltage', 'current', 'temperature','Cumulative_charge']]
    test_df_2=test_df_2[['cycle','voltage', 'current', 'temperature','Cumulative_charge']]

    
    #Conduct EMD on all variables and add to test_df table
    test_df['voltage']=emd_column(df,'voltage')
    test_df['temperature']=emd_column(df,'temperature')
    test_df['current']=df['current'].values
    test_df['cycle']=df['cycle'].values
    
    
    #Split data into train and test
    train_size = int(len(test_df) * split_factor)
    train_data = test_df.iloc[:train_size]
    test_data = test_df.iloc[train_size:]
    
    # Fetch data from train_data with length equal to test_data for 'current' column
    if len(test_data)<=len(train_data):
        for column in ['current','voltage','temperature','cycle']:
            train_current = train_data[column].tail(len(test_data)).reset_index(drop=True)
            test_df_2[column] = train_current.values
    else:
        for column in ['current','voltage','temperature','cycle']:
            train_current = train_data[column].reset_index(drop=True)
            test_df_2[column] = train_current.values
    
    #Calculate the average difference between consecutive cumulative charge values in the train dataset
    avg_diff_cumulative_charge = np.mean(np.diff(df['Cumulative_charge']))

    # Set the first cumulative charge value in the test dataset
    test_df_2.loc[test_df_2.index[0], 'Cumulative_charge'] = df['Cumulative_charge'].iloc[-1] + avg_diff_cumulative_charge

    # Calculate cumulative charge for subsequent rows
    for i in range(len(test_df_2)-1):
       test_df_2.loc[i+1, 'Cumulative_charge'] = test_df_2.loc[i, 'Cumulative_charge'] + avg_diff_cumulative_charge
    
    test_df_2['Cumulative_charge'] = test_df_2['Cumulative_charge'].astype('float64')
    
    return test_df_2



###############################################################################################################

###Implement functions to complete data cleaning and graphical analysis 
file_path = 'M:\OMSA\Practicuum\RW9.csv'
df = read_data(file_path)
df_clean = clean_data(df)
df_cap=calculate_capacity(df_clean)
df_cumc=calculate_cumulative_charge(df_cap)
df_filter=filter_rows(df_cumc)
plot_graph(df_filter,'cycle', 'Cumulative_charge', 'Cumulative Charge Vs Cycles (Battery 1)')
plot_graph(df_filter,'cycle', 'voltage', 'Voltage Vs Cycles (Battery 1)')
plot_graph(df_filter,'cycle', 'current', 'Current Vs Cycles (Battery 1)')
plot_graph(df_filter,'cycle', 'temperature', 'Temperature Vs Cycles (Battery 1)')
plot_graph(df_filter,'cycle', 'capacity', 'Capacity Vs Cycles (Battery 1)')

#check for correlation in dataset
corr(df_filter[['time','relativeTime','voltage','current','temperature','capacity','Cumulative_charge']])

###Implement Output Signal Cleaning using EMD and plot the output
df_emd=emd_application(df_filter)
plot_original_vs_residue(df_filter, df_emd,'capacity','emdcapacity')
plot_original_vs_residue(df_filter,df_emd,'voltage','voltage')
plot_original_vs_residue(df_filter,df_emd,'current','current')
plot_original_vs_residue(df_filter,df_emd,'temperature','temperature')


#Plot capacity chart with threshold
plot_capacity_vs_date_w_threshold(df_emd)

#analyze using acf and pacf charts for models to select
plot_acf_pacf(df_emd['emdcapacity'])


### Implement ARIMAX Model: Train and evaluate on first battery dataset
arimax_model,results,variables_test,response_test=train_arimax(df_emd,split_factor=0.70, p_range=range(1, 3), d_range=range(0, 2), q_range=range(1, 3))
implement_arimax(arimax_model, results, df_emd, split_factor=0.25)
implement_arimax_orginaldata(arimax_model, results, df_emd, split_factor=0.25)

###Implement CCN Model: Train and evaluate on first battery dataset
best_model=train_cnn(df_emd,split_factor=0.70,filters_values=[20, 30, 100], kernel_size_values=[4, 6, 10], lstm_units_values=[10, 20, 30])
evaluate_and_plot_cnn(best_model, df_emd, split_factor=0.25)
evaluate_and_plot_cnn_originaldata(best_model, df_emd, split_factor=0.25)

###############################################################################################################


###Implement functions to complete data cleaning and graphical analysis 
file_path_3 = 'M:\OMSA\Practicuum\RW11.csv'
df_3 = read_data(file_path_3)
df_clean_3 = clean_data(df_3)
df_cap_3=calculate_capacity(df_clean_3)
df_cumc_3=calculate_cumulative_charge(df_cap_3)
df_filter_3=filter_rows(df_cumc_3)
plot_graph(df_filter_3,'date', 'Cumulative_charge', 'Cumulative Charge Vs Date (Battery 3)')
plot_graph(df_filter_3,'date', 'capacity', 'Capacity Vs Date (Battery 3)')

###Implement Output Signal Cleaning using EMD and plot the output
df_emd_3=emd_application(df_filter_3)
plot_original_vs_residue(df_filter_3, df_emd,'capacity','emdcapacity')
plot_original_vs_residue(df_filter_3,df_emd,'voltage','voltage')
plot_original_vs_residue(df_filter_3,df_emd,'current','current')
plot_original_vs_residue(df_filter_3,df_emd,'temperature','temperature')

#Plot capacity chart with threshold
plot_capacity_vs_date_w_threshold(df_emd_3)

#analyze using acf and pacf charts for models to select
plot_acf_pacf(df_emd_3['emdcapacity'])

### Evaluate ARIMAX model performance on second battery dataset
implement_arimax(arimax_model, results, df_emd_3,split_factor=0.2)
implement_arimax_orginaldata(arimax_model, results, df_emd_3,split_factor=0.2)


###Evaluate CCN Model performance on second battery dataset
evaluate_and_plot_cnn(best_model, df_emd_3, split_factor=0.2)
evaluate_and_plot_cnn_originaldata(best_model, df_emd_3, split_factor=0.2)

###############################################################################################################


###Implement functions to complete data cleaning and graphical analysis 
file_path_4 = 'M:\OMSA\Practicuum\RW10.csv'
df_4 = read_data(file_path_4)
df_clean_4 = clean_data(df_4)
df_cap_4=calculate_capacity(df_clean_4)
df_cumc_4=calculate_cumulative_charge(df_cap_4)
df_filter_4=filter_rows(df_cumc_4)
plot_graph(df_filter_4,'date', 'Cumulative_charge', 'Cumulative Charge Vs Date (Battery 4)')
plot_graph(df_filter_4,'date', 'capacity', 'Capacity Vs Date (Battery 4)')

###Implement Output Signal Cleaning using EMD and plot the output
df_emd_4=emd_application(df_filter_4)
plot_original_vs_residue(df_filter_4, df_emd,'capacity','emdcapacity')
plot_original_vs_residue(df_filter_4,df_emd,'voltage','voltage')
plot_original_vs_residue(df_filter_4,df_emd,'current','current')
plot_original_vs_residue(df_filter_4,df_emd,'temperature','temperature')


#Plot capacity chart with threshold
plot_capacity_vs_date_w_threshold(df_emd_4)

#analyze using acf and pacf charts for models to select
plot_acf_pacf(df_emd_4['emdcapacity'])

### Evaluate ARIMAX model performance on second battery dataset
implement_arimax(arimax_model, results, df_emd_4,split_factor=0.2)
implement_arimax_orginaldata(arimax_model, results, df_emd_4,split_factor=0.2)

###Evaluate CCN Model performance on second battery dataset
evaluate_and_plot_cnn(best_model, df_emd_4, split_factor=0.2)
evaluate_and_plot_cnn_originaldata(best_model, df_emd_4, split_factor=0.2)