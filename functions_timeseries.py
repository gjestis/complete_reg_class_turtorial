## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## for statistical tests
from statsmodels.tsa.seasonal import seasonal_decompose

## for machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################

def decompose_ts(df, datetime_col, target_col, samples=250, period=24):
    if samples == 'all':
        # Decomposing all time series timestamps
        res = seasonal_decompose(df[target_col].values, period=period)
    else:
        # Decomposing a sample of the time series
        res = seasonal_decompose(df[target_col].values[-samples:], period=period)

    observed = res.observed
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid

    # Get the corresponding dates for the decomposed components
    dates = pd.to_datetime(df[datetime_col].iloc[-len(observed):], format='%d-%b-%y')

    # Plot the complete time series
    fig, axs = plt.subplots(4, figsize=(16, 10))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(dates, observed)
    axs[0].grid()

    # Plot the trend of the time series
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(dates, trend)
    axs[1].grid()

    # Plot the seasonality of the time series
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(dates, seasonal)
    axs[2].grid()

    # Plot the noise of the time series
    axs[3].set_title('NOISE', fontsize=16)
    axs[3].plot(dates, residual)
    axs[3].scatter(x=dates, y=residual, alpha=0.5)
    axs[3].grid()

    # Format the x-axis tick labels as dates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    for ax in axs:
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.show()


def train_time_series_with_avg(df, target_var, horizon=10, use_moving_avg=False, window=7):
    # Isolate the target variable
    y = df[target_var]

    # Create train and test
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    print("Start date for training:", y_train.index[0])
    print("End date for training:", y_train.index[-1])
    print("Start date for test:", y_test.index[0])
    print("End date for test:", y_test.index[-1])

    if use_moving_avg:
        # Calculate moving average
        moving_average = y_train.rolling(window=window).mean()
        last_window_average = moving_average.iloc[-1]

        # Create an array of moving average values for the length of the validation set
        predictions = np.full_like(y_test, fill_value=last_window_average)

        model_name = f'Moving Average (Window: {window})'
    else:
        # Calculate historical average
        historical_average = y_train.mean()

        # Create an array of historical average values for the length of the validation set
        predictions = np.full_like(y_test, fill_value=historical_average)

        model_name = 'Historical Average'

    # Calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)

    # Plot reality vs prediction
    fig = plt.figure(figsize=(16, 8))
    plt.title(f'Real vs Prediction - {model_name} (MAE: {mae})', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel(target_var, fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()


def average_forecast(df, start_date, end_date, target, moving_avg_window=None):
    # Create a new DataFrame for the forecasted values
    forecast_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='M'))

    if moving_avg_window is None:
        # Use historical average for the forecast
        forecast_df['average'] = df[target].mean()
        print("Forecasting using historical average:", df[target].mean())
    else:
        # Use moving average for the forecast
        df['moving_avg'] = df[target].rolling(window=moving_avg_window, min_periods=1).mean()
        forecast_df['average'] = df['moving_avg'].iloc[-1]
        print(f"Forecasting using moving average with window={moving_avg_window}:", df['moving_avg'].iloc[-1])

    # Plot the original time series and the forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Original')
    plt.plot(forecast_df.index, forecast_df['average'], label='Forecast')
    plt.title('Original Time Series and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_df


def find_optimal_lags(df, target_col, feature_col, max_lags):
    mae_scores = []

    # Iterate over different lag values
    for lag in range(1, max_lags + 1):
        # Create lagged features
        lagged_df = pd.DataFrame()
        lagged_df[feature_col] = df[feature_col].shift(lag)
        lagged_df[target_col] = df[target_col]

        # Drop missing values
        lagged_df.dropna(inplace=True)

        # Split the data into train and test sets
        train_size = int(len(lagged_df) * 0.8)
        X_train = lagged_df[feature_col].iloc[:train_size].values.reshape(-1, 1)
        y_train = lagged_df[target_col].iloc[:train_size].values.reshape(-1, 1)
        X_test = lagged_df[feature_col].iloc[train_size:].values.reshape(-1, 1)
        y_test = lagged_df[target_col].iloc[train_size:].values.reshape(-1, 1)

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate MAE score
        mae = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)

    # Find the lag value with the lowest MAE score
    optimal_lag = np.argmin(mae_scores) + 1
    min_mae = np.min(mae_scores)

    # Plot the MAE scores
    plt.plot(range(1, max_lags + 1), mae_scores)
    plt.xlabel('Number of Lag Values')
    plt.ylabel('MAE')
    plt.title('MAE vs Number of Lag Values')
    plt.grid(True)
    plt.show()

    return optimal_lag, min_mae


def create_lagged_matrix_for_train_test(df, target_column, feature_column, lag):
    """
    Creates a lagged matrix from a pandas DataFrame with a target variable and a feature variable.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series data.
        target_column (str): Name of the column containing the target variable.
        feature_column (str): Name of the column containing the feature variable.
        lag (int): Number of lagged values to include.

    Returns:
        numpy.array, numpy.array: X array with lagged feature values, y array with lagged target values.
    """
    lagged_df = pd.DataFrame()
    for i in range(lag):
        lagged_df[f'Lag{i + 1}_{feature_column}'] = df[feature_column].shift(i + 1)
    lagged_df[f'Lag1_{target_column}'] = df[target_column].shift(1)
    lagged_df.dropna(inplace=True)
    X = lagged_df.iloc[:, :lag].values
    y = lagged_df.iloc[:, lag].values
    return X, y


def train_time_series_with_linreg(X, y, horizon=24 * 7):
    # Convert to pd
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # Create train and test
    X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # Reset index for X_test and y_test
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Create, train, and do inference of the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # If predictons are negative map them to min value
    for i in range(len(predictions)):
        if predictions[i] <= 0:
            predictions[i] = 440
    # Calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    # Plot reality vs prediction for the last week of the dataset
    fig = plt.figure(figsize=(16, 8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red', marker='o')
    plt.plot(predictions, color='green', marker='o', linestyle='dashed')
    plt.xlabel('Days', fontsize=16)
    plt.ylabel('Gas Demand', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()

    return model, mae, predictions


def create_lagged_matrix_for_forecast(df, column, lag):
    """
    Creates a matrix with lagged values from a given column of a DataFrame
    and divides it into X and y arrays suitable for machine learning training.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series data.
        column (str): Name of the column containing the time series.
        lag (int): Number of lagged values to include.

    Returns:
        numpy.array, numpy.array: X array with lagged values, y array with target values.
    """
    lagged_df = pd.DataFrame()
    for i in range(lag):
        lagged_df[f'Lag{i + 1}'] = df[column].shift(i + 1)
    lagged_df.dropna(inplace=True)
    # X = lagged_df.iloc[:, 1:].values
    # y = lagged_df.iloc[:, 0].values
    return lagged_df


def plot_forecast(df, forecast_df, target):
    """
    Plot forecasted values on the original time series plot.

    Args:
        df (pandas.DataFrame): Historical data.
        forecast_df (pandas.DataFrame): Forecasted values.
        target (str): Target variable column name.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Original')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast')
    plt.title('Original Time Series and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def iterative_forecasting(X, model, num_forecasts, column_names):
    """
    Perform iterative forecasting using a machine learning model.

    Args:
        X (numpy.ndarray): Lagged features used for training the model.
        model: Trained machine learning model.
        num_forecasts (int): Number of forecasts to generate.
        column_names (list): Column names of the lagged features.

    Returns:
        pandas.DataFrame: Forecasted values for the specified periods.
    """
    # Create a copy of the original lagged features
    X_copy = pd.DataFrame(X.copy(), columns=column_names)

    # Create an empty DataFrame to store the forecasted values
    forecast_df = pd.DataFrame(index=range(1, num_forecasts + 1), columns=['Forecast'])

    for i in range(num_forecasts):
        print("Forecast nr:", i)
        # Get the last row in the lagged features
        last_row = X_copy.iloc[-1, :].values

        # Make a prediction using the lagged features
        prediction = model.predict([last_row])[0]
        print("Predicted:", prediction[0])

        # Shift the values in the row to the right
        new_row = np.roll(last_row, 1)

        # Update the first value in the row with the prediction
        new_row[0] = prediction[0]

        # Convert the new row back to a DataFrame and set the index
        new_row = pd.DataFrame([new_row], columns=column_names)
        print(new_row)

        # Store the prediction in the forecast DataFrame
        forecast_df.loc[i + 1, 'forecast'] = prediction[0]

        # Append the new row to the lagged features
        X_copy = X_copy.append(new_row, ignore_index=True)
        print(X_copy)

    return forecast_df
