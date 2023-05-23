import yfinance as yf
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import requests


# Prepare the features and target variables for training
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def lambda_handler(event, context):

    # Parse the JSON payload
    body = event.get("body", "{}")
    payload = json.loads(body)

    # Define the ticker symbol for item we want to get or default to bitcoin
    ticker_symbol = payload.get("ticker", "BTC-USD")

    # Define the start and end dates for historical data
    start_date = datetime.now() - timedelta(days=3650)  # ten year ago
    end_date = datetime.now()


    try:
        # Create a ticker object
        ticker = yf.Ticker(ticker_symbol)
        # Print the data
        print(ticker.info)
        # Rest of the code...
        # Fetch historical data from Yahoo Finance
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
    except requests.exceptions.HTTPError as e:
        error_message = f"Error fetching ticker information for {ticker_symbol}"
        print(error_message)
        return {
            'statusCode': 404,
            'body': error_message
        }


    # Prepare the data
    data = data[['Close']]
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Split the data into training and testing sets
    train_data = scaled_data[:int(len(dataset) * 0.8)]
    test_data = scaled_data[int(len(dataset) * 0.8):]

    time_steps = 90

    # Check if the length of training data is less than time_steps
    if len(train_data) < time_steps:
        error_message = "Insufficient training data: "+len(train_data)
        print(error_message)
        return {
            'statusCode': 400,
            'body': error_message
        }

    X_train, y_train = create_dataset(train_data, time_steps)
    X_test, y_test = create_dataset(test_data, time_steps)

    # Reshape the input data for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create and train the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Invert the scaling
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform([y_train])
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform([y_test])

    # Predict the Bitcoin price for the future
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=28, freq='D')
    future_prices = []

    # Create a DataFrame for the predicted prices
    predicted_data = pd.DataFrame(index=future_dates, columns=['Close'])

    last_60_days = data[-time_steps:].values

    for date in future_dates:
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        future_prices.append(predicted_price[0][0])
        predicted_data.loc[date] = predicted_price[0][0]

        # Update last_60_days with the new predicted price
        last_60_days = np.append(last_60_days[1:], [[predicted_price[0][0]]], axis=0)

    # Print the predicted prices
    print("Predicted Prices:")
    for date, price in zip(future_dates, future_prices):
        print(f"Date: {date}, Predicted Price: {price}")

    # Format the results
    results = []

    for date, price in zip(future_dates, future_prices):
        result = {
            'Date': date.strftime('%Y-%m-%d'),
            'Predicted Price': '${:.2f}'.format(price)
        }
        results.append(result)

    response = []
    # Add an index called "info" with a value
    info_index = {'info': ticker.info}
    response.append(info_index)
    # Add an index called "predictions" with a value
    prediction_index = {'predictions': results}
    response.append(prediction_index)

    return response
