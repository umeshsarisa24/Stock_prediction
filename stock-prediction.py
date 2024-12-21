import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class StockPredictor:
    def __init__(self, ticker="AAPL", period="2y"):
        self.ticker = ticker
        self.period = period
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=self.period)
        return df
    
    def prepare_data(self, df, lookback=60):
        """Prepare data for LSTM model"""
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i])
        
        X = np.array(X)
        y = np.array(y)
        

        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        return X_train, y_train, X_test, y_test, scaled_data
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        
        df = self.fetch_data()
        X_train, y_train, X_test, y_test, scaled_data = self.prepare_data(df)
        
    
        self.model = self.build_model((X_train.shape[1], 1))
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        

        train_predict = self.model.predict(X_train)
        test_predict = self.model.predict(X_test)
        
        
        train_predict = self.scaler.inverse_transform(train_predict)
        y_train_inv = self.scaler.inverse_transform(y_train)
        test_predict = self.scaler.inverse_transform(test_predict)
        y_test_inv = self.scaler.inverse_transform(y_test)
        
        
        train_rmse = np.sqrt(np.mean((train_predict - y_train_inv) ** 2))
        test_rmse = np.sqrt(np.mean((test_predict - y_test_inv) ** 2))
        print(f'Train RMSE: {train_rmse:.2f}')
        print(f'Test RMSE: {test_rmse:.2f}')
        
        return history, train_predict, test_predict, df
    
    def plot_results(self, history, train_predict, test_predict, df):
        """Plot training results"""
        
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-len(test_predict):], test_predict, label='Predicted')
        plt.plot(df.index[-len(test_predict):], df['Close'][-len(test_predict):], label='Actual')
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
    def predict_next_day(self):
        """Predict the next day's price"""
        df = self.fetch_data()
        scaled_data = self.scaler.transform(df['Close'].values.reshape(-1, 1))
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        
        prediction = self.model.predict(last_60_days)
        prediction = self.scaler.inverse_transform(prediction)
        
        return prediction[0][0]

def main():
    
    predictor = StockPredictor(ticker="AAPL", period="2y")
    
    
    print("Training model...")
    history, train_predict, test_predict, df = predictor.train(epochs=50)
    
    
    predictor.plot_results(history, train_predict, test_predict, df)
    
    
    next_day_price = predictor.predict_next_day()
    print(f"\nPredicted price for next day: ${next_day_price:.2f}")

if __name__ == "__main__":
    main()
