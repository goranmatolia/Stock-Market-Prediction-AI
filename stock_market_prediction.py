# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the stock prices data into a Pandas dataframe df = pd.read_csv('stock_prices.csv')
# Preprocess the data by cleaning, transforming, and scaling it # ...code for data preprocessing...
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Linear Regression model on the training data model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the testing data using the trained model y_pred = model.predict(X_test)
# Evaluate the model's performance using Mean Squared Error and R-squared metrics mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print the evaluation metrics print("Mean Squared Error: ", mse) print("R-squared: ", r2)
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load historical stock prices
df = pd.read_csv('historical_stock_prices.csv')
# Select relevant feature
data = df.filter(['Close']).values
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1)) scaled_data = scaler.fit_transform(data)
# Split the data into training and testing sets train_data = scaled_data[:int(0.8*len(data))] test_data = scaled_data[int(0.8*len(data)):]
# Create training data
def create_data(data, lookback):
X = []
y = []
for i in range(lookback, len(data)):
X.append(data[i-lookback:i, 0])
y.append(data[i, 0])
return np.array(X), np.array(y)
lookback = 60 # number of past days to look at X_train, y_train = create_data(train_data, lookback) X_test, y_test = create_data(test_data, lookback)
# Reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1))) model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=100)
# Make predictions on the testing set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
# Evaluate the model's performance
mse = np.mean((y_test - predictions)**2)
print('Mean squared error:', mse)