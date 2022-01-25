import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as smapi
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

def difference(dataset, interval=1): # Differenced seires
	diff = []
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

def predict(hist,yhat,interval=1): # Invert differenced value
    return (yhat + hist[-interval])


#   Choose your stock
symbol = 'NVDA'      
#   List of symbols for technical indicators
start = (datetime.date.today() - datetime.timedelta(10000) )
#   End time
end = datetime.datetime.today()
#   Data from yahoo finance
data = yf.download(symbol, start=start, end=end, interval="1d")
#   Rename columns
data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
#   Set index name
data["Date"] = data.index
data['DATE'] = data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
print(data.tail(10))

data = data[(data["Date"] > "2015-1-1")]

data1 = data[["DATE","close"]]

data1.reset_index(drop=True, inplace = True)

data1 = data1.values

data1 = data[["close"]].to_numpy() # Dataset to numpy

#   Arima parameters p,d,q :
p=4     # Number of lag AR term
d=1     # Number of times of differing required to achieve the stationary
q=0     # Number of lag MA term

days = 365 #  Days in year
differenced = difference(data1,days)
model = smapi.tsa.arima.ARIMA(differenced,order = (p,d,q)) # Fit model
model_fit = model.fit()
forecast = model_fit.forecast(steps=7) # Multi step
hist = [x for x in data1]
day = 1
for i in forecast:
    inv = predict(hist,i,days)
    print("Day %d: %f" % (day, inv)) # Print which day and price prediction
    hist.append(predict)
    day += 1

#   Let's put 70% of data to training and 30% to test
train_data = data[0:int(len(data)*0.70)]
train_data = train_data["close"].values
test_data = data[int(len(data)*0.70):]
test_data = test_data["close"].values

#   Predeictive ARIMA model
hist = [i for i in train_data]
predictions = []
observations = len(test_data)

for x in range(observations):
    model = smapi.tsa.arima.ARIMA(hist, order=(p,d,q))
    m_fit = model.fit()
    output = m_fit.forecast()
    y = output[0]
    predictions.append(y)
    test_value = test_data[x]
    hist.append(test_value)

#   Mean squared Error:
mse = mean_squared_error(test_data,predictions)
print("MSE error is {}".format(mse))

test_set_range = data[int(len(data)*0.7):].index
plt.plot(test_set_range, predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', label='Actual Price')
plt.title(symbol+' Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()















