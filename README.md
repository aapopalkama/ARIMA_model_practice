# ARIMA_model_practice
Machine learning practice

NVDIA stock price forecast:

The program was run on January 25, 2022 before the stock exchange opened.

January 24, 2022 NVDA closing price:
* 233,76 $

Predictions:
* Day 1: 234.039100 $, Closing price 25.2.2022 - 223.24 $

Arima problems:

* A problem with ARIMA is that it does not support seasonal data. That is a time series with a repeating cycle.

* ARIMA expects data that is either not seasonal or has the seasonal component removed.

# SARIMA_model_practice

* Sarima is an extension of ARIMA that support univariate time series data with a seasonal component.

SARIMA(p,d,q)x(P,D,Q)m

Where:
* p and seasonal P: indicate the number of AR terms (lags of the stationary series)
* d and seasonal D: indicate differencing that must be done to stationary series
* q and seasonal Q: indicate the number of MA terms (lags of the forecast errors)
* lag: indicates the seasonal length in the data


