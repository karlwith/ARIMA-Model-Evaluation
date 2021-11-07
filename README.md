# ARIMA Model Evaluation
Framework designed to assist with evaluation of ARIMA model configurations for time series imputation of Bluetooth MAC address scanner (BMS) traffic data.

Insights into initial ARIMA model configuration are given by Autocorrelation and Partial Autocorrelation plots. Each valid ARIMA model configuration is fitted to the complete dataset (70 days worth) then applied to 4 different imputation cases. Each imputation case consists of the entire dataset with either 1, 2, 3 or 4 randomly seeded consecutive missing values.

Root mean square error is used to evaluate the performance of each model configuration for each imputation case. A basic linear interpolation model is also implemented as a benchmark and found to be the most effective at imputing single value gaps.

Work completed individually as part of QUT research unit IFN712.
