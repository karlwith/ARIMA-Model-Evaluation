# ARIMA Model Evaluation
Framework designed to assist with evaluation of ARIMA model configurations for time series imputation of Bluetooth MAC address scanner (BMS) traffic data.

Insights into initial ARIMA model configuration are given by Autocorrelation and Partial Autocorrelation plots. All valid ARIMA model configurations are deployed to impute randomly seeded BMS time series gaps of sizes 1, 2, 3 and 4 time lag intervals of 5 minutes.

Root mean square error is used to evaluate the performance of each model configuration for all gap sizes. A basic linear interpolation model is also implemented as a benchmark and found to be the most effective at imputing single point gaps.

Work completed individually as part of QUT research unit IFN712.
