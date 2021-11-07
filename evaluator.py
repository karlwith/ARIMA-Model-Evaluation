import math
import operator
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# return:   cases (Dictionary)          - A nested dictionary of the form {case: {index: mean}}.
def load_data():
    cases = {}
    # Load .csv and parse BMS data for training and evaluation cases.
    for case in all_cases:
        # Load relevant columns from csv file of BMS data
        df = pd.read_csv('section_10278650/' + case + '.csv', usecols=['section - 10278650'])
        df.rename(columns={'section - 10278650': 'mean'}, inplace=True)
        cases[case] = df
    print('CSV data loaded.')
    return cases


# arg: case (String)                    - The time series to plot.
# return:                               - A window containing the ACF plot for 3 different orders of differencing.
def show_acf_plot(case):
    fig, axes = plt.subplots(4, 1, sharex=True)
    axes[0].set(xlim=(-1, 35))
    # Plot Original Series
    plot_acf(cases[case]['mean'], ax=axes[3], title='Original Series ACF')
    # Plot Differenced Series
    plot_acf(cases[case]['mean'].diff().dropna(), ax=axes[1], title='1st Order Differenced ACF')
    plot_acf(cases[case]['mean'].diff().diff().dropna(), ax=axes[2], title='2nd Order Differenced ACF')
    plot_acf(cases[case]['mean'].diff().diff().diff().dropna(), ax=axes[0], title='3rd Order Differenced ACF')
    plt.xlabel('Time Lag (5 minute scale)')
    plt.ylabel('ACF')
    plt.show()


# arg: case (String)                    - The time series to plot
# return:                               - A window containing the PACF plot for the 1st order differenced series.
def show_pacf_plot(case):
    plot_pacf(cases[case]['mean'].diff().dropna(), title='1st Order Differenced PACF')
    plt.xlabel('Time Lag (5 minute scale)')
    plt.ylabel('Partial ACF')
    plt.show()


# arg: predictions (Dictionary)         - The index and predicted values for each missing duration case.
# arg: model_order (String)             - Order of the ARIMA model to evaluate ('linear' for linear interpolation).
# return:                               - A window containing a plot of the actual and imputed values.
def show_prediction_plot(predictions, model_order):
    fig, axes = plt.subplots(4, 1, sharex=True)
    # Set appropriate title
    if model_order == 'linear':
        fig.suptitle('Linear Interpolation Model', fontsize='xx-large')
    else:
        fig.suptitle('ARIMA' + str(model_order) + ' Model', fontsize='xx-large')
    # Define the actual and predicted values to plot.
    i = 0
    while i < len(evaluation_cases):
        case = evaluation_cases[i]
        # Predicted values
        #pred = sorted(predictions[case].items())
        pred_x, pred_y = zip(*sorted(predictions[case].items()))
        axes[i].plot(pred_x, pred_y, 'o', color='red', markersize=3)
        # Actual values
        axes[i].plot(cases[case]['mean'], color='#1f77b4')
        axes[i].set_title(label="'"+case+"' Case With Predictions, RMSE:"+str(getattr(evaluated_models[model_order], 'err_'+case))[:6])
        i += 1
    plt.xlabel('Time Lag (5 minute scale)')
    plt.ylabel('Mean Travel Time (seconds)')
    plt.xlim([0, 75])
    plt.show()


# arg: model_order (String)             - Order of the ARIMA model to evaluate
# return: predictions (Dict)            - A nested dictionary of the form {testcase: {index: prediction}}.
def evaluate_arima_model(model_order):
    # Define and fit the model.
    model = ARIMA(cases['continuous_70day'], order=model_order)
    model_fit = model.fit()
    # Initialize dictionary to store associated original and predicted values.
    predictions = {'single_gap': dict(), 'double_gap': dict(), 'triple_gap': dict(), 'quad_gap': dict()}
    for case in evaluation_cases:
        # Initialize case specific cache.
        missing_indexes = []
        for index, row in cases[case].iterrows():
            # Start of missing value gap.
            if pd.isna(row.values):
                missing_indexes.append(index)
            # End of missing gap.
            elif len(missing_indexes):
                # Apply model to predict the missing values for this 'missing' segment.
                prediction = model_fit.predict(start=missing_indexes[0], end=missing_indexes[-1])
                # Assign predictions of this case to predictions dictionary.
                for ind in missing_indexes:
                    predictions[case][ind] = prediction[ind]
                # Clear missing duration gap cache.
                missing_indexes.clear()
    return predictions


# return: predictions (Dict)            - A nested dictionary of the form {testcase: {index: prediction}}.
def evaluate_linear_model():
    # Initialize dictionary to store associated original and predicted values.
    predictions = {'single_gap': dict(), 'double_gap': dict(), 'triple_gap': dict(), 'quad_gap': dict()}
    for case in evaluation_cases:
        # Identify missing indexes
        missing_indexes = [index for index, row in cases[case].iterrows() if pd.isna(row.values)]
        # Perform linear interpolation.
        prediction = cases[case].interpolate()
        # Convert df to series
        prediction = prediction.squeeze(axis=1)
        # Drop all non predicted values.
        for x, y in cases[case].iterrows():
            if x not in missing_indexes:
                prediction.drop(x, inplace=True)
        # Assign predictions of this case to predictions dictionary.
        for ind in missing_indexes:
            predictions[case][ind] = prediction[ind]
    return predictions


# arg: predictions  (Dict)              - A nested dictionary of the form {testcase: {index: prediction}}.
# arg: model_order (String)             - The model order if ARIMA, 'linear' if linear interpolation.
# return:                               - A dictionary of the form {model_order: Model}.
def calc_rmse(predictions, model_order):
    # Instantiate object to store model results, store within a dictionary.
    evaluated_models[model_order] = Model(model_order)
    # Determine RMSE for each test case.
    for case in evaluation_cases:
        err_sum = 0
        preds = sorted(predictions[case].items())
        for index, value in preds:
            err_sum += (value - cases['continuous_70day'].iloc[index])
        rmse = math.sqrt(err_sum**2 / len(preds))
        # Assign the case specific RMSE to the Model object.
        setattr(evaluated_models[model_order], 'err_'+case, rmse)
    # Following determination of RMSE for all cases, find the average.
    evaluated_models[model_order].set_err_avg()


# arg: measure (String)                 - The RMSE measure to rank models by (err_single_gap etc.)
# return:                               - Print the top 3 ranked models to the console.
def rank_models(measure):
    print('\nTop 3 models considering',measure,'(RMSE)')
    i = 0
    for model in (sorted(evaluated_models.values(), key=operator.attrgetter(measure))):
        print(model.model_type, '- '+str(getattr(model, measure)))
        i += 1
        if i == 3:
            return


# class: Used to store the evaluation metrics for each model (RMSE of various measures) in model specific objects.
class Model:
    def __init__(self, model_type):
        self.model_type = model_type

    def set_err_avg(self):
        self.err_avg = (self.err_single_gap + self.err_double_gap + self.err_triple_gap + self.err_quad_gap) / 4

    # Define instance variables for each model object, to be assigned after instantiation.
    err_single_gap: float
    err_double_gap: float
    err_triple_gap: float
    err_quad_gap: float
    err_avg: float

        
if __name__ == '__main__':
    ##
    ## LOAD DATASETS
    ##
    all_cases = ['continuous_1day', 'continuous_1week', 'continuous_70day', 'single_gap', 'double_gap', 'triple_gap', 'quad_gap']
    evaluation_cases = ['single_gap', 'double_gap', 'triple_gap', 'quad_gap']
    cases = load_data()

    ##
    ## INSIGHTS FOR ARIMA PARAMETER CONFIGURATION
    ##
    show_acf_plot('continuous_1week')  # Integration and Moving Average order determination
    show_pacf_plot('continuous_1week')  # Autoregressive order determination

    ##
    ## CREATE PREDICTIONS AND EVALUATE BY RMSE
    ##
    evaluated_models = dict()
    # ARIMA Model
    optimum_model_order = (1, 0, 0)
    valid_model_orders = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 2), (0, 2, 1), (0, 2, 2), (2, 0, 0))
    for model_order in valid_model_orders:
        arima_predictions = evaluate_arima_model(model_order)
        calc_rmse(arima_predictions, model_order)
    # Linear Interpolation Model
    linear_predictions = evaluate_linear_model()
    calc_rmse(linear_predictions, 'linear')

    ##
    ## SHOW PLOT OF ACTUAL AND PREDICTED VALUES OF THE BEST PERFORMING MODEL
    ##
    arima_predictions = evaluate_arima_model(optimum_model_order)
    show_prediction_plot(arima_predictions, optimum_model_order)
    show_prediction_plot(linear_predictions, 'linear')

    ##
    ## RANK MODELS BY A GIVEN RMSE METRIC
    ##
    rmse_metrics = ['err_avg', 'err_single_gap', 'err_double_gap', 'err_triple_gap', 'err_quad_gap']
    for metric in rmse_metrics:
        rank_models(metric)

    print('\nModel analysis complete.')





