import math
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import statsmodels.tsa.statespace.sarimax as statsmodels_sarimax
from statsmodels.tsa.statespace.varmax import VARMAX
import datetime

# add using vector autoregression to create 1 cln and use it further
plt.figure(figsize=(20, 10))

#functions for monitoring model
def make_cv_splits(data_df, n_splits):
    time_series_cv_splits = sk_model_selection.TimeSeriesSplit(n_splits=n_splits)
    data_df_cv_splits_indices = time_series_cv_splits.split(data_df)

    data_df_cv_splits = []
    for train_indices, test_indices in data_df_cv_splits_indices:
        train, test = data_df.iloc[train_indices], data_df.iloc[test_indices]
        data_df_cv_splits.append((train, test))

    data_df_cv_splits.pop(0)
    return data_df_cv_splits


def create_data_frame(values, last_date):
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=(len(values)), freq="MS")
    predicted_df = pd.DataFrame({"close": values}, index=dates)
    return predicted_df


def naive_prediction(train_df, observations_to_predict, **kwargs):
    values = [train_df.iat[-1, 0] for i in range(observations_to_predict)]
    return create_data_frame(values, train_df.index[-1])


def average_prediction(train_df, observations_to_predict, **kwargs):
    values = [train_df["close"].mean() for i in range(observations_to_predict)]
    return create_data_frame(values, train_df.index[-1])


def sarima_prediction(train_df, observations_to_predict, **kwargs):
    sarima_model = statsmodels_sarimax.SARIMAX(train_df, order=kwargs["order"], seasonal_order=kwargs["seasonal_order"])
    sarima_model_fit = sarima_model.fit(disp=False)
    values = sarima_model_fit.forecast(observations_to_predict)
    return create_data_frame(values, train_df.index[-1])


def make_cv_prediction(cv_splits, model, **kwargs):
    predictions = []
    for train_df, test_df in cv_splits:
        predicted_df = model(train_df, len(test_df), **kwargs)
        predictions.append(predicted_df)
    return pd.concat(predictions)


def calculate_errors(true_df, predicted_df):
    errors = dict()
    errors["MAE"] = sk_metrics.mean_absolute_error(true_df["close"], predicted_df["close"])
    errors["RMSE"] = math.sqrt(sk_metrics.mean_squared_error(true_df["close"], predicted_df["close"]))
    errors["RMSLE"] = math.sqrt(sk_metrics.mean_squared_log_error(true_df["close"], predicted_df["close"]))
    return errors


def main():
    start = datetime.datetime.now()
    print(start)
    data_df = pd.read_csv('dataset/btc_emotions', parse_dates=['date'])
    data_df.set_index("date", inplace=True)
    data_df.index.freq = data_df.index.inferred_freq
    number_of_splits = 6
    # utility_index_cv_splits = make_cv_splits(data_df, number_of_splits)
    model = VARMAX(data_df, order=(0, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.forecast()
    plt.plot(data_df.index, data_df["close"])
    pred_df = create_data_frame(yhat, data_df.index[-1])
    print(pred_df)


if __name__ == '__main__':
    main()
