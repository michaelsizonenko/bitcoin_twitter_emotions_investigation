import math
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import statsmodels.tsa.seasonal as statsmodels_seasonal
import statsmodels.tsa.statespace.sarimax as statsmodels_sarimax
from statsmodels.tsa.vector_ar.vecm import coint_johansen
# add using vector autoregression to create 1 cln and use it further

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
    predicted_df = pd.DataFrame({"value": values}, index=dates)
    return predicted_df


def naive_prediction(train_df, observations_to_predict, **kwargs):
    values = [train_df.iat[-1, 0] for i in range(observations_to_predict)]
    return create_data_frame(values, train_df.index[-1])


def average_prediction(train_df, observations_to_predict, **kwargs):
    values = [train_df["value"].mean() for i in range(observations_to_predict)]
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
    errors["MAE"] = sk_metrics.mean_absolute_error(true_df["value"], predicted_df["value"])
    errors["RMSE"] = math.sqrt(sk_metrics.mean_squared_error(true_df["value"], predicted_df["value"]))
    errors["RMSLE"] = math.sqrt(sk_metrics.mean_squared_log_error(true_df["value"], predicted_df["value"]))
    return errors


plt.figure(figsize=(20, 10))

utility_index_df = pd.read_csv("dataset/btc_emotions", parse_dates=["date"])
utility_index_df.set_index("date", inplace=True)
utility_index_df.index.freq = utility_index_df.index.inferred_freq


print(len(utility_index_df))
print(utility_index_df.index.min())
print(utility_index_df.index.max())

number_of_splits = 6
utility_index_cv_splits = make_cv_splits(utility_index_df, number_of_splits)

test_data = pd.concat([t for (_, t) in utility_index_cv_splits])

naive_predictions = make_cv_prediction(utility_index_cv_splits, naive_prediction)
naive_errors = calculate_errors(test_data, naive_predictions)
print("Naive errors:", naive_errors)

average_predictions = make_cv_prediction(utility_index_cv_splits, average_prediction)
average_errors = calculate_errors(test_data, average_predictions)
print("Average errors:", average_errors)

sarima_order_kwargs = {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)}
sarima_predictions = make_cv_prediction(utility_index_cv_splits, sarima_prediction, **sarima_order_kwargs)
sarima_errors = calculate_errors(test_data, sarima_predictions)
print("SARIMA errors:", sarima_errors)

plt.plot(utility_index_df.index, utility_index_df["value"])

plt.plot(naive_predictions.index, naive_predictions["value"], color="y")
plt.plot(average_predictions.index, average_predictions["value"], color="y")
plt.plot(sarima_predictions.index, sarima_predictions["value"], color="y")

sarima_extrapolation = sarima_prediction(utility_index_df, 80, **sarima_order_kwargs)
plt.plot(sarima_extrapolation.index, sarima_extrapolation["value"], color="g")

utility_index_additive_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df, model="additive")
utility_index_additive_decomposition.plot()
utility_index_multiplicative_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df,
                                                                                     model="multiplicative")
utility_index_multiplicative_decomposition.plot()

plt.show()
