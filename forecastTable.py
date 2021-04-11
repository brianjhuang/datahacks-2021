import numpy as np
import pandas as pd
from prophet import Prophet


def forecastTable(m, horizon, train_data, test_data, columns):
    """Build table for forecasting Prophet model.
    
    :params:
    m - fbprophet model
    horizon - int - number of days to forecast into future
    train_data - str - path to training observations
    test_data - str - path to test observations
    columns - list[str] - columns used as regressors
    date - pd.DateTime - date range of train data

    :returns:
    pd.DataFrame - cols: ds, y, *columns, index: datetimeIndex
    """
    normalize = lambda col: (col - col.mean()) / col.std()

    train_obs = pd.read_csv(train_data)
    test_obs = pd.read_csv(test_data)

    # normed_train_obs = train_obs.pivot_table(values="value", index="date", columns="series_id")
    # normed_train_obs = normed_train_obs.apply(normalize , axis=0)

    # normed_test_obs = test_obs.pivot_table(values="value", index="date", columns="series_id")
    # normed_train_obs = normed_train_obs.apply(normalize , axis=0)

    future = m.make_future_dataframe(periods=horizon)
    future.set_index(pd.to_datetime(future["ds"]), inplace=True)
    future.drop(columns="ds", inplace=True)

    for c in columns:
        train_feature = train_obs.query(f"""series_id == '{c}'""")
        train_feature.set_index(pd.to_datetime(train_feature["date"]), inplace=True)
        test_feature = test_obs.query(f"""series_id == '{c}'""")
        test_feature.set_index(pd.to_datetime(test_feature["date"]), inplace=True)
        mu = train_feature["value"].mean()
        s = train_feature["value"].std()
        future[c] = normalize(train_feature["value"]).append(
            ((test_feature["value"] - mu) / s)
        )  # must normalize test features to same linspace as training was done on
        future[c].fillna(method="bfill", inplace=True)
        future[c].fillna(method="ffill", inplace=True)

    future.reset_index(inplace=True)

    # print(future.head())

    # assert no nans
    assert all(future.isna().sum() == 0)

    # run forecasting
    forecast = m.predict(future)

    return forecast
