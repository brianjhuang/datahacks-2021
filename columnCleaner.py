def columnCleaner(path, columns, date="2011-02-14", normalize_sp5=True):

    """
    This function cleans and returns a dataframe that can be fit to our Prophet model.

    path: the file path/csv path that we want to read
    columns: the columns that we want to fit
    date: the date range we want for the SP500

    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if isinstance(path, str):
        messy_obs = pd.read_csv(path)
    else:
        messy_obs = path.copy(deep=True)  # if path is a dataframe
    # read the csv file

    messy_obs["date"] = pd.to_datetime(messy_obs["date"])
    messy_obs["series_id"] = messy_obs["series_id"].str.strip()
    # convert to date time

    obs_pivot = messy_obs.pivot(values="value", index="date", columns="series_id")
    # pivot the table
    temp = obs_pivot[obs_pivot.index >= date]

    mu = temp["SP500"].mean()
    s = temp["SP500"].std()

    normalize = lambda col: (col - col.mean()) / col.std()
    normed_obs = obs_pivot.apply(normalize, axis=0)
    # normalize data

    observations = obs_pivot = normed_obs[normed_obs.index >= date]
    # filter out any dates that are before designated date

    assert len(messy_obs["series_id"].unique()) == len(observations.columns)
    assert set(columns) not in set(observations.columns)

    if normalize_sp5:
        df = pd.DataFrame({"ds": observations.index, "y": observations["SP500"]})
    else:
        df = pd.DataFrame({"ds": observations.index, "y": temp["SP500"]})
    df.set_index("ds", inplace=True)
    # create the dataframe we return
    # try:
    #     observations.drop(np.nan, axis=1, inplace=True)
    # except KeyError:
    #     continue

    # print(observations.columns)
    # code above organizes table into usable format

    for col in columns:
        b = False
        try:
            fit = pd.DataFrame(
                {"ds": observations.index, str(col): observations[str(col)]}
            )
        except Exception as e:
            k = col in observations.columns
            print(f"Dropping Column: {col}", type(e), e, k)
            b = True

        if not b:
            # create our table for the column entry in our columns list
            fit.fillna(method="bfill", inplace=True)
            fit.fillna(method="ffill", inplace=True)
            # filter out any null values

            fit_column = fit[str(col)]
            # gets a series for our respective column

            df[str(col)] = fit_column
            # adds that column to our dataframe

    df.reset_index(inplace=True)

    return df, mu, s
