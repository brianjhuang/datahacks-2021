def columnCleaner(path, columns, date = "2011-02-14"):

    """
    This function cleans and returns a dataframe that can be fit to our Prophet model.

    path: the file path/csv path that we want to read
    columns: the columns that we want to fit
    date: the date range we want for the SP500

    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    messy_obs = pd.read_csv(path)
    #read the csv file

    messy_obs['date'] = pd.to_datetime(messy_obs['date'])
    #convert to date time

    obs_pivot = messy_obs.pivot(values = "value", index = "date", columns = "series_id")
    #pivot the table

    normalize = lambda col: (col-col.mean())/col.std()
    normed_obs = obs_pivot = obs_pivot.apply(normalize, axis=0)
    #normalize data

    observations = normed_obs[normed_obs.index >= date]
    #filter out any dates that are before designated date

    df = pd.DataFrame({'ds':observations.index})
    df.set_index('ds', inplace = True)
    #create the dataframe we return

    #code above organizes table into usable format

    for col in columns:
        fit = pd.DataFrame({'ds': observations.index, 'y': observations['SP500'], str(col): observations[str(col)]})
        #create our table for the column entry in our columns list

        fit.fillna(method = 'bfill', inplace = True)
        fit.fillna(method = 'ffill', inplace = True)
        #filter out any null values

        fit_column = fit[str(col)]
        #gets a series for our respective column

        df[str(col)] = fit_column
        #adds that column to our dataframe

    df.reset_index(inplace = True)

    return df
