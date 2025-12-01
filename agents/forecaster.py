import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class Forecaster:
    def forecast_next(self, df, column, days=7):
        y = df[column].values
        X = np.arange(len(y)).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        future_x = np.arange(len(y), len(y) + days).reshape(-1, 1)
        preds = model.predict(future_x)

        return preds.tolist()
