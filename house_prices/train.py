import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, \
    mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame, models_folder):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # save the model to a file called model.joblib in the models folder
    joblib.dump(model, os.path.join(models_folder, 'model.joblib'))

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmsle = compute_rmsle(y_test, y_pred)

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'rmsle': rmsle}
