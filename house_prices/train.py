import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, \
    mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd
from house_prices.preprocess import splitting_df, useful_df
from house_prices.preprocess import transform_scaler_encoder
from house_prices.preprocess import fit_scaler_encoder, preprocessing


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    # split data into train and test sets
    train_df, test_df = splitting_df(data)

    # useful features and label column
    useful_features = ['Foundation', 'KitchenQual',
                       'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']
    label_column = 'SalePrice'

    train_df = useful_df(train_df, useful_features, label_column)
    test_df = useful_df(test_df, useful_features, label_column)

    # preprocess the data
    continuous_columns = ['TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']
    categorical_columns = ['Foundation', 'KitchenQual']
    y_train = train_df[label_column]
    y_test = test_df[label_column]
    X_train = preprocessing(train_df, categorical_columns, continuous_columns)
    X_test = preprocessing(test_df, categorical_columns, continuous_columns)

    # fit scaler and encoder on the train data
    scaler, encoder = fit_scaler_encoder(
        X_train, categorical_columns, continuous_columns)

    X_train = transform_scaler_encoder(X_train,
                                       categorical_columns, continuous_columns,
                                       os.path.abspath('../models'))
    X_test = transform_scaler_encoder(X_test,
                                      categorical_columns, continuous_columns,
                                      os.path.abspath('../models'))

    # train the model on the train data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # save the model to a file called model.joblib in the models folder
    joblib.dump(model, os.path.join(
        os.path.abspath('../models'), 'model.joblib'))

    predictions = model.predict(X_test)

    # metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmsle = compute_rmsle(y_test, predictions)

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'rmsle': rmsle}
