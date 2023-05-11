import os
import joblib
import pandas as pd
import numpy as np
from house_prices.preprocess import preprocessing, transform_scaler_encoder


def make_predictions(test_df: pd.DataFrame) -> np.ndarray:
    # useful features and label column
    useful_features = ['Foundation', 'KitchenQual',
                       'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']
    continuous_columns = ['TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']
    categorical_columns = ['Foundation', 'KitchenQual']

    # df with only the useful features
    house_test = test_df[useful_features]

    # Preprocess the data usinf function from preprocess.py
    preprocessed_test = preprocessing(
        house_test, categorical_columns, continuous_columns)
    transformed_test = transform_scaler_encoder(preprocessed_test,
                                                categorical_columns,
                                                continuous_columns,
                                                os.path.abspath('../models'))

    # using joblib.load to make the predictions
    model = joblib.load(os.path.join(os.path.abspath('../models'),
                                     'model.joblib'))
    predictions = model.predict(transformed_test)

    return predictions
