import os
import joblib
import pandas as pd
import numpy as np


def make_predictions(test_df: pd.DataFrame, models_folder: str) -> np.ndarray:
    # using joblib.load to make the predictions
    model = joblib.load(os.path.join(models_folder, 'model.joblib'))
    predictions = model.predict(test_df)

    return predictions
