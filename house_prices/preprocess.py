import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from typing import List, Tuple


def splitting_df(initial_df: pd.DataFrame):
    train_df, test_df = train_test_split(initial_df,
                                         test_size=0.33, random_state=42)
    return train_df, test_df


def useful_df(data_df: pd.DataFrame, useful_features: List[str],
              label_column: str) -> pd.DataFrame:
    # Final df with only the useful features and the label
    data_df = data_df[useful_features + [label_column]]
    return data_df


def preprocessing(data: pd.DataFrame, categorical_columns: List[str],
                  continuous_columns: List[str]) -> pd.DataFrame:
    # Replace missing values in categorical columns with mode
    data[categorical_columns] = data[categorical_columns].fillna(
        data[categorical_columns].mode().iloc[0])
    # Replace missing values in continuous columns with mean
    data[continuous_columns] = data[continuous_columns].fillna(
        data[continuous_columns].mean())
    # Remove duplicated rows
    # data = data.drop_duplicates(keep='first')
    # Reset index
    preprocess_df = data.reset_index(drop=True)

    return preprocess_df


def fit_scaler_encoder(df_train: pd.DataFrame, categorical_columns: List[str],
                       continuous_columns: List[str]) \
        -> Tuple[StandardScaler, OneHotEncoder]:
    # Fitting the scaler on the train_df
    scaler = StandardScaler()
    scaler.fit(df_train[continuous_columns])

    # Fitting the encoder on the train_df
    encoder = OneHotEncoder()
    encoder.fit(df_train[categorical_columns])

    return scaler, encoder


def transform_scaler_encoder(data_df: pd.DataFrame,
                             categorical_columns: List[str],
                             continuous_columns: List[str],
                             models_folder: str) -> pd.DataFrame:
    # Using the saved encoder and scalers
    scaler = joblib.load(os.path.join(models_folder, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(models_folder, 'encoder.joblib'))
    scaled_data = scaler.transform(data_df[continuous_columns])
    scaled_df = pd.DataFrame(data=scaled_data, columns=continuous_columns)
    encoded_data = encoder.transform(data_df[categorical_columns])
    encoded_df = pd.DataFrame(data=encoded_data.toarray(),
                              columns=encoder.get_feature_names_out(
                                  categorical_columns))

    transformed_df = pd.concat([scaled_df, encoded_df], axis=1)

    return transformed_df
