import pandas as pd
import numpy as np


def df_of_coefs(lin_model: np.ndarray, columns: np.ndarray):
    """
    Inputs: 
    1. lin_model: linear model from sklearn like LinearRegression.
    2. columns: list of names of the features of the model
    
    Output: DataFrame that sumarize coeficients and features
    """
    coefs = lin_model.coef_[0]
    df = pd.DataFrame()
    df["Features"] = columns
    df["Coeficientes"] = coefs
    return df


def df_evaluation_model(X, y, y_pred, name="train"):
    """
    Inputs:
    1. X: ndarray with the features we want to evaluate the model
    2. y: ndarray with the real target values
    3. y_pred: ndarray with the predicted values of some model
    4. name: str with the type of evaluation data (train is default; cv=cross_validation; test=test_set)
    """
    df = pd.DataFrame(X.copy())
    df[f"y_{name}"] = y
    df[f"y_{name}_pred"] = y_pred
    df["|Errores|"] = np.abs(y - y_pred)
    df["Errores^2"] = (y - y_pred)**2
    return df