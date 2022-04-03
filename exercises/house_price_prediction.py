from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def date_to_int(dates: np.ndarray) -> np.ndarray:
    new_dates = dates.view((str, 1)).reshape(len(dates), -1)[:, 0:6]
    return np.fromstring(new_dates.tostring(), dtype=(str, 6))


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(str,
                     dtype={"id": "np.int64", "price": "np.int32", "bedrooms": "np.int32", "bathrooms": "np.float32",
                            "sqft_living": "np.int32", "sqft_lot": "np.int32", "floors": "np.float32",
                            "waterfront": "np.int32", "view": "np.int32", "condition": "np.int32", "grade": "np.int32",
                            "sqft_above": "np.int32", "sqft_basement": "np.int32", "yr_built": "np.int32",
                            "yr_renovated": "np.int32", "zipcode": "np.int32", "lat": "np.float64",
                            "long": "np.float32", "sqft_living15": "np.int32", "sqft_lot15": "np.int32"})
    # df = df.drop(df[df.id == 0].index) todo this is useful to delete rows by condition on cols
    dates = df[['date']].to_numpy().flatten()
    df.update(pd.DataFrame({'date': date_to_int(dates)}))
    df['date'] = df['date'].astype(np.int32)
    df = df[
        (df["id"] >= 1) and
        (df["date"] >= 20000000 and df["date"] <= 20220000) and
        (df["price"] >= 50000 and df["price"] <= 10000000) and
        (df["bedrooms"] >= 0 and df["bedrooms"] <= 11) and
        (df["bathrooms"] >= 0 and df["bathrooms"] <= 10) and
        (df["sqft_living"] >= 250 and df["sqft_living"] <= 10000) and
        (df["sqft_lot"] >= 450 and df["sqft_lot"] <= 1800000) and
        (df["floors"] >= 1 and df["floors"] <= 4) and
        (df["waterfront"] == 0 or df["waterfront"] == 1) and
        (df["view"] >= 0 and df["view"] <= 4) and
        (df["condition"] >= 1 and df["condition"] <= 5) and
        (df["grade"] >= 1 and df["grade"] <= 13) and
        (df["sqft_above"] >= 250 and df["sqft_above"] <= 10000) and
        (df["sqft_basement"] >= 0 and df["sqft_basement"] <= 5000) and
        (df["yr_built"] >= 1800 and df["yr_built"] <= 2022) and
        (df["yr_renovated"] >= 0 and df["yr_renovated"] <= 2022) and
        (df["zipcode"] >= 98000 and df["zipcode"] <= 99000) and
        (df["lat"] >= 47 and df["lat"] <= 48) and
        (df["long"] >= -123 and df["long"] <= -121) and
        (df["sqft_living15"] >= 350 and df["sqft_living15"] <= 7000) and
        (df["sqft_lot15"] >= 500 and df["sqft_lot15"] <= 1000000)
        ]

    # inserting the "yr_renovated" col the last year in which the building had had any renovation.
    df["yr_renovated"] = df[["yr_built", "yr_renovated"]].max(axis=1)

    prices_by_zipcode = pd.DataFrame({
        'zipcode': df[['zipcode']],
        'price': df[['price']]})
    mean_price = prices_by_zipcode.groupby('zipcode').mean()

    df.merge(mean_price, on='zipcode_mean_price', how='left')

    df = df.drop(['id','zipcode'], 1) #the df may need to change to pandas.DataFrame
    return df



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
