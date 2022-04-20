import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) ->pd.DataFrame:
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
    df = pd.read_csv(filename)
    df.fillna(0, inplace=True)
    df[["id", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
        "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]] = df[
        ["id", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
         "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
         "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]].apply(pd.to_numeric)

    df['date'] = df['date'].astype("str").apply(lambda s: s[:8])
    df['date'] = df['date'].astype('float64')

    df = df[
        (df["id"] >= 1) &
        (df["date"] >= 20000000) & (df["date"] <= 20220000) &
        (df["price"] >= 50000) & (df["price"] <= 10000000) &
        (df["bedrooms"] >= 0) & (df["bedrooms"] <= 15) &
        (df["bathrooms"] >= 0) & (df["bathrooms"] <= 12) &
        (df["sqft_living"] >= 200) & (df["sqft_living"] <= 100000) &
        (df["sqft_lot"] >= 450) & (df["sqft_lot"] <= 1800000) &
        (df["floors"] >= 1) & (df["floors"] <= 4) &
        (df["waterfront"] == 0) | (df["waterfront"] == 1) &
        (df["view"] >= 0) & (df["view"] <= 4) &
        (df["condition"] >= 1) & (df["condition"] <= 5) &
        (df["grade"] >= 1) & (df["grade"] <= 13) &
        (df["sqft_above"] >= 250) & (df["sqft_above"] <= 10000) &
        (df["sqft_basement"] >= 0) & (df["sqft_basement"] <= 5000) &
        (df["yr_built"] >= 1800) & (df["yr_built"] <= 2022) &
        (df["yr_renovated"] >= 0) & (df["yr_renovated"] <= 2022) &
        (df["zipcode"] >= 98000) & (df["zipcode"] <= 99000) &
        (df["lat"] >= 47) & (df["lat"] <= 48) &
        (df["long"] >= -123) & (df["long"] <= -121) &
        (df["sqft_living15"] >= 300) & (df["sqft_living15"] <= 10000) &
        (df["sqft_lot15"] >= 300) & (df["sqft_lot15"] <= 1000000)
        ]

    # inserting the "yr_renovated" col the last year in which the building had had any renovation.
    df["yr_renovated"] = df[["yr_built", "yr_renovated"]].max(axis=1)

    prices_by_zipcode = pd.DataFrame({'zipcode': df['zipcode'], 'price': df['price']})
    prices_by_zipcode = prices_by_zipcode.groupby('zipcode').mean()
    prices_by_zipcode.rename(columns={'price': 'mean_price'}, inplace=True)
    df = pd.merge(df, prices_by_zipcode, on='zipcode')

    df = df.drop(['id', 'zipcode', 'lat', 'long'], 1)
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
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for (feature, feature_data) in X.iteritems():
        r = str(feature_data.cov(y) / (feature_data.std() * y.std()))
        fig = go.Figure([go.Scatter(x=feature_data, y=y, mode='markers')],
                        layout=go.Layout(title="prices as a function of " + feature + " with Peasron Correlation:" + r,
                                         xaxis_title=feature, yaxis_title="prices", height=300))
        fig.write_image(os.path.join(output_path, feature + ".pdf"))


def fit_increasing_size(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, test_y: pd.Series):
    samples_p = np.arange(0.1, 1.01, 0.01)
    lin = LinearRegression(include_intercept=True)
    loss_avg = np.zeros(samples_p.size)
    loss_var = np.zeros(samples_p.size)
    curr_loss = np.zeros(10)
    for i, p in enumerate(samples_p):
        for j in range(10):
            void1, void2, samples_X, samples_y = split_train_test(train_X, train_y, p)
            lin._fit(samples_X.to_numpy(), samples_y.to_numpy())
            curr_loss[j] = lin._loss(test_X.to_numpy(), test_y.to_numpy())
        loss_avg[i] = curr_loss.mean()
        loss_var[i] = curr_loss.var()
    fig = go.Figure([go.Scatter(x=samples_p*100, y=loss_avg, mode='markers',showlegend=False),
                     go.Scatter(x=samples_p*100, y=loss_avg - 2 * np.sqrt(loss_var), fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=samples_p*100, y=loss_avg + 2 * np.sqrt(loss_var), fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(title="Loss mean and its variance as a function of percentage of sample size",
                                     xaxis_title="percentage of sample size", yaxis_title="Loss", height=500))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data("C:/Users/user/Desktop/Tom/iml/IML.HUJI-main/datasets/house_prices.csv")
    df = df.astype('int32')
    prices = df['price']
    df = df.drop(['price'], 1)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, prices, "plots_for_q2")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, prices, 0.25)

    # Question 4 - Fit model over increasing percentages of the overall training data
    fit_increasing_size(train_X, train_y, test_X, test_y)
