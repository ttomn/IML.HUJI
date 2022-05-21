from plotly.subplots import make_subplots

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"
days_until_month = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]


def parse_day_of_year(df):
    months = df["Month"].to_numpy()
    days = df["Day"].to_numpy()
    years = df["Year"].to_numpy()
    day_of_year = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        day_of_year[i] = days_until_month[months[i]] + days[i]
        if years[i] % 4 == 0 and months[i] > 2:
            day_of_year[i] += 1
    return day_of_year


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates={"date": [2]})
    df.fillna(0, inplace=True)
    df[["Year", "Month", "Day", "Temp"]].apply(pd.to_numeric)

    df = df[
        (df["Year"] >= 1800) & (df["Year"] <= 2022) &
        (df["Month"] >= 1) & (df["Month"] <= 12) &
        (df["Day"] >= 1) & (df["Day"] <= 31) &
        (df["Temp"] >= -50) & (df["Temp"] <= 60)]

    days = parse_day_of_year(df)
    df.insert(0, "DayOfYear", days)
    return df


def q2(df_israel):
    dkk = df_israel['Year'].astype("str")
    fig = px.scatter(df_israel, x='DayOfYear', y='Temp', color=dkk,
                     title="Temperature at any day of the year in Israel",
                     labels={"DayOfYear": "day of the year", "Temp": "temperature in Celsius", "color": "year"})
    fig.show()

    israel_by_month = df_israel.groupby('Month').agg({'Temp': 'std'})
    fig2 = px.bar(israel_by_month, x=[i for i in range(1, 13)], y="Temp",
                  title="Standard division of the temperature for every month in Israel",
                  labels={"x": "month", "Temp": "Standard division of the temp temperature"})
    fig2.show()


def q3(df):
    grouped_data = df.groupby(['Country', 'Month'], as_index=False).agg(temp_mean=('Temp', 'mean'),
                                                                        temp_std=('Temp', 'std'))
    fig = px.line(grouped_data, x='Month', y='temp_mean', error_y='temp_std', color='Country',
                  title="mean temperature and its variance in each country for every month",
                  labels={"temp_mean": "mean temperature", "Month": "month"})
    fig.show()


def q4(df_israel):
    israel_day = pd.DataFrame(df_israel['DayOfYear'])
    israel_temp = df_israel['Temp']
    day_train, temp_train, day_test, temp_test = split_train_test(israel_day, israel_temp, 0.25)
    loss_array = np.zeros(10)
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly._fit(day_train['DayOfYear'].to_numpy(), temp_train.to_numpy())
        loss = round(poly._loss(day_test['DayOfYear'].to_numpy(), temp_test.to_numpy()), 2)
        print("the degree is " + str(k) + " and the loss is " + str(loss))
        loss_array[k - 1] = loss
    k_values = [i for i in range(1, 11)]
    k_df = pd.DataFrame({'k': k_values, 'loss': loss_array})
    fig = px.bar(k_df, x='k', y='loss', title="Loss value for every degree", labels={"k": "degree"})
    fig.show()


def q5(df):
    df_israel = df[(df["Country"] == "Israel")]
    k = 5
    poly = PolynomialFitting(k)
    poly._fit(df_israel['DayOfYear'].to_numpy(), df_israel['Temp'].to_numpy())
    countries = df['Country'].unique()
    countries = countries[countries != 'Israel']
    loss_per_country = np.zeros(len(countries))
    for i in range(len(countries)):
        curr_country = df[(df['Country'] == countries[i])]
        loss_per_country[i] = poly._loss(curr_country['DayOfYear'].to_numpy(), curr_country['Temp'].to_numpy())
    countries_loss = pd.DataFrame({'countries': countries, 'loss': loss_per_country})
    fig = px.bar(countries_loss, x='countries', y='loss', title="Loss value for every other country with polynom of degree "+ str(k))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:\\Users\\ttomn\\OneDrive\\Desktop\\IML\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[(df["Country"] == "Israel")]
    q2(df_israel)

    # Question 3 - Exploring differences between countries
    q3(df)

    # Question 4 - Fitting model for different values of `k`
    q4(df_israel)

    # Question 5 - Evaluating fitted model on different countries
    q5(df)
