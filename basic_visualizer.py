from black_scholes import VolViz
import streamlit as st
import json
import yfinance as yf
import pandas as pd
import datetime
from datetime import date
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def convert_date_to_numeric(d):
    """
    Converts the datatype to a numeric value.

    Args:
        date (obj): the date from the dataframe

    Returns:
        an integer value for the date, relative to today.
    """
    if type(d) != "date":
        # convert date to string
        d = str(d)
        d = d.split("-")
        # split by - separator
        d = datetime.date(year=int(d[0]), month=int(d[1]), day=int(d[2]))
    # get the numeric value in second
    timestamp = time.mktime(d.timetuple())
    # there's 86400 seconds in a day, so convert it
    return timestamp / 86400

# Now, we need to calculate for time 
def calculate_T(expiration):
    """
    Calculates T  given the date of expiration,
    using datetime to extract the date.

    Args:
        expiration (str): The string form of the expiration date
    """
    # get current date
    expiration = convert_date_to_numeric(expiration)
    # convert dates to numeric values
    today = convert_date_to_numeric(date.today())
    # return T, which would be (expiration - current date) / 365
    return expiration - today

def get_price(ticker):
    """gets the current price of an asset

    Returns:
        float_: the price of an underlying asset, S
    """
    msft = yf.Ticker(ticker)
    S = float(msft.history(period="1d")["Close"][-1])
    return S

with open("stocks.json", "r") as f:
    s = f.read()
    d = json.loads(s)


st.title("Implied Volatility Visualizer")

st.subheader("Please pick a stock to see the implied volatility")

stock_choice = st.selectbox("Please pick a stock to see the implied volatility graph!", tuple(d.keys()), index=None, placeholder="Select a stock...")
st.subheader("Dataframe")

if stock_choice:
    # extract stock choice
    tk = yf.Ticker(d[stock_choice])
    # extract expiration dates
    exps = tk.options
    df = pd.DataFrame()
    for exp in exps:
        call_options = tk.option_chain(exp).calls
        call_options["expirationDate"] = exp
        df = pd.concat([df, call_options])
    df["T"] = df["expirationDate"].apply(lambda x: calculate_T(x))
    price = get_price(d[stock_choice])
    df["moneyness"] = df["strike"] / price
    # showcase dataframe
    st.dataframe(data=df)
    # showcase scatter plot implied volatility
    # create figure and axis
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # make 3-D data. x will be strike, y axis is time to expiry, and z being implied volatility.
    # since it's a 3d plot, x, y, and z must be 2 dimensional.
    X = df["strike"]
    Y = df["T"]
    Z = df["impliedVolatility"]
    # tri-surface
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(X, Y, Z,
                       cmap="viridis",
                       linewidth=0.1,
                       antialiased=True,
                       alpha=0.8)
    ax.set_xlabel("Moneyness (S/K)", labelpad=10)
    ax.set_ylabel("Time to Expiration", labelpad=10)
    ax.set_zlabel("Implied Volatility", labelpad=10)
    plt.title("Implied Volatility Surface", pad=20, size=14)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    st.pyplot(fig)


