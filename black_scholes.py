import numpy as np
import streamlit as sl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import time
import yfinance as yf
import math
from scipy.interpolate import griddata
import datetime
from scipy.stats import norm

#################
# BLACK SCHOLES #
#################

def black_scholes(S, K, T, r, sigma, q=0, option_type="call"):
    """_summary_

    Args:
        S (float): _description_
        K (_type_): _description_
        T (_type_): _description_
        r (_type_): _description_
        sigma (_type_): _description_
        q (int, optional): _description_. Defaults to 0.
        option_type (str, optional): _description_. Defaults to "call".

    Returns:
        _type_: _description_
    """
    d1 = float(float(math.log(S / K)) + float(r + (sigma**2) / 2) / float(sigma * math.sqrt(T)))
    d2 = float(d1 - (sigma * math.sqrt(T)))
    normal_distribution = norm(loc=0, scale=1)
    if option_type == "call":
        # it's a call option
        return S * (math.e**(-q * T)) * normal_distribution.cdf(d1) - K * (math.e**(-r * T)) * normal_distribution.cdf(d2)
    else:
        # it's a put, so we'll input the function for a put call
        return K * (math.e**(-r * T)) * normal_distribution.cdf(-d2) - S * (math.e**(-q * T)) * normal_distribution.cdf(-d1)

##########
# GREEKS #
##########

def vega(S, K, T, r, sigma, q=0, option_type="call"):
    d1 = ((math.log(S / K) + (r + (sigma**2) / 2)) / (sigma * math.sqrt(T)))
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)

def loss(S, K, T, r, price, sigma_guess=0.2, q = 0, option_type="call"):
    """_summary_

    Args:
        S (_type_): _description_
        K (_type_): _description_
        T (_type_): _description_
        r (_type_): _description_
        price (_type_): _description_
        sigma_guess (float, optional): _description_. Defaults to 0.2.
        q (int, optional): _description_. Defaults to 0.
        option_type (str, optional): _description_. Defaults to "call".

    Returns:
        _type_: _description_
    """
    # Price with the GUESS for the volatility
    theoretical_price = black_scholes(S, K, T, r, sigma_guess, q, option_type)
    
    # Actual Price
    market_price = price
    
    return theoretical_price - market_price

def solve_for_iv(S, K, T, r, price, sigma_guess = 0.2, q=0, option_type="call"):
    """_summary_

    Args:
        S (float): Underlying Price
        K (float): Strike Price
        T (float): date to expiration (days)
        r (float): risk free rate
        price (_type_): _description_
        sigma_guess (float, optional): _description_. Defaults to 0.8.
        q (int, optional): _description_. Defaults to 0.
        option_type (str, optional): _description_. Defaults to "call".

    Returns:
        _type_: _description_
    """
    tolerance = 1e-6
    for i in range(100):
        price_diff = loss(S, K, T, r, price, sigma_guess, q, "call")
        v = vega(S, K, T, r, sigma_guess)
        if v == 0:
            break
        next_sigma = sigma_guess - (price_diff / v)
        if abs(next_sigma - sigma_guess) < tolerance:
            return next_sigma
        sigma_guess = next_sigma
    return float(sigma_guess)

################
# GETTING DATA #
################

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

class VolViz:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = pd.read_csv("microsoft.csv")
        self.price = self.get_price()

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
    
    # the underlying price is the current price of the asset
    def get_price(self):
        """gets the current price of an asset

        Returns:
            float_: the price of an underlying asset, S
        """
        msft = yf.Ticker(self.ticker)
        S = float(msft.history(period="1d")["Close"][-1])
        return S
    
    def row_func(self, x):
        price = self.get_price()
        return solve_for_iv(price, x["strike"], x["T"], .0454, x["lastPrice"])
    
    def generate_df(self, ticker):
        tk = yf.Ticker(ticker)
        exps = tk.options
        df = pd.DataFrame()
        for exp in exps:
            call_options = tk.option_chain(exp).calls
            call_options["expirationDate"] = exp
            df = pd.concat([df, call_options])
        df["T"] = df["expirationDate"].apply(lambda x: calculate_T(x))
        df["impliedVolatilityNoAssumptions"] = df.apply(lambda x: self.row_func(x), axis=1)
        df["moneyness"] = df["strike"] / self.get_price()
        return df
    
    def visualize_scatter(self):
        # create figure and axis
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # make 3-D data. x will be strike, y axis is time to expiry, and z being implied volatility.
        # since it's a 3d plot, x, y, and z must be 2 dimensional.
        X = self.df["strike"]
        Y =self.df["T"]
        Z = self.df["impliedVolatility"]

        fig = plt.figure()

        ax.scatter(X, Y, Z, c=X + Y)
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Days to Maturity")
        ax.set_title("Microsoft's Implied Volatility Surface")
        return fig, ax

    def visualize_surface(self):
        # create figure and axis
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # make 3-D data. x will be strike, y axis is time to expiry, and z being implied volatility.
        # since it's a 3d plot, x, y, and z must be 2 dimensional.
        X = self.df["strike"]
        Y = self.df["T"]
        x, y = np.meshgrid(X, Y)
        R = np.sqrt(x**2, y**2)
        Z = griddata((X, Y), self.df["impliedVolatility"], (x, y), method="nearest")

        ax = fig.add_subplot(111, projection="3d")

        surface = ax.plot_surface(x, y, Z, cmap="viridis", edgecolor="none")


        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Days to Maturity")
        ax.set_title("Microsoft's Implied Volatility Surface")

        ax.view_init(elev=30, azim=-60) # Changed to improve viewing angle

        # Add a color bar
        fig.colorbar(surface, shrink=0.5, aspect=5)

        return fig, ax
        