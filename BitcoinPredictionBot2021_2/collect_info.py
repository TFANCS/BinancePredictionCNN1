from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
import mplfinance as mpf
import os
import const
import technical_indicators
import tensorflow as tf
import make_dataset


def collect_info(binance):

    df_list = {}
    df_list_test = {}

    print("collecting data")

    symbol = "LTCUSDT"


    df_list[symbol] = make_dataset.make_current_data(binance,symbol,128*2 + 70,70)
    df_list[symbol].to_csv("..\\Data\\" + symbol + "_data.csv") 


    img = make_dataset.make_current_image_data(binance,symbol,128*2 + 70,70)
    np.save("..\\Data\\" + symbol + "_image_data", img)


    for symbol in const.PAIR_SYMBOLS:
        base_asset = binance.get_base_asset(symbol)
        quote_asset = binance.get_quote_asset(symbol)


    for i in const.BALANCE_SYMBOLS:
        print(i + " Available : " + binance.get_balance(i)["free"])




    
