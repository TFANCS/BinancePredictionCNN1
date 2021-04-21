from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import tensorflow as tf
import os
import const
import make_dataset
import matplotlib.pyplot as plt





def sell(base_balance, quote_balance, quantity, price):
    base_balance -= quantity
    quote_balance += quantity*price * 0.999
    return base_balance, quote_balance


def buy(base_balance, quote_balance, quantity, price):
    quote_balance -= quantity*price * 0.999
    base_balance += quantity
    return base_balance, quote_balance







def simulation(binance,model):

    df_list = {}

    for symbol in const.PAIR_SYMBOLS:
        df_list[symbol] = make_dataset.make_current_data(binance,symbol,12,10)
        #mpf.plot(df_list[symbol], type='candle')

    symbol = "LTCUSDT"

    first_base_balance = 1.0
    first_quote_balance = 100
    base_balance = first_base_balance
    quote_balance = first_quote_balance
    price = df_list[symbol].iloc[0,0]
    #price = float(binance.get_ticker("BTCUSDT")["lastPrice"])

    data,target = make_dataset.make_dataset(df_list[symbol])
    print(df_list[symbol])
    print(data)



    model.load_weights(const.CHECKPOINT_PATH)
    prediction = pd.DataFrame(model.predict(data))
    prediction = prediction.idxmax(axis=1)


    
    first_balance = (base_balance*price)+quote_balance


    for i in range(len(prediction)):
        print("Period:" + str(i))
        price = df_list[symbol].iloc[i+const.TIME_LENGTH,0]
        print("Price:"+str(price))
        print("Balance:"+str((base_balance*price)+quote_balance) + " Base:" + str(base_balance) + " Quote:" + str(quote_balance))
        if prediction[i] == 0:
            print("SELL")
            base_balance, quote_balance = sell(base_balance, quote_balance,0.0002,price)
        elif prediction[i] == 3:
            print("BUY")
            base_balance, quote_balance = buy(base_balance, quote_balance,0.0002,price)
        print("")



    print("Start:" + str(first_balance) + "  Last" + str((base_balance*price)+quote_balance))
    print("Result:" + str((base_balance*price)+quote_balance-first_balance))
    print("Without Trading:" + str((first_base_balance*price + first_quote_balance)-first_balance))

    print()

    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    x=range(len(prediction))
    result = []
    win_num = 0
    total_num = 0
    for i in range(len(prediction)):
        if prediction[i] == 0 or prediction[i] == 3:
            total_num += 1
            if (prediction[i] == 0 and (target[i] == 0 or target[i] == 1)) or (prediction[i] == 3 and (target[i] == 2 or target[i] == 3)):
                result.append(1)
                win_num += 1
            else:
                result.append(-1)
        else:
            result.append(0)
    print("Accuracy:"+str(win_num/total_num))
    ax1.plot(x, df_list[symbol].iloc[-len(prediction):,1])
    ax1.set_title("Price")
    ax2.plot(x, target)
    ax2.set_title("Actual")
    ax3.plot(x, prediction)
    ax3.set_title("Prediction")
    ax4.plot(x, result)
    ax4.set_title("Result")
    plt.show()





