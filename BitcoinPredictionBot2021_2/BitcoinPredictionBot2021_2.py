from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
import mplfinance as mpf
import tensorflow as tf
import os
import const
import collect_info
import train
import simulation
import make_dataset
import technical_indicators


class BinanceAPI:

    def __init__(self, api_key, api_secret):
        API_KEY = api_key
        API_SECRET = api_secret

        self.client = Client(API_KEY, API_SECRET)


    def get_ticker(self, pair):
        try:
            value = self.client.get_ticker(symbol=pair)
            return value
        except Exception as e:
            print("Exception : " + str(e))


    def get_current_price(self, pair):
        try:
            ticker = self.client.get_symbol_ticker(symbol=pair)
            value = ticker["price"]
            return float(value)
        except Exception as e:
            print("Exception : " + str(e))


    def get_klines(self, pair, number):
        try:
            klines = pd.DataFrame(self.client.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_30MINUTE, limit=number),columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","TradeCount","TakerVolume","TakerQuoteVolume","Ignore"])
            value = klines[["OpenTime","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"]].copy()
            value.loc[:, "OpenTime"] = pd.to_datetime(value["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000))))
            value = value.set_index("OpenTime")
            return value
        except Exception as e:
            print("Exception : " + str(e))


    def get_historical_klines(self, pair, start, end):
        try:
            klines = pd.DataFrame(self.client.get_historical_klines(start_str = start,end_str = end,symbol=pair, interval=Client.KLINE_INTERVAL_30MINUTE, limit=500),columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume","TradeCount","TakerVolume","TakerQuoteVolume","Ignore"])
            value = klines[["OpenTime","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"]].copy()
            value.loc[:, "OpenTime"] = pd.to_datetime(value["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000))))
            value = value.set_index("OpenTime")
            return value
        except Exception as e:
            print("Exception : " + str(e))


    def get_balance(self, symbol):
        try:
            value = self.client.get_asset_balance(asset=symbol)
            return value
        except Exception as e:
            print('Exception : {}'.format(e))

    def get_free_balance(self, symbol):
        try:
            value = self.client.get_asset_balance(asset=symbol)
            return float(value["free"])
        except Exception as e:
            print('Exception : {}'.format(e))


    def get_futures_balance(self, symbol):
        try:
            value = self.client.futures_account_balance()
            balance = [balance["balance"] for balance in value if balance["asset"] == symbol]
            return float(str(*balance))
        except Exception as e:
            print('Exception : {}'.format(e))
            


    def create_limit_order(self, symbol, price, quantity, side_str):
        try:
            if side_str == "BUY":
                side = self.client.SIDE_BUY
            elif side_str == "SELL":
                side = self.client.SIDE_SELL
            order = self.client.order_limit(
            symbol=symbol,
            side=side,
            timeInForce=self.client.TIME_IN_FORCE_IOC,
            price=price,
            quantity=quantity)
            print(order)
            print("buy order created.\nSymbol:{0:5}\nPrice:{1:5}\nQuantity:{2:5}",symbol,price,quantity)
        except Exception as e:
            print("Exception : " + str(e))


    def create_market_order(self, symbol, quantity, side_str):
        try:
            if side_str == "BUY":
                side = self.client.SIDE_BUY
            elif side_str == "SELL":
                side = self.client.SIDE_SELL
            order = self.client.order_market(
            symbol=symbol,
            side=side,
            quantity=quantity)
            print(order)
            print("buy order created.\nSymbol:{0:5}\nPrice:{1:5}\nQuantity:{2:5}",symbol,price,quantity)
        except Exception as e:
            print("Exception : " + str(e))


    def create_test_order(self, symbol, quantity, side_str):
        try:
            if side_str == "BUY":
                side = self.client.SIDE_BUY
            elif side_str == "SELL":
                side = self.client.SIDE_SELL
            order = self.client.create_test_order(
            symbol=symbol,
            side=side,
            type=self.client.ORDER_TYPE_MARKET,
            quantity=quantity)
            print(order)
            print("buy order created.\nSymbol:{0:5}\nQuantity:{1:5}".format(symbol,quantity))
        except Exception as e:
            print("Exception : " + str(e))



    def create_futures_order(self, symbol, quantity, side_str):
        try:
            if side_str == "BUY":
                side = self.client.SIDE_BUY
            elif side_str == "SELL":
                side = self.client.SIDE_SELL
            order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=self.client.ORDER_TYPE_MARKET,
            quantity=quantity)
            #print(order)
            print("order created.\nSymbol:{0:5}\nQuantity:{1:5}".format(symbol,quantity))
        except Exception as e:
            print("Exception : " + str(e))


    def create_futures_stoploss_order(self, symbol, quantity, side_str, price):
        if side_str == "BUY":
            side = self.client.SIDE_BUY
        elif side_str == "SELL":
            side = self.client.SIDE_SELL
        order = self.client.futures_create_order(
        symbol=symbol,
        stopPrice = price,
        side=side,
        type="STOP_MARKET",
        quantity=quantity)
        #print(order)
        #print("order created.\nSymbol:{0:5}\nQuantity:{1:5}".format(symbol,quantity))


    def cancel_futures_orders(self,symbol):
        for i in range(0,50):
            try:
                self.client.futures_cancel_all_open_orders(symbol = symbol)
            except Exception as e:
                print("Exception : " + str(e))
                continue
            break



    def get_open_futures_orders(self,symbol):
        try:
            result = self.client.futures_get_open_orders(symbol = symbol)
            return result
        except Exception as e:
            print("Exception : " + str(e))



    def get_base_asset(self, symbol):
        try:
            return self.client.get_symbol_info(symbol)["baseAsset"];
        except Exception as e:
            print("Exception : " + str(e))


    def get_quote_asset(self, symbol):
        try:
            return self.client.get_symbol_info(symbol)["quoteAsset"];
        except Exception as e:
            print("Exception : " + str(e))

    def get_all_tickers(self):
        try:
            return self.client.get_all_tickers();
        except Exception as e:
            print("Exception : " + str(e))

    def get_all_orders(self):
        try:
            return self.client.get_all_orders();
        except Exception as e:
            print("Exception : " + str(e))


    def get_trades_data(self, symbol, start, end):
        try:
            data = []
            value = self.client.get_aggregate_trades(symbol = symbol, startTime = start, endTime = end)
            value_max = [-float('inf'), -float('inf')]
            value_min = [float('inf'), float('inf')]
            #price = self.client.get_historical_klines(start_str = start, end_str = start + 60000, symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0][1]
            for i in value:
                if len(data) == const.IMG_LENGTH:
                    break
                data.append([float(i["p"]), float(i["q"]), 0.3 if i["m"] else 0.6])
            if len(data) < const.IMG_LENGTH:
                data += [[0.0, 0.0, 0.0]]*(const.IMG_LENGTH-len(data))
            data = data[:const.IMG_LENGTH]
            data = pd.DataFrame(data)
            data=(data-data.min())/(data.max()-data.min())
            data=(data-data.mean())/data.std()
            data=(data-data.min())/(data.max()-data.min())
            return data.values.tolist()
        except Exception as e:
            print("Exception : " + str(e))
            data = []
            data += [[0.0, 0.0, 0.0]]*(const.IMG_LENGTH)
            return data



def make_model(column_num):

    input_features = tf.keras.layers.Input(shape=(const.TIME_LENGTH, column_num))
    features = tf.keras.layers.LSTM(32, activation="tanh", recurrent_activation="sigmoid", return_sequences = True)(input_features)
    features = tf.keras.layers.LSTM(16, activation="tanh", recurrent_activation="sigmoid")(features)

    features = tf.keras.layers.Dense(128, activation="swish")(features)
    features = tf.keras.layers.Dense(64, activation="swish")(features)
    features = tf.keras.layers.Dropout(0.4)(features)
    features = tf.keras.layers.Dense(32, activation="swish")(features)
    features = tf.keras.layers.Dense(16, activation="swish")(features)
    features = tf.keras.layers.Dropout(0.4)(features)


    input_image = tf.keras.layers.Input(shape=(const.TIME_LENGTH, const.IMG_LENGTH, 3))
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(16, 3, strides=1, padding="same", activation = "tanh"))(input_image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32, 3, strides=1, padding="same", activation = "tanh"))(image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, 3, strides=1, padding="same", activation = "tanh"))(image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.4))(image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="swish"))(image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="swish"))(image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.4))(image)
    image = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(image)

    image = tf.keras.layers.LSTM(16, activation="tanh", recurrent_activation="sigmoid", return_sequences = True)(image)
    image = tf.keras.layers.LSTM(8, activation="tanh", recurrent_activation="sigmoid")(image)

    image = tf.keras.layers.Dropout(0.4)(image)
    image = tf.keras.layers.Dense(32, activation="swish")(image)
    image = tf.keras.layers.Dense(16, activation="swish")(image)


    x = tf.keras.layers.Concatenate(axis = 1)([features,image])
    x = tf.keras.layers.Dense(32, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(16, activation="swish")(x)

    output = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs = [input_features, input_image], outputs = output)

    optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics = ['accuracy'])

    return model







def main():

    #with open("ApiKeyFuturesTestnet.txt") as f:
    with open("ApiKey.txt") as f:
        api_key = f.readline().rstrip('\n')
        api_secret = f.readline().rstrip('\n')


    binance = BinanceAPI(api_key, api_secret);

    

    df = make_dataset.make_current_data(binance,"LTCUSDT",0,0) #to get column size

    #model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200, activation="tanh"), input_shape=(const.TIME_LENGTH, len(df.columns))))
    #model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation="swish"), input_shape=(const.TIME_LENGTH, len(df.columns))))
    #model.add(tf.keras.layers.LSTM(32, activation="tanh", recurrent_activation="sigmoid", return_sequences = True))
    #model.add(tf.keras.layers.LSTM(16, activation="tanh", recurrent_activation="sigmoid"))
    #model.add(tf.keras.layers.Dense(500, activation="swish"))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(250, activation="swish"))
    #model.add(tf.keras.layers.Dense(125, activation="swish"))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(75, activation="swish"))
    #model.add(tf.keras.layers.Dense(50, activation="swish"))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(25, activation="swish"))
    #model.add(tf.keras.layers.Dense(4, activation="softmax"))
    #optimizer = tf.keras.optimizers.Adam(lr=0.001)
    #model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
    
    model = make_model(len(df.columns))



    print("0:CollectData 1:Train 2:Simulation")
    mode = input(">")
    if mode == "0":
        collect_info.collect_info(binance)
    elif mode == "1":
        train.train(binance,model)
    elif mode == "2":
        simulation.simulation(binance,model)




if __name__ == '__main__':
    main()


