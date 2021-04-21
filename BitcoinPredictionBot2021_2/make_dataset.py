from binance.client import Client
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime
import mplfinance as mpf
import tensorflow as tf
from PIL import Image
from datetime import datetime, timedelta
import os
import const
import technical_indicators
import math


def make_dataset(orig_data, orig_image):

    data, img_data, target = [], [], []

    #threshold = 0.0005

    df = orig_data.copy()

    df = df.fillna(0.0)

    df["Close"] = df["Close"].diff()
    df["High"] = df["High"].diff()
    df["Low"] = df["Low"].diff()
    df["WMA7"] = df["WMA7"].diff()
    df["EMA25"] = df["EMA25"].diff()
    df["MA99"] = df["MA99"].diff()
    df["BOLL_UP"] = df["BOLL_UP"].diff()
    df["BOLL_DOWN"] = df["BOLL_DOWN"].diff()
    df["SAR"] = df["SAR"].diff()

    #df["Close"] /= 10
    #df["High"] /= 10
    #df["Low"] /= 10
    #df["Volume"] /= 100000
    #df["QuoteVolume"] /= 100000000
    #df["TakerVolume"] /= 100000
    #df["TakerQuoteVolume"] /= 100000000
    #df["TradeCount"] /= 100000
    #df.loc[:,"WMA7":"SAR"] /= 10
    #df["SAR"] /= 100
    #df["MACD"] /= 100
    #df["MACD_SIGNAL"] /= 100
    #df["RSI12"] *= 10

    df.loc[:,"BTCUSDT":] = df.loc[:,"BTCUSDT":].diff()

    df=(df-df.min())/(df.max()-df.min())
    df=(df-df.mean())/df.std()

    #df["BTCUSDT"] /= 1000
    #df["ETHUSDT"] /= 100
    #df["XRPUSDT"] *= 10
    #df["BNBUSDT"] /= 10
    #df["BCHUSDT"] /= 100
    #df["DASHUSDT"] /= 100

    #df = df.clip(-1,1)


    df.to_csv("..\\Data\\aaa.csv")

    #for i in const.PAIR_SYMBOLS:
    #    if i in df.columns:
            #df[i] = df[i].diff()
            #df[i] = df[i].fillna(0.1)
            #print(0.1 ** int(math.log10(float(df[i].iloc[0])) + 2))
    #        df[i] *= 0.1 ** int(math.log10(float(df[i].iloc[0])) + 2)

    #for i in df.columns:
    #    for j in df[i]:
    #        if float(j) >= 1 or float(j) <= -1:
    #            print(i)
    #            print(j)


    for i in range(len(df)-(const.TIME_LENGTH+1)):  #get [i]~[i+const.TIME_LENGTH] as data and get [i+const.TIME_LENGTH] as target
        data.append(df.iloc[i:i + const.TIME_LENGTH,:])
        img_data.append(orig_image[i:i + const.TIME_LENGTH])
        dif = orig_data.iloc[i + const.TIME_LENGTH,0]-orig_data.iloc[i + const.TIME_LENGTH-1,0]  #difference between first open and last close
        threshold = orig_data.iloc[i+const.TIME_LENGTH,0]*0.005
        if dif > threshold:
            target.append(3)
        elif dif < -threshold:
            target.append(0)
        elif dif > 0:
            target.append(2)
        else:
            target.append(1)


    #down_num = 0
    #up_num = 0
    #for i in target:
    #    if i == 0:
    #        down_num += 1
    #    elif i == 2:
    #        up_num += 1
    #print(down_num/(len(df)-const.TIME_LENGTH))
    #print(up_num/(len(df)-const.TIME_LENGTH))

    #mpf.plot(df, type="candle")

    print(np.array(data).shape)
    print(np.array(img_data).shape)
    print(np.array(target).shape)

    re_data = np.array(data).reshape(len(data), const.TIME_LENGTH, len(df.columns))
    re_img_data = np.array(img_data).reshape(len(data), const.TIME_LENGTH, const.IMG_LENGTH, 3)
    re_target = np.array(target).reshape(len(data), 1)

    #re_data -= 0.0005

    np.nan_to_num(re_data, copy=False)
    np.nan_to_num(re_target, copy=False)
    np.nan_to_num(re_img_data, copy=False)

    return re_data, re_img_data, re_target






def make_current_data(binance,symbol, day_start,day_end):

    sub_pair_symbols = const.PAIR_SYMBOLS.copy()
    sub_pair_symbols.remove(symbol)

    df = pd.DataFrame(columns=["Time","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount","WMA7","EMA25","MA99","BOLL_UP","BOLL_DOWN","SAR","MACD","MACD_SIGNAL","RSI12"])
    #df = pd.DataFrame(columns=["Time","Open","Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount","MA7","MA25","MA99","EMA7","EMA25","EMA99","WMA7","WMA25","WMA99","BOLL_UP","BOLL_DOWN","VWAP","TEMA","SAR","MACD","MACD_SIGNAL","RSI6","RSI12","RSI24","K","D","J","OBV","WR","DI+","DI-","ADX","MTM","EMV"])
    df.loc[:, "Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")

    sub_pair_dict = {}

    for i in sub_pair_symbols:
        sub_pair_dict[i] = pd.DataFrame(columns=["Close","High","Low","Volume","QuoteVolume","TakerVolume","TakerQuoteVolume","TradeCount"])


    #to match the minimum size
    pad_multiplier = 128//const.TIME_LENGTH

    if day_start == 0 and day_end == 0:
        klines = binance.get_klines(symbol,const.TIME_LENGTH*pad_multiplier)
        df = df.append(klines)
        for i in sub_pair_symbols:
            sub_klines = binance.get_klines(i,const.TIME_LENGTH*pad_multiplier)
            sub_pair_dict[i] = sub_pair_dict[i].append(sub_klines)
    else:
        for i in range(day_start,day_end,-64):
            klines = binance.get_historical_klines(symbol, str(i+64) + " days ago UTC", str(i) + " days ago UTC")
            df = df.append(klines)
            for j in sub_pair_symbols:
                sub_klines = binance.get_historical_klines(j, str(i+64) + " days ago UTC", str(i) + " days ago UTC")
                sub_pair_dict[j] = sub_pair_dict[j].append(sub_klines)



    df = df.astype("float64")
    #mpf.plot(df, type="candle")

    #df.loc[:, "MA7"] =  technical_indicators.ma(df,7)
    #df.loc[:, "MA25"] =  technical_indicators.ma(df,25)
    df.loc[:, "MA99"] =  technical_indicators.ma(df,99)
    #df.loc[:, "EMA7"] =  technical_indicators.ema(df,7)
    df.loc[:, "EMA25"] =  technical_indicators.ema(df,25)
    #df.loc[:, "EMA99"] =  technical_indicators.ema(df,99)
    df.loc[:, "WMA7"] =  technical_indicators.wma(df,7)
    #df.loc[:, "WMA25"] =  technical_indicators.wma(df,25)
    #df.loc[:, "WMA99"] =  technical_indicators.wma(df,99)
    df.loc[:, "BOLL_UP"],df.loc[:, "BOLL_DOWN"] =  technical_indicators.boll(df,21)
    #df.loc[:, "VWAP"] =  technical_indicators.vwap(df,14)
    #df.loc[:, "TEMA"] =  technical_indicators.tema(df,9)
    df.loc[:, "SAR"] =  technical_indicators.sar(df)
    df.loc[:, "MACD"],df.loc[:, "MACD_SIGNAL"] =  technical_indicators.macd(df)
    #df.loc[:, "RSI6"] =  technical_indicators.rsi(df,6)
    df.loc[:, "RSI12"] =  technical_indicators.rsi(df,12)
    #df.loc[:, "RSI24"] =  technical_indicators.rsi(df,24)
    #df.loc[:, "K"],df.loc[:, "D"],df.loc[:, "J"] =  technical_indicators.kdj(df)
    #df.loc[:, "OBV"] =  technical_indicators.obv(df)
    #df.loc[:, "WR"] =  technical_indicators.wr(df,14)
    #df.loc[:, "DI+"],df.loc[:, "DI-"],df.loc[:, "ADX"] =  technical_indicators.dmi(df)
    #df.loc[:, "MTM"] =  technical_indicators.mtm(df)
    #df.loc[:, "EMV"] =  technical_indicators.emv(df)
    #df.loc[:, "PL"] =  technical_indicators.pl(df,12)
    #df = df.append(klines)


    for i in sub_pair_symbols:
        df.insert(len(df.columns),i,sub_pair_dict[i]["Close"])

    df = df.fillna(0.1)
    df = df.astype("float64")

    return df





def make_current_image_data(binance,symbol, day_start,day_end):

    print("generating img")

    #to match the minimum size
    pad_multiplier = 128//const.TIME_LENGTH

    if day_start == 0 and day_end == 0:
        day_start = const.TIME_LENGTH*pad_multiplier


    img = []
    for i in range(day_start, day_end, -1):
        now = datetime.utcnow() - timedelta(days=i)
        if now.minute >= 30:
            now = now.replace(minute=30, second=0, microsecond=0)
        else:
            now = now.replace(minute=0, second=0, microsecond=0)
        for j in range(0, 1440, 30):
            sleep(0.05)
            start = now + timedelta(minutes=j)
            end = now + timedelta(minutes=j+30)
            x = list(binance.get_trades_data("LTCUSDT",int(start.timestamp())*1000, int(end.timestamp())*1000))
            img.append(x)


    data = np.array(img)


    return data


