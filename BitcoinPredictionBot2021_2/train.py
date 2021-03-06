from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from time import sleep
import tensorflow as tf
import const
import os
import make_dataset
import kerastuner as kt


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



"""
def model_builder(hp): #for hyperparameter tuning
    hp_dense_units1 = hp.Int('dense_units1', min_value = 20, max_value = 800, step = 20)
    hp_dense_units2 = hp.Int('dense_units2', min_value = 20, max_value = 500, step = 20)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp_dense_units1, activation="tanh"), input_shape=(const.TIME_LENGTH, 31)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp_dense_units2, activation="swish"), input_shape=(const.TIME_LENGTH, 31)))
    model.add(tf.keras.layers.LSTM(80, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.5, return_sequences = True))
    model.add(tf.keras.layers.LSTM(80, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.5))
    model.add(tf.keras.layers.Dense(800, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(400, activation="tanh"))
    model.add(tf.keras.layers.Dense(200, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(75, activation="tanh"))
    model.add(tf.keras.layers.Dense(50, activation="swish"))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(25, activation="tanh"))
    model.add(tf.keras.layers.Dense(3, activation="swish"))
    optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
"""





def train(binance,model):


    df_list = {}
    df_list_test = {}

    symbol = "LTCUSDT"
    df_list[symbol] = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
    orig_img = np.load("..\\Data\\" + symbol + "_image_data.npy")

    

    #df_list_merged = pd.concat(df_list, axis=1)

    df_test = make_dataset.make_current_data(binance,"LTCUSDT",0,0)
    orig_img_test = make_dataset.make_current_image_data(binance,"LTCUSDT",0,0)

    data, data_img, target = make_dataset.make_dataset(df_list["LTCUSDT"], orig_img)
    data_test, data_img_test, target_test = make_dataset.make_dataset(df_test, orig_img_test)

    print(data_img.shape)
    print(data.shape)

    print(target_test)
    checkpoint_dir = os.path.dirname(const.CHECKPOINT_PATH)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        const.CHECKPOINT_PATH, 
        verbose=1, 
        save_weights_only=True,
        save_freq=1000)
    lr_change_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log", histogram_freq=1)

    epochs = 100
    batch_size = 8


    #tuner = kt.Hyperband(model_builder,
    #                 objective = "val_accuracy", 
    #                 max_epochs = 10,
    #                 factor = 3,
    #                 directory = "HyperparameterTunerData",
    #                 project_name = "tunerData")
    #tuner.search(data, target, epochs=3, validation_data=(data_test, target_test))
    #tuner.results_summary()

    print(target)

    stack = model.fit([data, data_img], target,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[checkpoint_callback,lr_change_callback, early_stop, tensorboard_callback],
              validation_data=([data_test, data_img_test], target_test)
              )
    model.summary()


    x = range(len(stack.history["accuracy"]))
    plt.plot(x, stack.history["accuracy"])
    plt.plot(x, stack.history["val_accuracy"])
    plt.legend(["accuracy", "val_accuracy"], loc="upper left")
    plt.title("accuracy")
    plt.show()
    

    #p_data, _ = make_dataset.make_dataset(df_list_test)
    #predicted = model.predict(p_data)
    #predicted = np.pad(predicted,[[0,const.TIME_LENGTH],[0,0]],"edge")
    #df = pd.DataFrame(predicted)
    #for i in range(100):
    #    print(df.iloc[i,:])
    #df = df.idxmax(axis=1)
    #df = df.columns.get_loc(df.idxmax(axis=1))
    #addplot = mpf.make_addplot(df)
    #mpf.plot(df_list["LTCUSDT"], type='candle', addplot = addplot)
    
    
    test_acc = model.evaluate([data_test, data_img_test], target_test, verbose=2)
    print("Test accuracy:", test_acc)






    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    prediction = pd.DataFrame(model.predict([data_test, data_img_test]))
    prediction = prediction.idxmax(axis=1)
    x=range(len(prediction))
    result = []
    win_num = 0
    for i in range(len(prediction)):
        if target_test[i] == prediction[i] or ((target_test[i] == 1 or target_test[i] == 2) and (prediction[i] == 1 or prediction[i] == 2)):
            result.append(1)
            win_num += 1
        else:
            result.append(0)
    print("Accuracy:"+str(win_num/len(prediction)))
    ax1.plot(x, df_test.iloc[-len(prediction):,1])
    ax1.set_title("Price")
    ax2.plot(x, target_test)
    ax2.set_title("Actual")
    ax3.plot(x, prediction)
    ax3.set_title("Prediction")
    ax4.plot(x, result)
    ax4.set_title("Result")
    plt.show()

    
