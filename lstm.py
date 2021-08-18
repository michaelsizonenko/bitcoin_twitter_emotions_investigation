import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False
win_length = 720
batch_size = 128
num_features = 2


def get_prediction_to_df(predictions, X_test, scaler, df_input):
    df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(X_test[:, 1:][win_length:])], axis=1)
    rev_trans = scaler.inverse_transform(df_pred)
    df_final = df_input[predictions.shape[0] * -1:]
    df_final['Pred'] = rev_trans.loc[:, 0]
    return df_final


def get_history(history):
    plt.plot(history.history['mean_absolute_error'])
    plt.show()
    print(history.history['mean_absolute_error'])


def df_featuring():
    df = pd.read_csv('dataset/btc_emotions')
    df = df[-10000:]
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    # df.set_index('date')[['close', 'mood']].plot(subplots=True)
    df_input = df[['close', 'mood']]

    return df_input


def model_create( train_generator, test_generator):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(win_length, num_features), return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))
    print(model.summary())
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min'
    )

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit_generator(train_generator, epochs=5,
                                  validation_data=test_generator,
                                  shuffle=False,
                                  callbacks=[early_stopping])
    return model, early_stopping, history


def main():
    df_input, = df_featuring()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_input)
    features = data_scaled
    target = data_scaled[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123,
                                                        shuffle=False)

    train_generator = TimeseriesGenerator(X_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
    test_generator = TimeseriesGenerator(X_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)
    model, early_stopping, history = model_create(win_length, num_features, train_generator, test_generator)

    model.evaluate_generator(test_generator, verbose=0)
    model.save('model_3')
    get_history(history)
    predictions = model.predict(test_generator)
    df_with_pred = get_prediction_to_df(predictions, X_test, scaler, df_input)


if __name__ == '__main__':
    main()
