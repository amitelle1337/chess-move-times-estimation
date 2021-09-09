import bz2
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, LSTM, concatenate, Masking
from keras_preprocessing import sequence


def normalize_data(data):
    for i in range(len(data)):
        # static
        data[i][0][0][0] /= 3000  # Elo
        data[i][0][0][1] /= 3000  # Elo
        data[i][0][0][2] /= 600  # starting time
        data[i][0][0][3] /= 10  # time increment
        data[i][0][0][4] /= 100000  # score
        data[i][0][0][5] /= 200  # legal moves
        data[i][0][0][6] /= 50  # pv

        # dynamic
        for j in range(len(data[i][0][1])):
            data[i][0][1][j][0] /= 100000  # score
            data[i][0][1][j][1] /= 200  # legal moves
            data[i][0][1][j][2] /= 50  # pv
            data[i][0][1][j][3] /= 200  # time

        data[i][1] /= 200  # time


def main():
    with bz2.open('../lichess/lichess_25k.pickel.bz2') as f:
        data = pickle.load(f)

    data = [[*row] for row in data]
    normalize_data(data)
    static_x = np.asarray([x[0] for x, _ in data])
    dynamic_x = np.asarray(sequence.pad_sequences([x[1] for x, _ in data], padding='post', value=-1))
    y = np.asarray([y for _, y in data])

    static_part = Sequential()
    static_part.add(Dense(5, input_shape=(7,), activation='relu'))
    static_part.add(Dense(3, activation='relu'))

    dynamic_part = Sequential()
    dynamic_part.add(Masking(mask_value=[-1, -1, -1, -1], input_shape=(None, 4)))
    dynamic_part.add(LSTM(4, activation='relu', return_sequences=True))
    dynamic_part.add(LSTM(4, activation='relu'))
    dynamic_part.add(Dense(3, activation='relu'))

    merged_part = concatenate([static_part.output, dynamic_part.output], axis=-1)
    merged_part = Dense(4, activation='relu')(merged_part)
    merged_part = Dense(2, activation='relu')(merged_part)
    merged_part = Dense(1, activation='relu')(merged_part)

    model = Model(inputs=[static_part.input, dynamic_part.input], outputs=merged_part)

    model.compile(loss='mse', optimizer='adam',
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(),
                  ])

    history = model.fit(x=[static_x, dynamic_x], y=y, validation_split=0.2, shuffle=True, epochs=5)

    model.save('move_times_model')

    plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
