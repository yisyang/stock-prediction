import click
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as L
from data_loader import DataLoader


DEVICE = None
TRAIN_X = []
TRAIN_Y = []
VAL_X = []
VAL_Y = []
BATCH_SIZE = 25
X_SEQ_LEN = 30
X_FEATURES = 6
Y_SEQ_LEN = 5


def create_model(load_checkpoint, batch_size=None):
    if load_checkpoint > 0:
        model = keras.models.load_model(f'checkpoints/k_{X_SEQ_LEN}_{X_FEATURES}_{Y_SEQ_LEN}_{load_checkpoint:04}')
        print(f'Continuing from episode {load_checkpoint}')
    else:
        model = keras.Sequential()
        model.add(L.Input(shape=[X_SEQ_LEN, X_FEATURES], batch_size=batch_size))
        model.add(L.LSTM(units=96, activation='tanh', return_sequences=False))
        model.add(L.Dropout(rate=0.3))
        model.add(L.Dense(5, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model


def train(load_checkpoint, total_episodes):
    data_loader = DataLoader(X_SEQ_LEN, X_FEATURES, Y_SEQ_LEN)
    model = create_model(load_checkpoint)
    episode = load_checkpoint
    s_x, s_y = data_loader.get_all_data()
    v_x, v_y = data_loader.get_all_data('validate')
    while episode < total_episodes:
        episode += 1

        history = model.fit(s_x, s_y, batch_size=BATCH_SIZE, validation_data=(v_x, v_y), verbose=1)
        print(f'Episode {episode}')
        print(history.history)

        # Save every N episodes
        if episode % 10 == 0:
            save_path = f'checkpoints/k_{X_SEQ_LEN}_{X_FEATURES}_{Y_SEQ_LEN}_{episode:04}'
            model.save(save_path)
            print(f'Model saved at {save_path}')
    return


def validate(load_checkpoint):
    np.set_printoptions(precision=4)
    data_loader = DataLoader(X_SEQ_LEN, X_FEATURES, Y_SEQ_LEN, 'validate')
    model = create_model(load_checkpoint)
    v_x, v_y = data_loader.get_batch(5, 'validate')
    for i in range(5):
        v_x_i = (v_x[i]).reshape(1, X_SEQ_LEN, X_FEATURES)
        pred = model.predict(v_x_i)
        print('pred  ', pred[0])
        print('actual', v_y[i])
    loss = model.evaluate(x=v_x, y=v_y, batch_size=5)
    print(loss)
    return


def predict(load_checkpoint):
    np.set_printoptions(precision=4)
    data_loader = DataLoader(X_SEQ_LEN, X_FEATURES, Y_SEQ_LEN, 'predict')
    v_x, v_y = data_loader.get_all_data('predict')
    size = len(v_x)
    model = create_model(load_checkpoint, size)
    pred = model.predict(v_x)
    print('pred')
    print(pred)
    return


@click.command()
@click.option('-m', '--mode', default='train', type=click.Choice(['train', 'validate', 'predict', 't', 'v', 'p']))
@click.option('-l', '--load-checkpoint', type=click.types.INT, default=0, help='Integer of episode checkpoint to load.')
@click.option('-te', '--total-episodes', default=5000, help='Total number of training episodes.')
def main(mode='train', load_checkpoint=0, total_episodes=5000):
    if mode in ['train', 't']:
        train(load_checkpoint, total_episodes)
    elif mode in ['validate', 'v']:
        validate(load_checkpoint)
    else:
        predict(load_checkpoint)


if __name__ == "__main__":
    main()
