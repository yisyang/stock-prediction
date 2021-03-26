import click
from data_loader import DataLoader
import numpy as np
import torch
from torch_simple_lstm import SimpleLstm


DEVICE = None
TRAIN_X = []
TRAIN_Y = []
VAL_X = []
VAL_Y = []
BATCH_SIZE = 25
X_SEQ_LEN = 30
X_FEATURES = 6
Y_SEQ_LEN = 5

torch.autograd.set_detect_anomaly(True)


def init_torch_device():
    global DEVICE
    if DEVICE is not None:
        return
    if torch.cuda.is_available():
        print('Cuda devices count', torch.cuda.device_count())
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')


def create_model(load_checkpoint):
    model = SimpleLstm(x_seq_length=X_SEQ_LEN, x_features=X_FEATURES, y_seq_length=Y_SEQ_LEN,
                       n_hidden=96, device=DEVICE)
    print('Model', model)
    print('Parameters')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    if load_checkpoint > 0:
        model.load_state_dict(torch.load(f'checkpoints/t_{X_SEQ_LEN}_{X_FEATURES}_{Y_SEQ_LEN}_{load_checkpoint:06}.zip', map_location=DEVICE))
        print(f'Continuing from episode {load_checkpoint}')
    return model


def train(load_checkpoint, total_episodes):
    data_loader = DataLoader(X_SEQ_LEN, X_FEATURES, Y_SEQ_LEN, 'train', 'torch', DEVICE)
    model = create_model(load_checkpoint)
    episode = load_checkpoint
    while episode < total_episodes:
        episode += 1

        s_x, s_y = data_loader.get_batch(BATCH_SIZE, 'train')
        # print('x shape', s_x.shape)
        # print('y shape', s_y.shape)
        # print('dtype', s_y.dtype)
        # print(s_x[:2])
        # print(s_y[:2])

        output, _ = model(s_x)
        loss = model.get_loss(output, s_y)

        print(f'step: {episode}  loss: {loss}')

        # Save every 100 episodes
        if episode % 100 == 0:
            save_path = f'checkpoints/t_{X_SEQ_LEN}_{X_FEATURES}_{Y_SEQ_LEN}_{episode:06}.zip'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at {save_path}')
            validate(episode)
    return


def validate(load_checkpoint, verbose=False):
    data_loader = DataLoader(X_SEQ_LEN, X_FEATURES, Y_SEQ_LEN, 'validate', 'torch', DEVICE)
    model = create_model(load_checkpoint)
    s_x, s_y = data_loader.get_batch(5, 'validate')
    output, _ = model(s_x)
    loss = model.get_loss(output, s_y)
    if verbose:
        print(output)
        print(s_y)
    print(f'val loss: {loss}')

    return


def predict(load_checkpoint):
    np.set_printoptions(precision=4)
    data_loader = DataLoader(X_SEQ_LEN, X_FEATURES, Y_SEQ_LEN, 'predict', 'torch', DEVICE)
    v_x, v_y = data_loader.get_all_data('predict')
    model = create_model(load_checkpoint)
    pred, _ = model(v_x)
    print('pred  ', pred)
    return


@click.command()
@click.option('-m', '--mode', default='train', type=click.Choice(['train', 'validate', 'predict', 't', 'v', 'p']))
@click.option('-l', '--load-checkpoint', type=click.types.INT, default=0, help='Integer of episode checkpoint to load.')
@click.option('-te', '--total-episodes', default=100000, help='Total number of training episodes.')
def main(mode='train', load_checkpoint=0, total_episodes=5000):
    init_torch_device()

    if mode in ['train', 't']:
        train(load_checkpoint, total_episodes)
    elif mode in ['validate', 'v']:
        validate(load_checkpoint, True)
    else:
        predict(load_checkpoint)


if __name__ == "__main__":
    main()
