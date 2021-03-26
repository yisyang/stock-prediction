import click
import csv
from decimal import Decimal, getcontext
import itertools
import numpy as np


getcontext().prec = 4  # Intentionally kept fuzzy to allow learning.
OFFSET_SEQ_DAYS = 5
VALIDATION_RATIO = 0.20  # Fraction of batches to reserve for validation

def get_rows(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            yield row


def count_lines(filename):
    i = 0
    with open(filename, 'r') as file:
        for i, l in enumerate(file, 1):
            continue
    return i


def compute_delta(new, old):
    delta = (Decimal(new) - Decimal(old)) / (Decimal(old) + Decimal(0.0000001))
    return delta


def create_normalized_csv_with_delta_price(filename_in, filename_out):
    """
    Partial normalization to convert prices and volumes into fractions of previous day values.

    :param filename_in: Raw stock price file w/ begin_price,end_price,low_price,high_price,b_large_volume,...
    :param filename_out: Normalized stock prices with all elements as fraction change from previous day.
    :return:
    """
    data = get_rows(filename_in)

    # Check headers.
    headers = next(data)
    print('====================')
    print('Creating appended CSV')
    print('====================')
    print('Headers', headers)

    # stock_id, days_1, begin_price, end_price, low_price, high_price,
    # b_large_volume, b_big_volume, b_middle_volume, b_small_volume,
    # s_large_volume, s_big_volume, s_middle_volume, s_small_volume

    stock_id_index = 0
    day_index = 1
    end_price_index = 3
    h_dict = {v: k for k, v in enumerate(headers)}
    assert h_dict['stock_id'] == stock_id_index
    assert h_dict['days_1'] == day_index
    assert h_dict['end_price'] == end_price_index

    # Start writing to new file.
    with open(filename_out, 'w') as file_new:
        writer = csv.writer(file_new)
        writer.writerow(headers)
        
        # We always need 2 rows to compute delta price.
        prev_row = next(data)
        print('Sample row', prev_row)
        for row in data:
            # If stock id changed, abandon prev row and restart computations.
            if row[stock_id_index] != prev_row[stock_id_index]:
                print(f'Normalizing: {row[stock_id_index]}        ', end='\r')
                prev_row = row
                continue
            assert row[day_index] > prev_row[day_index]
            
            # Append delta end price element to prev row, and write prev row.
            out_row = [
                prev_row[0], prev_row[1],
                compute_delta(row[h_dict['begin_price']], prev_row[end_price_index]),
                compute_delta(row[end_price_index], prev_row[end_price_index]),
                compute_delta(row[h_dict['low_price']], prev_row[end_price_index]),
                compute_delta(row[h_dict['high_price']], prev_row[end_price_index]),
                compute_delta(row[h_dict['b_large_volume']], prev_row[h_dict['b_large_volume']]),
                compute_delta(row[h_dict['b_big_volume']], prev_row[h_dict['b_big_volume']]),
                compute_delta(row[h_dict['b_middle_volume']], prev_row[h_dict['b_middle_volume']]),
                compute_delta(row[h_dict['b_small_volume']], prev_row[h_dict['b_small_volume']]),
                compute_delta(row[h_dict['s_large_volume']], prev_row[h_dict['s_large_volume']]),
                compute_delta(row[h_dict['s_big_volume']], prev_row[h_dict['s_big_volume']]),
                compute_delta(row[h_dict['s_middle_volume']], prev_row[h_dict['s_middle_volume']]),
                compute_delta(row[h_dict['s_small_volume']], prev_row[h_dict['s_small_volume']])
            ]
            writer.writerow(out_row)
            prev_row = row
    print('====================')
    print(f'Appended CSV saved at ${filename_out}')


def create_train_val_batches(filename_in, data_dir, x_seq_length=30, x_features=12, y_seq_length=5):
    """
    Creates data files for training and validation uses.

    :param filename_in: Input normalized data file.
    :param data_dir: Directory to save output data files into.
    :param x_seq_length: Number of input days per sample.
    :param x_features: Number of input features per day.
    :param y_seq_length: Number of output days to predict. If set to 0, data will be for prediction only.
    :return:
    """
    filename_train_x = f'{data_dir}/train_x_{x_seq_length}_{x_features}.csv'
    filename_train_y = f'{data_dir}/train_y_{y_seq_length}.csv'
    filename_val_x = f'{data_dir}/val_x_{x_seq_length}_{x_features}.csv'
    filename_val_y = f'{data_dir}/val_y_{y_seq_length}.csv'
    filename_pred_x = f'{data_dir}/pred_x_{x_seq_length}_{x_features}.csv'

    n_lines = count_lines(filename_in)
    print('====================')
    if y_seq_length > 0:
        print('Creating training and validation datasets')
    else:
        print('Creating prediction datasets')
    print('====================')
    print('Number of lines', n_lines)
    
    data = get_rows(filename_in)

    # Check headers.
    headers = next(data)
    print('Headers', headers)
    h_dict = {v: k for k, v in enumerate(headers)}
    assert h_dict['stock_id'] == 0
    assert h_dict['days_1'] == 1
    assert h_dict['end_price'] == 3
    assert len(headers) == 14

    # Reset data
    data = get_rows(filename_in)
    next(data)

    if y_seq_length > 0:
        with open(filename_train_x, 'w') as file_train_x, \
                open(filename_train_y, 'w') as file_train_y, \
                open(filename_val_x, 'w') as file_val_x, \
                open(filename_val_y, 'w') as file_val_y:
            writer_train_x = csv.writer(file_train_x)
            writer_train_y = csv.writer(file_train_y)
            writer_val_x = csv.writer(file_val_x)
            writer_val_y = csv.writer(file_val_y)
            res = generate_train_val_rows(data, x_seq_length, x_features, y_seq_length)
            for x_rows, y_rows, validation in res:
                if validation:
                    writer_val_x.writerow(x_rows)
                    writer_val_y.writerow(y_rows)
                else:
                    writer_train_x.writerow(x_rows)
                    writer_train_y.writerow(y_rows)
    else:
        with open(filename_pred_x, 'w') as file_pred_x:
            writer_pred_x = csv.writer(file_pred_x)
            res = generate_train_val_rows(data, x_seq_length, x_features, y_seq_length)
            for x_rows, y_rows, validation in res:
                writer_pred_x.writerow(x_rows)

    print('====================')
    if y_seq_length > 0:
        print(f'Training data saved at {filename_train_x} {filename_train_y}')
        print(f'Validation data saved at {filename_val_x} {filename_val_y}')
    else:
        print(f'Prediction data saved at {filename_pred_x}')

    return


def generate_rows_from_buffer(x_buffer, y_buffer, x_seq_length, y_seq_length):
    n = len(x_buffer)
    total_seq_length = x_seq_length + y_seq_length

    # Not enough items in buffer.
    if n < total_seq_length:
        yield None, None, None

    # x: (n, x_seq_length, x_features)
    # Each output row contains x_seq_length days of data.
    # Offset by OFFSET_SEQ_DAYS days.
    # begin_price,end_price,low_price,high_price,b_large_volume,b_big_volume,b_middle_volume,b_small_volume,s_large_volume,s_big_volume,s_middle_volume,s_small_volume

    # y: (n, y_seq_length)
    # Each output row contains delta_end_price for the next y_seq_length days.

    if y_seq_length == 0:
        # For prediction, output only the latest x_seq_length data.
        yield [i for il in x_buffer[-total_seq_length:] for i in il], [], False
    else:
        # Split x/y buffer, reserve the final data points for validation.
        n_rows = (n - total_seq_length) // OFFSET_SEQ_DAYS + 1  # n=40, tsl=35, xsl=30, offset=5, n_rows=2
        n_val = VALIDATION_RATIO * n_rows                       # n_train = 2
        for i in reversed(range(n_rows)):                       # i = 1, 0
            a = n - total_seq_length - i*OFFSET_SEQ_DAYS        # a = 0, 5
            b = a + x_seq_length                                # b = 30, 35
            c = b + OFFSET_SEQ_DAYS                             # c = 35, 40
            yield [i for il in x_buffer[a:b] for i in il], y_buffer[b:c], i < n_val


def generate_train_val_rows(data, x_seq_length, x_features, y_seq_length):
    x_buffer = []
    y_buffer = []
    prev_row = ['START']  # Dummy row to simplify stock ID logic.

    for row in data:
        # If stock id changed, clear buffer and restart computations.
        if row[0] != prev_row[0]:
            res = generate_rows_from_buffer(x_buffer, y_buffer, x_seq_length, y_seq_length)
            for i, j, k in res:
                if i is not None:
                    yield i, j, k
            x_buffer = []
            y_buffer = []
            print(f'Processing stock: {row[0]}        ', end='\r')

        if x_features == 12:
            # All features, just include all normalized features
            x_buffer.append(row[2:])
        elif x_features == 6:
            # Simple volumes
            b_vol = Decimal(np.array(row[4:8], dtype='f4').sum().item()) + 0
            s_vol = Decimal(np.array(row[8:12], dtype='f4').sum().item()) + 0
            x_buffer.append(row[2:6] + [b_vol, s_vol])
        elif x_features == 4:
            x_buffer.append(row[2:6])
        y_buffer.append(row[3])
        prev_row = row

    # Output final buffer.
    res = generate_rows_from_buffer(x_buffer, y_buffer, x_seq_length, y_seq_length)
    for i, j, k in res:
        if i is not None:
            yield i, j, k


@click.command()
@click.option('-d', '--data-dir', default='data')
@click.option('-f', '--filename', default='data/BkStock20210321.csv')
@click.option('-xl', '--x-seq-length', type=click.INT, default=30,
              help='Number of days as input.')
@click.option('-xf', '--x-features', type=click.Choice(['4', '6', '12']), default='6',
              help='Num of features: 4=open/high/low/close, 6=4/b_vol/s_vol, 12=4/b_l_vol/b_b_vol/.../s_l_vol/...')
@click.option('-yl', '--y-seq-length', type=click.INT, default=5,
              help='Number of days for fitting/predicting, set to 0 for predicting future data.')
def main(data_dir='data', filename='', x_seq_length=30, x_features=6, y_seq_length=5):
    x_features = int(x_features)  # Small hack
    # First we want to append delta_end_price as targets.
    filename_appended = filename.replace('.csv', '_appended.csv')
    create_normalized_csv_with_delta_price(filename, filename_appended)

    # Next we will create training and validation batches.
    create_train_val_batches(filename_appended, data_dir, x_seq_length, x_features, y_seq_length)
    

if __name__ == "__main__":
    main()
