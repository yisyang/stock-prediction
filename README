# Stock prediction

This is a toy project by a human learning to do some practical machine-learning.
The model used is just a simple LSTM and a Dense layer.


## Process

The initial implementation splits the data by stock_id into training and validation batches.
The losses decreased nicely, but the resulting model is not good at predicting real world data.

The current implementation splits the data by time, reserving the first N timesteps for training, and the rest for validation.
The result is that the training losses decrease nicely, and the validation losses stayed the same.

In other words... a nice exercise but it DOES NOT WORK at predicting real world data.


## Preparing data

Input data should be in the format of:
```
stock_id,days_1,begin_price,end_price,low_price,high_price,b_large_volume,b_big_volume,b_middle_volume,b_small_volume,s_large_volume,s_big_volume,s_middle_volume,s_small_volume
000800.SZ,2010-09-07 00:00:00,15.18,14.87,14.82,15.33,21570897,1445674,844167,0,21797815,996369,1066349,205
000800.SZ,2010-09-08 00:00:00,15.00,14.88,14.67,15.04,15448956,925000,1049072,0,16144750,617026,661073,179
000800.SZ,2010-09-09 00:00:00,14.65,14.42,14.27,14.80,20986988,1377623,1413536,0,21996045,840644,941249,209
000800.SZ,2010-09-10 00:00:00,14.55,14.08,14.00,14.62,15330849,1251200,871800,0,15859016,687931,906769,133
```

`prepare-data.py` is used to convert data into usable batches.
`run-keras.py` and `run-torch.py` are used to fit the data.
For all commands use --help to see parameters. Default parameters can also be changed by changing constants at the top of the files.


## Training and validation

**Prepare data into train/val batches:**
```
python prepare-data.py -f data/input.csv
```

**Training (w/ validation) from scratch**
```
python run-keras.py
```

**Training from previous checkpoint**
```
python run-keras.py -l 300
```
where 300 is the checkpoint saved after number of episodes of training.

**Validation only**
```
python run-keras.py -m v -l 300
```


## Prediction

**Prepare data into pred batches:**
```
python prepare-data.py -f data/input.csv -yl 0
```

**Prediction**
```
python run-keras.py -l 300 -m p
```
where 300 is the checkpoint saved after number of episodes of training.
