import sys
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split# , cross_val_score
from tensorflow import keras
from tensorflow.keras import layers
import optuna

objective = 'Y1' # 'Y1' or 'Y2' or 'Y3'
num_trial = 1000
for_prediction = False # True or False

dataset = pd.read_csv(f'csv/{objective}.csv', index_col=-1)

number_of_test_samples = 0.2
fold_number = 10

if objective == 'Y1':
    random_state = 21
if objective == 'Y2':
    random_state = 13
if objective == 'Y3':
    random_state = 21
    
y = dataset.iloc[:, -1].copy()

if for_prediction:
    y_train = dataset.iloc[:, -1].copy()
    x_train = dataset.iloc[:, :-1]
    x_train = (x_train.T / x_train.T.sum()).T
else:
    x = dataset.iloc[:, :-1]
    x = (x.T / x.T.sum()).T
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = number_of_test_samples, shuffle=True, random_state=random_state)

std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis = 1)

autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()

autoscaled_x_train = autoscaled_x_train.dropna()
autoscaled_y_train = autoscaled_y_train.dropna()

def objective(trial):
    unit_layer1 = trial.suggest_int("unit_layer1", 64, 1024, 32)
    unit_layer2 = trial.suggest_int("unit_layer2", 64, 1024, 32)
    unit_layer3 = trial.suggest_int("unit_layer3", 64, 1024, 32)
    unit_layer4 = trial.suggest_int("unit_layer4", 64, 1024, 32)
    activation_layer1 = trial.suggest_categorical('activation_layer1', ["relu", "sigmoid", 'tanh'])
    activation_layer2 = trial.suggest_categorical('activation_layer2', ["relu", "sigmoid", 'tanh'])
    activation_layer3 = trial.suggest_categorical('activation_layer3', ["relu", "sigmoid", 'tanh'])
    optimizer = trial.suggest_categorical("optimizer", ["RMSprop", "Adam"])
    
    model = keras.Sequential([
        layers.Dense(unit_layer1, activation=activation_layer1, input_shape=[len(autoscaled_x_train.keys())]),
        layers.Dense(unit_layer2, activation=activation_layer2),
        layers.Dense(unit_layer3, activation=activation_layer3),
        layers.Dense(unit_layer4, activation='relu'),
        layers.Dense(1)
        ])
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    epoch = 1000
    model.fit(autoscaled_x_train, autoscaled_y_train, epochs=epoch, validation_split = 0.2, verbose=0)
    mae = model.evaluate(autoscaled_x_train, autoscaled_y_train, verbose=0)
    keras.backend.clear_session()

    return mae

study = optuna.create_study()
study.optimize(objective, n_trials = num_trial)

print('\nBest trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
