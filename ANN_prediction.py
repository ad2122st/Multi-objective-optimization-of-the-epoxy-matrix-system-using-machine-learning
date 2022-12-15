import sys
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

objective = 'Y1' # 'Y1' or 'Y2' or 'Y3'

dataset = pd.read_csv(f'csv/{objective}.csv', index_col=-1)

y_train = dataset.iloc[:, -1].copy()
x_train = dataset.iloc[:, :-1]
x_train = (x_train.T / x_train.T.sum()).T

std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis = 1)

autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_train = autoscaled_x_train.dropna()
autoscaled_y_train = autoscaled_y_train.dropna()

if objective == 'Y1':
    unit_layer1 = 64
    unit_layer2 = 64
    unit_layer3 = 64
    unit_layer4 = 64
    activation_layer1 = 'relu'
    activation_layer2 = 'relu'
    activation_layer3 = 'relu'
    optimizer = 'Adam'
    epoch = 1000
elif objective == 'Y2':
    unit_layer1 = 800
    unit_layer2 = 864
    unit_layer3 = 160
    unit_layer4 = 1024
    activation_layer1 = 'tanh'
    activation_layer2 = 'tanh'
    activation_layer3 = 'tanh'
    optimizer = 'RMSprop'
    epoch = 1000
elif objective == 'Y3':
    unit_layer1 = 64
    unit_layer2 = 64
    unit_layer3 = 64
    unit_layer4 = 64
    activation_layer1 = 'relu'
    activation_layer2 = 'relu'
    activation_layer3 = 'relu'
    optimizer = 'Adam'
    epoch = 1000

model = keras.Sequential([
    layers.Dense(unit_layer1, activation=activation_layer1, input_shape=[len(autoscaled_x_train.keys())]),
    layers.Dense(unit_layer2, activation=activation_layer2),
    layers.Dense(unit_layer3, activation=activation_layer3),
    layers.Dense(unit_layer4, activation='relu'),
    layers.Dense(1)
    ])
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(autoscaled_x_train, autoscaled_y_train, epochs=epoch, validation_split = 0.2, verbose=0)


estimated_y_train = model.predict(autoscaled_x_train) * y_train.std() + y_train.mean()
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index, columns=['estimated_y'])

plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))
plt.subplots_adjust(left=0.25)
plt.subplots_adjust(bottom=0.2) 
plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')
y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())
y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel(f'Actual {objective}')
plt.ylabel(f'Predicted {objective}')
plt.savefig(f"img/{objective}_train_for_prediction_ANN.pdf")


print(f'\n{objective} with ANN for train data')
print(f'r^2 :', metrics.r2_score(y_train, estimated_y_train))
print(f'RMSE :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
print(f'MAE :', metrics.mean_absolute_error(y_train, estimated_y_train))


y_train_for_save = pd.DataFrame(y_train)
y_train_for_save.columns = [f'Actual {objective}']
y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train)
y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
results_train.to_csv(f'csv/estimated_{objective}_train_for_prediction.csv')


dataset_prediction = pd.read_csv(f'csv/{objective}_prediction.csv', index_col=-1)

x_prediction = dataset_prediction.iloc[:, :-1]
x_prediction = (x_prediction.T / x_prediction.T.sum()).T


autoscaled_x_prediction = (x_prediction - x_train.mean()) / x_train.std()
predicted_y = model.predict(autoscaled_x_prediction) * y_train.std() + y_train.mean()
predicted_y = pd.DataFrame(predicted_y, index=x_prediction.index, columns=['predicted_y'])
results_prediction_y = pd.concat([predicted_y], axis=1)
results_prediction_y.to_csv(f'csv/predicted_{objective}_ANN.csv')