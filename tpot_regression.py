import sys
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from tpot import TPOTRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

objective = 'Y3' # 'Y1' or 'Y2' or 'Y3'
dataset = pd.read_csv(f'csv/{objective}.csv', index_col=-1)
number_of_test_samples = 0.2

if objective == 'Y1':
    random_state = 21
if objective == 'Y2':
    random_state = 13
if objective == 'Y3':
    random_state = 21
    
y = dataset.iloc[:, -1].copy()
x = dataset.iloc[:, :-1]
x = (x.T / x.T.sum()).T

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = number_of_test_samples, shuffle=True, random_state=random_state)

std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis = 1)
x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis = 1)

autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()
autoscaled_x_train = autoscaled_x_train.dropna()
autoscaled_y_train = autoscaled_y_train.dropna()

model = TPOTRegressor(scoring='neg_mean_absolute_error',
                     generations=5,
                     population_size=25,
                     random_state=42,
                     verbosity=2,
                     n_jobs=-1
                    )
model.fit(autoscaled_x_train, autoscaled_y_train)

for x, y, word in [(autoscaled_x_train, y_train, 'train'), (autoscaled_x_test, y_test, 'test')]:
    estimated_y = model.predict(x) * y_train.std() + y_train.mean()
    estimated_y = pd.DataFrame(estimated_y, index=x.index, columns=[f'Predicted {objective}'])
    
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=figure.figaspect(1))
    plt.subplots_adjust(left=0.25)
    plt.subplots_adjust(bottom=0.2) 
    plt.scatter(y, estimated_y.iloc[:, 0], c='blue')
    y_max = max(y.max(), estimated_y.iloc[:, 0].max())
    y_min = min(y.min(), estimated_y.iloc[:, 0].min())
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel(f'Actual {objective}')
    plt.ylabel(f'Predicted {objective}')
    plt.savefig(f"img/{objective}_{word}_tpot.pdf")

    print(f'\n{objective} with tpot for {word} data')
    print(f'r^2 :', metrics.r2_score(y, estimated_y))
    print(f'RMSE :', metrics.mean_squared_error(y, estimated_y) ** 0.5)
    print(f'MAE :', metrics.mean_absolute_error(y, estimated_y))

    y_for_save = pd.DataFrame(y)
    y_for_save.columns = [f'Actual {objective}']
    y_error_train = y_for_save.iloc[:, 0] - estimated_y.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)
    y_error_train.columns = [f'error_of_{objective}(actual_{objective}-predicted_{objective})']
    results_train = pd.concat([estimated_y, y_for_save, y_error_train], axis=1)
    results_train.to_csv(f'csv/Predicted_{objective}_{word}_tpot.csv')
