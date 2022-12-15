import sys
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

objective = 'Y1' # 'Y1' or 'Y2' or 'Y3'
method_name = 'pls' # 'pls' or 'svr' or 'kr' or 'rf'

if method_name != 'pls' and method_name != 'svr' and method_name != 'kr' and method_name != 'rf':
    sys.exit(f'There is no regression method {method_name}. Please choose from "pls", "svr", "kr" or "rf".')

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

if method_name == 'pls':
    if objective == 'Y1':
        optimal_component_number = 16
    if objective == 'Y2':
        optimal_component_number = 4
    if objective == 'Y3':
        optimal_component_number = 12

    model = PLSRegression(n_components = optimal_component_number)

if method_name == 'svr':
    if objective == 'Y1':
        optimal_svr_gamma = 0.0022699420693605816
        optimal_svr_c = 25.735742734476617
        optimal_svr_epsilon = 0.003049050229232518
    if objective == 'Y2':
        optimal_svr_gamma = 0.01929591346012153
        optimal_svr_c = 70.5947773620326
        optimal_svr_epsilon = 0.2881869684674881
    if objective == 'Y3':
        optimal_svr_gamma = 0.05666645456567773
        optimal_svr_c = 11.449838183377452
        optimal_svr_epsilon = 0.0005398260722818336
    
    model = svm.SVR(kernel='rbf', C = optimal_svr_c, epsilon = optimal_svr_epsilon, gamma = optimal_svr_gamma)

if method_name == 'kr':
    if objective == 'Y1':
        alpha = 0.011552066380281046
        gamma = 0.010004994507661022
    if objective == 'Y2':
        alpha = 0.04418163719405123
        gamma = 0.01556495309293993
    if objective == 'Y3':
        alpha = 0.12241231159632308
        gamma = 0.010026083066201497

    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    
if method_name == 'rf':
    if objective == 'Y1':
        bootstrap = 'False'
        criterion = 'absolute_error'
        max_depth = 983
        max_features = 'log2',
        max_leaf_nodes = 176
        n_estimators = 625
        min_samples_split = 3
        min_samples_leaf = 1
        n_jobs = 2
    if objective == 'Y2':
        bootstrap = 'False'
        criterion = 'absolute_error'
        max_depth = 756
        max_features = 'auto'
        max_leaf_nodes = 306
        n_estimators = 461
        min_samples_split = 2
        min_samples_leaf = 1
        n_jobs = 2
    if objective == 'Y3':
        bootstrap = 'True'
        criterion = 'squared_error'
        max_depth = 100
        max_features = 'sqrt'
        max_leaf_nodes = 346
        n_estimators = 453
        min_samples_split = 3
        min_samples_leaf = 1
        n_jobs = 2

    model = RandomForestRegressor(
        bootstrap=bootstrap, criterion=criterion, 
        max_depth=max_depth, max_features=''.join(max_features), 
        max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
        n_jobs=n_jobs
        )

model.fit(autoscaled_x_train, autoscaled_y_train)

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
plt.savefig(f"img/{objective}_train_for_prediction_{method_name}.pdf")

print(f'\n{objective} with {method_name} for train data')
print(f'r^2 :', metrics.r2_score(y_train, estimated_y_train))
print(f'RMSE :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
print(f'MAE :', metrics.mean_absolute_error(y_train, estimated_y_train))

y_train_for_save = pd.DataFrame(y_train)
y_train_for_save.columns = [f'Actual {objective}']
y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train)
y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
results_train.to_csv(f'csv/Predicted_{objective}_train_for_prediction.csv')


dataset_prediction = pd.read_csv(f'csv/{objective}_prediction.csv', index_col=-1)

x_prediction = dataset_prediction.iloc[:, :-1]
x_prediction = (x_prediction.T / x_prediction.T.sum()).T


autoscaled_x_prediction = (x_prediction - x_train.mean()) / x_train.std()
autoscaled_x_prediction = autoscaled_x_prediction.fillna(0)
predicted_y = model.predict(autoscaled_x_prediction) * y_train.std() + y_train.mean()
predicted_y = pd.DataFrame(predicted_y, index=x_prediction.index, columns=['predicted_y'])
results_prediction_y = pd.concat([predicted_y], axis=1)
results_prediction_y.to_csv(f'csv/Predicted_{objective}_{method_name}.csv')