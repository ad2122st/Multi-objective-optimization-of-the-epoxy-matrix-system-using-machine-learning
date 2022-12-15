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

objective = 'Y2' # 'Y1' or 'Y2' or 'Y3'
method_name = 'kr' # 'pls' or 'svr' or 'kr' or 'rf'
number_of_test_samples = 0.2

if method_name != 'pls' and method_name != 'svr' and method_name != 'kr' and method_name != 'rf':
    sys.exit(f'There is no regression method {method_name}. Please choose from "pls", "svr", "kr" or "rf".')
    
dataset = pd.read_csv(f'csv/{objective}.csv', index_col=-1)

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
        alpha = 0.006483355871050879
        gamma = 0.01006433422359515
    if objective == 'Y2':
        alpha = 0.0510985382936244
        gamma = 0.018432581258103327
    if objective == 'Y3':
        alpha = 0.12355759143144339
        gamma = 0.046724961048862444

    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    
if method_name == 'rf':
    if objective == 'Y1':
        bootstrap = 'True'
        criterion = 'absolute_error'
        max_depth = 345
        max_features = 'sqrt'
        max_leaf_nodes = 328
        n_estimators = 391
        min_samples_split = 2
        min_samples_leaf = 1
        n_jobs = 2
    if objective == 'Y2':
        bootstrap = 'False'
        criterion = 'absolute_error'
        max_depth = 852
        max_features = 'sqrt'
        max_leaf_nodes = 290
        n_estimators = 631
        min_samples_split = 2
        min_samples_leaf = 1
        n_jobs = 2
    if objective == 'Y3':
        bootstrap = 'True'
        criterion = 'absolute_error'
        max_depth = 159
        max_features = 'log2'
        max_leaf_nodes = 332
        n_estimators = 152
        min_samples_split = 2
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
    plt.savefig(f"img/{objective}_{word}_{method_name}.pdf")

    print(f'\n{objective} with {method_name} for {word} data')
    print(f'r^2 :', metrics.r2_score(y, estimated_y))
    print(f'RMSE :', metrics.mean_squared_error(y, estimated_y) ** 0.5)
    print(f'MAE :', metrics.mean_absolute_error(y, estimated_y))

    y_for_save = pd.DataFrame(y)
    y_for_save.columns = [f'Actual {objective}']
    y_error_train = y_for_save.iloc[:, 0] - estimated_y.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)
    y_error_train.columns = [f'error_of_{objective}(actual_{objective}-predicted_{objective})']
    results_train = pd.concat([estimated_y, y_for_save, y_error_train], axis=1)
    results_train.to_csv(f'csv/Predicted_{objective}_{word}_{method_name}.csv')
