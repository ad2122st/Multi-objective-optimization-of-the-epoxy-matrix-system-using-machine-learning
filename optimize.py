import sys
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import optuna

objective = 'Y1' # 'Y1' or 'Y2' or 'Y3'
method_name = 'pls' # 'pls' or 'svr' or 'kr' or 'rf'
num_trial = 1000
for_prediction = False # True or False

dataset = pd.read_csv(f'csv/{objective}.csv', index_col=-1)

number_of_test_samples = 0.2
fold_number = 10

if method_name != 'pls' and method_name != 'svr' and method_name != 'kr' and method_name != 'rf':
    sys.exit(f'There is no regression method {method_name}. Please choose from "pls", "svr", "kr" or "rf".')
    
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
    if method_name == 'pls':
        component = trial.suggest_int('component', 1, 25)

        model = PLSRegression(n_components = component)
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv = fold_number))
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)

        indicator = -r2_in_cv
    
    else:
        if method_name == 'svr':
            gamma = trial.suggest_loguniform('gamma',1e-5,1e5)
            c = trial.suggest_loguniform('C',1e-5,1e5)
            epsilon = trial.suggest_loguniform('epsilon',1e-5,1e5)

            model = svm.SVR(kernel='rbf', C = c, epsilon = epsilon, gamma = gamma)

        if method_name == 'rf':
            bootstrap = trial.suggest_categorical('bootstrap',['True', 'False'])
            criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
            max_depth = trial.suggest_int('max_depth', 1, 1000)
            max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
            n_estimators =  trial.suggest_int('n_estimators', 2, 1000)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            
            model = RandomForestRegressor(bootstrap = bootstrap, 
                                    criterion = criterion,
                                    max_depth = max_depth, max_features = max_features,
                                    max_leaf_nodes = max_leaf_nodes, n_estimators = n_estimators,
                                    min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf,
                                    n_jobs=2)

        if method_name == 'kr':
            alpha = trial.suggest_uniform('alpha', 0.0, 100)
            gamma = trial.suggest_loguniform('gamma', 1e-2, 1e+1)

            model = KernelRidge(kernel = 'rbf', alpha = alpha, gamma = gamma)
        
        score = cross_val_score(model, autoscaled_x_train, autoscaled_y_train, cv = fold_number, scoring = "neg_mean_absolute_error")
        indicator = -1.0*score.mean()
        
    return indicator

study = optuna.create_study()
study.optimize(objective, n_trials = num_trial)

print('\nBest trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))