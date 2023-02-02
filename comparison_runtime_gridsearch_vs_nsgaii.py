import sys
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import time
import optuna
import csv

def create(objective):
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
        alpha = 0.006483355871050879
        gamma = 0.01006433422359515
    if objective == 'Y2':
        alpha = 0.0510985382936244
        gamma = 0.018432581258103327
    if objective == 'Y3':
        alpha = 0.12355759143144339
        gamma = 0.046724961048862444

    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    model.fit(autoscaled_x_train, autoscaled_y_train)
    return model, x_train, y_train


model_y1, x_train_y1, y_train_y1 = create('Y1')
model_y2, x_train_y2, y_train_y2 = create('Y2')
model_y3, x_train_y3, y_train_y3 = create('Y3')

# Grid search
start = time.time()

x_prediction_y1 = pd.read_csv(f'csv/data_new_y1.csv', index_col=-1)
x_prediction_y1 = (x_prediction_y1.T / x_prediction_y1.T.sum()).T
autoscaled_x_y1_prediction = (x_prediction_y1 - x_train_y1.mean()) / x_train_y1.std()
autoscaled_x_y1_prediction = autoscaled_x_y1_prediction.fillna(0)
predicted_y1 = model_y1.predict(autoscaled_x_y1_prediction) * y_train_y1.std() + y_train_y1.mean()
predicted_y1 = pd.DataFrame(predicted_y1, columns=['Y1'], index=x_prediction_y1.index)

x_prediction_y2 = pd.read_csv(f'csv/data_new_y2.csv', index_col=-1)
x_prediction_y2 = (x_prediction_y2.T / x_prediction_y2.T.sum()).T
autoscaled_x_y2_prediction = (x_prediction_y2 - x_train_y2.mean()) / x_train_y2.std()
autoscaled_x_y2_prediction = autoscaled_x_y2_prediction.fillna(0)
predicted_y2 = model_y2.predict(autoscaled_x_y2_prediction) * y_train_y2.std() + y_train_y2.mean()
predicted_y2 = pd.DataFrame(predicted_y2, columns=['Y2'], index=x_prediction_y2.index)

x_prediction_y3 = pd.read_csv(f'csv/data_new_y3.csv', index_col=-1)
x_prediction_y3 = (x_prediction_y3.T / x_prediction_y3.T.sum()).T
autoscaled_x_y3_prediction = (x_prediction_y3 - x_train_y3.mean()) / x_train_y3.std()
autoscaled_x_y3_prediction = autoscaled_x_y3_prediction.fillna(0)
predicted_y3 = model_y3.predict(autoscaled_x_y3_prediction) * y_train_y3.std() + y_train_y3.mean()
predicted_y3 = pd.DataFrame(predicted_y3, columns=['Y3'], index=x_prediction_y3.index)

predictions = pd.concat([predicted_y1, predicted_y2, predicted_y3], axis=1)
predictions = predictions.query('Y1 < 1').query('Y2 > 1').query('Y3 < 1')
predictions.to_csv('csv/best_predictions_grid_search.csv')

grid_search_runtime = time.time() - start



# NSGAII search
start = time.time()

def objective(trial):
    a1 = trial.suggest_int('A1', 30, 100, 5)
    a7 = trial.suggest_int('A7', 0, 70, 5)
    a8 = trial.suggest_int('A8', 0, 35, 5)
    a10 = trial.suggest_int('A10', 0, 10, 2)
    b1 = trial.suggest_int('B1', 0, 50, 5)
    b2 = trial.suggest_int('B2', 0, 30, 5)
    b6 = trial.suggest_int('B6', 0, 40, 5)
    c1 = trial.suggest_int('C1', 0, 7)
    c4 = trial.suggest_int('C4', 0, 7)
    d2 = trial.suggest_int('D2', 5, 9)
    d6 = trial.suggest_int('D6', 3, 5)
    
    c0 = (a1 + a7 + a8 + a10 - 100) ** 2 - 10 ** -5
    c1 = 24 - b1 - b2 - b6
    c2 = b1 + b2 + b6 - 31
    c3 = (d2 + d6 - 12) ** 2 - 10 ** -5
    
    trial.set_user_attr("constraint", (c0, c1, c2, c3))

    x_prediction = pd.DataFrame(
        (a1, a7, a8, a10, b1, b2, b6, c1, c4, 12, d2, d6), 
        index=['A1', 'A7', 'A8', 'A10', 'B1', 'B2', 'B6', 'C1', 'C4', 'D1', 'D2', 'D6'], 
        columns=[trial.number]).T
    x_prediction = (x_prediction.T / x_prediction.T.sum()).T

    autoscaled_x_y1_prediction = (x_prediction - x_train_y1.mean()) / x_train_y1.std()
    autoscaled_x_y1_prediction = autoscaled_x_y1_prediction.fillna(0)
    predicted_y1 = model_y1.predict(autoscaled_x_y1_prediction) * y_train_y1.std() + y_train_y1.mean()

    autoscaled_x_y2_prediction = (x_prediction - x_train_y2.mean()) / x_train_y2.std()
    autoscaled_x_y2_prediction = autoscaled_x_y2_prediction.fillna(0)
    predicted_y2 = model_y2.predict(autoscaled_x_y2_prediction) * y_train_y2.std() + y_train_y2.mean()

    autoscaled_x_y3_prediction = (x_prediction - x_train_y3.mean()) / x_train_y3.std()
    autoscaled_x_y3_prediction = autoscaled_x_y3_prediction.fillna(0)
    predicted_y3 = model_y3.predict(autoscaled_x_y3_prediction) * y_train_y3.std() + y_train_y3.mean()

    return predicted_y1, predicted_y2, predicted_y3

def constraints(trial):
    return trial.user_attrs["constraint"]

sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
study = optuna.create_study(
    directions=["maximize", "minimize", "minimize"],
    sampler=sampler,
)
study.optimize(objective, n_trials=10000)

trials = sorted(study.best_trials, key=lambda t: t.values)

with open('csv/best_predictions_NSGAII_search.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([None, 'A1', 'A7', 'A8', 'A10', 'B1', 'B2', 'B6', 'C1', 'C4', 'D1', 'D2', 'D6', 'Y1', 'Y2', 'Y3'])
    f.close()

for trial in trials:
    sample = list(trial.params.values()) + trial.values
    sample.insert(0, f'Trial {trial.number}')
    sample.insert(10, 12)
    
    with open('csv/best_samples.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(sample)
        f.close()

nsgaii_search_runtime = time.time() - start


print(len(pd.read_csv('csv/best_samples.csv', index_col=0)))
print(f'Time difference\nAll search vs. NSGAII search\n{grid_search_runtime}  :  {nsgaii_search_runtime}')