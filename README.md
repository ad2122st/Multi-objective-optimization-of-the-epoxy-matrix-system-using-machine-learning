# Multi-objective optimization of the epoxy matrix system using machine learning
This repository contains the code used in our survey paper: Multi-objective optimization of the epoxy matrix system using machine learning. It has been submitted to [Results in Materials](https://www.sciencedirect.com/journal/results-in-materials).

## Table of contents
* [Multi-objective optimization of the epoxy matrix system using machine learning](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning)
  * [Code](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#code)
  * [Machine Learning Models](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#machine-learning-models)

## Code

Regression code
>The code with a "regression" at the end of the filename builds the regression model to predict the material properties. The ratio of the training data to testing data is 8 to 2. You can choose a regression method from the five methods PLS, SVR, RF, KRR, and ANN. And the built model is evaluated by scatter plots and MAE and RMSE.

Optimization code
>The code with an "optimization" at the end of the filename optimizes the hyperparameters of each regression model by using [Optuna](https://dl.acm.org/doi/10.1145/3292500.3330701).


Prediction code
>The code with a "prediction" at the end of the filename predicts the material properties. The dataset is not split into test
and training data.

TPOT code
>The py file with "tpot" at the top is the code to predict the properties by using [Tree-based Pipeline Optimization Tool (TPOT)](https://academic.oup.com/bioinformatics/article/36/1/250/5511404), one of Automated machine learning (AutoML). TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

## Machine Learning Models
Artificial Neural Network (ANN) Models
>It was composed with the code based on [Keras API](https://www.tensorflow.org/tutorials/keras/regression).

Traditional Machine Learning Models
>Partial Least Squares (PLS), Support Vector Regression (SVR), Random Forest (RF), and Kernel Ridge Regression(KRR) are provided on [Scikit-learn](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html).
