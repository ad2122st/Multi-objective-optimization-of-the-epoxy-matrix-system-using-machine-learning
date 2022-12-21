# Multi-objective optimization of the epoxy matrix system using machine learning
This module contains the code used in our survey paper: Multi-objective optimization of the epoxy matrix system using machine learning. It has been submitted by [Results in Materials](https://www.sciencedirect.com/journal/results-in-materials).

# Table of contents
* [Multi-objective optimization of the epoxy matrix system using machine learning](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning)
  * [Code](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#code)
  * [Appendix](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#appendix)
  * [Requirement](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#requirement)
  * [Author](https://github.com/ad2122st/Multi-objective-optimization-of-the-epoxy-matrix-system-using-machine-learning/blob/main/README.md#author)

# Code

Regression code
>The py file with _regression at the end is the code to build the regression model to precict property from composition. The ratio of the taraining data to testing data is 8 to 2. You can choose a regression method from the four methods PLS, SVR, RF, and ANN. And the built model is evaluated by distribution plots and MAE and RMSE.

Optimization code
>The py file with _optimize at the end is the code to optimize the hyperparameters of each regression method. We used Optuna for optimizing the hyperparameters.

Prediction code
>The py file with _presdction at the end is the code to predict the unknown properties. The difference from regression code is ratio of trainig data to testing data, and it has no test data. It is able to predict the properties by entering untested samples.

# Machine Learning Models
Aritificial Neural Network (ANN) Models
>Keras API based 

Traditional Machine Learning Models
>

# Appendix
The py file with TPOT at the top is the code to predict the properties byusing Tree-based Pipeline Optimization Tool (TPOT), one of Automated machine learning (AutoML). TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

# Requirement
* scikit-learn 0.24.2
* tensorflow 2.4.0
* optuna 2.8.0
* TPOT 0.11.7

# Author
* Shigeru Taniguchi, Shogo Tamaki
* National Institute of Technology, Kitakyushu College Kitakyushu, JAPAN
