# Multi-objective optimization of the epoxy matrix system using machine learning

論文の趣旨

# Explanation of sample code

+ Regression
>末尾が_regression.pyのファイル

>組成から物性予測モデルを構築する。
>分布図とMAE、RMSEでモデルの評価する。

+ Optimize
>末尾が_optimize.pyのファイル

>Optunaを用いて各回帰手法のハイパーパラメータの最適化を行う。

+ Prediction
>末尾が_prediction.pyのファイル

>未知のサンプルから物性を予測する。

# Appendix
AutoMLの一つTPOTを用いて物性予測モデルを構築する。

# requirement
* scikit-learn 0.24.2
* tensorflow 2.4.0
* optuna 2.8.0
* TPOT 0.11.7

# Author
* 作成者
* 所属
* eメール
