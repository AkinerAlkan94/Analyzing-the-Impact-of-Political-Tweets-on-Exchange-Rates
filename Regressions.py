import xgboost
import numpy
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso, LassoLars, TweedieRegressor, \
    SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor

# from Constants import linearRegressionParameters, ridgeParameters, bayesianRidgeParameters, lassoParameters, \
#     lassoLarsParameters, tweedieParameters, svrParameters, sgdParameters, kNeighborsParameters, \
#     gaussianProcessorParameters, decisionTreeParameters
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBModel

from Constants import mlpParameters, linearRegressionParameters, ridgeParameters, bayesianRidgeParameters, \
    lassoParameters, lassoLarsParameters, kNeighborsParameters, tweedieParameters, svrParameters, sgdParameters, \
    gaussianProcessorParameters
from Utilities import load_data, make_predictions, score_regressions, plot_graph, All, convert_to_numpy, \
    convert_as_classification, score_classifications, OnlyCurrency, OnlyStock, OnlyTweet
from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')


# linearRegressionParameters = [{'copy_X': [True], 'fit_intercept': [False], 'normalize': [True]}]
# ridgeParameters = [{'alpha': [0.6], 'copy_X': [True], 'fit_intercept': [True], 'normalize': [False], 'tol': [1]}]
# bayesianRidgeParameters = [
#     {'compute_score': [True], 'copy_X': [True], 'fit_intercept': [True], 'normalize': [False], 'tol': [0.2]}]
# lassoParameters = [
#     {'alpha': [1e-05], 'copy_X': [True], 'fit_intercept': [True], 'normalize': [True], 'positive': [False],
#      'precompute': [True], 'selection': ['cyclic'], 'tol': [0.001], 'warm_start': [True]}]
# lassoLarsParameters = [
#     {'alpha': [1e-05], 'copy_X': [True], 'fit_intercept': [True], 'fit_path': [True], 'normalize': [True],
#      'positive': [False]}]
# tweedieParameters = [{'alpha': [0.001], 'fit_intercept': [True], 'power': [0], 'tol': [1], 'warm_start': [True]}]
# svrParameters = [{'C': [500], 'epsilon': [0.001], 'shrinking': [True], 'tol': [0.001]}]
# sgdParameters = [
#     {'alpha': [1.0], 'average': [True], 'early_stopping': [False], 'epsilon': [1], 'fit_intercept': [False],
#      'shuffle': [True], 'warm_start': [False]}]
# kNeighborsParameters = [{'algorithm': ['brute'], 'leaf_size': [1], 'n_neighbors': [50], 'weights': ['distance']}]
# gaussianProcessorParameters = [{'alpha': [1.0], 'copy_X_train': [True], 'normalize_y': [True]}]

def find_best_parameters(label, model, parameters):
    clf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', verbose=0)
    clf.fit(normalized_train_x, numpy.ravel(y_train))
    # print('')
    # print('---' + label + '---')
    # print(clf.best_params_)
    return clf.best_params_


def execute_model(label, parameters, model):
    try:
        prms = find_best_parameters(label, model(), parameters)
        model_best = model(**prms)
        model_best.fit(normalized_train_x, numpy.ravel(y_train))
        train_predict, test_predict = make_predictions(model_best, normalized_train_x, normalized_test_x)
        score_regressions(label, y_train, train_predict, y_test, test_predict)
        # score_classifications(label, y_train, train_predict, y_test, test_predict)

        # plot_graph(y_test, test_predict, label)
        return test_predict
    except Exception as e:
        print('Problem occurred *** ' + label + ' ***')
        print(e)


def run_regressors():
    # pyplot.plot(y_test, label='Actual')
    #
    # pyplot.legend()
    # pyplot.xlabel('Time')
    # pyplot.ylabel('USD/TRY')
    # pyplot.show()

    xgb = execute_model('Extreme Gradient Boost Regressor', {}, XGBRegressor)
    linearRegression = execute_model('Linear Regression Regressor', linearRegressionParameters, LinearRegression)
    ridge = execute_model('Ridge Regressor', ridgeParameters, Ridge)
    bayesianRidge = execute_model('Bayesian Ridge Regressor', bayesianRidgeParameters, BayesianRidge)
    lasso = execute_model('Lasso Regressor', lassoParameters, Lasso)
    lassoLars = execute_model('Lasso Lars Regressor', lassoLarsParameters, LassoLars)
    tweedie = execute_model('Tweedie Regressor', tweedieParameters, TweedieRegressor)
    svr = execute_model('SVR Regressor', svrParameters, SVR)
    sgd = execute_model('SGD Regressor', sgdParameters, SGDRegressor)
    kNeighbors = execute_model('K Neighbors Regressor', kNeighborsParameters, KNeighborsRegressor)
    gaussian = execute_model('Gaussian Process Regressor', gaussianProcessorParameters, GaussianProcessRegressor)
    mlp = execute_model('MLP Regressor ( FeedForward ANN )', mlpParameters, MLPRegressor)


x_train, y_train, x_test, y_test = load_data(OnlyStock)
x_train, y_train, x_test, y_test = convert_to_numpy(x_train, y_train, x_test, y_test)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler2 = preprocessing.MinMaxScaler()
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
quantile_transformer2 = preprocessing.QuantileTransformer(random_state=0)
binarizer = preprocessing.Binarizer()
binarizer2 = preprocessing.Binarizer()
max_abs_scaler = preprocessing.MaxAbsScaler()
max_abs_scaler2 = preprocessing.MaxAbsScaler()
yeo_johnson_power_transformer = preprocessing.PowerTransformer(standardize=False)
yeo_johnson_power_transformer2 = preprocessing.PowerTransformer(standardize=False)
yeo_johnson_power_transformer_standardized = preprocessing.PowerTransformer()
yeo_johnson_power_transformer_standardized2 = preprocessing.PowerTransformer()

normalized_train_x = x_train
normalized_test_x = x_test
print('With No Alteration')
run_regressors()

normalized_train_x = preprocessing.scale(x_train)
normalized_test_x = preprocessing.scale(x_test)
print('\n\n\n\n Only Scale')
run_regressors()

normalized_train_x = preprocessing.normalize(x_train)
normalized_test_x = preprocessing.normalize(x_test)
print('\n\n\n\n Only Normalize')
run_regressors()

normalized_train_x = preprocessing.normalize(preprocessing.scale(x_train))
normalized_test_x = preprocessing.normalize(preprocessing.scale(x_test))
print('\n\n\n\n Scale and Normalize')
run_regressors()

normalized_train_x = min_max_scaler.fit_transform(x_train)
normalized_test_x = min_max_scaler.transform(x_test)
print('\n\n\n\n Only MinMax Scale')
run_regressors()

normalized_train_x = preprocessing.normalize(min_max_scaler2.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(min_max_scaler2.transform(x_test))
print('\n\n\n\n MinMax Scale and Normalize')
run_regressors()

normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
print('\n\n\n\n Only Quantile Transform')
run_regressors()

normalized_train_x = preprocessing.normalize(quantile_transformer2.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(quantile_transformer2.transform(x_test))
print('\n\n\n\n Quantile Transform and Normalize')
run_regressors()

normalized_train_x = binarizer.fit_transform(x_train)
normalized_test_x = binarizer.transform(x_test)
print('\n\n\n\n Only Binarizer')
run_regressors()

normalized_train_x = preprocessing.normalize(binarizer2.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(binarizer2.transform(x_test))
print('\n\n\n\n Binarizer and Normalize')
run_regressors()

normalized_train_x = max_abs_scaler.fit_transform(x_train)
normalized_test_x = max_abs_scaler.transform(x_test)
print('\n\n\n\n Only Max Absolute Scale')
run_regressors()

normalized_train_x = preprocessing.normalize(max_abs_scaler2.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(max_abs_scaler2.transform(x_test))
print('\n\n\n\n Max Absolute Scale and Normalize')
run_regressors()

normalized_train_x = yeo_johnson_power_transformer.fit_transform(x_train)
normalized_test_x = yeo_johnson_power_transformer.transform(x_test)
print('\n\n\n\n Only Yeo Johnson Power Transform')
run_regressors()

normalized_train_x = preprocessing.normalize(yeo_johnson_power_transformer_standardized2.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(yeo_johnson_power_transformer_standardized2.transform(x_test))
print('\n\n\n\n Only Yeo Johnson Power Transform and Normalize')
run_regressors()
