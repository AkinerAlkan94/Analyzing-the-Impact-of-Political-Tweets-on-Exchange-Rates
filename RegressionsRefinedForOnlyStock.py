import numpy
from matplotlib import pyplot
from pyFTS.models.multivariate import mvfts, variable
from pyFTS.partitioners import Grid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso, LassoLars, TweedieRegressor, \
    SGDRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from Constants import mlpParameters, linearRegressionParameters, ridgeParameters, bayesianRidgeParameters, \
    lassoParameters, lassoLarsParameters, kNeighborsParameters, tweedieParameters, svrParameters, sgdParameters, \
    gaussianProcessorParameters, currencies, stocks
from Utilities import load_data, make_predictions, score_regressions, plot_graph, All, convert_to_numpy, \
    convert_as_classification, score_classifications, read_file, OnlyCurrency, OnlyTweet, OnlyStock
from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')


def create_min_max_scaler():
    return preprocessing.MinMaxScaler()


def create_quantile_transformer():
    return preprocessing.QuantileTransformer(random_state=0)


def create_binarizer():
    return preprocessing.Binarizer()


def create_max_abs_scaler():
    return preprocessing.MaxAbsScaler()


def create_yeo_johnson_power_transformer():
    return preprocessing.PowerTransformer()

def create_yeo_johnson_power_transformer_non_standardized():
    return preprocessing.PowerTransformer(standardize=False)



def find_best_parameters(label, model, parameters):
    clf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', verbose=0)
    clf.fit(normalized_train_x, numpy.ravel(y_train))
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


x_train, y_train, x_test, y_test = load_data(OnlyStock)
x_train, y_train, x_test, y_test = convert_to_numpy(x_train, y_train, x_test, y_test)

normalized_train_x = preprocessing.normalize(x_train)
normalized_test_x = preprocessing.normalize(x_test)
gaussian = execute_model('Gaussian Process', gaussianProcessorParameters, GaussianProcessRegressor)

quantile_transformer = create_quantile_transformer()
normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
linearRegression = execute_model('MLR', linearRegressionParameters, LinearRegression)


normalized_train_x = x_train
normalized_test_x = x_test
ridgeRegression = execute_model('Ridge', ridgeParameters, Ridge)

quantile_transformer = create_quantile_transformer()
normalized_train_x = preprocessing.normalize(quantile_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(quantile_transformer.transform(x_test))
mlp = execute_model('MLP', mlpParameters, MLPRegressor)

normalized_train_x = x_train
normalized_test_x = x_test
xgb = execute_model('Extreme Gradient Boost Regressor', {}, XGBRegressor)

yeo_johnson_power_transformer = create_yeo_johnson_power_transformer()
normalized_train_x = preprocessing.normalize(yeo_johnson_power_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(yeo_johnson_power_transformer.transform(x_test))
bayesianRidgeRegression = execute_model('Bayesian Ridge', bayesianRidgeParameters, BayesianRidge)

normalized_train_x = x_train
normalized_test_x = x_test
lassoRegression = execute_model('Lasso Regressor', lassoParameters, Lasso)

normalized_train_x = x_train
normalized_test_x = x_test
lassoLarsRegression = execute_model('Lasso Lars Regressor', lassoLarsParameters, LassoLars)

normalized_train_x = preprocessing.normalize(x_train)
normalized_test_x = preprocessing.normalize(x_test)
tweedieRegression = execute_model('Tweedie Regressor', tweedieParameters, TweedieRegressor)

min_max_scaler = create_min_max_scaler()
normalized_train_x = min_max_scaler.fit_transform(x_train)
normalized_test_x = min_max_scaler.transform(x_test)
svrRegression = execute_model('SVR', svrParameters, SVR)

binarizer = create_binarizer()
normalized_train_x = preprocessing.normalize(binarizer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(binarizer.transform(x_test))
sgdRegression = execute_model('SGD', sgdParameters, SGDRegressor)

normalized_train_x = x_train
normalized_test_x = x_test
kNeighborsRegression = execute_model('K Neighbors', kNeighborsParameters, KNeighborsRegressor)

quantile_transformer = create_quantile_transformer()
normalized_train_x = preprocessing.normalize(quantile_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(quantile_transformer.transform(x_test))
mlpRegression = execute_model('MLP', mlpParameters, MLPRegressor)