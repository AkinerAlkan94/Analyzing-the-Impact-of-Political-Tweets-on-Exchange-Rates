import time

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
    preprocessing.Binarizer()


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
        print("---" + label + "---")
        print("PossibleParameters:")
        print(parameters)
        print("BestParameters:")
        print(prms)
        start = time.time()
        model_best.fit(normalized_train_x, numpy.ravel(y_train))
        train_predict, test_predict = make_predictions(model_best, normalized_train_x, normalized_test_x)
        end = time.time()
        time_lapsed = round(end - start, 4)
        print("Time Elapsed: " + str(time_lapsed))
        score_regressions(label, y_train, train_predict, y_test, test_predict)
        print()
        print()
        # score_classifications(label, y_train, train_predict, y_test, test_predict)

        # plot_graph(y_test, test_predict, label)
        return test_predict
    except Exception as e:
        print('Problem occurred *** ' + label + ' ***')
        print(e)


x_train, y_train, x_test, y_test = load_data(All)
x_train, y_train, x_test, y_test = convert_to_numpy(x_train, y_train, x_test, y_test)

min_max_scaler = create_min_max_scaler()
normalized_train_x = preprocessing.normalize(min_max_scaler.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(min_max_scaler.transform(x_test))
gaussian = execute_model('Gaussian Process', gaussianProcessorParameters, GaussianProcessRegressor)

quantile_transformer = create_quantile_transformer()
normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
linearRegression = execute_model('MLR', linearRegressionParameters, LinearRegression)

quantile_transformer = create_quantile_transformer()
normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
ridgeRegression = execute_model('Ridge', ridgeParameters, Ridge)

quantile_transformer = create_quantile_transformer()
normalized_train_x = preprocessing.normalize(quantile_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(quantile_transformer.transform(x_test))
mlp = execute_model('MLP', mlpParameters, MLPRegressor)

quantile_transformer = create_quantile_transformer()
normalized_train_x = preprocessing.normalize(quantile_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(quantile_transformer.transform(x_test))
xgb = execute_model('Extreme Gradient Boost Regressor', {}, XGBRegressor)

quantile_transformer = create_quantile_transformer()
normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
bayesianRidgeRegression = execute_model('Bayesian Ridge', bayesianRidgeParameters, BayesianRidge)

yeo_johnson_power_transformer = create_yeo_johnson_power_transformer()
normalized_train_x = yeo_johnson_power_transformer.fit_transform(x_train)
normalized_test_x = yeo_johnson_power_transformer.transform(x_test)
lassoRegression = execute_model('Lasso Regressor', lassoParameters, Lasso)

quantile_transformer = create_quantile_transformer()
normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
lassoLarsRegression = execute_model('Lasso Lars Regressor', lassoLarsParameters, LassoLars)

quantile_transformer = create_quantile_transformer()
normalized_train_x = quantile_transformer.fit_transform(x_train)
normalized_test_x = quantile_transformer.transform(x_test)
tweedieRegression = execute_model('Tweedie Regressor', tweedieParameters, TweedieRegressor)

min_max_scaler = create_min_max_scaler()
normalized_train_x = preprocessing.normalize(min_max_scaler.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(min_max_scaler.transform(x_test))
svrRegression = execute_model('SVR', svrParameters, SVR)

quantile_transformer = create_quantile_transformer()
normalized_train_x = preprocessing.normalize(quantile_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(quantile_transformer.transform(x_test))
sgdRegression = execute_model('SGD', sgdParameters, SGDRegressor)

yeo_johnson_power_transformer = create_yeo_johnson_power_transformer_non_standardized()
normalized_train_x = preprocessing.normalize(yeo_johnson_power_transformer.fit_transform(x_train))
normalized_test_x = preprocessing.normalize(yeo_johnson_power_transformer.transform(x_test))
kNeighborsRegression = execute_model('K Neighbors', kNeighborsParameters, KNeighborsRegressor)

max_abs_scaler = create_max_abs_scaler()
normalized_train_x = max_abs_scaler.fit_transform(x_train)
normalized_test_x = max_abs_scaler.transform(x_test)
mlpRegression = execute_model('MLP', mlpParameters, MLPRegressor)


#Delete Me Afterwardss
LinearRegression()
GaussianProcessRegressor()
Ridge()
BayesianRidge()
Lasso()
LassoLars()
TweedieRegressor()
SVR()
SGDRegressor()
KNeighborsRegressor


# y_test_for_plot = y_test
# x_train, y_train, x_test, y_test = load_data(All)
# # x_train, y_train, x_test, y_test = convert_to_numpy(x_train, y_train, x_test, y_test)
# full = read_file("CurrencyHistoryHourlyNicerTest.csv").dropna()
# train_full, test_full = train_test_split(full, test_size=55, shuffle=False)
#
# model = mvfts.MVFTS()
#
# allFeatures = ["dollar", "usa", "eurUsd", "gbpUsd", "trumpPosOverTotal", "obamaNegOverTotal"]
# features = allFeatures
#
# for feature in features:
#     try:
#         modVariable = variable.Variable(feature, data_label=feature, data=train_full)
#         model.append_variable(modVariable)
#     except:
#         print(feature)
#
# target = variable.Variable("usdTry", data_label="usdTry", data=train_full)
# model.target_variable = target
#
# model.fit(train_full)
# train_predict = numpy.array(model.predict(train_full))
# test_predict = numpy.array(model.predict(test_full))
# score_regressions('Fuzzy Regressor', y_train, train_predict, y_test, test_predict)
# fuzzy = test_predict
#
# plotting a line plot after changing it's width and height
# f = pyplot.figure()
# f.set_figwidth(8)
# f.set_figheight(4)
#
# pyplot.plot(y_test_for_plot, label='Actual', color='#000000', linewidth=3)
# pyplot.plot(linearRegression, label='MLR', color='#000000', linestyle="-")
# pyplot.plot(ridgeRegression, label='Ridge', color='#000000', linestyle=":")
# pyplot.plot(mlp, label='MLP', color='#000000', linestyle="-.")
# pyplot.plot(fuzzy, label='Fuzzy', color='#000000', linestyle="--")
#
# pyplot.legend()
# pyplot.xlabel('Time')
# pyplot.ylabel('USD/TRY')
# pyplot.show()
