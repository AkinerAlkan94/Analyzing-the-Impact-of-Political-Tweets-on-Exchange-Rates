import time

import numpy
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Constants import logisticRegressionParameters, kNeighborsParameters, decisionTreeParameters, \
    randomForestParameters, gradientBoostingParameters, svcParameters, gaussianNbParameters, multinomialNbParameters, \
    complementNbParameters, bernoulliNbParameters, linearDiscriminantParameters, adaBoostParameters, mlpParameters, \
    tol_values, bool_values, multi_class_values, random_state_values, solver_values
from Utilities import convert_as_classification, make_predictions, plot_graph, load_data, All, convert_to_numpy, \
    OnlyStock, OnlyCurrency, OnlyTweet

import warnings

warnings.filterwarnings('ignore')

def find_best_parameters(label, model, parameters):
    clf = GridSearchCV(model, parameters, scoring='accuracy', verbose=0)
    clf.fit(normalized_train_x, numpy.ravel(class_y_train))
    # print('')
    # print('---' + label + '---')
    # print(clf.best_params_)
    return clf.best_params_


def execute_model(label, parameters, model):
    try:
        prms = find_best_parameters(label, model(), parameters)
        start = time.time()
        model_best = model(**prms)
        model_best.fit(normalized_train_x, numpy.ravel(class_y_train))
        train_predict, test_predict = make_predictions(model_best, normalized_train_x, normalized_test_x)
        end = time.time()
        time_lapsed = round(end - start, 4)
        print(label + " Time Elapsed: " + str(time_lapsed))
        test_acc = accuracy_score(class_y_test, test_predict)
        print('Test Acc: ' + format(test_acc, '.4') + ' (' + label + ')')
    except Exception as e:
        print('Problem occurred *** ' + label + ' ***')
        print(e)


def run_classifiers():
    # execute_model('LogisticRegression Classifier', logisticRegressionParameters, LogisticRegression)
    # execute_model('KNeighbors Classifier', kNeighborsParameters, KNeighborsClassifier)
    # execute_model('Extreme Gradient Boosting Classifier', [{}], XGBClassifier)
    # execute_model('Decision Tree Classifier', decisionTreeParameters, DecisionTreeClassifier)
    # execute_model('Random Forest Classifier', randomForestParameters, RandomForestClassifier)
    # execute_model('Gradient Boosting Classifier', gradientBoostingParameters, GradientBoostingClassifier)
    # execute_model('Support Vector Classifier', svcParameters, SVC)
    # execute_model('Gaussian Naive Bayes Classifier', gaussianNbParameters, GaussianNB)
    # execute_model('Bernoulli Naive Bayes Classifier', bernoulliNbParameters, BernoulliNB)
    # execute_model('Linear Discriminant Classifier', linearDiscriminantParameters, LinearDiscriminantAnalysis)
    # execute_model('Ada Boost Classifier', adaBoostParameters, AdaBoostClassifier)
    # execute_model('MLP Classifier ( FeedForward ANN )', mlpParameters, MLPClassifier)
    execute_model('LogisticRegression Classifier', [{}], LogisticRegression)
    execute_model('KNeighbors Classifier', [{}], KNeighborsClassifier)
    execute_model('Extreme Gradient Boosting Classifier', [{}], XGBClassifier)
    execute_model('Support Vector Classifier', [{}], SVC)
    execute_model('Gaussian Naive Bayes Classifier', [{}], GaussianNB)
    execute_model('Bernoulli Naive Bayes Classifier', [{}], BernoulliNB)
    execute_model('MLP Classifier ( FeedForward ANN )', [{}], MLPClassifier)

train_x, train_y, test_x, test_y = load_data(OnlyStock)
train_x, train_y, test_x, test_y = convert_to_numpy(train_x, train_y, test_x, test_y)

class_y_train = convert_as_classification(train_y)
class_y_test = convert_as_classification(test_y)

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

normalized_train_x = train_x
normalized_test_x = test_x
print('With No Alteration')
run_classifiers()

# normalized_train_x = preprocessing.scale(train_x)
# normalized_test_x = preprocessing.scale(test_x)
# print('\n\n\n\n Only Scale')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(train_x)
# normalized_test_x = preprocessing.normalize(test_x)
# print('\n\n\n\n Only Normalize')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(preprocessing.scale(train_x))
# normalized_test_x = preprocessing.normalize(preprocessing.scale(test_x))
# print('\n\n\n\n Scale and Normalize')
# run_classifiers()
#
# normalized_train_x = min_max_scaler.fit_transform(train_x)
# normalized_test_x = min_max_scaler.transform(test_x)
# print('\n\n\n\n Only MinMax Scale')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(min_max_scaler2.fit_transform(train_x))
# normalized_test_x = preprocessing.normalize(min_max_scaler2.transform(test_x))
# print('\n\n\n\n MinMax Scale and Normalize')
# run_classifiers()
#
# normalized_train_x = quantile_transformer.fit_transform(train_x)
# normalized_test_x = quantile_transformer.transform(test_x)
# print('\n\n\n\n Only Quantile Transform')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(quantile_transformer2.fit_transform(train_x))
# normalized_test_x = preprocessing.normalize(quantile_transformer2.transform(test_x))
# print('\n\n\n\n Quantile Transform and Normalize')
# run_classifiers()
#
# normalized_train_x = binarizer.fit_transform(train_x)
# normalized_test_x = binarizer.transform(test_x)
# print('\n\n\n\n Only Binarizer')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(binarizer2.fit_transform(train_x))
# normalized_test_x = preprocessing.normalize(binarizer2.transform(test_x))
# print('\n\n\n\n Binarizer and Normalize')
# run_classifiers()
#
# normalized_train_x = max_abs_scaler.fit_transform(train_x)
# normalized_test_x = max_abs_scaler.transform(test_x)
# print('\n\n\n\n Only Max Absolute Scale')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(max_abs_scaler2.fit_transform(train_x))
# normalized_test_x = preprocessing.normalize(max_abs_scaler2.transform(test_x))
# print('\n\n\n\n Max Absolute Scale and Normalize')
# run_classifiers()
#
# normalized_train_x = yeo_johnson_power_transformer.fit_transform(train_x)
# normalized_test_x = yeo_johnson_power_transformer.transform(test_x)
# print('\n\n\n\n Only Yeo Johnson Power Transform')
# run_classifiers()
#
# normalized_train_x = preprocessing.normalize(yeo_johnson_power_transformer_standardized2.fit_transform(train_x))
# normalized_test_x = preprocessing.normalize(yeo_johnson_power_transformer_standardized2.transform(test_x))
# print('\n\n\n\n Only Yeo Johnson Power Transform and Normalize')
# run_classifiers()
