import pandas
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, mean_squared_log_error, \
    median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, accuracy_score
from math import sqrt

from sklearn.model_selection import train_test_split

from Constants import stocks, currencies, tweets

All = 0
OnlyTweet = 1
OnlyStock = 2
OnlyCurrency = 3
StockAndCurrency = 4


def read_file(filename):
    data = pandas.read_csv(filename, sep=',', header=[0])
    return data


def plot_graph(actual_data, predictions, label='Predicted'):
    # plot
    pyplot.plot(predictions, label=label)
    pyplot.plot(actual_data, label='Actual')
    pyplot.legend()
    pyplot.xlabel('Time')
    pyplot.ylabel('USD/TRY')
    pyplot.show()


def make_predictions(model, x_train, x_test):
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    return train_predict, test_predict


def score_regressions(header, y_train, train_predict, y_test, test_predict):
    # print("\n--- Printing Scores " + header + " ---")
    # scorings = ''
    # scorings = scorings + 'Train'
    # scorings = scorings + ' MSE:' + format(mean_squared_error(y_train, train_predict), '.10f')
    # scorings = scorings + ' / RMSE:' + format(sqrt(mean_squared_error(y_train, train_predict)), '.10f')
    # scorings = scorings + ' / MAE:' + format(mean_absolute_error(y_train, train_predict), '.10f')
    # print(scorings)

    scorings = ''
    scorings = scorings + 'Test'
    scorings = scorings + ' MSE:' + format(mean_squared_error(y_test, test_predict), '.10f')
    scorings = scorings + ' / RMSE:' + format(sqrt(mean_squared_error(y_test, test_predict)), '.10f')
    scorings = scorings + ' / MAE:' + format(mean_absolute_error(y_test, test_predict), '.10f')
    scorings = scorings + ' (' + header + ')'
    print(scorings)


def score_classifications(label, y_train, train_predict, y_test, test_predict):
    class_y_train = convert_as_classification(y_train)
    class_y_train_predicted = convert_as_classification(train_predict)
    class_y_test = convert_as_classification(y_test)
    class_y_test_predicted = convert_as_classification(test_predict)

    train_acc = accuracy_score(class_y_train, class_y_train_predicted)
    test_acc = accuracy_score(class_y_test, class_y_test_predicted)
    print('Train Acc: ' + format(train_acc, '.4'))
    print('Test Acc: ' + format(test_acc, '.4'))


def load_data(strategy=All, target_feature='usdTry'):
    full = read_file("CurrencyHistoryHourlyNicerTest.csv").dropna()
    full_x = []

    if strategy == All:
        full_x = full[tweets + stocks + currencies]

    elif strategy == OnlyTweet:
        full_x = full[tweets]

    elif strategy == OnlyStock:
        full_x = full[stocks]

    elif strategy == OnlyCurrency:
        full_x = full[currencies]

    elif strategy == StockAndCurrency:
        full_x = full[currencies + stocks]

    full_y = full[[target_feature]]

    # Drop Last Record
    full_x = full_x.iloc[:-1]
    # Drop First Y for shift
    full_y = full_y.iloc[1:]

    train_x, test_x = train_test_split(full_x, test_size=55, shuffle=False)
    train_y, test_y = train_test_split(full_y, test_size=55, shuffle=False)


    # return x_train, y_train, x_test, y_test
    return train_x, train_y, test_x, test_y


def convert_to_numpy(a, b, c, d):
    return a.to_numpy(), b.to_numpy(), c.to_numpy(), d.to_numpy()


def convert_as_classification(target_feature):
    rates = target_feature
    trends = []

    i = 1
    while i < rates.size:
        diff = rates[i] - rates[i - 1]
        value = 0
        if diff > 0:
            value = 1
        trends.append(value)
        i = i + 1

    trends.append(0)
    return trends
