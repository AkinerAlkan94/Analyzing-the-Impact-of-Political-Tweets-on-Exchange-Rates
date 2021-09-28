# Working Finished. Always following one step back because only checks the output(1 feature)

import pandas
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

import warnings

warnings.filterwarnings('ignore')


def read_file(filename):
    data = pandas.read_csv(filename, sep=' ', header=[0])
    return data


my_train = read_file("Train.txt").dropna()
y_train = my_train.iloc[:, -1]

my_test = read_file("Test.txt").dropna()
y_test = my_test.iloc[:, -1]

history = [x for x in y_train]
predictions = list()
for t in range(len(y_test)):
    model = ARIMA(history, order=(2, 2, 2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = y_test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(y_test, predictions)
print('Test MSE: %.5f' % error)

# plot
pyplot.plot(y_test)
pyplot.plot(predictions, color='red')
pyplot.show()
