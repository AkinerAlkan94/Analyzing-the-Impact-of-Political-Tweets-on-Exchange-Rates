import time

import pyFTS
import dill
import numpy as np

from pyFTS.models.multivariate import variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.partitioners import Grid, Simple
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from Constants import tweets, currencies, stocks
from Utilities import All, load_data, convert_to_numpy, score_regressions, read_file, OnlyCurrency, OnlyStock, OnlyTweet

x_train, y_train, x_test, y_test = load_data(All)
# x_train, y_train, x_test, y_test = convert_to_numpy(x_train, y_train, x_test, y_test)
full = read_file("CurrencyHistoryHourlyNicerTest.csv").dropna()
train_full, test_full = train_test_split(full, test_size=55, shuffle=False)

model = mvfts.MVFTS()

stocks = ["dollar", "usa"]
tweets = ["trumpPosOverTotal", "obamaNegOverTotal"]
currencies = ["eurUsd","gbpUsd"]

# "trumpPosOverTotal", "obamaNegOverTotal", "eurUsd", "usdChf"

allFeatures = ["dollar","usa" ,"eurUsd","gbpUsd","trumpPosOverTotal", "obamaNegOverTotal"]
features = allFeatures

for feature in features:
    try:
        modVariable = variable.Variable(feature, data_label=feature, data=train_full)
        model.append_variable(modVariable)
    except:
        print(feature)

start = time.time()
target = variable.Variable("usdTry", data_label="usdTry", data=train_full)
model.target_variable = target

model.fit(train_full)
train_predict = np.array(model.predict(train_full))
test_predict = np.array(model.predict(test_full))
end = time.time()
time_elapsed = round(end - start, 2)
print(time_elapsed)
score_regressions('Fuzzy Regressor', y_train, train_predict, y_test, test_predict)
