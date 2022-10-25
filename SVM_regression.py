import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from data_preparation import create_learning_df
import pickle
from sklearn.linear_model import LinearRegression
import os

def build_fit_regressor(ticker, degree, split, visualize = False):

    regressor = svm.SVR(kernel='poly', degree=degree, verbose=3)
    poly = PolynomialFeatures(degree=degree)
    regressor = LinearRegression()

    master = create_learning_df(ticker)
    master.set_index(pd.DatetimeIndex(master['for index']), inplace=True)
    master.drop('for index', axis=1, inplace=True)

    # scaler = StandardScaler()
    # master[master.columns] = scaler.fit_transform(master[master.columns])

    data = master.drop('target', axis=1)
    target = master['target']

    data = master.drop('marketClose', axis = 1)
    target = master['marketClose']
    closes = master['marketClose']

    train_data = data.iloc[:round(split * len(data)), :]
    train_data = poly.fit_transform(train_data)
    train_target = target[:round(split * len(data))]

    test_data = data.iloc[round(split * len(data)):, :]
    test_data = poly.fit_transform(test_data)
    test_target = target[round(split * len(data)):]

    data = poly.fit_transform(data)

    if len(train_data) != len(train_target):
        raise IndexError('train data len', len(train_data), 'and train target len', len(train_target),
                         'do not match.')
    if len(test_data) != len(test_target):
        raise IndexError('test data len', len(test_data), 'and test target len', len(test_target), 'do not match.')

    if f'{ticker}_polyreg_model.sav' not in os.listdir():
        regressor.fit(train_data, train_target)
        prediction = regressor.predict(test_data)
        filename = f'{ticker}_polyreg_model.sav'
        pickle.dump(regressor, open(filename, 'wb'))
        score = regressor.score(test_data, test_target)

    else:
        regressor = pickle.load(open(f'{ticker}_polyreg_model.sav', 'rb'))
        prediction = regressor.predict(test_data)
        full_prediction = regressor.predict(data)
        score = regressor.score(test_data, test_target)

    print(score)

    if visualize:
        plt.plot(range(len(closes)), closes, color = 'red')
        #plt.plot(range(len(target)), full_prediction, color = 'green', linewidth = .5)

        # plt.plot(target.index.tolist(), target, color = 'red')
        # plt.scatter(test_target.index.tolist(), prediction, color = 'green')
        plt.axvline(round(split*len(data)))
        plt.show()
    return prediction, score

prediction, score = build_fit_regressor('SPY', 3, 3/4, visualize = True)

