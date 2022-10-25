import pandas as pd
import numpy as np
import os
import datetime as dt
import math
from getting_intraday_data import get_data
from indicators import SMA, EMA, Stochastic, MACD, RSI, ichimoku
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def combine_days(ticker):
    datelist = os.listdir(f'{ticker}_intraday_data')
    current_day = datelist[-1][:-4]

    if current_day != dt.datetime.today().strftime('%Y-%m-%d'):
        get_data(ticker)
        datelist = os.listdir(f'{ticker}_intraday_data')
    else:
        print('Up to date.')

    #datelist = [day for day in datelist if len(pd.read_csv(f'{ticker}_intraday_data/{day}'))]

    master = pd.DataFrame()
    dt_index = []
    
    for day in datelist:
        current_data = pd.read_csv(f'{ticker}_intraday_data/{day}')
        if not len(current_data):
            continue

        current_data.dropna(axis=0, inplace=True)
        dt_index += current_data.iloc[:, 0].tolist()
        current_data = current_data[['marketOpen', 'marketHigh', 'marketLow', 'marketClose', 'marketVolume', 'marketNumberOfTrades', 'marketNotional', 'marketChangeOverTime']]


        # SCALING
        # for label in ['marketOpen','marketHigh', 'marketLow', 'marketClose', 'marketVolume']:
        #     try:
        #         current_data[label] = current_data[label]/current_data[label][0]
        #         skip = False
        #     except:
        #         print(f'{day} is invalid')
        #         skip = True
        #
        # if skip:
        #     continue

        master = master.append(current_data)


    master['for index'] = dt_index

    return master




def make_new_cols(ticker, target_shift = -30):
    master = combine_days(ticker)

    open = master['marketOpen']
    high = master['marketHigh']
    low = master['marketLow']
    close = master['marketClose']

    for period in [9, 20, 50, 100, 200]:
        # master[f'{period}SMA'] = master['marketClose'].rolling(window = period).mean()
        master[f'{period}SMA'] = SMA(close, period)
        master[f'{period}EMA'] = EMA(close, period)

    master['MACD'] = MACD(close)
    master['Stochastic'] = Stochastic(high, low, close)
    master['RSI'] = RSI(close)

    for i, item in enumerate(ichimoku(high,low, close)):
        master[f'ich{i}'] = item

    master['target'] = master['marketClose'].shift(target_shift) - master['marketClose']

    # ind = pd.DatetimeIndex(master['for index'])
    # master.set_index(ind, inplace=True)
    # master.drop('for index', axis=1, inplace=True)
    master = master.loc[:,'marketOpen':]
    print(len(master))
    master.replace([np.inf, -np.inf], np.nan, inplace = True)
    master.dropna(axis = 0, inplace = True)
    print(len(master))
    # master.dropna(axis = 0, inplace = True)

    return master



#
# df = pd.read_csv('SPY_master_df')
# ind = pd.DatetimeIndex(df['for index'])
# df.set_index(ind, inplace = True)
# df.drop('for index', axis = 1, inplace = True)
# target = df['target']
# print(df.columns, df.index)
# plt.hist(target, bins=  100)
# plt.show()

'''

THIS PART WAS JUST TESTING INDEX AND GRAPHING AND SHIT
FROM HERE JUST BUILD THE ACTUAL MODEL I THINK
'''

def create_learning_df(ticker):
    if type(ticker) != str:
        raise TypeError('TICKER NAME MUST BE STRING')

    if f'{ticker}_master_df' not in os.listdir():
        master = make_new_cols(ticker)
        new_save = True

    else:
        print('df already exists in directory')
        master = pd.read_csv(f'{ticker}_master_df')
        new_save = False
        return master

    current_cols = master.columns.tolist()
    actual_cols = ['marketOpen', 'marketHigh', 'marketLow', 'marketClose', 'marketVolume',
       'marketNumberOfTrades', 'marketNotional', 'marketChangeOverTime', 'for index',
       '9SMA', '9EMA', '20SMA', '20EMA', '50SMA', '50EMA', '100SMA', '100EMA',
       '200SMA', '200EMA', 'MACD', 'Stochastic', 'RSI', 'ich0', 'ich1', 'ich2',
       'ich3', 'ich4', 'target']

    if current_cols != actual_cols:
        print(current_cols,actual_cols)
        missing = [element for element in actual_cols if element not in current_cols]
        raise ValueError('Column(s)', missing,'is missing.')

    if master.isnull().values.any():
        nans = np.where(np.asanyarray(np.isnan(master)))
        # row_cords = nans[0]
        # col_cords = nans[1]
        # pairs = []
        # 
        # for i,item in enumerate(row_cords):
        #     pairs.append(item, col_cords[i])

        raise ValueError('NaN values at', nans)
    
    print('Data is good')
    if new_save:
        master.to_csv(f'{ticker}_master_df')
    
    return master










def prepare_data(symbol):

    datelist = os.listdir(f'{symbol}_intraday_data')

    init_price = True
    dates = []
    master_list = []
    times = []

    if f'{symbol}_vis_data.csv' not in os.listdir():

        for day in datelist:

            current_data = pd.read_csv(f'{symbol}_intraday_data/{day}')
            try:
                current_data = current_data[['label', 'marketClose']]
                current_data.dropna(axis=0, inplace=True)
            except:
                current_data = []



            if not len(current_data):

                if not len(master_list):
                    continue

                master_list += [master_list[-1]] * 30
                dates += [day[:-3]] * 30
                times += [midnight] * 30

            else:
                closes = current_data['marketClose'].tolist()
                curr_times = current_data['label'].tolist()

                if not len(master_list):
                    master_list += (closes + [closes[-1]] * 30)
                    dates += [day] * (len(curr_times) + 30)
                    times += (curr_times + [midnight] * 30)

                else:
                    master_list += (closes + [closes[-1]] * 30)
                    dates += [day] * (len(curr_times) + 30)
                    times += (curr_times + [midnight] * 30)

        master_df = pd.DataFrame({'dates' : dates,
                                  'times' : times,
                                  'closes' : master_list})
        master_df.to_csv(f'{symbol}_vis_data.csv')

    else:
        print(f'Already have data for {symbol}')
        master_df = pd.read_csv(f'{symbol}_vis_data.csv')

    return master_df

def line_data(ticker):
    data = prepare_data(ticker)
    closes = data['closes'].tolist()
    times = data['times'].tolist()
    ind = data.index.tolist()

    times = [str(c) for c in times]

    return [closes,times,ind]
