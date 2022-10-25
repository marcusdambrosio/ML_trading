import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pandas as pd
import datetime as dt
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import Stock
import os

os.environ['IEX_TOKEN'] = 'sk_129d923338c640f99167d869165e8f52'

today = dt.datetime.today()
dates = [today - dt.timedelta(days = x) for x in range(400)]


def get_data(stock):

    if os.path.exists(f'{stock}_intraday_data'):

        print('Folder already exists.')

    else:

        os.mkdir(f'{stock}_intraday_data')
        print('Folder created.')

    count = 0
    for day in dates:

        if os.path.exists(f'{stock}_intraday_data' + '/' + f'{str(day)[:10]}.csv'):
            continue

        else:
            count += 1
            days_data = get_historical_intraday(stock, day, output_format = 'pandas')
            days_data.to_csv(f'{stock}_intraday_data' + '/'+ f'{str(day)[:10]}.csv')

    print(f'{count} CSV files created.')



