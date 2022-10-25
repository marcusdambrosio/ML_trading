import pandas as pd
import datetime as dt


df1 = pd.read_csv('SPY_intraday_data/2020-01-06.csv')
df2 = pd.read_csv('SPY_intraday_data/2020-01-07.csv')
df = df1.append(df2)


ind = pd.DatetimeIndex(df['date'])


dates = df.date
times = df.label
dtime = dates+' ' +times

new = []

# for item in dtime:
#     cur = dt.datetime.strptime(item, '%Y-%m-%d %H:%M')

print(type(df.iloc[:,0]))
df = df.set_index(pd.DatetimeIndex(df.iloc[:,0]))
print(df)

print(type(df.index))