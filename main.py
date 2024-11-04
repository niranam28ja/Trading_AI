from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import datetime
from matplotlib import pyplot as plt
import statsmodels.api as sm    
import pandas as pd 
import numpy as np
import yfinance as yf
import pandas_ta
import warnings

warnings.filterwarnings("ignore")
sp500  = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
sp500['sysbol'] = sp500['Symbol'].str.replace('.','-')
symbols_list = sp500['Symbol'].unique().tolist()
end_date = "2024-01-01"
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=8)

df = yf.download(tickers = symbols_list, start = start_date, end = end_date).stack()
df.index.names = ['Date','Ticker']
df.columns = df.columns.str.lower()
df['garman_klass_vol'] = ((np.log(df['high']) - np.log(df['low']))**2)/2 - (2*np.log(2)-1) * ((np.log(df['adj close']) - np.log(df['open']))**2)
df['rsi'] =  df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(x, length=20))

def calc_bbands_low(x):
    log_x = np.log1p(x)
    bb = pandas_ta.bbands(close=log_x, length=20)
    if bb is not None:
        return bb.iloc[:,0].set_axis(x.index)
    else:
        return pd.Series([np.nan] * len(x), index=x.index)
    
def calc_bbands_mid(x):
    log_x = np.log1p(x)
    bb = pandas_ta.bbands(close=log_x, length=20)
    if bb is not None:
        return bb.iloc[:,1].set_axis(x.index)
    else:
        return pd.Series([np.nan] * len(x), index=x.index)
    
def calc_bbands_high(x):
    log_x = np.log1p(x)
    bb = pandas_ta.bbands(close=log_x, length=20)
    if bb is not None:
        return bb.iloc[:,2].set_axis(x.index)
    else:
        return pd.Series([np.nan] * len(x), index=x.index)        

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

# pandas_ta.bbands(close=df.xs('AAPL', level=1)['close'], length=20)
df['bb-low'] = df.groupby(level=0)['adj close'].apply(calc_bbands_low).reset_index(level=0,drop=True)
df['bb-mid'] = df.groupby(level=0)['adj close'].apply(calc_bbands_mid).reset_index(level=0,drop=True)
df['bb-high'] = df.groupby(level=0)['adj close'].apply(calc_bbands_high).reset_index(level=0,drop=True)
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6
# df['bb-low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20)).iloc[:,0] # it's old one 
# df['bb-mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])  # this too 
# df['bb-high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2]) # this too 
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume','volume','open','high','low','close']]
data = (pd.concat([df.unstack('Ticker')['dollar_volume'].resample('M').mean().stack('Ticker').to_frame('dollar_volume'),df.unstack()[last_cols].resample('M').last().stack('Ticker')],axis=1)).dropna()
data['dollar_volume'] = data['dollar_volume'].unstack('Ticker').rolling(window=5*12,min_periods=1).mean().stack()
data['dollar_vol_rank'] = data.groupby('Date')['dollar_volume'].rank(ascending=False)
data = data[data['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
def calculate_returns(df):
    outlier_cuteoff = 0.005
    lags = [1,2,3,6,9,12]

    for lag in lags:
        df[f'return_{lag}m'] = df['adj close'].pct_change(lag).pipe(lambda x:x.clip(lower=x.quantile(outlier_cuteoff),upper=x.quantile(1-outlier_cuteoff))).add(1).pow(1/lag).sub(1)
    
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
data