import pandas as pd

def true_range(df): 
  datesIndex = df.index  
  ind = range(0,len(df))
  indexlist = list(ind)
  df.index = indexlist

  for index, row in df.iterrows():
    if index != 0:
      tr1 = row["High"] - row["Low"]
      tr2 = abs(row["High"] - df.iloc[index-1]["Close"])
      tr3 = abs(row["Low"] - df.iloc[index-1]["Close"])

      true_range = max(tr1, tr2, tr3)
      df.set_value(index,"True Range", true_range)
  df.index = datesIndex    
  return df

def transform(df,window=30):
    '''Return dataframe with Techinal Variables 
    '''
    # Return
    df['return'] = (df['Adj Close'] - df['Adj Close'].shift(periods=1))/df['Adj Close'].shift(periods=1)

    # Simple Moving Average for window = 6
    df['sma6'] = df['Adj Close'].rolling(window = 6).mean()

    # Simple Moving Average for window = 12
    df['sma12'] = df['Adj Close'].rolling(window = 12).mean()
    
    # Exponential Moving Average for window = 12
    k = 2.0/(12+1)
    df['ema12'] = df['Adj Close']
    df['ema12'] = (df['Adj Close'] - df['ema12'].shift())*k + df['ema12'].shift()

    # Exponential Moving Average for window = 26
    k = 2.0/(26+1)
    df['ema26'] = df['Adj Close']
    df['ema26'] = (df['Adj Close'] - df['ema26'].shift())*k + df['ema26'].shift()
    
    # dif = ema12 - ema26
    df['dif'] = df['ema12'] - df['ema26']

    # MACD for window 
    df['macd'] = df['Adj Close'].rolling(window = window).mean() - df['Adj Close'].rolling(window = 21).mean()

    # Stochastic %K
    df['%k'] = (df['Adj Close'] - df['Low'].rolling(window = window).min())/(df['High'].rolling(window = window).max() - df['Low'].rolling(window = window).min())*100
    
    # Stochastic %D
    df['%d'] = df['%k'].rolling(window = 3).mean()
    
    # ROC for window of 3 days
    df['roc'] = (df['Adj Close'] * 1.0) / (df['Adj Close'].shift(periods=3)) - 1

    # True range of price Movements
    true_range(df)
    
    # 6-day Momentum 
    df['momentum'] = df['Adj Close'] - df['Adj Close'].shift(periods=6)
    
    # 12-day Momentum 
    df['momentum'] = df['Adj Close'] - df['Adj Close'].shift(periods=12)

    # Williams index
    df['r%10'] = (df['High'].rolling(10).max() - df['Adj Close'])/(df['High'].rolling(10).max() - df['Low'].rolling(10).min())*-100
    df['r%5'] = (df['High'].rolling(5).max() - df['Adj Close'])/(df['High'].rolling(5).max() - df['Low'].rolling(5).min())*-100
    
    # Stochastic Oscillator for window 
    df['osc6'] = 1.0*(df['Close']-df['Adj Close'].rolling(window=6).min())/(df['Adj Close'].rolling(window=6).max()-df['Adj Close'].rolling(window=6).min())
    df['osc12'] = 1.0*(df['Close']-df['Adj Close'].rolling(window=12).min())/(df['Adj Close'].rolling(window=12).max()-df['Adj Close'].rolling(window=12).min())
    #df['slow_stochastic_oscillator'] =  df['fast_stochastic_oscillator'].rolling(window = window).mean()
    #df['smoothed_slow_stochastic_oscillator'] = df['slow_stochastic_oscillator'].rolling(window = window).mean()

    # Relative strength index
    df['shift']=df['Adj Close'].shift()
    df['gain']=df['Adj Close']-df['shift']
    df.loc[df['gain'] < 0,'gain']=0
    df['loss']=df['Adj Close']-df['shift']
    df.loc[df['loss'] >= 0,'loss']=0
    df['rs6']=(df['gain'].rolling(window=14).sum())/(df['loss'].rolling(window=6).sum()*-1.0)
    df['rsi6'] = 100 - (100.0/(1+df['rs6']))
    df['rs12']=(df['gain'].rolling(window=14).sum())/(df['loss'].rolling(window=12).sum()*-1.0)
    df['rsi12'] = 100 - (100.0/(1+df['rs12']))
    df.drop(['shift','gain','loss','rs6','rs12'],axis=1,inplace=True)
    
    # Phycholoigical Line
    
    # On Balance Volume Indicator for window size of window
    df['daily_returns'] = ((df['Adj Close'])/(1.0*df['Adj Close'].shift())-1) 
    df['positive_signed_volume'] = df[df['daily_returns'] >= 0]['Volume']
    df['positive_signed_volume'].replace(float('nan'),0,inplace=True)
    df['negative_signed_volume'] = df[df['daily_returns'] < 0]['Volume']*(-1.0)
    df['negative_signed_volume'].replace(float('nan'),0,inplace=True)
    df['signed_volume'] = df['positive_signed_volume'] + df['negative_signed_volume']
    df['obv'] = df['signed_volume'].rolling(window=window).sum()
    df=df.drop(['positive_signed_volume','negative_signed_volume','signed_volume','daily_returns'],axis = 1)
    
    # Boll line
    # Adding Middle Bollinger Band
    df['mb'] = df['sma6']    # Same as smiple moving average
    # Adding upper Bollinger Band
    df['up'] = 2*df['Adj Close'].rolling(window = window).std() + df['sma6']
    # Adding lower Bollinger Band
    df['dw'] = df['sma6'] - (2*df['Adj Close'].rolling(window = window).std())
    
    # K(t)-K(t-1)
    df['k(t)'] = df['%k']
    df['k(t-1)'] = (df['Adj Close'].shift(periods=1) - df['Low'].rolling(window = window).min())/(df['High'].rolling(window = window).max() - df['Low'].rolling(window = window).min())*100
    df['k(t)-k(t-1)'] = df['k(t)'] - df['k(t-1)']
    
    # D(t)-D(t-1)
    df['d(t)'] = df['k(t)'].rolling(window = 3).mean()
    df['d(t-1)'] = df['k(t-1)'].rolling(window = 3).mean()
    df['d(t)-d(t-1)'] = df['d(t)'] - df['d(t-1)']
    df=df.drop(['k(t)','k(t-1)','d(t)','d(t-1)'],axis = 1)
    
    # (MA6(t)-MA6(t-1))/MA6(t-1)
    df['ma6(t)'] = df['sma6']
    df['ma6(t-1)'] = df['Adj Close'].rolling(window = 6).mean().shift(1)
    df['(ma6(t)-ma6(t-1))/ma6(t-1)'] = (df['ma6(t)'] - df['ma6(t-1)'])/df['ma6(t-1)']
    
    # (MA12(t)-MA12(t-1))/MA12(t-1)
    df['ma12(t)'] = df['sma12']
    df['ma12(t-1)'] = df['Adj Close'].rolling(window = 6).mean().shift(1)
    df['(ma12(t)-ma12(t-1))/ma12(t-1)'] = (df['ma12(t)'] - df['ma12(t-1)'])/df['ma12(t-1)']
    
    # (MA6(t)-MA12(t-1))/MA12(t-1)
    df['(ma6(t)-ma12(t-1))/ma12(t-1)'] = (df['ma6(t)'] - df['ma12(t-1)'])/df['ma12(t-1)']
    
    # (x(t)-MA12(t))/MA12(t)
    df['(x(t)-MA12(t))/MA12(t)'] = (df['Adj Close']-df['ma12(t)'])/df['ma12(t)']
    df=df.drop(['ma6(t)','ma6(t-1)','ma12(t)','ma12(t-1)'],axis = 1)
    
    # (x(t)-xl(t))/(xo(t)-xl(t))
    df['(x(t)-xl(t))/(xo(t)-xl(t))'] = (df['Adj Close'] - df['Low'])/(df['Open'] - df['Low'])

    return df
