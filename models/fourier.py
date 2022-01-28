import pylab as pl
from numpy import fft
    
def fourierExtrapolation(x, n_predict, n_param=5):
    n = x.size
    n_harm = 100                # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    h=np.sort(x_freqdom)[-n_param]
    x_freqdom=[ x_freqdom[i] if np.absolute(x_freqdom[i])>=h else 0 for i in range(len(x_freqdom)) ]

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t
  
### ARIMA RESIDUAL MODIFIED FOURIER TERMS ###

import statsmodels.api as sm

# INITIALIZE TRANSFORMED DATA & SELECTED SERIES/CHANNELS
df = data.copy()
list_channel = ['CBS', 'NBC', 'ABC', 'FOX', 'MSNBC', 'ESPN', 'CNN', 'UNI', 'DISNEY CHANNEL', 'MTV']
list_target = ['18+', 'F18-34']
list_day_part = ['Morning', 'Daytime', 'Early Fringe', 'Prime Time', 'Late Fringe']

end_date = '2019-09-28'

list_fourier = {}

# PERFORM ARIMA MODEL OVER SELECTED CHANNELS, TARGET, AND DAY PARTS
list_preds_parametric = {}
list_resid_parametric = {}
for channel in tqdm(list_channel):
  list_preds_channel = {}
  list_resid_channel = {}

  list_fourier_channel = {}
  for target in list_target:
    list_preds_target = {}
    list_resid_target = {}

    list_fourier_target = {}
    for day_part in list_day_part: 
      random_seed = 1234

      # INITIALIZE DATAFRAME X
      df_pivot = df.pivot(index=['date', 'daypart', 'weekday', 'month', 'quarter', 'year', 'holiday', 'Events'], columns='target', values=channel).reset_index()
      df_pivot = df_pivot.merge(data[data['target'] == target][['date', 'daypart']], how='left', on=['date', 'daypart'])
      df_pivot = df_pivot[df_pivot['daypart'] == day_part]
      df_pivot['date'] = df_pivot['date'].astype('datetime64[ns]')

      # SPLIT INTO TRAINING & TEST SET
      X_train = df_pivot[df_pivot['date'] < end_date][['date', 'daypart', target]].set_index('date')
      X_train[target] = X_train[target]*1000
      X_test = df_pivot[df_pivot['date'] >= end_date][['date', 'daypart', target]].set_index('date')
      X_test[target] = X_test[target]*1000
      X_exog = pd.concat([df_pivot[['date', 'daypart']], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)

      # ADD EXTERNAL VARIABLES TO X_TRAIN & X_TEST
      X_exog_train = X_train.merge(X_exog, on=['date', 'daypart']).set_index('date').drop(columns=['daypart', target])
      X_exog_test = X_test.merge(X_exog, on=['date', 'daypart']).set_index('date').drop(columns=['daypart', target])

      # FILTER ON DAY PARTS
      X_train = X_train.drop(columns='daypart')
      X_test = X_test.drop(columns='daypart')

      # ARIMA MODEL MULTI-STEP AHEAD
      ARIMA = auto_arima(X_train, exog=X_exog_train, max_p=10, max_q=10, seasonal=False, n_jobs=-1, random_state=random_seed).fit(X_train, exogenous=X_exog_train)
      ARIMA_preds = ARIMA.predict(n_periods=len(X_test), exogenous=X_exog_test)
      ARIMA_resid = pd.Series(ARIMA.resid())

      # OBTAIN FOURIER TERMS
      Fourier_terms = fourierExtrapolation(ARIMA_resid, n_predict=len(X_test), n_param=50)
      
      # ADD EXTERNAL VARIABLES
      X_exog = pd.concat([df_pivot[['date']], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)
      X_exog['fourier'] = Fourier_terms

      list_fourier_target[day_part] = X_exog

    list_preds_channel[target] = list_preds_target
    list_resid_channel[target] = list_resid_target
    list_fourier_channel[target] = list_fourier_target

  list_preds_parametric[channel] = list_preds_channel
  list_resid_parametric[channel] = list_resid_channel
  list_fourier[channel] = list_fourier_channel

list_results_ARIMA_external = list_preds_parametric
list_resids_ARIMA_external = list_resid_parametric
list_fourier
