### PARAMETRIC MODELS: ARIMA -- BASELINE + ADDITIONAL EXPLANATORY VARIABLES + CLUSTERING ###
 
import statsmodels.api as sm
 
# INITIALIZE TRANSFORMED DATA & SELECTED SERIES/CHANNELS
df = data.copy()
list_channel = ['ESPN', 'CNN', 'DISNEY CHANNEL']
list_target = ['18+', 'F18-34']
list_day_part = ['Morning', 'Daytime', 'Early Fringe', 'Prime Time', 'Late Fringe']
 
# FUNCTION TO CREATE LAGS USED FOR PERFORMING MODELS
def create_lag(data, lags=[]):
  output = pd.DataFrame({'value': data.values.ravel()})
  for i in lags:
    output['lag_{}'.format(i)] = output['value'].shift(i)
  output = output.dropna()
  return output
 
# PERFORM PARAMETRIC MODELS OVER ALL SELECTED SERIES
list_preds_parametric = {}
list_resid_parametric = {}
for channel in tqdm(list_channel):
  list_preds_channel = {}
  list_resid_channel = {}
  for target in list_target:
    random_seed = 1234
 
    # INITIALIZE DATAFRAME X
    df_pivot = df.pivot(index=['date', 'daypart', 'weekday', 'month', 'quarter', 'year', 'holiday', 'Events'], columns='target', values=channel).reset_index()
    df_pivot = df_pivot.merge(data[data['target'] == target][['date', 'daypart', 'fourier_'+channel]], how='left', on=['date', 'daypart'])
    df_pivot['daypart'] = df_pivot['daypart'].astype('category')
    df_pivot['daypart'] = df_pivot['daypart'].cat.reorder_categories(['Morning', 'Daytime', 'Early Fringe', 'Prime Time', 'Late Fringe'])
    df_pivot['date'] = df_pivot['date'].astype('datetime64[ns]')
    df_pivot = df_pivot.sort_values(['date', 'daypart']).reset_index(drop=True)
 
    # PERFORM FEATURE SELECTION FOR ALL DAY PARTS
    list_RF_selected_lags = {}
    for i in range(len(list_day_part)):
      # ADD LAGGED VARIABLES AS PREDICTORS: 1-14 DAYS, 1-52 WEEKS, 1-3 YEARS
      X = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
      X = pd.DataFrame(X)
      lag_range = [i+1 for i in range(2*5*7)] + [i for i in range(105, 35*53, 35)]
      X = create_lag(X.iloc[:,:1], lag_range)
 
      # SPLIT INTO TRAINING & TEST SET
      X_train = X.iloc[:len(X)-4+i, 1:]*1000
      y_train = X.iloc[:len(X)-4+i, :1]*1000  
 
      # FEATURE SELECTION ON LAGGED REGRESSORS USING RF BASED ON PERMUTATION PERFORMANCES
      RF = LGBMRegressor(boosting_type='rf', bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, random_state=random_seed).fit(X_train, y_train)
      result = permutation_importance(RF, X_train, y_train, n_repeats=10, random_state=random_seed)
 
      # USE SELECTED LAGS OBTAINED FROM FEATURE SELECTION
      lags_selected = 1 + np.sort(result.importances_mean.argsort()[::-1][:10])
      list_RF_selected_lags[list_day_part[i]] = lags_selected
 
    # SPLIT INTO TRAINING & TEST SET
    X_train = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
    X_test = df_pivot[df_pivot['date'] >= '2018-09-28'][target].reset_index(drop=True)*1000
    X_exog = pd.concat([df_pivot[['date', 'daypart', 'fourier_'+channel]], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)
 
    # ADD CLUSTERS
    X_clusters = data_clusters[(data_clusters['channel'] == channel) & (data_clusters['target'] == target)].drop(columns=['channel', 'target', 'value'])
    X_clusters['date'] = X_clusters['date'].astype('datetime64[ns]')
    X_exog = X_exog.merge(X_clusters, on=['date', 'daypart'], how='left')

    # OBTAIN OPTIMAL ARIMA & SARIMA MODEL USING GRIDSEARCH
    list_ARIMA_optimal = {}
    df_resid = pd.DataFrame()
    for i in range(len(list_day_part)):
      X_train_lagged = create_lag(X_train[:len(X_train)-4+i], lags=list_RF_selected_lags[list_day_part[i]])
      X_train_lagged = X_train_lagged.merge(df_pivot[['date', 'daypart']].reset_index(drop=True), left_index=True, right_index=True).set_index('date')
 
      # ADD EXTERNAL VARIABLES
      X_train_lagged = X_train_lagged.merge(X_exog, on=['date', 'daypart']).set_index('date')
 
      # FILTER ON DAY PARTS
      X_train_lagged = X_train_lagged[X_train_lagged['daypart'] == list_day_part[i]].drop(columns='daypart').dropna(axis=1, how='all').fillna(0)
      X_train_lagged_exog = X_train_lagged.iloc[:, 1:]
 
      # OBTAIN BEST ARIMA MODEL
      ARIMA = auto_arima(X_train_lagged.iloc[:, :1], max_p=10, max_q=10, stationary=True, seasonal=False, exog=X_train_lagged_exog, n_jobs=-1, random_state=random_seed).fit(X_train_lagged.iloc[:, :1], exogenous=X_train_lagged_exog)
      ARIMA_resid = pd.Series(ARIMA.resid())
      temp = pd.DataFrame({'Date': X_train_lagged.index, 'Day part': list_day_part[i], 'ARIMA': ARIMA_resid})
      df_resid = df_resid.append(temp, ignore_index=True)
 
      ARIMA_sm = sm.tsa.SARIMAX(X_train_lagged.iloc[:, :1], order=ARIMA.order, exog=X_train_lagged_exog, trend='c', random_seed=random_seed).fit()
      list_ARIMA_optimal[list_day_part[i]] = [ARIMA, ARIMA_sm]
 
    # SPLIT INTO TRAINING & TEST SET
    X_train = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
    X_test = df_pivot[df_pivot['date'] >= '2018-09-28'][target].reset_index(drop=True)*1000
    X_exog = pd.concat([df_pivot[['date', 'daypart', 'fourier_'+channel]], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)
 
    # ADD CLUSTERS
    X_clusters = data_clusters[(data_clusters['channel'] == channel) & (data_clusters['target'] == target)].drop(columns=['channel', 'target', 'value'])
    X_clusters['date'] = X_clusters['date'].astype('datetime64[ns]')
    X_exog = X_exog.merge(X_clusters, on=['date', 'daypart'], how='left')

    # APPLY RECURSIVE MULTI-STEP AHEAD FORWARD PREDICTION FOR ALL DAY PARTS
    ARIMA_preds = []
    for i in range(int(len(X_test)/5)):
      for day_part in list_day_part:
        X_train_lagged = create_lag(X_train, lags=list_RF_selected_lags[day_part])
        X_train_lagged = X_train_lagged.merge(df_pivot[['date', 'daypart']].reset_index(drop=True), left_index=True, right_index=True).set_index('date')
 
        X_test_lagged = X_train.sort_index(ascending=False).reset_index(drop=True)[[i-1 for i in list_RF_selected_lags[day_part]]]
        X_test_lagged = pd.DataFrame(X_test_lagged).T
        X_test_lagged.columns = X_train_lagged.columns[1:-1]
        if (day_part == 'Morning'):
          X_test_lagged['date'] = X_train_lagged.index[-1] + dt.timedelta(days=1)
        else:
          X_test_lagged['date'] = X_train_lagged.index[-1]
        X_test_lagged['daypart'] = day_part
 
        # ADD EXTERNAL VARIABLES TO X_TRAIN & X_TEST
        X_train_lagged = X_train_lagged.merge(X_exog, on=['date', 'daypart']).set_index('date')
        X_test_lagged = X_test_lagged.merge(X_exog, on=['date', 'daypart']).set_index('date')
 
        # FILTER ON DAY PARTS
        X_train_lagged = X_train_lagged[X_train_lagged['daypart'] == day_part].drop(columns='daypart').dropna(axis=1, how='all').fillna(0)
        X_test_lagged = X_test_lagged.drop(columns='daypart').dropna(axis=1, how='all').fillna(0)
        X_train_lagged_exog = X_train_lagged.iloc[:, 1:]    
 
        # ARIMA MODEL MULTI-STEP AHEAD
        ARIMA_optimal = sm.tsa.SARIMAX(X_train_lagged.iloc[:, :1], order=list_ARIMA_optimal[day_part][0].order, exog=X_train_lagged_exog, trend='c', random_seed=random_seed)
        ARIMA_optimal = ARIMA_optimal.filter(list_ARIMA_optimal[day_part][1].params)
        ARIMA_temp_preds = ARIMA_optimal.forecast(steps=1, exog=X_test_lagged).values[0]
 
        # ADD PREDICTION AS NEXT INPUT
        ARIMA_preds.append(ARIMA_temp_preds)
        X_train = X_train.append(pd.Series(ARIMA_temp_preds), ignore_index=True)
    
    # SPLIT INTO TRAINING & TEST SET
    X_train = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
    X_test = df_pivot[df_pivot['date'] >= '2018-09-28'][target].reset_index(drop=True)*1000
    X_exog = pd.concat([df_pivot[['date', 'daypart', 'fourier_'+channel]], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)
 
    # ADD CLUSTERS
    X_clusters = data_clusters[(data_clusters['channel'] == channel) & (data_clusters['target'] == target)].drop(columns=['channel', 'target', 'value'])
    X_clusters['date'] = X_clusters['date'].astype('datetime64[ns]')
    X_exog = X_exog.merge(X_clusters, on=['date', 'daypart'], how='left')

    # APPLY RECURSIVE ONE-STEP AHEAD FORWARD PREDICTION FOR ALL DAY PARTS
    ARIMA_preds_1_step = []
    for i in range(int(len(X_test)/5)):
      count = 0
      for day_part in list_day_part:
        X_train_lagged = create_lag(X_train, lags=list_RF_selected_lags[day_part])
        X_train_lagged = X_train_lagged.merge(df_pivot[['date', 'daypart']].reset_index(drop=True), left_index=True, right_index=True).set_index('date')
 
        X_test_lagged = X_train.sort_index(ascending=False).reset_index(drop=True)[[i-1 for i in list_RF_selected_lags[day_part]]]
        X_test_lagged = pd.DataFrame(X_test_lagged).T
        X_test_lagged.columns = X_train_lagged.columns[1:-1]
        if (day_part == 'Morning'):
          X_test_lagged['date'] = X_train_lagged.index[-1] + dt.timedelta(days=1)
        else:
          X_test_lagged['date'] = X_train_lagged.index[-1]
        X_test_lagged['daypart'] = day_part
 
        # ADD EXTERNAL VARIABLES TO X_TRAIN & X_TEST
        X_train_lagged = X_train_lagged.merge(X_exog, on=['date', 'daypart']).set_index('date')
        X_test_lagged = X_test_lagged.merge(X_exog, on=['date', 'daypart']).set_index('date')
 
        # FILTER ON DAY PARTS
        X_train_lagged = X_train_lagged[X_train_lagged['daypart'] == day_part].drop(columns='daypart').dropna(axis=1, how='all').fillna(0)
        X_test_lagged = X_test_lagged.drop(columns='daypart').dropna(axis=1, how='all').fillna(0)
        X_train_lagged_exog = X_train_lagged.iloc[:, 1:]      
 
        # ARIMA MODEL 1-STEP AHEAD
        ARIMA_optimal_1_step = sm.tsa.SARIMAX(X_train_lagged.iloc[:, :1], order=list_ARIMA_optimal[day_part][0].order, exog=X_train_lagged_exog, trend='c', random_seed=random_seed)
        ARIMA_optimal_1_step = ARIMA_optimal_1_step.filter(list_ARIMA_optimal[day_part][1].params)
        ARIMA_temp_preds_1_step = ARIMA_optimal_1_step.forecast(steps=1, exog=X_test_lagged).values[0]
 
        # ADD TRUE VALUE AS NEXT INPUT
        ARIMA_preds_1_step.append(ARIMA_temp_preds_1_step)
        X_train = X_train.append(pd.Series(X_test[i*5+count]), ignore_index=True)
        count = count + 1
 
    # STORE RESULTS ARIMA
    df_results = pd.DataFrame({'Date': df_pivot[df_pivot['date'] >= '2018-09-28']['date'],
                               'Day part': ['Morning', 'Daytime', 'Early Fringe', 'Prime Time', 'Late Fringe']*int(len(X_test)/5), 
                               'ARIMA_preds': ARIMA_preds, 'ARIMA_preds_1_step': ARIMA_preds_1_step})
    list_preds_channel[target] = df_results
 
    # STORE RESIDUALS OF ARIMA
    list_resid_channel[target] = df_resid
 
  # STORE DICTIONARY OF CHANNEL
  list_preds_parametric[channel] = list_preds_channel
  list_resid_parametric[channel] = list_resid_channel
 
list_results_ARIMA_cluster = list_preds_parametric
list_resids_ARIMA_cluster = list_resid_parametric
