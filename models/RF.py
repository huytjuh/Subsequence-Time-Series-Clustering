### NON-PARAMETRIC MODELS: RF --- CLUSTERING ###

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

# PERFORM NON-PARAMETRIC MODELS OVER ALL SELECTED SERIES
tcv = TimeSeriesSplit(n_splits=4)
list_preds_non_parametric = {}
for channel in tqdm(list_channel):
  list_preds_channel = {}
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
      lags_selected = 1 + np.sort(result.importances_mean.argsort()[::-1][:20])
      list_RF_selected_lags[list_day_part[i]] = lags_selected

    # SPLIT INTO TRAINING & TEST SET
    X_train = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
    X_test = df_pivot[df_pivot['date'] >= '2018-09-28'][target].reset_index(drop=True)*1000
    X_exog = pd.concat([df_pivot[['date', 'daypart', 'fourier_'+channel]], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)

    # ADD CLUSTERS
    X_clusters = data_clusters[(data_clusters['channel'] == channel) & (data_clusters['target'] == target)].drop(columns=['channel', 'target', 'value'])
    X_clusters['date'] = X_clusters['date'].astype('datetime64[ns]')
    X_exog = X_exog.merge(X_clusters, on=['date', 'daypart'], how='left')

    # OBTAIN OPTIMAL RF MODEL USING GRIDSEARCH
    list_RF_optimal = {}
    for i in range(len(list_day_part)):
      X_train_lagged = create_lag(X_train[:len(X_train)-4+i], lags=list_RF_selected_lags[list_day_part[i]])
      X_train_lagged = X_train_lagged.merge(df_pivot[df_pivot['date'] < '2018-09-28'][['date', 'daypart']].reset_index(drop=True), left_index=True, right_index=True, how='right')
      X_train_lagged = X_train_lagged[:len(X_train_lagged)-4+i]

      X_impute = X_train_lagged.shift(-364*5)[:len(X_train_lagged[X_train_lagged.isna().any(axis=1)])].copy()
      X_impute['date'] = X_impute['date'] - dt.timedelta(days=364)
      X_train_lagged[X_train_lagged.isna().any(axis=1)] = X_impute
      X_train_lagged = X_train_lagged.set_index('date')

      # ADD EXTERNAL VARIABLES
      X_train_lagged = X_train_lagged.merge(X_exog, on=['date', 'daypart']).set_index('date')

      # FILTER ON DAY PARTS
      X_train_lagged = X_train_lagged[X_train_lagged['daypart'] == list_day_part[i]].drop(columns='daypart').dropna(axis=1, how='all').fillna(0)

      # OBTAIN BEST HYPERPARAMETERS OF RANDOMFOREST USING BAYESIAN OPTIMIZATION
      def lgb_eval(n_estimators, num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'boosting_type':'rf', 'bagging_freq':1, 'learning_rate':0.001, 'early_stopping_round':100, 'metric':'rmse'}
        params['num_iterations'] =  int(n_estimators)
        params['num_leaves'] = int(num_leaves)
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(max_depth)
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight

        train_data = lgb.Dataset(data=X_train_lagged.iloc[:, 1:], label=X_train_lagged.iloc[:, :1], free_raw_data=False)
        cv_result = lgb.cv(params, train_data, folds=tcv, seed=random_seed, verbose_eval=False, metrics=['rmse'])
        return -min(cv_result['rmse-mean'])

      lgbBO = BayesianOptimization(lgb_eval, {'n_estimators': (250, 1000),
                                              'num_leaves': (20, 40),
                                              'feature_fraction': (0.5, 0.9),
                                              'bagging_fraction': (0.5, 0.9),
                                              'max_depth': (5, 10),
                                              'lambda_l1': (0, 5),
                                              'lambda_l2': (0, 3),
                                              'min_split_gain': (0.001, 0.1),
                                              'min_child_weight': (5, 50)}, random_state=random_seed, verbose=0)
      lgbBO.maximize(init_points=50, n_iter=50)
      RF_optimal = LGBMRegressor(boosting_type='rf', bagging_freq=1, **convert_to_int(lgbBO.max['params']), random_state=random_seed).fit(X_train_lagged.iloc[:, 1:], X_train_lagged.iloc[:, :1])
      list_RF_optimal[list_day_part[i]] = RF_optimal

    # SPLIT INTO TRAINING & TEST SET
    X_train = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
    X_test = df_pivot[df_pivot['date'] >= '2018-09-28'][target].reset_index(drop=True)*1000
    X_exog = pd.concat([df_pivot[['date', 'daypart', 'fourier_'+channel]], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)

    # ADD CLUSTERS
    X_clusters = data_clusters[(data_clusters['channel'] == channel) & (data_clusters['target'] == target)].drop(columns=['channel', 'target', 'value'])
    X_clusters['date'] = X_clusters['date'].astype('datetime64[ns]')
    X_exog = X_exog.merge(X_clusters, on=['date', 'daypart'], how='left')

    # APPLY RECURSIVE MULTI-STEP AHEAD FORWARD PREDICTION FOR ALL DAY PARTS
    RF_preds = []
    for i in range(int(len(X_test)/5)):
      count = 0
      for day_part in list_day_part:
        X_train_lagged = create_lag(X_train, lags=list_RF_selected_lags[day_part])
        X_train_lagged = X_train_lagged.merge(df_pivot[df_pivot['date'] < str(dt.date(2018, 9, 28) + dt.timedelta(days=i))][['date', 'daypart']].reset_index(drop=True), left_index=True, right_index=True, how='right')
        X_train_lagged = X_train_lagged[:len(X_train_lagged)-4+count]

        X_impute = X_train_lagged.shift(-364*5)[:len(X_train_lagged[X_train_lagged.isna().any(axis=1)])].copy()
        X_impute['date'] = X_impute['date'] - dt.timedelta(days=364)
        X_train_lagged[X_train_lagged.isna().any(axis=1)] = X_impute
        X_train_lagged = X_train_lagged.set_index('date')

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

        RF_optimal = list_RF_optimal[day_part]#.fit(X_train_lagged.iloc[:, 1:], X_train_lagged.iloc[:, :1])
        RF_temp_preds = RF_optimal.predict(X_test_lagged)[0]

        # ADD PREDICTION AS NEXT INPUT
        RF_preds.append(RF_temp_preds)
        X_train = X_train.append(pd.Series(RF_temp_preds), ignore_index=True)
        count = count + 1

    # SPLIT INTO TRAINING & TEST SET
    X_train = df_pivot[df_pivot['date'] < '2018-09-28'][target].reset_index(drop=True)*1000
    X_test = df_pivot[df_pivot['date'] >= '2018-09-28'][target].reset_index(drop=True)*1000
    X_exog = pd.concat([df_pivot[['date', 'daypart', 'fourier_'+channel]], pd.get_dummies(df_pivot[['weekday', 'holiday', 'Events']], drop_first=True)], axis=1)

    # ADD CLUSTERS
    X_clusters = data_clusters[(data_clusters['channel'] == channel) & (data_clusters['target'] == target)].drop(columns=['channel', 'target', 'value'])
    X_clusters['date'] = X_clusters['date'].astype('datetime64[ns]')
    X_exog = X_exog.merge(X_clusters, on=['date', 'daypart'], how='left')
    
    # APPLY RECURSIVE ONE-STEP AHEAD FORWARD PREDICTION FOR ALL DAY PARTS
    RF_preds_1_step = []
    for i in range(int(len(X_test)/5)):
      count = 0
      for day_part in list_day_part:
        X_train_lagged = create_lag(X_train, lags=list_RF_selected_lags[day_part])
        X_train_lagged = X_train_lagged.merge(df_pivot[df_pivot['date'] < str(dt.date(2018, 9, 28) + dt.timedelta(days=i))][['date', 'daypart']].reset_index(drop=True), left_index=True, right_index=True, how='right')
        X_train_lagged = X_train_lagged[:len(X_train_lagged)-4+count]

        X_impute = X_train_lagged.shift(-364*5)[:len(X_train_lagged[X_train_lagged.isna().any(axis=1)])].copy()
        X_impute['date'] = X_impute['date'] - dt.timedelta(days=364)
        X_train_lagged[X_train_lagged.isna().any(axis=1)] = X_impute
        X_train_lagged = X_train_lagged.set_index('date')

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

        RF_optimal_1_step = list_RF_optimal[day_part]#.fit(X_train_lagged.iloc[:, 1:], X_train_lagged.iloc[:, :1])
        RF_temp_preds_1_step = RF_optimal_1_step.predict(X_test_lagged)[0]

        # ADD PREDICTION AS NEXT INPUT
        RF_preds_1_step.append(RF_temp_preds_1_step)
        X_train = X_train.append(pd.Series(X_test[i*5+count]), ignore_index=True)
        count = count + 1
    
    # STORE RESULTS RF
    df_results = pd.DataFrame({'Date': df_pivot[df_pivot['date'] >= '2018-09-28']['date'],
                               'Day part': ['Morning', 'Daytime', 'Early Fringe', 'Prime Time', 'Late Fringe']*int(len(X_test)/5), 
                               'RF_preds': RF_preds, 'RF_preds_1_step': RF_preds_1_step})
    list_preds_channel[target] = df_results

  # STORE DICTIONARY OF CHANNEL
  list_preds_non_parametric[channel] = list_preds_channel

list_results_RF_cluster = list_preds_non_parametric
