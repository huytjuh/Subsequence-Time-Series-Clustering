### SELF-ORGANIZING MAPS (SOM) CLUSTERING ON TIME-HORIZON ###

# INITIALIZE TRANSFORMED DATA & SELECTED SERIES/CHANNELS
df = data.copy()
list_channel = ['CBS', 'NBC', 'ABC', 'FOX', 'MSNBC', 'ESPN' ,'CNN', 'UNI', 'DISNEY CHANNEL', 'MTV']
list_target = ['18+', 'F18-34']
list_day_part = ['Morning', 'Daytime', 'Early Fringe', 'Prime Time', 'Late Fringe']

end_date = '2019-09-28'

def split_dataframe(df, chunk_size=7): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

df_output_final = pd.DataFrame()
for channel in tqdm(list_channel):
  for target in list_target:
    for day_part in list_day_part:
      random_seed = 1234
      df_all = data.copy()
      df_all = df_all[(df_all['daypart'] == day_part) & (df_all['target'] == target)]
      df_all['Week'] = df_all['date'].apply(lambda x: x.strftime('%V'))

      df = data[(data['date'] < end_date) & (data['daypart'] == day_part) & (data['target'] == target)][['date', 'target', 'daypart', channel]].reset_index(drop=True)

      if channel == 'MTV':
        impute = df[df['MTV'] == 0]
        impute = df.loc[df['date'].isin(impute['date'] - dt.timedelta(days=364*2)), 'MTV']
        df.loc[df['MTV'] == 0, 'MTV'] = impute.values

      df['Week'] = df['date'].apply(lambda x: x.strftime('%V'))
      df['Trend'] = sm.tsa.seasonal_decompose(df[channel], model='additive', freq=7, extrapolate_trend='freq').trend
      df['Seasonal_weekly'] = sm.tsa.seasonal_decompose(df[channel], model='additive', freq=7, extrapolate_trend='freq').seasonal
      df['Fourier'] = data[(data['date'] < end_date) & (data['daypart'] == day_part) & (data['target'] == target)]['fourier_' + channel].reset_index(drop=True)
      df = df[df['date'] != '2016-02-29'].reset_index(drop=True)

      for week in df['Week'].unique():
        for df_split in split_dataframe(df.loc[df['Week'] == week, :], 7):
          if df_split.empty:
            continue
          # TREND & SEASONALITY
          Yt = df.loc[df_split.index, channel] - df.loc[df_split.index, 'Trend'] - df.loc[df_split.index, 'Seasonal_weekly']
          Zt = df.loc[df_split.index, channel] - df.loc[df_split.index, 'Seasonal_weekly']
          Xt = df.loc[df_split.index, channel] - df.loc[df_split.index, 'Trend'] 
          df.loc[df_split.index, 'Trend_aggr'] = 1 - np.var(Yt) / np.var(Zt)
          df.loc[df_split.index, 'Seasonal_aggr'] = 1 - np.var(Yt) / np.var(Xt)

          # FOURIER TERMS AS PERIODICITY
          df.loc[df_split.index, 'Fourier_aggr'] = df.loc[df_split.index, 'Fourier'].mean()

          # KURTOSIS & SKEWNESS
          df.loc[df_split.index, 'Kurtosis'] = kurtosis(df_split[channel])
          df.loc[df_split.index, 'Skewness'] = skew(df_split[channel])

          # SERIAL CORRELATION --- USING LJONBOX TEST
          res = sm.tsa.SARIMAX(df.loc[df_split.index, channel], order=(1,0,1), random_seed=random_seed).fit(disp=-1)
          df.loc[df_split.index, 'Serial_correlation'] = sm.stats.acorr_ljungbox(res.resid, boxpierce=True, lags=1)[3][0]

          # NON-LINEARITY --- USING BDS TEST
          df.loc[df_split.index, 'NON_LINEARITY'] = sm.tsa.stattools.bds(df.loc[df_split.index, channel])[0]
          
          # SELF-SIMILARITY --- USING HURST EXPONENT
          df.loc[df_split.index, 'Self_similarity'] = nolds.hurst_rs(df.loc[df_split.index, channel])

          # CHAOS --- USING LYAPUNOV EXPONENT
          df.loc[df_split.index, 'Chaos'] = nolds.lyap_r(df.loc[df_split.index, channel], emb_dim=1, min_neighbors=1, trajectory_len=2)

      df_cluster = df[-365:].reset_index(drop=True).merge(df.iloc[-365*2:-365, 8:].reset_index(drop=True), left_index=True, right_index=True, how='left', suffixes=('', '_YEAR2'))
      df_cluster = df_cluster.merge(df.iloc[-365*3:-365*2, 8:].reset_index(drop=True), left_index=True, right_index=True, how='left', suffixes=('', '_YEAR3'))
      df_cluster = df_cluster.drop(columns=['date', channel, 'Trend', 'Seasonal_weekly', 'Fourier']).groupby('Week').mean().reset_index()
      df_cluster.iloc[:, 1:] = MinMaxScaler().fit_transform(df_cluster.iloc[:, 1:])

      def SOM_evaluate(som1, som2, sigma, learning_rate):
        som_shape = (int(som1), int(som2))
        som = MiniSom(som_shape[0], som_shape[1], df_cluster.iloc[:, 1:].values.shape[1], sigma=sigma, learning_rate=learning_rate,
                    neighborhood_function='gaussian', random_seed=random_seed)
        som.train_batch(df_cluster.iloc[:, 1:].values, 10000, verbose=False)
        return -som.quantization_error(df_cluster.iloc[:, 1:].values)

      SOM_BO = BayesianOptimization(SOM_evaluate, {'sigma': (1, 0.01), 'som1': (1, 10), 'som2': (5, 15),
                                                  'learning_rate': (0.1, 0.001)},
                                                  random_state=random_seed, verbose=0)
      SOM_BO.maximize(init_points=20, n_iter=20)
      som_shape = (int(SOM_BO.max['params']['som1']), int(SOM_BO.max['params']['som2']))
      som = MiniSom(som_shape[0], som_shape[1], df_cluster.iloc[:, 1:].values.shape[1], sigma=SOM_BO.max['params']['sigma'], learning_rate=SOM_BO.max['params']['learning_rate'],
                    neighborhood_function='gaussian', random_seed=random_seed)
      winner_coordinates = np.array([som.winner(x) for x in df_cluster.iloc[:, 1:].values]).T
      cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
      df_cluster['cluster'] = cluster_index

      df = df_all.merge(df_cluster[['Week', 'cluster']], on='Week', how='left')
      df_final = pd.concat([df[['date', 'daypart', 'target', channel]], pd.get_dummies(df['cluster'], prefix='Cluster', drop_first=True)], axis=1)
      df_final = df_final.rename(columns={channel: 'value'})
      df_final.insert(0, column='channel', value=[channel]*len(df_final))

      df_output_final = df_output_final.append(df_final, ignore_index=True)

df_output_final
