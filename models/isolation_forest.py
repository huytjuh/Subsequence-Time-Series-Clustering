#%%
def getiForest(df, channel):
    #Outlier detection using isolation forest
    iForest = IsolationForest(contamination = 0.01, random_state=12345)
    iForest.fit(df[[channel]])
    
    #reset index
    df = df.reset_index()
    df['anomaly'] = pd.Series(iForest.predict(df[[channel]]))
    
    outliers = df.loc[df['anomaly'] == -1]
    outliers = outliers[['date', 'daypart', 'target','quarter','weekday', 'month', channel]]
    #outliers = outliers[['date', 'daypart', 'target','quarter','weekday']]
    outliers['Channel'] = channel
    
    return outliers 
#%%
def outliers(df, channel):
    
    df_Morning = df.loc[df['daypart'] == 'Morning']
    df_Daytime = df.loc[df['daypart'] == 'Daytime']
    df_EarlyFringe = df.loc[df['daypart'] == 'Early Fringe']
    df_Primetime = df.loc[df['daypart'] == 'Prime Time']
    df_LateFringe = df.loc[df['daypart'] == 'Late Fringe']
    
    outliers_Morning = getiForest(df_Morning, channel)
    outliers_Daytime = getiForest(df_Daytime, channel)
    outliers_EarlyFringe = getiForest(df_EarlyFringe, channel)
    outliers_Primetime = getiForest(df_Primetime, channel)
    outliers_LateFringe = getiForest(df_LateFringe, channel)
    
    return outliers_Morning, outliers_Daytime, outliers_EarlyFringe, outliers_Primetime, outliers_LateFringe
#%%
def channelOutliers(df, channel, group):
    #extract channel
    df_channel = df[['date', 'daypart', 'target', 'quarter', 'weekday', 'month',channel]]
    
    #18 
    df_channel_18 = df_channel.loc[df['target'] == '18+']
    #F18-34
    df_channel_F = df_channel.loc[df['target'] == 'F18-34']
    
        
    #outliers for 18+
    outliers_18_Morning, outliers_18_Daytime, outliers_18_EarlyFringe, outliers_18_Primetime, outliers_18_LateFringe = outliers(df_channel_18, channel)
    
    #outliers for F18-34
    outliers_F_Morning, outliers_F_Daytime, outliers_F_EarlyFringe, outliers_F_Primetime, outliers_F_LateFringe = outliers(df_channel_F, channel)
    
    if group == 1:
        return outliers_18_Morning, outliers_18_Daytime, outliers_18_EarlyFringe, outliers_18_Primetime, outliers_18_LateFringe
    else:
        return outliers_F_Morning, outliers_F_Daytime, outliers_F_EarlyFringe, outliers_F_Primetime, outliers_F_LateFringe
#%% row combine
def rowCombine(df1, df2, df3, df4, df5):
    
    combine = [df1, df2, df3, df4, df5]
    final = pd.concat(combine)
    return final

#%% get date 

def datetimeCombine(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10):
    
    combine = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
    final = pd.concat(combine)
    #final = final[['date']]
    return final

#%% 
def groupOutput(group):
    espn1,espn2,espn3,espn4,espn5 = channelOutliers(data, "ESPN", group)
    cnn1,cnn2,cnn3,cnn4,cnn5 = channelOutliers(data, "CNN", group)
    cbs1,cbs2,cbs3,cbs4,cbs5 = channelOutliers(data, "CBS", group)
    nbc1,nbc2,nbc3,nbc4,nbc5 = channelOutliers(data, "NBC", group)
    abc1,abc2,abc3,abc4,abc5 = channelOutliers(data, "ABC", group)
    fox1,fox2,fox3,fox4,fox5 = channelOutliers(data, "FOX", group)
    uni1,uni2,uni3,uni4,uni5 = channelOutliers(data, "UNI", group)
    msbnc1,msbnc2,msbnc3,msbnc4,msbnc5 = channelOutliers(data, "MSNBC", group)
    mtv1,mtv2,mtv3,mtv4,mtv5 = channelOutliers(data, "MTV", group)
    disney1,disney2,disney3,disney4,disney5 = channelOutliers(data, "DISNEY CHANNEL", group)


    all_Morning = datetimeCombine(cnn1, fox1, nbc1, msbnc1, abc1, cbs1, espn1, uni1, mtv1, disney1)
    all_Daytime = datetimeCombine(cnn2, fox2, nbc2, msbnc2, abc2, cbs2, espn2, uni2, mtv2, disney2)
    all_EarlyFringe = datetimeCombine(cnn3, fox3, nbc3, msbnc3, abc3, cbs3, espn3, uni3, mtv3, disney3)
    all_PrimeTime = datetimeCombine(cnn4, fox4, nbc4, msbnc4, abc4, cbs4, espn4, uni4, mtv4, disney4)
    all_LateFringe= datetimeCombine(cnn5, fox5, nbc5, msbnc5, abc5, cbs5, espn5, uni5, mtv5, disney5)

    all = pd.concat([all_Morning, all_Daytime, all_EarlyFringe, all_PrimeTime, all_LateFringe])
    all = all.sort_values(by=['date'])
    
    return all

#%%
def checkDuplicate(df):
    
    df['duplicated'] = df[['date']].duplicated(keep=False)
    
    df_yes = df.loc[df['duplicated'] == True]
    #df_yes = df_yes.drop_duplicates(keep='first')
    df_no =  df.loc[df['duplicated'] == False]
    
    return df_yes.sort_values(by='date'), df_no.sort_values(by='date')
#%%
#18+
all_18 = groupOutput(1)
duplocate_all_18, unique_all_18 = checkDuplicate(all_18)

#F18-34
all_F = groupOutput(2)
duplocate_all_F, unique_all_F = checkDuplicate(all_F)

#%%
all_group = pd.concat([all_18, all_F])
#duplocate_all_group, unique_all_group = checkDuplicate(all_group)

#%%
def getChannel(df, list_of_channel, i, matrix):
    
    which = i
    
    for chan in list_of_channel:
        number = len(df[chan].loc[df[chan] >= 0])
        
        matrix.loc[[which],[chan]] = number
        
    return matrix
#%%
def getWeekdays(matrix, df, list1, list_of_channel):
  
    
    for i in list1:
        
        weekday = df.loc[df['weekday'] == i ]
        
        weekday_amount = getChannel(weekday, list_of_channel, i, matrix)
        
    return matrix    
#%%
def getMonths (matrix, df, list1, list_of_channel):
    
    
    
    for i in list1:
        
        weekday = df.loc[df['month'] == i ]
        
        weekday_amount = getChannel(weekday, list_of_channel, i, matrix)
        
    return matrix 

#%%
df = all_group

#monday = df.loc[df['weekday'] == 'Monday']

#test = len(monday['CBS'].loc[monday['CBS'] > -1 ])

list_of_channel = ['CBS', 'CNN','ABC',	'FOX','NBC','ESPN',	'MSNBC','UNI', 'DISNEY CHANNEL', 'MTV']
list_of_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
matrix = pd.DataFrame(None,  index= list_of_weekdays, columns= list_of_channel)
outlier_matrix = getWeekdays(matrix, df,list_of_weekdays, list_of_channel)
'''
writer = pd.ExcelWriter('outlier_matrix.xlsx')
outlier_matrix.to_excel(writer,"30", index=True)
writer.save()
'''
#%%
list_of_channel = ['CBS', 'CNN','ABC',	'FOX','NBC','ESPN',	'MSNBC','UNI', 'DISNEY CHANNEL', 'MTV']
list_of_Months = ['January','February','March','April','May','June','July', 'August','September', 'October', 'November', 'December']
matrix = pd.DataFrame(None,  index= list_of_Months, columns= list_of_channel)
outlier_matrix_monthly = getMonths(matrix, df,list_of_Months, list_of_channel)

writer = pd.ExcelWriter('outlier_matrix_monthly_0.05.xlsx')
outlier_matrix_monthly.to_excel(writer,"15", index=True)
writer.save()
