-------------------------------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv(r'D:\Data\bike.csv')
pd.set_option('display.max_rows',4 )
df
-------------------------------------------------------------------------------------------------
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
datetime      10886 non-null object     #时间和日期
season        10886 non-null int64      #季节,  1 =春季，2 =夏季，3 =秋季，4 =冬季  
holiday       10886 non-null int64      #是否是假期, 1=是, 0=否
workingday    10886 non-null int64      #是否是工作日, 1=是, 0=否
weather       10886 non-null int64      #天气,1:晴朗，很少有云，部分多云，部分多云; 2:雾+多云，雾+碎云，雾+少云，雾; 3:小雪，小雨+雷雨+散云，小雨+散云; 4:大雨+冰块+雷暴+雾，雪+雾
temp          10886 non-null float64    #温度
atemp         10886 non-null float64    #体感温度
humidity      10886 non-null int64      #相对湿度
windspeed     10886 non-null float64    #风速
casual        10886 non-null int64      #未注册用户租赁数量
registered    10886 non-null int64      #注册用户租赁数量
count         10886 non-null int64      #所有用户租赁总数
dtypes: float64(3), int64(8), object(1) 
memory usage: 1020.6+ KB
-------------------------------------------------------------------------------------------------
df.describe()
-------------------------------------------------------------------------------------------------
for i in range(5, 12):
    name = df.columns[i]
    print('{0}偏态系数为 {1}, 峰态系数为 {2}'.format(name, df[name].skew(), df[name].kurt()))
temp偏态系数为 0.003690844422472008, 峰态系数为 -0.9145302637630794
atemp偏态系数为 -0.10255951346908665, 峰态系数为 -0.8500756471754651
humidity偏态系数为 -0.08633518364548581, 峰态系数为 -0.7598175375208864
windspeed偏态系数为 0.5887665265853944, 峰态系数为 0.6301328693364932
casual偏态系数为 2.4957483979812567, 峰态系数为 7.551629305632764
registered偏态系数为 1.5248045868182296, 峰态系数为 2.6260809999210672
count偏态系数为 1.2420662117180776, 峰态系数为 1.3000929518398334
-------------------------------------------------------------------------------------------------
print('未去重: ', df.shape)print('去重: ', df.drop_duplicates().shape)
未去重:  (10886, 12)
去重:  (10886, 12)
-------------------------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
#绘制箱线图
sns.boxplot(x="windspeed", data=df,ax=axes[0][0])
sns.boxplot(x='casual', data=df, ax=axes[0][1])
sns.boxplot(x='registered', data=df, ax=axes[1][0])
sns.boxplot(x='count', data=df, ax=axes[1][1])
plt.show()
-------------------------------------------------------------------------------------------------
#转换格式, 并提取出小时, 星期几, 月份
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df.datetime.dt.hour
df['week'] = df.datetime.dt.dayofweek
df['month'] = df.datetime.dt.month
df['year_month'] = df.datetime.dt.strftime('%Y-%m')
df['date'] = df.datetime.dt.date
#删除datetime
df.drop('datetime', axis = 1, inplace = True)
Df
-------------------------------------------------------------------------------------------------
import matplotlib
#设置中文字体
font = {'family': 'SimHei'}
matplotlib.rc('font', **font)
#分别计算日期和月份中位数
group_date = df.groupby('date')['count'].median()
group_month = df.groupby('year_month')['count'].median()
group_month.index = pd.to_datetime(group_month.index)
plt.figure(figsize=(16,5))
plt.plot(group_date.index, group_date.values, '-', color = 'b', label = '每天租赁数量中位数', alpha=0.8)
plt.plot(group_month.index, group_month.values, '-o', color='orange', label = '每月租赁数量中位数')
plt.legend()
plt.show()
-------------------------------------------------------------------------------------------------
import seaborn as sns
plt.figure(figsize=(10, 4))
sns.boxplot(x='month', y='count', data=df)
plt.show()
-------------------------------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(x='season', y='count', data=df)
plt.show()
-------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
sns.boxplot(x="week",y='casual' ,data=df,ax=axes[0])
sns.boxplot(x='week',y='registered', data=df, ax=axes[1])
sns.boxplot(x='week',y='count', data=df, ax=axes[2])
plt.show()
-------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 7))
sns.boxplot(x='holiday', y='casual', data=df, ax=axes[0][0])
sns.boxplot(x='holiday', y='registered', data=df, ax=axes[1][0])
sns.boxplot(x='holiday', y='count', data=df, ax=axes[2][0])
sns.boxplot(x='workingday', y='casual', data=df, ax=axes[0][1])
sns.boxplot(x='workingday', y='registered', data=df, ax=axes[1][1])
sns.boxplot(x='workingday', y='count', data=df, ax=axes[2][1])
plt.show()
-------------------------------------------------------------------------------------------------
#绘制第一个子图
plt.figure(1, figsize=(14, 8))
plt.subplot(221)
hour_casual = df[df.holiday==1].groupby('hour')['casual'].median()
hour_registered = df[df.holiday==1].groupby('hour')['registered'].median()
hour_count = df[df.holiday==1].groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.xticks(hour_casual.index)
plt.title('未注册用户和注册用户在节假日自行车租赁情况')
#绘制第二个子图
plt.subplot(222)
hour_casual = df[df.workingday==1].groupby('hour')['casual'].median()
hour_registered = df[df.workingday==1].groupby('hour')['registered'].median()
hour_count = df[df.workingday==1].groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.title('未注册用户和注册用户在工作日自行车租赁情况')
plt.xticks(hour_casual.index)
#绘制第三个子图
plt.subplot(212)
hour_casual = df.groupby('hour')['casual'].median()
hour_registered = df.groupby('hour')['registered'].median()
hour_count = df.groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.title('未注册用户和注册用户自行车租赁情况')
plt.xticks(hour_casual.index)
plt.show()
-------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
sns.boxplot(x='weather', y='casual', hue='workingday',data=df, ax=ax[0])
sns.boxplot(x='weather', y='registered',hue='workingday', data=df, ax=ax[1])
sns.boxplot(x='weather', y='count',hue='workingday', data=df, ax=ax[2])
-------------------------------------------------------------------------------------------------
df[df.weather==4]
-------------------------------------------------------------------------------------------------
sns.boxplot(x='season', y='month',data=df)
-------------------------------------------------------------------------------------------------
import numpy as np
df['group_season'] = np.where((df.month <=5) & (df.month >=3), 1,
                        np.where((df.month <=8) & (df.month >=6), 2,
                                 np.where((df.month <=11) & (df.month >=9), 3, 4)))
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
#绘制气温和季节箱线图
sns.boxplot(x='season', y='temp',data=df, ax=ax[0])
sns.boxplot(x='group_season', y='temp',data=df, ax=ax[1])
-------------------------------------------------------------------------------------------------
df.drop('season', axis=1, inplace=True)
df.shape
(10886, 16)
-------------------------------------------------------------------------------------------------
sns.pairplot(df[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])
-------------------------------------------------------------------------------------------------
df['windspeed']
0         0.0000
1         0.0000
2         0.0000
          ...   
10883    15.0013
10884     6.0032
10885     8.9981
Name: windspeed, Length: 10886, dtype: float64
-------------------------------------------------------------------------------------------------

df.loc[df.windspeed == 0, 'windspeed'] = np.nan
df.fillna(method='bfill', inplace=True)
df.windspeed.isnull().sum()
0
-------------------------------------------------------------------------------------------------
#对数转换
df['windspeed'] = np.log(df['windspeed'].apply(lambda x: x+1))
df['casual'] = np.log(df['casual'].apply(lambda x: x+1))
df['registered'] = np.log(df['registered'].apply(lambda x: x+1))
df['count'] = np.log(df['count'].apply(lambda x: x+1))
sns.pairplot(df[['windspeed', 'casual', 'registered', 'count']])
-------------------------------------------------------------------------------------------------
correlation = df.corr(method='spearman')
plt.figure(figsize=(12, 8))#绘制热力图
sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w',
            annot=True,annot_kws={'size':8},square=True)
-------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
#由于所有用户的租赁数量是由未注册用户和注册用户相加而成, 故删除.
df.drop(['casual','registered'], axis=1, inplace=True)
X = df.drop(['count'], axis=1)
y = df['count']
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
-------------------------------------------------------------------------------------------------
from sklearn.linear_model import Ridge#这里的alpha指的是正则化项参数, 初始先设置为1.
rd = Ridge(alpha=1)
rd.fit(X_train, y_train)print(rd.coef_)print(rd.intercept_)
[ 0.00770067 -0.00034301  0.0039196   0.00818243  0.03635549 -0.01558927
  0.09080788  0.0971406   0.02791812  0.06114358 -0.00099811]
2.6840271343740754
-------------------------------------------------------------------------------------------------
#设置参数以及训练模型
alphas = 10**np.linspace(-5, 10, 500)
betas = []
for alpha in alphas:
    rd = Ridge(alpha = alpha)
    rd.fit(X_train, y_train)
    betas.append(rd.coef_)
#绘制岭迹图
plt.figure(figsize=(8,6))
plt.plot(alphas, betas)
#对数据进行对数转换, 便于观察.
plt.xscale('log')
#添加网格线
plt.grid(True)
#坐标轴适应数据量
plt.axis('tight')
plt.title(r'正则化项参数$\alpha$和回归系数$\beta$岭迹图')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()
-------------------------------------------------------------------------------------------------
from sklearn.linear_model import RidgeCVfrom sklearn import metrics
rd_cv = RidgeCV(alphas=alphas, cv=10, scoring='r2')
rd_cv.fit(X_train, y_train)
rd_cv.alpha_
805.0291812295973
-------------------------------------------------------------------------------------------------
rd = Ridge(alpha=805.0291812295973) #, fit_intercept=Falserd.fit(X_train, y_train)print(rd.coef_)print(rd.intercept_)
[ 0.00074612 -0.00382265  0.00532093  0.01100823  0.03375475 -0.01582157
  0.0584206   0.09708992  0.02639369  0.0604242  -0.00116086]
2.7977274604845856
-------------------------------------------------------------------------------------------------
from sklearn import metrics
from math import sqrt
#分别预测训练数据和测试数据
y_train_pred = rd.predict(X_train)
y_test_pred = rd.predict(X_test)
#分别计算其均方根误差和拟合优度
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = rd.score(X_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = rd.score(X_test, y_test)
print('训练集RMSE: {0}, 评分: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, 评分: {1}'.format(y_test_rmse, y_test_score))
-------------------------------------------------------------------------------------------------
from sklearn.linear_model import Lasso
alphas = 10**np.linspace(-5, 10, 500)
betas = []
for alpha in alphas:
    Las = Lasso(alpha = alpha)
    Las.fit(X_train, y_train)
    betas.append(Las.coef_)
plt.figure(figsize=(8,6))
plt.plot(alphas, betas)
plt.xscale('log')
plt.grid(True)
plt.axis('tight')
plt.title(r'正则化项参数$\alpha$和回归系数$\beta$的Lasso图')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()
-------------------------------------------------------------------------------------------------
from sklearn.linear_model import LassoCVfrom sklearn import metrics
Las_cv = LassoCV(alphas=alphas, cv=10)
Las_cv.fit(X_train, y_train)
Las_cv.alpha_
0.005074705239490466
-------------------------------------------------------------------------------------------------
Las = Lasso(alpha=0.005074705239490466) #, fit_intercept=FalseLas.fit(X_train, y_train)print(Las.coef_)print(Las.intercept_)
[ 0.         -0.          0.          0.01001827  0.03467474 -0.01570339
  0.06202352  0.09721864  0.02632133  0.06032038 -0.        ]
2.7808303982442952
-------------------------------------------------------------------------------------------------
#用Lasso分别预测训练集和测试集, 并计算均方根误差和拟合优度
y_train_pred = Las.predict(X_train)
y_test_pred = Las.predict(X_test)
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = Las.score(X_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = Las.score(X_test, y_test)
print('训练集RMSE: {0}, 评分: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, 评分: {1}'.format(y_test_rmse, y_test_score))
-------------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
#训练线性回归模型
LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.coef_)
print(LR.intercept_)
#分别预测训练集和测试集, 并计算均方根误差和拟合优度
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = LR.score(X_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = LR.score(X_test, y_test)
print('训练集RMSE: {0}, 评分: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, 评分: {1}'.format(y_test_rmse, y_test_score))
[ 0.00775915 -0.00032048  0.00391537  0.00817703  0.03636054 -0.01558878
  0.09087069  0.09714058  0.02792397  0.06114454 -0.00099731]
2.6837869701964014
