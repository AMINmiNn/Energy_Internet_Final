# 导入库
import os

import numpy
import numpy as np  # numpy库
import sklearn.neighbors._regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import GridSearchCV
import distfit
import scipy.stats as st
from sklearn import linear_model
import time


# %%数据导入
dataset = pd.read_csv('AT_power2.csv', header=0, index_col=0)

dataset['time'] = pd.to_datetime(dataset['time'])
# train_df['label'] = (train_df['repay_date'] - train_df['auditing_date']).dt.days
dataset.loc[:, 'dayofweek'] = dataset['time'].dt.dayofweek
dataset.loc[:, 'month'] = dataset['time'].dt.month

dataset['VirtualDay'] = (dataset.index / 24).astype('int')
dataset['VirtualDay'] = dataset['VirtualDay'] % 365

dataset.loc[:, 'VDay_sin'] = round(np.sin(2 * np.pi * dataset['VirtualDay'] / 365), 5)
dataset.loc[:, 'VDay_cos'] = round(np.cos(2 * np.pi * dataset['VirtualDay'] / 365), 5)

dataset.loc[:, 'hour_sin'] = round(np.sin(2 * np.pi * dataset['time'].dt.hour / 24), 2)
dataset.loc[:, 'hour_cos'] = round(np.cos(2 * np.pi * dataset['time'].dt.hour / 24), 2)

dataset.loc[:, 'month_sin'] = round(np.sin(2 * np.pi * dataset['time'].dt.month / 12), 2)
dataset.loc[:, 'month_cos'] = round(np.cos(2 * np.pi * dataset['time'].dt.month / 12), 2)

dataset = dataset.drop(['time'], axis=1)
values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
# scaled = scaler.fit_transform(values)
# frame as supervised learning
# values = scaled
for i in range(values.shape[1]):
    col = values[:, i]  # 获取当前列数据

    #     判断当前列的数据中是否含有nan
    nan_col = np.count_nonzero(col != col)
    if nan_col != 0:  # 条件成立说明含有nan
        not_nan = col[col == col]  # 找出不含nan的
        col[np.isnan(col)] = np.mean(not_nan)  # 将nan替换成这一列的平均值

n_train_hours = 365 * 24  # 1年数据
train = values[:n_train_hours, :]
test = values[n_train_hours:n_train_hours + 24 * 365, :]  # 预测24*7小时后数据
# split into input and outputs
train_x_raw, train_y_raw = train[:, 1:], train[:, 0]
test_x_raw, test_y_raw = test[:, 1:], test[:, 0]
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
scaler4 = MinMaxScaler()
train_x = scaler1.fit_transform(train_x_raw)
train_y = scaler2.fit_transform(train_y_raw.reshape(-1, 1))
test_x = scaler3.fit_transform(test_x_raw)
test_y = scaler4.fit_transform(test_y_raw.reshape(-1, 1))

# reshape input to be 3D [samples, timesteps, features]
# %%
# 模型
model_SVR = SVR(kernel='rbf', gamma=0.01, C=450)  # 建立支持向量机回归模型对象

model_MLP = MLPRegressor(alpha=0.03, hidden_layer_sizes=(180, 90), activation='relu', solver='adam', random_state=11)
# model_RF = RandomForestRegressor(n_estimators=460, random_state=11, max_depth=21, max_features=12)
model_RF = RandomForestRegressor(random_state=11, n_estimators=460, max_features=None, max_depth=19,
                                 min_samples_split=2, min_samples_leaf=1)
model_GBDT = GradientBoostingRegressor(random_state=11, n_estimators=350, learning_rate=0.05, max_depth=7,
                                       subsample=0.55, min_samples_split=100)
model_ABR = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, min_samples_split=20, min_samples_leaf=5),
                              random_state=11, n_estimators=400, learning_rate=0.05, loss='linear')

model_names = ['SVM', 'MLP', 'RF', 'GBDT']  # 不同模型的名称列表
# model_dic = [model_SVR, model_MLP, model_RF, model_GBDT]  # 不同回归模型对象的集合
model_dic = [model_ABR]

# %%model_SVR
model_SVR = SVR(kernel='rbf')
pre_y_list = []  # 各个回归模型预测的y值列表
# '''
gamma = [0.3, 0.1, 0.03, 0.01]
C = [10, 25, 450]
# C = range(330, 500, 20)
param_grid = dict(gamma=gamma, C=C)
gsearch1 = GridSearchCV(model_SVR, param_grid, cv=4, verbose=2, scoring="neg_mean_squared_error", )
gsearch1.fit(train_x, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))

print(gsearch1.best_params_)
print(gsearch1.best_score_)

# '''
'''
gammas = []
Cs=[]
score = []
for gamma in np.arange(0.015, 0.035, 0.002):
    for C in np.arange(100, 118, 2):
        print('~' * 30, '\ngamma={}:, C={}:\n'.format(gamma, C))
        gammas.append(gamma)
        Cs.append(C)
        model_SVR = SVR(kernel='rbf', gamma=gamma, C=C)
        sc = np.sqrt(
            -cross_val_score(model_SVR, train_x, train_y.ravel(), cv=n_folds, scoring="neg_mean_squared_error",
                             verbose=1))
        score.append(sc.mean())
plt.plot(gammas, score)
plt.xlabel('gamma')
plt.ylabel('score')
plt.show()
'''
pre_y_list = []  # 各个回归模型预测的y值列表
model_SVR = SVR(kernel='rbf', gamma=0.01, C=10)
pre_y_list.append(scaler4.inverse_transform(model_SVR.fit(train_x, train_y.ravel()).predict(test_x).reshape(-1, 1)))

# %%model_RF
n_folds = 6  # 设置交叉检验的次数
'''
model_RF = RandomForestRegressor(
    n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=11, verbose=1,
    warm_start=False
)
'''
model_RF = RandomForestRegressor(random_state=11, n_estimators=460, max_depth=19, max_features=None, oob_score=True)
n_e = [460, 480]
m_f = [13, 14, 15]
m_d = range(19, 24, 1)
m_ss = range(5, 11, 5)
m_sl = range(5, 11, 5)
param_grid = dict(min_samples_leaf=m_sl, min_samples_split=m_ss)
gsearch1 = GridSearchCV(model_RF, param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error', )
gsearch1.fit(train_x, train_y.ravel())

print(gsearch1.best_params_)
print(gsearch1.best_score_)

pre_y_list = []
model_RF = RandomForestRegressor(random_state=11, n_estimators=300, max_depth=5, max_features=None,
                                 min_samples_split=5,
                                 min_samples_leaf=10)
pre_y_list.append(scaler4.inverse_transform(model_RF.fit(train_x, train_y.ravel()).predict(test_x).reshape(-1, 1)))
# %%model_GBDT
model_GBDT = GradientBoostingRegressor(random_state=11)
pre_y_list = []  # 各个回归模型预测的y值列表
lr = [0.01, 0.03, 0.05]
n_e = [300, 350, 400]
subsample = [0.55, 0.65, 0.75]
m_d = [5, 6, 7, 8, 9]
m_ss = [40, 100]
param_grid = dict(learning_rate=lr, n_estimators=n_e, subsample=subsample, max_depth=m_d, min_samples_split=m_ss)
gsearch1 = GridSearchCV(model_GBDT, param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error', )
gsearch1.fit(train_x, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
print(gsearch1.best_params_)
print(gsearch1.best_score_)
'''
for m_d in np.arange(10, 101, 10):
    print('~' * 30, '\nn_e={}:,\n'.format(m_d))
    m_ds.append(m_d)
    model_RF = GradientBoostingRegressor(n_estimators=m_d, random_state=11)
    sc = np.sqrt(
        -cross_val_score(model_RF, train_x, train_y.ravel(), cv=n_folds, scoring="neg_mean_squared_error",
                         verbose=1))
    score.append(sc.mean())
plt.plot(m_ds, score)
plt.xlabel('m_f')
plt.ylabel('score')
plt.show()
'''
pre_y_list = []
model_GBDT = GradientBoostingRegressor(random_state=11, n_estimators=400, learning_rate=0.01, subsample=0.55,
                                       max_depth=5, min_samples_split=100)
pre_y_list.append(scaler4.inverse_transform(model_GBDT.fit(train_x, train_y.ravel()).predict(test_x).reshape(-1, 1)))

# %%
model_ABR = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, min_samples_split=20, min_samples_leaf=5),
                              random_state=11, n_estimators=400, learning_rate=0.05, loss='linear')
# AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, min_samples_split=20, min_samples_leaf=5),
#                               random_state=11, n_estimators=400, learning_rate=0.05, loss='linear')
pre_y_list = []  # 各个回归模型预测的y值列表
lr = [0.2, 0.5]
n_e = [450, 500]
# subsample = [0.55, 0.65, 0.75]
m_d = [5, 7, 9]
m_ss = [40, 100]
param_grid = dict(learning_rate=lr, n_estimators=n_e)
gsearch1 = GridSearchCV(model_ABR, param_grid, cv=4, verbose=2, scoring='neg_mean_squared_error', )
gsearch1.fit(train_x, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
print(gsearch1.best_params_)
print(gsearch1.best_score_)

pre_y_list = []
model_ABR = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, min_samples_split=20, min_samples_leaf=5),
                              random_state=11, n_estimators=450, learning_rate=0.2, loss='linear')
pre_y_list.append(scaler4.inverse_transform(model_ABR.fit(train_x, train_y.ravel()).predict(test_x).reshape(-1, 1)))

# %%model_KNR
model_KNR = KNeighborsRegressor(weights="distance", algorithm="auto")

pre_y_list = []  # 各个回归模型预测的y值列表
l_s = [5]
n_n = [14, 15, 16]
param_grid = dict(leaf_size=l_s, n_neighbors=n_n)
gsearch1 = GridSearchCV(model_KNR, param_grid, cv=4, verbose=2, scoring='neg_mean_squared_error', )
gsearch1.fit(train_x, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
print(gsearch1.best_params_)
print(gsearch1.best_score_)

pre_y_list = []
model_KNR = KNeighborsRegressor(weights="distance", algorithm="auto")
pre_y_list.append(scaler4.inverse_transform(model_KNR.fit(train_x, train_y.ravel()).predict(test_x).reshape(-1, 1)))

# %%kernel=RBF,  normalize_y=False, random_state=11
model_KR = KernelRidge(kernel='rbf')
# AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, min_samples_split=20, min_samples_leaf=5),
#                               random_state=11, n_estimators=400, learning_rate=0.05, loss='linear')
pre_y_list = []  # 各个回归模型预测的y值列表
al = [0.03, 0.1]
ga = [0.3, 0.1]
param_grid = dict(alpha=al, gamma=ga)
gsearch1 = GridSearchCV(model_KR, param_grid, cv=4, verbose=2, scoring='neg_mean_squared_error', )
gsearch1.fit(train_x, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# %%model_MLP
# model_MLP = MLPRegressor(
#     alpha=0.03, hidden_layer_sizes=(140, 35), activation='relu', solver='adam', random_state=11
# )
model_MLP = MLPRegressor(random_state=11, activation='relu', solver='adam')
pre_y_list = []  # 各个回归模型预测的y值列表
al = [0.02, 0.03, 0.05]
h_l = [(i, j) for i in range(160, 210, 10) for j in range(80, 105, 5)]
param_grid = dict(hidden_layer_sizes=h_l, alpha=al)
gsearch1 = GridSearchCV(model_MLP, param_grid, scoring='neg_mean_squared_error', cv=4, verbose=2)
gsearch1.fit(train_x, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
print(gsearch1.best_params_)
print(gsearch1.best_score_)

pre_y_list = []
model_MLP = MLPRegressor(alpha=0.02, hidden_layer_sizes=(170, 90), activation='relu', solver='adam', random_state=11)
pre_y_list.append(scaler4.inverse_transform(model_MLP.fit(train_x, train_y.ravel()).predict(test_x).reshape(-1, 1)))
# %%
a = 5000
b = 24
# plt.plot(train_y[a:a + b], label='train')

plt.plot(test_y[a:a + b], label='real')
plt.plot(pre_y_list[0][a:a + b], label='pred')
# plt.plot(pre_y_list[1][a:a + b], label='ANN')
# plt.plot(pre_y_list[2][a:a + b], label='SVM')
plt.legend()
plt.show()
# %%预测结果处理
# 平移处理，负荷无负值
if np.min(pre_y_list[0]) < 0:
    temp_min = np.min(pre_y_list[0])
    for i in range(len(pre_y_list[0])):
        pre_y_list[0][i] = pre_y_list[0][i] - temp_min

real = test_y_raw.reshape(-1, 1)
# 作图
plt.plot(test_y_raw, label='real')
plt.plot(pre_y_list[0], label='pred')
plt.title("MAPE = {:.2f}%".format(100 * mean_absolute_percentage_error(test_y_raw, pre_y_list[0])))
plt.legend()
plt.show()
# %%求特征重要性
fi = model_GBDT.feature_importances_.reshape(-1, 1)

# %%求误差分布
predict = pre_y_list[0]
percentage_error_up = []
percentage_error_low = []

k = 0
for i in range(len(real)):
    if real[i] != 0:
        p_e = (predict[i] - real[i]) / real[i]
        if p_e >= 0:
            percentage_error_up.append(p_e)
        else:
            percentage_error_low.append(-p_e)

percentage_error_up = np.array(percentage_error_up)
percentage_error_low = np.array(percentage_error_low)
dist1 = distfit.distfit(todf=True, alpha=0.05)
dist1.fit_transform(percentage_error_up)
dist1.plot()
plt.show()
dist2 = distfit.distfit(todf=True, alpha=0.05)
dist2.fit_transform(percentage_error_low)
dist2.plot()
plt.show()
# %%绘制概率分布图
x = range(168)
plt.title("AT_wind_interval_prediction")
plt.xlabel("t/hour")
plt.ylabel("AT_wind_power/MW")
predict_up_90 = predict + predict * 0.491348592248658
predict_up_80 = predict + predict * 0.36800481592970944
predict_up_60 = predict + predict * 0.2891681902876261
predict_up_40 = predict + predict * 0.21777260713011395
predict_up_20 = predict + predict * 0.17254764502471595
predict_up_10 = predict + predict * 0.15360228719589675

predict_low_90 = predict - predict * 0.44443957248573357
predict_low_80 = predict - predict * 0.3441282064736304
predict_low_60 = predict - predict * 0.24327358312921868
predict_low_40 = predict - predict * 0.18386844327028132
predict_low_20 = predict - predict * 0.14142363237771383
predict_low_10 = predict - predict * 0.12394431130732217

plt.plot(x, predict, label='predict', color='darkgreen')
plt.plot(x, real,label='real',  color='darkblue')
plt.fill_between(x, predict_up_90.ravel(), predict_up_80.ravel(),
                 label='Confidence Interval:90%',  # 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=0.4)  # 透明度
plt.fill_between(x, predict_up_80.ravel(), predict_up_60.ravel(),  # 上限，下限
                 label='Confidence Interval:80%',
                 facecolor='blue',  # 填充颜色
                 alpha=0.5)  # 透明度
plt.fill_between(x, predict_up_60.ravel(), predict_up_40.ravel(),  # 上限，下限
                 label='Confidence Interval:60%',
                 facecolor='blue',  # 填充颜色
                 alpha=0.7)  # 透明度
plt.fill_between(x, predict_up_40.ravel(), predict_up_20.ravel(),  # 上限，下限
                 label='Confidence Interval:40%',
                 facecolor='blue',  # 填充颜色
                 alpha=0.8)  # 透明度
plt.fill_between(x, predict_up_20.ravel(), predict_up_10.ravel(),  # 上限，下限
                 label='Confidence Interval:20%',
                 facecolor='blue',  # 填充颜色
                 alpha=0.9)  # 透明度
plt.fill_between(x, predict_up_10.ravel(), predict.ravel(),  # 上限，下限
                 label='Confidence Interval:10%',
                 facecolor='blue',  # 填充颜色
                 alpha=1)  # 透明度


plt.fill_between(x, predict_low_90.ravel(), predict_low_80.ravel(),# 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=0.4)  # 透明度
plt.fill_between(x, predict_low_80.ravel(), predict_low_60.ravel(),  # 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=0.5)  # 透明度
plt.fill_between(x, predict_low_60.ravel(), predict_low_40.ravel(),  # 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=0.7)  # 透明度
plt.fill_between(x, predict_low_40.ravel(), predict_low_20.ravel(),  # 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=0.8)  # 透明度
plt.fill_between(x, predict_low_20.ravel(), predict_low_10.ravel(),  # 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=0.9)  # 透明度
plt.fill_between(x, predict_low_10.ravel(), predict.ravel(),  # 上限，下限
                 facecolor='blue',  # 填充颜色
                 alpha=1.0)  # 透明度
plt.legend()
plt.show()
