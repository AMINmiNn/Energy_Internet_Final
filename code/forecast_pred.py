# 导入库
import os
import numpy as np  # numpy库
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # RF,GBDT集成算法
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error, \
    mean_squared_error, r2_score
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import GridSearchCV
from ShapleyValue_new import shapley_value, federate
from sklearn import linear_model


def record_metrics(test_y, pred_y):
    model_metrics_name = [explained_variance_score, mean_absolute_percentage_error, mean_squared_error,
                          r2_score]  # 回归评估指标对象集
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(test_y, pred_y)  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    return tmp_list


def ensemble(A, coef):
    pred_en = np.ones(A.shape[0]) * coef[-1]
    for i in range(A.shape[1]):
        pred_en += coef[i] * A[:, i]
    return pred_en


def leaveoneout_m(pre_y_list2, test_y2, pred_mean2):
    allo_loo_mm = []
    pred_3mean = [
        federate([pre_y_list2[2], pre_y_list2[3]]),
        federate([pre_y_list2[1], pre_y_list2[3]]),
        federate([pre_y_list2[1], pre_y_list2[2]]),
    ]

    Lomega_m = mean_absolute_percentage_error(test_y2, pred_mean2)
    for i in range(3):
        Lxk_3mean = mean_absolute_percentage_error(test_y2, pred_3mean[i])
        allo_loo_mm.append(Lxk_3mean - Lomega_m)
    allo_loo_mm = allo_loo_mm / np.sum(np.array(allo_loo_mm), axis=0)
    return allo_loo_mm


# %%    Read dataset
dataset = pd.DataFrame()
for i in range(1, 16):
    pathcsv = os.path.abspath("GEFCom2014 Data/GEFCom2014-L_V2/Load/Task " + str(i) + "/L" + str(i) + "-train.csv")
    file = pd.read_csv(pathcsv)
    if i == 1:
        dataset = file.dropna(axis=0)
    else:
        dataset = pd.concat([dataset, file], axis=0, ignore_index=True)

dataset['TIMESTAMP'] = pd.date_range(start='2005-1-1 1:00:00', end='2011-12-1 00:00:00', freq='H')
temp = dataset[['w' + str(i) for i in range(1, 26)]].apply(pd.to_numeric, downcast='float')
dataset = dataset.drop(['ZONEID'] + ['w' + str(i) for i in range(1, 26)], axis=1)
dataset = dataset.copy()
dataset['w'] = temp.mean(axis=1)
time_before = [24, 48, 72, 96, 120, 144, 168]
for i in time_before:
    dataset['L-' + str(i)] = np.nan
    dataset.loc[i:, 'L-' + str(i)] = dataset['LOAD'].values[:-i]
    dataset['w-' + str(i)] = np.nan
    dataset.loc[i:, 'w-' + str(i)] = dataset['w'].values[:-i]
for i in [1, 2, 3]:
    dataset['w-' + str(i)] = np.nan
    dataset.loc[i:, 'w-' + str(i)] = dataset['w'].values[:-i]
dataset = dataset[23:].reset_index(drop=True)

dataset = dataset[(dataset['TIMESTAMP'].dt.year >= 2008) & (dataset['TIMESTAMP'].dt.year <= 2011)].reset_index(
    drop=True)
# dataset.loc[:, 'hour'] = dataset['TIMESTAMP'].dt.hour
# dataset.loc[:, 'month'] = dataset['TIMESTAMP'].dt.month
# dataset.loc[:, 'dayofweek'] = dataset['TIMESTAMP'].dt.dayofweek
dataset.loc[:, 'day'] = dataset['TIMESTAMP'].dt.day
dataset.loc[:, 'hour_sin'] = np.sin(2 * np.pi * dataset['TIMESTAMP'].dt.hour / 24)
dataset.loc[:, 'hour_cos'] = np.cos(2 * np.pi * dataset['TIMESTAMP'].dt.hour / 24)
dataset.loc[:, 'dayofweek_sin'] = np.sin(2 * np.pi * dataset['TIMESTAMP'].dt.dayofweek / 7)
dataset.loc[:, 'dayofweek_cos'] = np.cos(2 * np.pi * dataset['TIMESTAMP'].dt.dayofweek / 7)
dataset.loc[:, 'month_sin'] = np.sin(2 * np.pi * dataset['TIMESTAMP'].dt.month / 12)
dataset.loc[:, 'month_cos'] = np.cos(2 * np.pi * dataset['TIMESTAMP'].dt.month / 12)
# %%
# 数据准备
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
# 2008 2009
train = dataset[(dataset['TIMESTAMP'].dt.year >= 2008) & (dataset['TIMESTAMP'].dt.year <= 2009)].reset_index(
    drop=True)  # 分割自变量
train_x = train.drop(['TIMESTAMP', 'LOAD'], axis=1).reset_index(drop=True)
train_y = train['LOAD'].values.reshape(-1, 1)
train_x = scaler1.fit_transform(train_x)
train_y = scaler2.fit_transform(train_y)
# 2010
test1 = dataset[(dataset['TIMESTAMP'].dt.year == 2010)].reset_index(drop=True)  # 分割因变量
test_x1 = test1.drop(['TIMESTAMP', 'LOAD'], axis=1).reset_index(drop=True)
test_y1 = test1['LOAD'].values.reshape(-1, 1)
test_x1 = scaler1.transform(test_x1)
# 2011
test2 = dataset[(dataset['TIMESTAMP'].dt.year == 2011)].reset_index(drop=True)  # 分割因变量
test_x2 = test2.drop(['TIMESTAMP', 'LOAD'], axis=1).reset_index(drop=True)
test_y2 = test2['LOAD'].values.reshape(-1, 1)
test_x2 = scaler1.transform(test_x2)
# %%
# 模型
model_SVR = SVR(kernel='rbf', gamma=0.01, C=450)  # 建立支持向量机回归模型对象
model_MLP = MLPRegressor(
    alpha=0.03, hidden_layer_sizes=(180, 90), activation='relu', solver='adam', random_state=11
)
# model_RF = RandomForestRegressor(n_estimators=460, random_state=11, max_depth=21, max_features=12)
model_RF = RandomForestRegressor(random_state=11, n_estimators=460, max_features=None, max_depth=19,
                                 min_samples_split=2, min_samples_leaf=1)
model_GBDT = GradientBoostingRegressor(random_state=11, n_estimators=350, learning_rate=0.05, max_depth=7,
                                       subsample=0.55, min_samples_split=100)

model_names = ['SVM', 'MLP', 'RF', 'GBDT']  # 不同模型的名称列表
model_dic = [model_SVR, model_MLP, model_RF, model_GBDT]  # 不同回归模型对象的集合
# %%    fit and predict
pre_y_list1 = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    '''
    model.fit(train_x, train_y.ravel())
    pred_y = []
    for i in range(len(test_x)):
        pred_point = model.predict(test_x[i].reshape(1, -1))
        if i + 3 < len(test_x):
            test_x[i + 1, 1] = pred_point
            test_x[i + 2, 2] = pred_point
            test_x[i + 3, 3] = pred_point
        elif i + 2 < len(test_x):
            test_x[i + 1, 1] = pred_point
            test_x[i + 2, 2] = pred_point
        elif i + 1 < len(test_x):
            test_x[i + 1, 1] = pred_point
        pred_y.append(pred_point)
    pre_y_list.append(pred_y)
    '''
    pre_y_list1.append(scaler2.inverse_transform(
        model.fit(train_x, train_y.ravel()).predict(test_x1).reshape(-1, 1)))  # 将回归训练中得到的预测y存入列表



# %%
# 模型效果指标评估
n_samples, n_features = train_x.shape  # 总样本量,总特征数
model_metrics_list1 = []  # 回归评估指标列表
for i in range(len(model_dic)):  # 循环每个模型索引
    model_metrics_list1.append(record_metrics(test_y1, pre_y_list1[i]))  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(model_metrics_list1, index=model_names, columns=['ev', 'mape', 'mse', 'r2'])  # 建立回归指标的数据框

print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('regression metrics:')  # 打印输出标题
print(df1)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance_score')
print('mape \t mean_absolute_percentage_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线
# %%
'''
# 模型效果可视化
color_list = ['c', 'r', 'b', 'g', 'y']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
plt.figure()  # 创建画布
plt.plot(np.arange(test_x1.shape[0]), test_y1, color='orange', label='actual data')  # 画出原始值的曲线
for i in range(len(pre_y_list1) - 1):  # 读出通过回归模型预测得到的索引及结果
    i += 1
    plt.plot(np.arange(test_x1.shape[0]), pre_y_list1[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
a = 5000
b = 168
plt.figure()  # 创建画布
plt.plot(np.arange(test_x1.shape[0])[a:a + b], test_y1[a:a + b], color='orange', label='actual data')  # 画出原始值的曲线
# color_list = ['c', 'r', 'b', 'g', 'y']  # 颜色列表
# linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i in range(len(pre_y_list1) - 1):  # 读出通过回归模型预测得到的索引及结果
    i += 1
    plt.plot(np.arange(test_x1.shape[0])[a:a + b], pre_y_list1[i][a:a + b], color_list[i],
             label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper center')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
'''
# %%
pred_mean1 = np.sum(pre_y_list1[1:], axis=0) / 3
df_pm1 = pd.DataFrame(np.array(record_metrics(test_y1, pred_mean1)).reshape((1, 4)), index=['pred_mean'],
                      columns=['ev', 'mape', 'mse', 'r2'])
df1 = pd.concat([df1, df_pm1], axis=0)
print('regression metrics:')  # 打印输出标题
print(df_pm1)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线

plt.plot(test_y1[5000:5000 + 168], label='real')
plt.plot(pred_mean1[5000:5000 + 168], label='pred_mean')
plt.title("MAPE = {:.2f}%".format(100 * mean_absolute_percentage_error(test_y1, pred_mean1)))
plt.legend()
plt.show()
# %%
allo_sv_m1 = shapley_value(test_y1, pre_y_list1[1:], pre_ori=pre_y_list1[0], methods='mean')
allo_loo_mm1 = leaveoneout_m(pre_y_list1, test_y1, pred_mean1)
allo_mape_mm1 = [df1['mape'][0] - df1['mape'][i + 1] for i in range(3)]
allo_mape_mm1 = allo_mape_mm1 / np.sum(np.array(allo_mape_mm1), axis=0)
print(allo_sv_m1, allo_loo_mm1, allo_mape_mm1)
# %%
A1 = np.asarray(pre_y_list1[1:]).reshape(3, 8760).transpose()
model_Li1 = linear_model.LinearRegression()
model_Li1.fit(A1, test_y1)
coef1 = np.concatenate((model_Li1.coef_.reshape(A1.shape[1]), model_Li1.intercept_))
print('sum of coef1:', np.sum(model_Li1.coef_))
pred_en1 = ensemble(A1, coef1)
df_en1 = pd.DataFrame(np.array(record_metrics(test_y1, pred_en1)).reshape((1, 4)), index=['pred_en1'],
                      columns=['ev', 'mape', 'mse', 'r2'])

df1 = pd.concat([df1, df_en1], axis=0)
print(df_en1)
print(70 * '-')  # 打印分隔线
print(df1)
# %%
pre_y_list2 = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    pre_y_list2.append(scaler2.inverse_transform(model.predict(test_x2).reshape(-1, 1)))  # 将回归训练中得到的预测y存入列表

# 模型效果指标评估
n_samples, n_features = train_x.shape  # 总样本量,总特征数
model_metrics_list2 = []  # 回归评估指标列表
for i in range(len(model_dic)):  # 循环每个模型索引
    model_metrics_list2.append(record_metrics(test_y2, pre_y_list2[i]))  # 将结果存入回归评估指标列表
df2 = pd.DataFrame(model_metrics_list2, index=model_names, columns=['ev', 'mape', 'mse', 'r2'])  # 建立回归指标的数据框

pred_mean2 = np.sum(pre_y_list2[1:], axis=0) / 3
df_pm2 = pd.DataFrame(np.array(record_metrics(test_y2, pred_mean2)).reshape((1, 4)), index=['pred_mean2'],
                      columns=['ev', 'mape', 'mse', 'r2'])
df2 = pd.concat([df2, df_pm2], axis=0)
# print('regression metrics:')  # 打印输出标题
# print(df_pm2)  # 打印输出回归指标的数据框
# print(70 * '-')  # 打印分隔线
# %%
A2 = np.asarray(pre_y_list2[1:]).reshape(3, len(test_y2)).transpose()
pred_en2 = ensemble(A2, coef1)
df_en2 = pd.DataFrame(np.array(record_metrics(test_y2, pred_en2)).reshape((1, 4)), index=['pred_en2'],
                      columns=['ev', 'mape', 'mse', 'r2'])

df2 = pd.concat([df2, df_en2], axis=0)

print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance_score')
print('mape \t mean_absolute_percentage_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线
# %%
'''
Shapley Value
'''
allo_sv_m = shapley_value(test_y2, pre_y_list2[1:], pre_ori=pre_y_list2[0], methods='mean', metrics=mean_squared_error)
allo_sv_w = shapley_value(test_y2, pre_y_list2[1:], pre_ori=pre_y_list2[0], metrics=mean_squared_error,
                          methods='weight', test_y_w=test_y1, pre_all_w=pre_y_list1[1:], pre_ori_w=pre_y_list1[0])
# %%
'''
# leave one out: mean

allo_loo_mm = []
allo_loo_mp = []

pred_3mean = [
    federate([pre_y_list2[0], pre_y_list2[2], pre_y_list2[3]]),
    federate([pre_y_list2[0], pre_y_list2[1], pre_y_list2[3]]),
    federate([pre_y_list2[0], pre_y_list2[1], pre_y_list2[2]]),
]
pred_2mean = [
    federate([pre_y_list2[0], pre_y_list2[1]]),
    federate([pre_y_list2[0], pre_y_list2[2]]),
    federate([pre_y_list2[0], pre_y_list2[3]]),
]

Lwi = mean_absolute_percentage_error(test_y2, pre_y_list2[0])
Lomega_m = mean_absolute_percentage_error(test_y2, pred_mean2)
for i in range(3):
    Lxk_3mean = mean_absolute_percentage_error(test_y2, pred_3mean[i])
    Lxk_2mean = mean_absolute_percentage_error(test_y2, pred_2mean[i])
    allo_loo_mm.append(leaveoneout(Lwi, Lomega_m, Lxk_3mean, method='minus'))
    allo_loo_mp.append(leaveoneout(Lwi, Lomega_m, Lxk_2mean, method='plus'))
allo_loo_mm = allo_loo_mm / np.sum(np.array(allo_loo_mm), axis=0)
allo_loo_mp = allo_loo_mp / np.sum(np.array(allo_loo_mp), axis=0)

# %%
# leave one out: mse ensemble

allo_loo_enm = []
allo_loo_enp = []

pred_3en = [
    federate([pre_y_list2[0], pre_y_list2[2], pre_y_list2[3]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[0], pre_y_list1[2], pre_y_list1[3]]),
    federate([pre_y_list2[0], pre_y_list2[1], pre_y_list2[3]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[0], pre_y_list1[1], pre_y_list1[3]]),
    federate([pre_y_list2[0], pre_y_list2[1], pre_y_list2[2]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[0], pre_y_list1[1], pre_y_list1[2]]),
]
pred_2en = [
    federate([pre_y_list2[0], pre_y_list2[1]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[0], pre_y_list1[1]]),
    federate([pre_y_list2[0], pre_y_list2[2]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[0], pre_y_list1[2]]),
    federate([pre_y_list2[0], pre_y_list2[3]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[0], pre_y_list1[3]]),
]

Lwi = mean_absolute_percentage_error(test_y2, pre_y_list2[0])
Lomega_en = mean_absolute_percentage_error(test_y2, pred_en2)
for i in range(3):
    Lxk_3en = mean_absolute_percentage_error(test_y2, pred_3en[i])
    Lxk_2en = mean_absolute_percentage_error(test_y2, pred_2en[i])
    allo_loo_enm.append(leaveoneout(Lwi, Lomega_en, Lxk_3en, method='minus'))
    allo_loo_enp.append(leaveoneout(Lwi, Lomega_en, Lxk_2en, method='plus'))
allo_loo_enm = allo_loo_enm / np.sum(np.array(allo_loo_enm), axis=0)
allo_loo_enp = allo_loo_enp / np.sum(np.array(allo_loo_enp), axis=0)
'''
# %%
'''
leave one out:mean
'''
allo_loo_mm = []
pred_3mean = [
    federate([pre_y_list2[2], pre_y_list2[3]]),
    federate([pre_y_list2[1], pre_y_list2[3]]),
    federate([pre_y_list2[1], pre_y_list2[2]]),
]
measure = mean_squared_error
Lomega_m = measure(test_y2, pred_mean2)
for i in range(3):
    Lxk_3mean = measure(test_y2, pred_3mean[i])
    allo_loo_mm.append(Lxk_3mean - Lomega_m)
allo_loo_mm = allo_loo_mm / np.sum(np.array(allo_loo_mm), axis=0)
# %%
'''
leave one out:mse ensemble
'''
allo_loo_enm = []

pred_3en = [
    federate([pre_y_list2[2], pre_y_list2[3]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[2], pre_y_list1[3]]),
    federate([pre_y_list2[1], pre_y_list2[3]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[1], pre_y_list1[3]]),
    federate([pre_y_list2[1], pre_y_list2[2]], methods='weight', test_y_w=test_y1,
             pre_all_w=[pre_y_list1[1], pre_y_list1[2]]),
]
measure = mean_squared_error
Lomega_en = measure(test_y2, pred_en2)
for i in range(3):
    Lxk_3en = measure(test_y2, pred_3en[i])
    allo_loo_enm.append(Lxk_3en - Lomega_en)
allo_loo_enm = allo_loo_enm / np.sum(np.array(allo_loo_enm), axis=0)
# %%
allo_mape_mm = [df2['mape'][0] - df2['mape'][i + 1] for i in range(3)]
allo_mape_mm = allo_mape_mm / np.sum(np.array(allo_mape_mm), axis=0)

allo_mse_mm = [df2['mse'][0] - df2['mse'][i + 1] for i in range(3)]
allo_mse_mm = allo_mse_mm / np.sum(np.array(allo_mse_mm), axis=0)
# %%
# 模型效果可视化
color_list = ['c', 'r', 'b', 'g', 'y']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
plt.figure()  # 创建画布
plt.plot(np.arange(test_x2.shape[0]), test_y2, color='orange', label='actual data')  # 画出原始值的曲线
for i in range(len(pre_y_list2) - 1):  # 读出通过回归模型预测得到的索引及结果
    i += 1
    plt.plot(np.arange(test_x2.shape[0]), pre_y_list2[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像


a = 3000
b = 168
plt.figure()  # 创建画布
plt.plot(np.arange(test_x2.shape[0])[a:a + b], test_y2[a:a + b], color='orange', label='actual data')  # 画出原始值的曲线
# color_list = ['c', 'r', 'b', 'g', 'y']  # 颜色列表
# linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i in range(len(pre_y_list2) - 1):  # 读出通过回归模型预测得到的索引及结果
    i += 1
    plt.plot(np.arange(test_x2.shape[0])[a:a + b], pre_y_list2[i][a:a + b], color_list[i],
             label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper center')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像


#%%

