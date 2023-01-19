import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
import math
from pylab import *
import matplotlib.mlab as mlab
from sklearn.utils import shuffle
import math

i = 0
j = []
data = []
X = []
indicess = []
xback = 24
with open(r'D:\error01冬季雨天.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row[:])  # 提取出每一行中的2:14列
data1 = []
data = np.array(data)
m, n = np.shape(data)
for i in range(m):
    for j in range(n):
        # print(data[i][j])
        data[i][j] = data[i][j].astype('float64')  # 是从第三列开始的
for i in range(m):
    for j in range(n):
        # print(data[i][j])
        data1.append(data[i][j])
print("the type of data1", type(data1[1]))
data = data.astype('float64')

# print(data)
print("the shape of data", len(data))


# 定义最大似然函数后的结果
def mle(x):
    u = np.mean(x)
    thea = np.std(x)
    return u, thea


# 确定了分布
print(mle(data))
u, thea = mle(data)
print(u)
print(thea)
y = st.norm.pdf(data[:6], u, thea)
print(y)
count, bins, ignored = plt.hist(data, bins=20, normed=False)
print("count", len(count))
print("bins", len(bins))
plt.plot(bins[:20], count, "r")
pro = count / np.sum(count)
plt.xlabel("x")
plt.ylabel("probability density")
plt.show()

plt.plot(bins[:20], pro, "r", lw=2)
plt.show()
low = -1.65 * thea + u  # 对应90%的置信度
up = 1.65 * thea + u
data0 = []
print("下界为", low)
print("上界为：", up)

with open(r'D:\真实值冬季雨天.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data0.append(row[:])  # 提取出每一行中的2:14列
data01 = []
data0 = np.array(data0)
# print(data0)
m, n = np.shape(data0)
print("the shape of data0", np.shape(data0))
for i in range(m):
    for j in range(n):
        # print(data0[i][j])
        data0[i][j] = data0[i][j].astype('float64')  # 是从第三列开始的
for i in range(m):
    for j in range(n):
        # print(data[i][j])
        data01.append(data0[i][j])
# print("the type of data1",type(data1[1]))
data0 = data0.astype('float64')
data01 = map(eval, data01)
print(np.shape(data0))
print(data0[:4])
print(data0[:2, 0])
datamax = np.max(data0[:, 0])
datamax = np.max(data0[:, 0])
p_low = list(map(lambda x: (x - abs(low) * datamax), data0[:, 0]))
p_up = list(map(lambda x: (x + up * datamax), data0[:, 1]))
x = [i for i in range(len(p_low))]
print(x)
# 显示置信区间范围
l = 90
k = 0
plt.plot(x[k:l], p_low[k:l], 'g', lw=2, label='下界曲线')
plt.plot(x[k:l], p_up[k:l], 'g', lw=2, label='上界曲线')
plt.plot(x[k:l], data0[k:l, 0], 'b', lw=2, label='真实值')
plt.plot(data0[k:l, 1], 'r', lw=2, label='预测值')
plt.fill_between(x[k:l], p_low[k:l], p_up[k:l], color="c", alpha=0.1)
plt.title('置信区间', fontsize=18)  # 表的名称
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize='small')
# 负责绘制与图或轴相关的数据
# savefig('D:/十折交叉验证/LSTM1.jpg')
plt.show()
# 评价置信区间PICP,PINAW,CWC，PICP用来评价预测区间的覆盖率，PINAW预测区间的宽带
count = 0

for i in range(len(p_low)):
    if data0[i][1] >= p_low[i] and data0[i][1] <= p_up[i]:
        count = count + 1

PICP = count / len(p_low)
print("PICP", PICP)

# 对于概率性的区间预测方法，在置信度一样的情况下，预测区间越窄越好
max0 = np.max(data0[:, 1])
min0 = np.min(data0[:, 1])
sum0 = list(map(lambda x: (x[1] - x[0]), zip(p_low, p_up)))
sum1 = np.sum(sum0) / len(sum0)
PINAW = 1 / (max0 - min0) * sum1
print("PINAW", PINAW)
# 综合指标的评价cwcCWC = PINAW*(1+R(PICP)*np.exp(-y(PICP-U)))
g = 90  # 取值在50-100
e0 = math.exp(-g * (PICP - u))
if PICP >= u:
    r = 0
else:
    r = 1
CWC = PINAW * (1 + r * PICP * e0)
print("CWC", CWC)
