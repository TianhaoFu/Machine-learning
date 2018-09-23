import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
关键：matplotlib库的很多函数的灵活使用方能使数据可视化
fig = plt.figure()#幕布
ax = fig.add_subplot(3, 4, 12)#截取幕布的一部分
x = np.linspace(-5, 5, 10)
y = x ** 2 + 1
ax.plot(x, y)
ax = fig.add_subplot(211)#截取幕布的一部分
ax.plot(x, y)
plt.show()
"""
def cost_function(X ,y ,theta):
     inner = np.power((( X *theta.T ) -y) ,2)  # 列向量
     return np.sum(inner ) /( 2 *len(X))   # 对列向量进行求和取平均返回

def gradient_descent(X ,y ,theta ,alpha=0.01 ,iters=1000): #返回一个给选定形状和类型的用0填充的数组
    cost = np.zeros(iters)
    all_theta = np.zeros((iters ,X.shape[1]))
    temp = np.matrix(np.zeros(theta.shape))
    for i in range(iters):
        errors = X * theta.T -y
        for j in range(theta.shape[1]):
            theta[0, j] = theta[0, j] - alpha * (np.sum(np.multiply(errors, (X[:, j]))).len(X))
    all_theta[i, 0] = theta[0, 0]#all_theta是对所有的theta进行存储
    all_theta[i, 1] = theta[0, 1]#theta是每次 计算时候的值
    cost[i] = compute_cost(X, y, theta)#cost是每次迭代theta之后的值，也一并存储下来

    return theta, cost, all_theta
path = 'F:\ML\MachineLearningEx\MachineLearningEx\machine - learning - ex1\ex1\ex1data1.txt'
    data = pd.read_csv(path, names=['population', 'profit'])

def prob1(): #读取表格，生成pd.DataFrame对象

#seaborn库线性粗略模拟画图
    sns.regplot('population', 'profit', data, fit_reg=False)
# 创建一个sns对象
    plt.show()

def prob2
"""
首先录入数据 录入数据并不是矩阵 转换成矩阵
"""
columns = data.shape[1]
X = data.iloc[:, :columns - 1]
y = data.iloc[:, columns - 1:columns]
# 创立一个一维向量 与自变量的取值个数相同
w = pd.Series(np.ones(X.shape[0]))

X.insert(0,'Ones', w)
X = np.matrix(X.values)
y = np.matrix(y.values)
# 创立参数列向量
theta = np.matrix(np.array([0, 0]))

theta, cost, all_theta = gradient_descent(X, y, theta)
x = np.linsapce(data.population.min(),data.population.max(),100) #在最大值和最小值之间均匀的取100个值
f = theta[0,0] + (theta[0,1]*x)#g[0,0]相当于十个100维的列向量 f是最后一次迭代后出来的theta函数
fig,ax = plt.subplots(figsize=(12,8))
#figure使用add_subplot，pyplot使用的是subplot生成一个 figure
#figure里面再包含多个axes，axes本质上数组
#pyplot由两个函数fugure(figsize=指定画板大小)
#和subplot(nrows=,ncols,figsize=(大小))
#fig =plt.figure(figsize=(12,8))
ax.plot(x,f,'r',label='prediction')
#真正画图的数据
ax.scatter(data.population,data.profit,marker='x',label='training data')
#画散点图 原始数据
ax.set_title('predicted profit vs population size')
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.legend(loc=2)
plt.show()
#画预测出的函数与population之间的关系

def pro3:
    """画出1000次迭代后的tcost值"""
    fig,ax = plt.subplot(figsize=(12,8))
    ax.plot(np.arrange(1000),cost,'blue',label='error')
    #真正画图的数据
    ax.set_title('Error vs. Triaining Epoch')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    plt.show()

def pro4:
    path1 = os.path.join(r'C:\Users\TenSh\Desktop\MachineLearningEx\ex1\ex1', 'ex1data2.txt')
    #os.path.join(os.getcwd(),'data')就是获取当前目录，并组合成新目录
    data1 = pd.read_csv(path1, names=['size', 'bedrooms', 'price'])

    #均值归一化
    data1 = (data1 - data1.mean())/data.std()
    # 简洁的语句实现了标准化变换，类似地可以实现任何想要的变换。
    # data.mean(axis=0) 输出矩阵为一行,求每列的平均值,同理data.mean(axis=1) 输出矩阵为一列,求每行的平均值
    # data.std(axis=0) 输出矩阵为一列,求每列的标准差,同理data.std(axis=1) 输出矩阵为一列,求每行的标准差
    # 标准差也成为标准偏差,表示数据的离散程度,和标准差大小成反比
    col1 = pd.Series(np.ones(data1.shape[0]))
    #对象 不是数组 也不是矩阵？
    data1.insert(o,'ones',col1)
    cols = data1.shape[1]
    X1 = np.matrix(data1.iloc[:,0:cols - 1].values)
    #x1 = np.matrix(data2.iloc[:,:])
    y1 = np.matrix(data1.iloc[:,cols-1:cols].values)
    theta1 = np.matrix([0,0,0])
    #构建 自变量 因变量（预测数据） 矩阵
    #构建函数矩阵

    #得出预测数据 画图
    theta1,cost1,all_theta1 = gradient_descent(X1,y1,theta1)
    fig1,ax1 = plt.subplots(figsize=(12,8))
    ax1.plot(np.arange(1000),cost1,'r',label='multierror')
    ax.set_title('multi varible linear regression')
    plt.show()

def main():
    """
    1. 画出 data1 散点图
    这里我用了seaborn库画了一个粗糙的模拟图 另外用scatter方法画了图
    2. 画出 data1 只有一个变量的线性回归图像
    用了matplotlib.pyplot来画图
    需要 自变量矩阵因变量矩阵 自变量名称 因变量名称 表头
    自变量矩阵采用导入数据转换成矩阵的形式
    因变量矩阵 首先写出代价函数 再写出梯度下降方法
    这其中巧妙的是 梯度下降方法中的 theta 采用 数组
    并把所有的theta全存到一个大的矩阵当中
    还把所有的cost存到一个大的矩阵当中
    自已一直所困扰的是 一直没有具体思路
     一旦把其全部转化为矩阵运算 转化为矩阵运算一切迎刃而解
     同样的不只是有theta迭代中用到 用theta预测输出值也用到了
     包括 自变量 也是简单的用到了一次
    3. 画出 1000 次迭代以后的 cost 值
    4. 画出 多变量函数 cost 值的变化
    :return:
    """














