# @Author  : 小贤
# @Time    : 2022/11/8 09:58

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#（1）获取数据，按时间间隔1h记录的电力数据
filepath = 'Slope500.csv'        # 文件的路径
data = pd.read_csv(filepath)    # 借助panda读取电量时间序列数据，两列特征值：时间和电量 'D:\coder\python_program\Books.csv'
# print(type(data))
# data[:, 1] = data[:, 1] * 100   # 将里程化为整数
print(data.head())              # head( )函数只能读取前五行数据

# （2）构建时间序列采样函数
# 通过时间序列滑动窗口 选择特征值及其对应的标签值
'''
dataset为输入的特征数据，选取用哪些特征
start_index 这么多数据选择从哪个开始，一般从0开始取序列
history_size表示时间窗口大小；若为20，代表从起始索引开始找20个样本当作x，下一个索引当作y。用多少数据来进行预测。
target_size表示需要预测的结果时窗口后的第几个时间点；0表示下一时间点的预测结果，取其当作标签；若为一个序列，预测一个序列的指标。 预测序列的话 target_size 怎么写？
indices=range(i, i+history_size) 代表窗口序列的索引，i表示每个窗口的起始位置，窗口中所有数据的索引
'''
def database(dataset, start_index, end_index, history_size, target_size):
    data = []     # 存放特征值  history_size中的历史数据用来预测。
    labels = []   # 存放目标值

    # 初始的取值片段[0:history_size]
    start_index = start_index + history_size

    # 如果不指定特征值终止索引，就取到最后一个分区前
    if end_index is None:
        end_index = len(dataset) - target_size

    # 遍历整个电力数据，取出特征及其对应的预测目标
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)     # 窗口内的所有元素的索引  history数据的长度
        # 保存特征值和标签值
        data.append(np.reshape(dataset[indices], (history_size, 1)))  # 按时间窗口把历史数据取出来
        labels.append(dataset[i + target_size])  # 预测未来几个片段的天气数据
        # print("data",data)
        # print("labels", labels)
    # 返回数据集
    return np.array(data), np.array(labels)

# 取前90%个数据作为训练集
train_num = int(len(data) * 0.8)                # 333*8 2664
# 90%-99.8%用于验证
val_num = int(len(data) * 0.9)                  # 2954
# 最后1%用于测试   剩下的数用于验证

# （3）选择特征
temp = data['Slope']                        # 获取电力数据，取AFP电量特征列作为训练的特征（单个特征）
# temp.index = data['Distence']             # 将索引改为时间序列 不用距离作为索引！！！！！！！！！！！！！
temp.plot()  # 绘图展示
print('temp', temp)
print('temp.index', temp.index)

# （4）对训练集预处理
temp_mean = temp[:train_num].mean()        # 均值
temp_std = temp[:train_num].std()          # 标准差
# 由于原始数据最大值和最小值之间相差较大，为了避免数据影响网络训练的稳定性，对训练用的特征数据进行标准化处理。
inputs_feature = (temp - temp_mean) / temp_std  # 将temp中的所有数进行归一化

# （5）划分训练集和验证集
# 窗口为20条数据，预测下一时刻气温
history_size = 20                        # 一万多的数据是20，
target_size = 0                           # target_size 为0表示下一时间点的预测结果，

# 训练集
x_train, y_train = database(inputs_feature.values, 0, train_num,  history_size, target_size)

# 验证集
x_val, y_val = database(inputs_feature.values, train_num, val_num,
                        history_size, target_size)

# 测试集
x_test, y_test = database(inputs_feature.values, val_num, None,
                          history_size, target_size)
print('train_num', train_num)
print('val_num', val_num)
print('inputs_feature.values', inputs_feature.values)
print('x_test', x_test)
print('y_test11111',y_test)
# print('x_val', x_val)
# 查看数据信息matlab实现LSTM预测NGSIM轨迹
print('x_train.shape:', x_train.shape)  # x_train.shape: (109125, 20, 1)

# （6）构造tf数据集    将划分好的numpy类型的训练集和验证集转换为tensor类型，用于网络训练。
# 训练集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))   # 加载数据
print("train_ds", train_ds)
train_ds = train_ds.shuffle(1000).batch(64)                       # 使用shuffle()函数打乱训练集数据 # 原10000 128
print("train_ds", train_ds)
# 验证集
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(64)                         # batch()函数指定每个step训练多少组数据。h的数据用于验证
print("val_ds ", val_ds)
# 查看数据信息
sample = next(iter(train_ds))                      # 借助迭代器iter()使用next()函数从数据集中取出一个batch
print('x_batch.shape:', sample[0].shape, 'y_batch.shape:', sample[1].shape)
print('input_shape:', sample[0].shape[-2:])
# x_batch.shape: (128, 20, 1) y_batch.shape: (128,)
# input_shape: (20, 1)

# 由于特征只有一个，使用一个LSTM层用于提取特征，一个全连接层用于输出预测结果。
# 构造输入层
inputs = keras.Input(shape=sample[0].shape[-2:])    # 输入
# print(inputs, "inputs ")
# 搭建网络各层
x = keras.layers.LSTM(8)(inputs)                    # 8为输出空间的维度
x = keras.layers.Activation('relu')(x)
outputs = keras.layers.Dense(1)(x)                  # 输出结果是1个
# 构造模型
model = keras.Model(inputs, outputs)
# 查看模型结构
model.summary()

# （8）模型编译
opt = keras.optimizers.Adam(lr=0.001)      # 优化器     # keras=2.2.4 写法 lr = learning_rate
model.compile(optimizer=opt, loss='mae')   # 平均误差损失

# （9）模型训练
epochs = 20           # epochs迭代次数，即全部样本数据将被“轮”多少次，轮完训练停止
# history = model.fit(train_ds, epochs=epochs, validation_data=val_ds) epochs迭代次数，即全部样本数据将被“轮”多少次，轮完训练停止
history = model.fit(x_val, y_val, epochs=epochs, validation_data=[x_val, y_val])
print("history", history)

# （10）获取训练信息
history_dict = history.history        # 获取训练的数据字典
train_loss = history_dict['loss']     # 训练集损失
val_loss = history_dict['val_loss']   # 验证集损失

# （11）绘制训练损失和验证损失
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
plt.plot(range(epochs), val_loss, label='val_loss')      # 验证集损失
plt.legend()  # 显示标签
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# （12）预测   # 最后一段的预测结果
y_predict = model.predict(x_test)  # 对测试集的特征值进行预测  y_predict类型<class 'numpy.ndarray'>[[]]  y_predict.shape (588, 1)
# x_test 等同于经过预处理后的 temp[val_num:-20].values
dates = temp[val_num:-20].index  # 获取预测时间的初始索引                     # 这个得改原来是20
# x_test, y_test = database(inputs_feature.values, val_num, None,
#                           history_size, target_size)
print("dates", dates)
print("y_test", y_test)
print("y_predict", y_predict)
# 误差值
y_error = np.squeeze(y_predict) - y_test     # np.squeeze(y_predict)将y_predict降维， y_test类型<class 'numpy.ndarray'>[] (588,)

# （13）绘制预测结果和真实值对比图
fig = plt.figure(figsize=(10, 5))
# 真实值
axes = fig.add_subplot(111)
axes.plot(dates, y_test, 'bo', label='actual')
# 预测值，红色散点
axes.plot(dates, y_predict, 'ro', label='predict')
# # # 测量值（预测值）-真实值的误差
# axes.plot(dates, y_error, 'go', label='error')
# 设置横坐标刻度
axes.set_xticks(dates[::30])
axes.set_xticklabels(dates[::30], rotation=45)

plt.legend()  # 注释
plt.grid()  # 网格
plt.show()





