"""
日期: 年12月04日
"""
# with open("test.txt", "a") as f:
#     f.write("\n这是fddf")  # 这句话自带文件关闭功能，不需要再写f.close()
# import torch as t
# from torch.autograd import Variable as v
#
# a = v(t.FloatTensor([2, 3]), requires_grad=True)
# b = a + 3
# c = b * b * 3
# out = c.mean()
# out.backward(retain_graph=True) # 这里可以不带参数，默认值为‘1’，由于下面我们还要求导，故加上retain_graph=True选项
#
# print(a.grad) # tensor([15., 18.])

import os

os.environ['TF_KERAS']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
import numpy as np

plt.ion()

n = 100  #噪声数量
batch_size = 20

x = np.linspace(0, 10, n)
random.shuffle(x)
noise = np.random.randn(n)
y = 2.5 * x + 0.8 + 2.0 * noise



train_x, train_y = [[x[j] for j in range(i, i+batch_size)] for i in range(0, len(x), batch_size)], [[y[j] for j in range(i, i+batch_size)] for i in range(0, len(y), batch_size)]

model = Sequential()
model.add(layers.Dense(units=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

w, b = model.layers[0].get_weights()

plt.scatter(x, y, s=1)
plt.plot([0, 10], [b, 10*w+b])
plt.pause(0.001)

for batch_x, batch_y in zip(train_x, train_y):
    time.sleep(5)
    batch_x, batch_y = np.array(batch_x), np.array(batch_y)
    model.train_on_batch(batch_x, batch_y)
    w, b = model.layers[0].get_weights()
    print(w, b)
    plt.cla()
    plt.scatter(x, y, s=1)
    plt.plot([0, 10], [b, 10*w+b])
    plt.pause(0.001)

plt.show()

