# coding:utf8

"""
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


# 基于pytorch框架编写模型训练
# 实现一个自行构造的找规律(机器学习)任务
# 规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数, 计算预测输出
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失， 计算预测输出与真实输出的距离

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval() #将模型设置成测试模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测等价于 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器, 用于进行梯度更新，寻找最小值。
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() #将模型设置成训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss, 等价于model.forward(x, y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果，用于验证
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，等价于model（x）
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_vec)

"""

# 基于pytorch框架编写模型训练
# 实现一个自行构造的找规律(机器学习)任务
# 规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = nn.functional.sigmoid
        self.loss = nn.functional.mse_loss


    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def BuildData():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0

def BuildDatabase(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = BuildData()
        X.append(x)
        Y.append([y])
    # print("正样本数量：%d, 负样本数量：%d" % (np.sum(Y), sample_num-np.sum(Y)))
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

def Evaluate(model):
    eva_sample_num = 200
    val_x, val_y = BuildDatabase(eva_sample_num)
    correct_n, incorrect_n = 0, 0

    model.eval()
    with torch.no_grad():
        y_pred = model(val_x)
        for y_pre, y_true in zip(y_pred, val_y):
            if y_pre > 0.5 and y_true == 1:
                correct_n += 1
            elif y_pre <= 0.5 and y_true == 0:
                correct_n += 1
            else:
                incorrect_n += 1

    return float(correct_n/(correct_n+incorrect_n))



def main():
    batch_size = 20
    sample_num = 5000
    lr = 0.001
    log = []
    epoch_num = 20
    input_size = 5

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    X, Y = BuildDatabase(sample_num)

    print("==================训练开始================")
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []
        for j in range(sample_num//batch_size):
            train_x = X[j * batch_size : (j + 1) * batch_size]
            train_y = Y[j * batch_size : (j + 1) * batch_size]
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            epoch_loss.append(loss.item())

        acc = Evaluate(model)
        print("训练第%d轮，损失值为：%f, 准确率为：%f" % (epoch+1, np.mean(epoch_loss), acc))

        log.append([acc, np.mean(epoch_loss)])

    torch.save(model.state_dict(), '01model_state.pth')
    # torch.save(model, 'model.pt')

    print("=================训练结束！===================")


    plt.title("The Performance of Model")
    plt.plot(range(len(log)), [l[0] for l in log], label = 'acc')
    plt.plot(range(len(log)), [l[1] for l in log], label = 'loss')
    plt.legend(['acc', 'loss'])
    plt.show()


def PredictTest(model_path, test_data):
    input_size = 5
    # state_dict = torch.load(model_path)
    model = TorchModel(input_size)
    # model.load_state_dict(state_dict)
    model.load_state_dict(torch.load('01model_state.pth'))

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.FloatTensor(test_data))
        for x, y in zip(test_data, y_pred):
            print("测试数据：%s, 预测分类：%d, 预测准确率：%s" % (x, round(float(y)), y.item()))


if __name__ == "__main__":
    # main()

    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.19349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]

    PredictTest("model.pt", test_vec)






