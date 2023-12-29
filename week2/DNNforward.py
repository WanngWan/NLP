#coding:utf8

"""
import torch
import torch.nn as nn
import numpy as np

numpy手动实现模拟一个线性层


#搭建一个2层的神经网络模型（3-5-2）
#每层都是线性层
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) #w：3 * 5
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 5 * 2

    def forward(self, x):
        x = self.layer1(x)   #shape: (batch_size, input_size) -> (batch_size, hidden_size1) 
        y_pred = self.layer2(x) #shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2) 
        return y_pred

#自定义模型
class DiyModel:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        hidden = np.dot(x, self.w1.T) + self.b1 #1*5
        y_pred = np.dot(hidden, self.w2.T) + self.b2 #1*2
        return y_pred



#随便准备一个网络输入
x = np.array([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])
#建立torch模型
torch_model = TorchModel(3, 5, 2)

print(torch_model.state_dict())

print("-----------")
#打印模型权重，权重为随机初始化
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
print(torch_model_w1, "torch w1 权重")
print(torch_model_b1, "torch b1 权重")
print("-----------")
print(torch_model_w2, "torch w2 权重")
print(torch_model_b2, "torch b2 权重")
print("-----------")
#使用torch模型做预测
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)


# #把torch模型权重拿过来自己实现计算过程
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
# #用自己的模型来预测
y_pred_diy = diy_model.forward(np.array(x))
print("diy模型预测结果：", y_pred_diy)
"""

##自由练习区
import torch
import torch.nn as nn
import numpy as np

#1、使用nn构建一个（3-5-2）两层的模型
class Model00(nn.Module):
    def __init__(self, inp_size, hid_size1, hid_size2):
        super(Model00, self).__init__()
        self.layer1 = nn.Linear(inp_size, hid_size1)
        self.layer2 = nn.Linear(hid_size1, hid_size2)

    def forward(self, x):
        x = self.layer1(x)
        y_pred = self.layer2(x)
        return y_pred

#2、使用numpy自定义模型
class DiyModel():
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        x_hid1 = np.dot(x, self.w1.T) + self.b1
        y_pred = np.dot(x_hid1, self.w2.T) + self.b2
        return y_pred

#3、准备数据
x = np.array([[1., 2.5, 4.3],
              [3.2, 6.2, 5.]])

#4 使用torch模型进行计算
#4.1 简历torch模型
Torch_Model = Model00(3, 5, 2)

print("模型整体状态：", Torch_Model.state_dict()) #模型的权重、偏置参数每次都会随机初始化

model_w1 = Torch_Model.state_dict()['layer1.weight'].numpy()
model_b1 = Torch_Model.state_dict()['layer1.bias'].numpy()
model_w2 = Torch_Model.state_dict()['layer2.weight'].numpy()
model_b2 = Torch_Model.state_dict()['layer2.bias'].numpy()
print("=" * 90)
print("layer1 weight:", model_w1)
print("layer1 bias:", model_b1)
print("=" * 90)
print("layer2 weight:", model_w2)
print("layer2 bias:", model_b2)
print("=" * 90)

#4.2 用模型计算
x_tensor = torch.FloatTensor(x)
y_pred = Torch_Model.forward(x_tensor)
print("torch模型预测输出：", y_pred)
print("=" * 90)
#5、使用自定义模型进行计算
diy_model = DiyModel(model_w1, model_b1, model_w2, model_b2)
y_pred_diy = diy_model.forward(x)
print("DIY模型预测输出：", y_pred_diy)