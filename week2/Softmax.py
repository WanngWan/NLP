#coding:utf8
import torch
import numpy

'''
softmax的计算
'''

def softmax(x):
    res = []
    for i in x:
        res.append(numpy.exp(i))
    res = [r / sum(res) for r in res]
    return res

#e的1次方
print(numpy.exp(1))

x = [[1,2,3,4],
     [4,3,2,1]]
#torch实现的softmax
print(torch.softmax(torch.Tensor(x), 0))
print(torch.softmax(torch.Tensor(x), 1))

#自己实现的softmax
print(softmax(x))