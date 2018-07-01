#! /usr/bin/env python
# -*- coding=utf-8 -*-

"""
练习map和reduce函数
"""
from functools import reduce

def str2float(s):
    a1,a2 = s.split('.') #分隔符split()
    def string(s):
        dight = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}#字典用{}
        return dight[s]
    def num1(x,y):
        return x*10+y
    def num2(x,y):
        return x/10.0+y  #此处高能注意！x为整数，若x/10，则结果为0，所以此处采用/10.0，计算结果为小数
    zheng = reduce(num1,map(string,a1))
    xiao = reduce(num2,map(string,reversed(a2))) #字符串反转reversed(),注意reversed反转结果为iterator,要用list显示出
    return num2(xiao,zheng)

print('str2float(\'123,456\') =',str2float('123.456'))
if abs (str2float('123.456') - 123.456) < 0.000001:
    print "测试成功!"
else:
    print "测试失败"
    
