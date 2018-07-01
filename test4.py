#! /usr/bin/env python
# -*- coding=utf-8 -*-

"""
  测试sorted函数

L=[('Bob',75),('Adam',92),('Bart',66),('Lisa',88)]
def by_name(t):
    return t[0].lower()

def by_score(t):
    return -t[1]

print sorted(L,key=by_name)
print sorted(L,key=by_score)
"""
'''
  闭包
'''
def CreateCounter():
    i = [0]
    def counter():
        i[0] =i[0] + 1  #这一行不能放到return i[0]后面，用i[0]是因为int(i)无法调用,（外部变量），在python3.0中有函数nonlocal解决调用外部变量的问题
        return i[0]
    return counter

counterA = CreateCounter()
print(counterA(),counterA(),counterA(),counterA())
counterB=CreateCounter()
if [counterB(),counterB(),counterB(),counterB()] == [1,2,3,4]:
    print '测试通过!'
else:
    print "测试失败"
