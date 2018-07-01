#/usr/bin/env python
# -*- coding:utf-8 -*-

"""
  测试filter函数
"""
#  在3.0中 filter结果为iterator；2.7中则为list。所以如求素数的代码，循环过后就出问题，因为list没有next
"""
def not_empty(s):
    return s and s.split()    #""=false,所以此处判定结果ok

print list(filter(not_empty,['A','','B','C',' '])) #此处出了点小问题，原题有None,None.split()会报错，python3.0则不会
#引申，filter(None,a)会保留a中所有为真的元素,''、0等则会被剔除
"""

"""

def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n

def _not_divisible(n):
    return lambda x:x % n >0
    
def primes():
    yield 2
    it = _odd_iter()
    while True:
        n = it.next()
        yield n
        it = filter(_not_divisible(n),it)
        

for n in primes():
    if n < 1000:
        print(n)
    else:
      break
"""
def is_palindrome(n):
    string=str(n)
    length = len(string)
    if length == 1:
        return True
    else:
        for i in range(0,length//2+1):
            if string[i] != string[length-i-1]: #小心，string索引最大为length-1，不是length
                return False
        return True

output = filter(is_palindrome,range(1,1000))
print('1~1000:', list(output))
if list(filter(is_palindrome, range(1, 200))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]:
    print('测试成功!')
else:
    print('测试失败')
