# Learning
Some exersices of learing and notes.
教材：<TensorFlow实战Google深度学习框架> 郑泽云、梁博文、顾思宇著 （第2版）



some mistakes about this book:
1、pg21 表1-2 中，如今tensorflow也支持windows
2、pg89 lambda在python中是一个函数名字，不能设为参数
3、pg96 给定地址中没有下载好的数据，并不会自动下载数据，只能创建空文件夹，需要自己从前文给的网站中把资料下载好。此问题不知道是不是因为时windows系统的问题。类似同一问题出现在pg101的程序中。
4、pg100 从优化损失函数开始到主程序之前的程序段都少了4格缩进格，此段程序都在train函数中。
5、pg112 代码会生成4个文件而不是三个。.meta保存了计算图结构，.index保存了当前参数名，.data保存了当前参数值，checkpoint保存记录信息，通过它可以定位最新保存的类型。
6、【不是错误，但在idle中无法实现tensorflow模型持久化，无论是保存还是加载都会报错】
