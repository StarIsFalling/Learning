# Learning
Some exersices of learing and notes.
教材：<TensorFlow实战Google深度学习框架> 郑泽云、梁博文、顾思宇著 （第2版）



some mistakes about this book:
1、pg21 表1-2 中，如今tensorflow也支持windows
2、pg89 lambda在python中是一个函数名字，不能设为参数
3、pg96 给定地址中没有下载好的数据，并不会自动下载数据，只能创建空文件夹，需要自己从前文给的网站中把资料下载好。此问题不知道是不是因为时windows系统的问题。类似同一问题出现在pg101的程序中。
4、pg100 从优化损失函数开始到主程序之前的程序段都少了4格缩进格，此段程序都在train函数中。
5、pg112 代码会生成4个文件而不是三个。.meta保存了计算图结构，.index保存了当前参数名，.data保存了当前参数值，checkpoint保存记录信息，通过它可以定位最新保存的类型。在5.4.2节中则提出了修改。这是由于tensorflow版本造成的，0.11版本后的都是四个文件，本书第二版主要针对的是1.4版本，而不是第一版的0.9.
5.5、【不是错误，但在idle中无法实现tensorflow模型持久化，无论是保存还是加载都会报错】
6、 pg125 文字段tensorflow拼写错误。
6.5 pg127中依据PEP8,最好将if regularizer != NONE 改为 if regularizer is not NONE
7、 pg163中，file_list = [] 应该放在for extension in extensions里，不然图片重复计算。另外，在window下，文件扩展名不区分大小写，所以，extensions里可以去掉大写部分。
