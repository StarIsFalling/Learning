import tensorflow as tf
"""
#神经网络实例1

#Numpy是一个科学计算的工具包，这里通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8

#定义神经网络的参数，这里还是沿用3.4.2小节中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2, 3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3, 1],stddev=1,seed=1))

#在shape的一个维度上使用None可以方便实用不同的batch大小。在训练时需要把数据分成比较小的batch，但是在测试时，可以一次性使用全部的数据。
#当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
x = tf.placeholder(tf.float32,shape=(None, 2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None, 1),name="y-input")

#定义神经网络前向传播的过程。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播的算法。
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ *  tf.log(tf.clip_by_value(y,1e-10,1.0))
    +(1-y)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size =128
X = rdm.rand(dataset_size,2)

#定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），而其他为负样本（比如零件不合格）。和Tensorflow游乐场中
#的表示法不太一样的地方时，在这里使用0来便是负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法。
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

#创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))
    
    #设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        
        #通过选取的样本神经网络并更新参数
        sess.run(train_step,feed_dict={x: X[start:end],y_: Y[start:end]})
        if i % 1000 == 0:
            #每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x: X,y_: Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
    
"""
"""
from numpy.random import RandomState
batch_size = 8
#训练神经网络
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_) * loss_more,(y_-y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x: X[start:end],y_:Y[start:end]})
        print(sess.run(w1))

"""
"""
#5层神经网络带L2正则化的损失函数的计算方法
def get_weight(shape,lamb):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合。
    #这个函数的第一个参数“losses"是集合的名字，第二个参数是要加入这个集合的内容。
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamb)(var))
    #返回生成的变量
    return var


x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
batch_size = 8
#定义了每一层网络中节点的个数
layer_dimension = [2,10,10,10,1]
#神经网络的层数
n_layers = len(layer_dimension)

#这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
#当前层的节点个数
in_dimension = layer_dimension[0]

#通过一个循环来生成5层全连接的神经网络结构
for i in range(1,n_layers):
  # layer_dimension[i]为下一层的节点个数。
    out_dimension = layer_dimension[i]
    #生成当前层中权重的变量，并将这个变量的L2正则化损失加上计算图上的集合。
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    #使用ReLU激活函数
    cur_layer =  tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    #进入下一层之前将下一层的节点个数更新为当前层节点个数。
    in_dimension = layer_dimension[i]

#在定义神经网络前向传播的同时已经将所有的L2正则化损失加上了图上的集合，
#这里只需要计算刻画模型在训练数据上表现的损失函数。
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
#将均方误差损失函数加入损失集合
tf.add_to_collection('losses',mse_loss)
#get_collection返回一个列表，这个列表是所有这个集合中的元素。在这个样例中，这些元素就是损失函数的不同部分，将他们加起来就可以得到最终的损失函数
loss = tf.add_n(tf.get_collection("losses"))
"""
"""
#ExponentialMovingAverage的使用

#定义一个变量用于计算滑动平均，这个变量的初始值为0.注意这里手动指定了变量的类型为tf.float32,因为所有需要计算滑动平均的变量必须是实数型。
v1 = tf.Variable(0,dtype=tf.float32)
#这里step变量模拟神经网络中迭代的轮数，可以用来动态控制衰减率
step = tf.Variable(0,trainable=False)

#定义一个滑动平均的类（class).初始化时给定了衰减率（0.99）和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99,step)
#定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个列表时，这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #通过ema.average(v1)获取滑动平均之后的取值。在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run([v1,ema.average(v1)]))   #输出[0.0，0.0]

    #更新变量v1的值为5
    sess.run(tf.assign(v1,5))
    #更新v1的滑动平均值，衰减率为min(0.99,(1+step)/(10+step)=0.1）=0.1，所以v1的滑动平均值为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))  #输出[5.0,4.5]
    
    #更新step的值为10000
    sess.run(tf.assign(step,10000))
    #更新v1的值为10
    sess.run(tf.assign(v1,10))
    #更新v1的滑动平均值，衰减率为min(0.99,(1+step)/(10+step)≈0.999)=0.99，所以v1的滑动平均值为0.99*4.5+0.01*10=4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))  #输出[10.0,4.555]
    
    #再次更新滑动平均值，得到的新滑动平均值为0.99*4.555+0.01*10=4.60495
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))  #输出[10.0,4.60945]
"""
"""
#mnist数据集的使用
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)

print("Training data size:",mnist.train.num_examples)

print("Validating data size:",mnist.validation.num_examples)

print("Testing data size:",mnist.test.num_examples)

print("Example training data:",mnist.train.images[0])

print("Example training data lable:",mnist.train.labels[0])
"""

from tensorflow.examples.tutorials.mnist import input_data

#mnist数据集相关的常熟
INPUT_NODE = 784  #输入层的节点数。对于MNIST数据集，这个就等于图片的像素。
OUTPUT_NODE = 10  #输出层的节点数。这个等于类别的数目。因为在MNIST数据集中需要区分的是0-9这10个数字，所以这里输出层的节点数为10.

#配置神经网络的参数
LAYER1_NODE = 500  #隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例。这个隐藏层有500个节点。
BATCH_SIZE = 50    #一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降。
LEARNING_RATE_BASE = 0.8      #基础的学习率
LEARNING_RATE_DECAY = 0.99    #学习率的衰减率
REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则化项在损失函数中的系数。
TRANING_STEPS = 30000         #训练轮数
MOVING_AVERAGE_DECAY = 0.99   #滑动平均衰减率。

#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里定义了一个使用ReLU激活函数的三层全连接神经网络。通过加入隐藏层实现了
#多层网络结构，通过ReLU激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型。
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里使用了ReLU激活函数。
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)

        #计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因
        #为预测时使用的时不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果的计算没有影响。于是在计算整个神经网络的前向传播时可以不加
        #入最后的softmax层。
        return tf.matmul(layer1,weights2) + biases2

    else:
        #首先使用avg_class.average函数来计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算在当前参数下神经网络前向传播的结果。这里给出的时用于计算滑动平均的类为None，所以函数不会使用参数的滑动平均值
    y = inference(x,None,weights1,biases1,weights2,biases2)

    #定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False)。在使用TensorFlow训练神经网络
    #时，一般会将代表训练轮数的变量指定为不可训练的变量。
    global_step = tf.Variable(0,trainable=False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章中介绍过给定训练轮数的变量可以加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如global_step)就不需要了。tf.trainable_variables返回的就是图上集合。
    #GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定trainable=False的参数。
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用了滑动平均之后的前向传播结果。第4章中介绍过滑动平均不会改变变量本身的取值，而是会维护一个影子变量来记录其滑动平均值。所以当需要使用这个
    #滑动平均值时，需要明确调用average函数。
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了TensorFlow中提供的sparse_softmax_cross_entropy_with_logits函数来计算
    #交叉熵。当分类问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST问题的图片中只包含了0~9中的一个数字，所以可以使用这个函数来计算
    #交叉熵损失。这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。因为标准答案是一个长度为10的一维数组，而
    #该函数需要提供的是一个正确答案的数字，所以需要使用tf.argmax函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数。
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项。
    regularizion = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss =cross_entropy_mean + regularizion
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,   #基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减。
        global_step,          #当前迭代的轮数
        mnist.train.num_examples/BATCH_SIZE,  #过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY)  #学习率衰减速度
    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失。
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了
    #tf.control_dependencies和tf.group两种机制。下面两行程序和train_op = tf.group(train_step,variables_averages_op)时等价的。
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')

    #检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y,1)计算每一个样例的预测答案。其中average_y是一个batch_size * 10
    #的二维数组，每一行表示一个样例的前向传播结果。tf.argmax的第二个参数“1”表示选取最大值的操作仅在第一个维度中进行，也就是说，只在每一行选取最大值对应
    #的下表。于是得到的结果是一个长度为batch的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。tf.equal判断两个张量的每一维是否相等，
    #如果相等返回True，否则返回False。
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
#这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        #准备测试数据。在真实的应用中，这部分数据在训练时时不可见的，这个数据只是作为模型优劣的最后评价标准。
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        #迭代地训练神经网络
        for i in range(TRANING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
            #计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更
            #小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存溢出的错误。
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy "
                      "using average model is %g" % (i,validate_acc))
          #     test_acc = sess.run(accuracy, feed_dict=test_feed)
          #     print("After %d training step(s),validation accuracy using average "
          #           "model is %g,test accuracy using average model is %g" %
          #            (i,validate_acc,test_acc))
          #  比较不同迭代论述下滑动平均模型在验证数据集和测试数据集上的正确率
            #产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs, ys =mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average"
              "model is %g"% (TRANING_STEPS,test_acc))
#主程序入口
def main(argv=None):
    #声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)

#TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数。
if __name__== "__main__":
    tf.app.run()