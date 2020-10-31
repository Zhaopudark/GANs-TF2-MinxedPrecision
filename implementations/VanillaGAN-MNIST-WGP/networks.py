"""
需要的模型容量和数据集本身相关
从networks 开始 往上的高阶封装 
都需要结合数据集进行相应的调整
"""
import tensorflow as tf 
import blocks as blocks
###################################################################
class Generator(tf.keras.Model):
    def __init__(self,**kwargs):
        super(Generator,self).__init__()
        self.block_list = []
        base_v = 64
        self.block_list.append(blocks.Dense(units=128,activation="relu",**kwargs))
        self.block_list.append(blocks.Dense(units=784,activation="sigmoid",**kwargs))
        self.last_activitation=tf.keras.layers.Activation('linear',dtype='float32')
    def build(self,input_shape):
        for item in self.block_list:
            item.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        in_put = x
        for item in self.block_list:
            out_put = item(in_put,training=training)
            in_put = out_put
        out_put = self.last_activitation(in_put)
        out_put = tf.reshape(out_put,[-1,28,28,1])
        return out_put
class Discriminator(tf.keras.Model):
    def __init__(self,use_sigmoid=True,**kwargs):
        super(Discriminator,self).__init__()
        self.block_list=[]
        base_v = 64
        self.block_list.append(blocks.Flatten_Dense(units=128,activation="relu",**kwargs))
        if use_sigmoid:
            self.block_list.append(blocks.Dense(units=1,activation="sigmoid",**kwargs))
        else:
            self.block_list.append(blocks.Dense(units=1,activation=None,**kwargs))
        self.last_activitation=tf.keras.layers.Activation('linear',dtype='float32')
    def build(self,input_shape):
        """
        dense 对输入的特殊处理
        """
        if len(input_shape)!=2:
            buf = 1
            for i in range(1,len(input_shape),1):
                buf *= input_shape[i]
            input_shape = [None,buf]
        for item in self.block_list:
            item.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        in_put = x
        for item in self.block_list:
            out_put = item(in_put,training=training)
            in_put = out_put
        out_put = self.last_activitation(in_put)
        return out_put
if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    g = Generator(name="g_fat")
    g.build(input_shape=[None,100])
    z1 = tf.random.normal(shape=[128,100])
    print(g(z1).shape)
    print(g(z1).dtype)
    print(len(g.trainable_variables))

    d = Discriminator(name="dd")
    d.build(input_shape=[None,28,28,1])
    z2 = tf.random.normal(shape=[128,28,28,1])
    print(d(z2).shape)
    print(d(z2).dtype)
    print(len(d.trainable_variables))
    # g.summary()
    # d.summary()
    # tmp = g(z1).numpy()
    # print(tmp.min(),tmp.max())
    # import time
    # start = time.time()
    # for _ in range(1000):
    #     g(z1)
    #     g(z1)
    #     d(z2)
    #     d(z2)
    # print(time.time()-start)