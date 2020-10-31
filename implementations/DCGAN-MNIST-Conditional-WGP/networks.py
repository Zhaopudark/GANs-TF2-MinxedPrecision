"""
数据集特有信息 如维度 或者其他等 都可以放入到该层实现
"""
import tensorflow as tf 
import blocks as blocks
###################################################################
class Generator(tf.keras.Model):
    def __init__(self,**kwargs):
        super(Generator,self).__init__()
        self.block_list = []
        base_v = 64
        self.block_list.append(blocks.vec2img(units=7*7*base_v*4,**kwargs))
        self.block_list.append(blocks.uk_5s1_2D(filters=base_v*2,**kwargs))
        self.block_list.append(blocks.uk_5s2_2D(filters=base_v,**kwargs))
        self.block_list.append(blocks.lats_up(filters=1,activation="sigmoid",**kwargs))
        self.last_activitation=tf.keras.layers.Activation('linear',dtype='float32')
    def build(self,input_shape,label_shape):
        for i,item in enumerate(self.block_list):
            if (i+1) == 1:
                input_shape[-1] += label_shape[-1]
                item.build(input_shape=input_shape)
                input_shape = [None,7,7,256]
            else:
                item.build(input_shape=input_shape)
        self.built = True
    def call(self,x,label,training=True):
        in_put = x
        for i,item in enumerate(self.block_list):
            if (i+1) == 1:
                out_put = item(in_put,label,training=training)
                in_put = tf.reshape(out_put,shape=[-1,7,7,256])
            else:
                out_put = item(in_put,training=training)
                in_put = out_put
        out_put = self.last_activitation(in_put)
        return out_put

class Discriminator(tf.keras.Model):
    def __init__(self,use_sigmoid=True,**kwargs):
        super(Discriminator,self).__init__()
        self.block_list=[]
        base_v = 64
        self.block_list.append(blocks.ckd_5s2_2D(filters=base_v,**kwargs))#1
        self.block_list.append(blocks.ckd_5s2_2D(filters=base_v*2,**kwargs))
        if use_sigmoid:
            self.block_list.append(blocks.Flatten_Dense_Concat(units=50,activation="relu",**kwargs))
            self.block_list.append(blocks.Flatten_Dense(units=1,activation="sigmoid",**kwargs))
        else:
            self.block_list.append(blocks.Flatten_Dense_Concat(units=50,activation="relu",**kwargs))
            self.block_list.append(blocks.Flatten_Dense(units=1,activation=None,**kwargs))
        self.last_activitation=tf.keras.layers.Activation('linear',dtype='float32')
    def build(self,input_shape,label_shape):
        for i,item in enumerate(self.block_list):
            if (i+1) in [1,2,4]:
                item.build(input_shape=input_shape)
            else:
                item.build(input_shape=input_shape,label_shape=label_shape)
        self.built = True
    def call(self,x,label,training=True):
        in_put = x
        for i,item in enumerate(self.block_list):
            if (i+1) in [1,2,4]:
                out_put = item(in_put,training=training)
                in_put = out_put
            else:
                out_put = item(in_put,label,training=training)
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
    g.build(input_shape=[None,100],label_shape=[10])
    z1 = tf.random.normal(shape=[128,100])
    label = tf.random.normal(shape=[128,10])
    print(g(z1,label).shape)
    print(g(z1,label).dtype)
    print(len(g.trainable_variables))

    d = Discriminator(name="dd")
    d.build(input_shape=[None,28,28,1],label_shape=[10])
    z2 = tf.random.normal(shape=[128,28,28,1])
    print(d(z2,label).shape)
    print(d(z2,label).dtype)
    print(len(d.trainable_variables))
    g.summary()
    d.summary()
    tmp = g(z1,label).numpy()
    print(tmp.min(),tmp.max())
    import time
    start = time.time()
    for _ in range(1000):
        g(z1,label)
        g(z1,label)
        d(z2,label)
        d(z2,label)
    print(time.time()-start)