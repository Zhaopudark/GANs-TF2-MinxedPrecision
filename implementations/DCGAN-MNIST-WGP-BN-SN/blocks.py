"""
构建CycleGAN需要的各个模块
层内build原则 非惰性shape计算 
虽然norm actviation dropout等没有shape需求 但是会迫使其前一层的shape必须计算输出后shape
则层上封装就可以直接获得输出shape(input_shape传递了层内shape变化)
抽象每一层最后都增加一个linear层 则linear输如shape在计算后就是原层的输出shape
"""
import tensorflow as tf
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
# import basemodels.BaseLayers_UnMixed as BaseLayers
import basemodels.BaseLayers_Mixed as BaseLayers
###########################################################Generator blocks###########################################################
class vec2img(tf.keras.Model):
    """
    后级调用需要补上reshape
    """
    def __init__(self,units,sn=False,**kwargs):
        super(vec2img,self).__init__()
        self.units_zp = units
        self.l1_dense = BaseLayers.Dense(units=units,use_bias=False,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.BatchNormalization()
        self.l3_activation = BaseLayers.LeakyReLU()
    def build(self,input_shape):
        self.l1_dense.build(input_shape=input_shape)
        input_shape[-1]=int(self.units_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.l3_activation.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_dense(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
class uk_5s1_2D(tf.keras.Model):
    def __init__(self,filters,sn=False,**kwargs):
        super(uk_5s1_2D,self).__init__()
        self.filters_zp = filters
        self.l1_conv2d = BaseLayers.Conv2DTranspose(filters=filters,kernel_size=[5,5],strides=[1,1],padding="SAME",use_bias=False,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.BatchNormalization()
        self.l3_activation = BaseLayers.LeakyReLU()
    def build(self,input_shape):
        self.l1_conv2d.build(input_shape=input_shape)
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.l3_activation.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv2d(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
class uk_5s2_2D(tf.keras.Model):
    def __init__(self,filters,sn=False,activation=None,**kwargs):
        super(uk_5s2_2D,self).__init__()
        self.filters_zp = filters
        self.l1_conv2d = BaseLayers.Conv2DTranspose(filters=filters,kernel_size=[5,5],strides=[2,2],padding="SAME",use_bias=False,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.BatchNormalization()
        if activation == None:
            self.l3_activation = tf.keras.layers.Activation("linear")
        elif activation.lower()=="relu":
            self.l3_activation = BaseLayers.ReLU()
        elif activation.lower()=="leaky_relu":
            self.l3_activation = BaseLayers.LeakyReLU()
        elif activation.lower()=="sigmoid":
            self.l3_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l3_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation"+activation)
    def build(self,input_shape):
        self.l1_conv2d.build(input_shape=input_shape)
        for i in range(2):
            input_shape[i+1] = input_shape[i+1]*2
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.l3_activation.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv2d(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y       

class lats_up(tf.keras.Model):
    def __init__(self,filters,sn=False,activation=None,**kwargs):
        super(lats_up,self).__init__()
        self.filters_zp = filters
        self.l1_conv2d = BaseLayers.Conv2DTranspose(filters=filters,kernel_size=[5,5],strides=[2,2],padding="SAME",use_bias=False,sn=sn,**kwargs)
        if activation == None:
            self.l2_activation = tf.keras.layers.Activation("linear")
        elif activation.lower()=="relu":
            self.l2_activation = BaseLayers.ReLU()
        elif activation.lower()=="leaky_relu":
            self.l2_activation = BaseLayers.LeakyReLU()
        elif activation.lower()=="sigmoid":
            self.l2_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l2_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation"+activation)
    def build(self,input_shape):
        self.l1_conv2d.build(input_shape=input_shape)
        for i in range(2):
            input_shape[i+1] = input_shape[i+1]*2
        input_shape[-1]=int(self.filters_zp)
        self.l2_activation.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv2d(x,training=training)
        y = self.l2_activation(x,training=training)
        return y  
###########################################################Discriminator blocks###########################################################
class ckd_5s2_2D(tf.keras.Model):
    def __init__(self,filters,sn=False,**kwargs):
        super(ckd_5s2_2D,self).__init__()
        self.filters_zp = filters
        self.l1_conv2d= BaseLayers.Conv2D(filters=filters,kernel_size=[5,5],strides=[2,2],padding='SAME',sn=sn,**kwargs)
        self.l2_activation = BaseLayers.LeakyReLU()
        self.l3_dropout = BaseLayers.Dropout(rate=0.3)
    def build(self,input_shape):
        self.l1_conv2d.build(input_shape=input_shape)
        for i in range(2):
            input_shape[i+1] = int(tf.math.ceil(input_shape[i+1]/2))
        input_shape[-1]=int(self.filters_zp)
        self.l2_activation.build(input_shape=input_shape)
        self.l3_dropout.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv2d(x,training=training)
        x = self.l2_activation(x,training=training)
        y = self.l3_dropout(x,training=training)
        return y
class Flatten_Dense(tf.keras.Model):
    def __init__(self,units,sn=False,activation=None,**kwargs):
        super(Flatten_Dense,self).__init__()
        self.units_zp = units
        self.l1_flatten = tf.keras.layers.Flatten()
        self.l2_dense = BaseLayers.Dense(units=units,use_bias=True,sn=sn,**kwargs)
        if activation == None:
            self.l3_activation = tf.keras.layers.Activation("linear")
        elif activation.lower()=="relu":
            self.l3_activation = BaseLayers.ReLU()
        elif activation.lower()=="sigmoid":
            self.l3_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l3_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation "+activation)
    def build(self,input_shape):
        self.l1_flatten.build(input_shape=input_shape)
        buf = 1
        for _ in range(len(input_shape)-1):
            buf *= int(input_shape.pop(-1))
        input_shape.append(buf)
        self.l2_dense.build(input_shape=input_shape)
        input_shape[-1]=int(self.units_zp)
        self.l3_activation.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_flatten(x,training=training)
        x = self.l2_dense(x,training=training)
        y = self.l3_activation(x,training=training)
        return y

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    a = vec2img(7*7*256)
    a.build(input_shape=[None,100])
    w = tf.random.normal([128,100])
    y = a(w)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))
    a = uk_5s1_2D(128)
    a.build(input_shape=[None,7,7,256])
    y = a(tf.reshape(y,[-1,7,7,256]))
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))

    a = uk_5s2_2D(64)
    a.build(input_shape=[None,7,7,128])
    y = a(y)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))

    a = lats_up(1)
    a.build(input_shape=[None,14,14,64])
    y = a(y)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))

    a = ckd_5s2_2D(64)
    a.build(input_shape=[None,28,28,1])
    y = a(y)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))

    a = ckd_5s2_2D(128)
    a.build(input_shape=[None,14,14,64])
    y = a(y)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))

    a = Flatten_Dense(1)
    a.build(input_shape=[None,7,7,128])
    y = a(y)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))


