"""
原始GAN所有层都采用了bias
"""
import tensorflow as tf
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
# import basemodels.BaseLayers_UnMixed as BaseLayers
import basemodels.BaseLayers_Mixed as BaseLayers
###########################################################Generator blocks###########################################################
class Dense(tf.keras.Model):
    def __init__(self,units,sn=False,activation=None,**kwargs):
        super(Dense,self).__init__()
        self.units_zp = units
        self.l1_dense = BaseLayers.Dense(units=units,use_bias=True,sn=sn,**kwargs)
        if activation == None:
            self.l2_activation = tf.keras.layers.Activation("linear")
        elif activation.lower()=="relu":
            self.l2_activation = BaseLayers.ReLU()
        elif activation.lower()=="sigmoid":
            self.l2_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l2_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation "+activation)
    def build(self,input_shape):
        self.l1_dense.build(input_shape=input_shape)
        input_shape[-1]=int(self.units_zp)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_dense(x,training=training)
        y = self.l2_activation(x,training=training)
        return y
class Flatten_Dense(tf.keras.Model):
    def __init__(self,units,sn=False,activation=None,**kwargs):
        super(Flatten_Dense,self).__init__()
        self.units_zp = units
        self.l0_flatten = tf.keras.layers.Flatten()
        self.l1_dense = BaseLayers.Dense(units=units,use_bias=True,sn=sn,**kwargs)
        if activation == None:
            self.l2_activation = tf.keras.layers.Activation("linear")
        elif activation.lower()=="relu":
            self.l2_activation = BaseLayers.ReLU()
        elif activation.lower()=="sigmoid":
            self.l2_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l2_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation "+activation)
    def build(self,input_shape):
        self.l1_dense.build(input_shape=input_shape)
        input_shape[-1]=int(self.units_zp)
        self.built = True
    def call(self,x,training=True):
        x = self.l0_flatten(x,training=training)
        x = self.l1_dense(x,training=training)
        y = self.l2_activation(x,training=training)
        return y
###########################################################Discriminator blocks###########################################################
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    # a = Dense(100,sn=True,activation="relu")
    # a.build(input_shape=[None,784])
    # w = tf.random.normal([128,784])
    # y = a(w)
    # print(y.shape,y.dtype)

    # a = Flatten_Dense(100,sn=True,activation=None)
    # a.build(input_shape=[None,784])
    # w = tf.random.normal([128,784])
    # y = a(w)
    # print(y.shape,y.dtype)
    a=Dense(units=128,activation="relu")
    a.build(input_shape=[None,100])
    b=Dense(units=784,activation="sigmoid")
    b.build(input_shape=[None,128])
    c=tf.keras.layers.Activation('linear',dtype='float32')
    z= tf.random.uniform(shape=[128,100],minval=-1.0,maxval=1.0)
    y1 = c(b(a(z)))
    out = tf.reduce_mean(y1)
    print(out.numpy())
    z= tf.random.uniform(shape=[128,100],minval=0.0,maxval=1.0)
    y2 = c(b(a(z)))
    out = tf.reduce_mean(y2)
    print(out.numpy())

    y1 = tf.reshape(y1,shape=[-1,28,28,1])
    y2 = tf.reshape(y2,shape=[-1,28,28,1])
    a=Flatten_Dense(units=128,activation="relu")
    a.build(input_shape=[None,784])
    b=Dense(units=1,activation="sigmoid")
    b.build(input_shape=[None,128])
    c=tf.keras.layers.Activation('linear',dtype='float32')

    out = tf.reduce_mean(c(b(a(y1))))
    print(out.numpy())
    out = tf.reduce_mean(c(b(a(y2))))
    print(out.numpy())

    print(a.trainable_variables[0].numpy())