import tensorflow as tf
import os 
import sys
import time
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
# import basemodels.BaseLayers_UnMixed as BaseLayers
import basemodels.BaseLayers_Mixed as BaseLayers
###########################################################Generator blocks###########################################################


physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

class c7s1_k_3D(tf.keras.Model):
    """ 
    A 7x7x7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
    Args:
        input: 5D tensor [Batchs,D,Width,Height,Channers],[B,D,W,H,C]
        k: integer, number of filters (output depth)
        norm: 'instance' or 'batch' or None
        activation: 'relu'
        training: boolean or BoolTensor
    Returns:
        5D tensor [B,D,W,H,C]
    """
    def __init__(self,filters,sn=False,use_bias=False,activation="relu",**kwargs):
        super(c7s1_k_3D,self).__init__()
        self.filters_zp = filters
        self.l1_conv3d = BaseLayers.Conv3D(filters=filters,kernel_size=[7,7,7],strides=[1,1,1],padding="REFLECT",use_bias=use_bias,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.InstanceNorm3D()
        if activation.lower()=="relu":
            self.l3_activation = BaseLayers.ReLU()
        elif activation.lower()=="sigmoid":
            self.l3_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l3_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation "+activation)
    def build(self,input_shape):
        """
        build 目的是确定训练参数 无参数层本就不应该被build
        合成的模型本身也不需要再次build
        super(c7s1_k_3D,self).build(input_shape)
        self.built=True 就不会被二次build
        """
        self.l1_conv3d.build(input_shape=input_shape)
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv3d(x,training=training)
        x = self.l2_norm(x,training=training)
        # y = self.l3_activation(x,training=training)
        return x
a = c7s1_k_3D(64,sn=True,use_bias=True,activation="relu")
a.build(input_shape=[None,128,128,128,8])
w = tf.random.normal([1,128,128,128,8])
for __ in range(10):
    start = time.time()
    for _ in range(10):
        y = a(w)
    print(time.time()-start)

# print(l1_conv3d(w).shape)
# 
# for __ in range(5):
#     start = time.time()
#     for _ in range(10):
#         y = l1_conv3d(w)
#     print(time.time()-start)
# l1_conv3d = BaseLayers.Conv3D(filters=64,kernel_size=[7,7,7],strides=[1,1,1],padding="REFLECT",use_bias=True,sn=True)
# l1_conv3d.build(input_shape=[None,128,128,128,8])
# w = tf.random.normal([1,128,128,128,8])
# print(l1_conv3d(w).shape)
# import time
# for __ in range(5):
#     start = time.time()
#     for _ in range(10):
#         y = l1_conv3d(w)
#     print(time.time()-start)

# l1_conv2d = tf.keras.layers.Dense(4096)
# l1_conv2d.build(input_shape=[None,784])
# w = tf.random.normal([4096,784])
# import time
# for __ in range(5):
#     start = time.time()
#     for _ in range(1000):
#         y = l1_conv2d(w)
#     print(time.time()-start)    