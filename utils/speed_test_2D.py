import tensorflow as tf
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
# import basemodels.BaseLayers_UnMixed as BaseLayers
import basemodels.BaseLayers_Mixed as BaseLayers
###########################################################Generator blocks###########################################################
class c7s1_k_2D(tf.keras.Model):
    def __init__(self,filters,sn=False,use_bias=False,activation="relu",**kwargs):
        super(c7s1_k_2D,self).__init__()
        self.filters_zp = filters
        self.l1_conv2d = BaseLayers.Conv2D(filters=filters,kernel_size=[7,7],strides=[1,1],padding="REFLECT",use_bias=use_bias,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.InstanceNorm2D()
        if activation.lower()=="relu":
            self.l3_activation = BaseLayers.ReLU()
        elif activation.lower()=="sigmoid":
            self.l3_activation = tf.keras.layers.Activation("sigmoid")
        elif activation.lower()=="tanh":
            self.l3_activation = tf.keras.layers.Activation("tanh")
        else:
            raise ValueError("Un supported activation "+activation)
        
    def build(self,input_shape):
        self.l1_conv2d.build(input_shape=input_shape)
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv2d(x,training=training)
        # x = self.l2_norm(x,training=training)
        # y = self.l3_activation(x,training=training)
        # return y
        return x

physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# a = c7s1_k_2D(6,sn=True,use_bias=True)
# a.build(input_shape=[None,256,256,3])
# w = tf.random.normal([1,256,256,3])
# import time
# for __ in range(5):
#     start = time.time()
#     for _ in range(1000):
#         y = a(w)
#     print(time.time()-start)

l1_conv2d = BaseLayers.Conv2D(filters=64,kernel_size=[7,7],strides=[1,1],padding="REFLECT",use_bias=False,sn=False)
# l1_conv2d = tf.keras.layers.Conv2D(filters=64,kernel_size=[7,7],strides=[1,1],padding="VALID",use_bias=False)
l1_conv2d.build(input_shape=[None,256,256,8])
w = tf.random.normal([8,256,256,8])
import time
for __ in range(5):
    start = time.time()
    for _ in range(1000):
        y = l1_conv2d(w)
    print(time.time()-start)

# l1_conv2d = tf.keras.layers.Dense(4096)
# l1_conv2d.build(input_shape=[None,784])
# w = tf.random.normal([4096,784])
# import time
# for __ in range(5):
#     start = time.time()
#     for _ in range(1000):
#         y = l1_conv2d(w)
#     print(time.time()-start)    