"""
构建CycleGAN需要的各个模块
"""
import tensorflow as tf
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
import basemodels.BaseLayers_Mixed as BaseLayers
# import basemodels.BaseLayers_UnMixed as BaseLayers
###########################################################Generator blocks###########################################################
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
        y = self.l3_activation(x,training=training)
        return y
class dk_3D(tf.keras.Model):
    """ 
    A 3x3x3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
    Args:
        input: 5D tensor
        k: integer, number of filters (output depth)
        norm: 'instance' or 'batch' or None
        training: boolean or BoolTensor
    Returns:
        5D tensor
    """
    def __init__(self,filters,sn=False,use_bias=False,**kwargs):
        super(dk_3D,self).__init__()
        self.filters_zp = filters
        self.l1_conv3d = BaseLayers.Conv3D(filters=filters,kernel_size=[3,3,3],strides=[2,2,2],padding="REFLECT",use_bias=use_bias,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.InstanceNorm3D()
        self.l3_activation = BaseLayers.ReLU()
    def build(self,input_shape):
        self.l1_conv3d.build(input_shape=input_shape)
        #B D W H C
        #0 1 2 3 4
        for i in range(3):
            input_shape[i+1] = int(tf.math.ceil(input_shape[i+1]/2))
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv3d(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
class rk_3D(tf.keras.Model):
    """ 
    ResNet 模块要求输入维度和输出维度必须一致 不然加不起来 所以设计中间的卷积层时需要谨慎考虑参数
    A residual block that contains two 3x3 convolutional layers with the same number of filters on both layer
    Args:
        input: 5D Tensor
        k: integer, number of filters (output depth)
    Returns:
        5D tensor (same shape as input)
    """
    def __init__(self,filters,sn=False,use_bias=False,**kwargs):
        super(rk_3D,self).__init__()
        self.l1_conv3d= BaseLayers.Conv3D(filters=filters,kernel_size=[3,3,3],strides=[1,1,1],padding="REFLECT",use_bias=use_bias,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.InstanceNorm3D()
        self.l3_activation = BaseLayers.ReLU()
        self.l4_conv3d= BaseLayers.Conv3D(filters=filters,kernel_size=[3,3,3],strides=[1,1,1],padding="REFLECT",use_bias=use_bias,sn=sn,**kwargs)
        self.l5_norm = BaseLayers.InstanceNorm3D()
    def build(self,input_shape):
        self.l1_conv3d.build(input_shape=input_shape)
        self.l2_norm.build(input_shape=input_shape)
        self.l4_conv3d.build(input_shape=input_shape)
        self.l5_norm.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        y = self.l1_conv3d(x,training=training)
        y = self.l2_norm(y,training=training)
        y = self.l3_activation(y,training=training)
        y = self.l4_conv3d(y,training=training)
        y = self.l5_norm(y,training=training)
        return y+x

class n_res_blocks_3D(tf.keras.Model):
    def __init__(self,filters,sn=False,use_bias=False,n=6,**kwargs):
        super(n_res_blocks_3D,self).__init__()
        self.rs_list = []
        for i in range(n):
            self.rs_list.append(rk_3D(filters=filters,sn=sn,use_bias=use_bias,**kwargs))
    def build(self,input_shape):
        for item in self.rs_list:
            item.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        in_put = x
        for item in self.rs_list:
            out_put = item(in_put,training=training)
            in_put = out_put
        return out_put

class uk_3D(tf.keras.Model):
    """ 
    A 3x3x3 fractional-strided-Convolution-BatchNorm(InstanceNorm)-ReLU layerwith k filters, stride 1/2 (transpose strides = 2)
    Args:
        input: 5D tensor
        k: integer, number of filters (output depth)
        norm: 'instance' or 'batch' or None
        training: boolean or BoolTensor
        output_size: integer, desired output size of layer
    Returns:
        5D tensor
    """
    def __init__(self,filters,sn=False,use_bias=False,**kwargs):
        super(uk_3D,self).__init__()
        self.filters_zp = filters
        self.l1_up= BaseLayers.UpSample3D(filters=filters,kernel_size=[3,3,3],strides=[2,2,2],padding="SAME",use_bias=use_bias,sn=sn,**kwargs)
        self.l2_norm = BaseLayers.InstanceNorm3D()
        self.l3_activation = BaseLayers.ReLU()
    def build(self,input_shape):
        self.l1_up.build(input_shape=input_shape)
        for i in range(3):
            input_shape[i+1] = input_shape[i+1]*2
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_up(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
###########################################################Discriminator blocks###########################################################
class ck_3D(tf.keras.Model):
    """ 
    A 4x4x4 Convolution-BatchNorm(InstanceNorm)-LeakyReLU layer with k filters and stride 2
    Args:
        input: 5D tensor
        k: integer, number of filters (output depth)
        slope: LeakyReLU's slope
        stride: integer
        norm: 'instance' or 'batch' or None
        training: boolean or BoolTensor
    Returns:
        5D tensor
    """
    def __init__(self,filters,sn=False,norm=True,use_bias=False,**kwargs):
        super(ck_3D,self).__init__()
        self.filters_zp = filters
        self.l1_conv3d= BaseLayers.Conv3D(filters=filters,kernel_size=[4,4,4],strides=[2,2,2],padding='SAME',use_bias=use_bias,sn=sn,**kwargs)
        if norm:
            self.l2_norm = BaseLayers.InstanceNorm3D()
        else:
            self.l2_norm = tf.keras.layers.Activation("linear")
        self.l3_activation = BaseLayers.LeakyReLU(alpha=0.2)
    def build(self,input_shape):
        self.l1_conv3d.build(input_shape=input_shape)
        for i in range(3):
            input_shape[i+1] = int(tf.math.ceil(input_shape[i+1]/2))
        input_shape[-1]=int(self.filters_zp)
        self.l2_norm.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv3d(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
class last_conv_3D(tf.keras.Model):
    """
    Last convolutional layer of discriminator network (1 filter with size 4x4x4, stride 1)
    Args:
        input: 4D tensor
        use_sigmoid: boolean (False if use lsgan)
        name: string, e.g. 'C64'
    """
    def __init__(self,use_sigmoid=True,sn=False,**kwargs):
        super(last_conv_3D,self).__init__()
        self.l1_conv3d= BaseLayers.Conv3D(filters=1,kernel_size=[4,4,4],strides=[1,1,1],padding='SAME',use_bias=True,sn=sn,**kwargs)
        if use_sigmoid:
            self.l2_activation = tf.keras.layers.Activation("sigmoid")
        else:
            self.l2_activation = tf.keras.layers.Activation("linear")
    def build(self,input_shape):
        self.l1_conv3d.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        x = self.l1_conv3d(x,training=training)
        y = self.l2_activation(x,training=training)
        return y
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    a = c7s1_k_3D(6,sn=True,use_bias=True)
    a.build(input_shape=[None,180,256,256,3])
    w = tf.random.normal([1,180,256,256,3],dtype=tf.float16)
    y=a(w)
    w = tf.random.normal([1,180,256,256,3],dtype=tf.float16)
    y=a(w)
    print(y.shape)
    print(y.dtype)
    print(len(a.trainable_variables))
    a = dk_3D(64)#(256+1*2-3)//2 +1=128
    a.build(input_shape=[None,180,256,256,3])
    w = tf.constant(tf.random.normal(([1,180,256,256,3])))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = rk_3D(32)#(256+1*2-3)//2 +1=128
    a.build(input_shape=[None,18,256,256,32])
    w = tf.constant(tf.random.normal(([1,18,256,256,32])))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = n_res_blocks_3D(32,n=9)
    a.build(input_shape=[None,20,256,256,32])
    w = tf.constant(tf.random.normal([1,20,256,256,32]))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = n_res_blocks_3D(32,n=9,use_bias=True,sn=False)
    a.build(input_shape=[None,20,256,256,32])
    w = tf.constant(tf.random.normal([1,20,256,256,32]))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = n_res_blocks_3D(32,n=9,use_bias=True,sn=True)
    a.build(input_shape=[None,20,256,256,32])
    w = tf.constant(tf.random.normal([1,20,256,256,32]))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = uk_3D(5)
    a.build(input_shape=[None,12,256,256,3])
    w = tf.constant(tf.zeros((1,12,256,256,3)))#生成随机值会溢出 所以用全0值代替
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))

    a = ck_3D(64)
    a.build(input_shape=[None,180,256,256,3])
    w = tf.constant(tf.random.normal((1,180,256,256,3)))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))  

    a = ck_3D(64,norm=False)
    a.build(input_shape=[None,180,256,256,3])
    w = tf.constant(tf.random.normal((1,180,256,256,3)))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))  
    a = last_conv_3D(use_sigmoid=True)
    a.build(input_shape=[None,180,512,512,3])
    w = tf.constant(tf.random.normal((1,180,512,512,3)))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = last_conv_3D(use_sigmoid=False)
    a.build(input_shape=[None,180,512,512,3])
    w = tf.constant(tf.random.normal((1,180,512,512,3)))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))
    a = last_conv_3D(use_sigmoid=False,sn=True)
    a.build(input_shape=[None,180,512,512,3])
    w = tf.constant(tf.random.normal((1,180,512,512,3)))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))

    a = last_conv_3D(use_sigmoid=False,sn=False)
    a.build(input_shape=[None,180,512,512,3])
    w = tf.constant(tf.random.normal((1,180,512,512,3)))
    print(a(w).shape)
    print(a(w).dtype)
    print(len(a.trainable_variables))


