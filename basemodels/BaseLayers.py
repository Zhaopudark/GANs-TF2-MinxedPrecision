"""
定义基本模型层
基于tf.keras.Model
"""
import tensorflow as tf
import os
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.BuildHelper import Reconstruction
from utils.SnOption import SpectralNormalization
class Dense(tf.keras.layers.Layer):
    def __init__(self,input_shape,
                      kernel_size,
                      kernel_initializer="glorot_uniform",
                      using_bias=False,
                      activation="relu",**kwargs):
        """
        input_shape: list [None,shape] or [shape]
        kernel_size: int
        using_bias: bool
        activation: string like "relu","sigmoid","leaky_relu","tanh"
        """
        super(Dense,self).__init__()
        self.initializer = Reconstruction.initializer(kernel_initializer)#定义初始化采用的方法
        self.input_shape_zp = Reconstruction.remake_shape(shape=input_shape,dims=2)
        self.w = tf.Variable(initial_value=self.initializer(shape=[self.input_shape_zp[-1],kernel_size],dtype=tf.float32),
                             trainable=True)
        self.using_bias = using_bias
        if using_bias :
            self.b = tf.Variable(initial_value=tf.zeros(shape=(1,kernel_size),dtype=tf.float32),
                                 trainable=True)#节点的偏置也是行向量 才可以正常计算 即对堆叠的batch 都是加载单个batch内
        self.activation = Reconstruction.activation(activation)
    def call(self,x,*args,**kwargs):
        x = tf.matmul(x,self.w)
        if self.using_bias:
            x = x+self.b
        y = self.activation(x)
        return y
class Dropout(tf.keras.Model):
    def __init__(self,rate=0.3,
                      noise_shape=None,
                      seed=None,**kwargs):
        super(Dropout,self).__init__()
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        # print(kwargs)
    def call(self,x,training=True,**kwargs):
        if training == True:
            y = tf.nn.dropout(x,rate=self.rate,noise_shape=self.noise_shape,seed=self.seed)
            return y
        else:
            y = x
            return y
class BatchNormalization(tf.keras.Model):
    def __init__(self,input_shape,
                      epsilon=1e-3,
                      momentum=0.99):
        super(BatchNormalization,self).__init__()
        """
        input_shape 不考虑batch维度
        """
        self.offset = tf.Variable(tf.zeros((1),dtype=tf.float32),trainable=True)
        self.scale = tf.Variable(tf.ones((1),dtype=tf.float32),trainable=True)
        self.mean = tf.Variable(tf.zeros(input_shape),dtype=tf.float32,trainable=False)#参与测试不参与训练
        self.variance = tf.Variable(tf.ones(input_shape),dtype=tf.float32,trainable=False)#参与测试不参与训练
        self.epsilon = epsilon
        self.momentum = momentum
    def call(self,x,training=True,**kwargs):
        if training:
            mean,variance = tf.nn.moments(x,axes=[0],keepdims=True)
            self.mean.assign(self.mean*self.momentum+(1-self.momentum)*self.mean)
            self.variance.assign(self.variance*self.momentum+(1-self.momentum)*self.variance)
            y = tf.nn.batch_normalization(x,mean,variance,self.offset,self.scale,self.epsilon)
            return y
        else:
            y = tf.nn.batch_normalization(x,self.mean,self.variance,self.offset,self.scale,self.epsilon)
            return y
class InstanceNorm3D(tf.keras.Model):
    def __init__(self,epsilon=1e-3):
        """
        instance Norm
        对于三通道彩图 是每个通道 每个batch 下做 WH的norm的
        对于3D医学图像 是每个通道 每个batch下做 DWH的norm的
        """
        super(InstanceNorm3D,self).__init__()
        self.epsilon = epsilon
        self.scale = tf.Variable(tf.ones((1),dtype=tf.float32),trainable=True)
        self.offset = tf.Variable(tf.zeros((1),dtype=tf.float32),trainable=True)
    def call(self,x):
        mean,variance = tf.nn.moments(x,axes=[1,2,3],keepdims=True)
        x_hat = (x-mean)/tf.sqrt(variance+self.epsilon)
        y = self.scale*x_hat+self.offset
        return y
class InstanceNorm2D(tf.keras.Model):
    def __init__(self,epsilon=1e-3):
        """
        instance Norm
        对于三通道彩图 是每个通道 每个batch 下做 WH的norm的
        对于3D医学图像 是每个通道 每个batch下做 DWH的norm的
        """
        super(InstanceNorm2D,self).__init__()
        self.epsilon = epsilon
        self.scale = tf.Variable(tf.ones((1),dtype=tf.float32),trainable=True)
        self.offset = tf.Variable(tf.zeros((1),dtype=tf.float32),trainable=True)
    def call(self,x):
        mean,variance = tf.nn.moments(x,axes=[1,2],keepdims=True)
        x_hat = (x-mean)/tf.sqrt(variance+self.epsilon)
        y = self.scale*x_hat+self.offset
        return y
class Conv3D(tf.keras.Model):
    def __init__(self,input_shape,filters,kernel_size,strides=[1,1,1],padding="SAME",using_bias=False,sn=False):
        """
        将卷积统一为 
        计算参数量
        自行padding 
        然后使用VALID卷积
        这样就可以嵌入任意的padding模式了
        则该层的输入shape对于SN而言，就需要加上padding_vect变成padding后的值
        """
        super(Conv3D,self).__init__()
        self.input_shape_zp = Reconstruction.remake_shape(input_shape,dims=5)#5维的输出张量
        self.kernel_size_zp = Reconstruction.remake_kernel_size(kernel_size,dims=3)#3维卷积核 变为[1,x,x,x,1]5D形式
        self.strides_zp = Reconstruction.remake_strides(strides,dims=3)#3维步长 变为[1,x,x,x,1]5D形式
        self.output_shape_zp,self.padding,self.padding_vect = Reconstruction.ConvCalculation(self.input_shape_zp[1:-1],filters,self.kernel_size_zp[1:-1],self.strides_zp[1:-1],padding) 
        weight_shape = self.kernel_size_zp[1:-1]+[self.input_shape_zp[-1]]+[filters]#[D,H,W,IC,OC]
        self.initializer = Reconstruction.initializer("random_normal",mean=0.0,stddev=0.02)

        self.w = tf.Variable(initial_value=self.initializer(shape=weight_shape),
                             trainable=True,
                             dtype=tf.float32)
        self.using_bias = using_bias
        if using_bias == True:
            self.b = tf.Variable(tf.zeros(self.output_shape_zp,dtype=tf.float32),trainable=True)
        self.sn = sn
        
    def call(self,x):
        x = tf.pad(x,self.padding_vect,self.padding)
        if self.sn:
            sigma=self.w_bar
        else:
            sigma=1
        print(self.w.shape)
        w=self.w/sigma
        print(w.shape)
        y = tf.nn.conv3d(input=x,filters=w,strides=self.strides_zp,padding="VALID")
        if self.using_bias == True:
            return y+self.b
        else:
            return y
    @property
    def w_bar(self,iter_k=5):
        sigma = SpectralNormalization.SN(self.w,iter_k=iter_k)
        return sigma
class Conv3DTranspose(tf.keras.Model):
    def __init__(self,input_shape,output_shape,filters,kernel_size,strides=[1,1,1],padding="SAME",using_bias=False,sn=False):
        super(Conv3DTranspose,self).__init__()
        self.input_shape_zp = Reconstruction.remake_shape(input_shape,dims=5)#变成5维的输出张量 None补充不足的维度(batch)
        self.output_shape_zp = Reconstruction.remake_shape(output_shape+[filters],dims=5)#变成5维的输出张量
        self.kernel_size_zp = Reconstruction.remake_kernel_size(kernel_size,dims=3)#3维卷积核 变为[1,x,x,x,1]5D形式
        self.strides_zp = Reconstruction.remake_strides(strides,dims=3)#3维步长 变为[1,x,x,x,1]5D形式 
        self.padding = padding
        Reconstruction.ConvTransCheck(input_shape=self.input_shape_zp[1:-1],
                                      output_shape=self.output_shape_zp[1:-1],
                                      filters=filters,
                                      kernel_size=self.kernel_size_zp[1:-1],
                                      strides=self.strides_zp[1:-1],
                                      padding=self.padding)
        weight_shape = self.kernel_size_zp[1:-1]+[filters]+[self.input_shape_zp[-1]]#[D,H,W,OC,IC]
        self.initializer = Reconstruction.initializer("random_normal",mean=0.0,stddev=0.02)

        self.w = tf.Variable(initial_value=self.initializer(shape=weight_shape),
                             trainable=True,
                             dtype=tf.float32)

        self.using_bias = using_bias
        if using_bias == True:
            self.b = tf.Variable(tf.zeros(self.output_shape_zp[1:],dtype=tf.float32),trainable=True)
        self.sn = sn
    def call(self,x):
        if self.sn:
            sigma = self.w_bar
        else:
            sigma = 1
        w=self.w/sigma #避免直接用assign修改
        y = tf.nn.conv3d_transpose(input=x,
                                   filters=w,
                                   output_shape=[x.shape[0]]+self.output_shape_zp[1:],
                                   strides=self.strides_zp,
                                   padding=self.padding)
        return y
    @property
    def w_bar(self,iter_k=5):
        sigma = SpectralNormalization.SN(self.w,iter_k=iter_k)
        return sigma
class Conv2D(tf.keras.Model):
    def __init__(self,input_shape,filters,kernel_size,strides=[1,1],padding="SAME",using_bias=False,sn=False):
        """
        将卷积统一为 
        计算参数量
        自行padding 
        然后使用VALID卷积
        这样就可以嵌入任意的padding模式了
        则该层的输入shape对于SN而言，就需要加上padding_vect变成padding后的值
        """
        super(Conv2D,self).__init__()
        self.input_shape_zp = Reconstruction.remake_shape(input_shape,dims=4)#4维的输出张量
        self.kernel_size_zp = Reconstruction.remake_kernel_size(kernel_size,dims=2)#2维卷积核 变为[1,x,x,1]4D形式
        self.strides_zp = Reconstruction.remake_strides(strides,dims=2)#2维步长 变为[1,x,x,1]4D形式
        self.output_shape_zp,self.padding,self.padding_vect = Reconstruction.ConvCalculation(self.input_shape_zp[1:-1],filters,self.kernel_size_zp[1:-1],self.strides_zp[1:-1],padding) 
        weight_shape = self.kernel_size_zp[1:-1]+[self.input_shape_zp[-1]]+[filters]#[H,W,IC,OC]
        self.initializer = Reconstruction.initializer("random_normal",mean=0.0,stddev=0.02)

        self.w = tf.Variable(initial_value=self.initializer(shape=weight_shape),
                             trainable=True,
                             dtype=tf.float32)
        self.using_bias = using_bias
        if using_bias == True:
            self.b = tf.Variable(tf.zeros(self.output_shape_zp,dtype=tf.float32),trainable=True)
        self.sn = sn
        
    def call(self,x):
        x = tf.pad(x,self.padding_vect,self.padding)
        if self.sn:
            sigma=self.w_bar
        else:
            sigma=1
        print(self.w.shape)
        w=self.w/sigma
        print(w.shape)
        y = tf.nn.conv2d(input=x,filters=w,strides=self.strides_zp,padding="VALID")
        if self.using_bias == True:
            return y+self.b
        else:
            return y
    @property
    def w_bar(self,iter_k=5):
        sigma = SpectralNormalization.SN(self.w,iter_k=iter_k)
        return sigma
class Conv2DTranspose(tf.keras.Model):
    def __init__(self,input_shape,output_shape,filters,kernel_size,strides=[1,1],padding="SAME",using_bias=False,sn=False):
        super(Conv2DTranspose,self).__init__()
        self.input_shape_zp = Reconstruction.remake_shape(input_shape,dims=4)#变成4维的输出张量 None补充不足的维度(batch)
        self.output_shape_zp = Reconstruction.remake_shape(output_shape+[filters],dims=4)#变成4维的输出张量
        self.kernel_size_zp = Reconstruction.remake_kernel_size(kernel_size,dims=2)#2维卷积核 变为[1,x,x,1]4D形式
        self.strides_zp = Reconstruction.remake_strides(strides,dims=2)#2维步长 变为[1,x,x,1]4D形式 
        self.padding = padding
        Reconstruction.ConvTransCheck(input_shape=self.input_shape_zp[1:-1],
                                      output_shape=self.output_shape_zp[1:-1],
                                      filters=filters,
                                      kernel_size=self.kernel_size_zp[1:-1],
                                      strides=self.strides_zp[1:-1],
                                      padding=self.padding)
        weight_shape = self.kernel_size_zp[1:-1]+[filters]+[self.input_shape_zp[-1]]#[D,H,W,OC,IC]
        self.initializer = Reconstruction.initializer("random_normal",mean=0.0,stddev=0.02)

        self.w = tf.Variable(initial_value=self.initializer(shape=weight_shape),
                             trainable=True,
                             dtype=tf.float32)

        self.using_bias = using_bias
        if using_bias == True:
            self.b = tf.Variable(tf.zeros(self.output_shape_zp[1:],dtype=tf.float32),trainable=True)
        self.sn = sn
    def call(self,x):
        if self.sn:
            sigma = self.w_bar
        else:
            sigma = 1
        w=self.w/sigma #避免直接用assign修改
        y = tf.nn.conv2d_transpose(input=x,
                                   filters=w,
                                   output_shape=[x.shape[0]]+self.output_shape_zp[1:],
                                   strides=self.strides_zp,
                                   padding=self.padding)
        return y
    @property
    def w_bar(self,iter_k=5):
        sigma = SpectralNormalization.SN(self.w,iter_k=iter_k)
        return sigma
######################################################################################################################
class UpSampaleConv2D(tf.keras.layers.Layer):
    """
    将转至卷积变成上采样和卷积的组合
    input_shape
    output_shape
    strides
    padding
    之间必须满足符合一般转至卷积的计算逻辑
    """
    def __init__(self,input_shape,output_shape,filters,kernel_size,strides=[2,2],padding="SAME",using_bias=False,sn=False):#输出的深度已经给出 只要给出每个通道的H,W即可
        super(UpSampaleConv2D,self).__init__()
        self.input_shape_zp = Reconstruction.remake_shape(input_shape,dims=4)#变成4维的输出张量 None补充不足的维度(batch)
        self.output_shape_zp = Reconstruction.remake_shape(output_shape+[filters],dims=4)#变成4维的输出张量
        self.kernel_size_zp = Reconstruction.remake_kernel_size(kernel_size,dims=2)#2维卷积核 变为[1,x,x,1]4D形式
        self.strides_zp = Reconstruction.remake_strides(strides,dims=2)#2维步长 变为[1,x,x,1]4D形式 
        self.padding = padding

        self.padding,self.padding_vect,self.cut_flag = Reconstruction.Trans2UpsampleCal(self.input_shape_zp[1:-1],
                                                                                        self.output_shape_zp[1:-1],
                                                                                        filters,
                                                                                        self.kernel_size_zp[1:-1],
                                                                                        self.strides_zp[1:-1],
                                                                                        self.padding)

        self.up_size = self.strides_zp[1:-1]
        self.up_op = tf.keras.layers.UpSampling2D(size=self.up_size)#上采样只对非通道和非batch进行 剔除冗余维度
        self.strides_zp = Reconstruction.remake_strides([1,1],dims=2)

        self.initializer = Reconstruction.initializer("random_normal",mean=0.0,stddev=0.02)
        weight_shape = self.kernel_size_zp[1:-1]+[self.input_shape_zp[-1]]+[filters]#[H,W,IC,OC]
        self.w = tf.Variable(initial_value=self.initializer(shape=weight_shape),
                             trainable=True,
                             dtype=tf.float32)

        self.using_bias = using_bias
        if using_bias == True:
            self.b = tf.Variable(tf.zeros(self.output_shape_zp[1:],dtype=tf.float32),trainable=True)
        self.sn = sn

    def call(self,x):
        x = self.up_op(x)
        if self.cut_flag:
            x = x[0-self.padding_vect[0][0]:int(x.shape[0])+self.padding_vect[0][1],
                  0-self.padding_vect[1][0]:int(x.shape[1])+self.padding_vect[1][1],
                  0-self.padding_vect[2][0]:int(x.shape[2])+self.padding_vect[2][1],
                  0-self.padding_vect[3][0]:int(x.shape[3])+self.padding_vect[3][1]]
        else:
            x = tf.pad(x,self.padding_vect,"CONSTANT")   
        if self.sn == True:
            sigmma = self.w_bar
        else:
            sigmma = 1
        w=self.w/sigmma #避免直接用assign修改
        y = tf.nn.conv2d(input=x,filters=w,strides=self.strides_zp,padding="VALID")
        return y
    @property
    def w_bar(self,iter_k=5):
        sigma = SpectralNormalization.SN(self.w,iter_k=iter_k)
        return sigma
class UpSampaleConv3D(tf.keras.layers.Layer):
    """
    将转至卷积变成上采样和卷积的组合
    input_shape
    output_shape
    strides
    padding
    之间必须满足符合一般转至卷积的计算逻辑
    """
    def __init__(self,input_shape,output_shape,filters,kernel_size,strides=[2,2,2],padding="SAME",using_bias=False,sn=False):#输出的深度已经给出 只要给出每个通道的H,W即可
        super(UpSampaleConv3D,self).__init__()
        self.input_shape_zp = Reconstruction.remake_shape(input_shape,dims=5)#变成5维的输出张量 None补充不足的维度(batch)
        self.output_shape_zp = Reconstruction.remake_shape(output_shape+[filters],dims=5)#变成5维的输出张量
        self.kernel_size_zp = Reconstruction.remake_kernel_size(kernel_size,dims=3)#3维卷积核 变为[1,x,x,x,1]5D形式
        self.strides_zp = Reconstruction.remake_strides(strides,dims=3)#3维步长 变为[1,x,x,x,1]4D形式 
        self.padding = padding
        
        self.padding,self.padding_vect,self.cut_flag = Reconstruction.Trans2UpsampleCal(self.input_shape_zp[1:-1],
                                                                                        self.output_shape_zp[1:-1],
                                                                                        filters,
                                                                                        self.kernel_size_zp[1:-1],
                                                                                        self.strides_zp[1:-1],
                                                                                        self.padding)

        self.up_size = self.strides_zp[1:-1]
        self.up_op = tf.keras.layers.UpSampling3D(size=self.up_size)#上采样只对非通道和非batch进行 剔除冗余维度
        self.strides_zp = Reconstruction.remake_strides([1,1,1],dims=3)

        self.initializer = Reconstruction.initializer("random_normal",mean=0.0,stddev=0.02)
        weight_shape = self.kernel_size_zp[1:-1]+[self.input_shape_zp[-1]]+[filters]#[D,H,W,IC,OC]
        self.w = tf.Variable(initial_value=self.initializer(shape=weight_shape),
                             trainable=True,
                             dtype=tf.float32)

        self.using_bias = using_bias
        if using_bias == True:
            self.b = tf.Variable(tf.zeros(self.output_shape_zp[1:],dtype=tf.float32),trainable=True)
        self.sn = sn

    def call(self,x):
        x = self.up_op(x)
        if self.cut_flag:
            x = x[0-self.padding_vect[0][0]:int(x.shape[0])+self.padding_vect[0][1],
                  0-self.padding_vect[1][0]:int(x.shape[1])+self.padding_vect[1][1],
                  0-self.padding_vect[2][0]:int(x.shape[2])+self.padding_vect[2][1],
                  0-self.padding_vect[3][0]:int(x.shape[3])+self.padding_vect[3][1],
                  0-self.padding_vect[4][0]:int(x.shape[4])+self.padding_vect[4][1]]
        else:
            x = tf.pad(x,self.padding_vect,"CONSTANT")   
        if self.sn == True:
            sigmma = self.w_bar
        else:
            sigmma = 1
        w=self.w/sigmma #避免直接用assign修改
        y = tf.nn.conv3d(input=x,filters=w,strides=self.strides_zp,padding="VALID")
        return y
    @property
    def w_bar(self,iter_k=5):
        sigma = SpectralNormalization.SN(self.w,iter_k=iter_k)
        return sigma
# ###########################################################################################################
# if __name__ == "__main__":
#     from tensorflow.keras.mixed_precision import experimental as mixed_precision
#     policy = mixed_precision.Policy('mixed_float16')
#     mixed_precision.set_policy(policy)
#     a = UpSampaleConv2D([4,4,5],[10,10],1,kernel_size=(3,3),strides=(2,2),padding="VALID",sn=True)
#     x = tf.constant(tf.random.normal((1,4,4,5),dtype=tf.float16))
#     y = a(x)
#     print(y.shape,y.dtype)
#     print(len(a.trainable_variables))  
#     print(a.output_shape_zp)
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    x = tf.random.normal(shape=[5,2], mean=0.0, stddev=1.0)
    a_ = tf.keras.layers.Dense(3)
    a = Dense([2],3)
    print(a(x))
    print(a.trainable_variables)
    a = Dense([2],3)
    print(a(x))
    print(a.trainable_variables)
    a = Dense(2,3)
    print(a(x))
    print(a.trainable_variables)
    a = Dense(2,3,kernel_initializer="glorot_uniform",
                using_bias=True,
                activation="relu")
    print(a.trainable_variables)
    print(a(x))
    # b = tf.constant([[1.0,1.0],[2.0,2.0]])
    b = tf.random.normal(shape=[7,3], mean=0.0, stddev=1.0)
    bn = BatchNormalization(3)
    print(tmp1:=bn(b))
    print(tmp2:=bn(b,training=False))
    print(tmp1-tmp2)
    print(x:=tf.random.normal(shape=[5,10], mean=0.0, stddev=1.0))
    dp = Dropout(rate=0.2,noise_shape=[5,1])#在非1维度随机出0 然后在是1维度广播
    print(dp(x))
    dp = Dropout(rate=0.2,noise_shape=[1,10])
    print(dp(x))
    print(dp(x))
    print(dp(x))
    print(x:=tf.constant([[1.0,1.0],[1.0,1.0]]))
    dp = Dropout(rate=0.2,noise_shape=[2,2])
    print(dp(x))
    dp = Dropout(rate=0.5,noise_shape=[1,2])
    print(dp(x))
    dp = Dropout(rate=0.5,noise_shape=[2,1])
    print(dp(x))
    dp = Dropout(drop_rate=0.5,noise_shape=[2,1])
    print(x:=tf.reshape(tf.range(27.0),[1,3,3,3,1]))
    i_n = InstanceNorm3D()
    print(i_n(x))
    print(x:=tf.reshape(tf.range(27.0),[1,3,3,3]))
    i_n = InstanceNorm2D()
    print(i_n(x))
    cv3d = Conv3D(input_shape=[64,64,64,3],filters=16,kernel_size=[3,3,3],strides=[1,1,1],padding="SAME",using_bias=False,sn=True)
    print(x:=tf.reshape(tf.range(64*64*64*3.0),[1,64,64,64,3]))
    print(cv3d(x).shape)

    a = Conv3DTranspose([4,4,4,5],[8,8,8],100,kernel_size=(3,3,3),strides=(2,2,2),padding="SAME",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)

    a = Conv2DTranspose([4,4,5],[7,7],1,kernel_size=(3,3),strides=(2,2),padding="SAME",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)

    cv2d = Conv2D(input_shape=[4,4,1],filters=4,kernel_size=[2,2],strides=[1,1],sn=True)
    x = tf.random.normal((1,4,4,1))
    print(cv2d(x).shape)
    for item in cv2d.trainable_variables:
        print(item.name)
    a = UpSampaleConv2D([4,4,5],[10,10],1,kernel_size=(3,3),strides=(2,2),padding="VALID",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)
    a = UpSampaleConv2D([4,4,5],[10,10],1,kernel_size=(2,2),strides=(3,3),padding="SAME",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)
    a = UpSampaleConv3D([4,4,4,5],[10,10,10],1,kernel_size=(2,2,2),strides=(3,3,3),padding="SAME",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)
    a = UpSampaleConv3D([4,4,4,5],[7,7,7],1,kernel_size=(4,4,4),strides=(2,2,2),padding="SAME",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)
    a = UpSampaleConv3D([4,4,4,5],[7,7,7],1,kernel_size=(2,2,2),strides=(2,2,2),padding="SAME",sn=True)
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(len(a.trainable_variables))  
    print(a.output_shape_zp)