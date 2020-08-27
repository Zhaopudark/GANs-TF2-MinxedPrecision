"""
tf.nn不支持混合精度
直接继承tf.keras.layers.xxx 构建支持混合精度的模型层  
将复杂的API使用逻辑交给keras内部调和
介入卷积的参数计算与padding等，增强模型层的功能
baselayer 设计原则 API功能只增不减 一切原父类支持的参数都可以支持
"""
import tensorflow as tf
import os
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.BuildHelper import Reconstruction
from utils.SnOption import SpectralNormalization
########################################################################################
"""
需要涉及参数选择的API 都单独集成实现一次
"""
class ReLU(tf.keras.layers.ReLU):
    def __init__(self,*args,**kwargs):
        super(ReLU,self).__init__(*args,**kwargs)
    def call(self,x,**kwargs):
        return super(ReLU,self).call(x)
class LeakyReLU(tf.keras.layers.LeakyReLU):
    def __init__(self,*args,**kwargs):
        super(LeakyReLU,self).__init__(*args,**kwargs)
    def call(self,x,**kwargs):
        return super(LeakyReLU,self).call(x)
########################################################################################
class Dense(tf.keras.layers.Dense):
    """
    参照官方API
    增加sn
    layer=Dense(units,activation="relu,sigmoid,tanh,leaky_relu",use_bias=False)
    layer=Dense(units=100,use_bias=True)
    """
    def __init__(self,*args,sn=False,**kwargs):
        self.sn = sn
        super(Dense,self).__init__(*args,**kwargs)
    def build(self,*args,**kwargs): 
        super(Dense,self).build(*args,**kwargs)
    def call(self,x,**kwargs):
        if self.sn:
            # print(self.kernel)
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x / sigma
        return super(Dense,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma
################################## Different when training and not training ###########################################
class Dropout(tf.keras.layers.Dropout):
    """
    参照官方API (rate,noise_shape=None,seed=None,**kwargs)
    noise_shape =[x,x,x]
    noise_shape =[x,1,x]
    表示产生随机的0,1后 向完整的维度广播  
    剩余维度会产生1/(1 - rate)
    call() training==True 才会进行dropout
    """
    def __init__(self,*args,**kwargs):
        super(Dropout,self).__init__(*args,**kwargs)
    def build(self,*args,**kwargs):
        super(Dropout,self).build(*args,**kwargs)
    def call(self,x,training=None):
        return super(Dropout,self).call(x,training=training)

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self,*args,**kwargs):
        super(BatchNormalization,self).__init__(*args,**kwargs)
    def build(self,*args,**kwargs):
        super(BatchNormalization,self).build(*args,**kwargs)
    def call(self,x,training=None):
        return super(BatchNormalization,self).call(x,training=training)

class InstanceNorm3D(tf.keras.layers.Layer):
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
    def build(self,*args,**kwargs):
        super(InstanceNorm3D,self).build(*args,**kwargs)
    def call(self,x,**kwargs):
        x=tf.cast(x,tf.float32)
        mean,variance = tf.nn.moments(x,axes=[1,2,3],keepdims=True)#[B,D,W,H,C]
        x_hat = (x-mean)/tf.sqrt(variance+self.epsilon)
        y = self.scale*x_hat+self.offset
        return tf.cast(y,tf.float16)

class InstanceNorm2D(tf.keras.layers.Layer):
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
    def build(self,*args,**kwargs):
        super(InstanceNorm2D,self).build(*args,**kwargs)
    def call(self,x,**kwargs):
        x=tf.cast(x,tf.float32)
        mean,variance = tf.nn.moments(x,axes=[1,2],keepdims=True)#[B,W,H,C]
        x_hat = (x-mean)/tf.sqrt(variance+self.epsilon)
        y = self.scale*x_hat+self.offset
        return tf.cast(y,tf.float16)
###############################################Convolution###############################################################
class Conv3D(tf.keras.layers.Conv3D):
    def __init__(self,*args,filters=None,kernel_size=None,strides=None,padding=None,sn=False,**kwargs):
        """
        参照官方API
        增加sn
        filters = 8
        kernel_size = (10,10,10) 
        strides = (1,1,1)
        padding = 'valid'
        use_bias = False
        
        valid 时 不进行padding 正常卷积
        SAME 时 默认zero padding
        CONSTANT 时 默认 zero padding 
        REFLECT  时 REFLECT padding
        SYMMETRIC 时 SYMMETRIC padding
        """
        self.sn = sn
        self.filters_zp = filters
        self.kernel_size_zp = kernel_size
        self.strides_zp = strides
        self.padding_zp = padding.upper()
        super(Conv3D,self).__init__(*args,filters=filters,kernel_size=kernel_size,strides=strides,padding="valid",**kwargs)
    def build(self,*args,input_shape=None,**kwargs):
        out_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape[1:-1],
                                                                        filters=self.filters_zp,
                                                                        kernel_size=self.kernel_size_zp,
                                                                        strides=self.strides_zp,
                                                                        padding=self.padding_zp)
        self.padding_zp = padding
        self.padding_vect_zp = padding_vect                                                
        super(Conv3D,self).build(*args,input_shape=input_shape,**kwargs)
    def call(self,x,**kwargs):
        x = tf.pad(x,self.padding_vect_zp,self.padding_zp)
        if self.sn:
            # print(self.kernel.shape)
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x/sigma
        return super(Conv3D,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma

class Conv2D(tf.keras.layers.Conv2D):
    def __init__(self,*args,filters=None,kernel_size=None,strides=None,padding=None,sn=False,**kwargs):
        """
        参照官方API
        增加sn
        filters = 8
        kernel_size = (10,10) 
        strides = (1,1)
        padding = 'valid'
        use_bias = False
        """
        self.sn = sn
        self.filters_zp = filters
        self.kernel_size_zp = kernel_size
        self.strides_zp = strides
        self.padding_zp = padding.upper()
        super(Conv2D,self).__init__(*args,filters=filters,kernel_size=kernel_size,strides=strides,padding="valid",**kwargs)

    def build(self,*args,input_shape=None,**kwargs):
        out_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape[1:-1],
                                                                        filters=self.filters_zp,
                                                                        kernel_size=self.kernel_size_zp,
                                                                        strides=self.strides_zp,
                                                                        padding=self.padding_zp)
        self.padding_zp = padding
        self.padding_vect_zp = padding_vect                            
        super(Conv2D,self).build(*args,input_shape=input_shape,**kwargs)
    def call(self,x,**kwargs):
        x = tf.pad(x,self.padding_vect_zp,self.padding_zp)
        if self.sn:
            # print(self.kernel.shape)
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x/sigma
        return super(Conv2D,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma
class Conv3DTranspose(tf.keras.layers.Conv3DTranspose):
    def __init__(self,*args,sn=False,**kwargs):
        """
        参照官方API
        增加sn
        filters = 8
        kernel_size = (10,10,10) 
        strides = (1,1,1)
        padding = 'valid'
        use_bias = False
        """
        self.sn = sn
        super(Conv3DTranspose,self).__init__(*args,**kwargs)
    def build(self,*args,**kwargs):
        super(Conv3DTranspose,self).build(*args,**kwargs)
    def call(self,x,**kwargs):
        if self.sn:
            # print(self.kernel.shape)
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x/sigma
        return super(Conv3DTranspose,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma
class Conv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self,*args,sn=False,**kwargs):
        """
        参照官方API
        增加sn
        filters = 8
        kernel_size = (10,10) 
        strides = (1,1)
        padding = 'valid'
        use_bias = False

        在转置卷积中 VALID情况下 tf会先满足target_shape作为输出 
        无法满足时，最小的shape作为转制后的shape
        (x-kernel_zize)//strides+1 == input_shape 的最小x作为output_shape
        SAME 情况下
        ceil(x/strides) == input_shape 
        y<=ceil(y)<y+1
        x/strides<=input_shape<x/strides+1
        input_shape-1<x/strides<=input_shape
        input_shape*strides-strides<x<=input_shape*strides
        为了唯一确定 取最大的x 即input_shape*strides 作为output_shape
        一般情况下 我就是想x2 x2 x2 x4 x8 那就直接SAME参数即可
        """
        self.sn = sn
        super(Conv2DTranspose,self).__init__(*args,**kwargs)
    def build(self,*args,**kwargs):
        super(Conv2DTranspose,self).build(*args,**kwargs)
    def call(self,x,**kwargs):
        if self.sn:
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x/sigma
        return super(Conv2DTranspose,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma

class UpSample2D(tf.keras.layers.Conv2D):
    def __init__(self,*args,
                      sn=False,
                      filters=None,
                      kernel_size=None,
                      strides=None,
                      padding=None,
                      use_bias=None,
                      **kwargs):
        """
        本方式为了减少歧义，有效参数只接受关键字参数
        tf.keras.layers.UpSampling2D
        +
        tf.keras.layers.Conv2D
        参照官方API 模拟转置卷积的参数计算
        增加sn

        上采样与转置卷积之间有参数计算的差别
        接受的padding方式是转置卷积的padding方式 最后会转换为给予Conv操作的padding方式
        接受的strides方式也是转置卷积的strides方式 最后会转换为给予Conv操作的strides=[1,1] [1,1,1]

        第一步 计算output_shape 
        第二步 计算input_shape*strides和output_shape的差距
        """
        self.sn = sn
        self.filters_zp = filters
        self.kernel_size_zp=kernel_size
        self.strides_zp=strides
        self.padding_zp=padding
        self.use_bias_zp=use_bias

        self.up_op = tf.keras.layers.UpSampling2D(size=strides,**kwargs)
        super(UpSample2D,self).__init__(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=(1,1),
                                        padding="SAME",
                                        use_bias=use_bias,**kwargs)
    def build(self,*args,input_shape=None,**kwargs):
        self.cut_buf,target_output_shape= self.cal_cut_buf(input_shape[1:-1],self.kernel_size_zp,self.strides_zp,self.padding_zp)
        input_shape = [None]+target_output_shape+[input_shape[-1]]
        super(UpSample2D,self).build(input_shape=input_shape)
    def call(self,x,**kwargs):
        x=self.up_op(x)
        x=self.cut_pad_2D(x,self.cut_buf)
        if self.sn:
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x/sigma
        return super(UpSample2D,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma
    def cal_cut_buf(self,input_shape,kernel_size,strides,padding):
        #不完整的output_shape计算  指导input padding方式
        output_shape = []
        target_shape = []
        dim = len(kernel_size)
        if padding.upper() == "VALID":
            """
            先满足 target_shape = input_shape[i]*strides[i]
            不满足 target_shape 时，输出shape为满足计算的下界
            """
            for i in range(dim):
                target_shape.append(input_shape[i]*strides[i])
                if((input_shape[i]-1)*strides[i]+kernel_size[i]<=target_shape[i]<input_shape[i]*strides[i]+kernel_size[i]):
                    output_shape.append(target_shape[i])
                else:
                    output_shape.append((input_shape[i]-1)*strides[i]+kernel_size[i])
        elif padding.upper() == "SAME":
            for i in range(dim):
                target_shape.append(input_shape[i]*strides[i])
                output_shape.append(target_shape[i])
        else:
            raise ValueError("Unsupported padding ways " + padding)
        
        """
        output_shape 是等效为转置卷积的输出shape 不完整 不包含深度和batch部分
        因为用SAME对上采样后进行卷积  所以output_shape就是卷积操作的inpu_shape(虽然卷积操作需要知道的inpu_shape仅仅是输入深度 但是还是尽量还原真实的shape)
        对target_shape 进行单步长卷积 希望得到output_shape
        但是初始化就已经定下SAME的方式 所以直接将target_shape 统一到output_shape即可
        可能需要padding或者cut
        遵循右侧(后端)优先原则
        """
        diff_buf = []
        for i in range(dim):
            diff_buf.append(output_shape[i]-target_shape[i])
        cut_buf = []
        for i in range(dim):
            if diff_buf[i]<0:
                tmp = -diff_buf[i]
                left_pad = tmp//2
                right_pad = tmp-left_pad
                cut_buf.append([-left_pad,-right_pad])
            else:
                tmp = diff_buf[i]
                left_pad = tmp//2
                right_pad = tmp-left_pad
                cut_buf.append([left_pad,right_pad])
            # 不存在left right 符号互斥的情况
        return cut_buf,output_shape
    def cut_pad_2D(self,x,cut_buf):

        if(cut_buf[0][0]<0)or(cut_buf[0][1]<0):
            x = x[:,0-cut_buf[0][0]:int(x.shape[1])+cut_buf[0][1],:,:]
        else:
            x = tf.pad(x,[[0,0],cut_buf[0],[0,0],[0,0]],"CONSTANT")

        if(cut_buf[1][0]<0)or(cut_buf[1][1]<0):
            x = x[:,:,0-cut_buf[1][0]:int(x.shape[2])+cut_buf[1][1],:]
        else:
            x = tf.pad(x,[[0,0],[0,0],cut_buf[1],[0,0]],"CONSTANT")
        return x
        
class UpSample3D(tf.keras.layers.Conv3D):
    def __init__(self,*args,
                      sn=False,
                      filters=None,
                      kernel_size=None,
                      strides=None,
                      padding=None,
                      use_bias=None,
                      **kwargs):
  
        self.sn = sn
        self.filters_zp = filters
        self.kernel_size_zp=kernel_size
        self.strides_zp=strides
        self.padding_zp=padding
        self.use_bias_zp=use_bias

        self.up_op = tf.keras.layers.UpSampling3D(size=strides,**kwargs)
        super(UpSample3D,self).__init__(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=(1,1,1),
                                        padding="SAME",
                                        use_bias=use_bias,**kwargs)
    def build(self,*args,input_shape=None,**kwargs):
        # input_shape = None
        self.cut_buf,target_output_shape= self.cal_cut_buf(input_shape[1:-1],self.kernel_size_zp,self.strides_zp,self.padding_zp)
        input_shape = [None]+target_output_shape+[input_shape[-1]]
        super(UpSample3D,self).build(input_shape=input_shape)
    def call(self,x,**kwargs):
        x=self.up_op(x)
        x=self.cut_pad_3D(x,self.cut_buf)
        if self.sn:
            # print(self.kernel.shape)
            sigma = tf.cast(self.sn_op(tf.cast(self.kernel,tf.float32)),tf.float16)
        else:
            sigma = 1.0
        x = x/sigma
        return super(UpSample3D,self).call(x)
    def sn_op(self,w,iter_k=5):
        sigma = SpectralNormalization.SN(w,iter_k=iter_k)
        return sigma
    # @classmethod
    def cal_cut_buf(self,input_shape,kernel_size,strides,padding):
        #不完整的output_shape计算  指导input padding方式
        output_shape = []
        target_shape = []
        dim = len(kernel_size)
        if padding.upper() == "VALID":
            """
            先满足 target_shape = input_shape[i]*strides[i]
            不满足 target_shape 时，输出shape为满足计算的下界
            """
            for i in range(dim):
                target_shape.append(input_shape[i]*strides[i])
                if((input_shape[i]-1)*strides[i]+kernel_size[i]<=target_shape[i]<input_shape[i]*strides[i]+kernel_size[i]):
                    output_shape.append(target_shape[i])
                else:
                    output_shape.append((input_shape[i]-1)*strides[i]+kernel_size[i])
        elif padding.upper() == "SAME":
            for i in range(dim):
                target_shape.append(input_shape[i]*strides[i])
                output_shape.append(target_shape[i])
        else:
            raise ValueError("Unsupported padding ways " + padding)
        
        """
        output_shape 是等效为转置卷积的输出shape 不完整 不包含深度和batch部分
        因为用SAME对上采样后进行卷积  所以output_shape就是卷积操作的inpu_shape(虽然卷积操作需要知道的inpu_shape仅仅是输入深度 但是还是尽量还原真实的shape)
        对target_shape 进行单步长卷积 希望得到output_shape
        但是初始化就已经定下SAME的方式 所以直接将target_shape 统一到output_shape即可
        可能需要padding或者cut
        遵循右侧(后端)优先原则
        """
        diff_buf = []
        for i in range(dim):
            diff_buf.append(output_shape[i]-target_shape[i])
        cut_buf = []
        for i in range(dim):
            if diff_buf[i]<0:
                tmp = -diff_buf[i]
                left_pad = tmp//2
                right_pad = tmp-left_pad
                cut_buf.append([-left_pad,-right_pad])
            else:
                tmp = diff_buf[i]
                left_pad = tmp//2
                right_pad = tmp-left_pad
                cut_buf.append([left_pad,right_pad])
            # 不存在left right 符号互斥的情况
        return cut_buf,output_shape
    def cut_pad_3D(self,x,cut_buf):
        if(cut_buf[0][0]<0)or(cut_buf[0][1]<0):
            x = x[:,0-cut_buf[0][0]:int(x.shape[1])+cut_buf[0][1],:,:,:]
        else:
            x = tf.pad(x,[[0,0],cut_buf[0],[0,0],[0,0],[0,0]],"CONSTANT")

        if(cut_buf[1][0]<0)or(cut_buf[1][1]<0):
            x = x[:,:,0-cut_buf[1][0]:int(x.shape[2])+cut_buf[1][1],:,:]
        else:
            x = tf.pad(x,[[0,0],[0,0],cut_buf[1],[0,0],[0,0]],"CONSTANT")

        if(cut_buf[2][0]<0)or(cut_buf[2][1]<0):
            x = x[:,:,:,0-cut_buf[2][0]:int(x.shape[3])+cut_buf[2][1],:]
        else:
            x = tf.pad(x,[[0,0],[0,0],[0,0],cut_buf[2],[0,0]],"CONSTANT")  
        return x
        

#######################################################################################################################
if __name__ == "__main__":
    
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    
    x = tf.random.normal(shape=[5,2],mean=0.0,stddev=1.0)
    a = Dense(3,sn=True,use_bias=False)
    a.build(input_shape=[None,2])
    print(a(x))
    print(a.trainable_variables)

    print(x:=tf.random.normal(shape=[5,5], mean=0.0, stddev=1.0))
    dp = Dropout(rate=0.5)#在非1维度随机出0 然后在是1维度广播
    print(dp(x,training=True))
    print("##############################")
    b = tf.random.normal(shape=[7,3], mean=0.0, stddev=1.0)
    bn = BatchNormalization()
    print(b)
    print(tmp1:=bn(b))
    print(tmp2:=bn(b,training=False))
    print(tmp1-tmp2)
    print("##############################")
    b = tf.random.normal(shape=[7,3], mean=0.0, stddev=1.0)
    print(b)
    bn = BatchNormalization()
    print(tmp1:=bn(b))
    print(tmp2:=bn(b,training=True))
    print(tmp1-tmp2)
    
    cv3d = Conv3D(filters=16,kernel_size=[3,3,3],strides=[1,1,1],padding="SAME",use_bias=False)
    cv3d.build(input_shape=[None,64,64,64,3])
    print(x:=tf.reshape(tf.range(64*64*64*3.0),[1,64,64,64,3]))
    print(cv3d(x).shape)
    print(cv3d(x).dtype)
    cv3d = Conv3D(filters=16,kernel_size=[3,3,3],strides=[1,1,1],padding="SAME",use_bias=False,sn=True)
    cv3d.build(input_shape=[None,64,64,64,7])
    print(cv3d.kernel.shape)
    print(x:=tf.reshape(tf.range(64*64*64*7.0),[1,64,64,64,7]))
    print(cv3d(x).shape)
    print(cv3d(x).dtype)
    cv2d = Conv2D(filters=16,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=False,sn=True)
    cv2d.build(input_shape=[None,64,64,7])
    print(x:=tf.reshape(tf.range(64*64*7.0),[1,64,64,7]))
    print(cv2d(x).shape)
    print(cv2d(x).dtype)

    a = Conv3DTranspose(100,kernel_size=(3,3,3),strides=(2,2,2),padding="SAME",sn=True)
    a.build(input_shape=[None,4,4,4,5])
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(a(x).dtype)

    a = UpSample3D(filters=100,kernel_size=(3,3,3),strides=(2,2,2),padding="SAME",sn=True,use_bias=False)
    a.build(input_shape=[None,4,4,4,5])
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(a(x).dtype)

    a = Conv3DTranspose(100,kernel_size=(3,3,3),strides=(2,2,2),padding="VALID",sn=True)
    a.build(input_shape=[None,4,4,4,5])
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(a(x).dtype)

    a = UpSample3D(filters=100,kernel_size=(3,3,3),strides=(2,2,2),padding="VALID",sn=True,use_bias=True)
    a.build(input_shape=[None,4,4,4,5])
    x = tf.constant(tf.random.normal((1,4,4,4,5)))
    print(a(x).shape)
    print(a(x).dtype)

    a = Conv2DTranspose(100,kernel_size=(3,3),strides=(2,2),padding="VALID",sn=True)
    a.build(input_shape=[None,5,5,7])
    x = tf.constant(tf.random.normal((1,5,5,7)))
    print(a(x).shape)
    print(a(x).dtype)

    b = UpSample2D(sn=True,
                   filters=100,
                   kernel_size=(3,3),
                   strides=(2,2),
                   padding="VALID",
                   use_bias=False)
    b.build(input_shape=[None,5,5,7])
    x = tf.constant(tf.random.normal((1,5,5,7)))
    print(b(x).shape)
    print(b(x).dtype)

    a = Conv2DTranspose(100,kernel_size=(3,3),strides=(2,2),padding="SAME",sn=True)
    a.build(input_shape=[None,5,5,7])
    x = tf.constant(tf.random.normal((1,5,5,7)))
    print(a(x).shape)
    print(a(x).dtype)
    b = UpSample2D(sn=True,
                   filters=100,
                   kernel_size=(3,3),
                   strides=(2,2),
                   padding="SAME",
                   use_bias=False)
    b.build(input_shape=[None,5,5,7])
    x = tf.constant(tf.random.normal((1,5,5,7)))
    print(b(x).shape)
    print(b(x).dtype)

    # # print(a.output_shape)
    # print(a.compute_output_shape([None,6,6,7]))
    # for p in ["same","VALID"]:
    #     for k in range(1,100):
    #         for s in range(1,10):
    #             for i in range(k+1,200):
    #                 # print("***************************")
    #                 a = Conv2DTranspose(100,kernel_size=(k,k),strides=(s,s),padding=p,sn=True)
    #                 # print(tmp1:=a.compute_output_shape([None,i,i,7]))
    #                 # print(tmp2:=UpSample2D.cal_output_shape_zp(input_shape=[i,i],kernel_size=[k,k],strides=[s,s],padding=p))
    #                 tmp1=a.compute_output_shape([None,i,i,7])
    #                 tmp2=UpSample2D.cal_output_shape_zp(input_shape=[i,i],kernel_size=[k,k],strides=[s,s],padding=p)
    #                 if (tmp1[1]!=tmp2[0])and(tmp1[2]!=tmp2[1]):
    #                     print(tmp1,tmp2)

