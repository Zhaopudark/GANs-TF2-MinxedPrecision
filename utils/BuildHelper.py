import os
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.ErrorsDef import *
import tensorflow as tf
class Reconstruction():
    def __init__(self):
        "集中处理不规范的shape,给前面添加必要的None以规范化"
        "集中处理不规范的kernel_size"
        "做一个激活函数的集合,以调用类方法的形式返回一个类方法"
        "做一个变量初始化函数的集合"
        "做一个卷积的辅助计算函数，用于依据用户输入，给出真正的输出shape"
        "做一个转置卷积的辅助计算函数，计算过程中的输出shape"
    @classmethod
    def remake_shape(cls,shape,dims):
        try:
            if type(shape) == int:
                buf = [shape]
                for i in range(dims-1):
                    buf = [None]+buf
                return buf
            elif type(shape) == tuple:
                shape = list(shape)
                return cls.remake_shape(shape,dims)
            elif type(shape) == list:
                if len(shape) == dims:
                    return shape
                elif len(shape) < dims:
                    buf = shape
                    for i in range(dims-len(shape)):
                        buf = [None]+buf
                    return buf
                else:
                    raise ShapeError(shape)
            else:
                raise ShapeError(shape)
        except ShapeError as error:
                print(error.err_msg)
                print(error.shape)
        else:# normal condition
            pass
        finally:# any way
            pass
    @classmethod
    def remake_kernel_size(cls,kernel_size,dims):
        try:
            if type(kernel_size) == tuple:
                kernel_size = list(kernel_size)
                return cls.remake_kernel_size(kernel_size,dims)
            elif type(kernel_size) == list:
                if len(kernel_size) == dims:
                    return [1]+kernel_size+[1]
                elif len(kernel_size) == dims+2:
                    return kernel_size
                else:
                    raise KernelShapeError(kernel_size)
            else:
                raise KernelShapeError(kernel_size)
        except KernelShapeError as error:
                print(error.err_msg)
                print(error.shape)
        else:# normal condition
            pass
        finally:# any way
            pass
    @classmethod
    def remake_strides(cls,strides,dims):
        try:
            if type(strides) == tuple:
                strides = list(strides)
                return cls.remake_strides(strides,dims)
            elif type(strides) == list:
                if len(strides) == dims:
                    return [1]+strides+[1]
                elif len(strides) == dims+2:
                    return strides
                else:
                    raise StridesError(strides)
            else:
                raise StridesError(strides)
        except StridesError as error:
                print(error.err_msg)
                print(error.strides)
        else:# normal condition
            pass
        finally:# any way
            pass   
    @classmethod
    def activation(cls,activation):
        try:
            if activation == "relu":
                return tf.nn.relu
            elif activation == "leaky_relu":
                return tf.nn.leaky_relu
            elif activation == "sigmoid":
                return tf.nn.sigmoid
            elif activation == "tanh":
                return tf.nn.tanh
            elif activation == None:
                return lambda x:x
            else:
                raise ActivationError(activation)
        except ActivationError as error:
                print(error.err_msg)
                print(error.activation)
        else:# normal condition
            pass
        finally:# any way
            pass
    @classmethod
    def initializer(cls,initializer,*args,**kwargs):
        try:
            if initializer == "glorot_normal":
                return tf.keras.initializers.GlorotNormal(*args,**kwargs)
            elif initializer == "glorot_uniform":
                return tf.keras.initializers.GlorotUniform(*args,**kwargs)
            elif initializer == "random_normal":
                return tf.keras.initializers.RandomNormal(*args,**kwargs)
            elif initializer == "random_uniform":
                return tf.keras.initializers.RandomUniform(*args,**kwargs)
            else:
                raise InitializerError
        except InitializerError as error:
                print(error.err_msg)
                print(error.initializer)
        else:# normal condition
            pass
        finally:# any way
            pass
    @classmethod
    def ConvCalculation(cls,input_shape,filters,kernel_size,strides,padding,*args,**kwargs):
        """
        卷积过程的参数计算
        3D--BDHWC
        2D--BHWC
        input_shape [H,W]or[D,H,W] 不带Batch和Deepth
        filters 单值int
        kernel_size [Hy,Wx]or[Dz,Hy,Wx]
        strides [Hs2,Ws1]or[Ds3,Hs2,Ws1]
        padding "SAME","REFLECT","CONSTANT","SYMMETRIC",
        深度(卷积核个数)、单个卷积核大小、步长和pad方式后 计算相关的参数
        返回两个重要的内容(卷积输出)
        If padding == "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        If padding == "VALID": output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]). 向上取整 因为就算不够整除 第一个位置也是有的
        对于VALID的公式的解释 和中文教材另一个公式有出入 但是是一样的 即将第一个位置摆上卷积核后 如卷积核宽4 输入宽64 那么64-3后 有61个位置 可以给卷积核的最右端那一列进行移动 除以步长 计算剩下的位置即可 卷积核非最后一列的前列位置都是不考虑的 就是一个摆格子的算法
        而(input_spatial_shape[i]-spatial_filter_shape[i])//strides[i] +1 则是将第一个位置摆上卷积核后 计算剩余位置可以摆放的卷积核的个数 然后加上已经考虑的第一个 他们本质一致
        
        用pad+valid模拟same 
        """
        try:
            l = len(input_shape)
            padding_vect = []
            padding_list = ["SAME","REFLECT","CONSTANT","SYMMETRIC"]
            if padding == "VALID":
                for i in range(l):
                    padding_vect.append([0,0])
                padding = "CONSTANT"
            elif padding in padding_list:
                if padding=="SAME":
                    padding = "CONSTANT"#默认为一般的0 padding
                for i in range(l):
                    # padding目的就是保持一致 所以直接用结果推结论 不考虑中间的奇偶关系
                    pad_begin = kernel_size[i]//2#开头的padding是一定存在的 右边的padding
                    pad_end = kernel_size[i]//2
                    for j in range(0,pad_begin+pad_end+1,1):#kernel_size很大时 padding的大小有很多 找寻最小的padding方式
                        if tf.math.ceil((input_shape[i]+j-kernel_size[i]+1)/strides[i]) == tf.math.ceil(input_shape[i]/strides[i]):
                            break
                    """
                    tf内部的实现逻辑是  计算最小的padding数
                    (右端)后端先padding 然后 (左端)前端padding
                    实现时 begin = j//2 end=j-begin 即可
                    """
                    pad_begin = j//2
                    pad_end = j - pad_begin
                    padding_vect.append([pad_begin,pad_end])
            else:
                raise ConvParaError(padding)
            buf = []
            for i in range(l):
                buf.append((input_shape[i]+padding_vect[i][0]+padding_vect[i][1]-kernel_size[i])//strides[i]+1)
            out_shape = buf+[filters]
            return out_shape,padding,[[0,0]]+padding_vect+[[0,0]]
        except ConvParaError as error:
                print(error.err_msg)
                print(error.parameter)
        else:# normal condition
            pass
        finally:# any way
            pass
    @classmethod
    def ConvTransCheck(cls,input_shape,output_shape,filters,kernel_size,strides,padding,*args,**kwargs):
        """
        反卷积不再是使用pad+valid 模拟same padding 的方式 而是辅助计算卷积给定参数是否合理 是否满足卷积公式
        卷积公式是依旧成立的
        input_shape 是转置卷积的输入  等于正常卷积过程的输出
        """
        try:
            conv_output_shape = input_shape
            conv_input_shape = output_shape
            l = len(conv_input_shape)
            if padding == "VALID":
                for i in range(l):
                    if tf.math.ceil((conv_input_shape[i]-kernel_size[i]+1)/strides[i]) != conv_output_shape[i]:
                        raise ConvParaError((input_shape,output_shape))
            elif padding == "SAME":
                for i in range(l):
                    if tf.math.ceil(conv_input_shape[i]/strides[i]) != conv_output_shape[i]:
                        raise ConvParaError((input_shape,output_shape))
            else:
                raise ConvParaError(padding)
        except ConvParaError as error:
                print(error.err_msg)
                print(error.parameter)
                raise  ValueError("ConvTransCheck Not Passed because of bad conv paraments!")
        else:# normal condition
            pass
        finally:# any way
            pass
    @classmethod
    def Trans2UpsampleCal(cls,input_shape,output_shape,filters,kernel_size,strides,padding,*args,**kwargs):
        """
        反卷积过程转化为卷积的相关计算。
        满足转置卷积的参数条件后，开始卷积。

        希望up_op可以达到真正的大小 然后进行SAME卷积 需要深入分析

        x-1<ceil(x)<x+1
        当x为整数时  x=ceil(x)<x+1
        当x为非整数时 x<ceil(x)<x+1
        所以x<=ceil(x)<x+1
        原本指定SAME时 input_shape = ceil(output_shape/strides)
        output_shape/strides<=input_shape<output_shape/strides +1
        output_shape <=input_shape*strides<output_shape+strides
        input_shape*strides是上采样后得到的直接shape
        和output_shape目标之间存在差异 寻找办法使得卷积后得到output_shape 
        (input_shape*strides-kernelsize)/1 + 1 = input_shape*strides-kernelsize+1 <output_shape+strides-kernelsize+1

        原本指定VALID时 input_shape = ceil(output_shape[i]-kernel_size[i]+1)/strides[i])
        (output_shape-kernel_size+1)/strides<=input_shape< (output_shape-kernel_size+1)/strides +1
        output_shape-kernel_size+1<=input_shape*strides<output_shape-kernel_size+1+strides
        
        一般的 kernel_size>=3 strides<=2 delt=strides-kernelsize+1<=0
        本着对输入信息只增不减的原则 
        原本指定SAME时 input_shape*strides-kernelsize+1<output_shape+delt<=output_shape
        那么就存在正的padding方式 对input_shape*strides补齐 然后满足pad=valid,kernelsize不变,strides=1的卷积 实现指定的输出维度

        原本指定VALID时 input_shape*strides<output_shape+delt<=output_shape
        那么就存在正的padding方式 对input_shape*strides补齐到output_shape维度 然后进行pad=SAME,kernelsize不变,strides=1的卷积 实现指定的输出维度

        delt>0时 input_shape*strides如果小于output_shape 则做pad补齐 
                input_shape*strides如果大于output_shape 则做cut裁剪 
        """
        try:
            cls.ConvTransCheck(input_shape,output_shape,filters,kernel_size,strides,padding)
            l = len(input_shape)#dim
            padding_vect=[]
            padding_list = []
            for i in range(l):
                delt = strides[i]-kernel_size[i]+1
                if delt <= 0:
                    if padding == "VALID":
                        differ = output_shape[i]-input_shape[i]*strides[i]
                        pad_begin = differ//2
                        pad_end = differ - pad_begin
                        padding_vect.append([pad_begin,pad_end])
                        padding_list.append("SAME")
                    elif padding == "SAME":
                        differ = output_shape[i]-(input_shape[i]*strides[i]-kernel_size[i]+1)
                        pad_begin = differ//2
                        pad_end = differ - pad_begin
                        padding_vect.append([pad_begin,pad_end])
                        padding_list.append("VALID")
                    else:
                        pass
                else:
                    if input_shape[i]*strides[i] <= output_shape[i]:
                        differ = output_shape[i]-input_shape[i]*strides[i]
                        pad_begin = differ//2
                        pad_end = differ - pad_begin
                        padding_vect.append([pad_begin,pad_end])
                        padding_list.append("SAME")
                    else:
                        differ = -(output_shape[i]-input_shape[i]*strides[i])
                        pad_begin = differ//2
                        pad_end = differ - pad_begin
                        padding_vect.append([-pad_begin,-pad_end])
                        padding_list.append("SAME")
                        

            tmp = padding_list[0]
            for item in padding_list:
                if item != tmp:
                    raise ConvParaError("Padding ways not euqual "+str(item)+" and "+str(tmp))
            positive_pad = 0
            negative_pad = 0
            for item in padding_vect:
                if (item[0]<0)or(item[1]<0):
                    negative_pad += 1
                elif (item[0]>0)or(item[1]>0):
                    positive_pad += 1
                else:
                    pass 
            if (positive_pad>0)and(negative_pad>0):
                raise ConvParaError("Can not handle both positive_pad and negative_pad")
            if negative_pad>0:
                cut_flag = True
                padding = padding_list[0]
                return  padding,[[0,0]]+padding_vect+[[0,0]],cut_flag
            else:
                cut_flag = False
                padding = padding_list[0]
                return  padding,[[0,0]]+padding_vect+[[0,0]],cut_flag

        except ConvParaError as error:
                print(error.err_msg)
                print(error.parameter)
        else:# normal condition
            pass
        finally:# any way
            pass
if __name__ == "__main__":

    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    filters_zp=1
    inshape=[16,16]
    k_size = [4,4]
    strides = [8,8]
    # for i in range(16,128+1,1):
    #     for j in range(2,9+1,1):
    #         for k in range(1,4+1):
    #             inshape=[i,i]
    #             k_size = [j,j]
    #             strides = [k,k]
    #             padding = "SAME"
    #             # x = tf.random.normal([1]+inshape+[1])
    #             x = tf.ones([1]+inshape+[1])
    #             w = tf.ones(k_size+[1]+[filters_zp])
    #             y=tf.nn.conv2d(x,w,strides,padding)
    #             out_shape,padding,padding_vect=Reconstruction.ConvCalculation(input_shape=inshape,
    #                                     filters=filters_zp,
    #                                     kernel_size=k_size,
    #                                     strides=strides,
    #                                     padding=padding)
    #             x_ = tf.pad(x,padding_vect,"CONSTANT")
    #             y_ = tf.nn.conv2d(x_,w,strides,"VALID")
    #             temp = tf.reduce_mean((y-y_)[0,:,:,0])
    #             if temp !=0.0 :
    #                 print("**********")
    #                 print(padding_vect)
    #                 print(temp.numpy())
    #                 print(inshape,k_size,strides)
                            


    padding = "SAME"
    # x = tf.ones([1]+inshape+[1])
    x = tf.random.normal([1]+inshape+[1])
    w = tf.ones(k_size+[1]+[filters_zp])
    print(w[:,:,0,0])
    y=tf.nn.conv2d(x,w,strides,padding)
    
    # print(Reconstruction.ConvCalculation(input_shape=inshape,
                                        # filters=filters_zp,
                                        # kernel_size=k_size,
                                        # strides=strides,
                                        # padding=padding))
    out_shape,padding,padding_vect=Reconstruction.ConvCalculation(input_shape=inshape,
                                        filters=filters_zp,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding)
    print(padding_vect)
    x_ = tf.pad(x,padding_vect,"CONSTANT")
    print(x[0,:,:,0])
    print(x_[0,:,:,0])
    print(x.shape)
    print(x_.shape)
    y_ = tf.nn.conv2d(x_,w,strides,"VALID")
    print(y[0,:,:,0])
    print(y_[0,:,:,0])
    print(y.shape)
    print(y_.shape)
    print((y-y_)[0,:,:,0])
    # x conv2d(same) y                       
    # x pad(same)+conv2d(valid) y_ = y
    """
    x->y conv2d(same) === pad(same)[x_]+conv2d(valid)
    x->y conv2d(valid) === pad(0)+conv2d(valid)
    抽象出一个卷积操作 和tf的卷积一致 同时可以做reflect padding

    现在需要抽象出一个转置卷积操作 和tf的转置卷积一致 同时关注padding问题
    conv_transpose 特殊性在于 指定了same valid后 输入输出的维度必须满足一个逻辑约束才是正确可行的
    y->x conv2d_tanspose(valid)
    """

    x1 = tf.nn.conv2d_transpose(y_,w,x.shape,[1]+strides+[1],"SAME")
    print("padding_vect",padding_vect)
    
    x1_ = tf.pad(x1,padding_vect,"CONSTANT")
    y_2 = tf.pad(y_,[[0,0],[1,1],[1,1],[0,0]],"CONSTANT")
    print(y_2[0,:,:,0])
    x2 = tf.nn.conv2d_transpose(y_,w,x_.shape,[1]+strides+[1],"VALID")
    print(x1.shape)
    print(x1_.shape)
    print(x2.shape)
    print(x1[0,:,:,0])
    print(x1_[0,:,:,0])
    print(x2[0,:,:,0])

    print((x1_-x2)[0,:,:,0])
    filters_zp=[1]
    inshape=[16,16,16]
    k_size = [4,4,4]
    strides = [4,4,4]
    padding = "SAME"
    x = tf.random.normal([1]+inshape+[2])
    
    w = tf.random.normal(k_size+[2]+filters_zp)
    y=tf.nn.conv3d(x,w,[1]+strides+[1],padding,data_format='NDHWC')
    
    print(Reconstruction.ConvCalculation(input_shape=inshape,
                                        filters=filters_zp,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding))
    out_shape,padding,padding_vect=Reconstruction.ConvCalculation(input_shape=inshape,
                                        filters=filters_zp,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding)
    x_ = tf.pad(x,padding_vect,"CONSTANT")
    print(x.shape)
    print(x_.shape)
    y_ = tf.nn.conv3d(x_,w,[1]+strides+[1],"VALID")
    print(y.shape)
    print(y_.shape)
    print((y-y_)[0,:,:,:,0])

    x1 = tf.nn.conv3d_transpose(y_,w,x.shape,[1]+strides+[1],"SAME")
    print(x1[0,:,:,:,0])
    print(x1.shape)
    print("padding_vect",padding_vect)
    x1_ = tf.pad(x1,padding_vect,"CONSTANT")
    print(x1_[0,:,:,:,0])
    print(x1_.shape)
    x2 = tf.nn.conv3d_transpose(y_,w,x_.shape,[1]+strides+[1],"VALID")
    print(x2[0,:,:,:,0])
    print(x2.shape)
    print((x1_-x2)[0,:,:,:,0])
    padding,padding_vect,cut_flag = Reconstruction.Trans2UpsampleCal(
                                input_shape=[16,16],
                                output_shape=[32,32],
                                filters=8000,
                                kernel_size=[2,2],
                                strides=[2,2],
                                padding="SAME")
    print(padding,padding_vect,cut_flag)
    padding,padding_vect,cut_flag = Reconstruction.Trans2UpsampleCal(
                                input_shape=[16,16],
                                output_shape=[31,31],
                                filters=8000,
                                kernel_size=[2,2],
                                strides=[2,2],
                                padding="SAME")
    print(padding,padding_vect,cut_flag)


   