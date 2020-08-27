"""
定义Spectral Normal的计算函数
构建一个SN类 返回多种不同的SN计算
"""
import tensorflow  as tf
class SpectralNormalization():
    def __init__(self):
        """
        反卷积层不需要SN 因为反卷积不出现在判别器中
        但是为了开发的完整性 增加反卷积的真谱范数计算的选择
        卷积的卷积核的维度意义如下
        2D
        [filter_height, filter_width, in_channels, out_channels]
        3D
        [filter_depth, filter_height, filter_width, in_channels, out_channels]
        转置卷积的卷积核的维度意义如下
        2D
        [height, width, output_channels, in_channels]
        3D
        [depth, height, width, output_channels, in_channels] 
        """
        pass 
    @classmethod
    def SN_Conv3D(cls,u_shape,v_shape,weight,strides,padding,iter_k=5,is_transpose=False):
        """
        不论本层是卷积操作或者反卷积操作，都是一个稀疏矩阵W的矩阵乘法
        Y=W*X
        记u0 = like(y)的单位向量
        (以下计算需要计算后化为单位向量，为了简洁不写出单位化计算步骤)
        v1 = W.T*u0
        u1 = W*v1
        v2 = W.T*u1
        u2 = W*v2 
        ...
        ...
        v_n = W.T*u_n-1
        u_n = W*v_n
        v_n+1 = W.T*u_n
        
        u_n.T*W*W.T*u_n=l1(W*W.T最大特征值的近似)
        而W.T*U_n 就是v_n+1
        对于卷积而言 W.T 第一步就是转置卷积
        对于转置卷积而言 W.T 第一步就是卷积
        """
        if is_transpose:
            u_n = tf.random.normal(shape=u_shape)
            u_n = u_n/tf.linalg.norm(u_n,ord='euclidean')
            for i in range(iter_k):
                v_n = tf.nn.conv3d(input=u_n,filters=weight,strides=strides,padding=padding)
                v_n = v_n/tf.linalg.norm(v_n,ord='euclidean')
                u_n = tf.nn.conv3d_transpose(input=v_n,filters=weight,output_shape=u_shape,strides=strides,padding=padding)
                u_n = u_n/tf.linalg.norm(u_n,ord='euclidean')
            v_n = tf.nn.conv3d(input=u_n,filters=weight,strides=strides,padding=padding)#v_n+1
            v_n = tf.reshape(v_n,[-1])
            return tf.linalg.norm(v_n,ord='euclidean')
        else:
            u_n = tf.random.normal(shape=u_shape)
            u_n = u_n/tf.linalg.norm(u_n,ord='euclidean')
            for i in range(iter_k):
                v_n = tf.nn.conv3d_transpose(input=u_n,filters=weight,output_shape=v_shape,strides=strides,padding=padding)
                v_n = v_n/tf.linalg.norm(v_n,ord='euclidean')
                u_n = tf.nn.conv3d(input=v_n,filters=weight,strides=strides,padding=padding)
                u_n = u_n/tf.linalg.norm(u_n,ord='euclidean')
            v_n = tf.nn.conv3d_transpose(input=u_n,filters=weight,output_shape=v_shape,strides=strides,padding=padding)
            v_n = tf.reshape(v_n,[-1])
            return tf.linalg.norm(v_n,ord='euclidean')

    @classmethod
    def SN(cls,weight,iter_k=5):
        w = tf.reshape(weight,[weight.shape[0],-1])
        #v=w^H * u 所以u就是与w第一维度同
        u_n = tf.random.normal(shape=[w.shape[0],1]) 
        u_n = u_n/tf.linalg.norm(u_n,ord='euclidean')
        for i in range(iter_k):
            #v_n = W^H*u_(n-1) 归一化
            v_n = tf.linalg.matmul(tf.transpose(w),u_n)
            v_n = v_n/tf.linalg.norm(v_n,ord='euclidean')
            #u_n = W*v_n 归一化
            u_n = tf.linalg.matmul(w,v_n)
            u_n = u_n/tf.linalg.norm(u_n,ord='euclidean')
        #v_(n+1) = W^H*u_n 归一化
        v_n = tf.linalg.matmul(tf.transpose(w),u_n)
        v_n = v_n/tf.linalg.norm(v_n,ord='euclidean')
        #v^H w^H u
        sigmma = tf.matmul(tf.matmul(tf.transpose(v_n),tf.transpose(w)),u_n)
        return sigmma


