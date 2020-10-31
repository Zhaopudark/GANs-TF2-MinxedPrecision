import tensorflow as tf 
import blocks_2D as blocks
###################################################################
class Generator(tf.keras.Model):
    def __init__(self,**kwargs):
        super(Generator,self).__init__()
        self.block_list = []
        base_v = 64
        self.block_list.append(blocks.c7s1_k_2D(filters=base_v,activation="relu",**kwargs))
        self.block_list.append(blocks.dk_2D(filters=base_v*2,**kwargs))
        self.block_list.append(blocks.dk_2D(filters=base_v*4,**kwargs))
        self.block_list.append(blocks.n_res_blocks_2D(filters=base_v*4,n=9,**kwargs))
        self.block_list.append(blocks.uk_2D(filters=base_v*2,**kwargs))
        self.block_list.append(blocks.uk_2D(filters=base_v,**kwargs))
        self.block_list.append(blocks.c7s1_k_2D(filters=3,activation="sigmoid",**kwargs))
        self.last_activitation=tf.keras.layers.Activation('linear',dtype='float32')
    def build(self,input_shape):
        for item in self.block_list:
            item.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        """
        Args:
            input: batch_size x width x height x 3
        Returns:
            output: same size as input
        """
        in_put = x
        for item in self.block_list:
            out_put = item(in_put,training=training)
            in_put = out_put
        out_put = self.last_activitation(in_put)
        return out_put
class Discriminator(tf.keras.Model):
    def __init__(self,use_sigmoid=True,**kwargs):
        super(Discriminator,self).__init__()
        self.block_list=[]
        base_v = 64
        self.block_list.append(blocks.ck_2D(filters=base_v,norm=False,**kwargs))#1
        self.block_list.append(blocks.ck_2D(filters=base_v*2,**kwargs))#3
        self.block_list.append(blocks.ck_2D(filters=base_v*4,**kwargs))#3
        self.block_list.append(blocks.ck_2D(filters=base_v*8,**kwargs))#3
        self.block_list.append(blocks.last_conv_2D(use_sigmoid=use_sigmoid,**kwargs))#2
        self.last_activitation=tf.keras.layers.Activation('linear',dtype='float32')
    def build(self,input_shape):
        for item in self.block_list:
            item.build(input_shape=input_shape)
        self.built = True
    def call(self,x,training=True):
        """
        Args:
            input: batch_size x width x height x 3
        Returns:
            output: same size as input
        """
        in_put = x
        for item in self.block_list:
            out_put = item(in_put,training=training)
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
    g.build(input_shape=[None,256,256,1])
    z1 = tf.random.normal(shape=[1,256,256,1])
    print(g(z1).shape)
    print(g(z1).dtype)
    print(len(g.trainable_variables))
    d = Discriminator(name="dd")
    d.build(input_shape=[None,256,256,1])
    z2 = tf.random.normal(shape=[1,256,256,1])
    print(d(z2).shape)
    print(d(z2).dtype)
    print(len(d.trainable_variables))
    g.summary()
    d.summary()
    tmp = g(z1).numpy()
    print(tmp.min(),tmp.max())
    import time
    start = time.time()
    for _  in range(50):
        g(z1)
        g(z1)
        d(z2)
        d(z2)
    print(time.time()-start)
