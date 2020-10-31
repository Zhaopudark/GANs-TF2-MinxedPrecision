import networks 
import tensorflow as tf
import time
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../../'))
from basemodels.GanLosses import GanLoss
from basemodels.GanOptimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
from PIL import Image
import datetime
###############global paraments###################
"""
记录那些无法被模型定义传递的参数
尤其是@tf.function() 中需要的参数
学习率与损失函数系数则应当在模型训练过程中予以控制
"""
global_input_X_shape = [None,100]
global_input_Y_shape = [None,28,28,1]
################################################
class DCGAN(tf.keras.Model):
    """
    模型只负责给定训练集和测试(验证)集后的操作
    """
    def __init__(self,
                train_set,
                test_set,
                loss_name="Vanilla",
                mixed_precision=False,
                learning_rate=2e-4,
                tmp_path=None,
                out_path=None):
        super(DCGAN,self).__init__()
        #接收数据集和相关参数
        self.train_set = train_set
        self.test_set = test_set
        self.tmp_path = tmp_path
        self.out_path = out_path
        #定义模型
        self.G = networks.Generator(name="G")
        if loss_name in ["WGAN-SN","WGAN-GP-SN"]:
            self.D = networks.Discriminator(name="If_is_real",use_sigmoid=False,sn=True)
            self.loss_name = loss_name[:-3]
        elif loss_name in ["WGAN","WGAN-GP"]:
            self.D = networks.Discriminator(name="If_is_real",use_sigmoid=False,sn=False)
            self.loss_name = loss_name
        elif loss_name in ["Vanilla-SN","LSGAN-SN"]:
            self.D = networks.Discriminator(name="If_is_real",use_sigmoid=True,sn=True)
            self.loss_name = loss_name[:-3]
        elif loss_name in ["Vanilla","LSGAN"]:
            self.D = networks.Discriminator(name="If_is_real",use_sigmoid=True,sn=False)
            self.loss_name = loss_name
        else: 
            raise ValueError("Do not support the loss "+loss_name)

        self.model_list=[self.G,self.D]
        #定义损失函数 优化器 记录等
        self.gan_loss = GanLoss(self.loss_name)
        self.optimizers_list = self.optimizers_config(mixed_precision=mixed_precision,learning_rate=learning_rate)
        self.mixed_precision = mixed_precision
        self.matrics_list = self.matrics_config()
        self.checkpoint_config()
        self.get_seed()
    def build(self,input_shape_G,input_shape_D):
        """
        input_shape必须切片 因为在底层会被当做各层的输出shape而被改动
        """
        self.G.build(input_shape=input_shape_G[:])#G X->Y
        self.D.build(input_shape=input_shape_D[:])#D Y or not Y
        self.built = True
    def optimizers_config(self,mixed_precision=False,learning_rate=2e-4):
        self.G_optimizer = Adam(2e-4)
        self.D_optimizer = Adam(2e-4)
        if mixed_precision:
            self.G_optimizer=self.G_optimizer.get_mixed_precision()
            self.D_optimizer=self.D_optimizer.get_mixed_precision()
        return [self.G_optimizer,self.D_optimizer]
    def matrics_config(self):
        current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_logdir = self.tmp_path+"/logs/" + current_time
        self.train_summary_writer = tf.summary.create_file_writer(train_logdir)
        return []
        # return None
    def checkpoint_config(self):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer=self.optimizers_list,model=self.model_list,dataset=self.train_set)
        self.manager = tf.train.CheckpointManager(self.ckpt,self.tmp_path+'/tf_ckpts', max_to_keep=3)
    def pix_gradient(self,x):
        x = tf.reshape(x,shape=[1,64,64,1])#在各batch和通道上进行像素梯度 对2D单通道而言其实没必要reshape
        dx,dy = tf.image.image_gradients(x)
        return dx,dy

    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),\
                                  tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),\
                                  tf.TensorSpec(shape=[4],dtype=tf.int32),\
                                  tf.TensorSpec(shape=[1],dtype=tf.uint32)])
    def train_step_D(self,trainX,trainY,y_shape,step):
        with tf.GradientTape(persistent=True) as D_tape:
            GeneratedY = self.G(trainX)
            D_real_out = self.D(trainY)
            D_fake_out = self.D(GeneratedY)

            e = tf.random.uniform(shape=y_shape,minval=0.0,maxval=1.0)
            mid_Y = e*trainY+(1-e)*GeneratedY
            with tf.GradientTape() as GP:
                GP.watch(mid_Y)
                inner_loss = self.D(mid_Y)
            penalty = GP.gradient(inner_loss,mid_Y)
            # penalty_norm = 10.0*tf.math.square(tf.maximum(tf.norm(penalty,ord='euclidean'),1.0)-1) #
            penalty_norm = 2.0*tf.math.square(tf.norm(penalty,ord='euclidean')-1)#这是按照算法愿意
            D_loss = self.gan_loss.DiscriminatorLoss(D_real_out,D_fake_out)+tf.reduce_mean(penalty_norm)

            if self.mixed_precision:
                scaled_D_loss = self.D_optimizer.get_scaled_loss(D_loss)
        if self.mixed_precision:
            scaled_gradients_of_D=D_tape.gradient(scaled_D_loss,self.D.trainable_variables)
            gradients_of_D = self.D_optimizer.get_unscaled_gradients(scaled_gradients_of_D)
        else:
            gradients_of_D = D_tape.gradient(D_loss,self.D.trainable_variables)
    
        self.D_optimizer.apply_gradients(zip(gradients_of_D,self.D.trainable_variables))

        return D_loss

    
    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),\
                                  tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),\
                                  tf.TensorSpec(shape=[4],dtype=tf.int32),\
                                  tf.TensorSpec(shape=[1],dtype=tf.uint32)])
    def train_step_G(self,trainX,trainY,y_shape,step):
        with tf.GradientTape(persistent=True) as G_tape:
            GeneratedY = self.G(trainX)
            # Dy_real_out = self.Dy(trainY)
            D_fake_out = self.D(GeneratedY)

            G_loss = self.gan_loss.GeneratorLoss(D_fake_out)

            if self.mixed_precision:
                scaled_G_loss = self.G_optimizer.get_scaled_loss(G_loss)
        if self.mixed_precision:
            scaled_gradients_of_G=G_tape.gradient(scaled_G_loss,self.G.trainable_variables)
            gradients_of_G = self.G_optimizer.get_unscaled_gradients(scaled_gradients_of_G)
        else:
            gradients_of_G = G_tape.gradient(G_loss,self.G.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_G,self.G.trainable_variables))
        return G_loss
    

    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),\
                                  tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),\
                                  tf.TensorSpec(shape=[4],dtype=tf.int32),\
                                  tf.TensorSpec(shape=[1],dtype=tf.uint32)])
    def train_step(self,trainX,trainY,y_shape,step):
        with tf.GradientTape(persistent=True) as gan_type:
            GeneratedY = self.G(trainX)
            D_real_out = self.D(trainY)
            D_fake_out = self.D(GeneratedY)

            D_loss = self.gan_loss.DiscriminatorLoss(D_real_out,D_fake_out)
            G_loss = self.gan_loss.GeneratorLoss(D_fake_out)

            if self.mixed_precision:
                scaled_D_loss = self.D_optimizer.get_scaled_loss(D_loss)
                scaled_G_loss = self.G_optimizer.get_scaled_loss(G_loss)

        if self.mixed_precision:
            scaled_gradients_of_D=gan_type.gradient(scaled_D_loss,self.D.trainable_variables)
            scaled_gradients_of_G=gan_type.gradient(scaled_G_loss,self.G.trainable_variables)
            gradients_of_D = self.D_optimizer.get_unscaled_gradients(scaled_gradients_of_D)
            gradients_of_G = self.G_optimizer.get_unscaled_gradients(scaled_gradients_of_G)
        else:
            gradients_of_D = gan_type.gradient(D_loss,self.D.trainable_variables)
            gradients_of_G = gan_type.gradient(G_loss,self.G.trainable_variables)
        
        self.D_optimizer.apply_gradients(zip(gradients_of_D,self.D.trainable_variables))
        self.G_optimizer.apply_gradients(zip(gradients_of_G,self.G.trainable_variables))
        return D_loss,G_loss

    def train(self,epoches):
        self.ckpt.restore(self.manager.latest_checkpoint)
        for _ in range(epoches):
            start = time.time()
            for trainX,trainY in self.train_set:
                self.ckpt.step.assign_add(1)
                step = int(self.ckpt.step)
                if self.loss_name in ["WGAN","WGAN-GP"]:
                    for __ in range(1):
                        D_loss = self.train_step_D(trainX,trainY,
                                                   tf.constant([trainY.shape[0],1,1,1],shape=[4],dtype=tf.int32),
                                                   tf.constant(step,shape=[1],dtype=tf.uint32))
                    for __ in range(3):
                        G_loss = self.train_step_G(trainX,trainY,
                                                   tf.constant([trainY.shape[0],1,1,1],shape=[4],dtype=tf.int32),
                                                   tf.constant(step,shape=[1],dtype=tf.uint32))
                elif self.loss_name in ["Vanilla","LSGAN"]:
                    D_loss,G_loss = self.train_step(trainX,trainY,
                                                    tf.constant([trainY.shape[0],1,1,1],shape=[4],dtype=tf.int32),
                                                    tf.constant(step,shape=[1],dtype=tf.uint32))
                else:
                    raise ValueError("Inner Error")
                
                if step % 100 == 0:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(step,save_path))
                    
                    self.G.save_weights(self.tmp_path+'/weights_saved/G.ckpt')
                    self.D.save_weights(self.tmp_path+'/weights_saved/D.ckpt')
                    
                    self.wirte_summary(step=step,
                                       seed=self.seed,
                                       G=self.G,
                                       G_loss=G_loss,
                                       D_loss=D_loss,
                                       out_path=self.out_path)
                    print ('Time to next 100 step {} is {} sec'.format(step,time.time()-start))
                    start = time.time()
    def test(self,take_nums):
        out_path = self.out_path+"/test"
        import os
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.ckpt.restore(self.manager.latest_checkpoint)
        seed_get = iter(self.test_set)
        for take in range(take_nums):
            plt.figure(figsize=(10,10))#图片大一点才可以承载像素
            for i in range(100):
                single_seed = next(seed_get)
                GeneratedY = self.G(single_seed,training=False)
                plt.subplot(10,10,(i+1))
                plt.imshow(GeneratedY[0,:,:,0],cmap='gray')
                plt.axis('off')
            plt.savefig(out_path+'/image_at_{}.png'.format(take))
            plt.close()

    def get_seed(self):
        self.seed = []
        seed_get = iter(self.test_set)
        for _ in range(100):
            seed = next(seed_get)
            self.seed.append(seed) 

    def wirte_summary(self,step,seed,G,G_loss,D_loss,out_path):
        plt.figure(figsize=(10,10))#图片大一点才可以承载像素
        for i,single_seed in enumerate(seed):
            GeneratedY = G(single_seed,training=False)
            plt.subplot(10,10,(i+1))
            plt.imshow(GeneratedY[0,:,:,0],cmap='gray')
            plt.axis('off')
        plt.savefig(out_path+'/image_at_{}.png'.format(step))
        plt.close()
        img = Image.open(out_path+'/image_at_{}.png'.format(step))
        img = tf.reshape(np.array(img),shape=(1,1000,1000,4))

        with self.train_summary_writer.as_default():
            ##########################
            tf.summary.scalar('G_loss',G_loss,step=step)
            tf.summary.scalar('D_loss',D_loss,step=step)
            tf.summary.image("img",img,step=step)

if __name__ == "__main__":
    y = tf.constant([128,1,1,1],shape=[4],dtype=tf.int32)
    print(list(y.numpy()))