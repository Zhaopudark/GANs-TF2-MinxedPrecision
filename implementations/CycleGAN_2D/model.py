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
global_input_X_shape = [1,128,128,1]
global_input_Y_shape = [1,128,128,1]
################################################
class CycleGAN(tf.keras.Model):
    """
    模型只负责给定训练集和测试(验证)集后的操作
    """
    def __init__(self,
                train_set,
                test_set,
                loss_name="WGAN-GP",
                mixed_precision=False,
                learning_rate=2e-4,
                tmp_path=None,
                out_path=None):
        super(CycleGAN,self).__init__()
        #接收数据集和相关参数
        self.train_set = train_set
        self.test_set = test_set
        self.tmp_path = tmp_path
        self.out_path = out_path
        #定义模型
        self.G = networks.Generator(name="G_X2Y")
        self.F = networks.Generator(name="G_Y2X")
        if loss_name in ["WGAN-SN","WGAN-GP-SN"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=False,sn=True)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=False,sn=True)
            self.loss_name = loss_name[:-3]
        elif loss_name in ["WGAN","WGAN-GP"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=False,sn=False)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=False,sn=False)
            self.loss_name = loss_name
        elif loss_name in ["Vanilla","LSGAN"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=True,sn=False)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=True,sn=False)
            self.loss_name = loss_name
        else: 
            raise ValueError("Do not support the loss "+loss_name)

        self.model_list=[self.G,self.F,self.Dy,self.Dx]
        #定义损失函数 优化器 记录等
        self.gan_loss = GanLoss(self.loss_name)
        self.optimizers_list = self.optimizers_config(mixed_precision=mixed_precision,learning_rate=learning_rate)
        self.mixed_precision = mixed_precision
        self.matrics_list = self.matrics_config()
        self.checkpoint_config()
        self.get_seed()
    def build(self,X_shape,Y_shape):
        """
        input_shape必须切片 因为在底层会被当做各层的输出shape而被改动
        """
        self.G.build(input_shape=X_shape[:])#G X->Y
        self.Dy.build(input_shape=Y_shape[:])#Dy Y or != Y
        self.F.build(input_shape=Y_shape[:])#F Y->X
        self.Dx.build(input_shape=X_shape[:])#Dx X or != X
        self.built = True
    def optimizers_config(self,mixed_precision=False,learning_rate=2e-4):
        self.G_optimizer = Adam(2e-4)
        self.Dy_optimizer = Adam(2e-4)
        self.F_optimizer = Adam(2e-4)
        self.Dx_optimizer = Adam(2e-4)
        if mixed_precision:
            self.G_optimizer=self.G_optimizer.get_mixed_precision()
            self.Dy_optimizer=self.Dy_optimizer.get_mixed_precision()
            self.F_optimizer=self.F_optimizer.get_mixed_precision()
            self.Dx_optimizer=self.Dx_optimizer.get_mixed_precision()
        return [self.G_optimizer,self.Dy_optimizer,self.F_optimizer,self.Dx_optimizer]
    def matrics_config(self):
        current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_logdir = self.tmp_path+"/logs/" + current_time
        self.train_summary_writer = tf.summary.create_file_writer(train_logdir)
        self.m_psnr_X2Y = tf.keras.metrics.Mean('psnr_y', dtype=tf.float32)
        self.m_psnr_Y2X = tf.keras.metrics.Mean('psnr_x', dtype=tf.float32)
        self.m_ssim_X2Y = tf.keras.metrics.Mean('ssim_y', dtype=tf.float32)
        self.m_ssim_Y2X = tf.keras.metrics.Mean('ssim_x', dtype=tf.float32)
        return [self.m_psnr_X2Y,self.m_psnr_Y2X,self.m_ssim_X2Y,self.m_ssim_Y2X]
        # return None
    def checkpoint_config(self):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer=self.optimizers_list,model=self.model_list,dataset=self.train_set)
        self.manager = tf.train.CheckpointManager(self.ckpt,self.tmp_path+'/tf_ckpts', max_to_keep=3)
    def pix_gradient(self,x):
        x = tf.reshape(x,shape=[1,64,64,1])#在各batch和通道上进行像素梯度 对2D单通道而言其实没必要reshape
        dx,dy = tf.image.image_gradients(x)
        return dx,dy
    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),tf.TensorSpec(shape=[1],dtype=tf.uint32)])
    def train_step_D(self,trainX,trainY,step):
        with tf.GradientTape(persistent=True) as D_tape:
            GeneratedY = self.G(trainX)
            Dy_real_out = self.Dy(trainY)
            Dy_fake_out = self.Dy(GeneratedY)

            GeneratedX = self.F(trainY)
            Dx_real_out = self.Dx(trainX)
            Dx_fake_out = self.Dx(GeneratedX)

            e = tf.random.uniform((trainY.shape[0],1,1,1),0.0,1.0)
            mid_Y = e*trainY+(1-e)*GeneratedY
            with tf.GradientTape() as gradient_penaltyY:
                gradient_penaltyY.watch(mid_Y)
                inner_loss = self.Dy(mid_Y)
            penalty = gradient_penaltyY.gradient(inner_loss,mid_Y)
            penalty_normY = 10.0*tf.math.square(tf.maximum(tf.norm(penalty,ord='euclidean'),1.0)-1)# 这是我自己认为的  因为只有梯度大于1的才需要优化哇

            e = tf.random.uniform((trainX.shape[0],1,1,1),0.0,1.0)
            mid_X = e*trainX+(1-e)*GeneratedX
            with tf.GradientTape() as gradient_penaltyX:
                gradient_penaltyX.watch(mid_X)
                inner_loss = self.Dx(mid_X)
            penalty = gradient_penaltyX.gradient(inner_loss,mid_X)
            penalty_normX = 10.0*tf.math.square(tf.maximum(tf.norm(penalty,ord='euclidean'),1.0)-1)

            Dy_loss = self.gan_loss.DiscriminatorLoss(Dy_real_out,Dy_fake_out)+tf.reduce_mean(penalty_normY)
            Dx_loss = self.gan_loss.DiscriminatorLoss(Dx_real_out,Dx_fake_out)+tf.reduce_mean(penalty_normX)

            if self.mixed_precision:
                scaled_Dy_loss = self.Dy_optimizer.get_scaled_loss(Dy_loss)
                scaled_Dx_loss = self.Dx_optimizer.get_scaled_loss(Dx_loss)

        if self.mixed_precision:
            scaled_gradients_of_Dy=D_tape.gradient(scaled_Dy_loss,self.Dy.trainable_variables)
            scaled_gradients_of_Dx=D_tape.gradient(scaled_Dx_loss,self.Dx.trainable_variables)
            gradients_of_Dy = self.Dy_optimizer.get_unscaled_gradients(scaled_gradients_of_Dy)
            gradients_of_Dx = self.Dx_optimizer.get_unscaled_gradients(scaled_gradients_of_Dx)
        else:
            gradients_of_Dy = D_tape.gradient(Dy_loss,self.Dy.trainable_variables)
            gradients_of_Dx = D_tape.gradient(Dx_loss,self.Dx.trainable_variables)

        self.Dy_optimizer.apply_gradients(zip(gradients_of_Dy,self.Dy.trainable_variables))
        self.Dx_optimizer.apply_gradients(zip(gradients_of_Dx,self.Dx.trainable_variables))
        return Dy_loss,Dx_loss
    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),tf.TensorSpec(shape=[1],dtype=tf.uint32)])
    def train_step_G(self,trainX,trainY,step):
        with tf.GradientTape(persistent=True) as G_tape:
            GeneratedY = self.G(trainX)
            # Dy_real_out = self.Dy(trainY)
            Dy_fake_out = self.Dy(GeneratedY)

            GeneratedX = self.F(trainY)
            # Dx_real_out = self.Dx(trainX)
            Dx_fake_out = self.Dx(GeneratedX)

            cycle_consistent_loss_X2Y = tf.reduce_mean(tf.abs(self.F(GeneratedY)-trainX))
            cycle_consistent_loss_Y2X = tf.reduce_mean(tf.abs(self.G(GeneratedX)-trainY))
            cycle_consistent = cycle_consistent_loss_X2Y+cycle_consistent_loss_Y2X

            if step>=0:#先不进行像素梯度和重建损失的使用
                cycle_l = 10.0
            else:
                cycle_l = 10.0
            G_loss = self.gan_loss.GeneratorLoss(Dy_fake_out)+cycle_l*(cycle_consistent)
            F_loss = self.gan_loss.GeneratorLoss(Dx_fake_out)+cycle_l*(cycle_consistent)

            if self.mixed_precision:
                scaled_G_loss = self.G_optimizer.get_scaled_loss(G_loss)
                scaled_F_loss = self.F_optimizer.get_scaled_loss(F_loss)
        if self.mixed_precision:
            scaled_gradients_of_G=G_tape.gradient(scaled_G_loss,self.G.trainable_variables)
            scaled_gradients_of_F=G_tape.gradient(scaled_F_loss,self.F.trainable_variables)
            gradients_of_G = self.G_optimizer.get_unscaled_gradients(scaled_gradients_of_G)
            gradients_of_F = self.F_optimizer.get_unscaled_gradients(scaled_gradients_of_F)

        else:
            gradients_of_G = G_tape.gradient(G_loss,self.G.trainable_variables)
            gradients_of_F = G_tape.gradient(F_loss,self.F.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_G,self.G.trainable_variables))
        self.F_optimizer.apply_gradients(zip(gradients_of_F,self.F.trainable_variables))
        return G_loss,F_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),tf.TensorSpec(shape=[1],dtype=tf.uint32)])
    def train_step(self,trainX,trainY,step):
        with tf.GradientTape(persistent=True) as cycle_type:
            GeneratedY = self.G(trainX)
            Dy_real_out = self.Dy(trainY)
            Dy_fake_out = self.Dy(GeneratedY)

            GeneratedX = self.F(trainY)
            Dx_real_out = self.Dx(trainX)
            Dx_fake_out = self.Dx(GeneratedX)

            cycle_consistent_loss_X2Y = tf.reduce_mean(tf.abs(self.F(GeneratedY)-trainX))
            cycle_consistent_loss_Y2X = tf.reduce_mean(tf.abs(self.G(GeneratedX)-trainY))
            cycle_consistent = cycle_consistent_loss_X2Y+cycle_consistent_loss_Y2X

            if step>=0:#先不进行像素梯度和重建损失的使用
                cycle_l = 10.0
            else:
                cycle_l = 10.0
            Dy_loss = self.gan_loss.DiscriminatorLoss(Dy_real_out,Dy_fake_out)
            Dx_loss = self.gan_loss.DiscriminatorLoss(Dx_real_out,Dx_fake_out)
            G_loss = self.gan_loss.GeneratorLoss(Dy_fake_out)+cycle_l*(cycle_consistent)
            F_loss = self.gan_loss.GeneratorLoss(Dx_fake_out)+cycle_l*(cycle_consistent)

        gradients_of_Dy = cycle_type.gradient(Dy_loss,self.Dy.trainable_variables)
        gradients_of_Dx = cycle_type.gradient(Dx_loss,self.Dx.trainable_variables)
        gradients_of_G = cycle_type.gradient(G_loss,self.G.trainable_variables)
        gradients_of_F = cycle_type.gradient(F_loss,self.F.trainable_variables)
        self.Dy_optimizer.apply_gradients(zip(gradients_of_Dy,self.Dy.trainable_variables))
        self.Dx_optimizer.apply_gradients(zip(gradients_of_Dx,self.Dx.trainable_variables))
        self.G_optimizer.apply_gradients(zip(gradients_of_G,self.G.trainable_variables))
        self.F_optimizer.apply_gradients(zip(gradients_of_F,self.F.trainable_variables))
        return G_loss,Dy_loss,F_loss,Dx_loss
    def train(self,epoches):
        self.ckpt.restore(self.manager.latest_checkpoint)
        for _ in range(epoches):
            start = time.time()
            for trainX,trainY in self.train_set:
                self.ckpt.step.assign_add(1)
                step = int(self.ckpt.step)
                if self.loss_name in ["WGAN","WGAN-GP"]:
                    for __ in range(1):
                        Dy_loss,Dx_loss = self.train_step_D(trainX,trainY,tf.constant(step,shape=[1],dtype=tf.uint32))
                    for __ in range(1):
                        G_loss,F_loss = self.train_step_G(trainX,trainY,tf.constant(step,shape=[1],dtype=tf.uint32))
                elif self.loss_name in ["Vanilla","LSGAN"]:
                    G_loss,Dy_loss,F_loss,Dx_loss = self.train_step(trainX,trainY,tf.constant(step,shape=[1],dtype=tf.uint32))
                else:
                    raise ValueError("Inner Error")
                
                if step % 100 == 0:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(step,save_path))
                    
                    self.G.save_weights(self.tmp_path+'/weights_saved/G.ckpt')
                    self.Dy.save_weights(self.tmp_path+'/weights_saved/Dy.ckpt')
                    self.F.save_weights(self.tmp_path+'/weights_saved/F.ckpt')
                    self.Dx.save_weights(self.tmp_path+'/weights_saved/Dx.ckpt')
                    
                    self.wirte_summary(step=step,
                                       seed=self.seed,
                                       G=self.G,
                                       F=self.F,
                                       G_loss=G_loss,
                                       Dy_loss=Dy_loss,
                                       F_loss=F_loss,
                                       Dx_loss=Dx_loss,
                                       out_path=self.out_path)

                    print ('Time to next 100 step {} is {} sec'.format(step,time.time()-start))
                    start = time.time()
    def get_seed(self):
        seed_get = iter(self.test_set)
        seed = next(seed_get)
        print(seed[0].shape,seed[1].dtype)
        plt.imshow(seed[0][0,:,:,0],cmap='gray')
        plt.show()
        plt.imshow(seed[1][0,:,:,0],cmap='gray')
        plt.show()
        self.seed = seed 

    def wirte_summary(self,step,seed,G,F,G_loss,Dy_loss,F_loss,Dx_loss,out_path):
        testX,testY= seed
        GeneratedY = G(testX)
        GeneratedX = F(testY)
        plt.figure(figsize=(5,5))#图片大一点才可以承载像素
        plt.subplot(2,2,1)
        plt.title('real X')
        plt.imshow(testX[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.title('fake Y')
        plt.imshow(GeneratedY[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.title('fake X')
        plt.imshow(GeneratedX[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.title('real Y')
        plt.imshow(testY[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.savefig(out_path+'/image_at_{}.png'.format(step))
        plt.close()
        img = Image.open(out_path+'/image_at_{}.png'.format(step))
        img = tf.reshape(np.array(img),shape=(1,500,500,4))

        with self.train_summary_writer.as_default():
            ##########################
            self.m_psnr_X2Y(tf.image.psnr(GeneratedY,testY,1.0,name=None))
            self.m_psnr_Y2X(tf.image.psnr(GeneratedX,testX,1.0,name=None)) 
            self.m_ssim_X2Y(tf.image.ssim(GeneratedY,testY,1, filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)) 
            self.m_ssim_Y2X(tf.image.ssim(GeneratedX,testX,1, filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)) 
            tf.summary.scalar('G_loss',G_loss,step=step)
            tf.summary.scalar('Dy_loss',Dy_loss,step=step)
            tf.summary.scalar('F_loss',F_loss,step=step)
            tf.summary.scalar('Dx_loss',Dx_loss,step=step)
            tf.summary.scalar('test_psnr_y', self.m_psnr_X2Y.result(), step=step) 
            tf.summary.scalar('test_psnr_x', self.m_psnr_Y2X.result(), step=step)
            tf.summary.scalar('test_ssim_y', self.m_ssim_X2Y.result(), step=step) 
            tf.summary.scalar('test_ssim_x', self.m_ssim_Y2X.result(), step=step)  
            tf.summary.image("img",img,step=step)

        ##########################
        self.m_psnr_X2Y.reset_states()
        self.m_psnr_Y2X.reset_states()
        self.m_ssim_X2Y.reset_states()
        self.m_ssim_Y2X.reset_states()

        
    