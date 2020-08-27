"""
定义GANs的相关损失函数类
调用直接返回相应的损失函数计算
Vanilla
WGAN
WGAN-GP
LSGAN
RSGAN
"""
import tensorflow as tf 
class GanLoss():
    """
    原则
    返回的loss必定被minimize 所以对应于公式中的符号在本类中直接修改
    """
    def __init__(self,loss_name):
        self.loss_name = loss_name
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    def DiscriminatorLoss(self,real_output,fake_output):
        if self.loss_name == "Vanilla":
            real_loss = self.cross_entropy(tf.ones_like(real_output),real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output),fake_output)
            total_loss = real_loss + fake_loss
            return total_loss
        elif self.loss_name == "WGAN":
            total_loss = -tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)#用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
            return total_loss
        elif self.loss_name == "WGAN-GP":
            total_loss = -tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)#用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
            return total_loss
        elif self.loss_name == "LSGAN":
            total_loss = (tf.reduce_mean(tf.math.squared_difference(real_output,1))+tf.reduce_mean(tf.math.squared_difference(fake_output,0)))*0.5
            return total_loss
        else:
            raise ValueError("Do not support the loss "+self.loss_name)
    def GeneratorLoss(self,fake_output):
        if self.loss_name == "Vanilla":
            return self.cross_entropy(tf.ones_like(fake_output),fake_output)
        elif self.loss_name == "WGAN":
            return -tf.reduce_mean(fake_output)
        elif self.loss_name == "WGAN-GP":
            return -tf.reduce_mean(fake_output)
        elif self.loss_name == "LSGAN":
            return tf.reduce_mean(tf.math.squared_difference(fake_output,1))#*0.5
        else:
            raise ValueError("Do not support the loss "+self.loss_name)
