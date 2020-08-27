import tensorflow as tf 
from tensorflow.keras.mixed_precision import experimental as mixed_precision
class Adam(tf.keras.optimizers.Adam):
    def __init__(self,*args,**kwargs):
        super(Adam,self).__init__(*args,**kwargs)
    def get_mixed_precision(self):
        optimizer = mixed_precision.LossScaleOptimizer(self,loss_scale='dynamic')
        return optimizer
if __name__ == "__main__":
    optimizer = tf.keras.optimizers.Adam(2e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    optimizer = Adam(2e-4)
    print(optimizer)
    optimizer = optimizer.get_mixed_precision()
    print(optimizer)