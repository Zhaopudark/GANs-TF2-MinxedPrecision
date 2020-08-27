import tensorflow as tf
import time
import os 
import sys
import model_nature as model 
 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../../'))

import datasets.Img2ImgPipeLine as train_dataset

physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

######################################################################################################


train_path_A = "G:\\Datasets\\Img2Img\\horse2zebra\\trainA"
train_path_B = "G:\\Datasets\\Img2Img\\horse2zebra\\trainB"
# test_path_A = "G:\\Datasets\\Img2Img\\horse2zebra\\testA"
# test_path_B = "G:\\Datasets\\Img2Img\\horse2zebra\\testB"
test_path_A = "G:\\Datasets\\Img2Img\\horse2zebra\\trainA"
test_path_B = "G:\\Datasets\\Img2Img\\horse2zebra\\trainB"
tmp_path = "D:/Work/Codes_tmp/2DCycleGAN-mixed-horse2zebra"
out_path = "D:/Work/Codes_tmp/2DCycleGAN-mixed-horse2zebra/out"

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
def map_func(x):
    # x shape = [batch,2,256,256,3]
    #必须归一化 对应于网络的tanh 但是暂时不知道用什么像素标准去归一化 可能需要遍历所有的值
    A = tf.reshape(x[:,0,:,:,:],[1,256,256,3], name=None)
    A = (A-0.0)/1
    B = tf.reshape(x[:,1,:,:,:],[1,256,256,3], name=None)
    B = (B-0.0)/1
    return A,B

EPOCHES = 200
BATCH_SIZE = 1


num_threads = 4
dataset = train_dataset.DataPipeLine(train_path_A,train_path_B)
dataset = tf.data.Dataset.from_generator(dataset.generator,output_types=tf.float32)\
            .batch(BATCH_SIZE)\
            .map(map_func,num_parallel_calls=num_threads)\
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

test_set = train_dataset.DataPipeLine(test_path_A,test_path_B)
test_set = tf.data.Dataset.from_generator(test_set.generator,output_types=tf.float32)\
            .batch(BATCH_SIZE)\
            .map(map_func,num_parallel_calls=num_threads)\
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)


model = model.CycleGAN(train_set=dataset,
                       test_set=test_set,
                       loss_name="LSGAN",
                       mixed_precision=True,
                       learning_rate=2e-4,
                       tmp_path=tmp_path,
                       out_path=out_path)
model.build(X_shape=[None,256,256,3],Y_shape=[None,256,256,3])
model.test(take_nums=200)