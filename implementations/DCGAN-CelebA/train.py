import tensorflow as tf
import time
import os 
import sys
import model 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../../'))
import datasets.CelebAPipeLine as train_dataset


physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

######################################################################################################
train_path = "G:\\Datasets\\CelebA\\Img\\img_align_celeba"
test_path = None
tmp_path = "D:/Work/Codes_tmp/DCGAN-mixed-CelebA-wgp"
out_path = "D:/Work/Codes_tmp/DCGAN-mixed-CelebA-wgp/out"

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
def map_func(img):
    """
    针对batch可变的不确定 先map在batch堆叠
    """
    img = tf.reshape(img,shape=[218,178,3])
    img = tf.image.resize(img,size=[112,80])
    noise = tf.random.uniform(shape=[100],minval=-1.0,maxval=1.0,dtype=tf.float32)
    return noise,img
def map_func_z(img):
    """
    针对batch可变的不确定 先map在batch堆叠
    """
    # img = tf.reshape(img,shape=[218,178,3])
    # img = tf.image.resize(img,size=[28,20,3])
    noise = tf.random.uniform(shape=[100],minval=-1.0,maxval=1.0,dtype=tf.float32)
    return noise

EPOCHES = 200
BATCH_SIZE = 128

num_threads = 4
dataset = train_dataset.DataPipeLine(train_path)
#拆包应当在第一步完成，不应当放在map中 因为会遇到可能无法堆叠的shape
dataset = tf.data.Dataset.from_generator(dataset.generator,output_types=(tf.float32),output_shapes=((218,178,3)))\
            .map(map_func,num_parallel_calls=num_threads)\
            .batch(BATCH_SIZE)\
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

# for A,b in dataset:
#     print(A.shape)
#     for i in range(100):
#         print(A[i,:])
#     break


test_set = train_dataset.DataPipeLine(train_path) #但其实毫无用处，仅仅是噪声作为输入的堆叠
test_set = tf.data.Dataset.from_generator(test_set.generator,output_types=(tf.float32),output_shapes=((218,178,3)))\
            .map(map_func_z,num_parallel_calls=num_threads)\
            .batch(1)\
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
# for i,noise in enumerate(test_set):
#     print(i,noise.shape)

model = model.DCGAN(train_set=dataset,
                       test_set=test_set,
                       loss_name="WGAN-GP",
                       mixed_precision=True,
                       learning_rate=2e-4,
                       tmp_path=tmp_path,
                       out_path=out_path)
model.build(input_shape_G=[None,100],input_shape_D=[None,112,80,3])
model.train(epoches=EPOCHES)