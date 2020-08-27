import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

inputs = keras.Input(shape=(784,), name='digits')
num_units = 4096
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

print('x.dtype: %s' % x.dtype.name)
# 'kernel' is dense1's variable
print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)

# INCORRECT: softmax and model output will be float16, when it should be float32
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

 # CORRECT: softmax and model output are float32
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)


# The linear activation is an identity function. So this simply casts 'outputs'
# to float32. In this particular case, 'outputs' is already float32 so this is a
# no-op.
outputs = layers.Activation('linear', dtype='float32')(outputs)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32') / 255
# x_test = x_test.reshape(10000, 784).astype('float32') / 255
# initial_weights = model.get_weights()
 


# optimizer = keras.optimizers.RMSprop()
# optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
#                  .shuffle(10000).batch(4096))
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(4096)
# for x,y in train_dataset:
#     break
import time 
print(x.shape)
x = tf.random.normal(shape=[4096,784])
for __ in range(5):
    start = time.time()
    for _ in range(400):
        predictions = model(x)
    print(time.time()-start)
# @tf.function()
# def train_step(x, y):
#     """
#     优化器增加 将损失乘以损失标度
#     get_scaled_loss(loss) ：将损失乘以损失标度
#     get_unscaled_gradients(gradients) ：接收一系列比例渐变作为输入，然后将每个比例除以损耗比例来取消比例
#     """
#     with tf.GradientTape() as tape:
#         predictions = model(x)
#         loss = loss_object(y, predictions)
#         scaled_loss = optimizer.get_scaled_loss(loss)
#     scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
#     gradients = optimizer.get_unscaled_gradients(scaled_gradients)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

# @tf.function()
# def test_step(x):
#   return model(x, training=False)

# model.set_weights(initial_weights)
# for epoch in range(5):
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#         name='test_accuracy')
#     for x, y in train_dataset:
#         loss = train_step(x, y)
#         epoch_loss_avg(loss)
#     for x, y in test_dataset:
#         predictions = test_step(x)
#         test_accuracy.update_state(y, predictions)
#     print('Epoch {}: loss={}, test accuracy={}'.format(epoch, epoch_loss_avg.result(), test_accuracy.result()))
    