"""
惰性迭代的方式读取训练数据
永远只返回一个数据样例 包括其标签
设置批数量 标签样本分离等等交给 如 train 文件处理(由用户自定义)
"""

import os 
import sys
from PIL import Image
import numpy as np 
import nibabel as nib
from scipy import ndimage
import random
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils import CutPadding
import random
import struct
def print_log(s):
    """
    对print函数的一种变相重写 防止在被其他文件调用时 输出不必要的调试信息
    """
    if __name__ =="__main__":
        print(s)
    else:
        pass
class DataPipeLine():
    def __init__(self,path,train=False,onehot=False):
        self.path = path 
        self.train_buf,self.test_buf = self.load_data(path)
        self.onehot = onehot
        self.train = train
    def loadMinistImage(self,filename):
        binfile = open(filename,'rb')
        buffers = binfile.read()
        head = struct.unpack_from('>4i',buffers,0)#默认小端格式 需要变成大端格式则改成<
        #一个i就是32位整数表示4字节,offset=4表示偏移4个字节
        if head[0]!=2051:
            raise ValueError("Errors occured in image file")
        else: 
            print_log("Read image file succeed")
            img_num = head[1]
            img_width = head[2]
            img_height = head[3]
            bits = img_num*img_height*img_width
            #60000*28*28的字节
            bits_string=">"+str(bits)+"B"#以一个字节为单位的连续结构，而i是以4个字节为单位
            offset = struct.calcsize('4i')  # 定位到data开始的位置
            imgs = struct.unpack_from(bits_string, buffers, offset)
            binfile.close()
            imgs = np.reshape(imgs,[img_num,img_width,img_height])
            #变成[60000 28 28]的矩阵
        return imgs
    def loadMinistLable(self,filename):
        binfile = open(filename,'rb')
        buffers = binfile.read()
        head = struct.unpack_from('>2i',buffers,0)#默认小端格式 要变成大端格式则改成<
        #一个i就是32位整数表示4字节,offset=4表示偏移4个字节 
        if head[0]!=2049:
            raise ValueError("Errors occured in label file")
        else:
            print_log("Read label file succeed")
            img_num = head[1]
            bits = img_num
            bits_string=">"+str(bits)+"B"#以一个字节为单位的连续结构，而i是以4个字节为单位
            offset = struct.calcsize('2i')  # 定位到data开始的位置
            labels = struct.unpack_from(bits_string, buffers, offset)
            binfile.close()
            labels = np.reshape(labels,[img_num])
        return labels 
    def load_data(self,path):
        if os.path.exists(path+"\\mnist.npz"):
            npzfile=np.load(path+'\\mnist.npz') 
            train_images = npzfile['k1']
            train_labels = npzfile['k2']
            test_images = npzfile['k3']
            test_labels = npzfile['k4']
        else:
            train_images = self.loadMinistImage(path+"\\train-images.idx3-ubyte")
            train_labels = self.loadMinistLable(path+"\\train-labels.idx1-ubyte")
            test_images = self.loadMinistImage(path+"\\t10k-images.idx3-ubyte")
            test_labels = self.loadMinistLable(path+"\\t10k-labels.idx1-ubyte")
            np.savez(path+"mnist.npz",k1=train_images,k2=train_labels,k3=test_images,k4=test_labels)
        return  zip(train_images,train_labels),zip(test_images,test_labels) 
          
    def __normalize(self,img_slice,dtype=np.float32):
        max_pix = img_slice.max()
        min_pix = img_slice.min()
        tmp = (img_slice-min_pix)/(max_pix-min_pix)
        return tmp.astype(dtype)
    def __remake_label(self,label,dtype=np.float32):
        if self.onehot:
            new_label = np.zeros(shape=(10))
            new_label[int(label)] = 1.0
            return new_label.astype(dtype)
        else:
            return label.astype(dtype)
    def generator(self):
        if self.train:
            for train_image,train_label in self.train_buf:
                yield (self.__normalize(train_image),self.__remake_label(train_label))
        else:
            for test_image,test_label in self.test_buf:
                yield (self.__normalize(test_image),self.__remake_label(test_label))
        return 
if __name__ == "__main__":
    # import tensorflow as tf 
    data = DataPipeLine("G:\\Datasets\\Mnist")
    for i,(a,b) in enumerate(data.generator()):
        print(i,a.shape,a.dtype,b.shape,b.dtype)
    data = DataPipeLine("G:\\Datasets\\Mnist",train=True)
    for i,(a,b) in enumerate(data.generator()):
        print(i,a.shape,a.dtype,b.shape,b.dtype)
    data = DataPipeLine("G:\\Datasets\\Mnist",onehot=True)
    for i,(a,b) in enumerate(data.generator()):
        print(i,a.shape,a.dtype,b.shape,b.dtype)
    data = DataPipeLine("G:\\Datasets\\Mnist",train=True,onehot=True)
    for i,(a,b) in enumerate(data.generator()):
        print(i,a.shape,a.dtype,b.shape,b.dtype)
    


