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
        self.datalist = self.__readDirFile(path)
        self.onehot = onehot
        self.train = train
    def __readDirFile(self,path):
        buf = []
        for (dirName, subdirList, fileList) in os.walk(path):
            for filename in fileList:
                if ".jpg" in filename.lower():  
                    buf.append(os.path.join(dirName,filename))
        return buf
    def read_file(self,path):
        image = Image.open(path)
        image_arr = np.array(image)
        return self.__normalize(image_arr)
    # def load_data(self,path):
    #     if os.path.exists(path+"\\mnist.npz"):
    #         npzfile=np.load(path+'\\mnist.npz') 
    #         train_images = npzfile['k1']
    #         train_labels = npzfile['k2']
    #         test_images = npzfile['k3']
    #         test_labels = npzfile['k4']
    #     else:
    #         train_images = self.loadMinistImage(path+"\\train-images.idx3-ubyte")
    #         train_labels = self.loadMinistLable(path+"\\train-labels.idx1-ubyte")
    #         test_images = self.loadMinistImage(path+"\\t10k-images.idx3-ubyte")
    #         test_labels = self.loadMinistLable(path+"\\t10k-labels.idx1-ubyte")
    #         np.savez(path+"mnist.npz",k1=train_images,k2=train_labels,k3=test_images,k4=test_labels)
    #     return  zip(train_images,train_labels),zip(test_images,test_labels) 
    def __normalize(self,img_slice,dtype=np.float32):
        max_pix = img_slice.max()
        min_pix = img_slice.min()
        tmp = (img_slice-min_pix)/(max_pix-min_pix)
        return tmp.astype(dtype)
    def generator(self):
        for image_path in self.datalist:
            yield self.read_file(image_path)
        return 
if __name__ == "__main__":
    # import tensorflow as tf 
    data = DataPipeLine("G:\\Datasets\\CelebA\\Img\\img_align_celeba")
    for i,image in enumerate(data.generator()):
        print(i,image.shape,image.dtype)
    


