"""
针对自然图像
Img2Img
本身是Unpaid 所以就应该操作两个图像才对

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
class DataPipeLine():
    def __init__(self,setA_path,setB_path):
        self.datalist_A = self.__readDirFile(setA_path)
        self.datalist_B = self.__readDirFile(setB_path)
        self.datalist = self.get_combined_datalist(self.datalist_A,self.datalist_B)
    def __readDirFile(self,path):
        buf = []
        for (dirName, subdirList, fileList) in os.walk(path):
            for filename in fileList:
                if ".jpg" in filename.lower():  
                    buf.append(os.path.join(dirName,filename))
        return buf
    # def read_file(file_):
    def get_combined_datalist(self,datalist_A,datalist_B):
        buf = []
        j = 0
        i = 0
        over_index_buf = [0,0]
        while(over_index_buf!=[1,1])and(i<len(datalist_A))and(j<len(datalist_B)):
            buf.append((datalist_A[i],datalist_B[j]))
            i += 1
            j += 1
            if i >= len(datalist_A):
                over_index_buf[0]=1
                random.shuffle(datalist_A)
                i = 0
            if j >= len(datalist_B):
                over_index_buf[1]=1
                j = 1 
                random.shuffle(datalist_B)
        return buf 
    def read_file(self,path):
        image = Image.open(path)
        image_arr = np.array(image)
        return self.__normalize(image_arr)
      
    def __normalize(self,img_slice,dtype=np.float32):
        max_pix = img_slice.max()
        min_pix = img_slice.min()
        tmp = (img_slice-min_pix)/(max_pix-min_pix)
        return tmp.astype(dtype)
    def __iter__(self):
        #实现__iter__ 本身就是一个迭代器 但是没有call方法 不能被tensorflow from_generator识别 所以必须在实现一个一般的生成器函数
        for A,B in self.datalist:
            yield (self.read_file(A),self.read_file(B))
        return
    def generator(self):
        for A,B in self.datalist:
            tmpA = self.read_file(A)
            tmpB = self.read_file(B)
            if (tmpA.shape!=(256,256,3))or(tmpB.shape!=(256,256,3)):
                continue
            else:
                yield (tmpA,tmpB)
        return 
if __name__ == "__main__":
    # import tensorflow as tf 
    a = DataPipeLine("E:\\Img2Img\\horse2zebra\\trainA","E:\\Img2Img\\horse2zebra\\trainB")
    # datalist = a.datalist
    # print(datalist)
    # for i,(itemA,itemB) in enumerate(datalist):
    #     image = Image.open(itemA)
    #     image_arr = np.array(image)
    #     if image_arr.shape == (256,256,3):
    #         print(i,image_arr.shape,image_arr.dtype,image_arr.max(),image_arr.min())
    for i,(imgA,imgB) in enumerate(a.generator()):
        print(i,imgA.shape,imgA.dtype,
                imgB.shape,imgB.dtype,
                imgA.max(),imgA.min(),
                imgB.max(),imgB.min())
    # a.chenk_saved_npy()
    # dataset = tf.data.Dataset.from_generator(iter(a),output_types=tf.float32)\
    #         .batch(1)\
    #         .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    # gen = DataPipeLine("E:\\Datasets\\BraTS\\MICCAI_BraTS2020_TrainingData",target_size=[64,64,64],update=True)
    # abc = gen.generator()
    # for i,(t1,t2) in enumerate(abc):
    #     print(i,t1.shape,t1.dtype,
    #             t2.shape,t2.dtype,
    #             t1.max(),t1.min(),
    #             t2.max(),t2.min())