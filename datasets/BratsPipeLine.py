"""
针对brats数据集
做包含预处理的数据管道(Python生成器)

每次优先读取npy 不存在则读取nii 同时保存npy

迫于无奈之举 必须降采样才可以训练 128 128 128 -> 64 64 64
同时 因为确实不知道怎么归一化 所以么得办法 只能最大最小归一化先

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
class UnpairedError(Exception):
    def __init__(self,path):
        self.err_msg = "There are not exixting paired samples! We can only find:"
        self.filename = path
class DataPipeLine():
    def __init__(self,path,target_size,remake_flag=False,random_flag=False):
        self.path = path
        self.datalist = self.__readDirFile(self.path,random_flag)
        self.target_size = target_size
        self.dims = len(target_size)
        self.remake_flag = remake_flag
        self.random_flag = random_flag
    def __readDirFile(self,path,random_flag=False):
        buf_A = []
        buf_B = []
        for (dirName, subdirList, fileList) in os.walk(path):
            try:
                for filename in fileList:
                    if "t1.nii" in filename.lower():  
                        buf_A.append(os.path.join(dirName,filename))
                    if "t2.nii" in filename.lower(): 
                        buf_B.append(os.path.join(dirName,filename))
                if len(buf_A) > len(buf_B):
                    raise UnpairedError(buf_A.pop(-1))
                elif len(buf_A) < len(buf_B):
                    raise UnpairedError(buf_B.pop(-1))
                else:
                    pass
            except UnpairedError as error:
                print(error.err_msg)
                print(error.filename)
            else:# normal condition
                pass
            finally:# any way
                pass
        if random_flag:
            random.shuffle(buf_A)
            random.shuffle(buf_B)
            return list(zip(buf_A,buf_B))
        else:
            return list(zip(buf_A,buf_B))
    def read_file(self,path):
        if self.dims == 3:
            temp_path = path[:-3]+"npy"
            if (os.path.exists(temp_path)==True)and(self.remake_flag==False):
                return np.load(temp_path)
            else:
                return self.load_nii_file(path)
        elif self.dims == 2:
            temp_path = path[:-3]+"2D.npy"
            if (os.path.exists(temp_path)==True)and(self.remake_flag==False):
                return np.load(temp_path)
            else:
                return self.load_nii_file(path)
        else:
            raise ValueError
    def __read_nii_file(self,path):
        img = nib.load(path)
        img = np.array(img.dataobj[:,:,:])
        return img
    def __cut_nii_file(self,img):
        return CutPadding.cut_img_3D(img)
    def __save_nii2npy(self,img,path):
        if self.dims == 3:
            temp_path = path[:-3]+"npy"
        elif self.dims ==2:
            temp_path = path[:-3]+"2D.npy"
        else:
            raise ValueError
        np.save(temp_path,img)
        return img
    def __cut_np_array(self,array,target_shape=[128,128,128]):
        old_shape = array.shape
        buf = [0,0,0]
        for i in range(3):
            buf[i]=old_shape[i]//2-target_shape[i]//2
            #左半部右下标+1 减去目标点数的一半 获得新的起始点 10//2 -6//2 = 2 从下标2开始然后到下标2+6-1结束
        return array[buf[0]:buf[0]+target_shape[0],buf[1]:buf[1]+target_shape[1],buf[2]:buf[2]+target_shape[2]]            
    def __normalize(self,slice,dtype=np.float32):
        tmp = slice/slice.max()
        return tmp.astype(dtype)
    def load_nii_file(self,path):
        img = self.__read_nii_file(path)#读取3D源文件 
        img = self.__cut_nii_file(img)#去除文件周围无意义的区域 3D去黑边
        #缩放到目标大小 最近邻插值
        if len(self.target_size)==2:
            temp_targer_size = self.target_size[:]+[self.target_size[-1]]
        else:
            temp_targer_size = self.target_size[:]
        ratio = [temp_targer_size[x]/img.shape[x] for x in range(3)]
        resize_image = ndimage.interpolation.zoom(img,ratio, mode='nearest')
        assert resize_image.shape==tuple(temp_targer_size)
        resize_image[resize_image<0]=0#去除插值后出现的负像素
        if self.dims == 3:
            resize_image = resize_image
        elif self.dims ==2:
            resize_image = resize_image[:,:,temp_targer_size[-1]//2]
        else:
            raise ValueError
        img_norm = self.__normalize(resize_image,dtype=np.float32)#归一化
        img_saved = self.__save_nii2npy(img_norm,path)#保存 并且返回保存的文件 将对2D 3D区别对待
        return img_saved
    def __iter__(self):
        #实现__iter__ 本身就是一个迭代器 但是没有call方法 不能被tensorflow from_generator识别 所以必须在实现一个一般的生成器函数
        for A,B in self.datalist:
            yield (self.read_file(A),self.read_file(B))
        return
    def generator(self):
        for A,B in self.datalist:
            yield (self.read_file(A),self.read_file(B))
        return 
    def chenk_saved_npy(self):
        #该方法直接进行一次全部迭代，将nii文件读取并且内容保存为预处理后的numpy矩阵 npy无压缩格式
        for i,(A,B) in enumerate(self):
            print(i+1,A.shape,B.dtype,
                    A.shape,B.dtype,
                    A.max(),B.min(),
                    A.max(),B.min())
if __name__ == "__main__":
    # import tensorflow as tf 
    a = DataPipeLine("G:\\Datasets\\BraTS\\Combine\\",target_size=[128,128],remake_flag=True,random_flag=False)
    a.chenk_saved_npy()
    a = DataPipeLine("G:\\Datasets\\BraTS\\Combine\\",target_size=[64,64,64],remake_flag=True,random_flag=False)
    a.chenk_saved_npy()
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
