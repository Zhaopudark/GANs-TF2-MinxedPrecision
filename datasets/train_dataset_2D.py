"""
从文件路径中，返回原始文件的最初读取内容
不做任何的附加操作
"""
import os 
from PIL import Image
import numpy as np 
import nibabel as nib
from utils import CutPadding
class DataPipeLine():
    def __init__(self,path,target_size=[64,64,64],update=False):
        self.path = path
        self.datalist = self.__readDirFile(self.path)
        self.update = update#指示是否更新文件（文件不存在时一定更新并且保存，文件存在时，update==False则直接从文件读取，update==True则保存新文件）
        self.get_saved = False#指示本次是否已经从保存文件中直接读取 每读取一次后回到False
        self.target_size = target_size
    def __readDirFile(self,path):
        train_AB_address = [] #二维列表 记录A
        for (dirName, subdirList, fileList) in os.walk(path):
            buf = ["A","B"]
            count = 0
            for filename in fileList:
                if "t1.nii" in filename.lower():  # check whether the file's jpg
                    buf[0]=os.path.join(dirName,filename)
                    count += 1
                if "t2.nii" in filename.lower():  # check whether the file's jpg
                    buf[1]=os.path.join(dirName,filename)
                    count += 1
            if count == 2:
                train_AB_address.append(tuple(buf))
        return train_AB_address
    def __read_nii_file(self,path):
        temp_path = path[:-3]+"2d.npy"
        if os.path.exists(temp_path)==True:
            if self.update==False:
                self.get_saved = True
                return np.load(temp_path)
        img = nib.load(path)
        img = np.array(img.dataobj[:,:,:])
        # print("*****")
        # print(img.shape,img.dtype)
        # print(img.min(),img.max())
        # print("*****")
        img = CutPadding.cut_img_3D(img)
        # print("*****")
        # print(img.shape,img.dtype)
        # print(img.min(),img.max())
        # print("*****")
        return img
    def __save_nii_npz(self,img,path):
        temp_path = path[:-3]+"2d.npy"
        if self.update==True:
            np.save(temp_path,img)
        else:
            pass
        self.get_saved = False #下一个文件不知道是否可以读取
    def __cut_np_array(self,array,target_shape=[128,128,128]):
        old_shape = array.shape
        buf = [0,0,0]
        for i in range(3):
            buf[i]=old_shape[i]//2-target_shape[i]//2
            #左半部右下标+1 减去目标点数的一半 获得新的起始点 10//2 -6//2 = 2 从下标2开始然后到下标2+6-1结束
        return array[buf[0]:buf[0]+target_shape[0],buf[1]:buf[1]+target_shape[1],buf[2]:buf[2]+target_shape[2]]            
    def __normalize(self,slice,dtype=np.float32):

        # """
        # normalize image with mean and std for regionnonzero,and clip the value into range
        # :param slice:
        # :param bottom:
        # :param down:
        # :return:
        # """
        # #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
        # b = np.percentile(slice, bottom)
        # t = np.percentile(slice, down)
        # slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)
    
        # #除了黑色背景外的区域要进行标准化
        # image_nonzero = slice[np.nonzero(slice)]
        # if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        #     return slice
        # else:
        #     tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        #     # since the range of intensities is between 0 and 5000 ,
        #     # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        #     # the min is replaced with -9 just to keep track of 0 intensities
        #     # so that we can discard those intensities afterwards when sampling random patches
        #     tmp[tmp == tmp.min()] = -9 #黑色背景区域
        #     return tmp
        tmp = slice/slice.max()
        # tmp[tmp<0.0]=0.0 无法改变负0的结果
        return tmp.astype(dtype)
    def __get_AB_nii(self,path):
        img = self.__read_nii_file(path)
        if self.get_saved == True:
            return img
        else:#没有读取保存的文件
            pass
        """
        迫于无奈之举 必须降采样才可以训练 128 128 128 -> 64 64 64
        同时 因为确实不知道怎么归一化 所以么得办法 只能最大最小归一化先
        """
        from scipy import ndimage
        ratio = [self.target_size[x]/img.shape[x] for x in range(3)]
        # print(ratio)
        resize_image = ndimage.interpolation.zoom(img,ratio, mode='nearest')
        assert resize_image.shape==tuple(self.target_size)
        # print(resize_image.min(),resize_image.max())
        resize_image[resize_image<0] = 0
        resize_image = resize_image[:,:,32] #2D图像必须单独提出且norm保存
        img_norm = self.__normalize(resize_image,dtype=np.float32)
        self.__save_nii_npz(img_norm,path)
        return img_norm


    def __img2numpy_bacth(self,fileList):
        pass
    def __img2numpy_single(self,path):
        pass
    def __load_dataset_slice(self,headpath,slice,detype=np.uint8):
        pass
    def __iter__(self):
        pass
    def generator(self):
        for item in self.datalist:
            yield (self.__get_AB_nii(item[0]),self.__get_AB_nii(item[1]))
        return  
if __name__ == "__main__":
    gen = DataPipeLine("E:\\Datasets\\BraTS\\Combine",target_size=[64,64,64],update=True)
    abc = gen.generator()
    for i,(t1,t2) in enumerate(abc):
        print(i,t1.shape,t1.dtype,
                t2.shape,t2.dtype,
                t1.max(),t1.min(),
                t2.max(),t2.min())
    # t1,t2 = next(abc)
    # print(t1.shape,t2.shape)
    # from matplotlib import pylab as plt
    # print(t1.dtype,t1.dtype)
    # from skimage import measure
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # def plot_3d(image, threshold=0):
    #     # Position the scan upright,
    #     # so the head of the patient would be at the top facing the camera
    #     p = image#.transpose(2,1,0)
    #     verts, faces, norm, val = measure.marching_cubes_lewiner(p,threshold,step_size=1, allow_degenerate=True)
    #     #verts, faces = measure.marching_cubes_classic(p,threshold)
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #     # Fancy indexing: `verts[faces]` to generate a collection of triangles
    #     mesh = Poly3DCollection(verts[faces], alpha=0.7)
    #     face_color = [0.45, 0.45, 0.75]
    #     mesh.set_facecolor(face_color)
    #     ax.add_collection3d(mesh)
    #     ax.set_xlim(0, p.shape[0])
    #     ax.set_ylim(0, p.shape[1])
    #     ax.set_zlim(0, p.shape[2])
    #     plt.show()
    # plot_3d(t1)


    








 
