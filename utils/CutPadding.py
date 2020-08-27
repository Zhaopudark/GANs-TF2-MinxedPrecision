"""
工具类
其实里面都只是数值计算函数
"""
import numpy as np 
def cut_img_3D(img):
    """
    将3D的数据进行剪裁，去除3D黑边
    img:numpy array
    return a cut img with same axis number but not a fixed one
    """
    # print(img.shape,img.dtype)
    buf=[]
    for i in range(img.shape[0]):
        temp = img[i,:,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[0]-1,-1,-1):
        temp = img[i,:,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[1]):
        temp = img[:,i,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[1]-1,-1,-1):
        temp = img[:,i,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[2]):
        temp = img[:,:,i]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[2]-1,-1,-1):
        temp = img[:,:,i]
        if(temp.sum()!=0):
            buf.append(i)
            break
    pw=1 # plus_width 前后增加的额外像素 防止3D图像缺失一小部分
    for i in range(3):
        if buf[2*i]-pw>=0:
            buf[2*i] -= pw
    for i in range(3):
        if buf[2*i+1]+pw<=(img.shape[i]-1):
            buf[2*i+1] += pw
    cut_img = img[buf[0]:buf[1]+1,buf[2]:buf[3]+1,buf[4]:buf[5]+1]
    # print(cut_img.shape) # buf 记录的是坐标下标 自身不涉及index+1 -1 认为考虑+-1
    max_length = max(cut_img.shape)
    zeros = np.zeros(shape=[1,cut_img.shape[1],cut_img.shape[2]],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[0]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=0)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=0)
    # print(cut_img.shape)
    zeros = np.zeros(shape=[cut_img.shape[0],1,cut_img.shape[2]],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[1]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=1)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=1)
    # print(cut_img.shape)
    zeros = np.zeros(shape=[cut_img.shape[0],cut_img.shape[1],1],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[2]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=2)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=2)
    # print(cut_img.shape)
    # print(cut_img.min())
    # print(cut_img.max())
    return cut_img
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pylab as plt
    import nibabel as nib
    from nibabel import nifti1
    from nibabel.viewers import OrthoSlicer3D

    example_filename = 'G:\\Datasets\\BraTS\\Collections\\HGG\\Brats18_2013_17_1\\Brats18_2013_17_1_t2.nii'

    img = nib.load(example_filename)
    img = np.array(img.dataobj[:,:,:])
    cut_img = cut_img_3D(img)
    print(cut_img.shape)
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    def plot_3d(image, threshold=0):
        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        p = image#.transpose(2,1,0)
        verts, faces, norm, val = measure.marching_cubes_lewiner(p,threshold,step_size=1, allow_degenerate=True)
        #verts, faces = measure.marching_cubes_classic(p,threshold)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
        plt.show()
    plot_3d(cut_img)