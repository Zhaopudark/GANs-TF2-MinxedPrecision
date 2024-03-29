
# 0.前言
鉴于目前tensorflow2.0已经高度依赖keras，不继承keras特性的模型、层、优化器等等都或多或少存在模型部署与保存的问题，因此，TF2下原本有依赖keras和不依赖keras两种开发模式，如今退化为仅依赖keras(tf.keras)的开发模式。


# 1.项目内容
基于Python 与tensorflow2.x构建生成对抗网络(Generative Adversarial Networks)(GANs)的若干算法的训练实例(Implementation)。力求简洁易用，有利于我自己的项目(医学图像合成)的开发与迭代过程。

+ 相关简写  
    + Binary Cross Entropy(BCE)
    + Multi-Layer Perceptron(MLP)
    + Fully Connected Layer(FCL)
    + Wasserstein Distance(W-Distance)
    + Gradient Penalty(GP)
    + Convolution (Conv)(Convolutional Option)(Convolutional Layer)
    + Spectral Normalization(SN)(Spectral Normalization Option)(Spectral Normalization Layer)

+ 实现算法如下([.]中为算法组成要素)
    + GAN(Original GAN)(Vanilla GAN)(Standard GAN)[BCE,MLP]
    + CGAN(Conditional GAN)[BCE,MLP]
    + DCGAN(Deep Convolutional GAN)[BCE,Conv+FCL]
    + C-DCGAN(Conditional - Deep Convolutional GAN)[BCE,Conv+FCL]
    + WGAN(Wasserstein GAN)[W-Distance,MLP]
    + WGAN-GP(Wasserstein GAN - Gradient Penalty)[W-Distance,MLP,GP]
    + SNGAN(Spectral Normalization GAN)[W-Distance,MLP,SN]
    + SN-W-DCGAN[W-Distance,Conv+FCL,GP,SN]
    + LSGAN()
    + Pix2Pix()
    + CycleGAN()
    + UNIT()
# 2.框架与环境
+ Python3.8  
+ Tensorflow2.2(基于Cuda10.2和所用系统自行编译)
+ Cuda10.2
+ Win10(2004) / Ubuntu20.04 (所有非可视化部分都可以在两个平台完全兼容，以Windows平台为主)
# 3.项目设计思路与特点
&ensp;&ensp;&ensp;&ensp;基于tf.nn和tf.keras.Model自定义网络层，模型层，训练部分等等。
# 4.文件结构
模型训练结果与中间值保存在外部指定位置，不会被云同步。
数据集保存在外部指定位置，不会被云同步

    -datasets
     --DataPipeLine.py(对指定数据集进行处理，返回一个)
    -utils
     --CutPadding.py(针对数据集做的剪裁等等操作)
    -models
     --BaseLayer.py(包含基本的模型构建)
     --GanModules.py(包含一般的GAN的生成器，判别器等)
     --GanLoss.py(定义了若干GAN是loss计算函数)
     --GanOptimizers.py(定义了若干GAN中常常使用的优化器)
    -gans
     --CycleGAN
      ---blocks.py
      ---nets.py
      ---train.py
      ---test.py

     --CGAN
     ...
     ...
