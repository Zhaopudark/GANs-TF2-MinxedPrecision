
# 0.前言
鉴于目前tensorflow2.0已经高度依赖keras，不继承keras特性的模型，层，优化器等等都或多或少存在模型部署与保存的问题(bugs)，或者是模型终结结果的输出问题，因此，TF2下原本有依赖keras和不依赖keras两种开发模式，如今退化为仅依赖keras(tf.keras)的开发模式。

另外，基于tf.keras也有两种开发模式

1. 借助keras的高阶API做开发，模型拥有build，compile，fit，evaluate等等高级初始化与训练方法，可以接入keras后端进行高级API操作。但是此类方法需要熟悉keras本身的特性，看似方便实则不灵活，需要自定义模型时，拥有功能的冗余

2. 基于tf.keras.Model自定义模型，自定义训练循环，自定义中间输出与保存。灵活方便，功能可增减，同时将模型细节完全展现，不隐藏起来。实现对我开发而言真正简明易用灵活的模式。(选择此模式开发)

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
&ensp;&ensp;&ensp;&ensp;为了便于二次开发，追求真正的易用性和易理解性，选择了tensorflow2.x的自定义训练模式进行算法构建。需要自定义网络层，模型层，训练部分等等。所以keras提供的强大接口愈加鸡肋。限制了自定义模型中的众多参数名称的自定义，且大部分功能用不到。

&emsp;&emsp;简单继承tf.Module,对模型保存，检查点等依旧支持不够友好，后续开发将基于tf.keras.Model进行。但是tf.keras.Model本身支持功能繁杂，为了不被干扰，需要将我自定义的特有变量加以新的标签

对比|tf.Module|tf.keras.Model
--|:--:|--:
trainable_variables|&heartsuit;|&heartsuit;
variables|&heartsuit;|&heartsuit;
tf.saved_model.save(this_model,path)|&heartsuit;|&heartsuit;
tf.saved_model.load(path)|can not be called|&heartsuit;
model.save(path)|未定义|&heartsuit;
model.load(path)|未定义|&heartsuit;
model.save_weights(path)|未定义|&heartsuit;
model.save_weights(path)|未定义|&heartsuit;

# 3.文件结构
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
