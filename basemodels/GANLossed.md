# GAN loss 相关公式与算法实现
包含以下的GAN loss理论公式与实际生成器判别器损失函数的对应关系。将会随着GAN理论的深入不断扩展

    Vanilla GAN 
    WGAN 
    WGAN-GP 
    RSGAN 
    LSGAN 


## 1. Vanilla GAN loss
$a + b = c$

$$lr_t = \mathrm{learning\_rate} *\sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
$$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
$$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
$$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$