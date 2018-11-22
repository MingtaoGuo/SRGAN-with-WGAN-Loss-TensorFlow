# SRGAN-with-WGAN-Loss-TensorFlow
SRGAN with WGAN loss function in TensorFlow
## Introduction
This code mainly address the problem of super resolution, [Super Resolution Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)
#### There are some places different from the paper:
1. The loss function, we use WGAN loss, instead of standard GAN loss.
2. The network architecture, Because of our poor device, in generator, we just use 5 residual block, and in discriminator, we use the standard DCGAN's discriminator.
3. The training set, device problem again,:cry: we just use a part of ImageNet ([ImageNet Val](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar)) which just contains 50,000 images.

![](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar)

## Results
#### Train procedure WGAN Loss
![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/wganloss.jpg)

|Down sampled|Bicubic (x4)|SRGAN (x4)|
|-|-|-|
|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/down1.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/bicubic1.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/sr1.jpg)|
|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/down2.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/bicubic2.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/sr2.jpg)|
|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/down3.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/bicubic3.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/sr3.jpg)|
|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/down4.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/bicubic4.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/sr4.jpg)|
|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/down5.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/bicubic5.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/sr5.jpg)|
|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/down6.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/bicubic6.jpg)|![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/sr6.jpg)|
