# SRGAN-with-WGAN-Loss-TensorFlow
SRGAN with WGAN loss function in TensorFlow
## Introduction
This code mainly address the problem of super resolution, [Super Resolution Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)
#### There are four different from the paper:
1. The loss function, we use WGAN loss, instead of standard GAN loss.
2. The network architecture, Because of our poor device, in generator, we just use 5 residual block (paper: 16), and in discriminator, we use the standard DCGAN's discriminator.
3. The training set, device problem again,:cry: we just use a part of ImageNet ([ImageNet Val](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar)) which just contains 50,000 images.
4. The max iteration, we just train the model about 100,000 iterations, instead of the paper 600,000.

![](https://github.com/MingtaoGuo/SRGAN-with-WGAN-Loss-TensorFlow/blob/master/IMAGES/networks.jpg)

## How to use 
1. Download the dataset [ImageNet Val](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar)
2. unzip dataset and put it into the folder 'ImageNet'
```
├── test
├── save_para
├── results
├── vgg_para
├── ImageNet
     ├── ILSVRC2012_val_00000001.JPEG
     ├── ILSVRC2012_val_00000002.JPEG
     ├── ILSVRC2012_val_00000003.JPEG
     ├── ILSVRC2012_val_00000004.JPEG
     ├── ILSVRC2012_val_00000005.JPEG
     ├── ILSVRC2012_val_00000006.JPEG
     ...
```
3. execute the file main.py
## Requirements
- python3.5
- tensorflow1.4.0
- pillow
- numpy
- scipy
- skimage
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
## Reference
[1] Ledig C, Theis L, Huszár F, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network[C]//CVPR. 2017, 2(3): 4.
