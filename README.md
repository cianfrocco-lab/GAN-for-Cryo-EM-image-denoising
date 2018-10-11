# GAN-for-Cryo-EM-image-denoising
Proof-of-principle studies with conditional generative adversarial networks to denoise Cryo-EM images. This project is the implementation of the Paper "Generative adversarial networks as a tool to recover structural information from cryo-electron microscopy data" on python.
# Network Architecture
Similar to the https://github.com/SpaceML/GalaxyGAN and pix2pix(https://github.com/phillipi/pix2pix) with some modifications. 
![image](https://github.com/cianfrocco-lab/GAN-for-Cryo-EM-image-denoising/blob/master/imgs/Figure1_v2.png)
Each encode and decode is a residual block
# Loss function 
GAN loss + L1 loss(similar to the loss used in pix2pix in https://arxiv.org/pdf/1611.07004.pdf ) 
# Dependencies
Tensorflow1.6 CUDA9.0 CuDNN 7.0 Anaconda

      
