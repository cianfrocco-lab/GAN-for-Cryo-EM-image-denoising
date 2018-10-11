# GAN-for-Cryo-EM-image-denoising
Proof-of-principle studies with conditional generative adversarial networks to denoise Cryo-EM images. This project is the implementation of the Paper "Generative adversarial networks as a tool to recover structural information from cryo-electron microscopy data"(https://www.biorxiv.org/content/biorxiv/early/2018/02/12/256792.full.pdf) on python.
# Network Architecture
Similar to the https://github.com/SpaceML/GalaxyGAN and pix2pix(https://github.com/phillipi/pix2pix) with some modifications. 
![image](https://github.com/cianfrocco-lab/GAN-for-Cryo-EM-image-denoising/blob/master/imgs/Figure1_v2.png)
Each encode and decode is a residual block
# Loss function 
GAN loss + L1 loss (similar to the loss used in pix2pix in https://arxiv.org/pdf/1611.07004.pdf ) 
# Dependencies
*Tensorflow1.6 CUDA 9.0 CuDNN 7.0 Anaconda
# Training 
python train.py 
(you need to modify the path in the config.py)
# Testing 
python test.py
# Results on the real data
![image](https://github.com/cianfrocco-lab/GAN-for-Cryo-EM-image-denoising/blob/master/imgs/Figure2.png)
the ground truth is the projection of the EM density map by Relion, the input particle is the corresponding particle with the same orientation. The FSC curve between the recovered image and the ground truth projection showed high correlation score for the low frequency information which below 25 Å, indicating that cGAN can effectively recover low resolution features. 
# Results on synthetic data
![image](https://github.com/cianfrocco-lab/GAN-for-Cryo-EM-image-denoising/blob/master/imgs/Figure5.png)
We use the GAN to try to help us pick the small particles such as the kinesin on the MT



      
