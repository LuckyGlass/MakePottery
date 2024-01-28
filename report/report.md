# CV大作业中期报告

2024.1.17 成员：王敏齐；徐灵昀；毛彦升

## Part1 Introduction

The topic we choose for final project is *Let's Make Pottery: GANS for Voxel Completion in Ancient Pottery Dataset* whose Quizmaster is class TA YuXuan, Luo.

This project mainly revolves around building a GAN that takes fragmented artifacts' voxel representation as input and outputs the missing part of the corresponding piece so as to assist the pottery reconstruction work for archaeological studies. Till the middle checkpoint, we've successfully visualized the dataset, implemented the strengthened data-loader and constructed the structure of a naive GAN which can smoothly compile and train. For further improvement, we plan to train it on a larger basis to collect long-term data of its performance and visualize its output. Eventually we will generalize our model to $64^3$ resolution and derive loss functions utilizing normal vectors and other significant information to improve the reconstruction performance.

## Part2 Related Works

### (1) GAN based models

*Basic GAN model (Goodfellow et al., 2014)* consists of two parts: generator and discriminator. In the first version, both the generator and the discriminator are MLPs. The advanced GANs make many modifications to the model structure, such as

- using CNNs - which is called *DCGAN (Radford et al., 2015)*;
- using Varation-AutoEncoder - which is called *VAE-GAN (Larsen et al., 2015)*;
- changing loss function from JS divergence to Wasserstein loss - which is called *WGAN (Arjovsky et al., 2017)*.

These modifications focused on higher ability and better stablity. Another kind of modification concerns how to generate higher resolutions and bigger results. One basic way is to adding new structure and stronger computation, such as

- *SA-GAN (Zhang et al., 2018)* using self-attention;
- *SN-GAN (Miyato et al., 2018)* using spectral normalization;
- *bigGAN (Brock et al., 2018)* using bigger batch size.

Some method changes the way the model works, including
- using progressive scale-growing GANs, which is called *ProGAN (Karras, 2017)*;
- using progressive pixel-growing models, which is called *PixelCNN (Oord et al., 2016)* (this is not a GAN, but a kind of generative model, so we put here).

Another improving way is adding controls to the result, such as *cGANs (Mirza et al., 2014)*, *styleGANs (Karras et al., 2018)*.

### (2) main paper introduction

- introduce the first two papers mentioned in writeup

Till now, we've read the first two papers mentioned in writeup, namely *Generative Adversarial Nets (Goodfellow et al., 2014)* and *3D Reconstruction of Incomplete Archaeological Objects Using Adversarial Network (Hermoza et al., 2018)*.

From the first paper, we've learnt the essence of a GAN which is a combination of a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. We've also gone through the mathematics foundation of the global optimality and convergence of a GAN-based algorithm, from which we derived several training strategies including alternating between the training of D and G. The paper also opens our minds to further improvement of GAN structure such as introducing a variational auto-encoder in the generator to allow for the learning of the conditional variance.

The second one mainly illustrated a variant of GAN, namely the ORGAN(OR is short for object reconstruction), drawing a fundamental picture of 3D GAN. It gave detailed description regarding its structure(a 3D CNN with **skip-connections**, as a **generator** on a Conditional GAN (CGAN) architecture. With two optimization targets: a mean absolute error (MAE) and an Improved Wasserstein GAN (IWGAN) loss) in additional to training details including the choice of hyper-parameters, which provided us with much reference and experience to rely on. Nevertheless, the dataset we use is quite different from the latter paper, which pushes us to design our own dataloader and consequently the exact training techniques need further experiments.

## Part3 Our Approach

### (1) Problem Definition
In a word, what we are going to do is to predict models of the complete pottaries, given models of the fragments. The 3D models are represented as voxels of shape $32\times32\times32$ or $64\times64\times64$.

It's worth noting that the task is a prediction task, instead of a generation task. The difference is that in out prediction task, the ground-truths are given, while in a generation task, they are not given. The task determined that our approach was fully-supervised.

Our task setting is almost the same as *"3D reconstruction of incomplete archaeological objects using a generative adversarial network" (Hermoza et al., 2018)*, with a slight difference that the inputs of our tasks were fragments (or combinations of fragments), while in Hermoza (2018)'s work, the inputs were randomly sampled from the complete models.

### (2) Model Setting

model layer structure/hyper parameters/loss function/optimizer/training strategy

We build a generator32 and a discriminator32 in `utils/model.py` separately and combine them into a GAN in `training.py`. The generator takes the input fragments and the label and outputs the complete model. The discriminator takes the complete model and the label and outputs a score. It should be emphasized that label are needed for both the generator and the discriminator, which means that the GAN model we built is actually a conditional-GAN.

The generator is built as the picture shown: In the first branch we use 1 Encoder (named as encoderl) to convert the label into a 1024 dim feature vector. In the second branch, we use 5 sequentially connected Encoder layers (named encoder_i) to encode the input fragments into a 1024 dim feature vector. Then we concatenate these two convert them into a 1024 dim feature vector. After these step, we use 5 Decoder layers (named as decoder_i) to decode the feature vector into a 3D-voxel, which is the output of the generator. The discriminator is similar in the first two step, and convert the feature vector into a 1 dim feature vector, which is the final output of the discriminator.

<img src='images/paper1.png' width = 300>
<img src='images/paper2.png' width = 300>
<img src='images/paper3.png' width = 300>
<img src='images/paper3.png' width = 300>

<center><b>picture in the paper</b></center>

<img src='images/model-G1.png' width =200>
<img src='images/model-G2.png' width =200>
<img src='images/model-G3.png' width =200>
<img src='images/model-G4.png' width =200>
<img src='images/model-G5.png' width =200>

<center><b>generator32 we build</b></center>

<img src='images/model-D1.png' width =200>
<img src='images/model-D2.png' width =200>
<img src='images/model-D3.png' width =200>

<center><b>discriminator32 we build</b></center>

<img src='images/model-SE.png' width =300>
<img src='images/model-enc1.png' width =300>
<img src='images/model-enc2.png' width =300>
<img src='images/model-dec.png' width =300>

<center><b>SE layer/Encoder/Decoder we build</b></center>

There're 2 points to note in this model: 1. Every encoder follows not only the 'Conv-Act-Norm' paradiam, but also follows a SE-layer. Decoders are the same except for replacing the 'Conv-layer' with 'TransConv-layer'. Details parameters are in the picture. 2. The generator use 'skip connection' technique which means directly concatenanting one feature map with a very late feature map. 

## Part4 Working Details

### (1) Visualization
introduce what kind of visualization we did(maybe some pictures)

We download the `pyvox` utils constituting `pyvox.parser` library from [github repository](https://github.com/gromgull/py-vox-io). 
With the help of `pyvox parser` and `plotly` library, we've realized the visualization of both the whole voxel object and its chosen fragments and enabled the interaction of a fragmentized voxel object. 
Furthermore, in order to control the resolution of voxel object, we utilize down-sampling to shrink object size from $64^3$ to $32^3$. For any dimension whose resolution is less than 64, we use concatenation for lengthening. 
The visualization result is as follows:
1) visualize the whole object
![](images\plot.png)
1) visualize the whole object with each fragment colored separatedly
![](images\plot_frag.png)
1) allowing for clicking specific fragment and mopping it off the plot
![](images\newplot.png)

### (2) Data Processing
#### a) the raw data
The raw data contained the voxel models of the fragments. The models were not larger than $64\times64\times64$. Each voxel of the models either belonged to a unique fragment or was empty.

There were $11$ categories of pottaries indexed from $1$ to $11$. Each category contains several pottaries with different shapes, and for each pottary, voxel models of fragments with different number of fragments (at least $2$ and no more than $17$) are given.

#### b) pre-processing
To stack the voxel data into Torch arrays, we first padded the voxel models to ensure they were of the shape $64\times64\times64$. In detail, we check whether the size of the dimension is less than $64$ for each dimension. If so, we inserted zeros to the back of the dimension.

We implemented two settings. For the low-resolution setting, we downsampled the voxel models by sampling the voxels with even indices (i.e. $0,2,\dots,63$). For the high-resolution setting, we did nothing.

For the training stage, we first shuffled the data. For each voxel model, we randomly selected several pieces of fragments and combined them as the input and took the complete model as the ground-truth. We guaranteed that the input contains at least one piece and not all the pieces.

For the test stage, to make the results stable, we only randomly sampled the pieces for each voxel model in the first epoch. And in the following epochs, we reused the inputs in the first epoch.

#### c) analysis
The same complete models were divided into different sets of fragments, which meant there were sereral data with the same ground-truths. This might break the property of IID.

However, it also allowed us to strengthen the dataset by randomly combining the fragments as inputs. We could generate a large number of combinations for each pottary, which contributed to preventing overfitting.

### (3) Training Framework

We used Pytorch to implement the training framework in training.py, named as `GAN_trainer`, shown in pictures below. In this model, we implements several functions, including args setting, data loading, model loading, model saving, model initialization, G_training, D_training, loss_drawing. In the main function, we initialize the model, load the data, and start the training process. We adapted the popular training way: update D every time and update G every 5 times. For better visualization, we also implemented a color progressing bar in console.

<img src='images/train1.png' width=300>
<img src='images/train4.png' width=300>
<img src='images/train2.png' width=300>
<img src='images/train3.png' width=300>

<center><b>class GAN_trainer and some of its member functions</b></center>

<img src='images/train0.png' width=300>
<img src='images/train-loss.jpg' width=300>

<center><b>left:training process;    right:loss picture in one trial</b></center> 

<img src='images/train-bar.png' width=400>

<center><b>colorful progressing bar in console</center></b>

### (4) Remote Environment Setting

- introduce how to set up environment in remote server

We aim to utilize [Google Colab](https://colab.research.google.com/) for free use of GPU training hours. Currently, we've gone through [this tips for using Colab](https://zhuanlan.zhihu.com/p/666938608) and are trying to adapt our code to fit the requirements of Colab. Major efforts including zipping the dataset to benefit uploading, adding command-line parameters to allow for adjustment of dataset path and model output path, equipping the remote environment to satisfy our dependency needs, adding checkpoint in case of sudden network connection shutdown and so on.

## Reference
Mirza, Mehdi and Simon Osindero. “Conditional Generative Adversarial Nets.” ArXiv abs/1411.1784 (2014): n. pag.

Goodfellow, Ian J. et al. “Generative Adversarial Nets.” Neural Information Processing Systems (2014).

Larsen, Anders Boesen Lindbo et al. “Autoencoding beyond pixels using a learned similarity metric.” ArXiv abs/1512.09300 (2015): n. pag.

Radford, Alec et al. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” CoRR abs/1511.06434 (2015): n. pag.

Oord, Aäron van den et al. “Conditional Image Generation with PixelCNN Decoders.” ArXiv abs/1606.05328 (2016): n. pag.

Arjovsky, Martín et al. “Wasserstein GAN.” ArXiv abs/1701.07875 (2017): n. pag.

Karras, Tero et al. “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” ArXiv abs/1710.10196 (2017): n. pag.

Zhang, Han et al. “Self-Attention Generative Adversarial Networks.” ArXiv abs/1805.08318 (2018): n. pag.

Miyato, Takeru et al. “Spectral Normalization for Generative Adversarial Networks.” ArXiv abs/1802.05957 (2018): n. pag.

Brock, Andrew et al. “Large Scale GAN Training for High Fidelity Natural Image Synthesis.” ArXiv abs/1809.11096 (2018): n. pag.

Karras, Tero et al. “A Style-Based Generator Architecture for Generative Adversarial Networks.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2018): 4396-4405.