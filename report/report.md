# CV大作业中期报告

2024.1.17 成员：xxx, xxx, xxx

## Part1 Introduction[xly]
The topic we choose for final project is *Let's Make Pottery: GANS for Voxel Completion in Ancient Pottery Dataset* whose Quizmaster is class TA YuXuan, Luo.
This project mainly revolves around building a GAN that takes fragmented artifacts' voxel representation as input and outputs the missing part of the corresponding piece so as to assist the pottery reconstruction work for archaeological studies. Till the middle checkpoint, we've successfully visualized the dataset, implemented the strengthened data-loader and constructed the structure of a naive GAN which can smoothly compile and train. For further improvement, we plan to train it on a larger basis to collect long-term data of its performance and visualize its output. Eventually we will generalize our model to $64^3$ resolution and derive loss functions utilizing normal vectors and other significant information to improve the reconstruction performance.

## Part2 Related Works

### (1) GAN based models[wmq]
introduce a variety of GANs

### (2) main paper introduction[xly]

- introduce the first two papers mentioned in writeup

Till now, we've read the first two papers mentioned in writeup, namely *Generative Adversarial Nets* (Goodfellow, 2014) and *3D Reconstruction of Incomplete Archaeological Objects Using Adversarial Network* (Hermoza, 2018).

From the first paper, we've learnt the essence of a GAN which is a combination of a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. We've also gone through the mathematics foundation of the global optimality and convergence of a GAN-based algorithm, from which we derived several training strategies including alternating between the training of D and G. The paper also opens our minds to further improvement of GAN structure such as introducing a variational auto-encoder in the generator to allow for the learning of the conditional variance.

The second one mainly illustrated a variant of GAN, namely the ORGAN(OR is short for object reconstruction), drawing a fundamental picture of 3D GAN. It gave detailed description regarding its structure(a 3D CNN with **skip-connections**, as a **generator** on a Conditional GAN (CGAN) architecture. With two optimization targets: a mean absolute error (MAE) and an Improved Wasserstein GAN (IWGAN) loss) in additional to training details including the choice of hyper-parameters, which provided us with much reference and experience to rely on. Nevertheless, the dataset we use is quite different from the latter paper, which pushes us to design our own dataloader and consequently the exact training techniques need further experiments.

## Part3 Our Approach

### (1) Problem Definition
In a word, what we are going to do is to predict models of the complete pottaries, given models of the fragments. The 3D models are represented as voxels of shape $32\times32\times32$ or $64\times64\times64$.

It's worth noting that the task is a prediction task, instead of a generation task. The difference is that in out prediction task, the ground-truths are given, while in a generation task, they are not given. The task determined that our approach was fully-supervised.

Our task setting is almost the same as *"3D reconstruction of incomplete archaeological objects using a generative adversarial network" (Hermoza, 2018)*, with a slight difference that the inputs of our tasks were fragments (or combinations of fragments), while in Hermoza (2018)'s work, the inputs were randomly sampled from the complete models.

### (2) Model Setting[wmq]
model layer structure/hyper parameters/loss function/optimizer/training strategy

## Part4 Working Details

### (1) Visualization[xly]
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

### (3) Training Framework[wmq]
introduce codes in training.py(mention this time, won't appear in final report)

### (4) Remote Environment Setting[xly]

- introduce how to set up environment in remote server

We aim to utilize [Google Colab](https://colab.research.google.com/) for free use of GPU training hours. Currently, we've gone through [this tips for using Colab](https://zhuanlan.zhihu.com/p/666938608) and are trying to adapt our code to fit the requirements of Colab. Major efforts including zipping the dataset to benefit uploading, adding command-line parameters to allow for adjustment of dataset path and model output path, equipping the remote environment to satisfy our dependency needs, adding checkpoint in case of sudden network connection shutdown and so on.
