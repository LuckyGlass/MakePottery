# CV大作业中期报告

2024.1.17 成员：xxx, xxx, xxx

## Part1 Introduction[xly]
(a brief introduction，such as“我们选择的题目是xxx，经过xxx时间我们目前完成了xxx内容、达到了xxx效果，后续计划完成xxx”)

## Part2 Related Works

### (1) GAN based models[wmq]
introduce a variety of GANs

### (2) main paper introduction[xly]

- introduce the first two papers mentioned in writeup

Till now, we've read the first two papers mentioned in writeup, namely ***Generative Adversarial Nets*** by Goodfellow and ***3D Reconstruction of Incomplete Archaeological Objects Using Adversarial Network*** by Renato.
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
introduce how to set up environment in remote server(mention this time, won't appear in final report)