# CV大作业中期报告

2024.1.17 成员：xxx, xxx, xxx

## Part1 Introduction[xly]
(a brief introduction，such as“我们选择的题目是xxx，经过xxx时间我们目前完成了xxx内容、达到了xxx效果，后续计划完成xxx”)

## Part2 Related Works

### (1) GAN based models[wmq]
introduce a variety of GANs

### (2) main paper introduction[xly]
introduce the first two papers mentioned in writeup

## Part3 Our Approach

### (1) Problem Definition[mys]
In a word, what we are going to do is to predict models of the complete pottaries, given models of the fragments. The 3D models are represented as voxels of shape $32\times32\times32$ or $64\times64\times64$.

It's worth noting that the task is a prediction task, instead of a generation task. The difference is that in out prediction task, the ground-truths are given, while in a generation task, they are not given. The task determined that our approach was fully-supervised.

Our task setting is almost the same as *"3D reconstruction of incomplete archaeological objects using a generative adversarial network" (Hermoza, 2018)*, with a slight difference that the inputs of our tasks were fragments (or combinations of fragments), while in Hermoza (2018)'s work, the inputs were randomly sampled from the complete models.

### (2) Model Setting[wmq]
model layer structure/hyper parameters/loss function/optimizer/training strategy

## Part4 Working Details

### (1) Visualization[xly]
introduce what kind of visualization we did(maybe some pictures)

### (2) Data Processing[mys]
introduce how data was processed and how the input and output data was organized

### (3) Training Framework[wmq]
introduce codes in training.py(mention this time, won't appear in final report)

### (4) Remote Environment Setting[xly]
introduce how to set up environment in remote server(mention this time, won't appear in final report)