# Part1 Introduction【复用】
【问题的重要性，问题的设置，简单提一下方法】
# Part2 Related Works【可以直接复用】
【整理的相关论文】
# Part3 Methods【复用+64】【mys-loss设置】
【模型】【配图：论文的图片（复用）】
# Part4 Experiments
## 1. Training details
We used the AdamW optimizer by default. Except the learning rates, the arguments of AdamW were set to the default setting. We trained the model in 3 stages.

In 1st stage, we set both the learning rates of the generator and the discriminator to $1\times10^{-5}$ and trained the GAN for $20$ epochs. The generator was trained every $3$ steps and the discriminator was trained every step. And the $\mathrm{DiffLoss}$ loss of the generator was not scaled in this stage.

In 2nd stage, we set the learning rate of the generator to $1\times10^{-3}$ and the learning rate of the discriminator to $1\times10^{-5}$ and trained the GAN for $20$ epochs. The generator was trained every $2$ steps and the discriminator was trained every step. In this stage, we magnified the weights of the non-empty voxels $100$ times.

In 3rd stage, the parameter setting was the same as that in 2nd stage, except that we magnified the weights of the non-empty voxels $20$ times and trained the GAN for only $10$ epochs.

1. Ablation Study【控制变量/对比实验】
	1. 测试输入为空，输出的结果是什么【xly】
		1. 【一个配图】
	2. 测试只训练gan的G部分，是否可以恢复模型【wmq】
		1. 【一个配图】
	3. 测试给gan的G部分的decoder投一个随机项链，是否可以生成结果【wmq】
		1. 【一个配图】
# Part5 results analysis
1. 通过指标比较两组模型的效果（wmq交两个模型，mys分析效果）
	1. ‘VAE vs GAN32 vs GAN64’
		1. 【一个表格】
2. title
	1. 测试同一个成品，随着输入碎片的数量增加，输出模型的变化【xly】
		1. 【一个表格+一个配图】
# Part6 Conclusion【mys】
