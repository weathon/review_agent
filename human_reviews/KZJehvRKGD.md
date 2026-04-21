# Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit

- Avg Score: 7.50
- Decision: Accept (poster)
- Scores: 8, 6, 8, 8

## Abstract
The cost of hyperparameter tuning in deep learning has been rising with model sizes, prompting practitioners to find new tuning methods using a proxy of smaller networks. One such proposal uses $\mu$P parameterized networks, where the optimal hyperparameters for small width networks *transfer* to networks with arbitrarily large width. However, in this scheme, hyperparameters do not transfer across depths. As a remedy, we study residual networks with a residual branch scale of $1/\sqrt{\text{depth}}$ in combination with the $\mu$P parameterization. We provide experiments demonstrating that residual architectures including convolutional ResNets and vision transformers trained with this parameterization exhibit transfer of optimal hyperparameters across width and depth on CIFAR-10 and ImageNet.  Furthermore, our empirical findings are supported and motivated by theory. Using recent developments in the dynamical mean field theory (DMFT) description of neural network learning dynamics, we show that this parameterization of ResNets admits a well-defined feature learning joint infinite-width and infinite-depth limit and show convergence of finite-size network dynamics towards this limit.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors study hyperparameter transfer between models at different scales with an emphasis on the hyperparameter transfer of ResNets (or ResNets sub-components) with depth. Following parts of the rational behind muP parameterization, they argue that good parameterizations are those having stable infinite width/depth limits. They then consider one such recent limit ("1/\sqrt{depth}" scaling) and extend prior theoretical results on ResNets at initialization to ResNets trained with gradient flow. Specifically, they establish that network updates are also stable at this limit. The manuscript contains several experimental results on CIFAR-10,Tiny ImageNet, and ImageNet showing successful hyper-parameter transfer based on this "1/\sqrt{depth}" scaling. As an additional theoretical side, the DMFT equations are solved for an infinite widh&depth linear network acting on a single data point.

### Strengths
The work offers a good combination of theory and experiment.  

It expands a useful and timely avenue of research, hyperparameter transfer, where theory can actually aid practitioners.

It studies relevant complex architecture acting on realistic datasets as opposed to linear or shallow networks.

### Weaknesses
Several gaps exist between the stated goal of the theory and its practical implementation. One such gap, which the authors themselves cite, is that the theory predicts stable results following a scale-up. Naturally, however, spending more compute and getting equal performance, is of little practical value. In practice, one expects to get better results which seems to be the case at least for the larger datasets they consider. This means that in examples where HP-transfer is worthwhile, one is by definition well away from the scaling limit. 

Another gap, of a similar kind, is that, as far as I could tell, their DMFT framework is based on gradient flow but it is then used to transfer the learning rate. While gradient flow may be a good approximation for SGD in the asymptotically low learning rate regime, typically the optimal learning rates lay far away from this regime. 

For CIFAR-10, the 1/\sqrt{depth} scaling leads to poorer performance compared to muP scaling. 

Reference to current literature could be broadened and sharpened. For instance, muP parameterization is also a unique parameterization having maximal updates, not just a stable limit (This fuller characterization also partially escapes the first gap described above). Citing further alternatives to muP scaling and adaptive kernel approaches to feature learning is also desirable.

### Questions
1. Can the authors rationalize why the above two gaps could be excused? 

2. Can the authors explain why for CIFAR-10 their parameterization seems non-optimal?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors' contribution in this work lies in their introduction of a straightforward parameterization for residual networks. Through empirical observations, they have demonstrated the remarkable consistency of hyperparameter transfer across various dimensions of network architecture, such as width and depth, as well as across diverse hyperparameters and datasets. This empirical evidence is further substantiated by theoretical analysis, which reveals that the behaviors of hidden layers remain intricate, unaffected by changes in network dimensions, and do not approach vanishing limits.

### Strengths
This paper demonstrates that a straightforward adaptation of the $\mu P$  parameterization enables the transfer of learning rates across both depth and width in residual networks, incorporating $1/\sqrt{depth}$ scaling on the residual branches. The authors conduct extensive experiments. The theory is also nice.

I think these findings have the potential to significantly reduce the computational expenses associated with hyperparameter tuning, thereby enabling practitioners to train large, deep, and wide models just once while achieving near-optimal hyperparameters.

### Weaknesses
1. Since that the $\mu P$ parameterization is not a commonly used method, the authors should provide a more comprehensive introduction.

2. The $1/\sqrt{depth}$ trick has been extensively investigated in previous works [1-3]. The novelty of the proposed method appears to be limited. Please clarify the differences. Moreover, many recent works propose initializing ResNet using "rezero"-like methods, which achieve similar performance to the $1/\sqrt{depth}$ trick. The authors should provide some comparisons. In particular, I would like to see some discussions on the scaling parameter. Why use $1/\sqrt{depth}$? What would happen if the scaling parameter is set as $1/depth$ or 0?

3. As the authors admitted in the "Limitation" paragraph, most of the experiments are confined to 10-20 epochs. This limitation makes the empirical evidence less convincing. In fact, I don't understand the difficulty in training a ResNet with more epochs if one can train it with 10-20 epochs.

[1] Tarnowski W, Warchoł P, Jastrzȩbski S, et al. Dynamical isometry is achieved in residual networks in a universal way for any activation function. The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019: 2221-2230.

[2] Yang G, Schoenholz S. Mean field residual networks: On the edge of chaos. Advances in neural information processing systems, 2017, 30.

[3] Zhang H, Dauphin Y N, Ma T. Fixup initialization: Residual learning without normalization. arXiv preprint arXiv:1901.09321, 2019.

### Questions
Please see  weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Paper proposes a novel parameterisation of deep residual networks that tackles width- and depth-dependent cost of hyper-parameter tuning, an issue, exacerbated by recent increase in SOTA models' sizes. A novel $1/L$ extension of $\mu P$ parameterization of residual networks is proposed. Paper argues that hyperparameter transfer is consistent across both width and depth for a variety of architectures, hyperparameters, and datasets. 
The work is primarily empirical whose experimental evidence is supported by statistical physics and NTK theory. In particular, the paper advances a width-invariant feature updates present in $\mu P$ parameterisation and in the same spirit derives depth scaling to ensure feature updates invariance over depth using dynamical mean field theory (DMFT). 

Exact solutions of the DMFT dynamics in the rich (i.e.non-kernel) regime for a simple deep linear network are provided. The suggested parameterisation is verified empirically by well selected range of experiments, mostly presented in appendices and  summarised in the Section 3. The exposition of the challenges, method and main results is well and accessibly written rounded by limitations and future directions.

### Strengths
+ Rounded, well written and easy to follow timely paper presenting valuable empirical evidence put into NTK theory context.
+ Future directions section proposes the method as suitable to study depth-width trade-offs while scaling neural networks since hyperparameter dependencies are consistent across widths/depths. In my opinion this can be very valuable addition to community even if the method does not turn out practically useful and adds positively to my overall rating.
+ Convincing experiments (with a caveat, that I hope will be addressed in the rebuttal, see Weaknesses, ad 1)) cover well selected range of settings and architectures

### Weaknesses
1. Fig. 1 Loss levels reached seem to be higher in case of proposed parameterization compared to alternatives. Could authors add some comments on the topic? Especially, is a proposed parameterization capable of reaching an optimal value of hyper parameters in scope, e.g., learning rate in practical settings? I rate paper 'accepted' conditioned on this issue will be alleviated in the camera-ready version.

2. Limited applicability(?) - Could authors argue otherwise? How could it be improved in the paper? Could authors elaborate on computational costs of proposed method vs. alternatives?

3. Paper heavily depends on $\mu P$ parameterisation properties derived in Yang & Hu (2021). In my view a short primer on $\mu P$ parameterisation (even in Appendix) would improve self-consistency and readability of the paper.

4. Proposition 2, Assumption $\gamma_0 \rightarrow 0$ renders ODE from Proposition 1 solvable due to time invariant kernels (and thus given by Gaussian initialization) and the same goes for the second example where linearity of network makes time invariance explicit. To what extend are conclusion transferable to any realistic non-linear scenario?

### Questions
See section Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a parameterization for Resnet neural networks which is claimed to enable the transfer of hyperparameters such as the learning rate over a large range of network widths and depths. The motivation for this work is to enable engineers to tune hyperpaprameters on small models and then apply them to large models, avoiding a costly fine tuning on large models.

The method extends the muP parameterization which, according to the authors, enables hyperparameter transfers across different widths but not different depths. The proposed parameterization consists of zero-mean unit-variance parameter initialization and scaling factors for the residual branches, output pre-activations and learning rate that depend on both the width and depth of the model, and no normalization layers.

The proposed parameterization is derived by theoretically analyzing the behavior of a (simplified) neural network in the limit of  infinite width and infinite depth, and choosing a parameterization which satisfy certain desiderata. Experiments on a small dataset CFAR-10 are used to support the analysis on realistic neural architectures.

### Strengths
- The method tackles an important practical problem: the efficient tuning of hyperparameters.
- Good presentation

### Weaknesses
- The experimental section is quite limited: only a single task and a very simple dataset are considered.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
