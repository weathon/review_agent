# Understanding Contrastive Learning through Variational Analysis and Neural Network Optimization Perspectives

- Avg Score: 4.75
- Decision: Reject
- Scores: 5, 5, 3, 6

## Abstract
The SimCLR method for contrastive learning of invariant visual representations has become extensively used in supervised, semi-supervised, and unsupervised settings, due to its ability to uncover patterns and structures in image data that are not directly present in the pixel representations. However, the reason for this success is not well-explained, since it is not guaranteed by invariance alone. In this paper, we conduct a mathematical analysis of the SimCLR method with the goal of better understanding the geometric properties of the learned latent distribution. Our findings reveal two things: (1) the SimCLR loss alone is not sufficient to select a "good" minimizer --- there are minimizers that give trivial latent distributions, even when the original data is highly clustered --- and (2) in order to understand the success of contrastive learning methods like SimCLR, it is necessary to analyze the neural network training dynamics induced by minimizing a contrastive learning loss. Our preliminary analysis for a one-hidden layer neural network shows that clustering structure can present itself for a substantial period of time during training, even if it eventually converges to a trivial minimizer.   To substantiate our theoretical insights, we present numerical results that confirm our theoretical predictions.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims to understand the performance of contrastive learning (especially the property of preserving cluster structures). This paper first proposes that studying the global minimizer of the contrastive loss is insufficient since (within the invariance of the map T) there exist minimizers that is free of the data distribution. The authors further propose that the good properties of contrastive learning is related to the training dynamics and justify this via gradient flow in Theorem 4.2.

### Strengths
1. This paper is well written and organized, and offers very interesting insights into the understanding of contrastive learning.
2. The presented theoretical results are solid, which is also justified by experimental results.

### Weaknesses
1. While theoretical results in this paper are solid and insightful, my concern lies in the reasoning following Proposition 2.1. This result is based on the assumption on the existence of an invariant T, thus the minimizer to the optimization problem (on line 195) is implicitly depending on the distribution of data via the invariant T. In my understanding, this proposition is similar to a two-step analysis, i.e. fixing a map that captures the information of the underlying data and then by optimizing (line 195), the loss enforces some properties e.g. uniformity. Hence, I think Proposition 2.1 is insufficient to justify that the global solution of the contrastive loss cannot preserve the structures in data and the fine-grained analysis may be needed.
2. It would be more helpful if the theoretical results (e.g. Proposition 4.3) based on gradient flow can be elaborated with some concrete examples of \mu so that the gradients can be intuitively understood.

### Questions
1. [Notations] I assume equation (3) is referring to "equal almost surely".
2. [Prop 2.1] I was wondering even if the invariance of T is given, what should be the minimizer of the optimization problem (on line 195). Should the uniformity of the distribution be enforced by this loss? I think it could be relevant to the property of Gaussian potential.
3. [Choice of similarity measure] As the theory applies to a general family of the similarity measure \eta, I was wondering if the gradient flow would suggest the optimal choice of \eta? Or, why is the cross-entropy type of \eta better than the linear functions in practice.
4. [Explicit form of the gradient flow] Similar to what's mentioned in Weakness, It would be more helpful if the theoretical results (e.g. Proposition 4.3) based on gradient flow can be elaborated with some concrete examples of \mu so that the gradients can be intuitively understood. In addition, there are several theoretical analysis of contrastive loss and its global minimizers using concrete data distributions that can be relevant:
	- Ji, Wenlong, Zhun Deng, Ryumei Nakada, James Zou, and Linjun Zhang. "The power of contrast for feature learning: A theoretical analysis." Journal of Machine Learning Research 24, no. 330 (2023): 1-78.
	- Wang, Tongzhou, and Phillip Isola. "Understanding contrastive representation learning through alignment and uniformity on the hypersphere." In International conference on machine learning, pp. 9929-9939. PMLR, 2020.
	- Gui, Yu, Cong Ma, and Yiqiao Zhong. "Unraveling Projection Heads in Contrastive Learning: Insights from Expansion and Shrinkage." arXiv preprint arXiv:2306.03335 (2023).
	- Wen, Zixin, and Yuanzhi Li. "The mechanism of prediction head in non-contrastive self-supervised learning." Advances in Neural Information Processing Systems 35 (2022): 24794-24809.
5. [Figure 4] Are the arrays representing the direction of the gradients? In addition, on row 3, by saying vanilla gradient descent, what is the encoder for this setting?

I will be happy to raise my score if some fine-grained analysis of the gradient flow can be presented.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores properties of self-supervised contrastive learning. In particular an optimality criterion for a generalized form of the popular NT-Xent loss function (used in the SimCLR framework) is derived and used to find a family of functions that satisfy this criterion. 
Of particular interest is the fact that optimal solutions exist that are in some sense "independent" of the input data distributions, i.e. they do not recapitulate the clustering structure of the inputs.
The author's then utilize the NTK framework to analyze the inductive biases of training a neural network using said loss function.
This theoretical analysis of the learning dynamics and empirical results both shed light on how optimization of SSL loss functions with neural networks tend to find useful solutions/representations despite the fact that uninteresting optima exist.

### Strengths
- The core topic, why SSL leads to useful representations despite the fact that uninteresting minimizers exist, is of interest to the community, and utilizing NTK to examine the inductive biases of neural network training is an interesting approach. 

- Including several different families of SSL (instance-contrastive/SimCLR, dimension-contrastive/VicReg, and non-contrastive/BYOL) in the analysis strengthens the generality of the analysis. 

- The toy-setting and accompanying theoretical analyses are well described and useful for building intuition.

### Weaknesses
- The first main contribution, showing that SSL losses have minimizers that are in some sense independent of the input data distribution, is well known and there are multiple existing studies that propose theoretical reasons for why useful representations emerge from training (and I feel this works contribution and novelty could be strengthened by interfacing with them). Two examples of this type are: 
  - "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" [Wang and Isola, 2023]: shows that contrastive learning optimizes for embedding uniformity. Given this, and the fact that the loss is clearly invariant to the permutation of (invariant) embeddings it is obvious that minimizers from which there is no ability to separate ground truth classes.
  - "Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss" [HaoChen et al., 2022]: Defines a concept of the "augmentation graph," and provides bounds on downstream class separability given that class-membership clusters form connected sub-graphs. Similarly "Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods" [Balestriero et al., 2022] derive optimality conditions for various losses and give performance guarantees in the condition that augmentation similarity graphs overlap with class similarity graphs. 

- It is not obvious to me what the implications of this theoretical work are for SSL practitioners. I believe the impact of the work would be strengthened substantially if the theory could be used to levy practical suggestions for improving the quality of learned representations (and strengthened even more if the author's could empirically demonstrate the efficacy of their suggestions!).

### Questions
- My understanding is that in the "Vanilla Gradient Descent" setting, the position of the embedding is directly optimized. My question is that, if the NT-Xent loss is used, how do you ensure that embeddings remain on the constraint surface (i.e. remain unit vectors)? Are embeddings projected to the unit sphere after each gradient step? Do the author's think there may be any implications for the results if some other way of remaining on the constraint surface (i.e. gradient steps that leave outputs on the unit sphere)?

- I wonder if the author's could comment on how their study (or its implications) are impacted by the near ubiquitous used of projector networks ("Guillotine Regularization: Why removing layers is needed to improve generalization in Self-Supervised Learning," [Bordes et al., 2022]) in SSL. Meaning most SSL methods actually discard the final layers of the network (where the loss is optimized) when transferring to downstream tasks. In light of this, might it be interesting to use a similar framework to analyze the dynamics of internal representations of deeper networks during SSL training (i.e. perhaps clustering structure is maintained longer or more pronounced in layers preceding loss evaluation)?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This study presents preliminary results investigating the SimCLR contrastive learning method to better understand its effectiveness. The findings suggest that the SimCLR loss function alone is insufficient for selecting an optimal minimizer, and a thorough analysis of the neural network training dynamics is essential.

### Strengths
This study examines the learning dynamics of the SimCLR loss function.

### Weaknesses
The motivation for the study in the introduction is unclear. There is a lack of a comprehensive overview of the numerous analytical studies aimed at understanding and improving the success of contrastive learning. It is essential to explicitly outline how this research differs from existing studies and what new findings it presents. A detailed section on related works should be included to address these points. Additionally, it should be clarified how this research serves as a complement to existing studies on dimensional collapse.

The two contributions claimed in this research lack sufficient evidence to support their relevance to the success of SimCLR. In fact, the results appear to highlight the shortcomings of contrastive learning rather than reinforcing its effectiveness. Clear justification is needed to demonstrate how these contributions substantiate the efficacy of contrastive learning.

Furthermore, the mathematical notation used in this research is not conventional, which diminishes readability. For instance, many subsequent studies, including those on SimCLR, commonly use T for augmentation and f for the encoder. In contrast, this study employs notations that are inverse to this convention.

Assumptions necessary for the validity of Equation 3 have been omitted. The absence of these assumptions may lead to an overgeneralization of the theoretical results derived from them. For example, if the augmentation is exceedingly strong—such as an augmentation that reduces all values to zero—Equation 3 would not hold true. Moreover, during the actual training of contrastive learning, the learning occurs in batches, further complicating the validity of Equation 3.

### Questions
The primary objective of contrastive learning is to encode various features present in the data into a representation. While clustering-related features are indeed included, there are numerous other features that also contribute to this encoding process. Given this consideration, the assumption in this study that clean data is highly structured or clustered may be viewed as a narrow and potentially unrealistic constraint.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes to study trivial minimizers of InfoNCE, their link to the data distribution as well as their appearance when using neural networks. Indeed, the InfoNCE criterion has trivial minimizers (collapsing to a single point, or a discrete measure) that can be learned by gradient descent. Studying neural network dynamics, the authors show that the clustering structure of the data can remain present during training, and may help explain the success of contrastive self-supervised learning.

### Strengths
- The focus on the training dynamics with a neural network leads to an analysis that is significantly closer to practice than what an analysis of the loss function alone would provide. While it is common knowledge that trivial solutions exist, it is important to understand how they emerge or not
- The study of the impact of the data distribution, while preliminary, sheds light into the success of SSL through the preservation of properties of the data distribution (e.g. clustering here)
- The tools used in this work could be repurposed to study in more depth the influence of the data distribution on the representations learned via SSL. Notably, the brittleness of methods to non-uniform data distributions that is observed empirically may be theoretically explainable.

### Weaknesses
- The caption of Figure 2 should be self-sufficient, describing every part of the figure. Notably, describing that $T$ is the embedding function, $f$ corresponds to the augmentations etc would make the plot more self-contained and help provide a more graphical description of the setup.
- Most of the analysis focuses on the setting where $T$ is already invariant to $\nu$. This means that the studied criterion is reduced to the 
repulsive force rather than the complete InfoNCE. A reference to [1] is also missing, which studied minimizers of InfoNCE, and notably the link between minimizing the repulsive force and the uniform distribution on the hypersphere.

[1] Wang, T., & Isola, P.. Understanding contrastive representation learning through alignment and uniformity on the hypersphere.ICML, 2020.

### Questions
- In figure 4, why is the choice of number of iterations for the two rows different ? Is it simply for visualization (to get similar “convergence” states), or does the behaviour change if we continue the first row to 1000 iterations ? I would suggest explaining the rationale behind the choices or show plots at convergence as well.
- More of an open ended discussion, but how do you think that the choice of architecture for the encoder influences how the underlying structure of the data is present in the embedding ? We see in figure 4/5 that with a 1/4 layer neural network, we roughly keep the clustered structure of the data (except the blue cluster which is split in two at 200 iterations). Perhaps more powerful architectures (ResNet,ViT) may have behaviours akin to the “vanilla gradient descent” setting, as asymptotically, a network with infinite capacity would be closer to this setting. Experiments are of course not expected here.
- Previous work [2] studied the importance of uniform clusters of features in self-supervised learning, mainly how the learned representation better capture  features present uniformly rather than following a power law. Perhaps some of the analysis presented here could be extended to a setting with non uniform clusters to help explain why contrastive learning tends to fail on non-uniformly distributed data. Again, this is mainly a discussion point.

[2] Assran, M., Balestriero, R., Duval, Q., Bordes, F., Misra, I., Bojanowski, P., ... & Ballas, N. (2022). The hidden uniform cluster prior in self-supervised learning. arXiv preprint arXiv:2210.07277.

### Soundness
3

### Presentation
3

### Contribution
3
