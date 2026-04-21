# Generalized Activation via Multivariate Projection

- Avg Score: 6.67
- Decision: Reject
- Scores: 8, 6, 6

## Abstract
Activation functions are essential to introduce nonlinearity into neural networks, with the Rectified Linear Unit (ReLU) often favored for its simplicity and effectiveness. Motivated by the structural similarity between a shallow Feedforward Neural Network (FNN) and a single iteration of the Projected Gradient Descent (PGD) algorithm, a standard approach for solving constrained optimization problems, we consider ReLU as a projection from $\mathbb{R}$ onto the nonnegative half-line $\mathbb{R}_+$. 
Building on this interpretation, we extend ReLU by substituting it with a generalized projection operator onto a convex cone, such as the Second-Order Cone (SOC) projection, thereby naturally extending it to a Multivariate Projection Unit (MPU), an activation function with multiple inputs and multiple outputs.
We further provide a mathematical proof establishing that FNNs activated by SOC projections outperform those utilizing ReLU in terms of expressive power. Experimental evaluations on widely-adopted architectures further corroborate MPU's effectiveness against a broader range of existing activation functions.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors introduce "Multivariate Projection Unit" (MPU), a novel activation function that, differently from a standard AF, takes multiple inputs and also returns multiple outputs (MIMO). The key observation is that ReLU can be seen as a a projection to the positive orthant, and linear multiplication + ReLU can be seen as a step of projected gradient descent. The MPU, instead, splits the output of the linear projection into blocks of same size (see Fig. 1), and projects each block onto a prespecified cone. This is motivated by the analogy with a PGD step on a different (more general) class of optimization problems. They validate the MPU on artificial benchmarks, a CNN on CIFAR-10 and ImageNet, and a vision transformer (only in the supplementary material), showing results that are similar or slightly better than ReLU and some ReLU variants (e.g., Leaky ReLU).

### Strengths
To the best of my knowledge, the MPU is novel, and it is an interesting variation of a standard ReLU with a good underlying motivation. The paper is well written, in particular the visualizations in Fig. 1 immediately show the basic idea of the paper. I am less convinced about the empirical evaluation (see below), so the practical value of the MPU is not clear.

### Weaknesses
I have a few general comments on the manuscript, making this (at the moment) a borderline paper for acceptance. I think most of these questions are addressable and I would be happy to increase my score.

EXPOSITION: I have found the exposition of the paper a bit strange, because there is a very long motivation for the MPU (both Section 2.1 and Section 3 serve as a motivation), but very little analysis of the MPU itself. For example: (i) there is no explicit definition of the MPU or the MPU layer; (ii) there is no discussion on how to choose the cone (they only specify which cone they use in the experiments); (iii) no discussion on how to perform the projection (which is relegated to an appendix); (iv) no discussion on the theoretical computational complexity (only an empirical computation of MACS).

RESULTS: The results do not seem very strong. Ignoring the artificial datasets, on CIFAR-10 (Table 1) it is inside 1 std of Leaky ReLU. On ImageNet (Table 2), LeakyReLU is superior but also inside 1 std. DeiT (Appendix F) is only provided for a single run and no comparison, and ReLU is still only at 0.07 distance. Also, they are not providing many important ablations and comparisons, including on the choice of the cone or its dimensionality. Many baselines are missing (see below).

RELATED WORKS: the related works section is very shallow. Strangely, they are mentioning some works on multi-input AFs which are not used in the comparisons (e.g., Maxout, CReLU). However, many other works are missing (see, e.g., https://arxiv.org/pdf/2005.00817.pdf), including winner-take-all AFs, network-in-network models. Also, complex activation functions can natively work with 2 inputs and 2 outputs and can be generalized (e.g., quaternion, octanion) for more dimensions.

### Questions
The major questions on the paper are related to the points above, in particular:
1. Provide a clear definition of the activation function, and add discussions on its design, including the choice of the cone and the computational complexity of the projection operation.
2. Show at least one use case where the AF provide significant improvements, either in accuracy or time.
3. Add more ablations and baselines, especially of AFs which are closely linked to the paper.

I also have a few minor additional questions:
1. In the citations there are multiple references to "MS Windows NT kernel description"?
2. P4: "the set S is a certain polyhedron", can you clarify what polyhedron or point to a specific definition in the paper.
3. I would advise to add a definition for the cone C_n.

EDIT AFTER REBUTTAL: point (1) was mostly solved. Point (2) is still standing. Point (3) was partially solved. Most minor questions were solved. I have increased my score from 6 to 8 because I believe this is an interesting research direction. However, experimental results (at the moment) are unconvincing.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper expands the SISO activation functions in neural networks by introducing the MPU as a MIMO activation function. This expansion is motivated by the structural resemblance between a shallow FNN and a single iteration of the PGD algorithm. Experiments test the effectiveness of the proposed approach.

### Strengths
1.	The idea that choose the activation function to be the projection onto the convex cone is very interesting.
2.	The paper provides some theoretical proofs.
3.	Some experiments demonstrate the effectiveness of the proposed MIMO activation function.

### Weaknesses
1.	The organization and writing of the paper need to be improved.
2.	The proposed Theorem 1 is not so rigorous.
3.	The experiments are not enough.

### Questions
1.	The projected gradient descent (PGD) algorithm in proposition 1 has many iterative steps. However, the Theorem1 has a single layer, which corresponds a single iteration of the PGD algorithm. The structural connection is weak.
2.	In Theorem 1, it seems that it needs to remove W^{(2)}=I, b^{(2)}=0. We only need  W^{(1)}, b^{(1)}, and ReLU activation function to represent a single step of PGD algorithm.
3.	How to set \alpha in Theorem 1 and Proposition 2 in the network training? It is especially to the convolutional network. It is better to give some discussion on the setting of \alpha.
4.	The paper only gives the derivations with fully-connected feedforward networks. It does not present the skip-connection. Since the experiments use ResNets, it is better to discuss this issue.
5.	The paper only tests the proposed MPU with ResNet18. It is not a very deep network.  Is the proposed MPU unable to train deep networks? It is best to give an experiment to train deep networks, such as resnet101? This is very critical, which determines whether the proposed MPU can be applied in practice. 
6.	The organization of section 2 needs to be improved. It is hard to follow.
7.	When referencing equations, it is better to use “\eqref”.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Most activations functions employed in DNNs are Single Input Single Output (SISO). One can naturally ask whether we can gain by using a Multiple Input Multiple Output (MIMO) activation function instead. Of course, the answer depends on how we design such MIMO activations. This paper explores a specific strategy for designing MIMO activations: first based on projections, and more generally based on proximal functions. 

The motivation the authors give for this two types of MIMO activations is that ReLU (one of the most widely used SISO activations) is a projection, and that a shallow network can be parameterized so to mimic a single iteration of Projected Gradient Descent. This is the high level observation that motivates their definition. To further motivate it mathematically, the authors prove a couple of theorems showing that ReLU can be, in a sense, mimicked by their activation, while the other direction does not.  The authors further explore proximal MIMO functions, by using proximal operator, and connecting to proximal gradient descent. They motivate it as a MIMO extension of leaky activation functions.

Finally, the authors show experiments that validate their theory and the fact that somewhat better empirical results can be obtained with their methods.

### Strengths
Overall, I liked the paper. It was very fun reading. In particular, the following are the main strengths: 

- Very well written paper.
- Clear motivation in what they want to achieve (MIMO activation) and in how they achieve it (based on projections/proximal functions).
- Supporting theory.
- Empirical results suggesting the benefits of their approach.

### Weaknesses
However, there are some very substantial weaknesses:

- *[Removed due to answers by authors:]* Theory is basic and simple. But more importantly: it does not really, at the essence, establish why they function should give better results. Sure, you can set the weights just correctly so that a layer is like an iteration. But, so what? The weights are actually learned, and our goal is not to really learn an iteration...

- *[Removed due to answers by authors:]* I am a bit skeptical about the motivation. The way I see it, the whole idea in the activation is to introduce *some* nonlinearity. It does not have to be a lot, to really get a low of expressive power! Just by doing many layers, we can take a really small amount of non-linearity and make it a lot. So, not sure that MIMO really will help...

- *[Added due to answers by authors:]* Limited potential improvements: the authors observed that only using very small input-output size (namely, m=2, which replaces one input one output with two input two output) is the best you can do. Going to m=3 and above does not help. This suggest that there is very little to gain from MIMO non-linearity.

- Empirical improvement is very small, and even then not always achieved.

### Questions
- How are MIMO activations implemented? *[Following rebuttal:]* Still unclear on how exactly the projection itself is implemented.
- Why use small m in the experiments? *[Following rebuttal:]* OK, understood. However, this raises the question on why MPU is a good idea if only very small m is used. See new weakness above.
- Table 1: improvements in training error should be compared with cost (MACS here). With MPU you get slighly better errors with slightly more MACS. What happens with ReLU if you increase the MACS a bit, e.g. by doing a few more epochs? 
Sure, you test ResNet 34 and 50  in which MACS increase. But there is more than one way to increase the MACS...
*[Following rebuttal:]* OK, doing more epochs might make sense less, but there are other ways. e.g. slightly wider network?
-*[New following rebuttal:]*  The authors use the same hyperparameters across all experiments, with the sole variation being the Activation Function (AF)." Why? And, more importantly, how the results will look if all algorithms use their best hyperparameters?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
