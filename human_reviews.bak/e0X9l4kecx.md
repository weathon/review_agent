# Convex Formulations for Training Two-Layer ReLU Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Solving non-convex, NP-hard optimization problems is crucial for training machine learning models, including neural networks. However, non-convexity often leads to black-box machine learning models with unclear inner workings. While convex formulations have been used for verifying neural network robustness, their application to training neural networks remains less explored. In response to this challenge, we reformulate the problem of training infinite-width two-layer ReLU networks as a convex completely positive program in a finite-dimensional (lifted) space. Despite the convexity, solving this problem remains NP-hard due to the complete positivity constraint. To overcome this challenge, we introduce a semidefinite relaxation that can be solved in polynomial time. We then experimentally evaluate the tightness of this relaxation, demonstrating its competitive performance in test accuracy across a range of classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies 2-layer ReLU neural networks trained with MSE loss and $l_2$ regularization. The main contribution of the paper is in showing that beyond certain critical width (of at most quadratic dependence on the number of samples), the standard optimization problem is equivalent to a convex problem over a matrix of quadratic number of parameters in number of samples and input/output dimensions. The optimal values of this problem match.

The equivalence further applies to infinite-width 2-layer ReLU networks with analogical loss function. The optimal solutions of the convex problem can be used to construct optimal solutions of the infinite-width problem. 

Finally, the paper proposes a computationally efficient algorithm that approximates the convex, yet copositive program. This new, semidefinite program is solvable in polynomial time and the solution of the original problem can be recovered from the solution of the SDP problem by a rounding scheme described in the paper, using existing approaches such as the three-operator splitting method. 

The authors then evaluate this method as well as the tightness of the SDP relaxation on some tabular datasets of small size, demonstrating (within these datasets) that the SDP relaxation is rather tight and the practical optimization algorithm works comparably to known optimization methods, despite using number of parameters that is not theoretically guaranteed to work.

### Strengths
- S1: The strongest part of the paper is, in my opinion, the theoretical contribution (Theorem 1). To the best of my knowledge, this convex formulation significantly improves the number of parameters (as a function of the number of samples and input/output dimension) in the convex program from high-degree polynomial [1], to quadratic. The price the authors pay is the minimal width required to achieve this equivalence, which is also quadratic in the number of samples, as opposed to linear in previous works. Nevertheless, this neat and parameter-light reformulation is of independent theoretical interest and I believe it can be used as a tool in theoretical analyses of 2-layer NNs outside the scope of this paper. I checked the proof of the result and everything seems to be all right with only a small issue that I will discuss in typos. 

- S2: The paper is overall well written, clear and interesting. 

[1] Sahiner, Arda, et al. "Vector-output relu neural network problems are copositive programs: Convex analysis of two layer networks and polynomial-time algorithms." arXiv preprint arXiv:2012.13329 (2020).

### Weaknesses
- W1: The practical contribution of this paper is limited. Even if we ignored the fact that the practical algorithm introduced in this paper is only suitable to train 2-layer neural networks, the main issue remains -- there already are algorithms in the literature that can obtain approximate solutions of the convex re-formulations using quadratic time per-iteration in the number of samples (see for instance [1]). Moreover, the experimental evaluation is not very convincing. The tested datasets are very small so the results are not very convincing. It is known (both theoretically and empirically), that gradient-based methods work well also with widths linear in the number of samples (see for instance [2], which considers only marginally different setting). Whether your method works well with R constant (the linear dependence on $n$ already comes from lifting into the higher-dimensional space) or at least sub-quadratic in $n$ is questionable (and not addressed in the paper). Moreover, the performance is comparable with already existing and well-established optimization methods (as demonstrated in the paper). 

- W2: While it is written in the paper that the algorithm scales quadratically with the number of samples (which is true in the plain CP-NN formulation), recovering a solution to the original problem actually requires cubic number of parameters and thus also time. This can be seen in equation (3), where it is clear that already the dimension of the $\lambda_j$ vectors scales with $n$ and $R$ scales quadratically with $n$, thus in total the dependence is cubic. Even if an oracle gave us *directly* a solution to CP-NN problem, recovering a solution to the original problem would still require $n^3$ number of parameters. 

- W3 (minor): The computational complexities of all the involved procedures should be explicitly discussed in the paper.

**Small issues:**

- Lemma 3 is wrong, the $w$ can have positive values where $\hat W$ has negative values. A correct version of this lemma removes $\lambda$ completely from the statement and replaces the $\lambda$ in the if and only if statement with $w$. This would also be consistent with the later use of this lemma in the actual proof of Theorem 1. 

- Typo: line 176 you refer to Lemma 4 but should be Lemma 2. 
- Typo: In line 167 index should be also added to the $u, v$ variables. 
- Typo: across the entire proof of theorem 1 including the lemmas that precede it, you use $W$ and $\Lambda$ supposedly to denote the same object. Please only use one of these (preferably $\Lambda$).

[1] Bai, Yatong, Tanmay Gautam, and Somayeh Sojoudi. "Efficient global optimization of two-layer relu networks: Quadratic-time algorithms and adversarial training." SIAM Journal on Mathematics of Data Science 5.2 (2023): 446-474.

[2] Bombari, Simone, Mohammad Hossein Amani, and Marco Mondelli. "Memorization and optimization in deep neural networks with minimum over-parameterization." Advances in Neural Information Processing Systems 35 (2022): 7628-7640.

### Questions
- Q1: Do you expect Theorem 1 to be applicable in theoretical analyses of 2-layer NNs in different contexts? 

- Q2:  Assume an oracle would give us a solution to the CP-NN problem directly. What is the computational complexity of an algorithm that would recover a solution to the NN-TRAIN problem? 

- Q3: Regarding Theorem 2, are there solutions to the NN-integral-train that are not discrete? 

- Q4: What would be the training loss of the recovered NN-TRAIN solution in Table 2? I am asking to see how far off would the projection back into feasible set be from the actual solution to NN-TRAIN. 

- Q5: What makes the bank notes dataset so different from the others that your method is so significantly worse than all the others, while being pretty good on all the other datasets? 

**Summary:** Despite several flaws outlined above, I still think this paper meets acceptance threshold due to the novel, compact and parameter-light convex formulation of the 2-layer NN training which does not rely on enumerating all the activation patterns of the intermediate layer (as in the previous literature). This result is of theoretical interest and, though simple in nature, it is novel and insightful. Therefore, I recommend acceptance. However, I condition this opinion as well as the score on the satisfying discussion period.

**After the rebuttal:** The authors satisfactorily answered my questions and discussed my concerns. I am keeping my original score.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel training method for two-layer neural networks based on a formulation of a convex completely positive program. The authors also propose a semidefinite relaxation which ensures that the problem can be solved in polynomial time. Experiments on several simple synthetic and real datasets are conducted to demonstrate the performance of the proposed method in comparison to the Neural Network Gaussian Process and Neural Tangent Kernel methods.

### Strengths
- The idea of the proposed method based on convex completely positive program and semidefinite relaxation is interesting. 

- The presentation of the paper is fairly clear.

### Weaknesses
- The proposed method is only for training wide two-layer neural networks with ReLU activation function. It seems that extending the method to deeper networks is non-trivial, which limits the practical value of the proposed method. It would be beneficial if the authors could discuss the possibility to implement the proposed methods to train modern deep neural networks.

- As the authors have commented, the problem (Cp-Nn) is NP-hard due to the complete positivity constraint, and as far as I can see, the corresponding (Sdp-Nn) is not guaranteed to find the actual solution to (Cp-Nn). It would be helpful if the authors can add discussions about this point.

- To my knowledge, the Neural Network Gaussian Process and Neural Tangent Kernel methods are also settings (mainly about parameter/initialization scale and network width) under which SGD training of neural networks is almost equivalent to a convex optimization procedure. It is widely agreed that NNGP and NTK cannot fully explain the success of deep learning. These models only characterize scenarios where the obtained neural networks have weights very close to their random initialization. As a result, they fail to explain how deep networks learn effective feature representations. Therefore, even if the proposed method achieves competitive results compared with NNGP and NTK, it does not necessarily mean that the proposed method can compete with actual neural network training beyond the “NTK regime”.

- The experiments are not representative enough. All datasets considered in this paper are very small datasets, and the performance of the proposed method on these datasets are insufficient to demonstrate its advantage. I think the authors should at least include results on datasets such as MNIST and CIFAR-10. 

- The experiment setup requires some clarification. As mentioned above, NNGP and NTK represent specific regimes in which SGD training of neural networks may fall [1,2,3,4]. Therefore, it is not reasonable to directly list “SGD”, “NNGP”, “NTK” as different training methods. Discussion how the SGD setting differs from NNGP and NTK and why they are not equivalent according to [1,2,3,4] is necessary to explain the experiment setting clearly.

- The authors should also discuss the relationship between the proposed method and the study in [5].


[1] Simon Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. ICLR 2019.

[2] Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via over-parameterization. ICML 2019.

[3] Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. ICML 2019.

[4] Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Gradient descent optimizes over-parameterized deep ReLU networks." Machine learning 2020

[5] Cong Fang, Yihong Gu, Weizhong Zhang, and Tong Zhang. Convex formulation of overparameterized deep neural networks. IEEE Transactions on Information Theory 2022

### Questions
I suggest that the authors should address the weaknesses point out above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the training of shallow NN using convex optimization techinques.
It provides a brief background on Semi-Definite programming and Copositive Programming. Then it provides a formulation for the training of a NN as a CP problem (for a sufficiently wide 2-layer net with ReLU activations), which is then relaxed to make it computationally feasible.
The tightness of the relaxation is evaluated empirically.

### Strengths
The paper is clear, and reads well.
The supplementary material provides the code to reproduce the results.
Section 2 provides a complete yet brief background on the relevant optimization topics and concepts.

### Weaknesses
The paper does not consider bias terms in the linear layers.
The tightness of the proposed relaxation is evaluated only empirically.
There is no convergence guarantee for the TOS rounding step.
The rounding step is performed using the critical width, however in practice this is unfeasible.
Time complexity is not considered in the evaluations.
The empirical evaluation is performed on classification tasks, using L2 loss.

### Questions
I have some concerns on the empirical evaluation. MSE is not a suitable loss function for classification problems, a selection of regression tasks would have been more representative. Regardless, the hyperparameters chosen for the SGD baseline are quite strange (especially for PIMA Indians). Finally, the dataset is divided in train-test split, this penalizes models that overfit on the training data. However, if the goal is to evaluate how well a model is solving the given optimization problem, it should be evaluated on the training set.

In most cases, the addition of a bias term in the linear projections significantly improves performance, especially for 2-layer ReLU networks. In fact, running the proposed SGD baselines with the addition of a bias term improves performance across the board (gaining up to +15% accuracy). Given this, the lack of a bias term and its implications should probably be noted in the text.

I don't understand the results in Table 2. The metric is the loss, lower mean better, i.e. SDP-NN significantly improves on SDG. However the way the approximation ratio is computed and is presented in the main text implies that the SDP-NN performs worse than SGD.

Some equation are labelled instead of being numbered. I personally find this beaks the flow of the main text as it is not immediately clear that it is a reference to an equation (as most people are used to see numbers). If you want to refer to some equation by a meaningful name, you can use the name in the definition, i.e. instead of "... in (CP-NN)" consider using something like "... in the CP-formulation (equation 2)".
If you decide to keep the labels, at least try to use them to refer to the equations only, not to the concepts they are tied to. For instance in "While the case of countably infinite number of hidden neurons – represented by an infinite sum in (NN) – is captured as a special case of (NN∫) with a discrete probability measure, the (NN∫) model ..." you use (NN∫) to denote both the infinite width model concept as a whole, and its equation.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
In this paper, the authors propose a novel model for infinite-width neural network training based on copositive programming. In theorems 1 and 2, the authors carefully demonstrate that the training of an RELU network of sufficient width using the MSE loss is equivalent to a certain convex copositive program. Next, the authors derive a semidefinite relaxation of the proposed copositive program and an associated rounding scheme to recover weights from the solution to the SDP. On a variety of synthetic and small real datasets, the authors demonstrate that their method recovers neural network weights that achieve smaller MSE (table 2) compared to networks trained via SGD and competitive predictive performance compared to NTK-based kernel methods.

### Strengths
This is an interesting paper that derives an equivalence between infinite-width RELU network training and solving a certain convex copositive program. This work contributes to the growing literature relating neural network training and convex optimization. The empirical results are also promising. Overall, I think this is an interesting work that provides valuable insight.

### Weaknesses
The empirical results seem somewhat weak to me. The authors acknowledge the similarity of their work to earlier work relating copositive programming and RELU network training. Although the existing methods make additional assumptions on the data distribution, a numerical comparison to prior work (e.g. the approximation ratio) would be beneficial in understanding how the proposed framework compares to earlier work in practice or further discussion on the applications of their technique.

### Questions
- can the authors comment on the runtime of solving the SDP?
- can the authors provide additional discussion about what kinds of insights can be gained by applying the proposed framework?

### Soundness
3

### Presentation
3

### Contribution
3
