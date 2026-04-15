# Semi-Supervised Learning of Tree-Based Models Using Uncertain Interpretation of Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 3

## Abstract
Semi-supervised learning (SSL) learns an estimator from labeled and unlabeled data. While diverse methods based on various assumptions have been developed for parametric models, SSL for tree-based models is largely limited to variants of self-training, for which decision trees are not well-suited. We introduce an intrinsic semi-supervised learning algorithm that achieves state-of-the-art performance for tree-based models. The algorithm first grows a tree to minimize a semi-supervised notion of impurity, then assigns leaf values using a leaf similarity graph to optimize either for smoothness or adversarial robustness of the estimator near the data. Our methods can be viewed as natural extensions of conventional tree induction methods emerging from an uncertain interpretation of model input, or alternatively as inductive tree-based approximations of well-established graph-based SSL algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a semi-supervised learning (SSL) algorithm for tree-based models. The proposed algorithm employs an intrinsic SSL approach that optimizes the estimator by leveraging the inherent structure of tree-based models. Specifically, it constructs a leaf similarity graph to capture similarities between leaf nodes of the tree and uses this graph to propagate label information from labeled to unlabeled data.

### Strengths
The paper is readable and well-written. The insight that the smoothness-based SSL approach can be seen as an approximation of label propagation using a kernel derived from the uncertain input interpretation is interesting. It provides a higher level of clarity regarding how the proposed method approximates label propagation using trees.

### Weaknesses
The proposed method combines various techniques, and it's unclear how beneficial each individual contribution is. While incorporating various strategies can be practically beneficial, I believe that in academic papers, there is a need for micro-level analyses of each component. 

Furthermore, since the algorithm is built with assumptions about the data, it performs well on data that aligns with those assumptions and poorly on data that does not. While this is evident, and the performance was often better on the datasets used in this experiment, there remain questions about the significance of this study. I believe that if there were advantages such as computational cost or support from theoretical superiority, they could serve as strong points beyond performance. 

Only prediction accuracy has been measured, and there has been no comparison in terms of computation time or memory usage. Although the order has been evaluated, a comparison with other methods is necessary. 
Upon reviewing the Appendix, it appears that the proposed method conducts parameter tuning and other fine-grained optimizations with great precision compared to other methods. I'm curious about the time required for these parameter search processes in comparison to other methods. It would be helpful to understand the impact of both the algorithm itself and the thorough parameter tuning on the results.

(Minor comment: The abbreviation 'KDDT' is used in the first paragraph before its definition is provided in the third paragraph of Section 3.)

### Questions
(See weakness part)

1: Could you provide ablation study results? In addition, there are various degrees of freedom when conducting experiments with the proposed method, such as the design of kernels in KDDT or the design of histograms when considering piecewise constant kernels. How would the experimental results change with variations in these designs?

2: Could you report the cost required for parameter search processes in comparison to other methods?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses the problem of training decision trees in a semi-supervised setting, i.e. only a fraction of the data are associated with labels. The authors root their proposal in label propagation, i.e. the classifier being trained is used to replace the missing labels. Two strategies for label propagation are considered: the first one (smooth leaf value assignment) amounts to replace the missing labels with the probabilistic predictions of the tree, while the second one makes use of randomized smoothing and seemingly consists in maximizing consistency between the missing label replacements and available labels. 

The paper features discussions on how the tree can be regularized, on how the SSL tree algorithm can be plugged into a random forest, and on the computational complexity of the approach. The proposals are then succinctly evaluated and compared on several datasets of the literature, before a conclusion is made.

### Strengths
The paper addresses an interesting problem, since training a tree in a semi-supervised setting remains open. 

The proposals are technically good, i.e. the math are correct.

### Weaknesses
The paper is not very well written. It lacks a clear presentation of the existing works upon which it builds (and/or a clear distinction between these works and the proposal); the proposals lack formalization; some mathematical objects are not properly introduced or described. Overall, it seems to lack structure. 

The paper focuses on the technical side rather than on the interpretation. As such, it well addresses how to solve the optimization problems presented (as I already stated above), but not (or not always) why these problems in particular were derived. 

Last, I have a mixed feeling towards the experiments. Indeed, the paper addresses training trees in a semi-supervised setting, the aim of which is to learn a classifier able to generalize well on new data. However, the accuracy of the model is evaluated on the unlabeled training data, which were used to train the model. This seems highly criticizable to me.

### Questions
The writing and the presentation of the paper should be improved. Some acronyms are not defined (e.g., KDDT), some mathematical objects are not properly introduced (e.g., \bf{y}). There are a few typos or clumsy expressions ("there is often copious data available", "we interpret [...] as random variable"), and several sentences are hard to understand ("By maximizing this quantity over training data $\sum_i p_{i,\text{max}}$, we maximize a robustness objective [...]", or "smaller tree models are more tree-like, while larger tree models are more kernel-like"). Some references (e.g., the first and the last) are incomplete. 

The description of the existing works is quite succinct, and may have been made more precise—and more formal. As well, some parts of the proposal (such as, notably, Section 3.3) lacks formalization, which does not help understanding nor fully appreciating its interest. 

Some notations are not appropriate (such as, e.g., the probability density attached to an instance, i.e. $f(\cdot,\boldsymbol{x})$)—note that this interpretation of an instance as the realization of a continuous random variable $\bf{x}$ may have been discussed further. It would have been nice to better present the matrices at the end of Section 3.2 (right before Sec. 3.2.1). 

Lemma 1 is not clear; it should be readable on its own. The first propagation strategy being equivalent to label propagation as per Zhu and Ghahramani, what is the added value ? 

To strategies were proposed to "identify" missing labels: how do they compare to each other ? An experimental comparison, if not a theoretical study, would have been interesting. 

There seems to be a major problem in the evaluation of the method. In the experiments, it seems that accuracy is computed over the unlabeled training data, not on some test data. This does not seem rigorous, as the actual generalization capacity is not evaluated, and the estimate thus obtained is biased since these data are used in training.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a method for learning trees in semi-supervised learning setting. This is done in two steps: 1) greedily grow a tree with modified gini splitting criterion; 2) use similarity-based graph to assign labels for each leaf. The authors propose two approaches for the second step: 1) smooth label assignment that relies on solving a linear system; 2) an assignment based on graph min-cut. 

The method is applicable to both classification and regression trees, and easily extendable to ensembles. Experiments demonstrate superior performance compared to baselines.

### Strengths
- the paper is well-motivated. In the era of data-driven decision making, we observe extensively increasing data with limited supervision. Thus, the paper considers a practical setup.
- the method is applicable to both classification and regression trees, and easily extendable to ensembles. 
- Experiments demonstrate superior performance compared to baselines.

### Weaknesses
Major comments:
- **Novelty**. An important and pertinent paper [1] appears to be absent from the list of references. It is worth noting that the tree growing procedure outlined in this work closely follows the methodology established by Levatic et al. However, the principal innovation lies in the leaf assignment process. The first (out of 2) technique involves solving a linear system on a similarity matrix, akin to the principles of graph Laplacian-based methods used in [1]. The authors demonstrate its equivalence to label propagation techniques. Notably, [1] explicitly builds upon this methodology, and it is noteworthy that one iteration in [1] bears a striking resemblance to the first approach elucidated in this paper. This correlation underlines the significance of [1] in the context of the current study.
- **Experiments**. As noted above, [1] is highly related to the proposed approach and should be compared as a baselines. Moreover, used benchmarks a somewhat limited in their sizes and do not resemble real-world semi-supervise scenario. 
- **Scalability**. [1] employs a sparse graph representation, necessitating the utilization of sparse linear algebra techniques for solving the linear system. However, it is important to acknowledge that in this particular context, scalability could potentially pose a challenge, primarily attributable to the dense representation of matrix A. This consideration prompts a thoughtful examination of the trade-offs associated with the chosen approach, especially in cases where computational resources may be constrained.


[1] Zharmagambetov, A. and Carreira-Perpinan, M. A. (2022): "Semi-supervised learning with decision trees: Graph Laplacian tree alternating optimization". Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 2392-2405.

### Questions
- The authors mention that the second approach is more robust against adversarial attacks. Just wondering whether it was demonstrated experimentally as I was unable to find?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The submission presents two methods for semi-supervised learning of decision trees, based on the tree construction method of predictive clustering trees, which have been applied to semi-supervised learning in earlier work, and the inference method from kernel density decision trees. The proposed two approaches apply graph-based semi-supervised learning strategies, label propagation and min-cut, and enable semi-supervised learning to propagate class information across similar leaves. Experiments evaluate the two methods for varying amounts of labeled training data on 14 datasets, for both stand-alone trees and random forests, and compare to supervised decision trees and random forests, self-trained random forests, and graph-based label propagation with an RBF kernel. The results are quite mixed when comparing the proposed methods to the purely supervised competitors, and there is no clear winner.

### Strengths
The submission is nicely written.

There is a lack of work on intrinsic methods for semi-supervised learning of decision trees, which is addressed by this work.

Theoretical results are proven for one of the two methods that are presented.

### Weaknesses
The experimental results do not show a consistent advantage compared to simple supervised baseline decision trees and random forests respectively. Moreover, this is in spite of the extensive hyperparameter tuning using cross-validation for the proposed methods and the absence of kernel smoothing in the decision tree baselines. The proposed approaches have two hyperparameters that are tuned using cross-validation. 

There is a lack of an ablation study where the proposed method is run in a purely supervised mode (for both stand-alone decision trees and random forests). 

There is no comparison to basic predictive clustering trees for semi-supervised learning.

No significance tests are performed and no confidence intervals are provided.

The proposed methods are advertised as inductive ones, but the evaluation appears to be transductive.

Cross-validation is applied for parameter tuning based on a tree structure learned from the entire dataset. This will yield overfitting. Computational complexity arguments are not very strong because the cross-validation can be trivially parallelized.

The submission references several papers that attempt to improve self-training in trees (e.g., Lie et al., 2020), but the experiments only include the basic variant of self-trained decision trees based on what is in scikit-learn.

Typos, etc.:

"a kind of automatic dimension reduction and a kernel-like representation" -- unclear

"domain, So"

### Questions
How many trees are used in the random forests of the proposed methods and how are hyperparameters selected for the trees in the random forests?

What is the difference between kernel density decision trees and the work presented in

@inproceedings{Geurts2005,
author = {Pierre Geurts and Lous Wehenkel},
booktitle = {Proceedings of the 22nd International Conference on Machine Learning},
pages = {233-240},
publisher = {ACM},
title = {Closed-form dual perturb and combine for tree-based models},
year = {2005}}

and why is this method of Geurts and Wehenkel not suitable?

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair
