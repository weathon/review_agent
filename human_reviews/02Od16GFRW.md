# Ensembles provably learn equivariance through data augmentation

- Decision: Reject
- Scores: 6, 6, 3

## Abstract
Recently, it was proved that group equivariance emerges in ensembles of neural networks as the result of full augmentation in the limit of infinitely wide neural networks (neural tangent kernel limit). In this paper, we extend this result significantly. We provide a proof that this emergence does not depend on the neural tangent kernel limit at all. We also consider stochastic settings, and furthermore general architectures. For the latter, we provide a simple sufficient condition on the relation between the architecture and the action of the group for our results to hold. We validate our findings through simple numeric experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a theoretical analysis showing that data augmentation can lead to equivariance in deep ensembles. The paper's main result is that under several assumptions (e.g. on initialization, architecture, etc.), deep ensembles trained with data augmentation are equivariant in mean, even when individual models are generally not. A similar result was previously presented, but the paper extends these previous results, which were primarily focused on infinitely wide NNs trained with gradient descent under full augmentation, to ensembles of finite-width trained with SGD and random augmentation.
The paper is mainly theoretical and validates the theoretical results through limited and small-scale empirical experiments.

### Strengths
1. The paper is well-structured and easy to follow.
1. The paper extends previous results to more reasonable and applicable settings. This is a significant extension.

### Weaknesses
I like the paper and believe it has a sufficient contribution and interesting results. However, there are several limitations stated below:

1. While the assumptions for the theoretical analysis are more applicable compared to previous works, they still hold only for infinite-size ensembles. Any analysis (including empirical) on the error bounds for finite ensembles would be beneficial.
1. While the results are important, the novelty is somewhat moderate in the sense that the emergent equivariance property of ensembles was previously proposed and the fact that the theoretical analysis heavily relies on previous works [1].
1. From the empirical evidence, it is unclear if some of the assumptions (like symmetric initialization) are indeed necessary. The authors discuss this, but I believe it can be extended further.
1. Empirical evaluation is limited. It would be beneficial to extend it to more settings, even by small modifications like considering cyclic groups C_k of different orders (k), different architectures, model sizes, etc.
1. It would be beneficial to see the impact of ensemble size on the metrics in Table 1, like adding a line plot for ensemble size vs. OSP. The authors show results for different sizes, but summarizing them in one clear view would make it easier to follow.
1. The paper could benefit from a clearer and more explicit discussion of the limitations of the results.
1. Minor:
    - Line 37: “... a definitive question to the question…”.

Reference

[1] Flinth & Ohlsson, Optimization Dynamics of Equivariant and Augmented Neural Networks, 2023.

### Questions
1. Why does the OSP not increase at initialization when ensemble size increases?
1. From the figures, it seems like the results could improve with more epochs (also for baselines). Could you please provide results with a larger number of epochs?

### Soundness
3

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
The paper expands the results of Gerken & Kessel that show that data augmentation produces equivariant ensembles of models using NTK, by looking at finite network sizes. They then show empirically that their theoretical results indeed hold in practice (up to sampling errors).

### Strengths
- It generalizes the results in Gerken & Kessel 
- The topic of invariance/equivariance is important so these results would be of interest to people in that community

### Weaknesses
My main issue is with the writing: 
- The results presented in the main text are quite trivial, that if you start with an invariant distribution and use an invariant flow you end up with an invariant distribution. The more interesting results are in the appendix (appendix B and C)
- You writing $\mathcal{L} = A_\mathcal{L} + T\mathcal{L}$ with $T\mathcal{L}$ the tangent space is very confusing, as tangent space is defined for a manifold and we are talking about a linear space. It needlessly complicates things as there is no need to involve differential geometry when we are working on linear spaces.

### Questions
The results in Table 1 aren't that clear to me. In the asymmetric case where you have a symmetric initialization, shouldn't you get results that are similar to the symmetric case? Yet there is a large gap

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper shows that an ensemble of models when trained with data augmentation leads to emergence of equivariance properties naturally. The results generalize over past known results based on NTKs. The theory assumes some basic assumptions on the architecture and shows that, when the initialization of the weights in an architecture has some symmetry, then, the expected architecture of the ensemble is equivariant. Experimental results with various ensembles validates the results for the C4 group of symmetries.

### Strengths
- The work show the emergence of equivariant in ensemble models
- The work generalizes previous works where the proof relied on NTKs
- Experiments with large ensemble of models show the emergence of equivariance

### Weaknesses
I have several concerns over the usefulness of the theory and the experimental results.

Usefulness of theory:
- What is the use of the theory in model design or practical use cases? Since equivariant models seems to give perfect equivariance and data augmentation techniques give approximate equivariance. So, I am wondering what is the use of ensemble technique for symmetries, especially, given that we need over 1000 models to get good equivariant results.
- What are the advantages of the proposed technique compared to existing symmetrization and canonicalization methods [1-4] that can convert non-equivariant models into equivariant ones using techniques somewhat similar to ensemble methods but with additional transformations that looks similar to augmentation.

Experimental Results:
- Although the experimental does show that the architecture with symmetric support does give invariant output, but even the asymmetric architecture seems to be giving invariant output, questioning the usefulness of the theory. It is also discussed in the paper about the symmetric states being attractors potentially, but, it still makes the current theory not very useful.
- Experiments are only shown for C4 symmetries

[1] Basu, Sourya, et al. "Equi-tuning: Group equivariant fine-tuning of pretrained models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 6. 2023.

[2] Mondal, Arnab Kumar, et al. "Equivariant adaptation of large pretrained models." Advances in Neural Information Processing Systems 36 (2023): 50293-50309.

[3] Basu, Sourya, et al. "Efficient equivariant transfer learning from pretrained models." Advances in Neural Information Processing Systems 36 (2024).

[4] Kaba, Sékou-Oumar, et al. "Equivariance with learned canonicalization functions." International Conference on Machine Learning. PMLR, 2023.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
