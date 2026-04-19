# SFESS: Score Function Estimators for $k$-Subset Sampling

- Decision: Accept (Poster)
- Scores: 6, 5, 5

## Abstract
Are score function estimators a viable approach to learning with $k$-subset sampling? Sampling $k$-subsets is a fundamental operation that is not amenable to differentiable parametrization, impeding gradient-based optimization. Previous work has favored approximate pathwise gradients or relaxed sampling, dismissing score function estimators because of their high variance. Inspired by the success of score function estimators in variational inference and reinforcement learning, we revisit them for $k$-subset sampling. We demonstrate how to efficiently compute the distribution's score function using a discrete Fourier transform and reduce the estimator's variance with control variates. The resulting estimator provides both $k$-hot samples and unbiased gradient estimates while being applicable to non-differentiable downstream models, unlike existing methods. We validate our approach experimentally and find that it produces results comparable to those of recent state-of-the-art pathwise gradient estimators across a range of tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the k-subset sampling and proposes a differentiable estimator that enjoys the benefits of score function estimators and the relaxed approperiate pathwise estimator. The proposed method extends the existing differentiable optimization of Bernoulli and categorical distributions and serves as a complement to existing methods.

### Strengths
I am not very familiar with the topic but I believe this work has made significant contributions. In traditional k-subset sampling methods, the score function estimators are rarely considered. This paper fills this gap by proposing the proposed method. The results are validated on diverse experiments and match the SOTA relaxed and approximate pathwise gradient methods. Therefore, the method is original and significantly improves the existing approaches.

### Weaknesses
The backgrounds and introduction to the studied topic are not very friendly. It is hard for me to understand why the k-subset sampling is needed and why the existing methods (which may not be based on the score function and potentially non-differentiable) are not sufficient to the related community. I hope the author could add more examples, explicitly stating where the k-subset sampling is used and why the existing methods fail to satisfy the demond.

### Questions
In Table 1: Method Comparison, the author lists four properties (Exact Samples, ...). Can the author explain why these properties are important or preferred to have? What will happen, for example, if the estimator is non-differentiable?  

When we need to have the gradient of some methods such as GS, STGS, or SIMPLE, can I simply use a gradient estimation approach to approximate their gradient? Does it make any difference from this method? 

A minor issue:

I probably disagree that the variance reduction using some control variates can be considered as any contribution, since it has been well-known in the literature. However, if the VR is not employed in the SFESS method, its performance in Figure 5, is nearly the worst among all methods. How should I understand this figure?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a method for computing the score function of the k-subset distribution using a discrete Fourier transform and employs control variates to address the high variance of the vanilla score function estimator.

### Strengths
SFESS broadens the potential applications of k-subset sampling, particularly in scenarios where the downstream model’s gradient is unavailable. The algorithm’s effectiveness is demonstrated through several experiments.

### Weaknesses
This paper appears to fall slightly below the standard expected at ICLR for several reasons:

1. A primary concern is the lack of a solid theoretical foundation; the paper focuses mainly on numerical results without a deeper mathematical analysis. For example, there is no mathematical description or convergence rate guarantee for Algorithm 2.

2. In terms of experiments, the explanations of the architecture design and choice of hyperparameters require more clarity and justification. I will outline these concerns in greater detail in the questions below.

3. Additionally, the paper would benefit from further polishing. For instance, on line 204, p_{theta}(b) is introduced without prior definition. On line 206, the phrase should be “Naively.” Also, on line 221, the operation ArgTopK(log(theta+g), k) would benefit from further explanation.

### Questions
1. Choice of Hyperparameter k: How do you select the hyperparameter k for different tasks? How does performance vary if k is adjusted? A deeper discussion on the choice of k would be beneficial.

2. Selection of Random Seeds: How did you choose your random seeds? It would be helpful to list the specific seeds used to facilitate reproducibility and to confirm that your choices were not made deliberately.

3. Performance Comparisons in Tasks: In the VAE task, it appears that SIMPLE outperforms SFESS+VR (Table 3), while for the kNN task, GS and STGS perform better (Table 4). What is the intuition behind these results, and what advantages does SFESS+VR offer in these contexts?

I would be willing to increase the score if the questions above are thoroughly addressed and if more in-depth theoretical results are provided.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a black-box gradient estimator for $k$-subset sampling using a score function estimator. 
In this approach, the discrete Fourier transform (DFT) is employed to compute the density function of the Poisson binomial distribution, and control variates is used to reduce variance.
The method is evaluated through multiple experiments, where it is compared against existing approaches.

### Strengths
1. The use of a score function to address the gradient estimation problem is straightforward yet effective. To compute the probability density of the Poisson binomial distribution, the paper leverages a closed-form expression based on the discrete Fourier transform (DFT). A variance reduction technique is then applied to enhance estimation efficiency.
2. The paper is well-written, providing a clear overview of the background, challenges, and gaps in the field, along with a detailed summary of $k$-subset sampling methods.
3. The proposed method is validated on both synthetic and real-world dataset.

### Weaknesses
1. While the paper proposes a score function estimator for $k$-subset sampling, Equation (4) appears to be a well-established technique, commonly known as the score function estimator (SFE) [1]. 
This suggests that the primary contribution may lie in the application of variance reduction within the Monte Carlo approximation. 
My main concern is that the novelty of this work may be insufficient for acceptance at a top-tier conference main track.

2. SFESS does not appear to demonstrate a consistently superior performance in the experimental results compared to other methods (e.g., SIMPLE). Could you clarify the specific advantages of SFESS and provide more insight into scenarios where it outperforms alternative approaches?

[1]. Kareem Ahmed, Zhe Zeng, Mathias Niepert, and Guy Van den Broeck. SIMPLE: A Gradient Estimator for k-Subset Sampling. In International Conference on Learning Representations, 2023.

### Questions
Please see Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2
