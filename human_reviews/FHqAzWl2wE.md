# Multimarginal Generative Modeling with Stochastic Interpolants

- Decision: Accept (poster)
- Scores: 5, 5, 8

## Abstract
Given a set of $K$ probability densities, we consider the multimarginal generative modeling problem of learning a joint distribution that recovers these densities as marginals. The structure of this joint distribution should identify multi-way correspondences among the prescribed marginals. We formalize an approach to this task within a generalization of the stochastic interpolant framework, leading to efficient learning algorithms built upon dynamical transport of measure. Our generative models are defined by velocity and score fields that can be characterized as the minimizers of simple quadratic objectives, and they are defined on a simplex that generalizes the time variable in the usual dynamical transport framework. The resulting transport on the simplex is influenced by all marginals, and we show that multi-way correspondences can be extracted. The identification of such correspondences has applications to style transfer, algorithmic fairness, and data decorruption.  In addition, the multimarginal perspective enables an efficient algorithm for optimizing the dynamical transport cost in the ordinary two-marginal setting. We demonstrate these capacities with several numerical examples.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Multimarginal problems are typically studied in the context of optimal transport, by contrast, the authors studied the multimarginal problem using stochastic interpolant with a high-dimensional α in a simplex. The multimarginal framework allows us to (K, 2) marginal transport problems using only K marginal vector fields. Interesting experiments on all-to-all image translation are studied.

### Strengths
1. by lifting alpha to high-dimensional vector, the work enables to solve (K,2) marginal transport problems using only K marginal vector fields.

2. the overall presentation is clear and the empirical experiments are comprehensive.

### Weaknesses
Optimizing the vector field on simplex alpha may require delicate parameterizions. It would be better if it is more connected optimal transport.

The theories look like a na\"ive extension from the work of "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions". The authors can kindly point out some innovations if I may miss some key points.

Section 2.2 may need some proper rewriting to be more clear.

### Questions
Table.1, what does the smoothing mean in "two-marginal with smoothing" ?


Minor:

typo? "color from a sample from a sample from one marginal"

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
The paper introduces a vector-field-based generative model (incorporating ODE/SDE mechanisms) capable of generating samples from multiple marginal distributions. The methodology pivots around a generalized form of stochastic interpolant, specifically a barycentric stochastic interpolant. The stochastic process, in this context, is formulated as a weighted mix of samples from distinct marginal distributions. These weights, restricted to the simplex, allow the model to generate a weighted combination, defined as $x(\alpha)=\sum_{k=0}^K x_k \alpha_k$, where $\alpha=\left(\alpha_0, \ldots, \alpha_K\right) \in \Delta^K$. The corresponding probability flows are derived for this multi-weight interpolant path. To identify the optimal path with the lowest transport cost, the model is trained to minimize the Wasserstein-2 metric, considering the path's velocities. Consequently, this learned model allows transitions between various marginal distributions following the optimized path. However, the semantic interpretation of these paths, especially between distributions like images from multiple modes, remains ambiguous. In the experiments,

### Strengths
- *Innovative Generative Model*: The paper's proposition of a vector-field-based generative model that utilizes a generalized form of stochastic interpolant to generate from multiple marginals is unique and intriguing.
- *Optimal Path Identification*: By minimizing the Wasserstein-2 metric, the methodology seeks the most efficient path for transitions, with theoretical groundings.
- *Barycentric Stochastic Interpolant*: The use of a barycentric stochastic interpolant offers a nuanced approach, ensuring the weights adhere to a simplex and providing a structured method to learn to interpolate samples from different distributions.

### Weaknesses
- *Ambiguous Interpretation*: The paper does not provide a clear semantic understanding of the paths between distributions, making it difficult to ascertain the practical implications or the broader relevance of the methodology. Also, what does multi-marginal optimal transport path mean for image generation? Does it lead to better quality? In this paper, it is not answered.
- *Unknown Utility*: There's a lack of clarity on the potential use-cases for this generative model. For instance, if it's geared towards style transfer, it's essential to determine which style gets preserved and how it compares with existing baselines in that domain.
- *Novelty*: The novelty of doing stochastic interpolant for multiple marginal distributions is limited given that it is done for the two marginal case. The theoretical result seems to be an extension from the two marginal case to the multi-marginal case as well.
- *Minor*: missing caption for Figure 2 (right).

### Questions
- Can the authors provide some more justification on why learning such an optimal transport path is useful? If learning the path in between distributions is to facilitate objectives like style transfer, then it should be compared with baselines on that task and show the utility of having this model. Right now, I am struggling to understand what is the use case especially for images when there is not a well-defined joint distribution between the multiple marginals.
- It is unclear to me what is the implication of the Monge type coupling theoretical result. Is this shown to be relevant to the experimental results? If not I would recommend it to be put in appendix. It is also worth explaining more about the what this result implies and when we would like to have this property.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors propose an extension of stochastic interpolants for multimarginal generative modeling, which involves learning a joint probability distribution that captures multiple marginal probability densities. The proposed framework is able to capture multi-way correspondences.

### Strengths
The work proposes a theoretically sound and practical approach based on stochastic interpolants for the multimarginal setting. The overall proposed scheme is mathematically sound and computationally more feasible than existing schemes. Extensive experiments are performed to illustrate the idea and different scenarios of the proposed method.

### Weaknesses
The overall algorithm of the proposed method might not be very clear or easy to follow, especially for mathematically less mature audience. I suggest the authors add the pseudocode of the main algorithm to improve clarity.

### Questions
Referring to the weaknesses part, how do you solve (16)? 

Other remarks:
- Incomplete caption for Figure 2 (Right). 
- Page 9 first paragraph: “from a sample from a sample”
- References: Song et al. (2021b, c) are repeated.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent
