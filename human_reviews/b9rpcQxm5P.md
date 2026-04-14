# Equivariant score-based generative models provably learn distributions with symmetries efficiently

- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Symmetry is ubiquitous in many real-world phenomena and tasks, such as physics, images, and molecular simulations. Empirical studies have demonstrated that incorporating symmetries into generative models can provide better generalization and sampling efficiency when the underlying data distribution has group symmetry. In this work, we provide the first theoretical analysis and guarantees of score-based generative models (SGMs) for learning distributions that are invariant with respect to some group symmetry and offer the first quantitative comparison between data augmentation and adding equivariant inductive bias. First, building on recent works on the Wasserstein-1 ($\mathbf{d}_1$) guarantees of SGMs and empirical estimations of probability divergences under group symmetry, we provide an improved $\mathbf{d}_1$ generalization bound when the data distribution is group-invariant. Second, we describe the inductive bias of equivariant SGMs using Hamilton-Jacobi-Bellman theory, and rigorously demonstrate that one can learn the score of a symmetrized distribution using equivariant vector fields without data augmentations through the analysis of the optimality and equivalence of score-matching objectives. This also provides practical guidance that one does not have to augment the dataset as long as the vector field or the neural network parametrization is equivariant. Moreover, we quantify the impact of not incorporating equivariant structure into the score parametrization, by showing that non-equivariant vector fields can yield worse generalization bounds. This can be viewed as a type of model-form error that describes the missing structure of non-equivariant vector fields. Numerical simulations corroborate our analysis and highlight that data augmentations cannot replace the role of equivariant vector fields.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
> This paper addresses how generative models can benefit from leveraging symmetries in data distributions to improve sampling efficiency and generalization on score-based generative models (SGMs). The authors focus on the empirical success of SGMs in scenarios where data has intrinsic symmetries (e.g., physics, molecular simulations). They argue that symmetries embedded in data distributions, known as group symmetries, can lead to improved model performance.

> First, they analyze how equivariant score-matching (with vector fields preserving symmetry) is an optimal method for training SGMs with symmetric data, bypassing the need for extensive data augmentation. Building on *Mimikos-Stamatopoulos et al. (2024)*, the authors generalize their bound errors in SGMs, breaking it down into errors stemming from non-equivariance, score-matching, sample complexity, and early stopping. They use the Wasserstein-1 distance (*d1*) as a metric to show how errors in these areas impact model performance, relying on HJB techniques and the mFG interpretation of SGMs.
The authors provides also numerical simulations aiming to corroborate their theoretical claims and to illustrate that embedding equivariant structures directly into SGMs consistently outperforms data-augmented models in generating high-quality samples of symmetric distributions, with lower Wasserstein-1 distance errors.

> This paper advances the theoretical understanding of symmetry in generative models, offering both conceptual and practical insights into how SGMs can be designed to better handle symmetric data distributions, efficiently using equivariant inductive biases instead of data augmentation.

### Strengths
The article demonstrates several notable strengths that advance the understanding and effectiveness of score-based generative models by focusing on equivariance:
- **Efficient Generalization of Existing Literature:** Building on the recent work by *Mimikos-Stamatopoulos et al. (2024)*, this article rigorously shows that $G$-invariance (group symmetry) is a key property that strengthens the structure and efficiency of SGMs. By extending these theoretical foundations, the authors highlight equivariance as crucial for improving model convergence and generalization.
- **Insightful Theoretical Framework Linking Key Concepts:** The paper effectively utilizes connections between MFG optimization, the HJB equation, and SGMs to clarify the structural properties of these models. This theoretical linkage reveals how SGMs can leverage inherent symmetries, providing a clearer perspective on the convergence bounds and making the role of symmetry more explicit.
- **Detailed Separation of Symmetry Components in Convergence Bounds:** The authors efficiently disentangle the components of symmetry within convergence bounds, offering a nuanced view of how these factors contribute to model stability and performance. This separation facilitates a better understanding of the role of symmetry in error decomposition and contributes to more accurate theoretical predictions.
- **Guidance for Model Enhancement:** The article provides practical suggestions for leveraging $G$-invariance directly within model architectures. This approach minimizes the need for data augmentation and improves computational efficiency while supporting more robust and generalizable models, laying groundwork for future research that optimally exploits symmetry.

### Weaknesses
- **Bounds Inefficiency:** The generalization bounds presented appear to lack efficiency, especially as they depend on the radius $R$ of the compact domain in which the data is confined. This leads to a bound that increases with the data’s spatial constraints. Additionally, the time horizon $T$ dependency in the bound further amplifies this issue, potentially limiting the practical applicability of the results.
- **Limited Novelty in Symmetry Component Analysis:** While the authors successfully separate the symmetry component in the bounds from *Mimikos-Stamatopoulos et al. (2024)*, the originality of this analysis seems limited. Revisiting classical results without substantial improvements in simulation outcomes weakens the impact of this contribution on advancing symmetry-related insights.
- **Insufficient Context for Symmetry in Examples:** Although the paper addresses how the absence of symmetry could affect convergence bounds, it lacks clear examples that naturally demonstrate symmetry or its absence. A more thorough description of practical cases where symmetry is essential would strengthen the authors’ arguments and provide clearer motivation.

### Questions
- **Limited Scope of Numerical Experiments:** The numerical experiments are minimal and fail to comprehensively illustrate the necessity of incorporating equivariance into the SGM architecture. The four-Gaussian example appears overly simplistic, raising questions about the generality of the findings. Moreover, even in this simple example the benefit of using equivariance adjustments does not seem to clear. Could a more complex real-world example better justify the proposed framework? Also, is the bound evaluated and, if so, is it tight?
- **Choice of Architecture:** Why did the authors choose the current architecture over U-Net, which has been crucial to the success of SGMs due to its computational efficiency and architectural adaptability? Clarifying the rationale could provide insights into the model’s design choices.
- **Inconsistencies Between Theory and Experiments:** The authors constrain the diffusion process to a compact set but then use a Gaussian distribution in experiments, which does not align with this assumption. How should the bound be applied or evaluated in this case? Addressing this discrepancy would clarify the applicability of the results across different setups.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
For learning distributions with certain symmetries using score-based generative models, this paper performs the first theoretical analysis and provides some guarantees. In particular, the authors show an improved error bound when the data distribution is group-invariant, use Hamilton-Jacobi-Bellman (HJB) theory to describe the inductive bias, and provide numerical tests to reinforce their claims.

### Strengths
The paper provides many theoretical results in a rigorous mathematical setting, when the target distribution has some symmetries. Furthermore, many of the results directly translate to practical use cases (e.g., Theorem 4).

### Weaknesses
While the (only) numerical example well demonstrates the theoretical results, it is a very simple one-dimensional experiment. It would have been much more convincing to see the theory validated with a higher-dimensional learning problem. For example, there are several high-dimensional experiments in Birrell et al. (2022) in ICML (cited by the present paper) and I wonder if the authors can validate their theoretical results in any of them.

### Questions
1. I understand that G is a given group when it comes to the theoretical results, but what exactly is the G group when it comes to the numerical results? For instance, in Section 6, is the group G the collection of four Gaussians or some properties of these four Gaussians (or other things)? How the elements of the example correspond to the elements of the theoretical parts need to be addressed in more detail. Specifically, I wonder if the authors can explicitly define the G group for the numerical example and provide a clear mapping between the theoretical elements and their counterparts in the experiment. 

2. In the Appendix, at lines 748-752, I do not understand how applying Theorem 6 yields the approximate inequality. If you already have the exact inequality in Theorem 6, why don't we just use the exact inequality (instead of some approximation)? I wonder if the authors can provide a step-by-step derivation of how they arrive at the approximate inequality from Theorem 6, or explain why an approximation is used if an exact inequality is available.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents the first theoretical analysis of score-based generative models (SGMs) for learning distributions with symmetries. It proves that incorporating symmetries through equivariant score approximations leads to improved generalization bounds and sampling efficiency. The study demonstrates that equivariant SGMs can effectively learn symmetric distributions without needing data augmentation and quantifies the impact of non-equivariant score parametrization on generalization error. The theoretical results are investigated with simple numerical experiments.

### Strengths
The paper is easy to read and provides a new theoretical anlysis for equivariant score-based generative models for learning symmetric distributions, providing valuable insights into their improved generalization and sample efficiency. The theoretical results are able to provide practical insights on comparison of equivariance with data augmentation. 

The paper presents a first-time quantitative comparison between data augmentation and the use of equivariant score parametrizations, demonstrating the advantages of the latter in terms of generalization error bounds.

### Weaknesses
There are multiple weakness of this paper: 
Major: 
The numerical experiments, are conducted on extremely simple examples and the conclusions reported do not appear to follow directly from the example considered. Score based models are often used in much more complex, higher-dimensional datasets. I expand further on this in the questions sections. Furthermore, the main result of Section 4.1 - Theorem 3 seems completely irrelevant to the rest of the paper (or at least not sufficiently connected to it - happy to be proven wrong). 

Minor: 
The analysis focuses on symmetries representable by unitary matrices, significantly limiting its applicability. While the paper advocates for equivariant SGMs, it doesn't fully address the potential increase in computational complexity associated with designing and training equivariant neural networks.

### Questions
* Line 140: Clasically the projection $S_G$ is referred to as the Reynolds operator. 
* In Line (3) - doesnt $\nabla \text{log} \rho$ transform as $A^{\top}\nabla \text{log} \rho(Ax)$?
* Line 296 - it seems that the main point is the improved scaling of $1/d^*$, though not emphasised.
* Theorem 3 appears to be almost completely irrelevant? Could you help me understand why the result is either not obvious or expand on why exactly it is relevant? 
* Line 370 onwards I am not sure how the analogy with symplectic integrators fits into the picture there
* I beleive that remark 3 once again is entirely obvious without the theorems from the objective function? 
* Line 430 - this seems to completely miss the point of non-optimisability. Oftentimes, optimizing equivariant functions is harder/more expensive. And while $e_nn$ can be made as small as possible in theory, practice will be different.
* Figure 1 - I am confused why blue and orange are not the same exact curves - the loss itself must be identical? Even more surprisingly - why are red and green so close?
* I believe that the approach that you use for approximating the Wasserstein distance is not correct. It is fairly well known in the literature that WGANs significantly overestimate the wasserstein distances: https://arxiv.org/pdf/2103.01678
* Line 491 - I believe that the provided results are not extensive enough to claim neither that training a non-equivariant with augmentation nor equivariant without are superior. What is more, training an equivariant network on augmented data does not appear to make much sense, as the parameter gradient are the same in both cases?

### Soundness
3

### Presentation
3

### Contribution
3
