# Regularization can make diffusion models more efficient

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 8, 2, 4

## Abstract
Diffusion models are one of the key architectures of generative AI. Their main
drawback, however, is the computational costs. This study indicates that the
concept of sparsity, well known especially in statistics, can provide a pathway to
more efficient diffusion pipelines. Our mathematical guarantees prove that sparsity
can reduce the input dimension’s influence on the computational complexity to that
of a much smaller intrinsic dimension of the data. Our empirical findings confirm
that inducing sparsity can indeed lead to better samples at a lower cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a regularization technique to accelerate diffusion models, with the core concept being the regularized denoising score-matching estimator (Equation 3). Based on this framework, the training efficiency of diffusion models is improved. The authors demonstrate the effectiveness of the method through both theoretical analysis and experimental validation.

### Strengths
A key strength of this paper lies in its theoretical rigor. The mathematical derivations of Theorem 1 are solid and clearly articulate the core conclusion. This high standard of rigor and clarity is also reflected in the mathematical sections and the explanation of the methods and motivations. Finally, the experimental results effectively validate this theoretical framework.

### Weaknesses
**Major concerns**
+ The TOY EXAMPLE in Figure 1 is confusing. The author states that “regularized denoising score matching predominantly adheres to the two-dimensional sub-manifold (along the Y and Z axes).” Can samples on both sides of the X-axis be effectively generated? Would this lead to a decrease in generated diversity?

+ The paper's evaluation currently relies primarily on visual comparisons, which are insufficient on their own. As previously mentioned, the manuscript does not include common quantitative metrics for generative models, such as Precision and Recall. The authors should incorporate these (or other relevant) metrics to provide a more objective assessment of the method's performance, specifically regarding its ability to approximate the original data (fidelity, Precision) and generate diverse samples (diversity, Recall).

> Improved Precision and Recall Metric for Assessing Generative Models. NeurIPS-2019

+ The current diffusion models adopt the Flow Matching framework, which constructs different intermediate distributions and shifts the model's prediction target from a score function to a velocity field. Can the conclusions of this paper be generalized to this setting?

> Flow Matching for Generative Modeling. arXiv:2210.02747

**Minor concerns**
+ The presentation of the figures could be improved. For instance, in Figures 2 and 3, enhancing the visual contrast between GT, baseline, and the proposed method would make the comparisons clearer and more impactful.

### Questions
+ Regarding the practical implementation of equation 3, **it is unclear how the parameters are constrained to satisfy the requirements of equation  2**. Clarification on this mechanism is needed. If the author could provide pseudocode for the implementation, it would make the article more persuasive.

+ Figure 5 shows relatively high FID values for CIFAR, while most diffusion models currently achieve FID values below 5. Could the authors explore more advanced network architectures to validate the method's generalizability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates how introducing l1-based regularization into score matching can make diffusion models more efficient, both theoretically and empirically. The authors assume that the true score function of the data distribution is approximately s-sparse, meaning that only a few dimensions matter for denoising. Under this sparsity assumption, they derive non-asymptotic bounds showing that the KL divergence scales with $s^2$ rather than $d^2$, and validate this through experiments on toy data and image datasets, where the regularized model achieves comparable or better quality with much fewer sampling steps.

### Strengths
1. Provides a clear theoretical link between sparsity regularization and improved non-asymptotic convergence, reducing dimensional dependence from $d$ to $s$.
2. The proposed regularized score-matching objective is simple to implement and compatible with existing diffusion frameworks

### Weaknesses
The main theoretical guarantee hinges on the unverified assumption that the true score function is s-sparse, and given the experiments are limited to low-dimensional datasets, it remains unclear whether the same efficiency extends to large-scale diffusion models.

### Questions
See Weakness

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to accelerate score-based diffusion model inference by introducing sparsity regularization into the score function. Theoretically, the authors provide proofs supporting the feasibility of the approach; however, the experimental results do not sufficiently validate these claims. The paper would benefit from an expanded experimental section, removal of the “paper outline” section, and more compact figure layouts to better utilize space and improve readability.

### Strengths
The paper provides detailed explanations for each assumption and theorem, offering clear intuition behind the proposed method. The theoretical derivations are comprehensive and rigorous, establishing a solid foundation for the concept of regularized score-based diffusion models.

### Weaknesses
1. The paper does not provide code or implementation details, making it difficult to assess the practical feasibility of the method. Moreover, there is no analysis of the hyperparameters $s,r,\kappa$, leaving the experimental section incomplete.

2. From the presented results, the method only shows effectiveness on datasets with inherently sparse structures, such as MNIST. Even for MNIST, where sparsity is visually evident, the sparsity-inducing SGM struggles to generate high-quality samples with a small number of timesteps. In Figure 2 (right column), the generated images at T=50 and T=20 still contain noticeable noise, and this issue becomes even more pronounced in Figures 3, 4, and 5.

3. Minor comment: The description of the diffusion process (Lines 29–34) should use consistent notation for the variable $x$ to avoid confusion.

### Questions
1. The sparsity-inducing regularization requires solving the optimization problem in Eq. (3), which introduces additional computational overhead. Although the paper emphasizes inference-time acceleration, it is important to analyze how this optimization affects training time and training stability.

2. Sparsity-inducing methods are typically applied for one of two purposes: (a) when genuine sparsity exists in the data or model structure with interpretable meaning, or (b) to improve computational efficiency in systems involving heavy matrix operations. However, diffusion processes do not inherently involve such high-order matrix computations. The authors should better justify the motivation. Why does enforcing sparsity in the score function improve inference efficiency, and under what conditions is this beneficial?

3. Adding sparsity constraints might reduce the model’s generalization ability to fit data distributions, especially for high-dimensional modalities such as images or text. How does the proposed method balance sparsity with the ability to capture complex data distributions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes adding a type of L^1 regularization to the score-matching objective in diffusion, and analyzes the iteration complexity of sampling from this model, and compares it to the standard score-matching objective -- it shows that convergence can be achieved using a number of iterations depending on the sparsity of the score function, rather than its ambient dimension. The paper shows empirical results on a synthetic task, MNIST and FashionMNIST to support its theoretical claims, and to support the assumption that score functions are sparse in practice.

### Strengths
The addition of an explicit L^1 regularization to the score-matching objective, and a guarantee on sampling iteration complexity depending on the sparsity of the score functions is interesting -- it's further interesting that the new objective does seem to give reasonable results on (very) small-scale experiments. The fact that the new objective provides a reasonable approximation to the score even in the absence of sparsity is a plus. The paper itself is easy to read, and the contributions are laid out clearly.

Overall, I like the contribution of this paper at a conceptual level, and the small-scale/initial empirical results are intriguing.

### Weaknesses
From a technical perspective, the theory seems to largely be a rehashing of previous results with some modifications. The empirical results are unfortunately too small-scale to be convincing to practitioners -- even the CIFAR experiments included in the appendix were run using a non-standard 32 channel network instead of the standard 128 channels. 

Additionally, it's not clear how the regularization parameter $r$ should be set in practice, and for instance, how sensitive the results are to this setting. The added complexity in the training could be a huge drawback for practitioners.

The theory could be made stronger by analyzing the *sample complexity* of learning the score using the new objective under sparsity constraints -- see [1] and [2]. I'd also be interested in knowing whether the recent more sophisticated sampling methods and analyses (see [3], [4]) translate to the sparse case analyzed in this paper.

[1]: Generative modeling with denoising auto-encoders and langevin sampling. A Block, Y Mroueh, A Rakhlin. 2020

[2]: Improved Sample Complexity Bounds for Diffusion Model Training. Shivam Gupta, Aditya Parulekar, Eric Price, and Zhiyang Xun. NeurIPS 2024.

[3]: Faster Diffusion Sampling with Randomized Midpoints: Sequential and Parallel. Shivam Gupta, Linda Cai, Sitan Chen. ICLR 2025.

[4]: Nearly -Linear Convergence Bounds for Diffusion Models via Stochastic Localization. J Benton, V De Bortoli, A Doucet, G Deligiannidis. ICLR 2024.

### Questions
1) What is the sample complexity of learning sparse scores using the objective you have proposed?
2) Can you achieve iteration complexity smaller than s^2 using the recent more sophisticated sampling algorithms and analyses?
3) Do the results translate to larger scale? For example LSUN or Celeb-A?
4) There are no quantitative results provided for many of the small scale experiments (MNIST, etc) -- can you report them?

### Soundness
3

### Presentation
3

### Contribution
2
