# Flow Matching with Semidiscrete Couplings

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Flow models parameterized as time-dependent velocity fields can generate data from noise by integrating an ODE.
These models are often trained using flow matching, i.e. by sampling random pairs of noise and target points $(x_0,x_1)$ and ensuring that the velocity field is aligned, on average, with $x_1-x_0$ when evaluated along a time-indexed segment linking $x_0$ to $x_1$. 
While these noise/data pairs are sampled independently by default, they can also be selected more carefully by matching batches of $n$ noise to $n$ target points using an optimal transport (OT) solver.
Although promising in theory, the OT flow matching (OT-FM) approach (Pooladian et al., 2023, Tong et al., 2024) is not widely used in practice. 
Zhang et al. (2025), pointed out recently that OT-FM truly starts paying off when the batch size $n$ grows significantly, which only a multi-GPU implementation of the Sinkhorn algorithm can handle.
Unfortunately, the pre-compute costs of running Sinkhorn can quickly balloon, requiring $O(n^2/\varepsilon^2)$ operations for every $n$ pairs used to fit the velocity field, where $\varepsilon$ is a regularization parameter that should be typically small to yield better results.
To fulfill the theoretical promises of OT-FM, we propose to move away from batch-OT and rely instead on a semidiscrete formulation that can leverage the fact that the target dataset distribution is usually of finite size $N$. The SD-OT problem is solved by estimating a dual potential vector of size $N$ using SGD; using that vector, freshly sampled noise vectors at train time can then be matched with data points at the cost of a maximum inner product search (MIPS) over the dataset.
Semidiscrete FM (SD-FM) removes the quadratic dependency on $n/\varepsilon$ that bottlenecks OT-FM. SD-FM beats both FM and OT-FM on all training metrics and inference budget constraints, across multiple datasets, on unconditional/conditional generation, or when using mean-flow models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Semidiscrete Flow Matching (SD-FM), a novel approach to define the noise-to-data coupling in flow matching. The core idea is to perform a offline pre-computation to solve a global, semidiscrete Optimal Transport (OT) problem, which yields a dual potential vector g*. This vector g* implicitly defines a partition of the noise space, where each region in the partition corresponds to a unique data point. During the online FM training phase, each sampled noise vector is coupled to its corresponding data point by performing a Maximum Inner Product Search (MIPS) to identify its region. This "pre-compute then query" strategy is designed to improve the efficiency and scalability of OT-guided flow matching. The authors demonstrate significant performance gains on some benchmarks.

### Strengths
- Solid Theoretical Analysis: The paper provides a convergence analysis for the core algorithm, with a systematic comparison with existing optimal transport-based methods. This lends to theoretical importance to the proposed method. 

- Clarity of Exposition: The paper is well-written and logically structured. Figure 1, in particular, is highly effective. It intuitively illustrates the core methodological differences between I-FM, OT-FM, and the proposed SD-FM, allowing readers to quickly grasp the paper's central contribution.

### Weaknesses
- Limited Experimental Scope: The method was only evaluated on two small-scale benchmarks. This is a significant omission, as the problem of computational inefficiency—which the paper claims to solve—is most critical on large-scale, high-dimensional datasets like ImageNet 256. The absence of such experiments makes it difficult to validate the method's scalability and practical utility.

- Limited Baseline Comparisons: The set of baseline methods is narrowly restricted to only two types of flow-matching models. The paper lacks a comparison against current state-of-the-art (SOTA) generative models in the related domain. To properly contextualize its contributions, the method must be benchmarked against SOTA approaches in terms of both generation quality and efficiency.

- Insufficient Analysis of the Key Hyperparameter ε: The vast majority of experiments are conducted with ε=0 (hard assignment). The parameter ε controls the stochasticity of the coupling (soft vs. hard assignment) and is highly likely to directly influence the diversity of the generated model. The paper lacks any investigation into how ε mediates the trade-off between generation quality (e.g., FID/precision) and diversity (e.g., recall).

### Questions
In table 2, the generation quality is better at $\epsilon = 0$, while some theoretical results are only valid when $\epsilon > 0$, e.g., Theorem 2, Proposition 3. Could you provide some explanation for this phenomenon?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To overcome the computational cost but fulfill the theoretical promises of OT-FM, this paper leverages the entropy regularized semidiscrete OT (SD-OT) and proposes a SD-OT coupling. In the coupling stage, the SD-OT problem is solved by estimating the dual potential $g$ by SGD; then in the FM training stage, noise samples are pared to data samples by maximum inner product search (MIPS, when $\epsilon = 0$) or SoftMax (when $\epsilon > 0$), using the learned $g$. They also provide the convergence analysis of SGD for SD-OT. The experiments using ImageNet, PetFace and CelebA show that the proposed SD-OT coupling is better than I-FM and OT-FM in main metrics.

### Strengths
The paper is well-motivated by practical drawbacks of OT-FM, and the general coupling method SD-FM can potentially integerate into many diffusion-/ flow-based models. The convergence analysis justify the case for both $\epsilon = 0$ and $\epsilon > 0$. The validity of unregularized $\epsilon = 0$ case avoids the choice of tuning parameter.

### Weaknesses
1. The paper mainly adapts SD-OT from Genevay et al., 2016 in FM setting, which may raise some novelty issues.
2. The scalabililty of SD-OT precompucate for large $N$ remains a concern.
3. In section 2 (Background and Related Work), are "$\ldots$" in subtites typos?

### Questions
1. Although $\epsilon = 0$ is valid, how sensitive are results to $\epsilon$? Also, what's the sensitivity for fitted $g$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Semidiscrete Flow Matching (SD-FM), a novel approach that addresses the computational bottlenecks of Optimal Transport Flow Matching (OT-FM). By reformulating the noise-data pairing problem as a semidiscrete optimal transport problem from a continuous noise distribution to a discrete dataset, SD-FM fits a dual potential vector via SGD during a precomputation phase. This allows each newly sampled noise vector to be matched to a data point at training time through a simple maximum inner product search, effectively eliminating the quadratic dependency on batch size that plagues batch-OT methods. Extensive experiments demonstrate that SD-FM outperforms both standard FM and OT-FM across various datasets and training settings, achieving superior results in unconditional/conditional generation tasks under all tested inference budget constraints.

### Strengths
- The paper introduces a fundamentally different approach to optimal transport in flow matching by leveraging the semidiscrete formulation, effectively circumventing the quadratic complexity that has limited practical adoption of OT-FM methods.
- Beyond the algorithmic contribution, the work provides thorough theoretical analysis including convergence guarantees for both regularized and unregularized settings and extends the Tweedie formula to the semidiscrete case, demonstrating strong mathematical rigor.
- The experimental design covers diverse scenarios including unconditional/conditional generation, super-resolution, and mean-flow models, with consistent improvements across different inference budgets, particularly notable in low-NFE regimes.
- The decoupling of precomputation from training, combined with efficient MIPS-based matching, offers a viable path for scaling OT-based methods to larger datasets while maintaining performance benefits.

### Weaknesses
- While the 12-hour precomputation on 2.56M samples is reasonable, the paper lacks systematic analysis of how the method scales to modern large-scale datasets (e.g., >10M samples). The O(N) training cost, though better than quadratic, may still become prohibitive for massive datasets.
- The comparison framework omits important contemporary approaches for improving flow straightness (e.g., Reflow, MinibatchOT variants), making it difficult to assess SD-FM's relative advantages in the broader landscape of efficient flow matching methods.
- The paper provides limited insights into hyperparameter selection, particularly regarding the regularization parameter ε. Practical recommendations for choosing ε based on dataset characteristics (dimensionality, size, noise levels) would significantly enhance reproducibility and usability.
- The experimental validation focuses primarily on lower-resolution datasets, leaving open questions about the method's effectiveness on high-resolution generation tasks (e.g., 256×256 and above) that are increasingly relevant in practical applications.

### Questions
See the "Weakness" section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Semidiscrete Flow Matching to tackle the training computational cost of OT-FM. It studies a semidiscrete OT problem from a continuous noise to a discrete data distribution. This is solved in two stages: a one-time precomputation of a dual potential vector, followed by a MIPS lookup during training to pair noise with data. Theoretical convergence analysis and empirical results show SD-FM beats I-FM and matches OT-FM at a fraction of the computational cost.

### Strengths
1. The core idea of reframing OT-FM as a continuous-to-dataset semidiscrete problem is novel and practically efficient. The associated convergence criterion is also a new contribution.
2. The paper is well-written. The motivation is clear. Figures and tables effectively communicate the method and its advantages.

### Weaknesses
1. The paper alludes to fast approximate MIPS but does not investigate the trade-off between approximation error and final model quality, measured by FID. For large $N$, what is the impact of the approximation error from MIPS?
2. The paper lacks validation on high-dimensional data. Current diffusion models often denoise in latent spaces, which have high data dimensions.

### Questions
1. Figure 5 presents a nice analysis of diversity control through guidance samples. Could you provide more extensive analysis on how different parameters affect the sample diversity?
2. When applied to datasets of billions of samples, what is the additional memory overhead of SD-FM compared to I-FM?

### Soundness
3

### Presentation
4

### Contribution
3
