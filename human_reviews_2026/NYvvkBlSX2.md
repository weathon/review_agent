# Incomplete Data, Complete Dynamics: A Diffusion Approach

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Learning physical dynamics from data is a fundamental challenge in machine learning and scientific modeling. Real-world observational data are inherently incomplete and irregularly sampled, posing significant challenges for existing data-driven approaches. In this work, we propose a principled diffusion-based framework for learning physical systems from incomplete training samples. To this end, our method strategically partitions each such sample into observed context and unobserved query components through a carefully designed splitting strategy, then trains a conditional diffusion model to reconstruct the missing query portions given available contexts. This formulation enables accurate imputation across arbitrary observation patterns without requiring complete data supervision. Specifically, we provide theoretical analysis demonstrating that our diffusion training paradigm on incomplete data achieves asymptotic convergence to the true complete generative process under mild regularity conditions. Empirically, we show that our method significantly outperforms existing baselines on synthetic and real-world physical dynamics benchmarks, including fluid flows and weather systems, with particularly strong performance in limited and irregular observation regimes. These results demonstrate the effectiveness of our theoretically principled approach for learning and imputing partially observed dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addressed the problem of imputing observational data under unknown dynamics. In diffusion-based imputation models, context mask and query mask are needed to train the model to extend the learned dynamics to impute unobserved data. The authors claim that the unobserved data is structured in reality and that diffusion and mask-based methods lack theoretical foundations, and then they introduce their novel pipeline to address the structured missing prior. The authors theoretically analyzed the correct strategy for context-query mask partitioning and extrapolation, and conducted comprehensive experiments on simulated and real-world datasets against competitive baseline methods. The results demonstrated the superiority of their method and the importance of catering to the structured missing prior in algorithm design. Their work could be a sound analysis platform for future diffusion-based and mask-based methods in data imputation.

### Strengths
S1: This paper gives a clear literature review and a contrasting comparison with similar models.
S2: The algorithm is backed by rigorous theorems and proofs, and additionally addresses the structured missing prior in reality.
S3: The experiments are thoughtful with ablation studies on the model design and components.

### Weaknesses
W1: The explanation of the mask selection strategy is confusing, and Figure 1 is hard to understand. In the series of training data with added noise, all data points seem to be observed once across time. No data point needs to be imputed.
W2: Sometimes, the mask distribution $p_{mask}$ depends on the data samples $x_0$. For example, a weather station may be damaged by a typhoon's direct impact. Also, cloud blocks weather satellite observing the sea temperature, and high sea temperature also promotes cloud generation.
W3: The authors proposed that the mask selection strategy for $M_{ctx},M_{qry}$ should align with the mask distribution prior $p_{mask}$ and used a block mask as an example. However, the authors didn't consider the case when the mask distribution pattern cannot tessellate on the whole space.
W4: The analysis of the paper's limitations is missing in the main text.
W5: The authors claim that some methods are not scalable, but they have not provided a scalability analysis and report on wall-time across baselines.

Minor remarks:
1. Line 723 VAE-based "and" GAN-based methods.
2. The definition of $x_\theta$ is unclear on line 138, and its input and output should be specified.
3. What is the definition of $i$ in eq. 2? Training instance or feature dimension?
4. A cross-reference to the proof of Thm 1 should be included in the paragraph starting from line 172.
5. On line 361, the improvement on MissDiff from noise matching to data matching needs to be quantified.
6. The margin of the caption of Figure 3 is too narrow.

### Questions
Q1: What is the term "feature" referring to in the main text? Does it refer to different data modalities or cell grid?
Q2: We observed a significant improvement from block-wise to pixel-level configuration on the Navier-Stokes dataset in Table 2. Is it because the authors tuned the hyperparameters on the NS dataset, as stated in tables 6-7? This could be a source of evaluation bias.
Q3: How is the physical dynamic learned? The data is not assumed to include the time dimension, and the "time" in the main text only refers to the diffusion time steps.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a diffusion-based framework for learning physical dynamics when the training data itself is incomplete. The core contribution is a training strategy termed "context-query partitioning." This strategy partitions the observed data within each incomplete sample into a "context" for model input and a "query" for loss computation. Theoretically, the authors demonstrate that this approach asymptotically recovers the true data distribution without full supervision, provided the partitioning strategy ensures a non-zero probability for any dimension to be queried. For inference, the paper proposes an ensemble sampling method that reconstructs the complete sample by averaging predictions over multiple, randomly drawn context masks. Empirical results on several PDE and climate datasets demonstrate superior performance over existing baselines, especially in highly sparse observation regimes.

### Strengths
Problem Significance: The paper tackles a crucial and pervasive problem in scientific computing: learning from inherently sparse and incomplete observational data.
Theoretical Rigor: The paper provides a theoretical justification (Theorem 1) for its proposed training strategy. This principled approach is a clear advantage over prior works that often rely on heuristics.
Strong Empirical Results: The method demonstrates significant performance improvements across multiple benchmarks, particularly in sparse regimes with very low data coverage (e.g., 1%-20%).

### Weaknesses
- Insufficient Discussion of Related Work: The primary weakness is the failure to adequately discuss the connection to masked signal modeling, such as Masked Autoencoders (MAEs). The "context-query partitioning" is conceptually very similar to masking part of the visible data to predict another part. The absence of this discussion makes the core idea seem more novel than it is.
- Limited Baseline Comparison: While the paper includes recent diffusion-based baselines, the modification of some baselines (e.g., adapting MissDiff to a data matching framework) makes it difficult to assess if the comparison is entirely fair. Furthermore, excluding some theoretically rigorous but computationally expensive methods, while understandable, would be more convincing with a small-scale comparison or a deeper discussion.
- Scope of Theory: While Theorem 1 is a highlight, it focuses on asymptotic convergence. It doesn't analyze finite sample complexity or the approximation error introduced by the model architecture. The theorem's premise—that the union of query masks covers all dimensions—is satisfied by the design of the sampling strategy, which makes the argument slightly circular.

### Questions
1. Could the authors elaborate on the conceptual similarities and differences between the "context-query partitioning" strategy and self-supervised learning paradigms like Masked Autoencoders (MAEs)? This is crucial for accurately positioning the paper's novelty.
2. In the baseline comparisons, were the hyperparameters for all baseline methods tuned with the same rigor as the proposed method? Ensuring a robust comparison is key to validating the claimed performance gains.
3. The "distribution-preserving" sampling strategy mentioned in line 264 appears critical to success. In real-world scenarios where prior knowledge of P_mask(M) might be inaccurate, how sensitive is the method's performance to this "mask distribution mismatch"?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a diffusion-based framework for learning physical dynamics from incomplete and irregularly sampled data, a common challenge in scientific domains. The core contribution is a novel training paradigm where each incomplete data sample is partitioned into a "context" set and a "query" set. A conditional diffusion model is then trained to reconstruct the query portions given the context, without requiring access to complete ground-truth data. The authors provide a theoretical analysis demonstrating that their method asymptotically converges to the true data distribution under certain conditions. Furthermore, they introduce an ensemble sampling technique for inference to reconstruct the complete data. The method is evaluated on several synthetic PDE datasets (Navier-Stokes, Shallow Water, Advection) and the real-world ERA5 climate dataset, showing superior performance over existing baselines, especially in highly sparse settings.

### Strengths
The paper tackles the critical and practical challenge of learning physical dynamics from inherently sparse and incomplete real-world data.

It provides a theoretical analysis (Theorem 1 and 2) that offers guarantees for the learning process, explaining how and why the model can learn from incomplete data. This is a notable strength compared to more heuristic approaches.

### Weaknesses
Limited Novelty: The core idea of context-query partitioning is conceptually similar to existing masked data training strategies in self-supervised learning and other generative models. The distinction from methods like Ambient Diffusion, which also operate on incomplete data, is not sufficiently pronounced. The contribution seems to be more about a specific, effective sampling strategy rather than a fundamentally new paradigm.

Lack of Clarity on Implementation: The "strategic context-query partitioning" is the cornerstone of the method, yet its implementation is vaguely described. The paper lacks a clear algorithm or pseudo-code explaining how M_ctx is sampled from a given observation mask M for different structural patterns (e.g., block-wise). This ambiguity hinders reproducibility.

Incomplete Baseline Comparison: Excluding recent methods like those by Chen et al. (2024b), Givens et al. (2025), and Zhang et al. (2025a) due to computational cost weakens the experimental validation. A more thorough comparison, even on a smaller scale, would be necessary to firmly establish state-of-the-art performance.

Unaddressed Inference Cost: The proposed ensemble sampling method for reconstruction requires running the model K times per sample. This introduces a significant computational overhead at inference time, which is a critical factor for many scientific applications. The paper does not analyze this trade-off or discuss how K is chosen.

### Questions
Could the authors please elaborate on the key conceptual difference between their method and Ambient Diffusion (Daras et al., 2023)? Both seem to train diffusion models by applying masks to incomplete data. What is the crucial insight that leads to the performance gains shown?

Could you provide a more concrete algorithm or pseudo-code for the "strategic context-query partitioning" procedure, especially for the block-wise missing data scenario? For a given observation mask M with 5/9 observed blocks, how exactly is M_ctx (e.g., with 4 blocks) sampled?

Regarding the excluded baselines: Could the authors provide a more detailed justification for their exclusion? Would it be possible to conduct a qualitative or small-scale quantitative comparison to give the reader a better sense of where the proposed method stands?

What is the inference-time cost of the ensemble sampling approach? How does the performance (e.g., MSE) and computational cost vary with the number of ensemble members K? What was the value of K used in the experiments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles a genuinely important problem: learning physical dynamics when training data is inherently incomplete. The authors propose training diffusion models by strategically splitting each incomplete sample into "context" (what the model sees) and "query" (what it tries to predict), then using ensemble averaging at inference time to reconstruct complete fields.
The key idea is that if you carefully design how you partition incomplete observations during training, matching the partition strategy to the underlying observation pattern (e.g., block-wise vs. pixel-wise missing data), one can train a model that learns meaningful conditional expectations even for dimensions that were never observed in the training data. The authors provide a theoretical analysis that proves convergence to the true complete data distribution and validate it on both synthetic PDEs (Shallow Water, Advection, and Navier-Stokes) and real climate data (ERA5).

### Strengths
- The incomplete data setting represents a fundamental constraint in scientific applications where complete ground truth observations are physically impossible to obtain (e.g., global weather systems, ocean dynamics). The motivation is well-articulated and substantiated.

- The context-query partitioning strategy constitutes a genuine advancement over existing approaches. The key differentiation with existing lies in training-time strategic partitioning that adapts to observation patterns, rather than inference-time conditioning or generic loss masking. 

- Theorem 1, concerning gradient scaling and parameter update frequency, provides meaningful insight into the significance of partitioning strategy selection. Theorem 2's ensemble convergence analysis demonstrates notable rigor, employing Martingale convergence theory rather than heuristic arguments. This theoretical grounding distinguishes the work from purely empirical approaches.

- The performance improvements documented in Table 2 for block-wise observations are substantial. The 1% observation regime results on ERA5 are particularly noteworthy, representing an extremely challenging setting.

### Weaknesses
A bit of nitpicking:
- The manuscript demonstrates conditions under which the method succeeds, but provides insufficient analysis of failure modes. Specifically, when does the information gap exceed the capacity of ensemble averaging to compensate? The cross-distribution experiments (Appendix G.3, Table 8) suggest performance degradation when the training and test observation ratios differ substantially; however, this warrants a more thorough discussion. Are there principled criteria for determining whether a test distribution lies outside the method's applicability range?

- Ensemble averaging during inference requires K forward passes through the model (the paper employs single-step sampling with K context masks). The selection of K and the resulting computational trade-offs are not discussed. For a method targeting scientific applications, these practical considerations are important. The multi-step sampling described in Appendix E further compounds this concern. Quantitative comparison of computational costs relative to baselines would be valuable.

### Questions
- How is K selected for ensemble averaging? Is there a principled approach to determining the number of context masks, or is selection based on empirical validation? Does the required K scale with observation sparsity?

- What is the computational overhead of the proposed method? Can you provide wall-clock training and inference times relative to baseline methods? This information is essential for practical implementation.

### Soundness
3

### Presentation
3

### Contribution
4
