# Bilateral Distribution Compression: Reducing Both Data Size and Dimensionality

- Avg Score: 5.20
- Decision: Reject
- Scores: 6, 4, 6, 4, 6

## Abstract
Existing distribution compression methods reduce dataset size by minimising the *Maximum Mean Discrepancy* (MMD) between original and compressed sets, but modern datasets are often large in both sample size and dimensionality. We propose *Bilateral Distribution Compression* (BDC), a two-stage framework that compresses along both axes while preserving the underlying distribution, with overall linear time and memory complexity in dataset size and dimension. Central to BDC is the *Decoded MMD* (DMMD), which quantifies the discrepancy between the original data and a compressed set decoded from a low-dimensional latent space. BDC proceeds by (i) learning a low-dimensional projection using the *Reconstruction MMD* (RMMD), and (ii) optimising a latent compressed set with the *Encoded MMD* (EMMD). We show that this procedure minimises the DMMD, guaranteeing that the compressed set faithfully represents the original distribution. Experiments show that across a variety of scenarios BDC can achieve comparable or superior performance to ambient-space compression at substantially lower cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Bilateral Distribution Compression (BDC), a two‑stage framework to compress datasets simultaneously along the sample and feature axes. Stage 1 trains an autoencoder by minimising a Reconstruction MMD (RMMD) between the data and its reconstruction. Stage 2 constructs a size‑$m$ compressed set in the latent space by minimising an Encoded MMD (EMMD) between encoded data and the compressed set. The paper defines the target discrepancy Decoded MMD (DMMD) between the original data and the decoded compressed set. It proves that vanishing RMMD and EMMD imply vanishing DMMD and also gives a bound using a pull‑back kernel . Empirically, BDC (linear and nonlinear variants) is evaluated on synthetic and real datasets for regression, classification, and clustering, reporting competitive accuracy with substantially lower runtime than ambient‑space compression baselines.

### Strengths
- DMMD formalises distributional fidelity after decoding; RMMD/EMMD decompose the otherwise entangled optimisation. This clarifies targets vs surrogates.
- Thm. 3.3 links zero RMMD/EMMD to zero DMMD (Sec. 3.1.4), aligning the two‑stage training with the main goal.
- Across Swiss‑Roll/CT‑Slice/MNIST/Clusters, BDC reaches comparable or superior predictive metrics with markedly lower runtime, reinforcing practical value (impact). Exact BDC verifies the RMMD+EMMD bound on DMMD with Gaussian mixtures, directly validating the theory.

### Weaknesses
- The DMMD ≤ RMMD+EMMD bound (Thm. 3.5) requires the pull‑back kernel, yet the main experiments typically define kernels directly in latent space (Sec. 4: “Unless otherwise stated…”), so the bound may not apply in most results.
- The hybrid loss (Eq. (5)) fixes the relative scale (RMMD + MSRE/nd) without a tunable weight or ablation over trade‑offs, limiting guidance.
- On MNIST, M3D is run without its up‑sampling trick explicitly “to enable a fair comparison,” potentially under‑representing that method’s best‑known configuration.
- The joint DJMMD/RJMMD/EJMMD framework is well‑defined, but it is not always clear which experiments actually used the joint objectives vs marginal ones.

### Questions
- For each supervised task, explicitly state whether RJMMD/EJMMD (joint) or RMMD/EMMD (marginal) were used, and report the corresponding values. Specify the response kernel choices ($l$) per dataset (delta vs Gaussian on one‑hots; real‑valued $y$) and include an ablation on $l$.
- Is it possible to introduce a tunable weight, in Eq. (5), sweep this weight, and report sensitivity of test metrics and RMMD?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Bilateral Distribution Compression (BDC), a framework for compressing datasets along both the sample and feature dimensions while preserving their underlying distribution. The method introduces three kernel-based discrepancy measures—Reconstruction MMD (RMMD), Encoded MMD (EMMD), and Decoded MMD (DMMD)—and shows theoretically that minimizing RMMD and EMMD ensures a small DMMD.

BDC operates in two stages: (1) learning a low-dimensional representation via RMMD-based autoencoder training, and (2) optimizing a compact latent set via EMMD. The authors also extend BDC to supervised data using joint feature–label kernels.

Experiments on several synthetic and real datasets show that BDC achieves similar or better performance than existing distribution compression methods, with much lower computational cost.

### Strengths
S1 (Problem Formulation):
Clearly defines a new problem—bilateral distribution compression—that reduces both the number of samples and feature dimensions. This formulation fills a gap between dataset condensation and dimensionality reduction.

S2 (Theoretical Guarantees):
Provides rigorous proofs showing that minimizing RMMD and EMMD ensures small or vanishing DMMD, establishing a solid theoretical foundation. The link to PCA under the quadratic kernel further supports correctness and interpretability.

S3 (Clarity and Presentation):
The paper is clearly structured and easy to follow. Key concepts and notation are introduced systematically, with intuitive explanations and helpful figures to illustrate the two-stage optimization idea.

### Weaknesses
W1 (Comparison to Sequential Baselines):
The paper does not clearly demonstrate how BDC improves over a simple two-step approach that first performs Kernel Herding (or other MMD-based compression) followed by dimensionality reduction (e.g., PCA or autoencoder). The advantage of the proposed integrated formulation remains conceptually underexplained.

W2 (Ablation and Sensitivity):
The analysis of hyperparameters—such as latent dimension p, compressed size m, and kernel type—is limited. More systematic ablation studies would help clarify robustness and the contribution of each component.

### Questions
Q1 (Regarding W1 – Comparison to Sequential Baselines):
How does BDC fundamentally differ from a sequential approach that first applies Kernel Herding (or similar MMD-based subset selection) and then performs dimensionality reduction with PCA or an autoencoder?
Specifically, does the proposed two-stage optimization provide any theoretical or empirical advantage beyond this straightforward pipeline?

Q2 (Regarding W2 – Ablation and Sensitivity):
Could the authors provide more analysis on how the choice of latent dimension p, compressed size m, and kernel type affects both reconstruction fidelity and downstream task performance?
For instance, is there a systematic way to choose these parameters, or are the results highly sensitive to them?

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
This paper introduces Bilateral Distribution Compression (BDC), a framework that simultaneously reduces both the number of samples and the dimensionality of a dataset while preserving its underlying distribution. The method combines ideas from kernel-based distribution matching (MMD) and autoencoding. It defines a new measure, the Decoded MMD (DMMD), which quantifies how well a compressed latent set (after decoding) approximates the original data distribution. The paper proposes a two-stage optimization: first, train an autoencoder using Reconstruction MMD (RMMD); second, optimize a compressed latent set using Encoded MMD (EMMD). The authors provide theoretical guarantees linking RMMD, EMMD, and DMMD, and show empirical results on synthetic and real datasets (e.g., Swiss-Roll, CT-Slice, MNIST) demonstrating improved or comparable performance to existing methods (e.g., ADC, M3D) at lower cost.

### Strengths
S1. The paper is well-motivated and addresses a relevant problem: compressing datasets in both sample and feature dimensions simultaneously.

S2. The method is conceptually elegant, bridging kernel-based distribution matching with representation learning.

S3. The theoretical analysis (linking RMMD, EMMD, and DMMD) provides clear intuition and guarantees for the proposed two-stage approach.

S4. The empirical results are diverse, spanning regression, classification, and clustering, with consistent improvements in both runtime and accuracy over strong baselines.

S5. The approach is domain-agnostic and task-agnostic, potentially useful beyond the demonstrated applications.

### Weaknesses
W1. The experimental section is somewhat limited: while results are shown on a few datasets, they are relatively small and mostly standard (swiss-roll, mnist, ce-slice). Evaluation on larger-scale or higher-dimensional modern datasets would strengthen the empirical case.

W2. There is no ablation study on key hyperparameters such as latent dimension, kernel choice, or autoencoder architecture. These factors likely affect the trade-off between reconstruction fidelity and compression rate.

W3. Can you compare withmore  recent strong baselines in dataset condensation and generative compression (e.g., diffusion-based or adversarial distillation methods)?

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework called Bilateral Distribution Compression (BDC) to address the problem of modern datasets being simultaneously large in both sample size ($n$) and dimensionality ($d$)1111. The authors introduce the Decoded Maximum Mean Discrepancy (DMMD) as a core metric to quantify the difference between the original data distribution and the compressed set after being decoded from a low-dimensional latent space.

### Strengths
1.Originality & Significance: The paper formulates a novel and highly significant problem. Most methods address either sample reduction or dimensionality reduction; tackling both simultaneously is a key challenge for modern, large-scale datasets. The proposed BDC framework is a practical, scalable solution with linear time/memory complexity ($O(nd)$), which is a major advantage.

2.Quality (Theory & Empirical): The method is well-grounded. The introduction of the three MMD variants clearly defines the objectives. The two-stage optimization is a logical decoupling of the problem, justified by sound theoretical bounds.

### Weaknesses
1.The method's efficacy relies heavily on the manifold hypothesis. If the data lacks a low-dimensional structure, BDC's forced dimensionality reduction can underperform ambient-space-only methods (ADC)

2.The choice of the latent dimension $p$ is purely heuristic and lacks principled guidance.

### Questions
1.Regarding the choice of $p$: Given that the choice of $p$ is critical to performance, can the authors provide a more principled strategy for selecting $p$? For example, could the RMMD-vs-$p$ curves be used to automatically find an 'Elbow point' for $p$?

2.Regarding the $RMMD+MSRE$ hybrid objective: In BDC-NL, how was the relative weighting between RMMD and MSRE set? How sensitive is the downstream task performance to this weight?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Bilateral Distribution Compression (BDC), a framework that jointly reduces both the sample size and the feature dimensionality of large datasets while preserving their underlying distribution. The method builds upon kernel-based distribution matching using Maximum Mean Discrepancy (MMD) and introduces a two-stage optimization:
(1) learning a low-dimensional representation via minimizing Reconstruction MMD (RMMD), and
(2) optimizing a latent compressed set via Encoded MMD (EMMD).

### Strengths
The idea of bilateral compression:  reducing both data size and dimensionality simultaneously.   is conceptually elegant and practically useful, extending traditional dataset condensation methods.

The paper provides theoretical guarantees linking RMMD, EMMD, and DMMD, giving the approach a clear probabilistic interpretation.

BDC is task-agnostic and can handle both supervised and unsupervised data. It’s applicable beyond specific domains like images.

The experiments, though modest in scale, demonstrate the method’s efficiency and ability to preserve downstream performance across regression, classification, and clustering tasks.

### Weaknesses
The framework mainly reorganizes existing MMD techniques (RMMD/EMMD/DMMD) in a two-stage pipeline; the novelty is more in formulation than in fundamental methodology.

Most experiments use relatively simple datasets (Swiss Roll, MNIST) and small-scale setups. It is unclear whether the method scales to large, high-dimensional real-world datasets (e.g., ImageNet or tabular domains).

Comparisons are limited to kernel MMD and ADC methods; more modern or neural-based data condensation approaches (e.g., Diffusion-based dataset distillation) are missing.

### Questions
How sensitive is the final performance to the choice of kernel functions (e.g., Gaussian vs. polynomial) in RMMD/EMMD?

How does BDC scale when applied to datasets with millions of samples or high-dimensional image data?

Could the method be integrated with diffusion-based or transformer-based representation learners instead of simple autoencoders?

In supervised settings, how are label distributions preserved during compression, especially for imbalanced datasets?

### Soundness
2

### Presentation
2

### Contribution
2
