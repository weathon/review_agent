# Stepwise Feature Learning in Self-Supervised Learning

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Recent advances in self-supervised learning (SSL) have shown remarkable progress in representation learning. However, SSL models often exhibit shortcut learning phenomenon, where they exploit dataset-specific biases rather than learning generalizable features, sometimes leading to severe over-optimization on particular datasets. We present a theoretical framework that analyzes this shortcut learning phenomenon through the lens of $\textit{extent bias}$ and $\textit{amplitude bias}$. By investigating the relations among extent bias, amplitude bias, and learning priorities in SSL, we demonstrate that learning dynamics is fundamentally governed by the dimensional properties and amplitude of features rather than their semantic importance. Our analysis reveals how the eigenvalues of the feature cross-correlation matrix influence which features are learned earlier, providing insights into why models preferentially learn shortcut features over more generalizable features.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper analyzes stepwise feature learning dynamics in SSL through the lens of feature cross-correlation eigenvalues. It introduces the notions of extent bias and amplitude bias. It provides theoretical derivations on toy examples, and experiments on synthetic and semi-synthetic data.

### Strengths
- The paper provides a theoretical exposition of stepwise feature learning in SSL, connecting it to short cut learning.

- In addition to the theoretical analysis, the paper presents synthetic experiments in more general settings as well as evaluations on semi-synthetic image datasets.

### Weaknesses
- The definition of extent bias is conceptually vague. The theoretical example in line 160 does not clearly distinguish “extent” from “amplitude”. The example effectively has just two 1-dimensional features (along different directions), but different magnitudes (one of size $\sqrt{m_l}$ and the other of size $\sqrt{m_s}$). Hence the analysis effectively reduces to showing that features with larger magnitudes are learned earlier.

- Following the previous point, both examples essentially show that features with larger magnitudes are learned earlier which is rather obvious, while offering little additional insight.

- Overall, I don’t see what new insights or significant practical implications the paper offers. From the theoretical side, the claimed novelty over Simon et al. (2023) is limited. The paper largely shows the same stepwise dynamics for SSL but just on two specific toy examples, without introducing fundamentally new mechanisms or insights. Additionally, theoretical analysis on how features with larger magnitudes and higher dimensions can suppress the learning of other features in SSL already appears in earlier works such as [1]. On the experimental side, the results only confirm the stepwise phenomenon. It’s unclear what specific new guidance for practice can be drawn from this study. Overall, the work seems to lack theoretical depth, novelty, and practical significance.

[1] Xue, Yihao, et al. "Which features are learnt by contrastive learning? on the role of simplicity bias in class collapse and feature suppression." International Conference on Machine Learning. PMLR, 2023.

### Questions
Please see the questions raised in the Weaknesses section.

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
4

### Summary
The paper analyzes self-supervised learning (SSL) algorithms through the theoretical framework introduced by Simon et al. (2023). This framework employs simplified toy one-layer linear models to examine the behavior of SSL objectives—particularly the Barlow Twins (BT) loss.

Within this setup, the authors first explore extent bias, aiming to understand how feature learning in SSL is influenced more by dimensional properties than by semantic content. They design a toy dataset in which each input is a concatenation of constant vectors of varying dimensionalities, randomly modulated in amplitude. Using the analytical tools developed by Simon et al. (2023), they study the temporal learning dynamics and show that the dimensional scale plays a key role in shaping learning behavior.

The second part of the study investigates amplitude bias, referring to the tendency of networks to favor low-frequency information during training. In this case, the inputs are modeled as superpositions of randomly weighted high- and low-frequency cosine signals. Applying the same analytical framework, the authors demonstrate that the learning dynamics depend primarily on the amplitude of spectral components rather than their frequency.

In the following sections, the paper examines the effect of the redundancy reduction coefficient of BT loss and elaborates on the extension of  the analysis to nonlinear networks. Finally, numerical experiments on the Colored MNIST and Waterbirds datasets are presented to support the theoretical findings.

### Strengths
- The authors present clear problem statements and theorems, supported by sound proofs in the Appendix, for both extent bias and amplitude bias. These analyses, based on linear toy models, are insightful and convincing. The fact that the findings are further validated through experiments on real data enhances their value.

- The article is well-presented and easy, even pleasant, to read.

- Overall, this work represents a valuable extension of prior research. Both extent bias and frequency (amplitude) bias have been recognized in the literature as important issues, and this paper provides additional insights by applying the analytical framework of Simon et al. (2023).

### Weaknesses
- The main limitation lies in the simplicity of the toy linear or single-layer model used in the analysis.

- More comprehensive experiments with real-world datasets would further strengthen the empirical validity of the presented analysis results.

### Questions
- What is the dependence of the $\Gamma$ matrix on the chosen data augmentations?

- Could you elaborate on the generalizability of the observations to other SSL algorithms? The derivations are based on the Barlow Twins loss, but can the underlying intuitions about common features of algorithms/losses to extend results to other losses or algorithmic designs?

- Regarding the extent vs. semantic impact: would it be possible to conduct controlled experiments with manipulated data (e.g., keeping the objects fixed while modifying the background—by expanding, changing texture, etc.) to test the influence of these properties?

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
This paper provides a theoretical framework analyzing shortcut learning in self-supervised learning (SSL) through the lens of eigenvalue decomposition of feature cross-correlation matrices. The authors introduce two concepts: extent bias (prioritizing features based on dimensional coverage) and amplitude bias (prioritizing features based on magnitude). Building on Simon et al. (2023)'s work on stepwise learning dynamics, they demonstrate that learning priority is fundamentally governed by eigenvalues of the feature cross-correlation matrix rather than semantic importance. The theoretical analysis is validated through toy models (linear networks, MLPs) and extended to semi-realistic datasets (Colored-MNIST, Modified Waterbirds).

### Strengths
Rigorous theoretical framework: The eigenvalue decomposition analysis (Theorems 4.1, 4.2, 4.3, 5.1) provides precise mathematical characterization of when features are learned, with critical time points τⱼ ∝ 1/γⱼ clearly derived.

Clear experimental validation: Figure 1 demonstrates excellent alignment between theoretical predictions (dashed lines) and empirical results (solid lines) for loss, eigenvalues, and feature alignment evolution.

Comprehensive scope: Analysis extends beyond basic Barlow Twins to multiple SSL methods (SimCLR, VICReg - Section C), network architectures (linear, DLN, MLP - Sections B.1-B.3), and the redundancy reduction coefficient λ (Section 6.1, Figure 2).

Novel formalization: Extent bias and amplitude bias provide useful conceptual frameworks. The connection between feature dimensionality (mₗ vs mₛ) and eigenvalue magnitude (γₗ = mₗ > γₛ = mₛ, Theorem 4.1) is elegantly established.

Some empirical validation: Colored-MNIST experiments (Section 7.1, Figure 5) show the plateau at 70% accuracy directly validates the extent bias hypothesis in a controlled setting with varying object ratios.

### Weaknesses
Limited novelty of core insight: The observation that models learn high-dimensional/high-amplitude features first is well-established in the spectral bias literature (Rahaman et al. 2019, Tancik et al. 2020 - cited by authors) and also formulated as various other names (easy-to-learn, low-variance features etc.) in the literature. The main contribution is formalizing this specifically for SSL eigenvalue dynamics.

Limited actionable insights: While the paper explains why shortcut learning occurs, it offers minimal guidance on how to mitigate it. The conclusion mentions "designing mechanisms to encourage learning of generalizable features" but provides no concrete methods.
Experimental scope:

Modified Waterbirds experiments (Section 7.2) are interesting but only briefly described in appendix.
No experiments on standard SSL benchmarks (ImageNet pretraining + downstream tasks) to assess real-world impact.
The 70% plateau observation in Figure 5 is compelling but limited to artificially constructed spurious correlations.

### Questions
What do the authors think the key takeaway from this work should be? 

Is the goal to provide a theoretical framework for future analysis or does this yield some clear empirical insights for mitigating or discovering spurious correlations in practice already?

### Soundness
3

### Presentation
3

### Contribution
3
