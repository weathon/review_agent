# Tracking the Discriminative Axis: Dual Prototypes for Test-Time OOD Detection Under Coveriate Shift

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
For reliable deployment of deep-learning systems, out-of-distribution (OOD) detection is indispensable. In the real world, where test-time inputs often arrive as streaming mixtures of in-distribution (ID) and OOD samples under evolving covariate shifts, OOD samples are domain-constrained and bounded by the environment, and both ID and OOD are jointly affected by the same covariate factors. Existing methods typically assume a stationary ID distribution, but this assumption breaks down in such settings, leading to severe performance degradation. We empirically discover that, even under covariate shift, covariate-shifted ID (csID) and OOD (csOOD) samples remain separable along a discriminative axis in feature space. Building on this observation, we propose DART, a test-time, online OOD detection method that dynamically tracks dual prototypes---one for ID and the other for OOD---to recover the drifting discriminative axis, augmented with multi-layer fusion and flip correction for robustness. Extensive experiments on a wide range of challenging benchmarks, where all datasets are subjected to 15 common corruption types at severity level 5, demonstrate that our method significantly improves performance, yielding 15.32 pp AUROC gain and 49.15 pp FPR@95TPR reduction on ImageNet-C vs. iNaturalist-C compared to established baselines. These results highlight the potential of the test-time discriminative axis tracking for dependable OOD detection in dynamically changing environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel problem setting for out-of-distribution (OOD) detection under
covariate shift and proposes a corresponding test-time OOD detection method specifically
designed for this scenario. The proposed approach maintains two prototypes—one for indistribution (ID) samples and another for OOD samples. It integrates the MSP score with a newly
designed RDS score, and employs the Otsu algorithm to partition each batch into potential ID and
OOD subsets. These subsets are then used to iteratively update their respective prototypes.
Furthermore, the authors analyze how different types of covariate shift manifest across the
model’s feature layers and fuse RDS scores from multiple layers to improve the model’s
robustness and detection reliability.

### Strengths
1. The observation that different types of covariate shift exhibit varying degrees of separability
in the scores computed across different layers of the model is novel and interesting.
2. This paper conducts extensive observations on the separability between in-distribution (ID)
and out-of-distribution (OOD) samples in the feature space, and provides a comprehensive
analysis of how different types of covariate shift affect the RDS scores computed across
various feature layers of the model. Experiments were conducted on multiple models,
demonstrating that the proposed method is model-agnostic.
3. The dynamic score switching mechanism enables the model to better adapt to complex test
environments.

### Weaknesses
1. Figure 4 shows that ID and OOD samples are separable even without considering class
labels. However, the OOD datasets used are all far-OOD. When OOD data are specifically
designed to resemble certain ID classes (i.e., near-OOD), this assumption may no longer
hold. Moreover, the class-agnostic aggregation strategy for ID and OOD prototypes might be
invalid in such a case.
2. The paper uses the Otsu algorithm to determine the optimal threshold. However, when ID
samples significantly outnumber OOD samples, the computed threshold might misclassify
some ID samples as OOD, leading to degraded model performance.
3. It is unclear whether the ID and OOD prototypes are shared across layers or maintained
independently. The paper mentions averaging scores from multiple layers but does not
include ablation studies to assess the contribution of each layer.
4. The paper involves multiple types of covariate shifts, but it is unclear whether results are
based on mixed-shift test sets. Additionally, in test-time setups, ID and OOD samples are
usually mixed, and performance may vary depending on random seeds used for shuffling.5. In Table 1, DART achieves very low FPR95 values on Places-C, SUN-C, and iNaturalist-C, yet
the corresponding AUROC scores are also low — an inconsistent and counterintuitive result.
6. The ID and OOD prototypes are derived from previously observed test data. If the earlier test
OOD samples belong to a single type, the performance may degrade when new or unseen
OOD types appear.
7. Both the empirical observations and the proposed method are largely based on
experimental findings and intuitive explanations, lacking rigorous theoretical analysis or
formal justification.

### Questions
1. Does the proposed assumption still hold under near-OOD conditions? Could you provide
additional experiments on near-OOD datasets such as NINCO [1], SSB-Hard [2], and
ImageNet-X [3] to verify this?
2. How does the proposed method handle class imbalance during threshold computation?
Have you observed performance degradation in scenarios where ID and OOD sample ratios
are highly unbalanced?
3. Are the prototypes shared or layer-specific during the Multi-Layer Score Fusion process?
Could you include ablation experiments comparing high-, mid-, and low-level feature layers
to clarify their individual contributions?
4. Are the reported results obtained from mixed covariate-shift test sets or evaluated per shift
type? Furthermore, how stable is the model’s performance variance across different random
seeds during test-time evaluation?
5. Could you explain the cause of this inconsistency between AUROC and FPR95?
6. How does the performance change when facing new OOD types?
[1] Bitterwolf J, Mueller M, Hein M. In or Out? Fixing Imagenet Out-Of-Distribution Detection
Evaluation[J]. arXiv preprint arXiv:2306.00826, 2023.
[2] Vaze S, Han K, Vedaldi A, et al. Open-Set Recognition: A Good Closed-set Classifier is All You
Need?[J]. 2021.
[3] Noda S, Miyai A, Yu Q, et al. A Benchmark and Evaluation for Real-world Out-Of-Distribution
Detection using Vision-Language Models[J]. arXiv preprint arXiv:2501.18463, 2025

### Soundness
3

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
5

### Summary
This paper addresses the challenging of OOD detection under concurrent covariate shifts, where both ID and OOD data streams are affected by the same environmental changes. The authors first empirically demonstrate a crucial insight: even as distributions drift, ID and OOD samples remain linearly separable along a "discriminative axis" in the feature space. Based on this observation, they propose DART, a test-time, online adaptation method. DART dynamically tracks this drifting axis by maintaining and updating dual prototypes (one for ID, one for OOD) for multiple feature layers. Extensive experiments on severely corrupted versions of CIFAR-100 and ImageNet benchmarks show that DART achieves state-of-the-art performance.

### Strengths
1. The paper formalizes a interesting setting where both ID and OOD data co-evolve under the same covariate shifts. 
2. DART is a simple, yet highly effective algorithm.

### Weaknesses
1. The foundational claim of a “recoverable linear axis” is a strong assumption, lacking rigorous theoretical justification. Although empirical evidence is provided via low-dimensional projections (Figure 4), the paper offers no theoretical basis to explain why such linear separability should persist in high-dimensional feature spaces, especially under complex, non-linear covariate shifts.
2. The authors assume that both ID and OOD data undergo the same covariate shift, which may simplify real-world scenarios. In practice, OOD samples may experience different types of shifts, making the task more challenging. A more realistic evaluation might involve distinguishing csID from near-OOD samples—for instance, using CIFAR-10 as OOD for CIFAR-100.
3. The paper does not discuss the potential impact of cross-domain OOD samples during test time, which could interfere with prototype updates. In real deployments, OOD inputs may originate from diverse domains, possibly leading to unstable prototype estimation and reduced detection robustness.
4. The ablation studies are presented only in qualitative or binary terms (with/without a module), failing to quantify the individual contributions of key components such as prototype tracking, multi-layer fusion, and flip correction.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work empirically shows that, even under covariate shift, covariate-shifted in-distribution (csID) and out-of-distribution (csOOD) samples remain separable along a discriminative axis in feature space. Building on this observation, this work proposes DART, a test-time online OOD detection method that dynamically tracks dual prototypes, one for ID and one for OOD, to recover the drifting discriminative axis, with multi-layer fusion and flip correction for robustness. Extensive experiments across challenging benchmarks, where all datasets are subjected to 15 common corruption types at severity level 5, demonstrate significant performance gains.

### Strengths
1. The paper targets a realistic test-time, streaming OOD scenario under covariate shift and presents a clear, well-motivated problem setup.

2. This work empirically shows that csID and csOOD remain separable along a drifting discriminative axis, and proposes dual-prototype tracking with flip correction and multi-layer fusion, a lightweight, training-free, and technically sound mechanism.

3. Experiments across multiple benchmarks and diverse corruption types demonstrate consistent improvements and robustness.

### Weaknesses
1. Report computational cost for the proposed method (FLOPs, parameters, and memory) and compare against baselines.

2. Evaluate across diverse corruptions and severities, and quantify the quality/effectiveness of the dual prototypes (e.g., axis drift, margin, stability).

3. Analyze sensitivity of multi-layer fusion and flip correction (number/choice of layers, fusion weights/thresholds), with stability curves.

4. Add comprehensive ablations isolating each component, such as window size, EMA momentum, seeding alternatives, and distance metrics.

### Questions
1. Please report computational cost to substantiate the “lightweight” claim, and compare against baselines.

2. Could you quantify the stability of the discriminative axis over time (e.g., angle drift, margin dynamics) and provide diagnostics/visualizations for scenarios where prototypes collapse?

3. Please detail which layers contribute most to fusion effectiveness and how performance varies with layer choices. Also report how frequently flip correction triggers and its impact on accuracy over time.

4. Results emphasize the highest severity; could you provide a breakdown across severities 1–5 and across corruption types?

5. Do you have theoretical support or an intuition for why csID vs. csOOD often remain separable along a dominant axis under covariate shift, and under what conditions this separability may fail?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DART (Discriminative Axis Real-time Tracker), a post-hoc, test-time method for out-of-distribution (OOD) detection under covariate shift. The key idea is that even when both in-distribution (ID) and OOD samples are corrupted (e.g., by blur or noise), their activations in feature space remain approximately linearly separable along a single discriminative direction. DART tracks this direction online by maintaining two evolving prototypes, one for ID and one for OOD, and updating them batch by batch through exponential moving averages. A relative distance score derived from these prototypes serves as the OOD metric. The method requires no retraining or labeled data and operates entirely on model activations, making it practical for deployment.

### Strengths
The paper is well written and the motivation is timely and relevant. The method is conceptually simple, computationally lightweight, and easy to integrate with pretrained networks. The experimental setup uses standard corruption benchmarks (CIFAR-100-C, ImageNet-C) and common evaluation metrics, which are appropriate for evaluating robustness. Results show consistent performance gains over established post-hoc baselines such as ODIN and MSP, indicating that the tracked discriminative axis indeed captures meaningful domain drift. The evaluation includes ablations (multi-layer fusion, flip correction) that demonstrate stable improvements. Overall, the experiments are sound and provide empirical evidence that the method works as intended.

### Weaknesses
Conceptually, DART relies on strong simplifying assumptions. It models the in-distribution as a single unimodal cluster, ignoring the inherently multimodal structure of multi-class ID data. This limits its interpretability and may cause misclassification for semantically similar but unseen classes. The discriminative axis a_t is updated empirically as the normalized difference between ID and OOD prototypes, which serves as a heuristic linear separator without theoretical justification of optimality or convergence. The approach assumes that covariate drift is smooth and linearly trackable, which may not hold for more complex domain shifts or multimodal data. Overall, while the study is well executed and empirically convincing, the approach lacks theoretical grounding and appears somewhat ad hoc in its construction. It would be beneficial to include discussion and examination on more complex scenarios. More discussions are deferred to the Questions section.

### Questions
The paper assumes that “temporally correlated csID and csOOD typically undergo the same covariate shift, so their distributions co-evolve during deployment.” What supports this assumption? Is there empirical evidence or theoretical reasoning showing that ID and OOD data indeed experience identical or synchronized covariate drift?

The experiments rely on CIFAR-C and ImageNet-C, which apply static corruptions (e.g., noise, blur, weather) to test sets. Do these settings genuinely reflect temporal covariate shift as described in the motivation, or are they only approximations of distributional change? If the latter, does the proposed online tracking offer a measurable advantage over one-shot detectors?

The method uses a single, class-agnostic ID prototype. How many ID classes are actually present in the experiments, and how does collapsing them into one centroid affect the results? Would performance degrade if class features overlap or are not linearly separable in activation space?

The approach appears to assume that covariate drift is smooth, shared across ID and OOD samples, and linearly trackable in the feature space. Are there any analyses or ablations testing sensitivity to violations of these assumptions? For example, class-dependent or abrupt shifts?

### Soundness
2

### Presentation
2

### Contribution
2
