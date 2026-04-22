# Self-Supervised Disentanglement via Cluster-Dependent Rotational Equivariance

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
Conventional self-supervised learning methods extract robust features by enforcing invariance to data augmentations. While effective for obtaining clustered representations, this objective provides limited control over how data variations structure the feature space, hindering disentanglement. Recent methods improve feature space structure by imposing equivariant predictability on feature transformations induced by data augmentations. However, existing approaches suffer from two significant limitations: (i) the incorporation of invariance in their final objective interferes with the learning of neat equivariance; (ii) the imposition of uniform equivariance across all samples forces semantic clusters into a parallel arrangement, leading to reduced inter-cluster distances (for features on the hypersphere). To overcome these issues, we propose in this paper Cluster-Dependent Rotational Equivariance for Disentanglement (CD-RED), a framework that enables learning neat equivariance and uniformly distributed clusters, while further supporting perfect disentanglement. Notably, CD-RED explicitly encodes variations as rotations via a direct product of $SO(2)$ groups within orthogonal hyperspherical subspaces, providing a principled mechanism for precise equivariance. We theoretically and experimentally establish that CD-RED achieves perfectly disentangled representations, suggesting a promising new direction for self-supervised disentanglement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to achieve self-supervised disentanglement of content and style factors in images without labels by enforcing geometric constraints on learned representations. Specifically, it proposes a method that learns features that form content-invariant clusters and conforms to style-equivariant rotations. Under assumptions that augmentations affect only one style factor at a time and leave content invariant, CD-RED provably recovers content and style subspaces up to rotation. Experiments on synthetic datasets show that the proposed approach can achieve near perfect disentanglement after post-processing, surpassing baselines.

### Strengths
1. The proposed approach unifies content invariance and style equivariance into a single self-supervised pipeline, hence it can in principle be built atop existing SSL backbones (SimCLR, MoCo,...) with minimal architectural changes.
2. It introduces a clean geometric model of disentanglement by representing each style factor as rotations in 2-D subspaces.
3. Within controlled, well-specified environments, the method consistently yields well-disentangled features.

### Weaknesses
- The proposed approach makes several strong assumptions that are hard to satisfy off the lab bench:
    1. The numbers of style factors are known a priori.
    2. $L_\theta$ assumes access to a user-provided proxy that bounds the true amount an augmentation moves the underlying latent.
    3. Augmentations change only the style but never content.
    4. User essentially has access to the ground-truth data generating process, being able to intervene the styles of images through augmentations while knowing the exact magnitude of the intervention.

- The strong disentanglement results rely on a hand-crafted post-processing stage. This leads to unfair comparisons with baselines, which do not leverage post-processing. For example, the post-processing includes concatenation of a one-hot content code, which can boosts completeness and disentanglement score.

- Experiments are only on synthetic datasets such as Shapes3D, MPI3D, 3DIdent, where they use the ground-truth latent transformations as augmentations (with the exact parameter 𝑡). This is impractical for the real self-supervised regime in the wild, where estimating 𝑡 and $q_a(t)$ is difficult. There’s no experimental evidence on natural images (ImageNet-like) or real augmentations with unknown latent magnitudes (e.g., crop, jitter).

### Questions
1. What happens if only a subset of ground-truth style augmentation can be intervened? For example, what if a dataset primarily contains objects from several viewpoints, and viewpoint (as a style latent) can not be intervened?
2. How are the hyperparameters chosen? Could you show a sensitivity analysis?
3. How does CD-RED perform on natural images with standard SSL augmentations?
4. Without appending the one-hot cluster code, how close do D/C get after the same post-processing?

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
- The paper introduces CD-RED, a self-supervised framework for learning disentangled and equivariant representations without labels.
- CD-RED addresses this via a two-stage training pipeline: Stage 1: Contrastive (InfoNCE) training to obtain semantic clusters with uniform hyperspherical distribution; Stage 2: Cluster-wise equivariance learning using explicit SO(2) block rotations within orthogonal subspaces, centered at cluster centroids and aligned via Householder transformations.
- Theoretically, the authors prove equivariance and strong disentanglement under mild conditions, supported by clear geometric reasoning.
- Empirically, CD-RED achieves near-perfect DCI scores on MPI3D, Shape3D, 3DIdent, and 3DIEBench, outperforming SimCLR, EquiMOD, and CARE.

### Strengths
- The paper is logically structured and clear geometric intuition (rotations on hypersphere) nicely complement theoretical intuitions. 
- Clean geometric design with explicit group structure where augmentations are encoded as a direct product of $SO(2)$ planes.

### Weaknesses
- The work assumes access to augmentation parameters (or good proxies) to drive the loss. In real SSL pipelines (random crops, color jitter, CutOut), reliable $q_{a(t)}$ may not exist or may be weakly correlated with semantic style change (e.g., crop boxes can be semantically neutral or catastrophic depending on content).
- The paper “specifies” which dimensions each augmentation acts on rather than learning the subspace; this is robust on synthetic datasets but could fail for real augmentations which often co-vary multiple unknown factors
- The method is beautifully matched to geometric transforms (rot/shift) that act like circle motions. Photometric transforms amongst others (e.g., color jitter, noise, blur) do not naturally map to SO(2) with clean anchors. 
- Empirical scope is limited to synthetic datasets (MPI3D, Shape3D, etc.); real-world or high-dimensional visual domains (ImageNet, CIFAR100, etc.) are not tested. This leaves open questions about scalability and robustness to complex augmentations. 
- The “perfect disentanglement” claim is heavily theoretical and relies on ideal cluster assignments and clean augmentation-factor separability. Performance under imperfect clustering would be interesting to study
- The assumption that augmentations can be neatly mapped to orthogonal 2D planes may not generalize to non-orthogonal or stochastic transformations (e.g., blur + rotation).

### Questions
See weaknesses above

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Cluster-Dependent Rotational Equivariance for Disentanglement (CD-RED), a self-supervised framework designed to achieve perfect disentanglement of latent factors without labels.
The method builds on the observation that existing self-supervised and equivariant learning frameworks (e.g., SimCLR and CARE) either lack precise geometric control or impose global equivariance that limits cluster separation.

CD-RED introduces two key innovations:

> A two-stage training scheme that decouples invariance (via InfoNCE-based clustering) from equivariance (via local rotation learning).

> A cluster-dependent rotational model that encodes data variations explicitly as products of SO(2) rotations in orthogonal hyperspherical subspaces.

Theoretical proofs establish that CD-RED achieves “perfect equivariance,” and consequently, “perfect disentanglement” under mild assumptions.
Empirically, the method achieves near-perfect DCI (Disentanglement/Completeness/Informativeness) scores across both discrete and continuous latent datasets (i.e., MPI3D, Shape3D, 3DIdent, and 3DIEBench) outperforming prior self-supervised baselines such as EquiMOD and CARE by large margins.
Visual analyses (Figures 3, 13–15) demonstrate interpretable, axis-aligned subspaces that correspond neatly to independent generative factors (e.g., translation, rotation, hue).

### Strengths
+ Originality: Introduces the first explicit cluster-dependent rotational system and two-stage self-supervised disentanglement framework.

+ Rigor: Comprehensive theoretical grounding with formal proofs, linking equivariance and disentanglement in a provable manner.

+ Empirical quality: Extensive evaluation on four benchmark datasets, including both synthetic and realistic 3D settings, with DCI ≈ 1.0 across the board (Tables 1 & 4–12).

+ Clarity: Figures 1–3 effectively contrast SimCLR, CARE, and CD-RED, showing superior geometric structure.

+ Significance: Demonstrates that perfect disentanglement is achievable without supervision, a long-standing challenge in representation learning.

### Weaknesses
- Generality: Current evaluation is confined to controlled 3D datasets with clean augmentation-latent correspondences. Real-world visual domains or partial symmetries are not tested.

- Computational complexity: The two-stage pipeline (InfoNCE + rotational training) and per-cluster SO(2) alignment may scale poorly with large-scale or high-dimensional datasets.

- Accessibility of proofs: Some theoretical sections (Appendix C) are exceedingly detailed and might hinder comprehension without accompanying intuition.

- Empirical baselines: While strong, the comparison set omits recent group-equivariant self-supervised models (e.g., SE(3)-Transformers, GroupVAE variants), which could contextualize the method’s advantages beyond CARE and EquiMOD.

### Questions
> Scalability: How does CD-RED perform with hundreds of semantic clusters or higher-dimensional representations (e.g., d > 512)?

> Robustness: What happens if augmentations are imperfect or stochastic (e.g., illumination, blur)? Does CD-RED remain stable when equivariance assumptions are only approximately valid?

> Continuous vs. discrete factors: Can the framework adaptively infer the number of independent rotational subspaces without prior knowledge of m?

> Extension to SE(3): Since the method is based on SO(2) subspaces, can it be generalized to 3D rotational groups SO(3) or product groups like SE(3) for spatial tasks?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Cluster-Dependent Rotational Equivariance for Disentanglement (CD‑RED), a self-supervised framework that addresses the limitations of existing methods in learning disentangled representations. CD‑RED overcomes the problems of existing methods by learning cluster-dependent rotational equivariance, explicitly encoding variations as rotations via a direct product of groups in orthogonal hyperspherical subspaces. The method enables neat equivariance, uniformly distributed clusters, and theoretically and experimentally achieves disentangled representations.

### Strengths
1. The paper identifies an important issue: the imposition of uniform equivariance across all samples could reduce inter‑cluster distances for features on the hypersphere.

2. The theoretical formulation based on rotational SO(2) groups and block‑diagonal equivariant mapping seems solid.

3. On benchmark datasets (Shapes3D, MPI3D, 3DIdent, 3DIEBench), CD‑RED achieves near‑perfect DCI metrics, demonstrating stable disentanglement performance.

### Weaknesses
1. The analysis of related work appears insufficient and somewhat unclear, which weakens the overall contribution of this paper. The proposed method lies at the intersection of self-supervised learning and unsupervised disentangled representation learning; however, the paper overlooks a substantial body of literature on self-supervised clustering methods that aim to learn meaningful cluster structures without relying on invariance to data augmentations. Likewise, numerous studies in disentangled representation learning incorporate conditional invariance constraints, which should be discussed to better position this work within existing research.

2. The experiments are conducted exclusively on synthetic datasets rather than real-world data, which limits the practical validity and generalizability of the proposed method. Evaluating on real-world benchmarks would strengthen the empirical analysis and demonstrate the method’s broader applicability.

3. Although the primary stage employs InfoNCE to establish non-collapsed semantic clusters, this alone may not be sufficient to achieve true disentanglement. Such limitations could potentially lead to clustering errors, which would further affects disentanglement in the later stage, as the clusters are given as conditions.

### Questions
Does the proposed method ensure that errors or biases from the initial InfoNCE-based clustering do not propagate and negatively affect disentanglement in the subsequent stages, given that these clusters serve as conditioning factors?

### Soundness
2

### Presentation
2

### Contribution
2
