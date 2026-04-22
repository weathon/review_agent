# Beyond Instance-Level Alignment: Dual-Level Optimal Transport for Audio-Text Retrieval

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Cross-modal matching tasks have achieved significant progress, yet remain limited by mini-batch subsampling and scarce labelled data. Existing objectives, such as contrastive losses, focus solely on instance-level alignment and implicitly assume that all feature dimensions contribute equally. Under small batches, this assumption amplifies noise, making alignment signals unstable and biased. We propose DART (Dual-level Alignment via Robust Transport), a framework that augments instance-level alignment with feature-level regularization based on the Unbalanced Wasserstein Distance (UWD). DART constructs reliability-weighted marginals that adaptively reweight channels according to their cross-modal consistency and variance statistics, highlighting stable and informative dimensions while down-weighting noisy or modality-specific ones. From a theoretical perspective, we establish concentration bounds showing that instance-level objectives scale with the maximum distance across presumed aligned pairs, while feature-level objectives are governed by the Frobenius norm of the transport plan. By suppressing unmatched mass and sparsifying the transport plan, DART reduces the effective transport diameter and tightens the bound, yielding greater robustness under small batches. Empirically, DART achieves state-of-the-art retrieval performance on three audio-text benchmarks, with particularly strong gains under scarce labels and small batch sizes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes DART, a framework for cross-modal retrieval that addresses the limitations of purely instance-level alignment under small batch sizes and scarce labels. DART combines the conventional instance-level IOT objective with a novel feature-level regularization based on the UWD. It employs RAM to reweight feature channels based on statistical cues (correlation, variance, kurtosis), guiding the transport plan towards stable semantic dimensions and suppressing noisy ones. The framework demonstrates state-of-the-art results on audio-text benchmarks, and generalizing to image-text retrieval.

### Strengths
1. The introduced feature-level alignment using UWD is new. It tries to address the known weakness of traditional contrastive losses (and instance-level IOT) that treat all feature dimensions equally.
2. DART achieves better performance compared to previous works.
3. Experiments on image-text retrieval shows the generalization ability of the proposed DART.

### Weaknesses
1. The writing of this paper is below ICLR's standard. For example, in Line 137 and 142, ``??'' apprear in the main text instead of the reference to equations, figures, or tables. In Eq. (6), x and y lack explanations.
2. Sec. 2.3, limitations of instance-level IOT lacks of theoretical or experimental evidence to support the claims.
3. Novelty is limited: The idea of learning feature importance (reweighting dimensions) is not new in cross-modal retrieval, as noted by the authors themselves when discussing Luong et al. (2024) . While DART's method uses cross-channel transport and richer statistics, the incremental novelty over existing per-channel weighting schemes is largely due to the expensive UWD machinery, which leads to a major concern about complexity.
4. RAM is based on simple, first-order statistics (cross-modal correlation, variance, and kurt). These static, hand-designed proxies for ``semantic stability'' may be insufficient for highly complex, evolving feature spaces. The reliance on simple statistics makes the mechanism feel more like a heuristic than a deep, learned principle. The ablation also shows that RAM improves performance (Tab 1, DART w/ RAM vs w/o RAM) with limited improvements.
5. Also, it would be interesting to see which part of RAM is most important.
6. Lack of running time, memory comparison.

### Questions
1. See weakness
2. In eq. 10, how to calculate kurt?

### Soundness
2

### Presentation
1

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
This paper presents a dual-level alignment via robust transport (DART) method for audio-text retrieval. The proposed method has been explained in detail and experiments have been condudcted for evaluation.

### Strengths
This proposed method combines both instance-level alignment and feature-level regularization for cross-modal retrieval. Plenty of experiments have been condcuted to answer four key questions.

### Weaknesses
The novelty of the proposed method needs more clarification. I don't think the introduction section gives a clear explanation on the connection between the proposed method and existing ones. Accroding to the Related Work section, channel-level considerations have been made in previous methods. Thus, the contribution of this paper would be not so significant if it only combines both instance and feature-level alignments. Besides, the paper writing still needs improvement. For example, there are ?? at line137/142.

### Questions
According to Table 1, using Beats as audio encoder achieves better performance. Please explain why this configuration was not adopted in following experiments.

### Soundness
2

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
The authors propose a dual-level optimal transport framework that integrates instance-level alignment with feature-level regularization via Unbalanced Wasserstein Distance and Reliability-Aware Marginals, achieving more robust and stable cross-modal (audio–text) retrieval under small-batch and noisy-label conditions.

### Strengths
1. The authors propose a novel and reasonable solution that incorporates instance-level inverse optimal transport and feature-level unbalanced Wasserstein regularization within a dual-level optimal transport framework.
2. Introducing reliability-aware marginals (RAM) to reweight feature channels based on cross-modal statistics is intuitive and effective.

### Weaknesses
1. Lack of Deeper Ablation for RAM Components: The existing ablation experiment table (1) only compares "DART w/ RAM" with "DART w/o RAM". I think more fine-grained ablation experiments should be provided. For example: (1) Use only corr, (2) Use only corr-var, (3) use other combinations (such as weighted sum). Without these experiments, we cannot determine if components like kurtosis are necessary or if the current combination is optimal.
2. Computational scalability concerns: The feature-level OT module introduces a cost matrix of size *d×d*, leading to quadratic complexity in feature dimensionality.   The paper does not provide sufficient discussion on scalability to high-dimensional encoders such as CLIP or BEATs.

### Questions
1. As mentioned in Weakness 1, the reliability score formula corr - var - kurt seems heuristic. Can the authors provide more theoretical or empirical evidence for choosing this specific combination?
2. As noted in Weakness 2, the O(d^2) complexity is a potential threat. Have the authors considered methods to mitigate this issue in high-dimensional settings (e.g., d > 2048)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DART, a framework that enhances cross-modal retrieval by combining instance-level alignment with feature-level regularization based on the Unbalanced Wasserstein Distance. By reweighting embedding channels according to cross-modal consistency, DART suppresses noisy dimensions and stabilizes learning under small batches and scarce labels.

### Strengths
* The paper goes beyond conventional instance-level contrastive alignment by introducing a feature-level regularization mechanism based on UOT and IOT. This dual-level design elegantly captures both instance-wise and dimension-wise consistency, addressing the long-standing assumption that all embedding dimensions are equally informative.

* The authors provide clear theoretical analysis showing how instance-level alignment objectives scale with the maximum distance among aligned pairs, while feature-level regularization scales with the Frobenius norm of the transport plan. This theoretical distinction explains DART’s robustness to noise and its improved generalization under small batch regimes.

### Weaknesses
* The ablation study is poorly written so that it is not clear how each critical design choice is justified.

* In the Introduction, the authors claim that ‘noisy channels tend to incur large transport costs.’ However, no empirical evidence is provided to support this claim, and it remains unclear whether the proposed feature alignment method effectively addresses this issue.

### Questions
The ablation study needs a major revision.

### Soundness
2

### Presentation
1

### Contribution
2
