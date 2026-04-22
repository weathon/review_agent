# Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Multimodal deep learning (MDL) has achieved remarkable success across various domains, yet its practical deployment is often hindered by incomplete multimodal data. Existing incomplete MDL methods either discard missing modalities, risking the loss of valuable task-relevant information, or recover them, potentially introducing irrelevant noise, leading to the discarding-imputation dilemma. To address this dilemma, in this paper, we propose DyMo, a new inference-time dynamic modality selection framework that adaptively identifies and integrates reliable recovered modalities, fully exploring task-relevant information beyond the conventional discard-or-impute paradigm. Central to DyMo is a novel selection algorithm that maximizes multimodal task-relevant information for each test sample. Since direct estimation of such information at test time is intractable due to the unknown data distribution, we theoretically establish a connection between information and the task loss, which we compute at inference time as a tractable proxy. Building on this, a novel principled reward function is proposed to guide modality selection. In addition, we design a flexible multimodal network architecture compatible with arbitrary modality combinations, alongside a tailored training strategy for robust representation learning. Extensive experiments on diverse natural and medical image datasets show that DyMo significantly outperforms state-of-the-art incomplete/dynamic MDL methods across various missing-data scenarios. Our code is available at https://github.com//siyi-wind/DyMo.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
**Summary**
To address the discard-fill dilemma faced by multimodal deep learning (MDL) in real-world scenarios due to modality loss， discarding the missing modality easily loses task-critical information, while filling the modality easily introduces low-fidelity/semantic misalignment noise. This paper proposes DyMo, a dynamic modality selection framework for inference, which aims to balance the utilization of missing modal information and noise avoidance.

### Strengths
**Strength**
1. The motivation is clear and convincing
2. The dataset and tasks are comprehensive. It covers five significantly different datasets, digital classification, attribute classification, disease diagnosis, and missing scenarios, fully verifying DyMo's adaptability in different scenarios.

### Weaknesses
**Weakness**
1. The primary concern is whether the noise in cross-modal generated data actually harms, especially with the advancement of diffusion model generation technology in recent years. The author should discuss and compare his work with the related recovery-based methods[1][2][3].

2. The comparison method is a bit outdated, and the author should consider comparing it with more 2024s methods of incomplete multimodal learning. 

3. The experiments should contain some classical multimodal datasets, such as CMU-MOSI or CREMAD, for a fair comparison. especially the CMU-MOSI, in which the text is the dominant modality, when it misses the performance degree. This can help to verify the effectiveness of the DyMo.

4.The method requires the label information to select the modality, but when the modality is missing, the prediction will be very incorrect. How can the performance of the model be guaranteed? The representation prototypes also cannot avoid the problem of representations being misclassified.



[1] Yuanzhi Wang.  Incomplete Multimodality-Diffused Emotion Recognition NIPS
[2] S Wei. Mmanet: Margin-aware distillation and modality-aware regularization for incomplete multimodal learning CVPR
[3] Yuntao Shou. GSDNet: Revisiting Incomplete Multimodality-Diffusion Emotion Recognition from the Perspective of Graph Spectrum

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a Dynamic Modality selection method, DyMo, for the missing modality machine learning setting. The key insight is that existing approaches either discard missing modalities (losing valuable information) or recover all missing modalities (potentially introducing noise), creating a fundamental trade-off the authors refer to the discarding-impuattion dilemma. DyMo overcomes this by adaptively selecting only the reliable recovered modalities that provide task-relevant information for each test sample.​

- Core Innovation: A Multimodal Task-Relevant Information Reward (MTIR) function that estimates incremental information gain from each recovered modality using task loss as a tractable proxy​

- Theoretical connection: MTIR is based in a connection between mutual information I(Y;Z) and classification loss, enabling inference-time selection without ground-truth labels​

- Evaluation: Extensive experiments on 5 datasets (PolyMNIST, MST, CelebA, DVM, UKBB) showing significant improvements, especially under severe missing scenarios​

### Strengths
- Strong Problem Formulation: This work substantiates discarding-imputation dilemma in incomplete multimodal learning providing a strong motivation for DyMo

- Comprehensive Technical Design: MTIR reward function handles both low-fidelity and semantically misaligned recovered modalities​ (e.g. when the image is blurry or the recovered image is not representative of the class)

- Strong additional features: Intra-class similarity calibration enhances reward reliability​, iterative modality selection to minimize noise, incomplete simulation training, auxiliary contrastive loss tested with 2 distance functions

- Extensive Experimental Validation: Evaluation across diverse domains (natural images, medical data, synthetic benchmarks)​, Consistent improvements over 12 baseline methods​, particularly strong performance under severe missing scenarios (e.g., 13.12% improvement on PolyMNIST with 80% missing modalities)​

- Thorough Analysis: Comprehensive ablation studies and visualization analyses (Figure 4 with the TSNE and PCA visualizations were particularly convincing that DyMo’s MNITR successfully adds recovered modality features when helpful and does not use them when it would hurt performance.

### Weaknesses
- Limited Recovery Method Diversity: While claiming generalizability, experiments primarily use VAE-based recovery methods (MoPoE, MMVAE+, CMVAE from Table S5) with limited evaluation of fundamentally different recovery approaches​

- Computational Overhead: The authors claim DyMo introduces minimal additional parameters and relies on a relatively simple training scheme. However, it seems give the features of the method including computing the MITR, which includes intra-class similarity calibration, and with iterative selection (average 1.38 iterations per sample from  appendix C.3), DyMo would be more computationally intensive. The inference-time latency, parameter count, or training time computational cost were not thoroughly analyzed​ in this work.

- Calibration Term Limitations: The intra-class similarity calibration shows inconsistent benefits across datasets (improves some tasks but hurts CAD/Infarction performance), suggesting the approach may not be universally optimal​

- Limited Analysis of Edge Cases and limitations: Insufficient discussion of common failure modes or limitations of when DyMo performs poorly or what assumptions DyMo makes.

### Questions
- Recovery Method Dependencies: How sensitive is DyMo's performance to the quality of the underlying recovery method? Could you provide analysis on performance degradation when recovery methods produce consistently low-quality reconstructions? For example, if the recovery method always produces noise how would DyMo perform. If it produces accurate recoveries 50% of the time, how well would DyMo perform?

- Computational Scalability Concerns: How does the computational overhead scale with the number of modalities and missing patterns? What is the practical upper limit for real-time applications?

- Hyperparameter Sensitivity: The framework introduces several hyperparameters (temperature t=0.1, calibration threshold alhpa). How sensitive is performance to these choices, and how should they be set for new domains?

- Theoretical Limitations: The mutual information lower bound assumes bounded loss values with conservative upper bound G. How is G estimated in practice, and how does this choice affect the bound's tightness?

- Class Imbalance: How does DyMo perform on highly imbalanced datasets where the equal class prior assumption may be violated? Could you provide analysis or modifications for such scenarios?

- Generalization Beyond Classification: While focused on classification, could this approach be extended to other multimodal tasks like regression, generation, or structured prediction? What modifications would be required?

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
This paper tackles incomplete multimodal classification by explicitly framing the discarding–imputation dilemma and proposing DyMo, an inference‑time dynamic modality selection framework. The method recovers missing modalities via any recovery model, then iteratively selects only those recoveries that yield positive task‑relevant information according to a reward grounded in a connection between mutual information and cross entropy, refined with an intra‑class similarity (ICS) calibration, and implemented through a lightweight greedy procedure. The paper pairs this with a transformer‑based multimodal backbone that accepts arbitrary modality subsets and a simple incomplete‑modality simulation training scheme. Extensive experiments on five datasets (natural and medical) show consistent gains, especially at high missing rates.

### Strengths
- This paper perceptively identifies discarding–imputation dilemma in incomplete multimodal learning, which is an interesting and practical research problem.

- Selection criterion grounded in standard MI-CE relations and rendered computable via prototype‑based energies (Eq. 5–7), yielding an interpretable "move‑toward‑the‑prototype" test. 

- Broad empirical coverage with competitive results on five datasets, strong robustness under severe missingness, and ablations demonstrating the contribution of each component. 

- Good practicality: recovery‑method agnostic with positive results across MoPoE/MMVAE+/CMVAE and modest extra inference steps on average.

### Weaknesses
- Prototype posterior in Eq. 5–6 presumes Bregman divergences; the cosine distance used in $\text{DyMo}_c$ lacks that guarantee, and the authors claim that both $\text{DyMo}_c$ and $\text{DyMo}_e$ achieved similar results, indicating that DyMo is robust to the choice of distance metric. However, as shown in the results of Table 1, the performance of  $\text{DyMo}_c$ and $\text{DyMo}_e$ across different settings (datasets/missing rates/metrics) does not seem to be consistent. 

- Sub-optimality of the greedy strategy: The algorithm relies on a greedy approach, selecting at each step the single modality that currently yields the highest reward. This strategy cannot guarantee finding the optimal combination of all missing modalities. There can be scenarios in which adding modality A alone is less beneficial than adding modality B, yet adding the combination of modalities A and C yields far greater gains than adding B alone. The effect of this kind of modality synergy can be significant in different medical imaging modalities. A greedy selection would overlook such combinatorial effects, leading to a sub-optimal subset of modalities.

- The author acknowledges that TIP has an upper limit for full-table reconstruction and uses this to explain why it performs on par with or worse than CONCAT in certain scenarios. However, this means that the "selection strategy" capability becomes strongly constrained by the reconstruction quality bottleneck. I highly recommend to include at least one alternative reconstructor in DVM/UKBB.

### Questions
- Could the authors provide a more in-depth analysis of the choice of distance metric, either theoretical or empirical?

- Please justify the selection of the greedy approach.

- Would the selection mechanism remain effective independent of the chosen reconstructor?

### Soundness
4

### Presentation
3

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
The paper tackles incomplete multimodal classification. It introduces an inference-time dynamic modality selection framework that (i) reconstructs missing modalities with an external recovery model, then (ii) selects only the “reliable” reconstructions to fuse with available modalities. The key idea is a multimodal task-relevant information reward (MTIR) as a proxy for information gain, plus a latent-space metric calibration to guard against low-fidelity reconstructions. Experiments show gains over both recovery-free and recovery-based baselines, especially in severe missingness.

### Strengths
1. Clear problem framing (“discarding–imputation dilemma”) and a practical angle: selecting only useful reconstructions instead of always discarding or always imputing.
2. The method is grounded in an information-theoretic heuristic that links mutual information and cross-entropy
3. The method has flexible architecture and straightforward training recipe. 
4. Experiments cover five datasets and multiple missing-modality regimes. Reported gains are meaningful in high-missingness settings.

### Weaknesses
1. Loose MI–CE bound.
(a) Heuristic bound. The proposed lower bound linking task-relevant information I(Y;Z) to the empirical CE loss is very loose and largely heuristic. The bound involves constants G, yet G can be arbitrarily large since the CE loss is unbounded in practice. As a result, the lower bound can collapse to a small or even meaningless value. Moreover, a reduction in CE loss does not necessarily imply increased mutual information.
(b) Dataset-level rather than per-sample validity.
The high-probability guarantee in the bound applies to the randomness of the training dataset \mathcal{D}, not to individual test samples. It does not provide theoretical support for the per-sample reward computation used during inference without many further assumptions.

2. Dependency on recovery quality.
DyMo’s selection quality depends entirely on the fidelity of recovered modalities. When all reconstructions are poor, the model effectively reverts to observed modalities while still incurring extra inference cost. 

3. Task scope.
The method is evaluated for classification. Since many multimodal applications are detection/segmentation/seq-to-seq, a short discussion of what’s needed to extend MTIR beyond CE classification would strengthen the impact.

### Questions
1. In Section 3.1, could the “dummy token” embeddings still introduce bias into positional encodings if not learned or masked carefully?

2. In Section 3.2, the computation of class prototypes as arithmetic means assumes locally Euclidean and unimodal feature geometry. This assumption may not hold in practice. Could you provide statistics or visualizations showing that class cluster means are representative?

3. Why model intra-class distances with a truncated normal distribution? Heavy-tailed or multi-modal classes may violate it. Have you explored nonparametric alternatives (e.g., kernel density) to avoid this assumption? Is there empirical evidence that ICS meaningfully quantifies the representativeness of z within its class cluster?

### Soundness
2

### Presentation
3

### Contribution
3
