# Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Class-incremental learning (CIL) in medical image-guided diagnosis requires models to preserve knowledge of historical disease classes while adapting to emerging categories. Pre-trained models (PTMs) with well-generalized features provide a strong foundation, yet most PTM-based CIL strategies, such as prompt tuning, task-specific adapters and model mixtures, rely on increasingly complex designs. While effective in general-domain benchmarks, these methods falter in medical imaging, where low intra-class variability and high inter-domain shifts (from scanners, protocols and institutions) make CIL particularly prone to representation collapse and domain misalignment. Under such conditions, we find that lightweight representation calibration strategies, often dismissed in general-domain CIL for their modest gains, can be remarkably effective for adapting PTMs in medical settings. To this end, we introduce Random Anchors with Low-rank Decorrelated Learning (RA-LDL), a minimalist representation-based framework that combines (a) PTM-based feature extraction with optional ViT-Adapter tuning, (b) feature calibration via frozen Random Anchor projection and a single-session-trained Low-Rank Projection (LRP), and (c) analytical closed-form decorrelated learning. The entire pipeline requires only one training session and minimal task-specific tuning, making it appealing for efficient deployment. Despite its simplicity, RA-LDL achieves consistent and substantial improvements across both general-domain and medical-specific PTMs, and outperforms recent state-of-the-art methods on four diverse medical imaging datasets. These results highlight that minimalist representation recalibration, rather than complex architectural modifications, can unlock the underexplored potential of PTMs in medical CIL. We hope this work establishes a practical and extensible foundation for future research in class-incremental image-guided diagnosis. Code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose a Random Anchors with Low-rank Decorrelated Learning (RA-LDL) approach to address Class-incremental Learning (CIL) in medical image diagnosis. The authors validate their idea with two base models (ViT-B/16-IN21K and BiomedCLIP) and four datasets (Covid, Blood, Skin8, MedMNIST-sub). The gains of RA-LDL are clear.

### Strengths
1. The experiments are comprehensive: the authors include various types of medical images and popular base models. In the comparisons, the authors also include most milestone methods in class incremental learning. All these experiments show a clear gain for RA-LDL.
2. Analytical closed-form decorrelated learning sounds good.

### Weaknesses
1. The method is not novel for the current stage. The main contributions of the paper include a random matrix projection and a LORA-based structure.
2. The author might need to justify more about the combination of two projectors. I understand the proof the author proposed that they aim to random anchor projections  to preserved the original feature characteristics and LRP residual reduces intra-class variance. However, mathemtically, could it possible to simplify the equation 4 into a single LORA-based structure $h(x) =h_{RA}(x) + h_{LRP}(x) \approx h_{LORA}(x)$?

3. The contribution of each projector ($h_{RA}(x)$, $h_{LRP}(x)$) is unclear. How do the authors combine each projection, and what is the weight for each part? Does the weight impact the results, and what is the robustness of the weight selection? These might impact the accuracy gain.

### Questions
1. What is RA-DL in Table 2? It only appears in Table 2 and without explanation.

### Soundness
3

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
The paper addresses class-incremental learning (CIL) in medical image classification, where models must learn new disease categories over time without forgetting previously learned ones. The authors argue that existing PTM-based CIL methods—often complex and designed for natural images—fail in medical settings due to low intra-class variability and high inter-domain shifts (e.g., from different scanners or protocols). In response, they propose RA-LDL, a minimalist, representation-based pipeline with three components:
(a) optional one-time ViT-Adapter tuning for domain adaptation,
(b) feature calibration via a frozen Random Anchor (RA) projection and a single-session-trained Low-Rank Projection (LRP), and
(c) an analytical, closed-form decorrelated classifier based on ridge regression.
RA-LDL requires only one training session, avoids replay or complex routing, and achieves strong performance across four medical CIL benchmarks, often approaching joint-training upper bounds. The method is shown to work robustly with both general-domain and medical-specific pre-trained models (PTMs).

### Strengths
Strengths:
1.The RA-LDL approach is innovative in its simplicity. While other methods for class-incremental learning in medical imaging often rely on complex architectures, this paper proposes a lighter solution that offers competitive performance, especially under domain shift conditions. It effectively merges generalization and adaptability without extensive tuning.
2.By focusing on feature recalibration (using random anchors and low-rank projections), RA-LDL addresses low intra-class variability and high inter-domain shifts directly and effectively. The method shows solid promise in medical domains like COVID-19 diagnostics, blood cell analysis, and skin lesion classification.
3.It only requires one training session with minimal task-specific tuning, making it very practical for real-world medical applications where data is continually evolving.The approach works across various imaging modalities, making it applicable in diverse medical settings.

### Weaknesses
Weaknesses:
1.it could benefit from a more detailed examination of RA-LDL’s long-term stability and its ability to mitigate forgetting. Specifically, investigating the model's ability to retain previously learned classes as more tasks are introduced would strengthen the paper’s argument.
2.The “one training session” claim is slightly misleading: while only the first session trains the LRP, the classifier is updated incrementally using accumulated statistics. This is still efficient, but the phrasing could confuse readers expecting fully frozen inference.
3.How much memory is required to store G and Cp? For large class sets or high-dimensional features, the Gram matrix (d₁×d₁) could become costly. Please report memory usage or discuss scalability.
4.Why not compare to recent replay-free methods? While Table 2 includes some, a more direct comparison to non-PTM-based CIL would strengthen the claim that PTMs + RA-LDL are uniquely effective.

### Questions
Questions:
1.The authors show that general-domain PTMs, like ViT-B/16-IN21K, outperform medical-specific models such as UniMedCLIP or RAD-DINO. However, I have a question whether the improvements are more due to the general-domain PTM itself rather than the RA-LDL recalibration technique. The paper could further clarify how RA-LDL benefits from this general-domain foundation and whether it can achieve similar results when applied to domain-specific PTMs.
2.While the manuscript evaluates the model’s robustness to task ordering, further analysis on how RA-LDL performs in highly unstructured or unpredictable sequences of class introduction (i.e., non-sequential class introduction) could be informative. For example, testing RA-LDL under random or shuffled class orders to assess how well it maintains generalizability when the sequence of tasks is not controlled or predictable.
3.Are there scenarios where RA-LDL underperforms? For example, in highly imbalanced tasks (e.g., Skin8), does decorrelation hurt minority classes?

### Soundness
3

### Presentation
3

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
The paper introduces a minimalist approach for class-incremental learning (CIL) in medical image classification called Random Anchors with Low-rank Decorrelated Learning (RA-LDL). The motivation is to address the unique challenges of medical CIL, which involves low intra-class variability and high inter-domain shifts, where complex pre-trained model (PTM)-based continual learning methods from the general domain often fail. RA-LDL combines a feature extraction method, lightweight feature calibration via a frozen random anchor projection and a single-session-trained Low-Rank Projection, and a closed-form analytical de-correlated classifier based on ridge regression.

### Strengths
The empirical evaluation is thorough, benchmarked across four realistic medical datasets as well as standard general-domain CIL benchmarks, and ablation studies clearly show the contribution of each component.

The work points out an overlooked strategy to CIL that offers better scaling properties/simplicity. It advances methodological understanding by showing that domain-specific challenges in medical continual learning are best addressed with statistical feature recalibration, not greater architectural/training complexity, which is an applicable insight to other fields with similar data properties.

Theoretical justifications for each step (random anchor, LRP, analytical de-correlated classifier) are provided, including proofs of distance and covariance preservation, variance reduction, and prototype de-correlation.

### Weaknesses
While the synergy and systematic evaluation of random anchors, low-rank projections, and de-correlated classifiers are novel for medical CIL, many components are incremental adaptations or well-established in general-domain CIL (Zhuang et al., 2022, McDonnell et al., 2024). More in-depth, quantitative ablation and comparison with these specific prior works could clarify the genuine novelty and highlight improvements beyond re-combination. Some discussion in appendix, but would be nice to see more quantitative evidence.

### Questions
See above.

### Soundness
4

### Presentation
3

### Contribution
3
