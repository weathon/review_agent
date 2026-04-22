# LCA: Local Classifier Alignment for Continual Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 2

## Abstract
A fundamental requirement for intelligent systems is the ability to learn continuously under changing environments. However, models trained in this regime often suffer from catastrophic forgetting. Leveraging pre-trained models has recently emerged as a promising solution, since their generalized feature extractors enable faster and more robust adaptation. While some earlier works mitigate forgetting by fine-tuning only on the first task, this approach quickly deteriorates as the number of tasks grows and the data distributions diverge. More recent research instead seeks to consolidate task knowledge into a unified backbone, or adapting the backbone as new tasks arrive. However, such approaches may create a (potential) \textit{mismatch} between task-specific classifiers and the adapted backbone. To address this issue, we propose a novel \textit{Local Classifier Alignment} (LCA) loss to better align the classifier with backbone. Theoretically, we show that this LCA loss can enable the classifier to not only generalize well for all observed tasks, but also improve robustness. Furthermore, we develop a complete solution for continual learning, following the model merging approach and using LCA. Extensive experiments on several standard benchmarks demonstrate that our method often achieves leading performance, sometimes surpasses the state-of-the-art methods with a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new loss function, called Local Classifier Alignment (LCA), which aims to address the potential mismatch between task-specific Gaussian-based classifiers and the continuously updated pre-trained models (PTMs, e.g., ViT) within the framework of PTM-based continual incremental learning (CIL). The paper adopts a model-merging approach for CIL, in which the PTM parameters fine-tuned on task-specific datasets are incrementally aggregated. Such incremental updates can lead to inconsistencies with previously trained (and typically frozen) task-specific classifiers — a problem that LCA is designed to mitigate. The key idea behind LCA is to minimize not only the class-wise loss but also the sensitivity of the loss to small perturbations in input samples. Through this simple yet effective regularization, LCA demonstrates clear improvements in CIL performance, along with enhanced robustness and stability across experiments.

### Strengths
1. This is a solid and well-motivated paper with a clear understanding of the underlying problem in PTM-based CIL.
2. The application of incremental model merging to continual learning appears to be novel and is one of the key contributions of this work.
3. The core idea of LCA is both insightful and elegant, achieving simplicity without unnecessary technical complexity.
4. LCA is evaluated comprehensively, both in terms of performance and robustness, across seven benchmarks, and its theoretical foundation is supported by a probabilistic guarantee.

### Weaknesses
1. While the paper is generally well-written, certain parts require improvement. In particular, Section 3.3 should be rewritten in direct connection with Section 3.1. The current version introduces new notations (e.g., C_t, x') without prior definition and refers to Gaussian-based classifiers (e.g., N_i) without adequate explanation.
2. Section 3.2 lacks novelty. Although applying model merging to continual learning may be a new attempt, the overall algorithmic formulation appears nearly identical to TIES-Merging, as also acknowledged by the authors.
3. Model merging itself is not sufficiently described in Section 3.2, making the paper somewhat less self-contained.
4. The paper does not provide an analysis of performance variation across different dataset characteristics. For example, model merging seems less effective on certain datasets (e.g., CUB, OB).

### Questions
1. What is the reason behind the relatively low performance on OB or CUB datasets?
2. Why is the OB graph missing in Figure 2?
3. Are there any prior studies that apply model merging to continual learning? If not, do you explicitly clarify that this is the first attempt at incremental model merging for CIL?
4. What exactly is the model architecture? Does it consist of a shared PTM backbone and multiple MLPs (one per task), where each MLP corresponds to multiple Gaussian components (one per class)?

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
4

### Summary
This paper addresses the critical issue of backbone-classifier mismatch in Pre-trained Model (PTM) based Class-Incremental Learning (CIL), where an evolving backbone diverges from fixed classifiers of previous tasks. The proposed solution consists of two components:

Incremental Merging (IM): Progressively merges PEFT modules (using LoRA) to create a unified backbone.

Local Classifier Alignment (LCA): A novel loss function applied after IM to align all classifiers (past and present) with the merged backbone. LCA approximates class features with Gaussian distributions and minimizes both classification loss and a local robustness term (sensitivity penalty) using sampled features.

The combined approach (IM+LCA) demonstrates strong performance, matching or exceeding state-of-the-art results on seven CIL benchmarks, with notable improvements in robustness evaluations (CIFAR100-C/P).

### Strengths
[S1] The paper clearly identifies and addresses a critical practical problem in PTM-based CIL methods: backbone-classifier misalignment that occurs when incrementally updating the backbone. This is a timely and important contribution to the CIL field.

[S2] Robustness and Generality: Demonstrates enhanced model stability through robustness tests and shows LCA can be complementarily applied to improve other CIL methods, proving its general applicability.

### Weaknesses
[W1] Since the robustness penalty is a core contribution, this ablation is essential to distinguish its specific benefit from the classifier alignment component (first term), which SLCA also employs. Without this experiment, it remains unclear whether the improvements stem from the novel robustness penalty or merely from classifier alignment.
I strongly encourage the authors to include this ablation, as it would clearly demonstrate the added value over existing approaches like SLCA and strengthen the paper's novelty claims.

[W2] The provided Theorem 3.1 assumes fixed class distributions (prototypes), whereas the IM component actively modifies the backbone and thus the feature space. This gap needs better reconciliation or clearer framing (e.g., viewing alignment as optimizing classifiers for a newly fixed merged backbone).

[W3] Cost Concerns (LCA Stage): Requires resampling features and retraining all classifiers after each task, leading to a computational cost that increases linearly with the number of tasks. 

[W4] Missing Related Work: While citing direct competitors (EASE, MOS), the paper misses relevant work on data-free replay for classifier calibration, such as: 
Feature Replay / Calibration: FeTrIL (WACV 2023), PASS (CVPR 2021). Classifier Calibration / Alignment: FOSTER (ECCV 2022).

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a continual learning method based on model merging. Specifically, it alleviates the mismatch between task-specific classifiers and the adapted backbone by a novel Local Classifier Alignment (LCA) loss. LCA keeps the in-task loss less sensitive to a small change in the input samples around the class prototypes. The effectiveness of LCA loss is supported both theoretically and empirically. The paper conducts experiments on multiple CIL datasets. Results show the improved performance compared with various pre-trained based CIL methods. A robustness measurement and ablation study are included.

### Strengths
1. Applying model merging methods to CL is an emerging topic and has potential to reduce forgetting for large scale models. 

2. The effectiveness of the LCA loss is justified both theoretically and experimentally. 

3. The paper conducts thorough experiments with comparison to recent CL baselines. The visualization of results is good.

### Weaknesses
1. It is unclear how LCA loss solves the mismatch between classifiers and the merged feature extractor. 
- The LCA loss is computed only on in-task samples, reducing $\epsilon_i$ in Eq. 4. However, the mismatch between classifier and feature extractor seems aiming to reduce $L(\mathbf D, h_t)$. Although LCA loss improves in-task robustness and reduces the loss upperbound, it’s unclear how it addresses the mismatch across tasks. 
- It could be helpful to show more evidence of the claim ‘LCA can reduce overlapping between classes’ in L220 and how this can reduce negative effect from potentially harmful samples (L224).

### Questions
1. Model merging based on magnitude selection is proposed in MagMax (Marczak et al., 2024) as well. What is the main difference/benefits of IM compared to MagMax?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of class incremental leanring with pretrained model. While pretrained model can act as a good general representation model, they still lack domain specific knowledge and also suffer from the catastrophic forgetting when trained on sequential tasks. To alleviate these problems, the authors propose icremental merging and local classifier alignment method with some theoretical/empirical analysis.

### Strengths
- Paper is well written and easy to follow. Especially, the authors tried to provide multiple visualisations to help the understanding of the readers.
- Instead of just giving empirical evidence, the authors also provide theoretical analysis as well

### Weaknesses
- My major concern about this paper is about novelty. To me, it looks like none of the proposed component is novel not just in entire deep learning literature but also in continual learning literature. For example, classifier alignment was studied even from few years ago as in [R1, R2]. Adopting model merging to continual learning is not a novel idea as well as in [R3]. I can not see the distinct novelty of the proposed method compared to the above cited papers.

- Since the pretrained model is evolving over time, there is no guarantee that the feature space structure is maintained. If so, LCA method and its theoretical analysis can not be justified. 

- Also, improvements in Table 1 is bery marginal, especially when IM is applied alone. What happens if LCA is applied to the existing methods? 

- In Figure2, why the starting point is different? The proposed methods has higher starting point but similar tendency. Due to this, I wonder the improvements comes just from good starting point, not by the proposed methods.

[R1]Zhu, Fei, et al. "Class-incremental learning via dual augmentation." Advances in neural information processing systems 34 (2021)
[R2]Kim, Taehoon, Jaeyoo Park, and Bohyung Han. "Cross-class feature augmentation for class incremental learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 12. 2024.
[R3]Marczak, Daniel, et al. "Magmax: Leveraging model merging for seamless continual learning." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
1
