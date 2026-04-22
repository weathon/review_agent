# EffSelect: Efficient Feature Value Selection for Deep Recommender Systems with Mini-Batch Training

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Features are critical to the performance of deep recommender systems, where they are typically represented as low-dimensional embeddings and fed into deep networks for prediction. However, a major challenge remains unaddressed: the sparsity and long-tail distribution in feature data result in a large number of non-informative feature values. These redundant values significantly increase memory usage and introduce noise, thereby impairing model performance. Most feature selection or pruning methods operate at a coarse granularity, either selecting entire features or fields, while finer-grained methods require a large number of additional learnable parameters. These methods struggle to effectively handle pervasive redundant features. To address these issues, we introduce EffSelect, a novel framework for finer-grained selection method at the level of feature values. Unlike previous methods, EffSelect directly quantifies the contribution to the prediction loss of each feature value as its importance. Specifically, we propose a mini-batch pre-training strategy that requires only 5% of the data for rapid warm-up, enabling real-time adaptation in dynamic systems. Using the trained model, we introduce an efficient and robust gradient-based mechanism to evaluate feature value contribution, discarding those features with low scores. EffSelect is theoretically guaranteed and achieves superior performance without introducing any additional learnable parameters to the base model. Extensive experiments on benchmark datasets validate the efficiency and effectiveness of EffSelect.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EffSelect, a novel framework for feature value selection in deep recommender systems. The method operates at the feature value level rather than the feature field level, aiming to reduce redundancy and noise in embedding tables without introducing additional learnable parameters. The approach combines a mini-batch pre-training strategy (MFCS) with a gradient-based importance scoring mechanism (FeatIS) to efficiently identify and retain only the most informative feature values. Extensive experiments on four public datasets demonstrate that EffSelect outperforms existing feature selection methods in both performance and efficiency.

### Strengths
1.EffSelect addresses a critical gap by performing feature value-level selection, which is finer-grained than field-level selection and more practical than gate-based value selection methods.
2.The method does not introduce additional learnable parameters, making it lightweight and suitable for dynamic recommender systems.
3.The use of numerical integration and gradient-based importance estimation (FeatIS) is well-motivated and theoretically justified, with error bounds provided in the appendix.
4.Requires only 5% of training data for warm-up, making it highly efficient and suitable for large-scale systems.
5.Experiments are conducted on four diverse datasets and two base models (DCN and MaskNet), with comparisons against a wide range of field-level and value-level baselines.

### Weaknesses
1.The method assumes a static dataset for pre-training and selection. In real-world systems, new feature values may emerge over time. A discussion or experiment on incremental or online adaptation of the feature set would strengthen the practicality claim.
2.The performance is sensitive to α (selection ratio), and the optimal α varies across datasets. This may require tuning in practice. The authors could propose a heuristic or adaptive method for setting α based on dataset characteristics.
3. While the paper compares with many strong baselines, some recent feature selection or embedding pruning methods are not included.
4. The description of the bitmap implementation in MFCS is somewhat brief. A more detailed pseudocode or example would help reproducibility.
5. The authors could explore adaptive N or early stopping strategies to reduce the number of gradient evaluations.

### Questions
1.The method assumes a static dataset for pre-training and selection. In real-world systems, new feature values may emerge over time. A discussion or experiment on incremental or online adaptation of the feature set would strengthen the practicality claim.
2.The performance is sensitive to α (selection ratio), and the optimal α varies across datasets. This may require tuning in practice. The authors could propose a heuristic or adaptive method for setting α based on dataset characteristics.
3. While the paper compares with many strong baselines, some recent feature selection or embedding pruning methods are not included.
4. The description of the bitmap implementation in MFCS is somewhat brief. A more detailed pseudocode or example would help reproducibility.
5. The authors could explore adaptive N or early stopping strategies to reduce the number of gradient evaluations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the problem of redundant and non-informative feature values in deep recommender systems, which increase memory usage and reduce model performance. The authors propose EffSelect, a fine-grained feature value selection framework that identifies important feature values based on their contribution to the prediction loss. The method combines a mini-batch sampling strategy that efficiently warms up embeddings with a gradient-based importance estimation that ranks and prunes uninformative values without adding extra parameters. Experimental results on multiple benchmark datasets demonstrate that EffSelect improves both accuracy and efficiency compared to existing field-level and gating-based selection methods.

### Strengths
1. High efficiency via mini-batch coverage sampling, especially compared to the learnable gate method OptFS.

2. The proposed method shows strong empirical results and robustness across datasets.

### Weaknesses
1. The baseline selection of this paper is narrow. The comparison scope is limited and does not fully reflect the landscape of recent progress in fine-grained feature value selection. Although the paper includes several classical baselines such as RF, XGBoost, and OptFS, it omits more recent and sophisticated methods like LPFS [1] and MultiFS [2]

2. The base model selection is inconsistent with the literature. The experimental design relies primarily on DCN and MaskNet, which, while valid architectures, do not align with the broader conventions in recommender system research. In particular, DeepFM, a foundational and widely adopted model for CTR prediction, has been consistently used in feature selection studies such as [2, 3, 4].

3. This paper claims to offer insightful ideas for selecting informative feature values. However, the authors do not explicitly demonstrate the conceptual insight or intuition underlying their method throughout the paper.

[1] Guo, Yi, et al. "LPFS: Learnable Polarizing Feature Selection for Click-Through Rate Prediction." arXiv preprint arXiv:2206.00267 (2022).

[2] Liu, Dugang, et al. "MultiFS: Automated multi-scenario feature selection in deep recommender systems." Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 2024.

[3] Lyu, Fuyuan, et al. "Optimizing feature set for click-through rate prediction." Proceedings of the ACM Web Conference 2023. 2023.

[4] Jia, Pengyue, et al. "Erase: Benchmarking feature selection methods for deep recommender systems." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

### Questions
Why you select MaskNet instead of DeepFM?

### Soundness
2

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
4

### Summary
This paper proposes EffSelect, a fine-grained feature-value selection framework for deep recommender systems. EffSelect introduces a mini-batch pre-training strategy (MFCS) to efficiently warm up embeddings and a gradient-based scorer (FeatIS) to measure each feature value’s contribution to prediction loss without adding learnable parameters. Experiments on four benchmark datasets (Criteo, Avazu, iPinYou, and Ali-CCP) show that EffSelect achieves state-of-the-art accuracy and efficiency, outperforming field- and gate-based baselines while reducing parameter and training costs.

### Strengths
1. The overall presentation of the paper is good, and the logic is clear.
2. The related work section provides a comprehensive survey and covers the key prior studies.
3. The code is open-sourced, which facilitates reproducibility and future research follow-up.
4. The selection of datasets and backbones is reasonable and generally consistent with previous works.

### Weaknesses
1. Why were these four datasets selected for the experiments? As I understand, Ali-CCP, Avazu, and Criteo are commonly used benchmarks in the feature selection domain, so the authors could provide more detailed explanations and relevant citations. In addition, the iPinYou dataset has very few fields, why was it chosen?
2. Is the feature sensitivity to the ratio α consistent across different datasets? From the main results table, the Avazu–DCN combination seems to perform best with the base model. The authors may need to include more experimental details to clarify this.
3. Could the authors provide some examples to help understanding? For instance, which feature values were filtered out by EffSelect, so that the effectiveness of feature selection can be illustrated more intuitively? just like a case study section.

### Questions
please refer to the weakness section

### Soundness
2

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
The work proposes a framework EffSelect for recommendation systems that is able to quantify each feature value's contribution to prediction loss for finer-grained selection. This method uses mini-batch pre-training for rapid adaptation and gradient-based evaluation to discard low-scoring features. It improves efficiency and robustness without extra learnable parameters.

### Strengths
1. The proposed fine-grained feature selection by considering its is contribution to loss in mini-batch is novel and effective.
2. The theoretical analysis on importance design is solid.
3. Experiments across four benchmarks show consistent AUC/logloss improvements and memory savings versus baselines like OptFS and AdaFS

### Weaknesses
I think the only concern for me is that the work should add more recommendation system's metrics in experiment.

### Questions
1. could the authors add more experiments on recommendation system's traditional metrics?
2. Could the authors provide an ablation on the trade-off between batch coverage and training loss convergence to confirm robustness?

### Soundness
3

### Presentation
3

### Contribution
3
