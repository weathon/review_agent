# Is Bidirectionality Necessary in Mamba for Time Series Forecasting?

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Mamba is a sequential model that has recently emerged as a promising alternative to Transformers, offering near-linear complexity.
However, although channels in time series (TS) data generally lack a sequential order, recent studies have adopted Mamba to capture channel dependencies (CD) in TS, introducing a sequential order bias. To address this, prior works have adopted bidirectional Mamba to scan channels in both forward and reverse orders. In this paper, we show that unidirectional Mamba can effectively replace the bidirectional Mamba with simple strategies. To this end, we propose FSMamba, a TS forecasting method employing a unidirectional Mamba that incorporates a regularization strategy to minimize the discrepancy between two embedding vectors generated from data with reversed channel orders, thereby enhancing robustness to channel order. Furthermore, we introduce channel similarity modeling, a pretraining task to preserve similarities between channels from the data space to the latent space to enhance the ability to capture CD. Extensive experiments demonstrate the efficacy of our method, achieving state-of-the-art performance on diverse datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies whether bidirectionality is necessary for Mamba in multivariate time series forecasting. Since channels lack natural order, applying bidirectional Mamba for channel dependencies (CD) adds complexity. The authors propose FSMamba, a unidirectional model with a regularization term that minimizes the distance between embeddings of reversed channel orders and a channel similarity modeling (CSM) pretraining step. Experiments on multiple datasets show FSMamba achieves comparable or better performance than bidirectional models with fewer parameters, proving that bidirectionality is unnecessary for effective CD modeling.

### Strengths
1. Clear and easy-to-understand writing with clean and visually appealing figures.
2. Detailed experimental analysis

### Weaknesses
1. Since Mamba itself is designed for TD, the motivation for applying it to CD is not well justified.
2. The evaluation of Mamba’s modeling capability on CD is insufficient.

### Questions
1. The paper proposes a regularization method that minimizes the distance between two embedding vectors generated with reversed channel orders. However, this seems to prevent Mamba, as a sequence modeling method, from capturing sequential information. I do not understand the rationale behind this design. The final architecture appears equivalent to using Selective SSM to capture CD. Please explain why this design is necessary and why it works.

2. Since the existing benchmarks contain too few dimensions, to further validate FSMamba’s capability in modeling CD, it would be helpful to include experiments on some datasets from Time-HD benchmark [1] and Wike2000 from TFB [2].

[1] Ni, J., Wang, S., Liu, Z., Shi, X., Zhong, X., Ye, Z., & Jin, W. (2025). U-Cast: Learning Hierarchical Structures for High-Dimensional Time Series Forecasting. arXiv preprint arXiv:2507.15119.
[2] Qiu, X., Hu, J., Zhou, L., Wu, X., Du, J., Zhang, B., ... & Yang, B. (2024). Tfb: Towards comprehensive and fair benchmarking of time series forecasting methods. arXiv preprint arXiv:2403.20150

### Soundness
3

### Presentation
4

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
This paper proposes FSMamba, a unidirectional Mamba model for time series forecasting that addresses the "sequential order bias" found in bidirectional models. It uses a "Flipped Siamese" regularization strategy to enforce robustness to channel order and removes the 1D-convolution layer from the Mamba block, arguing it is unsuited for non-sequential channel data. The authors also introduce Channel Similarity Modeling (CSM), a pretraining task designed to preserve channel correlations from the data space to the latent space.

### Strengths
1. Principled Bias Correction: The paper clearly identifies the sequential order bias from applying Mamba to non-sequential channels. The proposed "Flipped Siamese" regularization is an intuitive and effective method to enforce order robustness using a single, shared-weight unidirectional model, avoiding the inefficiency of two-model bidirectional approaches. 
2. Computational Efficiency: FSMamba achieves state-of-the-art performance while being significantly more efficient than bidirectional baselines like S-Mamba. It uses 37.6%–38.1% fewer parameters, consumes less GPU memory, and demonstrates faster training and inference times.
3. Thorough Validation: The method is validated on 13 diverse datasets against numerous strong baselines. The paper includes extensive ablation studies (Section 6) that confirm the benefits of the regularization, 1D-conv removal, and CSM pretraining.

### Weaknesses
1. Limited Permutation Robustness: The regularization strategy only enforces robustness against a single permutation (the reversed channel order) not general permutation invariance. The paper dismisses using random permutations as "unstable" without a deep investigation, meaning the model is order-robust only to a single, specific flip.
2. Disconnected Pretraining Task: The Channel Similarity Modeling (CSM) pretraining task, which aims to preserve channel correlations, feels disconnected from the paper's main thesis on sequential order bias and its novelty is debatable.
3. Contradictory 1D-Convolution Results: The paper justifies removing the 1D-convolution by stating channels lack sequential order. However, it also notes PEMS datasets do have a meaningful geographical order. The results in Table 9 show removing the 1D-conv does not harm performance on PEMS, a counter-intuitive finding that is not adequately explained and weakens the motivation for the removal.

### Questions
1. Why random permutation result in unstable training? Why can this strategy not lead to better robustness compared to the reverse flip?
2. Does the conclusion in Fig. 7  and the robustness generalize to even higher-dimensional data[1]?
3. For the CSM pretraining task, why only preserving the linear correlations? Will preserving non-linear correlations also help?
4. Can the two loss strategies (L_reg and L_CSM) be applied to other CD models like iTransformer[2] and Duet[3]?


References:
[1] Ni J, Wang S, Liu Z, Shi X, Zhong X, Ye Z, Jin W. U-Cast: Learning Hierarchical Structures for High-Dimensional Time Series Forecasting. arXiv preprint arXiv:2507.15119. 2025 Jul 20.
[2] Liu Y, Hu T, Zhang H, Wu H, Wang S, Ma L, Long M. itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625. 2023 Oct 10.
[3]Qiu X, Wu X, Lin Y, Guo C, Hu J, Yang B. Duet: Dual clustering enhanced multivariate time series forecasting. InProceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1 2025 Jul 20 (pp. 1185-1196).

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
4

### Summary
This paper investigates whether bidirectionality is essential in applying Mamba for modeling channel dependencies (CD) in multivariate time series forecasting. The authors propose FSMamba, a lightweight alternative that removes bidirectionality and introduces a regularization term to align embeddings generated from original and reversed channel orders. Experiments on thirteen benchmark datasets (including ETT, PEMS, Exchange, Weather, ECL, Solar, and Traffic) demonstrate that FSMamba achieves state-of-the-art accuracy with around 37% fewer parameters than prior Mamba-based models and exhibits enhanced robustness to channel-order permutations.

### Strengths
1. The paper asserts that minimizing embedding distance improves robustness to channel order but does not formally analyze why this suffices to approximate bidirectional behavior. A stronger theoretical connection between the regularizer and bidirectionality could strengthen the contribution.
2. The paper only reports λ-sensitivity results on the ETTh1 dataset (Table 14), showing stable performance within [0.01, 0.1], but it does not provide quantitative evidence across other datasets such as Weather, ECL, or PEMS. As the regularization term is central to FSMamba’s robustness design, a more systematic analysis of λ’s influence across datasets would strengthen the reliability and generality of the proposed approach.

### Weaknesses
1. The paper asserts that minimizing embedding distance improves robustness to channel order but does not formally analyze why this suffices to approximate bidirectional behavior. A stronger theoretical connection between the regularizer and bidirectionality could strengthen the contribution.
2. The paper only reports λ-sensitivity results on the ETTh1 dataset (Table 14), showing stable performance within [0.01, 0.1], but it does not provide quantitative evidence across other datasets such as Weather, ECL, or PEMS. As the regularization term is central to FSMamba’s robustness design, a more systematic analysis of λ’s influence across datasets would strengthen the reliability and generality of the proposed approach.

### Questions
1.Could you provide a more formal justification of how the proposed regularization approximates bidirectional scanning?
2.How sensitive is FSMamba to the regularization weight λ across datasets beyond ETTh1?

### Soundness
3

### Presentation
3

### Contribution
3
