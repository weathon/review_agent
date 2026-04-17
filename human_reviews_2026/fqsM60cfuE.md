# EntroPE: Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Transformer-based models have significantly advanced time series forecasting, with patch-based input strategies offering efficiency and improved long-horizon modeling. Yet, existing approaches rely on temporally-agnostic patch construction, where arbitrary starting positions and fixed lengths fracture temporal coherence by splitting natural transitions across boundaries. This naive segmentation often disrupts short-term dependencies and weakens representation learning. In response, we propose EntroPE (Entropy-Guided Dynamic Patch Encoder), a novel, temporally informed framework that dynamically detects transition points via conditional entropy and dynamically places patch boundaries. This preserves temporal structure while retaining the computational benefits of patching. EntroPE consists of two key modules, namely an Entropy-based Dynamic Patcher (EDP) that applies information-theoretic criteria to locate natural temporal shifts and determine patch boundaries, and an Adaptive Patch Encoder (APE) that employs pooling and cross-attention to capture intra-patch dependencies and produce fixed-size latent representations. These embeddings are then processed by a global transformer to model inter-patch dynamics. Experiments across long-term forecasting benchmarks demonstrate that EntroPE improves both accuracy and efficiency, establishing entropy-guided dynamic patching as a promising new paradigm for time series modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces EntroPE, a temporally informed framework that dynamically detects transition points via conditional entropy and dynamically places patch boundaries. Experiments across long-term forecasting benchmarks demonstrate that EntroPE improves both accuracy and efficiency.

### Strengths
1. The paper is well-structured.
2. The paper provides a ​​comprehensive​ review of the related work.
3. The experiments include extensive comparisons with state-of-the-art baselines.

### Weaknesses
1. The novelty and contribution of this paper are limited. I have carefully read the paper and check their repository, and I find that the method used in the text is exactly the same as Byte Latent Transformer's [1], and the code was copied directly with minor refactors. The only difference is that, unlike natural language, time series data is not discrete. Therefore, the authors adopt discretized tokenization from Chronos [2]. Therefore, the contribution of this work primarily constitutes an engineering-level transfer (from NLP to TSF), rather than high-level innovation.
2. The improvement of EntroPE is marginal. In Table 1, its optimal case shows an improvement of almost no more than 1% over the second-best method (e.g., 0.416 vs. 0.418, 0.378 vs. 0.380, 0.242 vs. 0.244). Empirically, a gain of this magnitude is often smaller than the standard deviation observed across multiple runs of the same model (though this variance is not reported in the table). This indicates that the migration from NLP to time series was unsuccessful.
3. The efficiency analysis is incomplete. Section 4.3 only provides comparative results with some baselines on the ETTm1 dataset. The claim of BLT's efficiency is based on large-scale pre-training, where the cost of the local encoder and decoder is negligible compared to the typically deep layers of a Global Transformer. However, transformers for time series forecasting are usually much shallower (e.g., 1 or 2 layers). In this context, the overhead of patching could even exceed that of the global Transformer. It can be inefficient in some scenarios, while this paper fails to analyze that.

[1] Pagnoni, A., et al. Byte Latent Transformer: Patches Scale Better Than Tokens. Proceedings Of The 63rd Annual Meeting Of The Association For Computational Linguistics. 9238-9258 (2025)

[2] Ansari A F, Stella L, Turkmen C, et al. Chronos: Learning the language of time series (2024)

### Questions
1. Many models utilize patching technique. Would replacing the patching module in these models with EntroPE components lead to any performance improvement? Supplementing these results would enhance the persuasiveness of the claim.
2. How to handle the case that the patch sizes are highly imbalanced (too long or too short)?

### Soundness
2

### Presentation
3

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
The paper proposes **EntroPE (Entropy-Guided Dynamic Patch Encoder)**, a novel framework for time series forecasting that addresses limitations of existing temporally-agnostic patch-based transformer models. By dynamically placing patch boundaries at natural temporal transitions via conditional entropy, EntroPE preserves temporal coherence, mitigates train-inference mismatch, and enhances both forecasting accuracy and computational efficiency. It outperforms 14 state-of-the-art models across seven benchmark datasets.

### Strengths
1. The insight and intuition of this paper are valuable. In recent time series forecasting models (large-scale time series foundational models in particular), patch-based modelling is normally taken by default. Investigations of the deficiencies of current patch modelling strategies are important.
2. This paper is well-written and well-presented. The descriptions are clearly structured and well-organized. The model structure is clearly demonstrated and explained with mathematical formulations. The experiments are conducted with sufficient baselines and ablations.
3. The model structure is simple, reasonable, and effective. The idea of allocating entropy calculation to the patch length decision is quite innovative. More importantly, such techniques avoid potential risks in multi-resolution patch modelling, such as additional training complexity introduced by multiple patch embedding projections.

### Weaknesses
1. The term “the global transformer” is not sufficiently explained in the original manuscript. The authors should not neglect this, as patch embeddings are often correlated with transformer attention modelling to generate better performance.
2. Similar to Chronos, the authors transform patch embedding into point embedding to resolve different patch lengths (with pooling and cross-attention token merging). This approach is straightforward, but also risky. As point embeddings are localized, they cannot extract adjacent relations like patch embeddings and might potentially rely on quantization like Chronos.
3. The baselines of the large-scale time series model are not sufficient. Some state-of-the-art LTMs, such as Moirai, Sundial, and TimeMOE, are not included.

### Questions
1. Is “global transformer” the same as “vanilla transformer”? What is the backbone structure of this “global transformer”?
2. The authors mentioned the non-casual bidirectional attention in the “global transformer”. Is this the BERT-styled encoder-only attention mechanism? 
3. Can we change the “global transformer” to different backbones and maintain similar enhancements listed in the current experiments?
4. If we use multi-resolution patch modelling (such as using 3 patch lengths: 4,8,16), can EntroPE still outperform?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces dynamic time series tokenization by leveraging predictive entropy from a small pre-trained causal model to adaptively segment patches. It uses an Adaptive Patch Encoder (APE) and Fusion Decoder (FD) to normalize these variable-length segments for a global transformer, achieving strong forecasting performance.

### Strengths
* The idea of using predictive entropy to guide dynamic patch boundaries is creative and addresses a real limitation of fixed patching schemes.
* The proposed APE and FD modules form a coherent pipeline that can handle variable-length patches while preserving temporal details.
* The model achieves strong or state-of-the-art performance on several benchmarks, with solid ablation and integration studies that demonstrate the value of the dynamic tokenization idea.

### Weaknesses
* The overall system feels overly engineered, involving a pre-trained entropy model, multi-stage encoding, and several cross-attention blocks. 
* The paper does not cite TimeCAT (ICLR 2025 submission), which similarly criticizes fixed patching and proposes dynamic grouping. The omission makes the novelty of the problem framing less convincing.
* The approach assumes that predictive entropy from a small proxy model is an accurate indicator for patch segmentation, but this connection is not well validated or theoretically supported.

### Questions
* How does EntroPE compare with recent adaptive segmentation or dynamic grouping approaches such as TimeCAT?
* Why rely on a separately pre-trained entropy model for segmentation instead of learning patch boundaries jointly with the forecasting model?

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
EntroPE introduces an information-theoretic approach to patch construction in Transformer-based time series forecasting. Instead of using fixed-length patches, it dynamically determines patch boundaries by detecting local maxima in the conditional entropy of a lightweight, pre-trained next-token predictor. These variable-length patches are then encoded via an adaptive patch encoder (APE) and processed by a global Transformer. The method is evaluated on seven standard benchmarks and compared against 14 baselines, showing consistent accuracy gains and favorable efficiency trade-offs.

### Strengths
(1) First work to use entropy-based, data-dependent boundary detection in time series patching; tackles a real limitation of fixed-length patches.
(2) Outperforms PatchTST by ~10–20% MSE on several datasets and remains competitive with larger foundation models (CALF, LangTime) while using  fewer parameters.
(3) Explicitly controls computational budget via entropy thresholds; MACs vs. MSE analysis shows favorable Pareto frontier.
(4) Ablation study demonstrates that dynamic patching contributes more than the adaptive encoder or fusion decoder alone.

### Weaknesses
(1) Core pipeline (patch → encode → global Transformer → cross-attention decoder) closely mirrors some previous works, likePatchTST and Perceiver; entropy-based segmentation is the only major novelty.
(2) No guarantee that entropy peaks coincide with optimal regression boundaries; no analysis of information loss when abrupt changes are missed or over-segmented.
(3) Entropy model is trained per dataset; cost of re-training for new domains or high-frequency data is ignored.
(4) Best-configuration results for FilterTS, LangTime, etc., are selected by the authors; fair comparison requires identical hyper-parameter search budgets.
(5) Only long-term forecasting under channel independence; no multivariate interaction, classification, or anomaly detection tasks.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
