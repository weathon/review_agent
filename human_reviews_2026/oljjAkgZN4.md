# MIAM: Modality Imbalance-Aware Masking for Multimodal Ecological Applications

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Multimodal learning is crucial for ecological applications, which rely on heterogeneous data sources (e.g., satellite imagery, environmental time series, tabular predictors, bioacoustics) but often suffer from incomplete data across and within modalities (e.g., unavailable satellite image due to cloud cover, missing records in a time series). While data masking strategies have been used to improve robustness to missing data by exposing models to varying input subsets during training, existing approaches typically rely on static masking and inadequately explore the space of input combinations. As a result, they fail to address modality imbalance, a critical challenge in multimodal learning where dominant modalities hinder the optimization of others. To fill this gap, we introduce Modality Imbalance-Aware Masking (MIAM), a dynamic masking strategy that: (i) explores the full space of input combinations; (ii) prioritizes informative or challenging subsets; and (iii) adaptively increases the masking probability of dominant modalities based on their relative performance and learning dynamics. We evaluate MIAM on two key ecological datasets, GeoPlant and TaxaBench, with diverse modality configurations, and show that MIAM significantly improves robustness and predictive performance over previous masking strategies. In addition, MIAM supports fine-grained contribution analysis across and within modalities, revealing which variables, time segments, or image regions most strongly drive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a masking strategy called MIAM. The objective is to improve the robustness of transformers to missing modalities at test time. The topic is important because there is a trend towards fusing more and more modalities, while it is crucial to maintain performance on any subset of modalities seen during training for practical utility. MIAM increases the probability of a modality being masked (not given as input) if its unimodal validation performance is high and not changing; conversely, modalities with low and increasing accuracy are masked less often (given as input). MIAM improves over other masking strategies for two multi-modal datasets.

### Strengths
- Multi-modal masked modelling is increasingly used, so the topic is important
- I learned from the paper and it may inform my research
- The idea of MIAM is intuitive, fairly simple, and seems to work — at least when the number of modalities > 2
- The paper is well written and has nice figures that aid understanding

### Weaknesses
Moderate weakness:
- I believe that according to the appendix, all transformers have 3 layers and embedding dimension 192, which is very very small. For reference, a ViT-Base has 12 layers and dimension 768. I understand that larger models are costlier but I suspect model size will interact with masking strategies. For example, larger models may be able to fit all modalities easier, and thus uniform masking may be fine. 

Minor weakness:
- The method seems to rely on computing validation accuracy after each epoch. Thus it cannot be directly used in self-supervised learning, e.g., masked autoencoding (MAE), since we aim to learn task-agnostic representations without using labelled data. Since multi-modal MAE is quite popular, having MIAM directly support MAE would be nice.

### Questions
None

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MIAM (Modality Imbalance-Aware Masking), a dynamic masking strategy for multimodal learning. MIAM formalizes masking as probability distributions over unit hypercubes and addresses three key principles often missing in prior methods: Full support for all input combinations, Corner prioritization to emphasize critical configurations, and Imbalance awareness by adapting masking probabilities based on modality dominance. To achieve this, MIAM constructs a mixture of product beta distributions and dynamically adjusts masking during training using modality-specific performance and learning speed. This design enables handling arbitrary missing inputs, mitigating modality imbalance, and supporting fine-grained contribution analysis across and within modalities.

### Strengths
-	Interesting approach to improve performance and robustness when modalities are missing. 
-	Strong average performance across different subsets of modalities compared to other sampling strategies. 
-	Methodology is clearly described and well-structured. 
-	Provides insightful analysis of how modality-specific values evolve during training.

### Weaknesses
-	Only tested on two classification downstream tasks. Broader applicability (e.g., segmentation tasks like flood mapping where optical data is often missing) is not demonstrated. Also includes frequent references to pre-training masking strategies in related work, while not compared for pre-training.
-	Downstream performance is sensitive to new hyperparameters. With ablation shown only on one dataset (Figure 10), it is difficult for readers to tune parameters effectively. 
-	Evaluated with only one architecture, which is not well explained, even though MIAM is likely applicable to many others. 
-	Training details and architecture description are insufficient. Before presenting results, it should be made explicit that one model per masking strategy was trained and then evaluated on different subsets.

### Questions
-	Could MIAM be adapted for other model architectures? If so, what could be challenges? 
-	How should practitioners choose λ and κ in practice for new datasets without extensive tuning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MIAM (Modality Imbalance-Aware Masking), a dynamic masking strategy for multimodal learning that addresses the challenge of modality imbalance in ecological applications. The key insight is to formalize masking strategies as probability distributions over unit hypercubes and design a principled approach with three properties: (i) full support over all input combinations, (ii) corner prioritization to favor complete/minimal modality combinations, and (iii) imbalance-awareness that adaptively masks dominant modalities based on their performance and learning dynamics. MIAM uses a mixture of product beta distributions whose parameters are dynamically adjusted during training based on per-modality performance scores and their temporal derivatives. The method is evaluated on two ecological benchmarks (GeoPlant and TaxaBench) and demonstrates consistent improvements over existing masking strategies while providing fine-grained contribution analysis.

### Strengths
1. Testing on two diverse ecological datasets (GeoPlant with 3 modalities, TaxaBench with 5 modalities) with multiple modality combinations demonstrates robustness.
2. The fine-grained contribution analysis (Fig. 5) demonstrates how the method can provide ecological insights (e.g., importance of NDVI bands, impact of extreme events), bridging ML and domain science.
3. The progressive ablation showing the contribution of each design principle (uniform hypercube → beta hypercube → MIAM) is convincing.

### Weaknesses
1. Only ecological datasets are thoroughly evaluated; broader applicability claims need support from other domains. The SatBird result (Appendix A.4.3) showing similar performance across strategies raises questions about when MIAM is beneficial. No comparison on standard multimodal benchmarks (e.g., vision-language tasks)
2. The choice of ε=3 and φ=10 appears arbitrary. THere is limited discussion of how to set these in practice. The corner weights (Eq. 3) use specific fractions (1/4, 1/2) without justification
3. OPM (Wei et al. 2024) is the main dynamic masking baseline, but other recent modality balancing methods are mentioned but not compared. Missing comparison with recent self-supervised multimodal methods.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a new masking algorithm that addresses the problem of modality competition, wherein one modality dominates the learned features. The algorithm uses a mixture of product beta distributions that concentrations probability mass in the corners (especially the all-on or all-off corners) of a unit hypercube representing the probability that tokens within each modality are masked. The weights are adjusted dynamically to adjust the masking probability depending on the performance and learning speed of each modality. Experiments are performed on two multimodal ecology datasets, GeoPlant and TaxaBench, using a transformer architecture. MIAM overall improves performance on these benchmarks compared to baselines including On-the-fly Prediction Modulation (OPM), which adjusts per-modality probabilities based on relative performance scores but applies the probability to the entire modality, not each token within each modality as in MIAM. The paper also shows how MIAM indicates which inputs drive performance, providing a measure of the contribution/importance of each modality.

### Strengths
- This paper addresses an understudied problem in multimodal learning of modality imbalance/competition. Despite being understudied, it is a common challenge in multimodal learning for remote sensing (and other domains).
- The paper is well written and easy to read. Technical details are clearly explained. Figures and tables are high quality and are helpful for understanding the paper’s results and ideas.
- The contribution analysis enabled by MIAM is a nice bonus and would be appreciated by domain experts in ecology and other domains (e.g. across remote sensing applications).
- The ablation experiment is well designed and effectively shows the contribution of each component of the proposed algorithm.
- The proposed algorithm is well-motivated, and the paper gives nice explanations and figures to support the motivation and intuition behind the algorithm’s design.

### Weaknesses
- The MIAM masking algorithm could be applied to any model architecture that implements masked multimodal learning. I think there was a missed opportunity to show the value of MIAM on existing multimodal remote sensing foundation models. If it worked, MIAM could significantly improve the utility of these models. (To me, this is the difference between a score of 8 and 10.)
- It seems that a natural extension (or even baseline?) of MIAM is to assign a masking probability to each token, rather than applying the same probability to all tokens within each modality. Did the authors consider or test this?
- I think the motivation for prioritizing the all-on/all-off corners (or corners in general) could be better explained in terms of the ecological application context. It doesn’t seem obvious to me why it would be beneficial to prioritize combinations with almost all tokens or almost no tokens from each modality.

### Questions
- Why is it beneficial to prioritize combinations with almost all tokens or almost no tokens from each modality?
- How does MIAM affect the performance of multimodal remote sensing foundation models?
- How does applying the same probability to all tokens in a modality compare to applying a different probability to each token regardless of modality?

### Soundness
4

### Presentation
4

### Contribution
4
