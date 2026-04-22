# MOMEMTO: Patch-based Memory Gate Model in Time Series Foundation Model

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Recently reconstruction-based deep models have been widely used for time series anomaly detection, but as their capacity and representation capability increase, these models tend to over-generalize, often reconstructing unseen anomalies accurately. Prior works have attempted to mitigate this by incorporating a memory architecture that stores prototypes of normal patterns. Nevertheless, these approaches suffer from high training costs and have yet to be effectively integrated with time series foundation models (TFMs). To address these challenges, we propose MOMEMTO, an improved variant of TFM for anomaly detection, enhanced with a patch-based memory module to mitigate over-generalization. The memory module is designed to capture representative normal patterns from multiple domains and enables a single model to be jointly fine-tuned across multiple datasets through a multi-domain training strategy. MOMEMTO initializes memory items with latent representations from a pre-trained encoder, organizes them into patch-level units, and updates them via an attention mechanism. We evaluate our method using 23 univariate benchmark datasets. Experimental results demonstrate that MOMEMTO, as a single model, achieves higher scores on AUC and VUS metrics compared to baseline methods, and further enhances the performance of its backbone TFM, particularly in few-shot learning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the over-generalization problem in anomaly detection models by extracting patch memories from different domains and reconstructing time series through linear combinations of these patch memories, thereby improving the accuracy of anomaly detection.

### Strengths
1. The problem that this paper aims to solve is clearly described, and the proposed solution is well articulated.

2. The paper demonstrates strong technical soundness, showing good practical and methodological rationality.

3. The authors conduct extensive experiments on 870 time series, comparing their approach against 11 popular time series anomaly detection methods, which convincingly validates the effectiveness of the proposed method.

### Weaknesses
1. Compared with MEMTO, the contributions and innovations of this paper are rather marginal. Both approaches build a prototype memory and reconstruct the original data through linear combinations of vectors within that memory. Overall, the work lacks novelty and makes limited contributions to the advancement of the field.

2. Although the paper points out that MEMTO heavily depends on memory initialization, the discussion of how the proposed method alleviates this issue is insufficient. For example, it is unclear why the proposed method is less dependent on memory initialization and whether there are experiments that support and verify this claim.

3. The paper states that MEMTO follows a “one-model-per-dataset” paradigm. However, the original MEMTO paper mentions that MEMTO can capture multiple prototypes to represent diverse patterns, implying that it could generalize across datasets and potentially achieve a “one-model-for-multiple-datasets” capability. Therefore, this claim in the reviewed paper requires further justification.

4. The core problem this paper aims to solve is preventing model over-generalization, which can lead to reduced reconstruction errors for anomalies and thus hinder anomaly discrimination. However, the experiments do not directly demonstrate whether the proposed method effectively mitigates this problem — for instance, by showing that its recall rate improves compared with baselines at their best F1 scores.

### Questions
1. Could you provide an additional experiment comparing the recall of the proposed method and the baselines at their best F1 scores?

2. Could you elaborate on how the proposed method addresses MEMTO’s strong dependency on memory initialization?

3. Could you summarize and highlight the specific contributions and innovations of the proposed method compared with MEMTO?

### Soundness
3

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
This paper proposes MOMEMTO, a Time Series Foundation Model (TFM) for reconstruction-based time series anomaly detection. The primary goals are to mitigate the "over-generalization" problem, where models accurately reconstruct anomalies, and to address the high computational cost of the "one-model-per-dataset" training paradigm.

Key Contributions:
- Architecture: MOMEMTO integrates a pre-trained TFM encoder from MOMENT with a patch-based memory module made by modifying MEMTO. This module stores prototypes of normal patterns at the patch level to prevent the reconstruction of anomalies.
- Training Strategy: The paper uses a multi-domain training strategy to jointly fine-tune a single model across multiple datasets, which is designed to be more efficient than training separate models.
- Empirical Results: The model is evaluated on 23 univariate benchmark datasets. As a single model, MOMEMTO achieves higher scores on AUC and VUS metrics compared to 13 baseline methods.
- Few-Shot Learning: The method is shown to enhance the performance of its backbone TFM, particularly in few-shot learning scenarios.

On initial read this paper seems very convincing, however a lack of empirical analysis on results and conclusion that contradict reported metrics leave me unconvinced. I want to give this paper a higher rating if these concerns are addressed.

### Strengths
Originality: Fair

Quality: Good

Clarity: Good

Significance: Good

Additional note: The paper highlights the key drawbacks of existing methodologies, and modifies MEMTO and MOMENT into one architecture to alleviate drawbacks of both models. The paper is well written and mostly, the motivation for design decisions is well motivated.

### Weaknesses
Discrepancy in results and conclusions: In section 4.2 where Figure 3 is discussed, the conclusion is that MOMEMTO exhibits better separation between anomalous and non anomalous scores as compared to MOMENT. However, the results in Appendix C.4 show little or no difference for certain datasets between the two models, and in certain datasets, (TAO-Environment, YAHOO-Synthetic, SMD-Facility, WSD-Webservices), MOMENT outperforms MOMEMTO. I see this as the biggest issue in the paper.

In appendix C.1 in table 9, out of 32 domains, MOMEMTO md has best or 2nd best performance in only 12.
Results: There was no empirical evaluation of the distribution of anomalous scores done, conclusion is made based on (what it seems to be) visual inspection of boxplot graphs.

Uni-variate focus: Only uni–variate datasets were considered in the experiments

### Questions
Please provide empirical analysis of results to accompany graphs.

The number of memory items is set to 32, matching the 32 domains identified in the benchmark38. While the update strategy is analyzed (Table 3)39, the paper does not provide a sensitivity analysis for the number of memory items. How would the performance differ if the the number of items < the number of domains. Would the memory items exhibit plasticity? Some experiments with accompanying analysis would be good.

Please provide an explanation for the lack of convincing results in Appendix C.4 and C.1 (Table 9).
A clarification on the ultimate objective of the paper would be appreciated. Is your claim that MOMEMTO is better than all baseline models mentioned? Or that it is a modification that improves MOMENT.

### Soundness
3

### Presentation
4

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
This paper introduces MOMEMTO, a memory-augmented anomaly detection framework that integrates the pre-trained MOMENT encoder with the memory update mechanism of MEMTO.
The model is designed for multi-domain, univariate time series and operates in a fully data-driven manner—learning cross-domain normality representations without domain labels.
Key components include a patch-level memory alignment strategy that adaptively refines prototypical memory items across heterogeneous domains.
Empirically, MOMEMTO achieves superior performance over strong baselines (MOMENT, MEMTO, TimesNet, Anomaly Transformer) on the TSB-AD-U benchmark while maintaining computational efficiency.

### Strengths
**Originality:**
MOMEMTO combine a pre-trained cross-domain encoder with a memory mechanism for anomaly detection. Its data-driven update policy allows unsupervised memory adaptation without explicit domain label

**Quality:**
Good

**Clarity:**
The formulation of memory alignment, gating, and normalization is well structured, and the Appendix provides sufficient hyperparameter details for reproducibility.

**Significance:**
The method simultaneously improves multi-domain generalization and computational efficiency—achieving better accuracy with fewer active memory items. This is practically important for scalable time-series anomaly detection in resource-constrained or heterogeneous environments.

### Weaknesses
**1. Limited scope of experimental validation:**
While the authors acknowledge this limitation in their conclusion, the empirical validation remains confined to univariate time series from a single benchmark (TSB-AD-U), which constrains the generalizability of the results. Additional experiments on either multivariate datasets or alternative univariate benchmarks would be essential to substantiate the robustness and broader applicability of the proposed framework.

**2. Interpretability of Figure 4:**
The description of “accumulated during training” remains ambiguous—it’s unclear whether the visualization reflects cumulative statistics over the entire training process or a specific epoch window.
Clarifying this temporal scope would make the observed concentration dynamics more interpretable.

**3. Domain-level selection imbalance:**
In Figure 4, the Daphnet (Human Activity) domain exhibits the strongest memory selection, while some domains appear nearly unused.
The paper does not analyze why this imbalance occurs—whether due to domain size, feature diversity, or intrinsic signal variance.
Additional quantitative analysis or ablation would strengthen the claim of cross-domain adaptability.

### Questions
**1. Comparison with H-PAD:**
The proposed framework shares conceptual similarities with H-PAD (Hybrid Prototype-based Anomaly Detection), which also employs patch-level prototypes but introduces periodic prototypes for long-range temporal modeling.
Could the authors elaborate on the principal design differences—especially regarding prototype granularity and update dynamics—and whether H-PAD could serve as a complementary baseline?

**2. Domain-level bias in memory utilization:**
What drives the dominant selection of the Daphnet :Human Activity domain in Figure 4?
Conversely, which domains are rarely or never selected, and do these cases correlate with domain size or anomaly density?
A systematic analysis would clarify whether the model’s “data-driven” selection is influenced by statistical imbalance.

**3.Definition of “accumulated during training”:**
Does Figure 4 aggregate similarities across all epochs or represent a snapshot at a particular stage of training (e.g., final epoch)?
A precise definition is necessary to interpret whether the observed concentration reflects progressive convergence or a steady-state behavior of the memory update dynamics.

### Soundness
3

### Presentation
4

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
This paper proposes the model MOMEMTO, which is a type of basic model for anomaly detection in time series. It is enhanced through a patch-based memory module by utilizing a pre-trained encoder and patch-based memory module, which somewhat alleviates the overgeneralization problem.

### Strengths
1. A more fair metric evaluation system was adopted, reasonably reflecting the model's performance in terms of accuracy; 

2. The idea of a model handling datasets from multiple domains has been widely explored in the time series field, but it still holds some value in the TSAD domain; 

3. The explanation of the method in the article is relatively clear.

### Weaknesses
1. The novelty of the article is relatively limited and needs to be restated, especially with further experimental or theoretical explanations of the mutual benefits among the components. The proposed framework combines three elements: the Patch mechanism, the memory mechanism, and the Moment model. However, these three components have likely been widely explored in the TSAD field, particularly the first two, which leads me to believe that this paper may have difficulty providing readers with novel insights.

2. The paper claims to be a TFM specifically designed for time series anomaly detection that supports multiple domains. On one hand, this is positive, as we encourage more research in this area adapting a single model to multiple time series anomaly datasets. On the other hand, according to experimental comparison results, machine learning models like Isolation Forest and deep learning models like LSTM-AD show nearly suboptimal performance. Although they do not achieve the performance level of existing models, they can reach performance close to that of the proposed model with a simpler structure and very low cost. In contrast, the TFM dedicated to time series anomaly detection may incur a much higher cost without showing remarkable performance, and to some extent, it loses the ability to handle other time series tasks. Therefore, I believe the motivation of this work should be re-evaluated. Additionally, I think comparisons with more integrated machine learning models should be conducted to assess whether higher costs are justified for relatively modest performance gains. More experiments and discussions are needed to support the rationale behind this model.

3. The comparative baselines in the article are relatively weak, with few methods from 2025, especially in the deep learning section where many methods are outdated. This makes the comparative experiments somewhat unreliable and further calls into question whether the motivation of the paper is reasonable or robust. It is recommended that the authors supplement more comparative baselines.

4. The discussion on computational efficiency is insufficient and too vague, and this indicator of other comparative algorithms should be included.

### Questions
1. The content regarding the novelty of the model and the rationality of combining existing technologies needs more theoretical and experimental evidence. It is recommended that the author provide additional support; 

2. The comparison methods are too outdated; 

3. The advantages of the model are not obvious, especially more comprehensive comparative experiments on parameters, training, and inference time are needed to highlight the advantages of the proposed method.

### Soundness
2

### Presentation
2

### Contribution
2
