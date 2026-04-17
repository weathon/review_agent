# CO-COME: A Contrastive Label Disambiguation Framework with Combined MTS Feature Encoder for Partial-Label Multivariate Time Series Classification

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Multivariate Time Series Classification (MTSC) is a conventional time series task applied in the fields of finance, healthcare, and weather forecasting. However, it is often plagued by a lack of high-quality labels in practical applications. To address the label quality issues in real-world scenarios, the Partial Label Learning (PLL) paradigm has been proposed. This paradigm solves the problem of ambiguous labels by allowing each training instance to be associated with a set of candidate labels. The superiority of PLL has been verified in the field of image classification. But due to the inherent difficulties in feature extraction for Multivariate Time Series (MTS) and the lack of appropriate data augmentation strategies, PLL has not been applied in MTSC tasks. Motivated by this, we propose a novel model: COntrastive label disambiguation framework with COmbined MTS feature Encoder (CO-COME), which integrates a contrastive learning-based label disambiguation framework with an efficient MTS feature representation encoder, CTFE. The contrastive learning module leverages label prototypes to effectively resolve label ambiguity under the PLL setting, meanwhile the CTFE encoder is designed to capture both explicit and latent representations of time series data, enabling robust and discriminative feature learning. Extensive experiments on 20 UEA benchmark datasets demonstrate that our model achieves state-of-the-art performance under partial-label conditions. Our method will be available in https://github.com/Noname9971/CO-COME once accepted.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CO-COME, a contrastive label disambiguation framework designed for Multivariate Time Series Classification (MTSC) under a Partial Label setting. The proposed method integrates a dynamic multivariate data augmentation strategy to enhance representation learning and employs a hybrid time series feature encoder that combines the lightweight MiniROCKET with a channel-independent transformer encoder. Experimental evaluations conducted on 20 UEA time series datasets demonstrate that CO-COME achieves superior performance compared to several baseline methods.

### Strengths
1.	This paper is among the first to apply Partial Label Learning to the task of Multivariate Time Series Classification (MTSC), which represents a meaningful contribution to the field.

2.	The figures and visualizations in the paper are well-designed, with clear layouts and effective color schemes, making the content easy to read and interpret.

3.	The paper provides a comprehensive and detailed review of traditional approaches to time series classification, demonstrating a good understanding of related work.

### Weaknesses
1.	Although combining Partial Label Learning with Multivariate Time Series Classification (MTSC) is an interesting direction, the paper does not clearly justify why MTSC faces partial label issues. In particular, the second paragraph of the introduction fails to cite any prior work supporting the existence of MTSC under partial label conditions, nor does the paper provide any real-world examples illustrating such scenarios.

2.	The proposed CO-COME framework shows limited novelty in partial label learning. Specifically, CO-COME largely follows the technical design of PiCO [1], with Figure 1 in this paper closely resembling Figure 2 in PiCO, the main difference being that PiCO is applied to image data while CO-COME extends the approach to multivariate time series via encoder adaptation.

3.	The paper overlooks modeling inter-variable relationships, which are crucial for MTSC. Using MiniROCKET and a channel-independent transformer primarily designed for univariate series weakens the claim of effective multivariate modeling.

4.	The baselines are all fully supervised methods, raising fairness concerns under the partial label setting. For instance, PiCO [1] compares against PLL-specific algorithms such as LWS, PRODEN, MSE, and EXP in image classification tasks. To ensure fair evaluation, the authors should also consider PLL-specific baselines (which are typically input data-type agnostic) for comparison.

5.	The review of MTSC methods omits several recent approaches from the past three years that should be discussed and included as baselines (e.g., [2–6]).


6.	The paper contains many minor formatting errors (e.g., missing spaces in “Classification(MTSC)” and “and PiCOWang et al.”).

7.	The provided code repository is empty, hindering reproducibility of the experimental results.




[1] PiCO: Contrastive Label Disambiguation for Partial Label Learning. ICLR, 2022.

[2] Shapeformer: Shapelet transformer for multivariate time series classification. KDD, 2024.

[3] TimeMIL: Advancing multivariate time series classification via a time-aware multiple instance learning. ICML, 2024.

[4] Graph-aware contrasting for multivariate time-series classification. AAAI, 2024.

[5] Abstracted shapes as tokens: a generalizable and interpretable model for time-series classification. NIPS, 2024.

[6] MPTSNet: Integrating multiscale periodic local patterns and global dependencies for multivariate time series classification. AAAI, 2025.

### Questions
1.	The authors employ MiniROCKET, which is originally designed for univariate time series classification, together with a channel-independent transformer for time series feature extraction. Given this design, why did the authors not evaluate their approach on the univariate UCR benchmark datasets [7], which would provide a more direct and consistent basis for comparison?

2.	According to Table 1, MiniROCKET achieves classification performance that is second only to the proposed method under the partial label learning setting. However, MiniROCKET itself does not incorporate any mechanisms specifically designed for partial label learning, and the proposed CO-COME framework uses MiniROCKET as one of its feature extraction components. Under such circumstances, is this experimental comparison fair and meaningful?

3.	If algorithms that are explicitly designed for partial label learning (such as PiCO) were uniformly equipped with MiniROCKET as the feature extraction module, how would their performance on multivariate time series classification compare to that of the proposed CO-COME method?

[7] The UCR time series archive[J]. IEEE/CAA Journal of Automatica Sinica, 2019, 6(6): 1293-1305.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles multivariate time-series classification (MTSC) under partial- label learning (PLL) conditions, where each training instance is annotated with a set of candidate labels rather than a single ground-truth label. The authors argue that label ambiguity is a critical issue in industrial time-series applications but has been largely ignored in existing MTSC research.
To address this, they propose CO-COME, a framework that integrates: Partial- label learning, a contrastive label disambiguation strategy (inspired by MoCo- style contrastive learning ), dynamic Multivariate Data Augmentation and A Combined Time-series Feature Encoder (CTFE).
The paper includes ablation studies (DMDA, CI-Transformer, MiniROCKET, FusionGate) and varying partial-rate experiments, showing that each component contributes positively, especially when label ambiguity increase

### Strengths
1. Effectively introduces partial-label learning into the multivariate time-series domain, a relevant and underexplored area.
2. Combines multiple strategies (contrastive disambiguation, augmentation, hybrid encoder) into a cohesive and well-motivated pipeline.
3. The inclusion of ablation studies provides good insights into the contribution of individual components.

### Weaknesses
1. Abbreviations overload: The paper introduces many abbreviations (DMDA, CTFE, CI-Transformer, etc.) without a clear reference list. A glossary or table of abbreviations in the appendix would aid readability.
2. Evaluation metrics: The authors report only accuracy, which limits our evaluation of the performance. Additional metrics such as F1-score, precision, recall, and confidence intervals (mean ± std) are necessary to assess robustness and class-wise performance, especially under label ambiguity.
3. The performance degradation on PM and HW datasets is notably large. Why the model performs differently on these datasets?
4. Incomplete notation: The variable σ is undefined in the time-mask section of the appendix; please clarify.
5. DMDA mechanism is unclear: While the paper states that DMDA “analyzes inter-variable correlations” and uses summary statistics (mean, max, min) to guide augmentation strength, the specific mapping strategy is not described. What are the parameter ranges or scaling rules derived from correlation metrics? How are weak and strong views quantitatively differentiated?
6. Simulation realism of partial labels: The paper uses uniform random super-setting to generate candidate-label sets, which is the simplest possible form of label ambiguity.
– This is reasonable for controlled studies (easy to tune via threshold) but does not capture real-world ambiguity, which is typically class- dependent (confusable or semantically similar labels).
– To strengthen the empirical claim, the authors should consider an alternative simulation where candidate sets are biased by inter-class confusion or nearest-class prototypes a more realistic approximation of ambiguous labeling in industrial datasets.
7. Compute reporting: How CO-COME’s compute time and GPU memory usage compares to other SOTA models.
8. No limitation section.

### Questions
Questions are listed above in the weaknesses section. Please clarify abbreviations and other notation, expand evaluation metrics and simulation settings.

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
This work proposes CO-COME, a contrastive label-disambiguation framework for partial-label multivariate time-series classification (MTSC), coupled with a Combined Time-series Feature Encoder (CTFE). DMDA builds weak/strong views; CTFE-W / CTFE-S (momentum pair) encode features; a feature memory bank and class prototypes drive a PiCO-style loss to refine pseudo-labels. Experiments claim strong results over 20 UEA datasets in fully-supervised MTSC.

### Strengths
1. Clear problem framing
2. On six highlighted datasets, accuracy decays more slowly than baselines

### Weaknesses
1. Novelty is incremental, since the contrastive+prototype core follows PiCO/PiCO+ closely; the contributions are (i) tailoring to time-series with CI-Transformer + MiniROCKET and (ii) DMDA that tunes aug strength from inter-channel correlations. 

2. PLL setting is synthetic, as partial labels are formed by uniform random thresholding over classes. This is convenient but unlikely to match real ambiguity.

3. Many compared methods are not designed for PLL and run with standard settings, while CO-COME explicitly targets PLL. Need adapted PLL variants.

4. Table 2 bundles four modules; deltas are sometimes modest or dataset-dependent. We don’t see effect sizes for each hyperparameter of DMDA or FusionGate; memory-bank size/temperature is unspecified.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents CO-COME, a method for Multivariate Time Series Classification (MTSC) when labels are ambiguous or noisy. 
It uses Partial Label Learning (PLL), where each sample is associated with a set of candidate labels and combines this with contrastive learning and a custom feature encoder.

### Strengths
- The problem of inaccurate labels in multivariate time series is an important and interesting problem setting 
- I believe applying partial label learning to multivariate time series classification is novel 
- The architecture diagrams are well done 
- The paper is mostly well written, and usually easy to follow (though there are some egregious formatting errors --- see weaknesses)

### Weaknesses
- The formatting of the paper is off. In-text citations often do not have a whitespace between the text of the citation and the character preceding it. Many parenthesis are also missing a whitespace.  Some text appears to have formatting errors caused from being copy and pasted from another source, such as in line 114: "contrastive representa- tion space". These formatting issues do not make the paper unreadable, but they show a lack of polish. 

- The method is rather ad hoc; there is no analysis showing that the proposed method will converge to the correct solution. Design choices are not given sufficiently grounded justification 

- I may be wrong, but I believe that most of the compared methods are general MVTSC methods, and aren't designed to handle noisy/ambiguous labels. In that case, it's not surprising that the proposed approach outperforms them when there is significant label noise. I believe the authors should compare against time series methods designed to handle noisy labels [1, 2, 3], or at least explain why comparison with them is not reasonable/feasible. 

- Discussion of related work focus primarily on PLL methods. More discussion should be given to how ambiguous / missing / noisy labels is handled in time series data, in general. 

[1] Nagaraj, Sujay, et al. "Learning under Temporal Label Noise." ICLR 2025. 
[2] Zhou, Zhi, Yi-Xuan Jin, and Yu-Feng Li. "Rts: learning robustly from time series data with noisy label." Frontiers of Computer Science 2024. 
[3] Atkinson, Gentry, and Vangelis Metsis. "Identifying label noise in time-series datasets." Adjunct proceedings of the 2020 ACM international joint conference on pervasive and ubiquitous computing and proceedings of the 2020 ACM international symposium on wearable computers. 2020.
[4] Cui, Beilei, et al. "Rectifying noisy labels with sequential prior: Multi-scale temporal feature affinity learning for robust video segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.

### Questions
- Which compared methods were designed to handle ambiguous / noisy / missing labels?

### Soundness
2

### Presentation
2

### Contribution
2
