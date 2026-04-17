# Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context–Detail Interaction

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Accurate detection of clinically meaningful events in healthcare time-series data is crucial for reliable downstream analysis and decision support. However, most existing methods struggle to jointly localize event boundaries and classify event types; even detection transformer (DETR)-based approaches show limited performance when confronted with extremely sparse events typical of clinical recordings. To address these challenges, we propose a coarse-to-fine detection framework combining a global context explorer, a local detail inspector, and an adaptive gating module (AGM) that fuses multiple label perspectives. The AGM uses transformed labels—encoding event presence and temporal position—to improve learning on sparse events. This design acts as a switch that selectively activates detailed feature extraction only when an event is likely, thereby reducing noise and improving efficiency in sparse settings. We evaluate our framework on diverse healthcare datasets—including arrhythmia detection, emotion recognition, and human-activity monitoring—and demonstrate substantial performance gains over existing DETR-based models, with particularly strong improvements in sparse event detection. With precise and robust event detection, our framework enables interpretation and actionable insights in real-world clinical applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a DETR-inspired framework for sparse event detection in healthcare time-series. It introduces a dual-branch encoder—a Global Context Explorer and a Local Detail Inspector—fused through an Adaptive Gating Module (AGM) that dynamically balances global and local information. To handle extreme event sparsity, the authors design Positional Gaussian Injection for soft temporal supervision and a Conditional Gate Scaler to mitigate class imbalance. Experiments on several physiological datasets show consistent improvements over DETR-based baselines, achieving more precise event localization and better data efficiency.

### Strengths
The paper’s novelty is moderate but meaningful. While the framework builds on DETR and existing ideas such as global–local fusion, adaptive gating, and soft temporal labels, these components are integrated with notable care. The design is elegant and well aligned with the characteristics of healthcare time-series—sparse, imbalanced, and context-dependent. The adaptive gate offers interpretable behavior, and the Gaussian label smoothing plus conditional weighting show clear practical insight. Overall, the work is not conceptually radical, but it represents a thoughtful and well-executed adaptation that brings tangible progress to sparse event detection in medical data. 

The presentation is clear and well-structured, making the technical ideas easy to follow. Overall, this is a thoughtful and well-executed adaptation that combines solid engineering with practical relevance.

### Weaknesses
The main weaknesses are twofold. 

First, the method’s generalization remains limited. It still requires dataset-specific fine-tuning, and the learned gating behavior may not transfer well across domains with different event sparsity or temporal patterns. 

Second, the architecture is relatively heavy, combining dual-branch encoders, gating, and DETR-style decoding, which raises computational cost and makes real-time healthcare deployment challenging.

### Questions
NA

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
3

### Summary
This paper introduces a lightweight deep learning model for time-series event detection with a novel evaluation metric called Affiliation F1 score (AF-F1). The proposed model builds upon Chronos's tiny-bolt architecture and consists of three primary components: a feature extractor (comprising GCE for capturing long-range temporal dependencies and LDI for extracting local temporal patterns), an Adaptive Interaction Module (AGM) containing CGS (adjusts feature importance based on event presence) and PGI (encodes positional information), and a query-based decoder for event prediction. The AF-F1 metric evaluates model performance at the event level by measuring how well predicted event segments align with ground-truth segments, considering both class labels and temporal boundaries. The model achieves a compact size of 11.0M-13.3M parameters and demonstrates strong performance on event detection tasks.

### Strengths
1. The introduction of AF-F1 as an event-level metric is a significant contribution, addressing a critical limitation of point-wise metrics by considering both temporal alignment and class accuracy, which is particularly valuable for event detection tasks.
2. The separation of components for different temporal scales (GCE for long-range dependencies, LDI for local patterns) demonstrates thoughtful design that effectively captures diverse temporal features.
3. The use of both AF-F1 and mAP as complementary metrics provides a more complete assessment of model performance, capturing not only event correctness but also the quality of prediction ranking based on confidence scores.

### Weaknesses
1. The evaluation appears to be limited to a single dataset (implied but not explicitly stated), which may not adequately demonstrate generalizability across different domains or event types.
2. There's no detailed analysis of model performance across different event types (e.g., short vs. long events, frequent vs. rare events), which could reveal important performance characteristics and limitations.

### Questions
No more questions

### Soundness
3

### Presentation
3

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
This paper proposes a coarse-to-fine detection framework for enhancing sparse event detection in healthcare time-series data. The framework combines a Global Context Explorer (GCE) and a Local Detail Inspector (LDI) with an Adaptive Gating Module (AGM). The AGM uses Positional Gaussian Injection (PGI) for refined temporal localization and a Conditional Gate Scaler (CGS) for adaptive rebalancing of sparse event features. The model leverages multiple label perspectives during training and employs a DETR-based architecture for joint event type and boundary prediction. Evaluated on diverse healthcare datasets (arrhythmia, emotion, activity monitoring), the proposed method demonstrates substantial performance gains, particularly for sparse events, over existing DETR-based baselines.

### Strengths
1.  **Targeting a Critical Problem (Sparse Event Detection in Healthcare Time-Series):** The paper addresses a highly relevant and challenging problem in healthcare—detecting rare, clinically meaningful events with precise boundaries in complex time-series data. This problem is clearly articulated as a limitation of existing methods.
2.  **DETR-based End-to-End Framework:** Adopting and adapting the DETR architecture for time-series event detection, combined with a coarse-to-fine feature extraction, offers an elegant end-to-end solution. The tailored Hungarian matching cost for temporal events is a thoughtful adaptation.
3.  **Comprehensive Evaluation:** The framework is rigorously evaluated on four diverse healthcare datasets across three different tasks (arrhythmia, emotion, activity recognition) and various class evaluation scenarios, demonstrating consistent performance gains over multiple strong DETR baselines. The use of PW-F1, AF-F1, and mAP provides a holistic view of performance.

### Weaknesses
1.  **Ambiguity in Time-Series Windowing and Event Truncation:** The use of fixed 10-second windows with a 1-second stride for long-duration healthcare time-series (e.g., Holter ECG) raises concerns. The paper lacks a clear explanation of how events that span across window boundaries are handled, potentially leading to event truncation or inaccurate boundary detection, which contradicts the goal of "precise event boundary detection." This is a critical aspect for reproducibility and clinical applicability.


2.  **Underspecified "Adaptive Gating Module" (AGM) Mechanism:** The core component, the AGM, specifically the precise mechanism for generating the `g` gate tensor (e.g., what layers/activation functions are applied to which input features to produce `g` as a R $\in$ B×τ×1  tensor), is not explicitly detailed. This lack of clarity hinders the understanding of a key novelty and impacts reproducibility.


3.  **Lack of Clarity in GCE/LDI "Coarse-to-Fine" Distinction and TCN Details:** The distinction between "global" (GCE) and "local" (LDI) features is primarily attributed to TCN kernel sizes (7 vs. 3). However, **without specifying the dilation rates** used in their TCN-Attention mechanisms, it's unclear if this difference is substantial enough to justify the "coarse-to-fine" claim, especially given the inherently short window length. Furthermore, the detailed operation and structure of the "FFN-based alignment layer" are insufficiently explained.

4.  **Information Dispersal and Lack of Self-Containedness:** Several crucial details, such as full metric definitions, complete quantitative results tables (beyond highlights), detailed dataset class distributions for specific class evaluation scenarios (e.g., "Class 3", "Class 6", "Class 15" for MIT-BIH), and specific hyperparameters/adaptation methods for baselines, are mostly relegated to the Appendix. This significantly impedes the reader's ability to grasp the main content and evaluate the results. Key figures like Table 1, Table 2, and Figure 2 (which is overly complex and lacks sufficient labeling for intermediate representations like `z` and `h`) are not self-contained, forcing constant reference to the appendix and hindering streamlined understanding.

5.  **Inconsistent and Undefined Notation:** Several fundamental variables used in equations, such as the batch size `B` (Eq. 2), the total number of events `N` (Eq. 1), and the subscript `b` in $w_b(t)$ (Eq. 6), are used without explicit definition early in the methodology. Furthermore, the input definition for `X` (Eq. 1) does not include the batch dimension `B`, leading to an inconsistency with later equations that use `B`.

6.  **Limited Discussion on Clinical Interpretability/Actionability:** While the abstract and introduction mention "actionable insights" and "reliable interpretation" in real-world clinical applications, the paper's analysis is predominantly quantitative. A deeper discussion or qualitative analysis on *how* the improved event detection truly translates into concrete clinical benefits (e.g., enabling earlier alerts, more precise diagnosis, or better treatment planning) would strengthen its real-world impact claim.

### Questions
1.  **Event Handling at Window Boundaries:** Given the 10-second windowing approach for continuous time-series, how are ground-truth events that are longer than 10 seconds or that span across window boundaries specifically annotated and handled during training and evaluation? Does this approach risk truncating events or causing ambiguities in boundary detection, especially for a framework aiming for "precise event boundary detection"?

2.  **Precise Generation of Gate Tensor `g`:** The paper states that "the AGM produces a gate tensor `g`... that controls the relative contributions of LDI and GCE" . However, the exact mechanism or specific layers (e.g., what output features, linear layers, and activation functions) through which this `g` tensor is computed from the features (e.g., after PGI and CGS) is not explicitly described. Please clarify the precise generation process of `g`.

3.  **TCN Dilations for GCE and LDI & Alignment Layer Details:** To substantiate the "coarse-to-fine" distinction between GCE and LDI, please specify the dilation rates used in their respective TCN-Attention mechanisms. This is crucial for understanding how kernel sizes 7 and 3 translate into genuinely "global" and "local" effective receptive fields. 

4.  **Clarification of Notation and Consistency:** Please provide clear and explicit definitions for all variables used in equations (e.g., `N` in Eq. 1, `B` in Eq. 2) early in the methodology section. Ensure consistency in input definitions, specifically by including the batch dimension `B` in the definition of input `X` (Eq. 1). Also, clarify the meaning of the subscript `b` in $w_b(t)$ (Eq. 6).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a novel approach for enhancing sparse event detection in healthcare time-series data through the use of an adaptive gate mechanism. The proposed method combines global context exploration, local detail inspection, and an adaptive gate module to improve the precision of event localization in medical datasets, such as ECG recordings. The authors have conducted experiments on various healthcare datasets and demonstrate substantial improvements over existing methods in terms of sparse event detection performance.

### Strengths
The introduction of the adaptive gate mechanism (AGM) is a significant contribution to sparse event detection. By leveraging both global and local perspectives, the method addresses key challenges in event localization. The explanation of the proposed framework and the AGM module is clear and well-structured. The authors provide sufficient details on the model's design and operational principles. The experiments on diverse healthcare datasets, including ECG signals, are comprehensive and effectively demonstrate the superiority of the proposed method over existing approaches. The results are promising and support the validity of the approach.

### Weaknesses
How does the proposed method handle cases with missing or incomplete data, particularly in sparse time-series recordings?
Could the adaptive gate mechanism be further enhanced by incorporating additional types of context information (e.g., patient demographics, clinical history)?

### Questions
How does the proposed method handle cases with missing or incomplete data, particularly in sparse time-series recordings?
Could the adaptive gate mechanism be further enhanced by incorporating additional types of context information (e.g., patient demographics, clinical history)?

### Soundness
4

### Presentation
4

### Contribution
4
