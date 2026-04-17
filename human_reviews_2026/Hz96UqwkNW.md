# DEEP STRATIGRAPHIC INFERENCE: A TWO-STAGE TRAINING CURRICULUM AND HEURISTIC GATE FOR HIGH-PRECISION CHANGE-POINT DETECTION

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Accurate geological characterization of subsurface reservoirs using well log data is essential for high-impact applications such as carbon sequestration and environmental monitoring. This task, which we term Deep Stratigraphic Inference, requires the high-precision localization of change-points within noisy time series. While transformers are powerful, a naive end-to-end regression approach fails due to training instabilities.  To address this, we propose CURT-Point (Curriculum-trained Regression Transformer for Point Localization), a comprehensive framework for robust time series localization. CURT-Point's core is a Two-Stage Training Curriculum that first pre-trains the transformer as an expert classifier, then fine-tunes a specialized regression head. To maximize robustness, the framework is completed by a post-processing Hybrid System incorporating a Heuristic Gate, which achieves the best overall performance by intelligently ensembling an attention-based regression with a robust peak-finding heuristic, both derived from the same unified Transformer backbone. The effectiveness of this framework hinges significantly on two additional advancements: we show that a fusion of specific data preprocessing with an innovative constrained data augmentation tactic is crucial for dealing with real-world signal flaws, and we establish that Rotary Positional Embeddings (RoPE) play a crucial role in attaining high performance.
Our final Hybrid System, validated on three real-world well-log datasets of increasing complexity, achieves state-of-the-art recall and median errors, providing a generalizable workflow for high-precision time series localization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, authors proposed CURT-Point, a holistic framework for robust and accurate localization in geological time series well-log data. The frameworks includes a Two-Stage Training Curriculum with a final Hybrid System that employs a simple Heuristic Gate to correct outlier predictions. They showed that a foundation of targeted data augmentation and RoPE encodings is critical. The final system achieves state-of-the-art results.

### Strengths
1. The proposed framework achieves strong performance across three datasets.
2. The paper proposes comprehensive ablations for different components.

### Weaknesses
1. Overreliance on heuristic designs. The main parts of the system rely on rule-based or heuristic procedures, which limit the novelty of the framework that targets an automated machine learning approach.

2. The writing needs to be improved, for example, 1) The introduction repeats introducing the name, CURT-Point, twice. 2) Several jargon terms are used without sufficient explanation. 3) Typos in page 2, line 63. 4) In Table 3, there are items that are wrongly highlighted.

3. Poor methodological writing in Section 3. The Methodology section mainly restates the conceptual overview, similar to the Introduction, with insufficient algorithmic or implementation detail. The overall organization makes it more difficult to follow how each component really works. Many designs that were not finally adopted are also included in this section (e.g, learnable gate shown in Figure 2).

4. Low-quality figures. Many figures are low-resolution and use too-small fonts, making them hard to read.

### Questions
1. Have you tested other backbones rather than Transformer to confirm that your framework is highly generalizable, as you claimed?

2. What motivates the simultaneous use of classification and regression tasks? From the perspective of your target problems and their physical meaning, why does it need two similar tasks as outputs?

3. Due to the writing issues in Section 3, it is confusing how the four models in Table 1 differ. Based on my understanding, the Heuristic-Based Method uses only the classification head, while the Two-Stage End-to-End variant uses only the sequence output head. Is this correct?

4. With CE loss only, how is the Attention-Based Regression Head trained? Section 3.5 states CE is for classification, and L1 is for localization precision.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CURT-Point, a framework for high-precision change-point detection in noisy time series, with a specific application to localizing geological formation tops in well-log data. The authors propose a multi-faceted approach. The core contributions include: 1) A two-stage training curriculum where a PatchTST-based Transformer is first pre-trained as a classifier and then fine-tuned for regression to avoid training instability. 2) A final hybrid system that uses a rule-based “Heuristic Gate" to ensemble predictions from the high-precision regression head and a more robust heuristic derived from the classifier's logits. 3) A data-centric pipeline involving robust preprocessing and a novel constrained data augmentation strategy. 4) An architectural analysis demonstrating the critical importance of RoPE for this task. The proposed hybrid system is shown to achieve state-of-the-art results on three real-world datasets of increasing complexity.

### Strengths
- The paper addresses a challenging and high-impact real-world problem (Deep Stratigraphic Inference) with a thorough and systematic engineering approach.
- The ablation studies are extensive and provide convincing evidence for the effectiveness of several individual components of the proposed framework, such as the constrained data augmentation (Table 2), the use of RoPE (Table 3), the attention-based regression head (Table 4), and the composite loss function (Table 5).
- The proposed system achieves strong empirical results, demonstrating significant improvements in recall and precision (MedAE) over baselines, particularly on the more complex datasets.

### Weaknesses
- The central claim motivating the paper that a naive end-to-end (E2E) model fails due to training instability, thus requiring the proposed Two-Stage Curriculum is not substantiated by a fair experiment. As detailed in Section 4.3, the 'Naive End-to-End' baseline was trained with a simple loss function (L_CE + L_MAE), while the proposed Two-Stage model benefited from a much more sophisticated composite loss including L1 and Order penalties. This introduces a major confounding variable, making it difficult to attribute the performance difference solely to the training curriculum. The performance gap could be due to the superior loss function, which undermines the paper's core motivation.
- The final proposed system, the Hybrid System (Gate), relies on a rule-based Heuristic Gate with hard-coded thresholds that were explicitly tuned on the validation set of one specific dataset (Dataset C, Appendix A.1.3). The paper provides no sensitivity analysis for these thresholds, and the claim of generalizability is weak. This makes the solution appear brittle and ad-hoc, making it hard to generalize. The reported failure of a Learned-Gate Model further suggests this is a difficult problem that was solved with a non-learned, hand-tuned rule.
- The robustness of the method to hyper parameter choices is questionable. Beyond the Heuristic Gate, there is no sensitivity analysis for the loss function weights, which appear critical for performance (Table 5). This suggests the method may require significant expert tuning for new applications.

### Questions
- Have you tried training a naive, single-stage E2E model with the full composite loss function (including L1 and Order penalties)? If so, how did it perform? Without this control experiment, the claim that the curriculum itself is the key to overcoming training instability is not well-supported.
- Could you provide a sensitivity analysis for Heuristic Gate showing how performance (MedAE, MAE, Recall) varies as these thresholds are changed? How confident are you that this gate would generalize to a new geological basin or a different problem domain (e.g., ECG analysis) without extensive re-tuning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CURT-Point, a two-stage transformer-based framework for high-precision change-point detection in well-log time series. It combines curriculum learning (classification → regression) with a post-hoc heuristic gate to improve robustness.

### Strengths
1. Sensible use of curriculum learning for stabilizing transformer regression.
2. Hybrid gate design seems practical and improves robustness.
3. Well-motivated problem setting.

### Weaknesses
1. The related work section and experiments do not compare against, or even cite, recent SOTA baselines for change-point detection and sequence segmentation, such as: CLASP/TIME2STATE/ISSD, and even classical method AutoPlait. All reported baselines are either simple heuristics or basic neural modules
2. The paper introduces three custom datasets (A/B/C) derived from public geological repositories but with heavy preprocessing and proprietary restrictions (Appendix A.2). Reproducibility is compromised, and comparison with existing public CPD/time-series datasets.
3. In Eq. (3.5), it defines L_spacing and L_order, but it’s unclear how mismatched marker counts are handled. Are missing markers ignored, padded, or penalized? The heuristic gate algorithm (Algorithm 2) mixes confidence thresholds in “feet” with patch size; a clearer rationale for these constants is needed.
4. No computational profiling (training time, inference latency, or model complexity) is presented, even though the method targets practitioner-facing geoscience applications. Given the hybrid multi-stage system, an analysis of computational overhead and scalability versus heuristic or end-to-end baselines is essential.
5. the lack of accessible data/code undermines reproducibility.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
