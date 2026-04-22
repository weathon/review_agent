# Textual and Temporal-Guided Feature Decoupling for Video Temporal Grounding

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Recent approaches to Video Temporal Grounding (VTG) predominantly rely on CLIP-based representations, often augmented with visual encoders such as SlowFast or C3D to enhance temporal modeling. However, the prevailing “concat-then-project” paradigm disrupts the inherent alignment between CLIP's visual and textual modalities and undermines the temporal modeling capabilities of the additional video encoder. To address these, we propose FDAP, a plug-and-play Feature Decoupling and Aggregation Paradigm. FDAP introduces two key components: a Textual-Guided Feature Decoupling Module (TGFDM) that preserves CLIP’s cross-modal alignment and SlowFast’s temporal modeling via independent attention maps, and a Dual-branch Feature Aggregation Module (DFAM) that dynamically adjusts feature weights during aggregation based on query-specific needs. Extensive experiments across four VTG methods (M-DETR, TR-DETR, CG-DETR, Flash-VTG) on three benchmark datasets (QVHighlights, Charades-STA, TACoS) demonstrate consistent performance gains, \emph{e.g.}, a 3\% improvement in M-DETR’s R1@0.7 metric. With minimal overhead (0.2M additional parameters), FDAP advances VTG feature modeling and generalizes effectively across diverse methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles Video Temporal Grounding and argues that the common “concat-then-project” way of fusing CLIP features with an auxiliary video encoder harms both CLIP’s cross-modal alignment and temporal modeling. It proposes FDAP (Feature Decoupling and Aggregation Paradigm) with two modules: (1) a Textual-Guided Feature Decoupling Module (TGFDM) that builds separate textual-guided and temporal-guided attention maps to enrich CLIP/SlowFast streams without collapsing them, and (2) a Dual-branch Feature Aggregation Module (DFAM) that uses Adaptive Weight Aggregation Modules (AWAM) to fuse the four enhanced streams with query-adaptive weights. FDAP is plugged into M-DETR, TR-DETR, CG-DETR, Flash-VTG and evaluated on QVHighlights, Charades-STA, TACoS, reporting consistent but modest gains and negligible extra parameters.

### Strengths
1. Clear target and modular design. The work squarely addresses the “concat-then-project” pitfall and proposes a clean decouple-then-fuse alternative that is easy to bolt onto popular VTG heads.
2. Query-adaptive aggregation. DFAM/AWAM explicitly models sample-level preference between textual and temporal cues, aligning with the task’s intuition; the paper visualizes these preferences.
3. Breadth of integration. FDAP is evaluated across four heads and three datasets with consistent gains, and reports seed averages for robustness.
4. Cost awareness. The paper quantifies parameter and FPS overhead (albeit without memory metrics).

### Weaknesses
1. Attention-pooling + MLP is fairly standard. Pooling attention maps and feeding a small MLP to predict dynamic fusion weights is a common adaptive-fusion pattern. Please present it as an engineering choice, cite close precedents, and include ablations against alternatives (gating, softmax/Dirichlet weighting, mixture-of-experts) to demonstrate why this variant is preferable here.
2. “Temporal-Guided Attention” vs. preserving SlowFast’s intrinsic dynamics. Eq. (5) defines a text-conditioned frame self-attention, where the affinity involves the query text. This is not the same as preserving SlowFast’s query-agnostic motion modeling. Please clarify whether the intent is to expose SlowFast cues under textual guidance or to truly preserve them. Consider: (i) a query-agnostic temporal attention variant as a control, and (ii) analyses showing how much the query changes SlowFast features.
3. Mixed comparison protocols reduce fairness/interpretability. The tables mix (i) prior-paper numbers and (ii) your re-implementations, likely under different training setups (data, augmentation, audio, pretraining). This makes it hard to attribute gains. Please separate these blocks, and provide a unified reproduction with the same recipe and modalities for all baselines, reporting mean±std (multiple seeds) and basic significance tests.
4, sqrt(L), sqrt(T) scaling needs explicit rationale. Unlike vanilla attention’s sqrt(d), Eqs. (4)/(5) scale by sequence-length factors. Please state the softmax axes precisely and justify why it is appropriate for your affinity definitions. 
5. T×T complexity isn’t reflected in the efficiency results. The module implies 𝑂(𝑇2) temporal attention, but Table 5 only reports aggregate FPS/params. Please plot FPS/peak memory as T grows. 
6. Constraints on AWAM weights (lambda1, lambda2). It’s not specified whether the weights are non-negative or normalized (sum to 1). Please state the parameterization (e.g., softmax over two logits with temperature), any normalization/regularization and numerical safeguards (epsilon/clamp), and provide weight distributions or stability checks.

### Questions
Please see weakness part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses key limitations in Video Temporal Grounding (VTG) caused by the common “concat-then-project” feature fusion strategy, which disrupts CLIP’s cross-modal alignment and degrades SlowFast’s temporal modeling. The authors propose FDAP, a lightweight, plug-and-play paradigm consisting of two modules: (1) TGFDM, which decouples CLIP and SlowFast features and enhances them via textual- and temporal-guided attention, preserving their respective strengths; and (2) DFAM, which dynamically fuses these features using query-adaptive weights.

### Strengths
1. the motivation is clear
2. the writing is clear and easy to understand

### Weaknesses
1. Related work is an important part of the article and should be placed in the main text, not in the appendix.

2. According to the paper, the text features Tcl extracted by clip are aligned with the visual features Vcl, so calculating their similarity Mcl is understandable. However, the text features Tcl extracted by clip are not aligned with the features Vsf extracted by slowfast, so it's difficult to understand why text features are introduced in Equation 5, which seems to contradict the starting point of this paper.

3. The description lacks many details. For instance, it only mentions that 0.2M parameters are introduced—but which specific components do these parameters belong to, and how exactly are they calculated? Additionally, in Table 3, when using only the DFAM or TGFDM module individually, how is the forward pass performed? After all, these two modules appear to be closely interdependent.

### Questions
1. The authors should consider how to better verify that the proposed module achieves the desired effect. It is difficult to prove that the three limitations of the current work have been solved by quantitative experiments and visualization of a few samples alone. 

2. Why did the last column in Table 3 undergo multiple experiments, while the preceding columns only underwent one? Why did Table 1 use the mIou metric on the Charades-STA dataset, while Table 3 uses the mAP metric?

Overall, I currently tend to give this article a negative review. However, I would consider revising my score if the author can address my concerns during the rebuttal process.

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
3

### Summary
This paper introduces FDAP (Feature Decoupling and Aggregation Paradigm) to address limitations of methods that use the "concat-then-project" approach in  Video Temporal Grounding (VTG) tasks. Specifically, it uses two main modules: the Textual-Guided Feature Decoupling Module (TGFDM) and the Dual-branch Feature Aggregation Module (DFAM). These modules preserve CLIP's cross-modal alignment and allow dynamic aggregation based on query-dependent preferences. Extensive experiments across various VTG methods and benchmarks demonstrate FDAP's effectiveness in improving performance with minimal computational overhead. However, the description of the query is not clear enough, and the relationship between the textual features and the query is quite confusing.

### Strengths
- Textual-Guided Feature Decoupling Module: Effectively preserves the cross-modal features alignment between CLIP's visual and textual features and maintains SlowFast's temporal modeling, ensuring better performance in the VTG task.
- Dual-branch Feature Aggregation Module: Dynamically adjusts the aggregation of textual and temporal features based on the specific preferences of each query, incorporating dynamic adaptive weights to analyze different query preferences for information and the impact of text temporal connectives on the weights.
- Plug-and-Play flexibility and generalizability: It brings minimal computation(only 0.2M parameters) and works across different architectures with preserved original experimental protocols.

### Weaknesses
- This paper does not clarify the initialization and updating strategy of the query. Given that the AWAM plays a crucial role in the model's ability to adjust feature fusion based on the query, the lack of detailed explanation on how the query is initialized and updated is highly ambiguous. 
- In Section 2.2, the paper mentions the interaction between query and CLIP's visual features, but there is some inconsistency between the formulas and the main text. Specifically, it's unclear which specific textual features from CLIP are interacting with the visual features. The formulas and the paper description seem to imply different interpretations, and this inconsistency should be addressed to provide a clearer understanding of how textual features and the query are aligned and integrated.

### Questions
- What is the relationship between the query and the text features?
- How is the query initialized? Is it learnable?
- How can it be applied to other models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents FDAP (Feature Decoupling and Aggregation Paradigm), a plug-and-play module designed to improve feature modeling in Video Temporal Grounding (VTG). The authors argue that most prior works follow a ‘concat-then-project’ paradigm when combining CLIP visual features with additional video encoders (e.g. SlowFast), which disrupts CLIP’s multimodal alignment and weakens temporal modeling. FDAP consists of two modules: 1) Textual-Guided Feature Decoupling Module (TGFDM) generates separate textual-guided and temporal-guided attention maps to preserve both CLIP alignment and SlowFast temporal capability; 2) Dual-branch Feature Aggregation Module (DFAM) employs three Adaptive Weight Aggregation Modules (AWAMs)to dynamically assign query-dependent weights between textual and temporal branches. FDAP is incorporated into four baselines, including M-DETR, TR-DETR, CG-DETR, and Flash-VTG, and evaluated on three benchmarks. The results show consistent gains and reduced variance with only 0.2M additional parameters.

### Strengths
**[S1]** The motivation is clearly presented, and the proposed method is clearly designed to address the presented issues.

**[S2]** Implementation details and experimental settings are well-reported, including fixed random seeds, resource settings, and parameter overhead.

**[S3]** The writing is clear and easy to follow.

### Weaknesses
**[W1]** Applicability
- The main concern is that the proposed method can be applied only to the ‘dual-encoder (e.g. CLIP)-based’ frameworks. Recent works for the VTG task have been actively employing the LLM-based frameworks due to their high performance and generalizability. While the reviewer acknowledges that there are several trade-offs in comparing the dual-encoder-based and LLMs-based approaches (e.g. computational efficiency vs. performance), the presented results and comparisons may not be significant to highlight the proposed method in video temporal grounding.

**[W2]** Novelty
- Cross-modal attention-based decoupling and weighted aggregation are quite incremental. Presenting the differences from similar works in multimodal learning would be helpful to argue novelty.

**[W3]** Missing reference
- Many of the recent works (presented in recent venues such as CVPR 2025) are missing.
- Additionally, all LLM-based approaches are missing. This can highlight the limited applicability of the proposed method.

### Questions
**[Q1]** Please see weaknesses.

**[Q2]** Alignment between $V_\text{sf}$ and $T_\text{cl}$
- In the TGFDM, textual-guided attention between $V_\text{cl}$ and $T_\text{cl}$ can be reasonably obtained since they are extracted from co-trained encoders. However, obtaining the precise attention between $V_\text{sf}$ and $T_\text{cl}$ may not be guaranteed. How can we confirm this attention is appropriately measured based on the correspondence between the video and textual features?

### Soundness
2

### Presentation
2

### Contribution
2
