# HiTeA: Hierarchical Temporal Alignment for Training-Free Long-Video Temporal Grounding

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Temporal grounding in long, untrimmed videos is critical for real-world video understanding, yet it remains a challenging task owing to complex temporal structures and pervasive visual redundancy. Existing methods rely heavily on supervised training with task-specific annotations, which inherently limits their scalability and adaptability due to the substantial cost of data collection and model retraining. Although a few recent works have explored training-free or zero-shot grounding, they seldom address the unique challenges posed by long videos. In this paper, we propose HiTeA (Hierarchical Temporal Alignment), a novel, training-free framework explicitly designed for long-video temporal grounding. HiTeA introduces a hierarchical temporal decomposition mechanism that structures videos into events, scenes, and actions, thereby aligning natural language queries with the most appropriate temporal granularity. Candidate segments are then matched with queries by leveraging pre-trained vision–language models (VLMs) to directly compute segment–text similarity, thereby obviating the need for any task-specific training or fine-tuning. Extensive experiments on both short- and long-video benchmarks show that HiTeA not only substantially outperforms all existing training-free methods (e.g., achieving 44.94% R\@0.1 on TACoS, representing an absolute gain of 12.4%) but also achieves competitive performance against state-of-the-art supervised baselines under stricter metrics. The code is available at https://anonymous.4open.science/r/HiTeA_code.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HiTeA, a training-free framework for temporal grounding in untrimmed videos. The core of HiTeA is its Hierarchical Temporal Decomposition (HTD) mechanism, which structures a video into multi-scale temporal units (events, scenes, actions). It leverages pre-trained vision-language models to match text queries with candidate segments without any task-specific training and employs a refinement module to generate final predictions. The authors report extensive experiments on Charades-STA, ActivityNet-Captions, Ego4D-NLQ, and TACoS datasets.

### Strengths
1. The manuscript is well-written and clearly structured. 

2. The supporting figures, particularly the conceptual diagrams, are well-designed and effectively aid in understanding the proposed framework.

### Weaknesses
1. The proposed hierarchical relationship among events, scenes, and actions appears to be somewhat problematical and lacks a strong empirical or theoretical justification. For instance, in real-world videos, the relationship can be more complex and non-hierarchical; a single scene might encompass multiple distinct events. A more detailed rationale or citation for this specific structural choice would strengthen the methodological foundation.

2. The experimental evaluation would benefit from a direct comparison with other recent, state-of-the-art training-free methods for temporal grounding or related video-language tasks. [a] Universal Video Temporal Grounding with Generative Multi-modal Large Language Models, arXiv 2025.  [b] Zero-Shot Video Grounding With Pseudo Query Lookup and Verification, TIP 2021.

3. In the ablation study of Table 3, the performance improvements from the individual components of the Hierarchical Temporal Decomposition (HTD) on the TACoS dataset seem relatively marginal, and in some configurations, a component even appears to degrade performance. The manuscript should provide an in-depth analysis discussing the potential reasons for this observed instability or limited impact. Furthermore, the ablation study appears incomplete as it does not present the results of using the Candidate Refinement module in isolation (e.g., on uniformly sampled segments). Showing this baseline is critical to disentangle and quantify its specific contribution to the overall performance.

4. The generalization and robustness of the method could be more convincingly demonstrated by including evaluations on the QVHighlights dataset. A key challenge of this dataset is that a single query can correspond to multiple moments. Testing on QVHighlights would provide valuable insights into the model's capability to handle such complex, multi-instance grounding scenarios.

5. The justification for the specific choice of pre-trained models for feature extraction (ViT-B/32 for events, DINO-v2 for scenes, RAFT-Large for actions) is insufficient. The authors should clarify the specific properties or prior evidence that make these particular models representative of their respective hierarchical levels. A discussion on whether alternative models were considered and why these were chosen would strengthen the methodological design.

6. The inference cost of utilizing five different pre-trained models is a significant practical concern, especially for processing long videos. The manuscript should include a discussion, and ideally a quantitative analysis, of the computational efficiency and inference speed. This is a critical factor for real-world applicability and should be transparently reported.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents HiTeA, a training-free framework for temporal grounding in long, untrimmed videos. The key innovation is a Hierarchical Temporal Decomposition (HTD) module that structures videos into multi-scale temporal units (events, scenes, actions) using off-the-shelf feature extractors. Candidate segments are scored using frozen vision-language models (VLMs), and a Candidate Refinement module produces final predictions. The method achieves strong results on both long-video benchmarks (Ego4D-NLQ, TACoS) and short-video datasets (Charades-STA, ActivityNet-Captions).

### Strengths
1) **Well-motivated problem.** The paper targets a real gap in current approaches—their difficulty scaling to long, temporally complex videos without costly supervised training—and clearly motivates a training-free alternative.

2) **Strong empirical performance.** HiTeA delivers compelling results, especially on long videos: 44.94% R@0.1 on TACoS (a +12.4% absolute gain).

3) **Robust generalization.** The method transfers across diverse domains and durations—from short clips (≈29 s, Charades-STA) to extended egocentric footage (≈472 s, Ego4D-NLQ)—indicating strong robustness to video type and length.

### Weaknesses
1) **Limited technical novelty.**
   While the system is effective, its building blocks are largely established:
   - Hierarchical temporal decomposition has prior art.
   - Zero-shot use of pretrained VLMs is well explored.
   - Change-point detection (e.g., PELT) is off-the-shelf.
   - The overall pipeline follows a familiar proposal-generation + scoring pattern.

2) **Method complexity and engineering burden.**
   The framework stitches together multiple components (three feature extractors, VideoCLIP filtering, VLM scoring, candidate refinement) and exposes many hyperparameters (α, β, λ, k, thresholds). This raises concerns about:
   - **Generalizability** beyond the tested datasets,
   - **Sensitivity** to hyperparameters (only β and λ receive analysis), and
   - Whether the gains stem from **principled design** versus careful tuning and component stacking.

3) **Unclear rationale for hierarchy effects across lengths.**
   Table 7 shows hierarchical merging helps long videos but harms short ones; the discussion in A.4.2 reads as a post-hoc explanation rather than a principled, testable design hypothesis.

4) **Domain specificity of the three-level hierarchy.**
   The event–scene–action granularity—especially the scene level—appears tailored to the evaluated domains. It’s unclear how this hierarchy transfers to other video genres (e.g., sports broadcasts, surveillance, instructional videos) without re-defining levels or retuning cues.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HiTeA, a training-free framework for long-video temporal grounding. The method tackles the challenge of locating temporal segments in long, untrimmed videos corresponding to natural language queries—without requiring any supervised training.

### Strengths
The Hierarchical Temporal Decomposition (HTD) effectively aligns video segments with textual queries at multiple granularities, bridging the gap between semantic understanding and temporal localization.

### Weaknesses
1. **Feature extraction choices** – Why were **ViT**, **DINO**, and **FLOW** specifically chosen for extracting event-level, scene-level, and action-level features, respectively? How do these choices contribute to the complementary strengths of each temporal layer?
2. **Inference efficiency** – The paper discusses computational complexity analytically, but could you provide a **runtime comparison** with representative baselines on long-video datasets to empirically demonstrate HiTeA’s scalability?

### Questions
Please refer to the points listed under *Weaknesses*.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes HiTeA, a training-free framework for Video Temporal Grounding (VTG) in long videos. The authors argue that existing methods either rely on expensive supervised annotations or perform poorly when handling the complex temporal structures and content redundancy in long videos. The core contribution of HiTeA is a Hierarchical Temporal Decomposition (HTD) mechanism. This mechanism simulates human search strategies, decomposing the video into three granularities: 'events', 'scenes', and 'actions'. It first uses various pre-trained models (like ViT, DINO, RAFT) to extract multi-level features and generate multi-scale candidate segments. Subsequently, the framework adopts a two-stage strategy: first, it pre-filters candidates using a lightweight VideoCLIP, and then it scores the candidates using a frozen large vision-language model (VLM). Experiments show that HiTeA significantly outperforms existing zero-shot (ZS) methods on multiple short- and long-video benchmarks. For example, on the TACOS dataset, it achieves a 12.4% absolute performance gain. More importantly, on long-video benchmarks like Ego4D-NLQ, the method achieves performance competitive with or even superior to state-of-the-art (SOTA) supervised methods.

### Strengths
-  **Significance and Impact:** The paper's strongest merit lies in its **groundbreaking results**. As a **training-free** framework, HiTeA's mIoU (8.12%) on Ego4D-NLQ (a highly challenging long-video benchmark) **surpasses** all listed fully-supervised SOTA methods (e.g., UniVTG's 4.91%). This challenges the traditional notion that SOTA performance must rely on dense supervision and holds significant implications for the VTG field.
- **Originality and Intuitive Method:** The core **HTD (Hierarchical Temporal Decomposition)** idea is highly novel. It does not blindly segment the video; instead, it emulates the human search intuition of progressing from 'event' to 'scene' to 'action' and elegantly maps this intuition to different pre-trained features (ViT for semantics, DINO for structure, RAFT for motion). This is a very elegant design.
- **High-Quality and Rigorous Experiments:**
    * **Main experiments** convincingly demonstrate its SOTA performance on both long and short videos.
    * The **ablation study (Table 3)** is solid. It clearly quantifies the contribution of each layer in HTD (Event, Scene, Action) and the necessity of the CR module, proving the effectiveness of each component in the framework.
    * The **efficiency analysis (Table 4)** adds significant practical value, showing that VideoCLIP pre-filtering reduces the computational load on Ego4D by 93.9%, addressing the pain point of slow VLM inference on long videos.

-  **Clarity:** As mentioned, the paper is well-written and logically organized. **Figures 1 and 2**, in particular, intuitively illustrate the difference between HiTeA and traditional methods, as well as its complex internal workflow.

### Weaknesses
My criticisms are  primarily focus on the rigor of its claims and the transparency of its details.

-   **Contradiction between the motivation for Score Fusion (Sec 3.5) and the results (Sec 4.4):**
    * **Issue:** In Sec 3.5, the authors introduce $s_{final}=\lambda \cdot s_{vlm}+(1-\lambda) \cdot s_{clip}$, motivated by the idea of fusing the VLM's semantic information with VideoCLIP's structural information.
    * **Contradiction:** However, in the parameter analysis in Sec 4.4 (Fig 4, Right), the optimal value for $\lambda$ is set to **0.99**. This implies that the contribution of $s_{clip}$ is almost negligible (1%).
    * **Suggestion:** The authors also admit in the appendix that $s_{clip}$ mainly serves as a "tie-breaker". It is suggested that the authors state this more candidly in the main text (Sec 3.5)—i.e., that the VLM is absolutely dominant and CLIP is only used for filtering and tie-breaking—rather than presenting it as a "balanced fusion." This would make the paper's motivation more rigorous.

-   **A key implementation detail is hidden in the appendix:**
    * **Issue:** Appendix A.4.2 (Table 7) reveals that for short videos (Charades-STA, ANet), the **Hierarchical Merging (HM) module is disabled** because it slightly harms performance.
    * **Suggestion:** This is a crucial implementation detail that reveals the framework's adaptability (prioritizing structure for long videos, diversity for short videos). This should not be hidden in the appendix. It is suggested that the authors move this detail (i.e., that HM is disabled for short videos) to the main paper's experimental setup (Sec 4.1) or method (Sec 3.3) for discussion.

### Questions
My main concerns are articulated in the "Weaknesses" section and are more about presentation and rigor. Here they are converted into specific questions for the authors' rebuttal:

-   **Regarding $\lambda=0.99$:** Given the optimal value of $\lambda$ is 0.99, this suggests the contribution of $s_{clip}$ is minimal. Does this imply that the "fusion" motivation proposed in Sec 3.5 is not practically realized, and the true role of $s_{clip}$ (as mentioned in the appendix) is merely for filtering and tie-breaking? Please clarify the exact role of $s_{clip}$ in the final score.
-  **Regarding the disabling of HM:** You mention in Appendix A.4.2 that the HM module is disabled for short videos because preserving the diversity of independent features is more effective than enforcing a hierarchy. Does this mean the best practice for HiTeA involves using **two different configurations based on video length (long/short)**? Could you more explicitly state this (very reasonable) design choice in the main text?
-  **Regarding the complexity of CR and HTD:** Your framework appears to have two "merging" steps: (1) the hierarchical merging in HTD (Alg 1) and (2) the progressive merging in CR (Alg 2). This makes the method (especially the CR module) seem somewhat complex. Table 3 shows that CR is crucial, but could you provide further intuition on why the merging in HTD is insufficient, necessitating a second round of merging in CR?

### Soundness
3

### Presentation
3

### Contribution
3
