# MomentSeg: Moment-Centric Sampling for Enhanced Video Pixel Understanding

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Referring Video Object Segmentation (RefVOS) seeks to segment target objects in videos guided by natural language descriptions, demanding both temporal reasoning and fine-grained visual comprehension. Existing sampling strategies for LLM-based approaches typically rely on either handcrafted heuristics or external keyframe models. The former often overlooks essential temporal cues, while the latter increases system complexity. To address this, we propose a unified framework that jointly optimizes Temporal Sentence Grounding (TSG) and RefVOS, naturally incorporating key moment grounding capability. During training, we introduce a novel TSG paradigm that employs a dedicated $\texttt{[FIND]}$ token for key moment identification through temporal token similarity matching, thereby avoiding the need for external timestamp encodings. For inference, we design a Moment-Centric Sampling (MCS) strategy that densely samples informative moments while sparsely sampling non-essential frames, preserving both motion details and global context. To further enhance tracking stability, we develop Bidirectional Anchor-updated Propagation (BAP), which leverages the most relevant moment as start point for high-quality mask initialization and dynamically updates at sampled points to mitigate accumulated errors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This thesis addresses the shortcomings of existing large language model (LMM)-based approaches in the referential video target segmentation (RefVOS) task i.e., either relying on external keyframe models to increase the system complexity or adopting a manual sampling strategy to lose the critical timing cues i.e., proposes a unified framework MomentSeg to achieve the joint optimization of temporal sentence localization (TSG) and RefVOS to improve the video pixel-level comprehension. In the training phase, [FIND] token is introduced to identify text-related key frames by frame-level temporal tokn similarity matching without external timestamp encoding; in the inference phase, a moment-centered sampling (MCS) strategy is designed to densely sample high-relevance key moments and sparsely sample non-key frames, and a bi-directional anchor update propagation (BAP) is proposed to bi-directionally propagate the forward and backward temporal sequence from key frames At the same time, we propose the bidirectional anchor update propagation (BAP), which propagates the mask from the key frames to the forward and backward time sequences in a bi-directional manner and updates it adaptively to reduce the tracking accumulation error. 
Its core contributions include 1. Proposing MomentSeg unified framework, which realizes native keyframe selection by matching the similarity between [FIND] token and frame token during training, which eliminates the need of relying on external keyframe models and the need of explicit timestamp encoding, and simplifies the system design. 2. Design Moment Centered Sampling (MCS): Based on the similarity distribution of [FIND]tokn output, dense sampling at text-related key moments and sparse sampling in non-critical regions, taking into account the preservation of key motion details and global timing context maintenance. 3. Bidirectional Anchor Update Propagation (BAP) is proposed: taking the key frames recognized by MCS as the starting point, the mask is propagated forward and backward chronologically in a bidirectional manner, and the mask and memory are updated adaptively at the sampling points, which can effectively alleviate the cumulative tracking error and improve the robustness of long video segmentation.

### Strengths
1. The moment-centric modeling formulation is conceptually sound and backed by a clear motivation:local semantic coherence is often more stable than frame-wise predictions.
2. The appendix adds significant methodological rigor,defining how temporal tokens and attention masks are constructed,and validating their influence.
3. The ablation studies are systematic and reproducible-exploring token number,temporal frame count,threshold sensitivity,and alternative output paradigms ([FIND]vs.text generation).

### Weaknesses
1. The moment definition remains implicit:moments are derived implicitly from attention maps,not through an explicit segmentation constraint This makes reproducibility somewhat architecture-dependent.
2. The causal or theoretical justification for why "moment aggregation"improves generalization is intuitive but lacks formal backing (e.g.,no information-theoretic analysis or invariance claim).
3. Temporal boundary precision is still limited;CRF helps smooth predictions but does not guarantee sharp semantic transition detection. 
4. Efficiency considerations are largely omitted (e.g.,FLOPs,latency),though essential for TAS scalability.

### Questions
1. Efficiency Analysis Needed: No report of inference time or computational overhead-important for real-world deployment.
2. Comparative Positioning vs.Prior Work: The relation to window-based methods (MS-TCN++,ASFormer)should be analytically clarified-e.g.,how MomentSeg differs from dynamic temporal pooling or multi-scale convolution.
3. Generalization Study Beyond Benchmarks: ncluding transfer experiments (e.g.,from GTEA>Breakfast)could demonstrate robustness beyond in-distribution evaluations.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
## Summary
This paper proposes MomentSeg for joint TSG and RefVOS training with [FIND] token-based keyframe selection, moment-centric sampling (MCS), and bidirectional anchor-updated propagation (BAP). While empirical results show improvements, the work suffers from limited novelty, questionable experimental design, and theoretical gaps.

### Strengths
## Strengths

1. **Comprehensive empirical evaluation**: The paper provides extensive experiments across multiple benchmarks including TSG (Charades-STA, ActivityNet), RefVOS (MeVIS, ReVOS, Ref-Youtube-VOS, Ref-DAVIS17), image segmentation (RefCOCO/+/g, ReasonSeg, GCG), and QA tasks. This breadth demonstrates the method's versatility.

2. **Consistent improvements across benchmarks**: The method shows meaningful gains over strong baselines, particularly +5% on MeVIS valu (62.0 vs 57.5) and +6% on ReVOS overall (62.6 vs 58.7) compared to Sa2VA baseline. The 3B model outperforms several 7B models on certain tasks.

3. **Thorough ablation studies**: Tables 6-10 provide detailed ablations of key components (TTI, MCS, BAP), different sampling strategies (Table 7), and joint training effects (Table 10). The analysis of sampling strategies in Fig. 2 effectively motivates the problem.

4. **Eliminates external keyframe selection models**: Unlike VISA, ViLLA, GLUS, and VRS-HQ which require separate models (LLaMA-VID, Grounded-VideoLLM, CLIP) for keyframe identification, this method learns keyframe selection end-to-end within the main model, reducing pipeline complexity.

5. **Well-motivated design**: The analysis showing that random sampling has high variance (Fig. 2) and that uniform sampling outperforms firstK on motion-driven datasets provides good empirical motivation for learned keyframe selection.

6. **Strong TSG results**: Table 2 shows impressive performance on temporal grounding (76.1 R@0.3, 58.2 R@0.5 on Charades-STA) despite using a smaller 3B model, outperforming methods like VTimeLLM-7B and HawkEye-7B.

7. **Detailed supplementary material**: The 16-page appendix includes additional experiments, visualizations, implementation details, and failure case analysis, demonstrating thoroughness.

8. **Clear presentation**: The paper is generally well-written with informative figures (especially Fig. 1, 3, 4) that effectively illustrate the approach. The distinction between training and inference pipelines is clearly explained.

### Weaknesses
### Concerns Regarding the Novelty of Proposed Components

While the paper achieves impressive empirical results, I have some concerns regarding the claimed novelty of the core technical components ([FIND] token, MCS, and BAP), which appear to be effective applications of well-established principles.

**1. On the [FIND] Token Paradigm**

The paper introduces the [FIND] token as part of a "novel TSG paradigm". However, the described mechanism—a learnable token that is matched against frame features using cosine similarity and trained with a contrastive loss—bears a strong resemblance to standard learnable queries used in attention-based models (e.g., DETR) and has been widely explored in video grounding literature.

* **Request for Clarification:** Could the authors elaborate on the specific technical distinctions that qualify this as a "novel paradigm" rather than a successful and well-executed application of existing query-key matching mechanisms for the task of temporal grounding?

**2. On Moment-Centric Sampling (MCS)**

The proposed MCS strategy samples frames based on the similarity distribution generated by the [FIND] token, effectively sampling more densely from regions of high relevance.

* **Observation:** This method appears to be a direct application of importance sampling. The use of Inverse CDF sampling, as detailed in Algorithm 1, is a textbook statistical method for drawing samples from an arbitrary probability distribution. While this is a sensible and effective choice, it may be an overstatement to present the sampling algorithm itself as a novel contribution.

**3. On Bidirectional Anchor-updated Propagation (BAP)**

The BAP mechanism is presented as a method to enhance tracking robustness.

* **Observation:** The key ideas of (1) initializing a tracker from a high-confidence "anchor" frame and (2) propagating both forward and backward in time are standard and widely-used techniques in the video object segmentation (VOS) field to prevent error accumulation.
* **Request for Clarification:** The ablation in Table 8 also shows that the full BAP (with Bidirectional Propagation) results in a slight performance drop on Ref-DAVIS17 (76.8 vs. 77.0) compared to just using "Moment-anchored Updating." This seems to slightly undermine the claim of its universal benefit. Could the authors clarify the specific novelty of BAP beyond combining these existing VOS techniques, and perhaps comment on this performance discrepancy?

**Summary of Concern:**

The primary contribution of this work appears to be the skillful combination and joint optimization of these components into a high-performing, unified framework for RefVOS and TSG. While the engineering is strong and the results are state-of-the-art, the paper would be more impactful if it more clearly distinguished its fundamental research contributions from the novel application of existing, well-established methods.

### Questions
1. Inference Cost
    - Could you provide a detailed breakdown of the inference latency and/or FLOPs for the full MomentSeg pipeline? How does it compare to other LMM-based methods that might use a simpler, single-pass sampling strategy?

### Soundness
3

### Presentation
4

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
This paper proposes MomentSeg, a novel framework for referring video object segmentation (RVOS) that addresses the limitations of sparse and fixed sampling in existing methods. The core contribution is the Moment-Centric Sampling (MCS) mechanism, which intelligently selects frames most relevant to the referring expression to improve temporal grounding and visual context. Furthermore, the model employs a Bidirectional Anchor-updated Propagation (BAP) strategy to ensure robust and temporally consistent mask tracking across the selected video segment. The experimental results, validated across multiple challenging datasets, demonstrate that MomentSeg achieves state-of-the-art or highly competitive performance, confirming the effectiveness of the proposed sampling and propagation strategies.

### Strengths
1. Clear Motivation: The work provides a clear and compelling motivation, effectively highlighting the inherent limitations of conventional fixed or simple keyframe sampling approaches when dealing with moment-centric video-language tasks.
2. Comprehensive Experiments and Strong Results: The experimental evaluation is notably comprehensive, providing ample technical detail and achieving excellent, state-of-the-art results across a diverse set of benchmark datasets.
3. Well-Structured and Accessible Methodology: The overall methodology and presentation of the paper are clear, logical, and easy to follow, making the proposed MomentSeg architecture and techniques accessible to the broader reader community.

### Weaknesses
1. Incremental Gain of MCS: The core idea of "keyframe mining" appears somewhat incremental, and the reported empirical gains are small; Figure 2 suggests the impact of the sampling strategy is marginal (around 2 points), and Table 7 confirms that the gain provided by MCS over the base keyframe methods is minimal.
2. Similarity to Prior Grounding Work: The approach of using the [FIND] token to localize keyframes is highly similar to the established "streaming EOS prediction" mechanism found in related work, such as VideoLLM-Online, which diminishes the technical novelty of this specific component.
3. Limited Applicability to Online Scenarios: Key components of the proposed method, including Moment-Centric Sampling (MCS) and Bidirectional Anchor-updated Propagation (BAP), fundamentally require access to the entire video's context, severely limiting their applicability to real-world online or streaming environments.
4. Novelty of Bidirectional Propagation: While effective within this framework, the concept of bidirectional propagation for ensuring temporal consistency is not a novel technique in the broader Video Object Segmentation (VOS) literature and represents mainly an adaptation to the SAM model rather than a core innovation.
5. Missing Robustness Analysis: The paper lacks a deep-dive analysis on the method's robustness, particularly how BAP performs when anchor frames contain significant noise or the model encounters high degrees of occlusion, which are common challenges in propagation-based models.

[*] VideoLLM-online: Online Video Large Language Model for Streaming Video, CVPR2024.

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a unified framework that jointly optimizes temporal sentence grounding and segmentation. A novel [FIND] token enables key moment identification without external timestamps, while a Moment-Centric Sampling (MCS) strategy balances dense and sparse frame sampling for efficiency and context preservation. Additionally, a Bidirectional Anchor-updated Propagation (BAP) mechanism enhances tracking stability by adaptively updating masks during inference.

### Strengths
1.	The paper addresses a valuable research question: Is keyframe selection necessary?
2.	The experiments cover a wide range of tasks and datasets, with comprehensive baseline comparisons.
3.	The ablation studies are well-designed and clearly demonstrate the contribution of each component.
4.	The proposed method achieves competitive performance.

### Weaknesses
1.	The FIND token plays a critical role in the proposed approach. However, the paper does not clearly explain how this token is supervised or optimized. Moreover, the semantic role of the FIND token within the model (e.g., whether it acts as a query, indicator, or moment aggregator) remains unclear.
2.	The Moment-Centric Sampling design appears rather complex, but the authors lacks justification for its necessity and detailed analysis of why this design is better.
3.	The overall framework is complicated with many hyperparameters, which increases the tuning cost.
4.	The inference efficiency is not adequately discussed. Considering the method includes multiple tokens and multiple steps, the inference cost may be relatively high.
5.	Figures and writing quality could be further polished for clarity and readability.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
