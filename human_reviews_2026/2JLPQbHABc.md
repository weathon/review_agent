# VideoAnchor: Reinforcing Subspace-Structured Visual Cues for Coherent Visual-Spatial Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision–language alignment, yet they remain limited in visual–spatial reasoning. We first identify that this limitation arises from the attention mechanism: visual tokens are overshadowed by language tokens, preventing the model from consistently recognizing the same visual cues across frames. To address this challenge, we draw a novel connection between the self-expressiveness property in sparse subspace clustering and the attention mechanism in Transformers. Building on this insight, we propose VideoAnchor, a plug-and-play module that leverages subspace affinities to reinforce visual cues across frames without retraining, effectively anchoring attention to shared visual structures. Extensive experiments across benchmarks and backbone models show consistent performance gains — e.g., 3.2% and 4.6% improvements on VSI-Bench and Video-MME (spatial-related tasks) with InternVL2-8B and Qwen2.5VL-72B—while qualitative analyses demonstrate more coherent subspace partitions and stronger visual grounding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes VideoAnchor, a training-free module that improves visual–spatial reasoning in MLLMs by anchoring attention on shared visual cues across frames. It leverages sparse subspace clustering to identify consistent visual structures and regularizes attention at inference time. Evaluated on VSI-Bench, All-Angles-Bench, and Video-MME, it achieves limited performance gains without retraining.

### Strengths
1.VideoAnchor operates purely at inference time, making it easily applicable to existing MLLMs without fine-tuning.

2.Comprehensive analysis: Includes ablation studies and visualization to explain the mechanism’s effect on attention coherence.

3.This work establishes a connection between sparse subspace clustering and attention mechanisms, offering a principled way to preserve visual consistency across frames.

### Weaknesses
1. The method primarily targets pairwise spatial relations by anchoring one shared object. It remains unclear how this mechanism would generalize to broader scenarios such as navigation, counting, or reasoning over complex multi-object relations involving three or more entities.

2. The paper evaluates InternVL with 8-frame input and Qwen2.5-VL/LLaVA-Video with 16 frames on VSI-Bench, but does not justify this inconsistency. Moreover, since most videos in VSI-Bench and Video-MME last longer than 60s, sparse frame sampling may miss visual continuity, raising questions about how anchoring remains effective when frame overlap is minimal.

3. While VideoAnchor fits multi-view scenarios, evaluation across more multi-view benchmarks such as SPAR-Bench [1] would strengthen claims of generalizability. Also, models vary across benchmarks; evaluating the same model series would provide a clearer understanding of cross-task generalization.

4. Important inference details (e.g., temperature, rounds) are missing. It is unclear whether the observed gains are statistically significant or could be influenced by inference variance rather than the VideoAnchor mechanism itself.

5. Since VideoAnchor modifies attention internally, I am curious how explicit prompt-based anchoring performs—for instance, instructing the model to first identify an anchor object and then describe spatial relations. 

[1] From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D, 2025.

### Questions
Please refer to the **Weaknesses**

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
This paper introduces VideoAnchor, a plug-and-play module designed to enhance visual–spatial reasoning in multimodal large language models (MLLMs) during inference. The key idea is to exploit the self-expressiveness property of Sparse Subspace Clustering (SSC) to uncover consistent visual structures across frames, and then translate these structural relations into token-wise scaling factors that modulate attention.

VideoAnchor consists of two components: 1. Subspace-to-Scaler Unit – Applies SSC to visual tokens extracted from multiple frames, computes each token’s “anchor strength” based on its intra-subspace connectivity, and generates scaling coefficients for the Q/K/V projections. 2. Attention Regularization Unit – Injects these scalers into the attention computation, amplifying stable visual regions and mitigating text-dominated biases without retraining. The method operates entirely at inference time, is model-agnostic, and can be seamlessly integrated into all self-attention layers of various MLLMs. Experiments across multi-frame and multi-view benchmarks demonstrate consistent accuracy gains (typically +1–3%).

### Strengths
1）Clear Conceptual Novelty. The paper introduces a distinctive idea by incorporating the self-expressiveness property of Sparse Subspace Clustering (SSC) into Transformer attention modulation. This provides a theoretically grounded and interpretable way to reinforce visual anchoring during inference.

2) Training-Free and Plug-and-Play Design. VideoAnchor operates entirely at inference time without parameter updates or retraining. Its plug-and-play nature makes it broadly compatible with various multimodal large language models (e.g., InternVL, LLaVA, Qwen-VL).

3) Strong Interpretability. The approach offers explicit geometric and semantic interpretability: subspace consistency identifies stable visual anchors, and attention amplification directly follows from these structural relations, unlike opaque learned gates.

4) Comprehensive Ablation Analysis. The paper includes systematic ablations on Q/K/V scaling, gating placement, and subspace cardinality, providing convincing evidence for the soundness of each design choice.

### Weaknesses
1）Lack of Quantitative Efficiency and Complexity Analysis. Although the paper emphasizes that VideoAnchor is lightweight, it provides no explicit measurements of runtime, FLOPs, or latency, leaving the actual computational efficiency unverified. While the ADMM optimization process is described, there is no formal analysis of its convergence behavior or computational scaling under realistic token counts.


2) No Discussion of Scalability to Long Videos or High-Resolution Inputs. Since the SSC step scales quadratically with the number of tokens, potential issues in long-sequence or high-resolution scenarios remain untested and unexplored.


3) Empirical Scope Limited to Spatial Reasoning Tasks. VideoAnchor appears to be a more general approach, but the evaluation focuses on visual–spatial reasoning; generalization to broader multimodal reasoning tasks (e.g., temporal dynamics, abstract understanding) remains unclear.

### Questions
1) Have the authors verified compatibility with high-performance inference frameworks (e.g., vLLM, FlashAttention, xFormers)? Does the multiplicative scaling require any modification to fused attention kernels or memory layouts?

2) The choice of subspace number and the scaling coefficients (α_Q, α_K, α_V) lacks principled derivation or sensitivity analysis beyond empirical tuning.

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
4

### Summary
This paper introduces *VideoAnchor*, a plug-and-play attention modulation module designed for test-time enhancement. The core idea is to cluster multi-frame or multi-view visual tokens into semantic subspaces using sparse subspace clustering, compute a *shared expression score* for each token, and convert it into scaling factors applied to QKV in the attention mechanism.  Extensive experiments on VSI-Bench, All-Angles-Bench, and Video-MME demonstrate consistent improvements across multiple multimodal LLM backbones, supported by ablation studies and visual analyses.

### Strengths
1. A training-free and model-agnostic *subspace-to-scaler* mechanism that can be universally applied across architectures, linking self-expressive subspace modeling with transformer attention in a novel way.
2. Comprehensive experimental validation on diverse benchmarks with consistent gains and detailed ablations covering QKV scaling, clustering methods, and subspace counts.

### Weaknesses
1. The comparison with other training-free enhancement approaches (e.g., DC2’s high-resolution perception [1], ControlMLLM’s visual prompting [2], VisionFuse’s multi-encoder fusion [3]) lacks equal experimental settings and quantitative cost-performance analysis.
2. Missing a deeper comparison or hybridization with learning-based reinforcement approaches such as Video-R1 or T-GRPO [4], the paper only briefly mentions them in the appendix without unified benchmarks.
3. The failure cases and generalization limits (e.g., occlusions, fast viewpoint changes, low-texture regions) are underexplored, with only anecdotal visual evidence.
4. The SSC plus spectral clustering steps could introduce high latency when processing long videos or thousands of tokens, yet no systematic runtime or throughput evaluation is provided.
5. Hyperparameter sensitivity is superficially treated. Although claimed “robust,” only fixed αQ/αK/αV and subspace counts are reported, without cross-dataset robustness or automatic selection strategies.

[1] Li et al. *DC2: Decoupled Cross-resolution Conditioning for High-Resolution Multimodal Models*.

[2] Hu et al. *VisionFuse: Multi-Encoder Fusion for Multimodal Reasoning*.

[3] Xu et al. *ControlMLLM: Test-time Control for Multimodal Large Language Models*.

[4] Wang et al. *Video-R1: Reinforcing Video Reasoning in Multimodal Large Language Models*.

### Questions
1. What is the computational complexity of VideoAnchor in terms of frames, tokens, and subspace counts? A runtime comparison with DC2 and VisionFuse would clarify its practicality.
2. Does the sharing expression score bias attention toward large clusters, potentially overlooking small but semantically critical objects? Could weighting or normalization mitigate this issue?
3. When strong appearance changes or motion blur occur, how does the linear self-expression assumption of SSC hold? Have the authors considered nonlinear or graph-based subspace alternatives?
4. Could VideoAnchor be combined with learning-based methods such as Video-R1 or GRPO fine-tuning to further improve robustness and visual grounding?
5. Across benchmarks,VSI-Bench, All-Angles-Bench, and Video-MME,which specific sub-tasks (e.g., spatial relation, counting, manipulation) benefit most, and does the improvement mainly come from enhanced perception or reasoning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the visual-spatial reasoning in the context of MLLMs, and proposes VideoAnchor, a plug-and-play module, to address the limitation of current MLLMs that they lack a mechanism to consistently preserve visual cues cross frames. Specifically, the paper aims to draw connection between the self-expressive property in sparse subspace clustering (SSC) and attention mechanisms in Transformers. To implement this, the proposed VideoAnchor consists of two units: (1) subspace-to-scaler unit, and (2) attention regularization unit. Methods and empirical evaluations are presented.

### Strengths
Overall, the paper is easy to follow. The problem setting, the intuition/motivation (that token organization in the semantic subspace helps), the technical approach, and the empirical evaluations are presented in a relatively clear and organized manner. The utilization and illustration of SSC (e.g., the discussions in Section 3.4, illustrations in Section 4.2) provide different perspectives and justify the incorporation of two units in VideoAnchor. Experimental results also show improvements.

### Weaknesses
The improvement over baseline is relatively modest. This is not entirely unexpected now that the module does not require re-training. That being said, I am wondering if author(s) can share thoughts on whether the plug-and-play module can have a more "integrated" version, in the sense that how would VideoAnchor operate if (lightweight) retraining is feasible. Additional discussion on the tradeoff between expected performance gain and the retraining effort may be preferable (e.g., to better justify the contribution), in addition to the claim that VideoAnchor is plug-and-play.

### Questions
(As mentioned in "Weakness" Section) Can author(s) provide further discussion/justification on the tradeoff between expected performance gain and the retraining effort?

### Soundness
3

### Presentation
3

### Contribution
3
