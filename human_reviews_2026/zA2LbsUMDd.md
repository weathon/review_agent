# VideoNSA: Native Sparse Attention Scales Video Understanding

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Video understanding in multimodal language models remains limited by context length: models often miss key transition frames and struggle to maintain coherence across long time scales. To address this, we adapt Native Sparse Attention (NSA) to video-language models.  **Our method, VideoNSA, adapts Qwen2.5-VL through end-to-end training on a 216K video instruction dataset. We employ a hardware-aware hybrid approach to attention, preserving dense attention for text, while employing NSA for video.** Compared to token-compression and training-free sparse baselines, VideoNSA achieves improved performance on long-video understanding, temporal reasoning, and spatial benchmarks. Further ablation analysis reveals four key findings: (1) reliable scaling to 128K tokens; (2) an optimal global–local attention allocation at a fixed budget; (3) task-dependent branch usage patterns; and (4) the learnable combined sparse attention help induce dynamic attention sinks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper adapts Native Sparse Attention (NSA) to video-language models, demonstrates its potential in tasks including long video understanding, temporal reasoning, and spatial intelligence, and conducts extensive and thorough ablation and analytical experiments. Additionally, to mitigate attention sinks in long-term visual contexts, this paper further proposes to dynamically integrate global and local attention via three complementary branches, effectively addressing this issue.

### Strengths
1. This paper attempts to apply Native Sparse Attention (NSA) to VideoMLLMs for the first time, and effectively demonstrates its potential in tasks such as long video understanding.

2. The ablation experiments in this paper are quite comprehensive and rigorous, which analyze the performance of NSA in video understanding tasks from multiple perspectives.

### Weaknesses
1. This paper appears to lack an analysis of the training and inference efficiency of VideoNSA. For instance, regarding Table 1, it would be desirable to know the comparisons between various methods and VideoNSA in terms of inference efficiency, latency, and FLOPs. For training efficiency, a comparison between VideoNSA and full attention under the same context length is also expected.

2. Regarding the application of NSA in video tasks, the primary video-related modification in this paper seems to be employing standard GQA for text while adopting NSA for video attention. It is of interest to understand the impact of this operation on the final performance. Does "Dense-NSA" in Table 3 refer to the use of NSA for all modalities? It is suggested that the authors further elaborate on the performance differences of NSA across different modalities (i.e., video and text).

### Questions
1. Does "Dense-NSA" in Table 3 refer to the use of NSA for all modalities?
2. Regarding Figure 2, does the performance under the 64k context length appear to be better than that under 128k? Does this indicate that training under a short context length (i.e., 36k) is insufficient to unlock the full 128k performance of VideoNSA?

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
This paper targets the challenge of long-video understanding in MLLMs, which is constrained by the quadratic complexity of standard attention. The authors propose VideoNSA, a hybrid attention mechanism that adapts Native Sparse Attention for video-language models. The core of the method is to apply standard Grouped-Query Attention to text tokens while using a learnable, three-branch sparse attention mechanism for the video tokens. This video-specific NSA dynamically combines a Token Compression branch, a Token Selection branch, and a Sliding Window branch using learnable gates. The model, which is an adaptation of Qwen2.5-VL , is trained end-to-end on a 216K video instruction dataset. The authors present experiments showing that VideoNSA achieves competitive performance on long-video understanding, temporal reasoning, and spatial benchmarks. The paper also provides a very detailed analysis of the model's scaling properties, attention budget allocation, internal branch usage, and its effect on mitigating attention sinks.

### Strengths
1. The paper addresses a highly significant and timely problem: scaling video-language models to handle long contexts (e.g., thousands of frames or 128K tokens) efficiently.

2. The proposed hybrid attention mechanism is well-motivated. Preserving dense attention for text tokens while applying aggressive, learnable sparsity to the highly redundant video tokens  is a sensible architectural choice.

3. The paper's main strength is its extensive analysis section. The authors provide commendable, deep insights into the model's behavior, including:
- A study of information scaling (spatial vs. temporal trade-offs) .
- A detailed breakdown of attention budget allocation (global vs. local).
- An analysis of the dynamic gate usage across layers.
- A novel and valuable investigation into how different sparse branches uniquely contribute to or mitigate attention sinks.

### Weaknesses
1. My most significant concern is the flawed "Dense-SFT" baseline. This baseline, which should serve as the primary control, was fine-tuned on the same 216K dataset as VideoNSA. However, this Dense-SFT model performed worse than the original, pre-trained Qwen2.5-VL on most benchmarks (e.g., LVB, TimeScope, Tomato). The authors attribute this to the "limited quality of the training data". This admission severely confounds the paper's central claim. We cannot know if VideoNSA's architectural improvements are genuine or if the VideoNSA architecture is simply more robust to this specific, low-quality training data than a dense model. The experiment fails to demonstrate that VideoNSA is better than a properly trained dense model.

2. While the paper claims "improved performance," the results in Table 1 are more accurately described as "competitive" or "on par" rather than a significant step forward.
- On Long VideoBench (LVB), VideoNSA (60.0) is outperformed by Video-XL-2 (61.0).
- On $MLVU_{test}$, VideoNSA (51.8) is outperformed by Video-XL-2 (52.2) and InternVL2.5-8B (55.8).
- On Long TimeScope (LTS) and TimeScope, its scores (44.4 and 83.7) are effectively tied with other sparse attention methods like MInference and XAttention.
- Given the confounding baseline (Weakness 1), these marginal gains are not sufficient to robustly claim superiority.

3. The paper is motivated by efficiency, but its own analysis (Finding 5) identifies the compression (CMP) branch as the dominant latency bottleneck as context length grows. The paper concludes that "the prefill stage remains the primary bottleneck". While the analysis is transparent and appreciated, the paper identifies a critical practical limitation of its own method without offering a solution. This undermines the practical efficiency claims of the work.

### Questions
1. Could you elaborate on the "Dense-SFT" baseline's performance drop? If the training data is of limited quality, how can you be sure it isn't also limiting VideoNSA's potential? Conversely, do you hypothesize that VideoNSA's gains over the dense baseline would be larger or smaller if trained on a much larger, higher-quality video instruction dataset?

2. Given the CMP branch is the bottleneck, do you have concrete suggestions for optimizing it? The paper states the block-level representation is obtained by "averaging all tokens" 29, but the preliminary definition (Eq. 2) mentions a "learnable MLP" ($\varphi$)30.
- First, please clarify this: is the learnable MLP simply performing a weighted average, or is it a more complex, non-linear projection?
- Second, if it is a simple average, the latency should be low. If it's a learnable MLP, did you experiment with replacing it with a fixed (non-learnable) pooling operation to see if the performance/latency trade-off improves?

3. The authors note the "strange behavior" in the last layer (L27) where all three branch gates become fully active. Do you have any hypothesis for this? Is it a learned behavior for final-layer aggregation, or could it be an artifact of training (e.g., the gates for that layer not receiving a strong gradient)?

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
This paper introduces VideoNSA, a method for scaling video understanding models to very long contexts by adapting NSA. The core idea is to apply a hybrid attention mechanism to a Qwen2.5-VL-7B. Specifically, text tokens are processed with standard GQA, while video tokens are handled by NSA, which dynamically combines three complementary sparse attention branches: CMP for global aggregation, SLC for salient information, and SWA for local context. The authors fine-tune this model on a 216K video instruction dataset. The resulting model scales effectively to 128K tokens and performs well at a series of challenging long-video benchmarks,

### Strengths
- Extensive Evaluation: The paper evaluates VideoNSA across a diverse set of challenging long-video benchmarks, demonstrating competitive or SOTA performance. The inclusion of strong baselines and thorough ablations validates the design choices.
- In-depth Analysis: The analysis in Section 4 is a standout feature. The structured "Findings" provide clear, actionable insights into how sparse attention behaves when scaled. The study of information scaling, budget allocation, and attention sinks goes far beyond a typical model performance paper and offers significant value to the research community.
- Efficiency and Scalability: The paper shows that VideoNSA scales effectively to 128K context lengths, far beyond its training regime. The finding that it achieves top-tier performance with only 3.6% of the dense attention budget is a powerful demonstration of the method's efficiency.

### Weaknesses
Novelty: The primary weakness is that the core technical component, NSA, is adapted from a previous work (Yuan et al., 2025b). The novelty is in the application, specific architectural choices for video, and the extensive analysis, rather than a new algorithm.

### Questions
In "Finding 5," you identify the token compression (CMP) branch as the main latency bottleneck. Given its importance (as shown in the gate analysis in "Finding 4"), what are your thoughts on potential avenues for optimizing this branch to further improve the model's overall efficiency?

### Soundness
3

### Presentation
3

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
This paper extends Native Sparse Attention (NSA), originally designed for long-context models, to the multimodal domain, proposing VideoNSA. This method applies NSA's three-branch hierarchical sparse structure and gating mechanism to video tokens, while employing Grouped Query Attention for text tokens. Furthermore, the paper provides a scalability analysis, offering new insights into the behavior of sparse attention in multimodal models.

### Strengths
1. Insightful Analysis: The scalability analysis provides a degree of interpretability for sparse attention, while also clarifying the advantages of the sparse mechanism (e.g., extensibility to contexts longer than those seen during training, control over attention sinks).
2. Comprehensive Experiments: The experimental validation is extensive, covering ablation studies for each branch, visualizations of gating distributions, context extension curves, and performance evaluations on multiple benchmarks.
3. Practical Guidance: The empirical findings offer practical guidance for deploying hardware-aligned sparse attention in long-context multimodal systems.

### Weaknesses
1. Limited Algorithmic Novelty: The core framework for the vision component (the three-branch sparse structure + learnable gating) is nearly identical to that of NSA (Yuan et al., 2025), meaning the method lacks fundamental innovation.
2. Lack of Discussion on Modality-Specific Sparsity: The paper does not discuss the differences in sparsity between text and video. In text, sparse attention primarily filters information at the syntactic and semantic levels, where token dependencies are relatively stable and one-dimensional. In video, however, sparsity involves spatiotemporal locality and motion redundancy, implying that the definition of semantic redundancy differs across modalities. This raises questions about whether it is appropriate to directly reuse a text-based sparse attention mechanism for the video modality.
3. Insufficient Theoretical or Analytical Depth: The scalability analysis is predominantly empirical. While it helps users better utilize the model, it lacks theoretical explanations or modeling to elucidate the underlying causes of the observed trends.

### Questions
1. The paper lacks a strong motivation for applying NSA, a text-modality method, to the video modality. Many methods exist for long-context modeling; why is NSA a good choice? In other words, do the characteristics of NSA offer unique advantages in the context of video?
2. The authors state that the model utilizes the selection and sliding-window branches in shallow layers to capture fine-grained details and local information, while relying more on the compression branch in deep layers to integrate and refine high-level global semantics. Do the sparse distributions and gating activations differ across various types of videos (e.g., high-speed motion, shot transitions, static scenes), or is the gating behavior solely dependent on layer depth? If the gating behavior is only correlated with layer depth and is independent of input features, does this imply that the model has limited adaptability to different types of video content?

### Soundness
2

### Presentation
4

### Contribution
3
