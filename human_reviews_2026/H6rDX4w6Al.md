# FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging

- Decision: Accept (Oral)
- Scores: 4, 4, 8, 6

## Abstract
Although Video Large Language Models (VLLMs) have shown remarkable capabilities in video understanding, they are required to process high volumes of visual tokens, causing significant computational inefficiency. Existing VLLMs acceleration frameworks usually compress spatial and temporal redundancy independently, which overlooks the spatiotemporal relationships, thereby leading to suboptimal spatiotemporal compression. The highly correlated visual features are likely to change in spatial position, scale, orientation, and other attributes over time due to the dynamic nature of video. Building on this insight, we introduce FlashVID, a training-free inference acceleration framework for VLLMs. Specifically, FlashVID utilizes Attention and Diversity-based Token Selection (ADTS) to select the most representative tokens for basic video representation, then applies Tree-based Spatiotemporal Token Merging (TSTM) for fine-grained spatiotemporal redundancy elimination. Extensive experiments conducted on three representative VLLMs across five video understanding benchmarks demonstrate the effectiveness and generalization of our method. Notably, by retaining only $\textbf{10}$% of visual tokens, FlashVID preserves $\textbf{99.1}$% of the performance of LLaVA-OneVision. Consequently, FlashVID can serve as a training-free and plug-and-play module for extending long video frames, which enables a $\textbf{10$\times$}$ increase in video frame input to Qwen2.5-VL, resulting in a relative improvement of $\textbf{8.6}$% within the same computational budget. Code is available at https://github.com/Fanziyang-v/FlashVID.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FlashVID, a unified and efficient framework for long-context video understanding. The core idea is to enable scalable temporal reasoning on long videos without incurring quadratic attention costs, while maintaining accuracy comparable to state-of-the-art multimodal large language models (MLLMs).  The authors propose three key innovations: Flash Temporal Attention (FTA), Segment-Level Progressive Compression (SPC), and Streaming Memory Replay (SMR).

### Strengths
1. The paper targets efficiency–accuracy tradeoffs in long-video understanding, a practical and timely problem for both academic and industry-scale multimodal systems.The proposed Flash Temporal Attention and Segment-Level Progressive Compression present new design patterns for balancing local precision with global awareness.
2. The methods are clearly described and experimentally validated across multiple datasets and evaluation settings. Ablation studies systematically analyze the contributions of FTA, SPC, and SMR, showing each component’s quantitative effect. The authors also provide a theoretical complexity analysis, demonstrating that temporal attention cost scales linearly rather than quadratically.
3. Figures (e.g., Fig. 2–4) effectively visualize the compression hierarchy and temporal flow.

### Weaknesses
1. While SPC achieves impressive speedup, the paper lacks qualitative or interpretive analysis of how compression affects temporal coherence or feature richness. Visualization (e.g., attention maps pre- and post-compression) would strengthen understanding.
2. Experiments primarily cover videos up to 2 hours. It would be interesting to see whether FlashVID maintains stability and context retention beyond this range (e.g., >5 hours or continuous streaming).
3. How does SMR handle forgetting or outdated context in long videos? Is there a mechanism for refreshing or discarding stale memory tokens?
4. Since SPC depends on segment-level aggregation, how sensitive is the model to poor segmentation? It would be valuable to test robustness on unstructured or synthetic long videos.

### Questions
1. Could you report end-to-end latency per video and memory usage per frame, especially during streaming inference? These practical details would help others reproduce the claimed efficiency.
2. Could you provide qualitative examples showing how SPC affects video understanding? For instance, do compressed representations preserve fine-grained temporal cues such as object motion or scene transitions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a training-free inference acceleration framework called FlashVID, aimed at addressing the high computational cost issues caused by processing massive visual tokens in video large language models (VLLMs). FlashVID employs a two-stage compression strategy: first, it selects representative tokens within each frame using the "Attention and Diversity-based Token Selection (ADTS)" module; second, it constructs a "spatio-temporal redundancy tree" across frames through the "Tree-based Spatio-Temporal Token Merging (TSTM)" mechanism to jointly model and merge similar tokens that change dynamically over time. Experiments were validated on three VLLMs and five video benchmarks, maintaining 99.1% of LLaVA-OneVision's performance while retaining only 10% of the tokens, and achieving a 10x increase in the number of frames processed by Qwen2.5-VL under the same computational budget, resulting in an 8.6% improvement.

### Strengths
1. FlashVID is a training-free framework, can be used as a plug-and-play module applied to existing trained VLLMs without expensive training costs.
2. Addressed a key pain point of previous VLLM acceleration methods: they typically compress spatial and temporal redundancies independently, or rely on a single spatial correspondence for temporal merging. TSTM can flexibly track and merge similar tokens that change dynamically over time in terms of spatial location, scale, or direction by building a "spatio-temporal redundancy tree," which better aligns with the dynamic characteristics of videos.

### Weaknesses
1. This method requires multiple hyperparameters that need empirical tuning, which may affect its plug-and-play performance across different models or datasets. For example $T_{\tau}$, $f_{e}$, $\alpha$。
2. The paper mentions the interesting phenomenon of "less is more" and attributes it to "excessive visual token input may introduce noise," which is a reasonable assumption but lacks more in-depth quantitative or qualitative analysis to clarify how these "noise" specifically affect VLLM's attention or internal representations. Moreover, this phenomenon was only tested on the llava-onevision model, making the argument less persuasive. If this phenomenon can also be observed on models like qwen-vl and internvl, it would further strengthen the argument.
3. ADTS introduced an "event relevance" to calibrate "importance", but it uses global average pooling (GAP) to average the entire frame information into a single vector. If the key event in a video is a local small action and most of the area is static background, the GAP vector will mainly represent the background, which may lead ADTS to mistakenly consider this key local event as "irrelevant" and possibly discard it. Is this global heuristic method a significant weakness for tasks requiring fine-grained understanding?

### Questions
1. "We set K = 20 for LLaVA-OneVision, LLaVA-Video, and Qwen2.5-VL." Has K=20 been analyzed? Or is it set based on experience, and if it's a new model, would this configuration also apply?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes FlashVID, a training-free inference acceleration framework for Video Large Language Models (VLLMs) that addresses the computational bottleneck of processing high volumes of visual tokens. FlashVID is a training-free acceleration framework for Video Large Language Models (VLLMs) that addresses computational bottlenecks from processing many visual tokens. Unlike existing methods that compress spatial and temporal redundancy separately, FlashVID jointly models spatiotemporal relationships through: (1) TSTM - tree-based merging that tracks tokens across frames despite spatial movement, and (2) ADTS - diversity-based selection of informative tokens per frame. Experiments on three VLLMs across five benchmarks show FlashVID retains 99.1% performance with only 10% of tokens and enables processing 10× more frames under fixed compute budgets.

### Strengths
- The paper identifies a clear limitation in existing methods that temporal redundancy is typically defined by fixed spatial locations, which fails to capture video dynamics where objects move, scale, and rotate. The tree-based spatiotemporal merging is an elegant solution to this problem.

- The experiments are comprehensive, covering three diverse VLLMs and five benchmarks. The results consistently show FlashVID outperforming baselines. 

- The method requires no additional training, making it immediately applicable to existing VLLMs and reducing deployment barriers.

- The paper includes proper ablations (Table 4, 5, 9-12), efficiency analysis (Table 13), and extensive visualizations (Figures 5-8) that support the main claims.

### Weaknesses
- Limited novelty in individual components: ADTS essentially combines existing techniques ([CLS] attention + diversity-based selection via MMDP). The calibrated MMDP formulation (Algorithm 4) is relatively straightforward. The tree construction in TSTM (Algorithm 1, lines 9-16) is a simple greedy nearest-neighbor matching with thresholding. The "tree" structure emerges naturally but isn't explicitly optimized.

- The paper states that depth and breadth constraints "yielded negligible gains" (page 6, line 297-299; Table 11-12), which raises questions about whether the tree structure itself is crucial or if simple nearest-neighbor merging suffices.

- The paper combines multiple techniques: video partitioning, ADTS, TSTM, and inner-LLM pruning. Table 4-5 only ablate ADTS variants (ATS, DTS, ADTS) and \alpha, not the full pipeline. More ablation to show each contribution would make the paper more stronger.

### Questions
- What is the individual contribution of TSTM vs. ADTS? How much does video partitioning contribute?

- TSTM vs. simpler alternatives: Have you compared TSTM against simpler temporal merging strategies like:
  - Merging based on global feature similarity (all-pairs matching) without spatial constraints?
  - k-means clustering across frames?
  - Optical flow-guided merging?

- Given that depth/breadth constraints don't help (Table 11-12), is the tree structure essential, or is TSTM effectively performing similarity-thresholded pairwise merging?

- How does FlashVID perform on video generation or video editing tasks where spatial precision is critical? Can it handle multi-view videos or videos with significant camera motion?

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
The paper proposes FlashVID, a training-free and model-agnostic token merging framework for video LLMs that jointly compresses both spatial and temporal redundancy. FlashVID introduces a two-stage pipeline: an Attention and Diversity-based Token Selection (ADTS) module that selects intra-frame tokens, and a Tree-based Spatiotemporal Token Merging (TSTM) module that merges redundant tokens across frames. Experimental results show that the proposed method can beat previous works using different backbones, and the ablation experiments are comprehensive.

### Strengths
1. The idea of jointly modeling spatial and temporal redundancy via a tree-based merging method is well-motivated and insightful.
2. Another component, ADTS, focusing on intra-frame redundancy, is organically combined with TSTM in the pipeline. And they are complementary.
3. Experiments are done in multiple benchmarks, with multiple backbones, compared with several previous SOTAs. The proposed model can consistently outperform them.

### Weaknesses
1. When comparing with other previous works, only the retention ratio and TFLOPs are used in the paper to measure the efficiency. However, token pruning is also time-consuming. Given the proposed method has many steps (eg, tree construction, similarity calculation, attention mask computation, etc), the pruning inference time complexity, and potential memory cost should also be comprehensively analyzed and compared with previous works.

2. In the paper, the author(s) claim to use a hybrid compression strategy (before-LLM + inner-LLM). However, it's unclear that given a retention ratio, how many tokens are dropped according to the proposed method (before-LLM compression), and how many tokens are dropped in inner-LLM compression. And also, how is the before-inner ratio compared to previous works? Without a detailed analysis, it is difficult to tell whether the benefit mainly comes from the proposed method or from inner-LLM compression.

### Questions
See weaknesses. 

Besides, one minor question is: why do you introduce TSTM in sec 3.2 before ADTS in sec 3.3, tho their orders in your pipeline are opposite?

### Soundness
3

### Presentation
4

### Contribution
3
