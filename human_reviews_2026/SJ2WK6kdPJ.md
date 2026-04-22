# NABLA: Neighborhood-Adaptive Block-Level Attention for Efficient Video Diffusion Transformers

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 2, 8

## Abstract
Full self‑attention in video diffusion transformers scales quadratically with the spatio‑temporal token count, making processing the high‑resolution clips prohibitively slow and memory‑heavy. We introduce NABLA, a Neighborhood‑Adaptive Block‑Level Attention mechanism that builds a per‑head sparse mask in three steps: (i) average‑pool queries and keys into $N\times N$ blocks, (ii) keep the highest‑probability blocks via a cumulative‑density threshold, and (iii) optionally union the result with Sliding‑Tile Attention (STA) to suppress border artefacts. NABLA drops straight into PyTorch's FlexAttention with no custom kernels or extra losses. On the Wan 2.1 14B text‑to‑video model at 720p, NABLA accelerates training and inference by up to $2.7\times$ while matching CLIP ($42.06\rightarrow42.08$), VBench ($83.16\rightarrow 83.17$) and FVD ($68.9\rightarrow 67.5$) scores. During pre‑training of a 2B DiT at $512^2$, iteration time falls from 10.9s to 7.5s ($1.46\times$) with lower validation loss. A link to the code and model weights will be published in the camera-ready version of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a Neighborhood-Adaptive Block-Level Attention mechanism (NABLA) designed to accelerate video diffusion transformers by exploiting sparsity in the attention map. The method operates in three steps: (1) average-pool queries and keys into N×N blocks; (2) apply a cumulative density threshold to select high-importance blocks; (3) optionally combine the result with Sliding Tile Attention (STA) to mitigate boundary artifacts. NABLA can be integrated directly into PyTorch’s FlexAttention without custom CUDA kernels. The authors claim up to 2.7× acceleration on the WAN 2.1 (14B) model, with comparable CLIP/FVD/VBench metrics to full attention.

### Strengths
1. NABLA is designed to fit into PyTorch’s FlexAttention API with minimal engineering cost, which makes it practically useful.

2. Although not deep, the experiments are at least conducted on a realistic backbone, lending some credibility to deployment feasibility.

### Weaknesses
1. The overall presentation of the paper is poor. For instance, Figure 1 is disproportionately large and occupies excessive space, reducing readability. In addition, the Introduction section devotes most of its content to related work, while providing only a brief, single paragraph describing the proposed method, which makes it difficult for readers to grasp the core contribution.

2. The proposed approach (average pooling + CDF thresholding) is a straightforward engineering heuristic, not grounded in theory or optimization principles. Could the author provide some theory guarantee?

### Questions
1. How does NABLA differ mathematically from prior dynamic sparse attention methods like DSV (Tan et al., 2025) or AdaSpa (Xia et al., 2025)?

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
5

### Summary
This paper tackles the computational bottleneck of full self-attention in video diﬀusion transformers (DiTs). The authors propose NABLA, a training-free, adaptive block-level attention mechanism. The core algorithm works by average-pooling Q/K matrices to compute a cheap, low-resolution attention map. It then applies a Cumulative Density Function (CDF) threshold to dynamically select the most important blocks. It proposes to combine the dynamic sparse mask with STA mask. The proposed method outperforms the STA baseline.

### Strengths
- The paper identifies a consistent failure mode of static sparse attention (such as STA): object duplication and boundary artifacts. It attributes this issue to limited global attention coverage. This observation clearly explains why fixed sparsity patterns can harm video generation quality.
  
- The proposed method is easily implemented using PyTorch without requiring custom CUDA kernels or model retraining.

### Weaknesses
**Contribution and novelty are limited.** The main idea (downsample Q/K, compute coarse attention, and guide sparse masking) is not new; it closely resembles SpargeAttention, MInference, and SeerAttention, etc. The proposed CDF-based thresholding is also previously used in SpargeAttention, FlexPrefill, Twilight, etc. In other words, the proposed main method is exactly the same as other works, and can not be considered a contribution.

**Misaligned self-positioning and unfair comparisons.** Although NABLA is a dynamic sparse method, all comparisons are made against static methods such as STA. Moreover, the mix of NABLA and STA, makes the results look better but unfair. It is hard to know how well NABLA works on its own. Proper baselines such as SparseVideoGen, SparseVideoGen2, SpargeAttention, and RadialAttention should be included.

**Unbalanced narrative and poor writing.** Section 2 (Background) spends excessive space reiterating standard attention equations (Eqs. 13) and describing STA in full detail. The writing and formatting of the paper should be improved. The presentation is sometimes unclear, and the structure and layout make it difficult to follow the main ideas.

**The experiments are not well designed.** They should report both end-to-end quality metrics and efficiency metrics together, rather than evaluating them separately. The baseline only includes one component of the proposed method, namely STA, and lacks comparisons with other relevant baselines.

**Missing empirical analysis.** The key hyperparameter $thr$ is not analyzed, and its chosen values (0.4 and 0.2) lack rationale. Table 3 shows that NABLA with threshold 0.4 already performs as well as or even better than full attention. This result disagrees with the paper’s claim that NABLA alone produces visible artifacts. The statements about border artifacts are only based on examples and are not supported by measurements.

### Questions
Why use CDF thresholding instead of Top-K or ﬁxed-threshold binarization?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces NABLA, a Neighborhood-Adaptive Block-Level Attention mechanism designed to improve the efficiency of video diffusion transformers. By constructing a sparse attention mask through block-wise pooling and adaptive thresholding, NABLA reduces the cost of full self-attention. The method integrates seamlessly into PyTorch’s FlexAttention and achieves substantial training and inference acceleration (up to 2.7×) while maintaining comparable quality across several benchmarks, including CLIP, VBench, and FVD scores.

### Strengths
- The work addresses an important problem in the video generation field—scaling self-attention efficiently. Given the rising cost of video diffusion models, efficiency-focused contributions are timely and valuable.
- The paper is well organized and easy to follow.

### Weaknesses
- The core idea of using pooling-based approximations and block-sparse attention is not new. Similar strategies have been explored in both large language model acceleration methods such as MInference and video diffusion models such as SparseVideoGen. The conceptual overlap reduces the originality of the contribution.
- The paper lacks comparisons with other block-sparse or spatially adaptive attention methods for video generation, such as SpargeAttention, SparseVideoGen, PowerAttention, RadialAttention, and XAttention, which are necessary to evaluate NABLA’s relative performance.
- The evaluation is limited to a single model (Wan 2.1-14B) and a small set of metrics. Broader experiments—including other video generation models (e.g., Hunyuan Video) and additional evaluation metrics like VisionReward—would provide stronger evidence of generalization and robustness.

### Questions
1. How does NABLA fundamentally differ from existing block-sparse attention mechanisms (e.g., MInference) beyond its adaptation to video data?

2. Can the authors provide quantitative or qualitative comparisons against other block sparse attention to better contextualize NABLA’s efficiency and quality trade-offs?

3. Have the authors tested NABLA on other video diffusion architectures, such as Hunyuan and CogvideoX, to verify model-agnostic performance improvements?

4. Could the authors include additional perceptual metrics (e.g., VisionReward, PickScore) or generated video samples to assess generation quality more comprehensively?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a dynamic sparse-attention mechanism for video DiTs. NABLA computes a low-resolution attention map and uses it to derive a dynamic attention mask. Because this alone can introduce visible seams at block boundaries, the method is combined with Sliding Tile Attention (STA). Using the union of STA and NABLA as the attention mask, the authors claim to accelerate both inference and training while preserving output quality.

### Strengths
- Despite its simplicity and the fact it can be implemented with FlexAttention, the method shows promising strong practical gains and is highly effective.

- It can be introduced via fine-tuning into models originally trained with full attention, which is very convenient and increases its applicability.

- The experiments cover the key axes of speed and quality to a reasonable extent.

### Weaknesses
- In Table 2, the runtime comparison between the Baseline (full attention) and the STA/NABLA variants appears to evaluate all settings with FlexAttention. However, the Baseline could (and in practice often would) leverage FlashAttention. Using FlashAttention for the Baseline would be a more realistic and informative comparison.

- Quantitative comparisons against other dynamic sparse-attention methods (e.g., AdaSpa, Sparse-VideoGen, SpargeAttention) are limited, making it hard to highlight NABLA’s relative advantages.

### Questions
- The explanation of Figure 2 feels somewhat brief. Some questions remain about the specific patterns shown and how they are derived/interpreted.

### Soundness
2

### Presentation
3

### Contribution
4
