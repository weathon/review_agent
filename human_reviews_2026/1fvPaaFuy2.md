# Entropy-Select: Training-Free Local Entropy Token Compression for Video LLMs

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Video Language Models (VLMs) emit visual tokens that grow linearly with video length while attention scales quadratically, making inference sensitive to token count and latency variance. In practice, an effective token compressor should be training-free to plug into strong off-the-shelf VLMs, architecture-agnostic to avoid model-specific hooks, and deliver selection runtime that is predictable regardless of the target retention—properties that many prior methods lack. We introduce EntropySelect, a training-free, architecture-agnostic framework that ranks tokens by local neighborhood entropy, an information-theoretic measure of unpredictability relative to nearby tokens. We compute temperature-scaled similarity distributions within fixed spatial windows to obtain normalized entropy, fuse it with gradient-based structural saliency, enforce coverage via grid quotas, and allocate per-frame budgets by saliency. Because windows/grids are fixed and selection reduces to a single global sort, EntropySelect runs in O(N log N) with latency effectively decoupled from the retention ratio. Across MVBench, ActivityNet-QA, VideoMME, and EgoSchema, EntropySelect matches or exceeds the uncompressed baseline at moderate retention and degrades gracefully under aggressive compression, yielding predictable latency and plug-and-play deployment without any model modification or internal-state access. Notably, we observe “enhancement under compression” on several benchmarks—for example, 35% and 50% retention surpass the full-token baseline on benchmarks such as MVBench and EgoSchema—suggesting that removing low-entropy redundancy can improve downstream accuracy.These results establish local entropy as a principled criterion for video token selection and a practical tool for scalable VLM inference. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ENTROPY-SELECT, a training-free, architecture-agnostic token selection framework for video language models (VLMs). It scores tokens by local neighborhood entropy, fuses gradient-based saliency, and enforces coverage via grid quotas and per-frame allocation. The method achieves O(N log N) complexity with latency decoupled from retention ratio, demonstrating comparable or even better performance than the uncompressed baseline across multiple benchmarks (MVBench, EgoSchema, VideoMME, ActivityNet-QA).

### Strengths
1. Clear motivation & strong deployment practicality – The paper convincingly motivates why a training-free, architecture-agnostic approach is essential for scalable VLM inference, addressing deployment gaps in previous works like LLaVA-Scissor or VisionZip.

2. Novel use of local entropy for token importance – Introducing normalized neighborhood entropy as an information-theoretic measure of token unpredictability is elegant and theoretically grounded.

3. Excellent empirical results – ENTROPY-SELECT achieves near-baseline or even superior results under 35–50% token retention, with measurable latency benefits and predictable runtime.

### Weaknesses
1. Limited theoretical depth – While the motivation is information-theoretic, the derivation lacks formal analysis connecting entropy scores to mutual information or rate–distortion objectives.

2. Scope of experiments – All evaluations are on LLaVA-OneVision; portability to other VLMs (e.g., InternVideo2, Video-LLaMA) is not demonstrated, which weakens the “architecture-agnostic” claim.

3. Missing ablation on hyperparameters & τ sensitivity – The paper mentions τ and α choices but lacks discussion on robustness across different datasets or architectures.

### Questions
1. How sensitive is the method to the feature quality of the vision encoder (e.g., SigLIP vs. CLIP)?

2. Does the local entropy score depend on absolute token feature magnitudes? Was any normalization across frames applied?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ENTROPY-SELECT, a training-free, architecture-agnostic framework that ranks tokens by local neighborhood entropy—an information-theoretic measure of unpredictability relative to nearby tokens. Temperature-scaled similarity distributions within fixed spatial windows are used to compute normalized entropy, which is then fused with gradient-based structural saliency; coverage is enforced via grid quotas, and per-frame budgets are allocated by saliency.

### Strengths
1. This paper presents a training-free, architecture-agnostic token compression pipeline that uses fixed-window entropy estimation and a single global sort, achieving O(NlogN) runtime with predictable latency.
2. Experiments on MVBench, ActivityNet-QA, VideoMME, and EgoSchema show that ENTROPY-SELECT matches or surpasses uncompressed accuracy at moderate retention and degrades gracefully under aggressive compression.

### Weaknesses
1. Since the method is claimed to be architecture-agnostic, broader validation on model families where many existing approaches are inapplicable is necessary.

2. The evaluation relies on LLaVA-OneVision, which is relatively outdated; stronger, modern models (e.g., Qwen-3-VL) should be included.

3. Please report performance under more extreme retention (e.g., 10% and 20% of tokens), which is common and necessary for video-LLM token/KV-cache compression.

4. Compare against finer-grained, head-level budget allocation methods (e.g., SparseMM) to test whether the proposed advantages persist.

5. Discuss recent lossless VLM acceleration via speculative decoding (e.g., EMNLP-2025-SpecVLM, NeurIPS-2025-ViSpec), clarify differences and complementarities, and consider a combined setting with token compression.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

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
This paper proposes a method, ENTROPY-SELECT, that prunes tokens for VLM inference to increase efficiency while minimizing the performance degradation. Compared to other methods, this paper posits that VLM token pruning should be (1) training free (2) architecture agnostic (3) have close to constant runtime. The proposed method achieves this by using heuristics on the input features to identify a fixed fraction of features to remove. This does not require access to any component of the attention mechanism, and is thus more architecture agnostic while also being training free. The experiments demonstrate that at the same compression levels, Entropy-Select outperforms other comparable baselines.

### Strengths
1) The writing and presentation of the paper is crisp and clear and it is easy to understand exactly how the method works and how it might be implemented. 

2) The experimental evaluation is comprehensive and clearly demonstrates that the proposed method does outperform other similar baselines.

3) The method's focus on working only on the `exported' attention features means it is genuinely more flexible across VLM architectures compared to most existing works which require access to some component of the attention computation.

### Weaknesses
1) I'm not convinced that the three main desiderata for the paper are really important. Why does it matter that the selection runtime remain perfectly predictable? I think for complex inputs, it is fine that the selection method takes slightly longer - as shown in the figure, this difference amounts to a rather small and negligible wall-clock difference. Furthermore, I don't quite agree that training-free is a strong desiderata. Trained methods generally outperform training-free ones, so while training-free methods are more easily adaptable, a trained method only requires a one-time cost and can potentially save more cost at inference time. 

2) The results are quite incremental. The accuracy boost is somewhat small (only really significant at 10%) and the speed-up over state of the art baselines is also rather small. 

3) There’s no real analysis on what tokens entropy-select decides to prune vs other methods, so it’s not clear why this performs better than others, especially learned or others. While I understand the intuition for using local entropy as a metric, why is it obvious that less predictable tokens are less helpful? A more rigorous analysis (qualitative or quantitative) would be very helpful here and make a stronger case for the paper.

4) The heuristics seem rather fragile and require tuning - wouldn't determining edge (gradient) and entropy thresholds change based on the model backbone?

5) Is removing a constant number of tokens from the input actually optimal? One could image a video that is just pure white, where almost everything could be removed, but entropy-select would remove a predefined compression ratio no matter what.

### Questions
1) Why does the entropy window only look at the past? Are all video LMs causal?

2) Complexity and FLOPs are not particularly useful in this setting - total memory usage and wall-clock inference time would be better metrics to report here. This is briefly discussed in 4.4 but a figure to show the end-to-end breakdown would be nice.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ENTROPY-SELECT, a training-free and model-agnostic token compression method for video LLMs. Specifically, it fuses local spatial-temporal neighborhood entropy and gradient-based structural saliency to rank the visual tokens. And then, it ensures spatial coverage via grid partitions, and assigns per-frame budgets by token ranking importance scores. Experimental results show that the proposed method can outperform previous works in multiple benchmarks and can even beat the uncompressed baseline in certain metrics.

### Strengths
1. The idea of mining information-diverse tokens via entropy and spatial salient tokens via gradient, and then fuse them for token importance ranking is well-motivated and reasonable. 

2. This paper further carefully developed several strategies to make sure the tokens are selected in a balance way, including gradient-entropy fusion, grid-aware coverage, and frame-based allocation.

3. The experiments are conducted on multiple benchmarks, and the proposed model generally outperforms previous works. Ablation is done on several key components of the method.

### Weaknesses
1. The author(s) claim the proposed method is model/architecture agnostic. However, all the experiments are only conducted on one model - LLaVA-OneVision model with SigLIP-SO400M-patch14-384 as visual encoder and Qwen-2.5-7B as the language model. To demonstrate the generalizability of this model-agnostic method, at least one or two additional different backbones should be experimented with.

2. There are too many hyperparameters that need to be manually set up in the proposed method. For example, in Local SpatioTemporal Window Entropy: the spatial window size `k`, temporal window depth `K_t`, and temperature `t` ; in section 3.3: region `g`; in section 3.4: top-q fraction, `B_min`, etc. While some of these are partially ablated (e.g., the entropy weight in Table 5), others appear fixed without justification. Having so many heuristic settings makes the method sensitive to specific configurations and difficult to adapt to different backbones and usecases (eg, longer video scenarios).

### Questions
See weaknesses

### Soundness
2

### Presentation
4

### Contribution
3
