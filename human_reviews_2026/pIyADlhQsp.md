# CausNVS: Autoregressive Multi-view Diffusion for Flexible 3D Novel View Synthesis

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Multi-view diffusion models have shown promise in 3D novel view synthesis, but most existing methods adopt a non-autoregressive formulation. This limits their applicability in world modeling, as they only support a fixed number of views and suffer from slow inference due to denoising all frames simultaneously. To address these limitations, we propose CausNVS, a multi-view diffusion model in an autoregressive setting, which supports arbitrary input-output view configurations and generates views sequentially. We train CausNVS with causal masking and per-frame noise, using pairwise-relative camera pose encodings (CaPE) for precise camera control. At inference time, we combine a spatially-aware sliding-window with key-value caching and noise conditioning augmentation to mitigate drift. Our experiments demonstrate that CausNVS supports a broad range of camera trajectories, enables flexible autoregressive novel view synthesis, and achieves consistently strong visual quality across diverse settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposed CausNVS, an autoregressive multi-view diffusion model for flexible NVS. Formally, CausNVS trained the multi-view diffusion with independent frame-wise noise to support the autoregressive generation. Then, CausNVS used CaPE to denote relative positional encoding, which is more robust than the popularly used absolute positional encoding (pluckr ray). Finally, the KV cache and window attention are used to improve the generation stability and reduce the inference cost. Experiments show the effectiveness of the proposed method.

### Strengths
1. This paper pointed out an interesting question for the autoregressive novel view synthesis, i.e., how to incrementally formulate the infinitely changed camera trajectories to AR models. Using relative positional encoding (CaPE) is somewhat effective, but this still suffers from some concerns, as detailed in the weaknesses.

2. The presentation of this paper is clear and easy to follow. The authors provided sufficient implementation details in the appendix.

### Weaknesses
1. This work presents an integrated system, but most of its core components (e.g., Diffusion Forcing, CaPE, KV cache, and window-wise attention map) have been previously proposed in existing literature, which weakens the paper’s novelty. The performance improvement is not significant (Table1 shows comparable and even inferior results compared to SEVA).

2. A major concern is the CaPE’s design and efficacy for "progressively larger poses" for AR. 

a) The first question is why CaPE does not require any normalization ("scale sweeping") as mentioned in Line 413-414? As formulated in Eq(2), CaPE is very similar to PRoPE[1] (it would be very helpful if the authors could discuss the relation between CaPE and PRoPE), while the unnormalized translation values will potentially undermine the numerical stability of attention computation. Depending on the metric methods and SfM pipelines used, some translation values could become excessively large, leading to unstable model training or inference. 

[1] Li R, Yi B, Liu J, et al. Cameras as relative positional encoding[J]. arXiv 2025.

b) As shown in Figure 2, the first frame is consistently retained in the window-wise (frame-wise) attention. When the disparity between the first and last frames is extremely large (i.e., large translation values), the relative translation term in Eq. (2) will introduce distribution shifts between the training and inference processes. This shift may degrade the model’s generalization to address large-distance moving. So simply using the relative position encoding instead of the absolute one without proper normalization may not effectively address the large-pose challenge.

3. Most experiments are just conducted on the in-domain dataset (real10k, dl3dv), while the out-of-distribution performance is not well studied. For example, the authors should consider datasets like Tanks-and-Temple, MipNerf360, and some stylized pictures to ensure generalization.

4. To demonstrate the effect of AR and invariant pose encoding, the authors should include some convincing evidence, such as infinite or extremely long NVS generation along a specific way with good memory maintenance.

### Questions
1. As shown in Figure 2, why is the first frame always included in the KV cache? I could not find any discussion for this point in the paper.

2. The authors should clearly clarfiy why using CaPE is convincing to address the large pose issue for AR.

### Soundness
2

### Presentation
3

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
This paper focuses on novel view synthesis. The technical approach is learning-based, leveraging a multiview diffusion model equipped with augmentations including a Relative Camera Pose Encoding and causal mask. Experiments show that the proposed approach is comparable to state-of-the-art NVS models.

### Strengths
- The figure illustrations are clear. The attached website is informative.

### Weaknesses
- I'm not too convinced by the results. In particular: 
(1) There is no comparison to camera-controlled video generation models such as https://research.nvidia.com/labs/toronto-ai/GEN3C/
(2) There is no comparison to large NVS transformers, such as https://haian-jin.github.io/projects/LVSM/ and its follow-up works.
(3) The video results in the website seem to have severe aliasing artifacts. Most examples have this issue.
(4) The video results in the website seem to have blurry artifacts. For example, in the first section, first row, 3rd column, the chair becomes pretty blurry when the camera goes forward.

### Questions
I think these may strengthen the submission:

- Comparison to baselines from camera controlled video generation and large NVS models. 
- More video results and discussion on the aliasing/blurry artifacts.

### Soundness
2

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
This paper introduces a causal diffusion model for novel view synthesis, which generates novel views sequentially. The proposed method adapts a multi-view diffusion model to a causal prediction setting by integrating several strategies, including causal masking in the attention layers, per-frame noise sampling, pairwise relative camera pose encodings, and a spatially-aware sliding-window mechanism with KV caching. Experiments demonstrate that this approach outperforms existing baseline models.

### Strengths
* The motivation of synthesizing novel views in an autoregressive manner is a promising and important direction for achieving arbitrary-length view synthesis, which has significant practical value.
* The proposed method achieves better results compared to the established baselines, demonstrating its effectiveness.
* The paper is well-written, clearly structured, and easy to understand.

### Weaknesses
* My primary concern is the paper's limited technical novelty. The method appears to be a skillful integration of several known techniques (causal masking[1], per-frame noise[1], relative pose encodings[2]) to adapt a non-causal model to a causal setting. While this combination is effective, the work does not seem to introduce new fundamental knowledge or concepts. The contribution lies more in the engineering and application of these components rather than in proposing a conceptually novel method.

[1] Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural
Information Processing Systems (NeurIPS), 2024a.


[2] Eschernet: A generative model for scalable view synthesis. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2024.

* Regarding the baselines, I noticed the comparisons are focused primarily on other multi-view synthesis methods. I am curious how the proposed method stacks up against modern video generation models. Since these models excel at generating temporally coherent new frames, which is conceptually similar to sequential novel view synthesis. Could the authors provide a comparison with video-based novel view synthesis methods? This would help to better position the paper's contribution within the broader field.

* The paper claims the ability to generate arbitrary-length sequences. However, the experiments seem to be conducted on relatively short sequences. I am skeptical about the model's ability to maintain long-term consistency. How does the model perform on much longer sequences (e.g., 100+ frames)? Does the quality degrade or do errors accumulate over time? An analysis or qualitative result on a long sequence would be necessary to fully substantiate this claim.

### Questions
* I have a question regarding the use of CAPE for positional embeddings. This technique was originally designed for object-centric representations. How does it handle arbitrary camera paths in large, unbounded scenes? A discussion on the suitability of CAPE for this new, more general setting would be insightful.

### Soundness
2

### Presentation
2

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
This paper proposes an autoregressive framework for novel view synthesis to take into account previously generates frames. The approach introduces relative camera pose encodings to reuse the KV cache during inference. Additionally, it incorporates noisy frame conditioning and teacher forcing for autoregressive generation. Experiments are performed on RealEstate10K,  LLFF, DL3DV,  Long (Short) datasets.

### Strengths
1. Teacher forcing and noisy frames during training a me noise conditioning during training encourage robustness to imperfect inputs and contexts. 
2. KV caching is used for efficiency at inference.
3. The work proposes relative camera pose encodings (CaPE) which helps the model to adapt to shifting camera trajectory and reuse the KV cache efficiently.

### Weaknesses
1. The focus is on improving the inference time performance and efficiency. Experiments do not show the inference time overhead of the models. The experiments do not report actual inference time, FLOPs, or latency metrics. Without these, it’s unclear whether the proposed design achieves meaningful speedups in practice.
2. Quantitative results show that the models lags behind prior art in terms of PSNR metrics.
3. What is the context window considered for autoregressive generation? How many past frames are considered?

### Questions
1. How does the approach compare in terms of inference time with respect to the prior work which generates views in parallel? How does this compare to the baselines without KV caching?
2. How does the model adapt to long horizons. How is the drift mitigated with the increasing sequence length? 
3. Is it feasible to use this approach in real time applications? 
4. What is the minimum difference between the relative camera poses for the CaPE embeddings to be useful?
5. Is it possible if the input frames are less than the generated that the model's performance can suffer compared to parallel approaches?

### Soundness
2

### Presentation
3

### Contribution
2
