# MIGA: Make Train-Free Infinite Frame Generation Great Again for Consistent Long Videos

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Without relying on significant computational or data resources, train-free long video generation aims to extend the duration of foundation video generation models, which are typically limited to short videos.
Direct noise prediction on the entire long latents incurs substantial computational overhead.
In contrast, frame-level autoregressive frameworks, e.g., FIFO-diffusion, offer the advantage of generating infinitely long videos with constant memory consumption.
However, the substantial gap between training and inference phases hinders the effective utilization of foundation models. Furthermore, maintaining long-term consistency is central to long video generation, yet existing methods pay insufficient attention to this aspect.
To mitigate these concerns, we propose **MIGA**, a novel infinite-frame long video generation method. 
**(i)** Firstly, considering that the training-inference gap mainly stems from the excessive noise span of latents fed to the model during inference, we propose an effective two-stage alignment mechanism. By partitioning the generation process of existing frameworks into two dedicated stages with reduced noise spans, the capabilities of advanced foundation models are efficiently unlocked.
**(ii)** Additionally, building upon the intrinsic properties of frame-level autoregressive frameworks, we introduce an innovative dual consistency enhancement mechanism. 
Specifically, our self-reflection approach evaluates and corrects early high-noise frames, while our long-range frame guidance approach leverages later low-noise frames with broad coverage to steer the generation process. These strategies jointly promote consistency in the generated content.
**(iii)** Finally, extensive experiments on VBench and NarrLV demonstrate the state-of-the-art performance of MIGA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MIGA, a training-free infinite-frame video generation method that extends foundation diffusion models for consistent long videos. It introduces two key designs: (1) a Two-Stage Training–Inference Alignment mechanism that reduces the noise-span mismatch between training and inference via zigzag denoising and unified refinement, and (2) a Dual Consistency Enhancement mechanism combining self-reflection–based anomaly correction and long-range frame guidance. Experiments on VBench and NarrLV benchmarks show that MIGA achieves state-of-the-art performance, improving subject and background consistency over prior training-free methods like FIFO-Diffusion and FreeLong, while maintaining infinite-frame generation capability.

### Strengths
1. The two-stage denoising process is intuitive yet effective, reducing the noise span mismatch with minimal modification to the inference pipeline.
2. The proposed self-reflection mechanism uses cosine similarity in latent space, eliminating dependence on external models like DINO and improving computational efficiency.
3. The figures and diagrams are highly informative and well-designed.

### Weaknesses
1. The proposed framework lacks a clear overarching motivation. Its components (e.g., TTA, self-reflection, and long-range frame guidance) appear to be designed independently rather than forming a unified, coherent approach to long video generation.
2. The paper does not demonstrate strong scientific contributions; the proposed method resembles more of an engineering pipeline than a conceptually novel framework.
3. Given the complexity of the pipeline, the authors should include a comparison of computational or time complexity with other sota methods.

minor weakness:
1. Line 161 mentions "its length L equals the total number," but L seems absent from the context; did you mean T?

### Questions
1. What is the key contribution of the paper, and which component highlights it?  
2. How do the proposed methods relate and work together to address the research problem, specifically the train-inference gap?

### Soundness
2

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
The paper introduces MIGA, a train-free framework that turns short-video diffusion models into infinite-frame generators with constant memory. It addresses two key issues: the training–inference gap and long-term consistency. First, a Two-Stage Training-Inference Alignment (TTA) reduces the span of noise levels the model must handle at inference: Stage-1 maintains a zigzag latent queue so noise changes more slowly across frames; Stage-2 performs unified-level denoising once all latents reach the same noise level, closely matching the training condition. This yields fewer artifacts and drift and is detailed with an explicit queue-update algorithm.
Second, Dual Consistency Enhancement (DCE) combines Self-Reflection (detects anomalies on high-noise tail latents and selectively re-samples to correct them) and Long-Range Frame Guidance (feeds sparse, cleaner anchor frames alongside local windows to promote distant-frame interaction). DCE adds little overhead for guidance and offers tunable test-time scaling for self-reflection; ablations show consistent gains on VBench metrics (subject/background consistency, motion smoothness, etc.). Demonstrations include ~1000-frame videos on Wan2.1-1.3B and ~600-frame on VideoCrafter2

### Strengths
MIGA’s big win is that it’s truly train-free yet still turns short-video diffusion models into infinite-frame generators with constant memory. It closes the training–inference gap with a neat two-step trick— a zigzag latent queue to keep neighboring frames at similar noise levels, followed by unified-level denoising—so no drift or flicker as videos get longer. Then it boosts long-range consistency with Dual Consistency Enhancement: lightweight self-reflection to catch and fix anomalies on the fly, plus long-range frame guidance that pulls in clean anchor frames to keep characters and backgrounds stable. It’s plug-and-play (works with VideoCrafter2, Wan2.1, etc.), scales to 600–1000+ frames, and shows solid gains on VBench/Narrative metrics—practical, efficient, and easy to adopt.

### Weaknesses
1. Novelty vs. prior art: The “train-free long-video” direction already includes methods like FreeNoise / FreeLong / FreePCA (extend duration by re-using or transforming noise) and Diffusion-Forcing / AR-Diffusion / FIFO-Diffusion (queue/FIFO autoregression with constant memory). MIGA builds on this paradigm by adding alignment and consistency enhancements (TTA + DCE), so its contribution is more incremental than first-of-its-kind on “∞-frame with constant memory.”
2. Model diversity: Experiments are run on two base models only—VideoCrafter2 and Wan2.1-1.3B—with some discussion that stylistic differences can sway different metrics. Evidence of broad cross-model generalization is therefore limited.
3. Benchmark/data breadth: The evaluation focuses mainly on VBench (subject/background consistency, motion smoothness, flicker) and NarrLV (narrative metrics) against baselines like FIFO-Diffusion, FreeLong, FreePCA, and ScalingNoise. The benchmark surface is relatively narrow—there’s little coverage of diverse, real-world datasets or large-scale human studies—so external validity remains to be strengthened. The appendix also shows failure cases (e.g., structural mismatches on objects), indicating consistency is not fully solved.

### Questions
See Weakness

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
This paper addresses the challenge of extending pre-trained short video generation models to create coherent, long videos in a train-free manner. It identifies two major flaws in existing frame-level autoregressive frameworks, such as FIFO-Diffusion: 1) A significant "training-inference gap," as models trained on latents at a single noise level are forced during inference to process a queue of latents with a wide span of different noise levels, leading to artifacts. 2) A failure to model long-term dependencies, which results in poor temporal consistency as the video progresses.

To solve these issues, the authors propose MIGA, a method with two main contributions. First, it introduces a Two-Stage Training-Inference Alignment (TTA) mechanism. Stage 1 uses "zigzag iterative denoising," which groups latents by noise level (using a width of $L_{zig}$) to reduce the noise span seen by the model. Stage 2 completes the process by performing a unified denoising step once all latents have reached the same intermediate noise level $\tau_{e-1}$, perfectly aligning with the model's training conditions. Second, MIGA employs a Dual Consistency Enhancement (DCE) mechanism. This includes a "self-reflection" approach to efficiently evaluate consistency on new, high-noise latents and trigger a corrective search if anomalies are detected, and a "long-range frame guidance" method that injects older, low-noise latents ($m_{guid}$) into the current processing window to enforce temporal coherence. Experiments show MIGA achieves state-of-the-art results on VBench and NarrLV benchmarks.

### Strengths
+ The paper tackles a well-known, difficult problem (long-term consistency in long video generation) from a novel and practical angle. While it builds on existing frame-level autoregressive (AR) frameworks like FIFO-Diffusion, its contributions are highly original. The "Two-Stage Training-Inference Alignment" (TTA) mechanism, particularly Stage 2 (Unified Denoising), is a very clever solution to the training-inference gap by forcing the final denoising steps to perfectly match the training distribution (noise span of 1). Furthermore, the "Dual Consistency Enhancement" (DCE) mechanism's "Self-Reflection" component is also very original. The insight to use noisy-latent self-similarity as a computationally cheap proxy for clean-latent consistency (as shown in Fig. 3) to trigger a test-time search is a significant and non-obvious contribution.

+ TThe methodology is well-motivated, and the two main components (TTA and DCE) directly address the two identified weaknesses of prior work (training-inference gap and long-term consistency). The experimental validation is thorough and convincing. The authors use two different foundation models (VideoCrafter2 and Wan2.1) and two standard benchmarks (VBench, NarrLV). The ablation study in Section 4.3 is exemplary; it methodically deconstructs the proposed system, providing strong evidence for the independent and combined contributions of TTA (Stage 1 and Stage 2) and DCE (Self-Reflection and Long-Range Guidance). The qualitative results, especially the step-by-step ablation in Figure 4, clearly visualize the impact of each component.

+ The paper is exceptionally well-written and easy to follow. The core limitations of the baseline (FIFO-Diffusion) are explained with intuition (training-inference gap) and illustrated well (Fig. 2a). The proposed MIGA solution is logically broken down into its TTA and DCE components, each of which is explained with clear diagrams (Fig. 2b, Fig. A1) and, in the appendix, detailed pseudocode. The connection between the observed problem and the proposed solution is direct and compelling.

+ Training-free methods for extending foundation models are of immense practical value, as they democratize access to powerful capabilities (like long video generation) without requiring massive computational resources for retraining. This paper provides a robust and well-engineered solution that demonstrably improves the state-of-the-art in a challenging domain. By solving key issues in AR frameworks, MIGA makes infinite-frame generation significantly more consistent and stable, pushing the boundaries of what can be achieved without additional training.

### Weaknesses
- "Infinite Generation" Claim vs. TTA Stage 2: There appears to be a contradiction between the paper's claim of "infinite-frame generation" with "constant memory" (as inherited from FIFO-Diffusion) and the mechanics of the proposed TTA Stage 2. As described in Section 3.2 and Algorithm 5 (lines 16-26), Stage 2 collects all N partially denoised latents (where N is the total length of the final video) into a queue Q_gen. It then performs unified denoising steps on this entire queue. This implies that memory usage scales with N, the total number of frames to be generated. This is a fundamental departure from the streaming, constant-memory paradigm of FIFO-Diffusion, which generates one clean frame at a time. This re-introduces the very memory-scaling problem that AR frameworks were designed to solve. While the method is still "autoregressive" in Stage 1, the overall process seems to be for fixed-length (though long) video generation, not a truly "infinite" stream. This is a major weakness that needs to be clarified, as it affects the paper's core premise.

- Missing Latency/Throughput Analysis: The paper focuses entirely on quality improvements (VBench/NarrLV scores) but provides no analysis of the computational overhead. The baseline FIFO-Diffusion is already computationally intensive. The proposed MIGA adds two new sources of overhead: TTA (specifically Stage 2, which processes all N frames e-1 times) and DCE (which involves a test-time search, as acknowledged in Appendix B.3). While Appendix B.3 discusses the cost, it provides no concrete numbers. A key part of evaluating a new method is understanding its trade-offs. How much slower is MIGA than FIFO-Diffusion? Is it 1.1x, 2x, or 10x slower? Without a "Latency vs. Quality" comparison (e.g., in Table 1), it's hard to judge the practical utility of the method. The gains in consistency may not be worth a massive drop in throughput.

### Questions
- Memory Scaling of TTA Stage 2: Could the authors please clarify the apparent contradiction regarding the "infinite" and "constant memory" claims? Algorithm 5 suggests memory scales with N (total frames). The authors may explain the memory-management mechanism of Stage 2 that maintains constant memory.

- Practical Latency Overhead: What is the wall-clock latency (or throughput in frames/sec) of the full MIGA (TTA+DCE) method compared to the baseline FIFO-Diffusion when generating a video of a fixed length (e.g., 128 or 161 frames, as in Table 1)? A concrete comparison is needed to evaluate the practical cost-benefit trade-off of the proposed mechanisms.

- Effectiveness of "Zigzag" Denoising (TTA Stage 1): The "zigzag" denoising in Stage 1 is proposed to "proactively narrow the noise span". However, any sliding window of size f_0 (e.g., f_0=16) would still seem to cover at least two different noise levels (e.g., ...2, 2, 2, 3, 3...), giving a span of 2. The baseline FIFO-Diffusion has a span of f_0. Is the improvement (which is confirmed in Table 6) really just from going from a span of f_0 to a span of 2? Or is there another factor at play?

- Robustness of Self-Reflection Proxy: The use of noisy-latent similarity as a proxy for clean-latent consistency (Fig. 3) is a key insight for the "Self-Reflection" component. How sensitive is this? The paper mentions it works even at high noise levels (e.g., 40/50). Does this proxy become more or less reliable at different noise levels? How was the judgment index f_judg (the noise step at which to perform this check) chosen?

### Soundness
2

### Presentation
3

### Contribution
2
