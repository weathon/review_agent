# Rolling Forcing: Autoregressive Long Video Diffusion in Real Time

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Streaming video generation as one fundamental component in interactive world models and neural game engines aims to generate high-quality, low-latency, and temporally coherent long stream videos. However, most existing work suffers from severe error accumulation that often significantly degrades the generated stream videos over long horizons. We design Rolling Forcing, a novel video generation technique that enables streaming long videos with minimal error accumulation. Rolling Forcing comes with three novel designs. First, instead of iteratively sampling individual frames which accelerates error propagation, we design a joint denoising scheme that simultaneously denoises multiple frames with progressively increasing noise levels. This design relaxes the strict causality across adjacent frames, effectively suppressing error growth. Second, we introduce the attention sink mechanism into the long-horizon stream video generation task, which allows the model to keep key–value states of initial frames as a global context anchor and thereby enhances long-term global consistency. Third, we design an efficient training algorithm that enables few-step distillation over largely extended denoising windows. This algorithm operates on non-overlapping windows and mitigates exposure bias conditioned on self-generated histories. Extensive experiments show that Rolling Forcing enables real-time streaming generation of multi-minute videos on a single GPU, with substantially reduced error accumulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Rolling Forcing (RF), a streaming text-to-video diffusion method that reduces long-horizon error accumulation while keeping real-time throughput. Key ideas: (i) a rolling denoising window that jointly denoises several consecutive frames with bidirectional attention inside the window but emits one clean frame per forward pass; (ii) a global/temporal KV cache with dynamic RoPE so early frames act as a stable “attention-sink” anchor over minutes; and (iii) few-step DMD distillation on non-overlapping windows plus a mixed Self-Forcing (SF) objective for more natural motion. On VBench, RF improves quality and notably lowers quality drift vs. open AR baselines, with ablations for each component.

### Strengths
Effective drift control: Rolling window enables mutual refinement before a frame is finalized; ablations show large Quality_Drift gains.  
 Long-term consistency trick: Global KV cache + dynamic RoPE keeps initial context usable without RoPE over-offset artifacts. 
 Real-time demo & interactivity: Sub-second steady-state latency and prompt-switching during streaming.

### Weaknesses
* Attention-sink rationale needs deeper evidence: “Attention sink” comes from LMs; here it’s adapted via KV caching + dynamic RoPE. The paper shows ablations, but provides limited analysis of why the sink phenomenon should emerge in video DiTs. As written, it risks reading as a concept transplant without sufficient mechanistic support.  
* Motion/scene diversity trade-off: Demos suggest mis-tuning can collapse into static outputs, RF may bias toward stillness to curb drift. More systematic tests compared to baselines on large motion and scene changes are needed, you may use like dynamic degree metric in Vbench.  
* Evaluation scope: Metrics center on 30 s clips (quality drift measured 0–5 s vs. last 5 s). Given the multi-minute claim, add standardized 2–5 min quantitative runs (and a small human preference study).  
* Scaling detail: Limited profiling across window/model sizes

### Questions
Please see weaknesses

### Soundness
3

### Presentation
2

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
This paper introduces Rolling Forcing, a novel framework for real-time autoregressive long-video diffusion. The key idea is to mitigate error accumulation in streaming generation by performing joint denoising over a rolling window of consecutive frames rather than processing single frames sequentially. Three main contributions are proposed: (1) a rolling-window joint denoising strategy enabling mutual refinement across neighboring frames while keeping real-time throughput; (2) integration of an attention-sink mechanism that anchors initial frames as a global context to maintain long-term temporal consistency; and (3) an efficient few-step distillation algorithm over non-overlapping windows that conditions on self-generated histories to reduce exposure bias.
Experimental results show that Rolling Forcing can stream multi-minute videos at 16 FPS on a single GPU with minimal drift, achieving state-of-the-art performance on the VBench benchmark and outperforming prior autoregressive baselines such as Self Forcing and CausVid in both visual quality and temporal stability.

### Strengths
1. Rolling forcing is a robust and effective framework. Components like progressively increasing noise levels, attention sink, and few-step distillation enhance long video generation.  
2. Compared to baselines like causvid and self-forcing, rolling forcing significantly reduces error accumulation and extends video length while maintaining content consistency.

### Weaknesses
The weaknesses primarily stem from the novelty and experiments, as outlined below:  
1. The proposed components are not entirely novel, as they have been used in video generation. For example, progressively increasing noise is used in Rolling Diffusion[1], MAGI-1[2], and FIFO-Diffusion[3]. Attention sink has been applied in LLMs[4], and step distillation is utilized in self-forcing[5]. This paper seems to integrate these components into an autoregressive video generation paradigm.  
2. The paper does not compare its approach to related works in autoregressive video generation, such as FramePack[6], which it cites.  
3. The training length in the paper is set to 27, with a KV cache length of 24 frames. It raises questions about how the attention sink performs with such a short training length.  
4. In Table 2, the second row (without RF training) achieves strong performance, though still lower than rolling forcing. This raises the question of the key benefits of RF training.

[1] Rolling Diffusion Models.
[2] MAGI-1: Autoregressive Video Generation at Scale.
[3] FIFO-Diffusion: Generating Infinite Videos from Text without Training.
[4] When Attention Sink Emerges in Language Models
[5] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
[6] Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models.

### Questions
1. What motivates the focus on long video generation, and how is this challenge tackled?  
2. How does this paper's contribution distinguish itself from the limitations of related works?  
3. How do the results compare to FramePack?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposed a method to generate long video without drifting error and enables streaming video. Instead of causal diffusion, it uses progressive noise level for denoising. It utilizes attention sink to keep long term video consistency and quality. Finally it proposed a few-step distillation method over extended window to mitigate exposure bias conditioned on self-generated histories.

### Strengths
- introducing rolling diffusion, i.e., progressive noise level for multiple video latents, into self-forcing pipeline to achieve smoother and lower error drifting denoising framework.
- history frame use causal attention so as to be saved in KV cache to save inference compute.
- adopted attention sink with a few global context at the beginning of the sequence. Adjust RoPE dynamically to avoid PE index out of range.
- novel design of DMD loss on progressive noised latent to save computation cost.
- the error drifting problem has been improved significantly.

### Weaknesses
- the design of using a subset of backpropogated gradients for DMD loss training alleviate the memory and compute cost. But has the training quality and efficiency been affected at the same time? For example, it is mentioned that the original SF training is still needed to regularize the training. It would be helpful to have a discussion about advantage and disadvantage of it since this is one of the core contribution.
- the overall design resolves the error drifting significantly, but other metrics, e.g., fps, latency, VBench score, are not improved or improved marginally.

### Questions
- How does attention sink perform for long video with context dramatically changed?
- It is not clear how the mixed training strategy (SF + Rolling Forcing) performed practically. Besides, is this training strategy stable? It would be helpful to add a more detailed discussion of it.
- Since RoPE is very sensitive, does dynamically changing RoPE for attention sink bring any artifacts or instability of the output videos?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes "Rolling Forcing," a method designed to alleviate error accumulation in long-horizon video generation. It uses a rolling denoising window and introduces an attention sink mechanism to improve temporal coherence and achieve real-time rendering through few-step distillation. The paper includes detailed pseudocode and extensive engineering contributions, clearly explaining the training process.

### Strengths
- The proposed attention sink clearly helps alleviate error accumulation issues, as demonstrated in Figure 4.
- Few-step distillation allows real-time rendering, making the approach practical for streaming video generation.
- The paper includes substantial engineering work, and the provided pseudocode clearly describes the training process in detail.

### Weaknesses
- A significant limitation of the proposed approach is that the use of the attention sink and Rolling Forcing may not generalize well to long videos with drastically changing content. This setup could restrict substantial changes in video content, causing generated videos to appear as repetitive sequences or stitched clips (as evident in all the provided supplementary videos). Additionally, continuity issues arise when key frames do not provide rich enough information.
- The evaluation is somewhat limited, relying only on metrics from VBench. A broader evaluation would strengthen the claims.

### Questions
- Could the authors discuss how well their approach handles scenarios with significant changes in video content?

### Soundness
4

### Presentation
3

### Contribution
3
