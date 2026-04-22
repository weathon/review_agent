# Streaming Visual Geometry Transformer

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Perceiving and reconstructing 3D geometry from videos is a fundamental yet challenging computer vision task. To facilitate interactive and low-latency applications, we propose a streaming visual geometry transformer that shares a similar philosophy with autoregressive large language models. We explore a simple and efficient design and employ a causal transformer architecture to process the input sequence in an online manner. We use temporal causal attention and cache the historical keys and values as implicit memory to enable efficient streaming long-term 3D reconstruction. This design can handle low-latency 3D reconstruction by incrementally integrating historical information while maintaining high-quality spatial consistency. For efficient training, we propose to distill knowledge from the dense bidirectional visual geometry grounded transformer (VGGT) to our causal model. For inference, our model supports the migration of optimized efficient attention operators (e.g., FlashAttention) from large language models. Extensive experiments on various 3D geometry perception benchmarks demonstrate that our model enhances inference speed in online scenarios while maintaining competitive performance, thereby facilitating scalable and interactive 3D vision systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an online, low-latency 3D reconstruction pipeline. It achieves this by (1) temporal causal attention, cross-attending to cached KV-tokens from previously observed frames, and (2) utilizing knowledge distillation during training by setting VGGT (a full SA-based reconstruction model) as the teacher to provide the GT for the causal student model.

### Strengths
1. The paper is well written and easy to follow. The problem is well motivated and explained.
2. The proposed causal attention is simple but effective, and performs well empirically.

### Weaknesses
1. In Table 6, the impact of KD is rather drastic, which makes the contribution of the causal attention questionable. When reporting results for w/o KD, is the 'true' GT used as supervision? The authors claim that KD helps reduce error accumulation, but this claim needs a more involved analysis.

### Questions
1. Did the authors explore strategies for constraining/truncating the memory usage for very long scenes? Perhaps simply incorporating a recency bias i.e. cacheing and cross-attending to only the most recent K scenes' tokens instead. An ablation over different K can be helpful.
2. Can the authors explain the results reported in Figure 6?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes StreamVGGT, which reconstructs 3D geometry from videos in an online manner. They do this by adopting a causal transformer architecture and caching the historical keys and values to use for each incremental new frame. They train StreamVGGT through knowledge distillation of VGGT, which is state-of-the-art, "offline", but slower. For 3D reconstruction, their approach increases inference speed relative to offline methods, but with a small performance reduction. They are competitive with (and sometimes slightly better than) other streaming methods.

### Strengths
1. This paper's approach of caching memory tokens for streaming reconstruction is important for efficiency, and it is useful for the field to see the results and comparisons they report. I think this makes the paper useful and worth publishing. 
2. The paper is written clearly and provides a very readable explanation of their method and other methods. 
3. StreamVGGT is competitive with other streaming methods, while adopting a significantly different approach. The additional experiments (inference speed, memory, distillation ablations) also serve as useful information.

### Weaknesses
1. It would be useful to explicitly compare the inference (and training, if possible, but understandably this may be harder) speed and memory consumption of their approach against other streaming reconstruction methods like CUT3R and Spann3R. I think this is crucial because the core argument of their paper is efficiency.
2. "StreamVGGT leverages the inherent sequential and causal nature of real-world video data, constraining the attention mechanism to past and current frames, thereby aligning with the causal structure observed in human perception." I agree that humans receive input frames in an online streaming manner. However, do we know if there is evidence that their representation of old views is not modified by (and so does not "attend to") new views, like in KV caching? One counterpoint is that our stored experiences and memories are modified by new information. It is possible that humans use some hybrid: KV-caching where stored keys/values are allowed to be modified. At a few points in this paper, the paper implicitly suggests that StreamVGGT is more human-like than full self-attention approaches. I think this requires more evidence, or otherwise hedge the claim.

### Questions
1. To what extent is the inference speedup and memory reduction (Figure 2) due to memory token caching versus FlashAttention-2?
2. I think it would be helpful to discuss: what are the different methods (e.g., StreamVGGT, CUT3R) good at and where do they fail? How is this linked to their method (architecture, training strategy, training data, etc.)?

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
5

### Summary
This paper proposes StreamVGGT, a causal transformer architecture designed for real-time streaming 3D visual geometry reconstruction. Inspired by the success of large language models with causal attention, the authors integrate temporal attention and a cached token memory mechanism to enable efficient, incremental 3D scene updates without needing to reprocess full sequences. Extensive experiments on multiple 3D reconstruction benchmarks demonstrate that Stream3R achieves competitive performance.

### Strengths
1. This demonstrates that using cache token technology can significantly improve VGGT's inference performance with long video inputs.
2. The technical details are clear and easy to follow.

### Weaknesses
1. The authors’ demo video showcases the model’s capability to handle scenes with significant motion. It would be helpful if the paper included more analysis or discussion regarding these dynamic scenarios.
2. I don't quite follow the motivation behind using knowledge distillation (KD). First, why not directly train the model on real-world data? Is it because the dataset size and quality are insufficient compared to VGGT, leading to suboptimal results? Second, since VGGT cannot be trained on long video sequences due to the computational cost of attention, wouldn't this limitation of the teacher model affect StreamVGGT’s performance on long sequences—especially for tasks like pose estimation?
3. Figure 2 showcases the time efficiency advantage over the vanilla VGGT. However, since the authors emphasize the streaming nature of their approach, it would be more convincing if they could also compare the overall reconstruction time with other similar methods, such as CUT3R.

### Questions
1. Since there are already many similar works (e.g., Stream3R, Lan et al.) that also adopt caching techniques, it would be beneficial if the authors could include a comparison with these approaches in the paper.
2. Adding a pose estimation visualization and a comparison with CUT3R would make the paper more complete.

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
This paper introduces StreamVGGT, a transformer architecture for streaming 3D visual geometry reconstruction. The model draws inspiration from autoregressive language models and replaces global self-attention with temporal causal attention, enabling incremental 3D reconstruction with cached token memory. StreamVGGT supports online inference while maintaining spatial and temporal consistency. The approach achieves comparable reconstruction and depth estimation performance to offline methods (e.g., VGGT) while substantially improving inference speed on long sequences.

### Strengths
* The paper addresses a practical problem: enabling low-latency 3D reconstruction suitable for real-time streaming systems.
* The temporal causal attention and cached memory are conceptually simple, computationally efficient, and align well with online mapping needs.
* The experimental evaluation is extensive, covering multiple datasets and comparing against both dense view and streaming baselines.
* The writing is clear, structured, and easy to follow.
* The paper demonstrates significant latency reduction (e.g., 10× faster inference for long sequences) while maintaining similar accuracy compared to VGGT.

### Weaknesses
* The issue of error accumulation over long sequences is not thoroughly analyzed. No explicit mechanisms (e.g., drift correction) are introduced beyond distillation, and the paper does not present accuracy trends across different sequence lengths compared to VGGT.
* The scalability and memory-growth behavior of cached tokens for very long video sequences remains unclear.
* The connection to causal or autoregressive modeling is somewhat superficial: there is no explicit probabilistic formulation or next-frame prediction, only causal masking.
* The paper does not explore qualitative failure cases, such as dynamic occlusions or fast ego-motion, which are critical for practical deployment.

### Questions
* How does the model handle error accumulation or drift over long sequences (e.g., hundreds or thousands of frames)?
* Does the cached token memory ever saturate or require pruning? If so, how is this managed without degrading accuracy?
* Can the causal model generalize beyond static datasets (e.g., driving, dynamic human motion)?
* How robust is StreamVGGT to missing or noisy frames, which are common in real streaming data?

### Soundness
3

### Presentation
4

### Contribution
3
