# TriangleMix: Accelerating Prefilling via Decoding-time Contribution Sparsity

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 6

## Abstract
Large Language Models (LLMs) incur quadratic attention complexity with input length, creating a major time bottleneck in the prefilling stage. Existing acceleration methods largely exploit attention score sparsity by estimating blocks with high attention scores and applying dynamic sparse attention. In this work, we identify another untapped form of sparsity in the prefilling stage, namely decoding-time contribution sparsity, where many attention blocks exhibit nontrivial attention scores during prefilling yet contribute negligibly to subsequent decoding, as indicated by gradient-based analysis. Building on this observation, we propose TriangleMix, a training-free static attention pattern that uses dense attention in a subset of layers and switches to Triangle attention in the others. Extensive experiments show that TriangleMix preserves nearly lossless performance relative to dense attention while substantially reducing attention overhead in Triangle layers. For 128K inputs, Triangle attention achieves a $15.3\times$ speedup in attention computation, significantly exceeding the acceleration of typical dynamic sparse methods ($1.9\times$–$3.4\times$). Furthermore, TriangleMix can be seamlessly combined with dynamic sparsity approaches, delivering an additional 6%–19% reduction in TTFT over using dynamic sparsity alone.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Through rigorous analysis and comparative experiments, this paper demonstrates that the proposed TriangleMix method is both simple and effective. The method achieves acceleration by masking out the central part of the attention matrices in subsequent layers during the pre-filling stage.

### Strengths
1.  The paper is exceptionally well-written: it is clear, focused, and effectively covers crucial logical justifications. For instance, to explain why masking the middle section has minimal impact on decoding predictions, the authors provide a gradient analysis. Furthermore, the separate discussion of kernel versus end-to-end efficiency is commendable, offering a comprehensive view of performance.

2.  The proposed method is well-supported both intuitively and empirically. The intuition that distant historical information can be effectively approximated using an "attention sink"-like map is logical, and this hypothesis is validated by the practical experiments.

### Weaknesses
1.  The proposed method appears overly simplistic. Its novelty is questionable, as similar masking techniques likely exist, even if not applied specifically to pre-filling. Furthermore, its performance is unlikely to substantially diverge from other lambda-shaped attention pre-filling methods (e.g., StreamingLLM), further limiting its innovative contribution.

2.  Despite modifying the model's forward pass computation, the method achieves only a modest TTFT acceleration (up to ~30%). This gain is limited and may not outweigh the potential negative impacts of this approximation. Notably, the method's performance degrades on Qwen2.5, which features a more information-dense KV cache, raising concerns about its generalizability.

### Questions
1.  Can the perplexity scores be validated across the Qwen2.5 models at three scales (3B, 7B, and 14B) to examine how the method's performance scales with increasing model size?

2.  How can the significant performance gap between StreamingMix and TriangleMix on the TREC and Passage Count subsets be explained?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TriangleMix, a method to accelerate the prefilling stage of Large Language Models (LLMs) by leveraging a newly identified form of sparsity called decoding-time contribution sparsity.

### Strengths
1.  **Nearly Lossless Performance:** Preserves model accuracy close to full dense attention on benchmarks like RULER and LongBench.
2.  **High Efficiency:**
    * For 128K inputs, Triangle attention achieves a **15.3x speedup** in attention computation per layer.
    * Reduces **Time-To-First-Token (TTFT)** by **12%-32%** for contexts from 32K to 128K.
3.  **Complementary to Dynamic Methods:** Can be combined with dynamic sparsity methods (MInference, FlexPrefill, XAttention), yielding a further **6%-19% reduction in TTFT**.
4.  **Simplicity:** The static pattern is easier to implement than dynamic methods and avoids the overhead of block index estimation.

### Weaknesses
* Limited Analysis on Longer Generations: The paper's core analysis focuses on the first token generation (TTFT). While TTFT is a critical metric, it does not assess whether skipping Middle Q-K computations leads to narrative drift, factual inconsistencies, or logic errors later in the generation. The impact on the quality of a full, multi-sentence paragraph is not measured.

* Combination Overhead: When combined with dynamic sparsity (e.g., "Ours + MInference"), it's not fully detailed how the two methods are integrated. The computational savings are reported, but the engineering complexity and potential overhead of managing multiple sparse attention schemes simultaneously are not discussed.

* Unexplained Performance Anomalies in Combined Methods: The paper demonstrates that combining TriangleMix with certain dynamic sparsity methods (notably FlexPrefill) leads to significant performance *improvements* over using the dynamic methods alone. While this is framed as a benefit, it is a double-edged sword. This phenomenon inadvertently highlights the instability and inherent weaknesses of the underlying dynamic methods, suggesting they are prone to significant performance degradation on their own. The paper does not adequately investigate the root cause of this synergy, leaving open questions about whether TriangleMix is merely compensating for the failures of other methods rather than solely contributing its own acceleration.

* Lack of Ablation Studies: A significant methodological shortcoming is the absence of crucial ablation experiments. The paper fails to rigorously demonstrate that the specific design choices of TriangleMix are optimal. Key unanswered questions include:
    *   How does the performance compare if layers are selected randomly or based on a different metric, rather than using the proposed gradient-based importance?
    *   What is the individual contribution of the triangle pattern's specific shape (e.g., the size of the "Last Q-K" region)?
    *   Is the performance gain due to the intelligent, structured sparsity of the triangle pattern, or simply a result of reduced FLOPs?
    Without these studies, the necessity and superiority of the proposed gradient-driven layer selection and the Triangle pattern itself remain not fully substantiated.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a simple method that use dense prefill in some layers while using triangle sparse attention in the rest based on some observation of its affection to decoding stage. Results are evaluated on Ruler and Longbench with llama3 and qwen2.5 models.

### Strengths
The design is simple and easy to understand. The results comparison covers most training-free methods.

### Weaknesses
1. The evaluation is not solid with many old models and benchmarks. Ruler score will be high as long as you have some dense layers and lower part of your triangle (doing dense attn in question and first few token generation). Longbench is also too short to reflect the long-context ability. With nowadays so many long-reasoning and agentic tasks, I do not see a reason why using this model-benchmark choices. 
2. The triangle attention speedup is very missleading in the paper as you are only showing the sparse layers instead of the e2e one. This is missleading for reader at a first glance. Without diving into the details, the readers can not know that your method is not called triangle attention but trainglemix. 
3. The related works do not include a category of leanble sparse attention methods such as native sparse attn.
4. Post-training training-free sparse attention methods is not going to be scalable and general. With new architecture tweaks like attention sink bias and gated attention, the triangle pattern may not exist anymore. This type of research is not going to solve the problem fundamentally.

### Questions
1. Can you show any reasoning model/benchmark results? 
2. Is there any insight of this work that will benefit decoding sparse attention? Prefill speedup is less important nowdays compare to long-reasoning time of the models. This is easy to understand as users most only care "time to first answer token" instead of "time to first token". The situation is more profound in agentic tasks.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the prefill bottleneck in long-context LLM inference and identifies a previously unexplored form of sparsity, termed decoding-time contribution sparsity. The paper shows that the Middle Q–K attention region in deeper layers contributes minimally to the first tokens during decoding.

### Strengths
1. well organized: this paper presents a promising idea of thinking the attention sparsity from the whole generation point and proved it as a practical method.

2. well evaluation: for accuracy, it evaluated the RULER/LongBench compared with other SOTA methods; for efficiency, test the triton-based kernel level speedup.

### Weaknesses
1. Is this observation general for reasoning model, like qwen-r1-distilled? Can you test this method on reasoning model and reasoning tasks?

2. It seems similar with streamingllm and other layer-skip [2] work? What is the fundamental difference between yours and others?

3. How do you define which layer is full attention and which is sparse? calibration or just direct half-half?

### Questions
All are in the Weaknesses part.

[1] Efficient Streaming Language Models with Attention Sinks, https://arxiv.org/abs/2309.17453

[2] LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

### Soundness
3

### Presentation
3

### Contribution
3
