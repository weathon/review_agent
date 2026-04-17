# Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Understanding

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Recurrent large language models (Recurrent LLMs) offer linear computational complexity as efficient alternatives to quadratic self-attention-based LLMs (Self-Attention LLMs). However, Recurrent LLMs underperform on long-context tasks due to limited fixed-size memory. Previous research focused on architectural innovations to enhance memory capacity, but failed to match Self-Attention LLM performance. We argue this limitation stems from processing entire contexts at once being ill-suited for Recurrent LLMs. We propose Smooth Reading, a co-design of recurrent architecture and inference method. It introduces a end-to-end multi-round inference method that processes context incrementally and iteratively summarizes information, reducing memory demands.
Methodologically, we reveal architecture-inference interactions play an important role for performance, efficiency and scalability, shedding light on future Recurrent LLM design.
Besides, our method substantially bridges the performance gap between Recurrent and Self-Attention LLMs on long-context tasks while preserving efficiency advantages.
Smooth Reading boosts SWA-3B-4k from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench, while maintaining 2.5× faster training and 2× faster inference at 64k context.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes smooth reading, a method that combines architecture design and inference optimization, which aims to minimize the memory demands during inference time.

### Strengths
1. The paper is well-written and easy to follow. Readers can easily get the design philosophy behind the proposed methodology.

2. The research question is practical and meaningful. The degradation of LLMs' capabilities under long context remains an unsolved problem, and the multi-turn chat is one scenario that such situation could usually happen.

3. The author conducts extensive experiments on many benchmarks, which show significant improvements and competatives to current self-attention LLMs.

### Weaknesses
1. The idea of context compression or LLM with memory has been extensively explored in prior works (e.g., [1][2][3]). 
Although the authors claim that EMR models are primarily designed for multi-turn chat scenarios, the proposed approach does not exhibit substantial conceptual or methodological differences from existing studies. [1] compresses context into special compression tokens in both training and inference time, [2][3] preserve the most significant information at head level.
As a result, the overall novelty of this work appears limited. The author should compare and explain the differences between these works and the proposed method in detail.

2. The paper sets its motivation in a multi-turn conversational setting; however, this aspect is not clearly reflected in either the formulation or the experimental design. The evaluations and problem setups mainly focus on single-turn instruction-following input–output pairs, while an EMR model is expected to demonstrate its advantage in multi-turn observation and planning. The authors should show their model's performance on such tasks to substantiate their claims.

[1] Zhang, Peitian, et al. "Soaring from 4k to 400k: Extending llm’s context with activation beacon." arXiv preprint arXiv:2401.03462 2.3 (2024): 5.

[2] Li, Yuhong, et al. "Snapkv: Llm knows what you are looking for before generation." Advances in Neural Information Processing Systems 37 (2024): 22947-22970.

[3] Zhang, Zhenyu, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models." Advances in Neural Information Processing Systems 36 (2023): 34661-34710.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper systematically studies the persistent performance gap between Recurrent LLMs and Self-Attention LLMs in long-context understanding. While Recurrent LLMs offer linear-time and constant-memory efficiency, they struggle with processing long inputs due to fixed-size memory constraints. To tackle this issue, the authors propose Smooth Reading, a co-design of architecture and inference method that uses End-to-End Multi-Round (EMR) inference. Instead of processing all tokens in one pass (“One-Round”), the model reads the input in chunks, produces contextual summaries, and updates its hidden state across rounds—effectively address the memory overwhelming issue of Recurrent models while not sacrificing the efficiency.

### Strengths
1. The paper provides a well-organized and systematic study to demonstrate that R-LLMs' performance depends critically on inference design—a neglected dimension in prior works.

2. The End-to-End Multi-Round inference approach is conceptually simple yet powerful, avoiding common pitfalls of non-end-to-end chunking.

3. Evaluation spans multiple long-context benchmarks (LongBench, NIAH, RULER, HELMET), multiple architectures, and comparisons with both Self-Attention and RAG methods.

### Weaknesses
The comparison has only been conducted using standard transformer models, so it would be valuable to evaluate whether the proposed method can also outperform more efficient transformer variants such as Mamba, Jamba, or Hymba.

### Questions
See above

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
3

### Summary
This paper presents an approach to enhance recurrent LLMs ability to preform long-context tasks. Instead of processing the entire input at once, the paper proposes a method called "smooth reading" -- essentially chunking the input and outputting a segment of summaries before outputting the next chunk of text. Experiments shows that two recurrent LLMs of 3B sizes trained in this manner achieve better performance by standard LLMs on long context tasks (LongBench and NIAH).

Overall I find the paper to present an interesting approach to improve recurrent LLMs with comprehensive experiments and analysis.

### Strengths
* The paper presents an innovative method to enhance recurrent LLMs ability to process long-context. The method is intuitive and relatively straightforward. 
* Experiment results demonstrate the effectiveness of the method.

### Weaknesses
* Missing reference: There has been some chunk reading work proposed before for standard LLMs, such as: [MemWalker](https://arxiv.org/pdf/2310.05029), [ReadAgent(ICML 2024)](https://arxiv.org/pdf/2402.09727).

### Questions
* Out-of-domain evaluation: the experiment for out-of-domain evaluation on HELMET (Table 9) is a bit incomplete -- what's the performance for QWEN-2.5-3B-NMR and SWA-3B-4k-OR?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Smooth Reading: an End-to-End Multi-Round (EMR) inference procedure co-designed with recurrent LLMs. Instead of one pass over the full context, the model reads chunks, emits a contextual summary, and updates hidden memory across rounds; this preserves recurrent efficiency while avoiding fixed-memory overload. On LongBench and NIAH, EMR closes or beats (but lacks SSM/RetNet baselines) a self-attention baseline (Qwen-2.5-3B-OR) while keeping linear scaling; at 64k tokens it reports ~2.5× faster training and ~2× faster inference than self-attention, with ~20% overhead vs the recurrent one-round baseline. Methods, datasets, and training/eval protocols are described, with Algorithm 1 outlining EMR.

### Strengths
### Originality

* Frames architecture–inference co-design for recurrent LLMs, advocating EMR over one-round or non-end-to-end multi-round strategies; analyzes complexity/performance trade-offs (Table 1, Eq. (1)).

### Quality

* Provides a concrete algorithm (Algorithm 1), chunking/summary scheme, and datasets (A.1), plus ablations on window/chunk sizes showing co-design effects.

### Clarity

* Clear figures/tables separating OR / NMR / EMR and self-attention / recurrent; centralized setup in §4.1.

### Significance

* Demonstrates gap-closing on LongBench (SWA-3B-4k-EMR avg 50.99 vs Qwen-2.5-3B-OR 47.38) and near-perfect NIAH at up to 256k, while maintaining recurrent efficiency.

### Weaknesses
1. Long-context SOTA families like Mamba/SSMs and RetNet are not benchmarked in Tables 2–3, limiting external validity of the “closes the gap” claim. Add head-to-head results (accuracy and throughput/memory).

2. Seed counts and uncertainty are not prominent in the main text. For stability/extrapolation claims, report ≥5 seeds on key curves with CIs. (If already in appendix, surface in main text.)

3. Realized wall-clock gains can be non-monotonic with skip/round settings; include a FLOPs vs. wall-clock breakdown and gate/round overheads alongside Figure 3, and consider a kernel-optimized path for recurrent EMR.

4. EMR details are spread across §3 and A.1; a compact main-text box with the chunking rules, summary template, and early-stop criteria would improve reproducibility at a glance.

### Questions
1. SOTA recurrent/SSM baselines: Can you add Mamba-7B and a RetNet-style model on LongBench/NIAH/RULER, with matched training protocol and report accuracy, tokens/s, and peak memory?

2. Seed & CIs: For Table 2–3 and Fig. 3, can you re-run with ≥5 seeds and include 95% CIs to calibrate the extrapolation/stability claims?

3. Wall-clock decomposition: Please provide a table with FLOPs saved vs. added (per-round summaries/decoding), and map it to the 2× inference speed at 64k (Fig. 3b).

4. EMR hyperparameters: What default chunk:window ratios do you recommend (Table 4 suggests ~1:2), and how sensitive is accuracy/speed to this ratio across tasks?

### Soundness
3

### Presentation
3

### Contribution
2
