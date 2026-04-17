# MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Large language models (LLMs) are deployed on mobile devices to power killer applications such as intelligent assistants.  LLMs pre-trained on general corpora often hallucinate when handling personalized or unseen queries, leading to incorrect or outdated responses. Knowledge editing addresses this by identifying and adjusting a small crucial portion of model weights, without compromising the general knowledge.  However, prior knowledge editing methods are impractical to run on local devices due to the resource-heavy backpropagation (BP) needed for updates.  We present MobiEdit, the first mobile knowledge editing framework that enables efficient LLM personalization on commercial off-the-shelf (COTS) mobile devices. MobiEdit replaces full-precision BP with quantized forward-only gradient estimation, thus compatible with the energy-efficient mobile neural processing units (NPUs).  To further improve gradient estimation efficiency, we introduce two optimizations: an early stopping mechanism that adaptively terminates editing upon success and prefix activation reusing that reduce redundant computation across steps. Our approach enables real-time editing of 3B-parameter models (Qwen2.5-3B-Instruct and Llama3.2-3B-Instruct) on COTS mobile devices with 7.1$\times$ less memory, 15.8 $\times$ less energy and 3.4$\times$ less latency compared to previous knowledge editing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents MobiEdit, the first practical framework for performing knowledge editing (KE) on large language models directly on commercial off-the-shelf (COTS) mobile devices. The work identifies that existing KE methods, which rely on backpropagation (BP), are fundamentally incompatible with mobile deployment due to (1) high memory/energy costs from storing activations, (2) instability when combined with quantization, and (3) incompatibility with mobile NPUs, which are typically forward-pass-only accelerators. MobiEdit proposed BP-Free Editing, NPU-Friendly Quantization, and System Optimizations to solve the challenges. Experiments on COTS mobile phones (e.g., Redmi K60 Pro) show that MobiEdit can edit 3B LLMs (Qwen2.5, Llama3.2) with 7.1x less memory, 15.8x less energy, and 3.4x less latency than BP-based methods, making on-device personalization feasible for the first time.

### Strengths
The paper tackles a critical bottleneck for the next generation of on-device personal AI: enabling models to learn from user interactions without resorting to the cloud. To our knowledge, this is the first work to demonstrate practical knowledge editing on commercial mobile NPUs. The "Facts edited successfully within 8 hours" metric (Table 3) is a killer demonstration of this practical feasibility, showing MobiEdit can edit 14 facts while ROME can only edit 2, all while avoiding device overheating. A major strength is the rigorous justification for using zeroth-order (ZO) optimization. The paper provides a clear theoretical analysis (Sec 2.3, Eq 4-9) arguing that BP-based gradient noise accumulates multiplicatively with network depth under quantization, while ZO-based noise remains constant.

### Weaknesses
The main results (Fig 5) clearly show a trade-off: MobiEdit has slightly lower "Edit Success," "Portability," and "Locality" than the (infeasible) FP32 ROME baseline. The main text does not fully analyze the source of this drop. Appendix A (Table 5) contains the crucial insight: the "prefix activation reusing" optimization is the main culprit, causing an 8.4-point drop in success, while quantization itself causes only a 2.1-point drop. This is a critical finding and must be discussed in the main paper, as it's the central trade-off of the work (speed vs. accuracy). The claim in Sec 2.4 that reusing stale activations "does not negatively affect the editing outcome" is directly contradicted by Appendix A.

### Questions
The main trade-off in the paper is efficiency vs. quality. Your appendix (Table 5) brilliantly isolates the source of this quality drop (mostly from prefix reusing, not quantization). Could you please move this critical analysis into the main paper (e.g., in Sec 3.2 or 2.4) and discuss this trade-off more explicitly? Why do you think reusing stale prefix activations hurts portability and locality so much?

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
5

### Summary
The paper presents MobiEdit, a framework for knowledge editing in large language models (LLMs) that is optimized for NPUs in mobile devices. MobieEdit works by performing a backpropagation-free, zeroth-order optimization method for factual editing using only forward passes, along with two optimizations to reduce latency and energy consumption. MobiEdit is evaluated on 2 datasets (ZsRE, CounterFact) using 2 3b models (qwen and llama) across 3 mobile devices. Mobiedit achieves up to 15.8× energy savings and 7.1× lower memory usage, with a significant trade-off in edit quality.

### Strengths
The paper is well written and has the following strengths:
- The paper explores the topic of on-device LLM personalization on resource-constrained devices(mobile devices), which is an interesting and relevant topic.
- MobiEdit is optimized towards a new trend of computing hardware(NPUs) via the use of forward-only editing.
- The paper provides a theoretical justification showing that quantization without backpropagation (BP) is inherently more resilient to noise than BP-based quantization.
- Compared to the given baselines, MobiEdit reduces energy consumption and memory usage by a good margin (7x-15x).
- The paper presents an extensive set of experiments, including additional results in the appendix, which serve as solid ablation studies and help clarify the sources of the observed performance gains.

### Weaknesses
While the topic is interesting, there are still a few weaknesses that need to be addressed:
- The novelty of the paper is somewhat incremental, as its primary contributions build heavily on prior work. The concepts of zeroth-order gradient estimation and quantization have been explored previously; however, there is still a degree of novelty in adapting and applying these ideas to NPUs.
- No empirical experiments to support the theoretical claim that MobiEdit quantization is more robust to noise than BP-based quantization.
- The degradation in edit quality is substantial even on relatively simple factual edits. Does the memory/energy gain justify this level of degradation?

### Questions
- Have you tested MobiEdit with different model sizes of the same family? It would be interesting to see the performance gains/accuracy degradations on even smaller models. Additionally, given the memory reduction, could that allow for running larger models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
this paper proposed MobiEdit, an on-device knowledge editing method that replaces backpropagation with forward-only zeroth-order updates in a locate-and-edit scheme, paired with NPU-friendly mixed-precision quantization that keeps only the edit-critical layers in floating point.

### Strengths
1. technically sound, the forward-only zeroth-order editing with NPU-friendly mixed-precision removes backpropagation memory needs and is shown to be more robust under low-bit quantization. prefix-activation reuse and early stopping further cut compute
3. Strong on-device results on commercial phones

### Weaknesses
1. zeroth-order editing needs much more optimization steps to reach similar convergence, without early-stopping and caching, wall-clock time can erase efficiency gains.
2. sensitivity to hyperparameters: performance depends on the number of sampled directions (loss stability varies across 1/3/5 vs 300 directions) and on the early-stopping confidence threshold.

### Questions
N/A

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
3

### Summary
This paper presents MobiEdit, a framework for knowledge editing of large language models directly on commercial mobile devices. The key contribution includes replacing backpropagation-based optimization with forward-only zeroth-order gradient estimation, making it compatible with mobile NPUs, mixed-precision quantization and two system optimizations - prefix activation reusing and early stopping.

### Strengths
S1. Timely work and well motivated design decisions. All the design choices like NPU compatibility, memory efficiency, mixed precision, prefix reuse and early stopping are justified for the  mobile environment.

S2.  Detailed empirical validation and system evaluation. The paper does analysis across different system metrics including energy profiling and thermal pressure.

### Weaknesses
W1. CPU baseline uses llm.c which is not an optimised implementation, as on the github of llm.c itself says it is slightly tweaked version of nanoGPT, which is a learning project. There are many optimised cpu implementation of different llms including llama.cpp [1] and many more, which authors could have used. So it is unclear if the gain is because of unoptimised cpu implementation (llm.c) vs optimised npu implementation or because of algorithm design.

W2. Even though paper's major claim is related to personalised on-device llm, the two benchmarks (ZsRE, CounterFact) used are for factual editing and not personalization-specific.

W3. The theoretical analysis assumes linear networks (f_l(x) = x, equation 5) which seems unrealistic for transformers with LayerNorm, GELU activations, and the claim that zeroth-order variance is "depth-independent" (line 252) contradicts their own equation 5 showing output noise variance $\sigma ^2_L$ grows with depth L.


---

1. https://github.com/ggml-org/llama.cpp

### Questions
Q1. Is it possible to provide result and comparison using the optimised CPU implementation?

Q2/Q3. Same as W2 and W3

Q4. Any plan to release the code.

### Soundness
2

### Presentation
3

### Contribution
3
