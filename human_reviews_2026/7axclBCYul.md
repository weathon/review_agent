# Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
We introduce Qronos---a new post-training quantization algorithm that not only explicitly corrects errors due to both weight and activation quantization, but also corrects errors accumulated from previously quantized layers. Our iterative algorithm is based on an interpretable and disciplined optimization framework that surpasses existing data-driven approaches. At each step, Qronos alternates between error correction and diffusion via optimal update rules. Importantly, we prove that Qronos admits an equivalent formulation that significantly improves algorithmic efficiency; we use our discovery to reduce peak memory usage by 18\times on Llama3 8B, and our scaling analysis shows a speedup of up to  13.8\times for a single-layer microbenchmark. We demonstrate compatibility with existing transformation techniques such as Hadamard-based incoherence processing and weight-activation scaling equalization, among others. We evaluate Qronos using recent language models in the Llama3 and Qwen3 families; Qronos consistently outperforms previous state-of-the-art adaptive rounding methods when quantizing the weights, activations, and/or KV caches to 4 bits or fewer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies post-training quantization (PTQ) problem and presents a method named Qronos for 4bits or lower. The key idea behind Qronos is to alternate between error correction and error diffusion via optimal update rules. Promising experimental results are reported for LLAMA3 and Qwen 3 models.

### Strengths
+ I have found the idea of alternating between error correction and diffusion novel and sensible.
+ The reported experimental results are promising and seem to be among the current SOTA.
+ The appendix contains detailed proofs for the theoretical results reported in Sec. 3.

### Weaknesses
- The literary presentation is still lacking. For example, "GPTAQ has been observed to be unstable in other reproductions." reads weird and the paper's introduction section needs to be revised significantly (too much emphasis on experimental results and little space for explaining the motivation or significance).
- Lemma 3.2 seems like a known result. I understand you provided rigorous proof for this lemma. But I believe that the connection between LS and Cholesky decomposition has been known in the literature. 
- A minor weakness is the notable artifacts left by AI models (the notorious dash). For example, four dashes can be found on page 1 alone.

### Questions
1. Why did you name the proposed method Qronos? What is the purpose of adding "..." in the paper title?
2. Did authors compare Qronos with SmoothRot [1] at 4bit?
3. Is Theorem 3.1, Lemma 3.2, and Corollary 3.3 your own results or borrowed from the literature? If later, reference should be given.
[1] https://arxiv.org/pdf/2506.05413

### Soundness
3

### Presentation
2

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
This paper investigates post-training quantization following a similar approach to the popular OPTQ/GPTQ framework. Specifically, the paper derives an optimization method that accounts for the accumulated errors from previously quantized layers. In addition, it proposes using Cholesky decomposition to accelerate the implementation. The method is evaluated on popular LLMs and combined with well-known quantization schemes, demonstrating improved accuracy in low-precision settings.

### Strengths
1). The paper is reasonably well written.

2). The intuition and analysis are solid, helping to clarify the motivation behind the proposed techniques as well as offering insights into the strengths and weaknesses of the widely used GPTQ method.

3). The experimental evaluation is comprehensive, particularly in combination with state-of-the-art techniques such as rotation and MagR.

4). The results generally show improved accuracy compared to previous methods.

### Weaknesses
1). The overall contribution is incremental.

2). The method introduces additional computational overhead to achieve accuracy gains over GPTQ. Since GPTQ also uses an approximate inverse Hessian to balance accuracy and speed, it is difficult to conclude that this method is definitively better than GPTQ.

3). The improvements are limited for 8B models, and no results are reported for very large models.

4). Some claims—particularly regarding implementation speedup and memory savings—seem misleading. The optimization process takes longer than GPTQ, yet the comparison is placed in the appendix. It would be more informative to present a detailed discussion of these trade-offs in the main paper to help assess the method’s advantages and limitations.

### Questions
1). The paper does not clearly explain how the algorithm is implemented. Could the authors include implementation details and pseudocode?

2). For the W4A4KV4 setting, the weight quantizer uses a linear search. How is this linear search integrated into the proposed optimization process?

### Soundness
3

### Presentation
3

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
This paper introduces Qronos, a new post-training quantization (PTQ) algorithm for large language models that explicitly corrects errors accumulated across layers during quantization. Unlike previous methods such as OPTQ, which assume identical activations before and after quantization, Qronos accounts for the drift that occurs when earlier layers are already quantized. It alternates between two key steps: correcting current quantization errors and diffusing residual errors into future weights to prevent accumulation.
The authors provide a theoretical interpretation linking Qronos to OPTQ and derive an efficient implementation based on Cholesky updates. Extensive experiments are conducted to compare with other PTQ methods.

### Strengths
1. Introduces a clear mismatch-aware formulation addressing activation drift across layers.
2. Provides an efficient implementation for the proposed algorithm. 
3. Shows consistent empirical gains over prior PTQ methods across various models and benchmarks. 
4. Can be easily used with various quantization algorithms and improve their performance. 
5. Offers useful theoretical insights linking and generalizing existing algorithms like OPTQ.

### Weaknesses
1. The proposed method mainly extends OPTQ rather than introducing a fundamentally new algorithmic framework.
2. The improvement on weight–activation quantization appears smaller than that on weight-only quantization; additional analysis would help clarify the underlying reason.
3. The main paper reports results primarily for 3- and 4-bit settings, where existing methods already perform well. Including 2-bit quantization experiments would better demonstrate the robustness of the proposed approach under more challenging conditions.

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
3
