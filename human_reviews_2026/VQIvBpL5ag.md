# Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR incurs only a 1.4 perplexity degradation on Llama2-7B to enable aggressive W4A4KV4 quantization with 50% sparsity, delivering up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The key idea is to model the effect of weight perturbations due to pruning + quantization via a second-order Hessian objective, approximate it tractably (row-wise decoupling, approximate Hessian via Fisher × identity factorization, then derive a closed‐form compensation term that “restores” performance by shifting information from more error‐sensitive weights to more robust ones.

### Strengths
Model compression for LLMs is a hot area and bridging quantization + pruning is of high practical relevance

### Weaknesses
1 The general idea of combining quantization + pruning is not new (the authors themselves reference earlier works on small networks like DJPQ (Wang et al., 2020) and OBQ (Frantar & Alistarh, 2022) for the joint problem). 

2 The Hessian‐based analysis for pruning quantization has precedent (e.g., OBD/OBQ for pruning, Fisher/Hessian analyses, quantization with perturbation bounds). The authors are extending to LLM scale, but one might argue that the core conceptual novelty is limited: the compensation term is derived, but the big leap is engineering (applying it to huge models) rather than algorithmic novelty.

3 Some sections feel dense and hard to follow

### Questions
1 For pruning, you adopt the mask from WANDA as default. Have you tested OBR using other pruning algorithms

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Optimal Brain Restoration (OBR), a general training-free framework for jointly achieving quantization and sparsification of Large Language Models (LLMs). Its core motivation stems from the fundamental conflict between quantization and pruning: quantization prefers compact weight distributions (to suppress outliers) while pruning favors high-variance distributions (to identify important weights). OBR reconciles these conflicting objectives by introducing a Hessian-based error compensation mechanism between pruning and quantization, which "transfers" the information lost due to pruning or quantization distortion to more robust weights. This method transforms the original second-order optimization problem into a closed-form solution via row-wise decoupling and group error compensation, making it efficient and interpretable. Experiments show that OBR achieves LLM compression of W4A4KV4 + 50% sparsity, incurring only a 1.4 perplexity increase on Llama2-7B while delivering 4.72× inference speedup and 6.4× memory reduction.

### Strengths
1.Applying Classical Hessian Optimization to Modern LLM Joint Compression Scenarios: Closed-Form Solution Design Balances Efficiency and Performance

2.OBR functions as a post-processing module, compatible with various pruning methods, quantizers, and rotation schemes.

3.This work systematically extends the classic Optimal Brain Damage/Surgeon (OBD/OBQ) theory, generalizing it from single-task scenarios to joint compression settings.

4.This work benchmarked the latency, FLOPs, and TOPS of INT4 2:4 sparse GEMM on the A100 GPU and implemented a custom CUTLASS kernel.

### Weaknesses
1.The calibration phase requires solving a system of linear equations for each row, resulting in significant time overhead during calibration.

2.The partitioning ratio α in the quantization compensation phase lacks theoretical grounding. Although validated through ablation studies, the optimal α may vary across layers and models.

3.OBR’s calibration data sensitivity is insufficiently analyzed—it is only evaluated within general-domain text corpora (WikiText2 and C4) and lacks cross-domain robustness experiments, making it difficult to guarantee the stability of its compensation mechanism in real-world, domain-specific deployment scenarios.

4.The inter-row decoupling hypothesis, due to ignoring the correlations between channels, may lead to underestimation of error propagation and accumulation of non-orthogonal errors in LLM compression.

### Questions
1.Table 6 demonstrates that employing different α values across various models can lead to superior performance. Is it feasible to design an adaptive α strategy?

2.OBR’s compensation relies on the calibration dataset; if the calibration data distribution differs significantly from that of the target downstream task (e.g., calibrating on general-domain text but deploying on mathematical reasoning), could the compensation amplify errors rather than mitigate them?

3.Is the row-wise decoupling assumption in Section 4.1 of the paper reasonable? Could the authors provide an ablation study comparing the error accumulation under the decoupling assumption versus partially coupled alternatives?

4.The paper compares OBR against baselines such as SparseGPT+GPTQ, but does not include comparisons with other joint compression methods like JSQ. Under the W4A4KV4 + 50% sparsity setting, is OBR’s advantage over JSQ and similar methods truly significant?

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
This paper proposes Optimal Brain Restoration (OBR), which reconciles the conflict between pruning and quantization via a second-order approximation and a closed-form group error compensation scheme: errors introduced on pruned/quantized weights are projected—through Hessian sub-blocks—onto the retained weights to minimize accuracy loss under joint compression. On models such as Llama-2/3 and Qwen-2.5, OBR achieves better zero-shot accuracy and WikiText-2 perplexity than strong baselines (e.g., SparseGPT+GPTQ and naïve “quantization+pruning”) under the W4A4KV4 + 50% sparsity setting. For efficiency, the authors implement an INT4 + 2:4 sparse GEMM using NVIDIA CUTLASS and report kernel-level latency/throughput curves (e.g., a 5.9× speedup over dense FP16 and 1.4× over dense INT4 at sequence length 4096).

### Strengths
1. Clear and well-justified motivation. The paper precisely frames the problem: quantization and pruning each approach their limits, and naïvely combining them causes mutual interference, motivating a method that explicitly resolves this conflict. It also provides empirical evidence that non-trivial sparsity naturally emerges after quantization, supporting the coexistence and complementarity of the two representations and thus the direction of joint compression.

2. Sound theoretical development. Starting from a second-order (Hessian-based) objective, the paper introduces group error compensation, derives a closed-form optimum, and explains the Hessian’s bridging role in transferring errors—offering good interpretability.

3. Methodological generality. The proposed approach can be combined with popular sparsity/quantization techniques such as SparseGPT, OBR, QuaRot, SpinQuant, and FlatQuant, making it a broadly applicable compression method.

4. Custom kernel implementation. The authors implement dedicated GEMM CUDA kernels with NVIDIA CUTLASS (INT4 with 2:4 sparsity), which strengthens practical usability and reflects substantial engineering effort.

5. Comprehensive setup and strong results. Experiments span multiple sizes of Llama-2/3 and Qwen-2.5, evaluating zero-shot benchmarks and WikiText-2 perplexity, with comparisons against quantization-only and strong prune+quantize baselines (e.g., SparseGPT+GPTQ). The method shows high speedups and substantial memory reduction.

### Weaknesses
1. Lack of end-to-end inference metrics. The paper does not report TTFT (time-to-first-token), tokens/s, or throughput under concurrency. Although kernel-level gains are provided, their translation to full-pipeline performance is unclear.


2. Narrow evaluation tasks. The experiments could be broadened to include more task types—such as code generation or long-context workloads—to better demonstrate generality.

### Questions
same as the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper propose Optimal Brain Restoration (OBR), combining the sparse pruning and quantization with error compensation for model compression. Experiments show that OBR incurs only a 1.4 perplexity degradation on Llama2-7B to enable aggressive W4A4KV4 quantization with 50% sparsity, delivering up to 4.72× speedup and 6.4× memory reduction compared to the FP16-dense baseline.

### Strengths
A training free method with the combination of pruning and quantization is achieved, by the intervene between pruning and quantization by computing an optimal compensation that minimizes the impact of compression on downstream tasks. 

The experiments show SOTA results, indicating the first algorithm to enable W4A4KV4+50% sparsity LLMs, without requiring any additional retraining.

### Weaknesses
The sparse pruning can be used based on 4:2 sparsity, the hardware implementation is efficient, but the accuracy drop of W4A4KV4+50% sparsity LLMs is larger than 5%, which hinders the application of this method. The experimental results are based on simple QA tasks, and lack of long context tasks and long-cot tasks, so the results are not convincing.

Also the algorithm should verify the results on the new data formats such as NVFP4 or MXFP4, which are popular on the modern hardware.

### Questions
1. Maybe the training aware method by combining the quantization and pruning is more practical, especially for the use of W4A4 + 4:2 sparsity.
2. What's dominant application of 4:2 sparsity? Since few examples of 4:2 sparsity are released in the industry.

### Soundness
3

### Presentation
3

### Contribution
3
