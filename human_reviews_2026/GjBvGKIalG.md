# EAGLE: Efficient Analytical Gradient LinearEvaluation for Enhanced Recomputation in Large Language Models

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Training large language models requires substantial memory to store intermediate activations, often exceeding the capacity of modern accelerators. Gradient checkpointing addresses this challenge by trading computation for memory, but introduces significant overhead due to forward passes during recomputation. In this work, we present EAGLE (Efficient Analytical Gradient Linear Evaluation), a recomputation strategy that leverages closed-form gradients for linear transformations and integrates FlashAttention's backward algorithm for attention blocks. Unlike traditional checkpointing that uniformly replays the forward pass with autograd, EAGLE computes gradients analytically for linear layers and calls FlashAttention's backward directly, while keeping standard recomputation for nonlinear operations. On production-scale models including DeepSeek-V2, DeepSeek-V3, and LLaMA3-70B (70B--694B parameters), EAGLE improves Model FLOPs Utilization by 18--33\% over Full Recompute and achieves up to $9.75\times$ module-level recomputation speedups, and our analysis and experiments show that these gains are achieved without changing training convergence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents EAGLE, a recomputation strategy that reduces the overhead of gradient checkpointing in large language model training. It computes analytical gradients for linear layers and integrates with FlashAttention’s backward pass to avoid redundant forward computations. Experiments on models such as LLaMA3-70B and DeepSeek-V3 show up to 9.7× module-level speedup and 18–33% improvement in model FLOPs utilization while keeping memory usage unchanged.

### Strengths
The paper introduces a recomputation method that uses analytical gradients for linear layers and integrates with FlashAttention’s backward pass, reducing redundant computation and improving efficiency while keeping memory usage constant.

### Weaknesses
The proposed approach offers only a small conceptual improvement over existing gradient checkpointing, mainly replacing autograd recomputation with analytical formulas.

The experimental evaluation is too limited, covering few configurations and lacking analysis of convergence, training stability, or scalability.

Comparisons with recent optimized methods such as CheckMate and Adacc are missing, so the claimed efficiency gains are not well contextualized.

The paper focuses heavily on large-model benchmarks without deeper investigation into when and why the method performs best.

### Questions
The experiments seem limited in scope. Can the authors provide more details on how many runs were conducted and whether the reported improvements are statistically consistent?

How does the proposed method affect convergence behavior or final model quality compared to standard checkpointing?

Why were methods such as CheckMate or Adacc not included in the comparison? Would the claimed gains still hold under those baselines?

The paper focuses heavily on large-model benchmarks without deeper investigation into when and why the method performs best？

### Soundness
2

### Presentation
2

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
The paper proposes EAGLE, a recomputation strategy for training Transformers that replaces autograd-based forward recomputation in linear layers with analytical gradients, and invokes FlashAttention’s backward to avoid the attention forward during checkpointed backprop. For a linear layer $y=Wx$, the method uses $\\nabla_W=\\nabla_y x^\\top$ and $\\nabla_x=W^\\top\\nabla_y$ inside the recompute region, skipping one matmul per region.  EAGLE reports FLOP reductions such as RMSNorm+Linear $8bshh_1 \\to 6bshh_1$ and MLP $24bshh_1 \\to 22bshh_1$, plus attention speedups via direct FlashAttention backward.   Empirically, module-level speedups range from $1.46\\times$ to $9.75\\times$, and end-to-end MFU gains are $18.18%$ to $33.33%$ with fixed $R$ and $7.69%$ to $20.00%$ with optimized $R$, on LLaMA3-70B, DeepSeek-V2, and DeepSeek-V3 up to 694B parameters.

### Strengths
* Clear derivation and insertion point of analytical gradients within a recompute block. 
* Integration with FlashAttention backward to remove attention forward recomputation while preserving memory. 
* Explicit FLOP accounting with parameter dependence and consistent module-level speedups.  
* End-to-end MFU gains across diverse architectures and parallelization regimes.

### Weaknesses
* No numerical gradient-check or stability analysis to support "identical gradient accuracy".
* Missing system throughput metrics such as tokens per second; only MFU and duration are reported.
* Limited sensitivity analysis beyond fixed vs optimized $R$; no study of sequence length or batch size effects in microbenchmarks.
* Scope of end-to-end evaluation excludes mid-scale models; all results are 70B to 694B.

### Questions
* On one configuration per model, report tokens per second (tokens/s) and peak memory alongside MFU and iteration duration, and specify the hardware (e.g., A100 80GB) and precision.
* Provide a short sensitivity sweep over sequence length, for example $s\\in\\{4\\mathrm{k},16\\mathrm{k},32\\mathrm{k}\\}$, showing MFU, iteration duration, and speedup. Use existing codepaths; no retraining needed.
* Could you run a minimal gradient check comparing EAGLE to standard Autograd? On a toy 2-layer MLP and a single self-attention block, run one forward-backward step with identical weights and RNG, dropout off, deterministic kernels, under both BF16 and FP32. Let $g_T^{\\text{EAGLE}}=\\partial\\mathcal{L}/\\partial T$ and $g_T^{\\text{AutoFP32}}=\\partial\\mathcal{L}/\\partial T$ computed by Autograd FP32. For each parameter tensor $T$ (flattened, with $n_T$ elements), please report:

  * (i) $\\|g_T^{\\text{EAGLE}}-g_T^{\\text{AutoFP32}}\\|_{\\infty}$,
  * (ii) $\\tfrac{1}{n_T}\\sum_i \\bigl|g_{T,i}^{\\text{EAGLE}}-g_{T,i}^{\\text{AutoFP32}}\\bigr|$,
  * (iii) $\\tfrac{1}{n_T}\\sum_i \\dfrac{\\bigl|g_{T,i}^{\\text{EAGLE}}-g_{T,i}^{\\text{AutoFP32}}\\bigr|}{\\max\\bigl(|g_{T,i}^{\\text{AutoFP32}}|,\\epsilon\\bigr)}$ with a fixed $\\epsilon$ (e.g., $10^{-10}$). A single step with batch 1 and sequence length around 1024 is sufficient.

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
See below

### Strengths
See below

### Weaknesses
See below

### Questions
The paper introduced a new gradient checkpointing-type algorithm called EAGLE. EAGLE eliminates redundant forward passes by computing gradients directly from cached inputs for linear operations. This method achieves speedup over the existing gradient-checkpointing methods without additional memory consumption.  

The paper is well-written, and the proposed method is clearly presented. According to the literature review in the paper, the proposed idea has not been applied in the existing literature.  The proposed method will be useful for compute-constrained LLM developers. 

To me, this paper starts from a very simple insight and designs a better gradient-accumulation approach. Mathematically, the insight itself is relatively straightforward, but from an engineering perspective, I believe implementing this method and achieving performance improvements requires significant effort. I appreciate the authors' hard work on the execution. 
In summary, I am leaning towards acceptance.

I  have several presentation suggestions: 

1. In Figure 1 and in the summary of the main contribution in Section 1 (~line 100). It would be better to explicitly specify "achieving recomputation speedups" over which baseline method.

2. In Section 1 or maybe in Figure 1, it would be better to explicitly show the memory comparison.

### Soundness
3

### Presentation
3

### Contribution
3
