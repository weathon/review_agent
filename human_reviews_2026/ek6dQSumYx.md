# The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Neural network pruning is a promising technique to mitigate the excessive computational and memory requirements of large language models (LLMs).
Despite its promise, however, progress in this area has diminished, as conventional methods are seemingly unable to surpass moderate sparsity levels (50-60\%) without severely degrading model accuracy.
This work breaks through the current impasse, presenting a principled and effective method called $ \text{Elsa}$, which achieves extreme sparsity levels of up to 90\% while retaining high model fidelity.
This is done by identifying several limitations in current practice, all of which can be traced back to their reliance on a surrogate objective formulation.
$ \text{Elsa}$ tackles this issue directly and effectively via standard and well-established constrained optimization techniques based on ADMM.
Our extensive experiments across a wide range of models and scales show that $ \text{Elsa}$ achieves substantial improvements over existing methods;
e.g., it achieves 7.8$ \times$ less perplexity than the best existing method on LLaMA-2-7B at 90\% sparsity.
Moreover, we show that $ \text{Elsa}$ remains stable even at extreme sparsity (e.g., 95\%), yielding up to $\times$3.98 inference speedup and $\times$7.80 memory compression over its dense counterpart.
We also present $ \text{Elsa}_ {-L}$, a quantized variant that scales to extremely large models (27B), and establish its theoretical convergence guarantees.
These results highlight meaningful progress in advancing the frontier of LLM sparsity, while promising that significant opportunities for further advancement may remain in directions that have so far attracted limited exploration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the "sparsity wall" (50--60\%) in large language model (LLM) pruning by introducing ELSA (Extreme LLM Sparsity via Surrogate-Free ADMM). ELSA formulates pruning as minimizing the true task loss $f(x)$ under an $\ell_0$ sparsity constraint and solves it via ADMM, which alternates between weight optimization and projection onto the sparse set. Instead of layer-wise reconstruction surrogates, it performs an objective-aware projection using diagonal Fisher/Adam second-moment statistics to guide weighted top-$k$ selection. Experiments on OPT, Gemma-2, and LLaMA-2 models show stable perplexity up to 90\% sparsity and consistent gains over SparseGPT, Wanda, ALPS, L-ADMM, and SAFE. The paper also mentions a quantized extension (ELSA-L) for 27B-scale models with theoretical guarantees, though detailed results appear later in the text.

### Strengths
1. The curvature-weighted projection is simple yet effective, using readily available second-moment statistics to guide pruning decisions.

2. Empirical results are strong and consistent, showing stable perplexity up to 90% sparsity across diverse LLM architectures.

3. The paper is well written and conceptually cohesive. The additional theories round up a solid work.

### Weaknesses
1. The experiments convincingly show perplexity improvements, but omit practical metrics such as wall-clock time, memory footprint, and actual inference acceleration. In reality, pruning methods—whether extremely local (e.g., SparseGPT, Wanda) or global (e.g., this work, SparseLLM and etc)—ultimately represent different points on a performance–cost tradeoff curve. With sufficient GPU resources, even global pruning becomes computationally feasible, reducing the motivation for additional algorithmic complexity like ADMM. It would strengthen the paper to explicitly highlight the practical scenarios or deployment constraints where ELSA provides tangible benefits over simpler baselines.

2. The evaluation focuses solely on perplexity and task accuracy, without measuring actual inference acceleration, FLOPs reduction, or memory throughput gains after pruning. This limits the practical impact, as sparsity alone does not guarantee real-world speed-ups. 

3. While the paper frames pruning as a constrained optimization problem and leverages ADMM elegantly, both the global-pruning perspective and ADMM-style decomposition have precedents (e.g., SparseLLM, L-ADMM, SAFE). The novelty lies mainly in unifying these ideas under a “surrogate-free” formulation rather than introducing a fundamentally new optimization principle.

4. [minor] Despite claiming global sparsity, the projection step is performed per-tensor using diagonal curvature estimates, so cross-layer dependencies are not modeled. The approach therefore remains an efficient approximation to the ideal global objective rather than a full solution.

### Questions
1. How does ELSA translate its high sparsity into real inference acceleration or memory savings on modern hardware?

2. Can you quantify the additional compute cost (training or calibration time) introduced by ADMM compared to simpler methods like SparseGPT?

3. How robust is the whole approach to the hyper-parameters? ADMM-based approaches performance could be quite sensitive to hyper-parameters and hard to tune in practice, especially for large-scale LLM pruning problem IMHO.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ELSA, a surrogate-free ADMM-based method for pruning large language models (LLMs) to extreme sparsity levels (up to 90\%) while preserving performance. It critiques existing layer-wise reconstruction approaches for their limitations and compounding errors. ELSA directly optimizes the sparsity-constrained objective, achieving significantly lower perplexity (e.g., 7.8× better on LLaMA-2-7B at 90\% sparsity) and higher zero-shot accuracy across models like OPT, Gemma, and LLaMA.

### Strengths
1. This paper has a reasonable structure and is clearly written. The authors explain their method in detail and make substantial theoretical contributions.
2. The authors conducted extensive experiments across various model architectures and scales, evaluating metrics including perplexity and zero-shot task performance (e.g., ARC, BoolQ) and performing fair comparisons with state-of-the-art methods (e.g., SparseGPT, ALPS, SAFE). The results show that ELSA exhibits significant performance advantages at high sparsity levels (80-90\%).

### Weaknesses
1. Both ALPS and L-ADMM have also proposed ADMM-based pruning algorithms for LLMs. The authors should clearly elaborate on the differences between ELSA and ALPS to highlight the unique contributions of ELSA.
2. At low sparsity rates (50% and 60%), the accuracy of ELSA is significantly lower than that of the optimal L-ADMM baseline (Tables 7 and 8).
3. Although ELSA has advanced the performance of LLMs at high sparsity rates (70%-90%), there remains a substantial performance gap between ELSA and dense LLMs.
4. Pruning a 7B-parameter model requires 4 A100 GPUs, while pruning 13B and 27B-parameter models requires 4 H200 GPUs. Compared with Wanda and SparseGPT, this constitutes a much higher computational overhead—since pruning a 27B-parameter model using the above two methods only requires at most 1 A100 GPU.

### Questions
1. The authors used such a large number of GPUs to prune LLMs, so why not perform LoRA fine-tuning on the pruned model to obtain a better pruned model? It is suggested that the authors compare the accuracy, computational resources used, and time overhead for obtaining sparse LLMs between ELSA and the "Wanda + LoRA" approach.
2. Can the LLM sparsification technique used in this paper improve the model's inference speed? What advantages does it have compared with quantization techniques?

### Soundness
4

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
4

### Summary
This paper introduces ELSA (Extreme LLM Sparsity via Surrogate-free ADMM), a method to prune large language models to extreme sparsity levels (up to 90\%) while preserving performance. It identifies limitations in existing pruning techniques, which rely on surrogate objectives like layer-wise reconstruction error minimization, leading to performance collapse beyond 50-60\% sparsity due to compounding errors and suboptimality. ELSA directly optimizes a sparsity-constrained problem using ADMM, incorporating objective-aware projections and avoiding surrogates. A quantized variant, ELSA-L, scales to 27B-parameter models with reduced memory. Experiments across models (OPT, Gemma, LLaMA) show ELSA achieves 5-30× lower perplexity and up to 6\% higher zero-shot accuracy at 90\% sparsity compared to baselines. Theoretical convergence is proven, highlighting potential for further LLM efficiency advancements.

### Strengths
1.	This paper is well-written and highly readable.

2.	At high sparsity rates, ELSA achieves significantly better accuracy than other baselines and successfully breaks through the performance limit of LLM sparsification.

3.	The experiments cover models of various parameter scales and different benchmarks, and conduct comprehensive evaluations of the LLMs' performance, including perplexity and zero-shot accuracy.

4.	The theoretical foundation is solid: convergence proofs for ELSA and ELSA-L are provided, based on standard assumptions (such as weak convexity and smoothness). This enhances the reliability of the method and aligns with the empirical results.

### Weaknesses
1.	ELSA's accuracy at low sparsity rates (50% and 60%) is lower than that of the baselines.

2.	It remains unclear how ELSA performs on larger models. Although the parameter scales of the models tested in the experiments range from 125 million to 27 billion, there is a lack of experimental results on even larger models, such as Llama-3-80B.

### Questions
1.	What is the computational efficiency of ELSA? How much time does it take to prune LLMs with different parameter scales?

2.	Can ELSA be used to prune Mixture-of-Experts (MoE) models?

### Soundness
3

### Presentation
3

### Contribution
3
