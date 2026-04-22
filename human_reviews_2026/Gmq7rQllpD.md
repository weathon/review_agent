# OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Post-training model pruning is a promising solution, yet it faces a trade-off: simple heuristics that zero weights are fast but degrade accuracy, while principled joint optimization methods recover accuracy but are computationally infeasible at modern scale. One-shot methods such as SparseGPT offer a practical trade-off in optimality by applying efficient, approximate heuristic weight updates. To close this gap, we introduce OPTIMA, a practical one-shot post-training pruning method that balances accuracy and scalability. OPTIMA casts layer-wise weight reconstruction after mask selection as independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian. Solving these QPs yields the per-row globally optimal update with respect to the reconstruction objective given the estimated Hessian. The shared-Hessian structure makes the problem highly amenable to batching on accelerators. We implement an accelerator-friendly QP solver that accumulates one Hessian per layer and solves many small QPs in parallel, enabling one-shot post-training pruning at scale on a single accelerator without fine-tuning. OPTIMA integrates with existing mask selectors and consistently improves zero-shot performance across multiple LLM families and sparsity regimes, yielding up to 2.53% absolute accuracy improvement. On an NVIDIA H100, OPTIMA prunes a 8B-parameter transformer end-to-end in 40 hours with 60GB peak memory. Together, these results set a new state-of-the-art accuracy-efficiency trade-offs for one-shot post-training pruning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper focuses on layer-wise pruning of LLMs. Instead of using a Hessian-type sensitivity metric to adjust the weights, the authors actually solve the QPs associated with the layer-wise problem, through some clever reductions that turn the constrained QPs into an unconstrained ones. They then provide a fast solver (based on Restarted Accelerated Primal-Dual Hybrid Gradient) and an efficient GPU/TPU implementation. The method shows some performance gains when combined with mask selection from other LLM pruning methods (WANDA, SparseGPT and others).

### Strengths
1. The paper is well-written. It is nicely focused yet not missing details in the key parts of the methodology section.
2. The QP reductions are sound and it is overall a nice principled way to solve the problem.
3. The experiments presented are quite comprehensive in terms of multiple architectures, lots of metrics, which makes the comparison with existing methods very clear.

### Weaknesses
1. My main concern with the paper is a lack of empirical contribution. Despite the claims in the introduction, the benefits of OPTIMA in e.g. Table 1 are relatively incremental compared to vanilla SparseGPT/Wanda.
2. Furthermore, the method takes 40 hours to run on LLama-8b. While the run-time of the pruning algorithm is typically not the main concern for pruning (since you prune once, and do inference forever), given the lack of empirical gains I think practitioners would simply use SparseGPT. SparseGPT can prune a 7B model (in my experience) in less than an hour.
3. In spite of this, I quite like the methodology of the paper, but wonder if it can be improved. I think the choice to only do weight optimization (as opposed to mask selection) may have been a limiting one. I would be curious to see if it's possible to incorporate mask selection into the formulation. For example in Eq (9) of the paper, one can consider doing greedy forward/backward selection with low-rank updates to the objective (for example, see Algorithm 2 of https://arxiv.org/pdf/1806.03756). My suspicion is that SparseGPT already does quite a good job at weight optimization, and that solving the QPs exactly (on the fixed mask) isn't actually adding much, while mask optimization may help.

### Questions
1. I am curious why the reductions in layer-wise error in Figure 2 do not translate into stronger accuracy gains. I suppose the reductions here are also relatively modest (~10-15% for SparseGPT), but I would be interested to hear the authors' opinion.
2. Another thing to consider as a means to improve the empirical contribution: SparseGPT is typically not scalable to very large LLMs (say 32B+). Hence, WANDA is often used as an alternative. However, WANDA typically performs quite badly on its own (i.e. without any weight adjustment). It does seem that OPTIMA improves WANDA quite a lot. If there is a way to reduce the computation time of your procedure (e.g. by using a stopping tolerance, or larger step sizes, ...) or if it otherwise scales well to larger LLMs, then I could see WANDA+OPTIMA being empirically useful. Based on the computation time results in the paper, this doesn't seem to be the case, but I would be happy to be convinced otherwise.
As I said, I am quite positive overall on the paper and approach, but I would need to see some empirical benefit to the procedure to be satisfied in accepting the paper.

### Soundness
4

### Presentation
4

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
This paper proposes a weight correction method for post-training pruning that relies on quadratic programming. The presented method, OPTIMA, takes in a sparsity mask and precomputed layerwise activation error Hessian and finds the weight that minimizes the immediate activation error. The authors show that OPTIMA improves upon the default weight update methods in Wanda, Sparse GPT, ProxSparse, and Thanos across some (but not all) metrics when evaluated on common open source LLMs.

### Strengths
- The method seems to generally improve downstream task accuracy across both Llama and Gemma models.
- The method is constructed to be tractable and outperforms gradient descent methods in terms of local error minimization.

### Weaknesses
- Equation 1 is the wrong equation to use when characterizing compression problems. The goal of compression methods (both quantization and pruning) is to minimize the end-to-end error, not the immediate activation error. The immediate activation error is only used when directly considering the end-to-end error is intractable. The effect of this is pretty clear in the empirical evaluations in this paper. Figure 2 shows that OPTIMA generally does a better job of minimizing the immediate activation error than the selected baselines, but OPTIMA actually hurts perplexity in many/most cases, especially in the 2:4 case, which is what people actually care about in practice due to hardware support.
- One benefit of Adam and other first order optimizers is that you can directly minimize the end to end error of the entire model by training all remaining parameters with a fixed sparsity mask. This paper doesn't evaluate that baseline, even if it might cost more than OPTIMA. My understanding is that PTP methods generally perform far worse than just doing some full model finetuning with a fixed mask, which makes it hard to justify doing PTP. In contrast, PTQ (quantization) methods are usually not much worse than QAT (and sometimes even better), and QAT is much harder to do than training a model with a fixed sparsity mask, so it is reasonable for PTQ methods to not compare to QAT.
- Equation 1 is written in a way that suggests OPTIMA is jointly finding an optimal mask and weight assignment, but at least from my reading of the paper, OPTIMA only finds the weight assignment and relies on a precomputed mask. Can the authors clarify if this is correct, and if so, what effect the precomputed mask has on the effectiveness of OPTIMA?

### Questions
See above

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
3

### Summary
The paper proposes OPTIMA, a post-mask reconstruction step for pruned LLM layers that turns reconstruction into many small per-layer QPs sharing a single Hessian $H=X^TX$. This shared-Hessian setup lets the authors batch thousands of solves efficiently and plug the method after common maskers (Wanda, SparseGPT, Thanos) for both unstructured and 2:4 sparsity. Across several LLaMA/Gemma models on zero-shot tasks, they report consistent accuracy gains (up to ~2.5 points) at the same sparsity with practical single-GPU runtimes; contributions are the shared-Hessian formulation, the accelerator-friendly solver pipeline, and empirical improvements.

### Strengths
* One shared per-layer $H=X^TX$ turns reconstruction into **uniform QPs**, which batch naturally and saturate GPU/TPU throughput, which is a great optimization fit.
* Practical plug-in: works after common maskers, pruning pipeline doesn't need to be changed.
* Evaluated multiple models and sparsity settings, with generally consistent lifts over the underlying mask baseline.
* Reports wall time and memory, and tries to be usable, not just theory.

### Weaknesses
* Hessian inconsistency: Sometimes reads like they use   (outputs) instead of  (inputs). Must be consistent and show the exact activation capture point.
* Notation/shape errors: Loss decomposes by output columns, not rows. Text/algorithms say "row-wise". This is more than cosmetic and risks correctness.
* Runtime under-specified: One headline number, no mean±std, no per-layer breakdown, or solver iteration counts, and unclear dependence on calibration dataset.
* Robustness/variance weak: No seeds, some outlier perplexities not contextualized, including variances would better explain such outliers, and strenghten the claim about the accuracy gains.
* Thin ablations: No clear sensitivity to calibration size(evaluation accuracy), solver tolerances/iterations, QP batch size, or which subsystems are pruned (attention vs. MLP) at matched global sparsity.
* Figure 2 anomaly unaddressed: MLP-Down and Attn-O show error-ratio ≈ 1 under update methods, which suggests skipped updates or ill-conditioning, or the need for larger calibration set. These are not explained well.
* Algorithm numbering is messy: Numbering resets in the middle, and Algorithm steps don’t match the symbols used in the text. This reads like it missed a final editorial pass.

### Questions
* Row vs. column: Your text repeatedly says the reconstruction is row-wise, but the objective $||X\Delta W||^2_F$ decomposes by output columns. Is the method truly row-wise, or is this a typo and it should actually be column-wise? Please correct shapes and notations.
* Hessian source: Did you compute  from pre-matmul activations? If not, why is  appropriate for the stated objective?
* Figure 2 behavior: Why do MLP-Down and Attn-O have error-ratio ≈ 1 under SparseGPT/Thanos? Were updates skipped (max-iter/tol)? Is too small?
* Runtime variance: Were runtimes measured on a single calibration shard or multiple? Please report mean±std per different calibration tokens for QP solve.
* Baseline overheads: What are the wall-times for SparseGPT/Thanos on the same hardware & token budget? Is mask selection time reused for OPTIMA or counted again?
* Averaging/seeds: Is the reported “average” a macro-average across tasks? How many seeds per model/task? Any confidence intervals?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces a post-pruning weight update method that refines the remaining unpruned weights of a pruned LLM.

### Strengths
- The methodology is presented clearly, and it’s nice to see that the authors also consider a practical implementation on actual hardware.

- Setting aside the fact that the method does not propose a way to find the pruning mask, the idea of further improving a pruned LLM itself seems reasonable.

### Weaknesses
- OPTIMA is not an algorithm for finding pruning masks, but rather one for updating the unpruned weights after a pruning mask has already been determined by some means. Therefore, referring to it as “one-shot pruning for LLMs” does not seem appropriate; it is more accurately described as a post-pruning weight update algorithm. While it certainly contributes to a one-shot pruning pipeline, I believe the essence of pruning lies in determining the sparse structure itself.

- A related work to mention would be Kwon et al. (2022), where (a) mask search and rearrangement is followed by (b) mask tuning. Since the latter essentially corresponds to a constrained optimization of the unpruned weights, (a) can be viewed as “Mask Selection”, and (b) as “Weight Update.” It would therefore be interesting to evaluate how well the proposed OPTIMA performs within such a two-stage pruning pipeline, where the sparse structure is first determined and then the pruned model is post-processed.

- Boža (2024) also considers “a scenario where one selects the pruning mask first and then updates the weights,” which directly corresponds to the setting assumed by OPTIMA. However, the current manuscript lacks any discussion or comparative analysis related to this line of work.

- As currently presented, the results do not appear to support the claim that “OPTIMA consistently improves the accuracy of the models across different tasks.” In particular, the most crucial case from a model compression perspective would be the 8B model, i.e., the largest model among those evaluated in this work. However, for this setting, there seems to be little reason to introduce OPTIMA on top of SparseGPT or Thanos, given the additional computational cost it incurs.

---
- Kwon et al. (2022), A fast post-training pruning framework for transformers.
- Boža (2024), Fast and effective weight update for pruned large language models.

### Questions
- In the case of SparseGPT (and others as well) w/ OPTIMA, does OPTIMA perform an additional update after the weights have already been updated once by SparseGPT? What would happen if we instead used only the sparse structure provided by SparseGPT and applied OPTIMA directly to the original weights?

- How much additional wall-clock time (relative to the base pruning methods) does OPTIMA introduce when applied on top of Wanda, SparseGPT, and Thanos? This is quite important, as the proposed method essentially serves as a post-processing step for refining a pruned LLM; its value ultimately depends on how much improvement it can achieve for the additional computational cost incurred.

### Soundness
2

### Presentation
3

### Contribution
2
