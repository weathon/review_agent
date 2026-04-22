# MaskPro: Linear-Space Probabilistic Learning for Strict (N:M)-Sparsity on LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
The rapid scaling of large language models(LLMs) has made inference efficiency a primary bottleneck in the practical deployment. To address this, semi-structured sparsity offers a promising solution by strategically retaining $N$ elements out of every $M$ weights, thereby enabling hardware-friendly acceleration and reduced memory. However, existing (N:M)-compatible approaches typically fall into two categories: rule-based layerwise greedy search, which suffers from considerable errors, and gradient-driven combinatorial learning, which incurs prohibitive training costs. To tackle these challenges, we propose a novel linear-space probabilistic framework named MaskPro, which aims to learn a prior  categorical distribution for every $M$ consecutive weights and subsequently leverages this distribution to generate the (N:M)-sparsity throughout an $N$-way sampling without replacement. Furthermore, to mitigate the training instability induced by the high variance of policy gradients in the super large combinatorial space, we propose a novel update method by introducing a moving average tracker of loss residuals instead of vanilla loss. Finally, we conduct comprehensive theoretical analysis and extensive experiments to validate the superior performance of MaskPro, as well as its excellent scalability in memory efficiency and exceptional robustness to data samples.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes **MaskPro**, a probabilistic framework to learn strict (N:M)-sparsity masks for LLM weights with **linear memory** in the number of parameters. It models each group of M weights with a categorical distribution and generates N active positions via sampling without replacement. It trains logits using a policy-gradient estimator on loss residuals with a moving average baseline to cut variance. A representation theorem rewrites N:M masks as a probabilistic sum of basis vectors; proofs claim unbiased gradients and reduced variance. Experiments on several 7B models show competitive zero-shot accuracy vs rule-based pruning and proximity to MaskLLM at much lower memory and data cost.

### Strengths
- **Memory scalability.** Reduces logits storage from combinatorial $O \left(\binom{M}{N}^{d/M}\right)$ to linear $O(d)$. This is the key lever for practicality.  
- **Clean probabilistic formulation.** N-way sampling without replacement per group with explicit $p(m\mid \pi)$ and REINFORCE-style training.  
- **Variance control.** Loss-residual update with a moving average tracker yields an unbiased estimator with lower variance than vanilla PG under stated conditions. Empirical curves support stability gains.  
- **Robustness to tiny data.** Reasonable performance even with very few calibration samples; plots include the 1-sample case.  
- **Empirical breadth.** Evaluates on multiple 7B models and diverse zero-shot tasks; MaskPro beats rule-based baselines by ~2 points on average and narrows the gap to MaskLLM.  
- **Training efficiency.** Orders-of-magnitude lower memory and dataset size than MaskLLM; single-GPU feasibility reported.

### Weaknesses
- **Novelty vs known tools.** The representation of N:M masks via basis-vector sums and a categorical policy with REINFORCE is conceptually straightforward; the main novelty is the linear-space parameterization plus the residual baseline. The paper should better position against prior PG-based pruning and Gumbel/relaxation methods.  
- **Sampling cost.** The approach still simulates N-way sampling per group at train time. The paper notes this as a time bottleneck but gives limited profiling or asymptotic-to-wall-clock mapping.  
- **Baseline fairness and scope.** Results focus on (2:4) with (4:8) pushed to appendix. No strong hardware-level throughput benchmarks for inference with real N:M kernels. Comparisons use C4 for all methods, but details like per-layer masks and calibration choices could bias outcomes; MaskLLM numbers are partly taken from prior work.  
- **Initialization dependence.** Logit init relies on a good starting mask (Top-N or SparseGPT). The sensitivity to poorer inits and to the magnitude hyperparameter is only partly explored.  
- **Theory conditions.** The variance-reduction claim depends on a condition $f(m_t \odot w,\xi) > \tfrac12 f(m_0\odot w,\xi)$. Practical validity across models and tasks is not deeply tested; switching $m_0$ mid-training is suggested but not ablated.  
- **Model scale.** Only 7B models are reported. No 13B–70B results, where memory and sampling costs and accuracy regressions are more consequential.

### Questions
1. How is $p(m\mid \pi)$ computed efficiently for N-way sampling without replacement? Provide explicit formulas and complexity, not just appendix references. Can you replace sampling with a differentiable top-N estimator while keeping linear memory?  
2. How sensitive is performance to the initial mask $m_0$ and the logit magnitude $C$? Report ablations that start from random $m_0$, weaker heuristics, and different $C$.  
3. Give wall-clock profiles: percent time in sampling, forward passes, and PG updates. Can cached logits or Gumbel-Top-k variants reduce sampling cost further?  
4. Show end-to-end throughput and latency on A100/H100 with vendor N:M kernels before and after MaskPro, at fixed accuracy. Accuracy-only tables are insufficient to claim deployment readiness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces MaskPro, a memory-efficient probabilistic framework for learning strict (N:M)-structured sparsity in large language models (LLMs). The method models each group of M weights using a categorical distribution and generates sparse masks via N-way sampling without replacement, reducing the memory complexity from exponential to linear. It also proposes a new policy gradient estimator (PGE) that replaces the vanilla loss metric with loss residuals and employs a smoothing tracker to stabilize optimization. Experiments on several 7B-scale models show moderate improvements in performance and significant reductions in memory usage compared to prior approaches such as MaskLLM.

### Strengths
- The proposed update rule based on loss residuals, combined with a moving average tracker, is an interesting and practical way to stabilize the inherently noisy gradients in policy-gradient-based mask learning. The authors demonstrate theoretically that this estimator remains unbiased while reducing variance.
- MaskPro achieves substantial memory and computational efficiency, requiring roughly 36 GB of memory compared to over 300 GB for MaskLLM, while running on a single GPU. 
- The method performs well even when trained with very small datasets (sometimes as few as one sample), showcasing robustness to data scarcity.

### Weaknesses
- The experimental evaluation focuses solely on mid-sized (7B) models and only on (2:4) sparsity configurations, with minimal exploration of other ratios or architectures. There is no validation on larger or smaller models, no finetuning experiments, and no runtime or latency measurements to verify practical benefits. As a result, the empirical evidence supporting MaskPro’s claimed generality and scalability remains limited.
- The method's success is critically dependent on a complex initialization strategy. Standard random or zero initializations are "ineffective", and the training fails if the initial logits magnitude $C$ is not large enough.
- The comparisons with baselines such as MaskLLM, SparseGPT, and Wanda are not entirely fair, as these methods differ in their reliance on fine-tuning versus frozen weights, training data sizes, and initialization strategies. Moreover, MaskPro benefits from initialization using precomputed SparseGPT masks, which gives it an advantage that is not acknowledged or ablated. The paper would be stronger if these factors were controlled more carefully.
- Although the authors emphasize MaskPro’s efficiency, the claimed linear-space scaling overlooks constant factors associated with sampling and softmax operations, and the actual runtime improvements are not measured. The claim that the method can train effectively with one sample seems implausible without relying on strong priors, undermining the claim of data robustness. More empirical validation is needed to substantiate these statements.
- The algorithmic description omits critical implementation details such as how N-way sampling without replacement is realized efficiently, how randomness affects reproducibility, and how logits are updated in parallel. The notation (e.g., the $\oplus$ operator) is nonstandard and lacks intuition, while figures are schematic and do not convey architectural structure or ablation results. Overall, the presentation is mathematically heavy and could be made clearer.

### Questions
- Can the authors provide wall-clock runtime and throughput comparisons with MaskLLM and rule-based baselines?
- How sensitive is MaskPro to the smoothing parameter $α$ and the initialization constant $C$?
- What happens when masks are initialized randomly instead of using SparseGPT priors?

### Soundness
1

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
This paper presents MaskPro, a linear-space probabilistic framework for learning strict N:M sparsity in LLMs. The work addresses the limitations of existing approaches, rule-based methods that introduce bias and learning-based ones that demand prohibitive computational cost, by proposing a lightweight probabilistic formulation. It models each group of weights with a simple categorical distribution and learns sparsity patterns through a refined policy gradient update that incorporates loss residuals and a moving average tracker, improving both stability and convergence. Empirically, MaskPro achieves strong results across multiple 7B-scale models, reaching good accuracy while reducing memory usage by an order of magnitude and requiring only a handful of training samples. The approach is conceptually clear, theoretically supported, and practically appealing.

### Strengths
1. The reformulation of the (N:M) sparsity problem into a linear-space probabilistic model is both interesting and convincing. It reduces the memory requirement from exponential to linear scale and lowers the data demand compared with prior learning-based methods.
2. MaskPro demonstrates consistently strong performance across diverse benchmarks and model backbones, achieving comparable or superior accuracy to established baselines while maintaining high efficiency.
3. The approach is remarkably data-efficient, as evidenced in Figure 3, where stable results are obtained even with minimal training samples, making the method highly practical for large-scale applications.

### Weaknesses
1. While the results on 2:4 sparsity are strong, it remains to be seen how well MaskPro generalizes to other configurations, such as 8:16. These settings involve significantly larger combinatorial spaces, roughly 12,870 combinations per group in MaskLLM, and would serve as a valuable test of the proposed probabilistic formulation’s scalability.
2. It would be interesting to explore whether MaskPro can be extended to jointly optimize both the sparsity pattern and the LLM parameters. Such a joint optimization could potentially yield further improvements in performance and adaptability.

### Questions
Please see the weaknesses.

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
3

### Summary
The paper introduces a new probabilistic framework for learning hardware-friendly semi-structured sparsity in LLMs. Unlike prior methods, MaskPro reduces both memory and computation overhead by modeling sparsity as an N-way sampling without replacement from categorical distributions over M consecutive weights. It leverages a linear-space parameterization that reduces memory complexity to linear. and optimizes via a refined PGE that replaces the raw loss with a loss residual tracked by a moving average, improving stability and variance reduction. Extensive experiments on multiple 7B LLMs show that MaskPro outperforms baselines while approaching the their accuracy at 10x less memory and training cost.

### Strengths
Sound theoretical explanation, linear memory efficiency, and comprehensive validation experiments.

### Weaknesses
Experiments focus mainly on 2:4 sparsity and 7B models; scaling to higher sparsity ratios or larger models, e.g., 70B, is not demonstrated. (might due to hardware constraints)

### Questions
How sensitive is MaskPro's performance to the choice of $\alpha$?

Does the sampling-without-replacement process still scale linearly in memory and remain computationally tractable for large M? This needs further discussion.

### Soundness
4

### Presentation
4

### Contribution
4
