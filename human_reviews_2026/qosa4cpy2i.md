# Compression Aware Certified Training

- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Deep neural networks deployed in safety-critical, resource-constrained environments must balance efficiency and robustness. Existing methods treat compression and certified robustness as separate goals, compromising either efficiency or safety. We propose CACTUS (Compression Aware Certified Training Using network Sets), a general framework for unifying these objectives during training. CACTUS models maintain high certified accuracy even when compressed. We apply CACTUS for both pruning and quantization and show that it effectively trains models which can be efficiently compressed while maintaining high accuracy and certifiable robustness. CACTUS achieves state-of-the-art accuracy and certified performance for both pruning and quantization on a variety of datasets and input specifications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CACTUS, a framework for unifying optimization techniques to achieve certified robustness in compressed models. It proposes a CACTUS loss to balance standard accuracy and robustness on a compressed model. While it is the first work to address certified robustness for model compression, its optimization approach is not well-motivated, lacks deep theoretical analysis, and the presentation is not good.

### Strengths
1. This paper is the first work to address certified robustness for model compression.
2. The background section is well-written, providing a clear foundation to understand the problem.

### Weaknesses
1. **Writing and Notation Quality**. The paper suffers from many writing and notation issues that significantly affect readability and clarity.
- Several key symbols are used before being defined, such as $\theta$, $Q(\cdot)$, $\Delta$, and $\eta$.
- The notation convention stated in Section 2 is inconsistent with later usage. For example, the authors claim lowercase bold letters denote vectors, but $\theta$, which may represent network parameters, should arguably be bold ($\boldsymbol{\theta}$) under that rule.
- The definition of $f$ is **ambiguous and inconsistent**. In Section 2.2, $f_k(\mathbf{x})$ denotes the score for class $k$, but later $f_\theta$ refers to a parameterized model. The paper never explicitly defines $f_\theta(\mathbf{x})$, though it is used in Equation 1. It is unclear whether $f$ denotes a function, a variable, or model logits. This confusion propagates through the theoretical and algorithmic sections, making the paper unnecessarily difficult to follow.
- missing “{”, “}” in the experiment section. 

2. **Theoretical Analysis (Theorem 4.1) Is Trivial and Problematic**
- Theorem 4.1 should used  $\Delta^*$ instead of   $\Delta$.
- Theorem 4.1 is almost tautological. Since $\Delta^*$ in Eq. 9 is defined as the optimal perturbation maximizing the loss within the admissible space, it is naturally larger than any feasible perturbation such as that induced by quantization. Hence, the inequality does not provide real theoretical insight.
- The proof in the appendix uses the condition $q_{\text{step}} < 2\eta$, while the main text lists $q_{\text{step}} < \eta$. This inconsistency undermines the claimed theoretical guarantee.

3. **Computational Overhead of CACTUS Loss**
The CACTUS objective (Eq. 8) involves averaging losses over a set of compressed models, $C(f_{\theta})$. In practice, this set can be extremely large because multiple compression configurations can satisfy the same compression ratio. The approach implies training many compressed models simultaneously, resulting in high computational and memory overhead. The paper does not quantify this cost or justify its scalability.

### Questions
1. The paper briefly mentions in the experimental setup that “we set $C(f_{\theta})$ to be the full unpruned network and a network pruned ..." Does this mean that $C(f_{\theta})$ only includes two models (the full and one pruned network)? Theoretically, $C(f_{\theta})$ should contain a large number of compressed models even for one pruning ratio. Could the authors clarify how $C(f_{\theta})$ is actually collected or sampled in practice, and whether the set is fixed or dynamically updated during training?
2. How is certified accuracy computed in experiments? 
3. The paper states that $\lambda$ is gradually increased from 0 to 0.75 during training. What motivates the choice of the upper bound 0.75? Did the authors conduct ablation studies on different $\lambda$ schedules or maximum values?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Introduces CACTUS, a joint training framework that optimizes for certified robustness while being compression-aware across pruning and quantization. Core idea: train over a set of compressed variants per batch, with a curriculum on the robust loss and a differentiable proxy for quantization via Adversarial Weight Perturbation (AWP), supported by a bound linking AWP to quantization error. Strong gains appear primarily under compression; uncompressed models remain comparable to robust baselines.

### Strengths
- Clear problem formulation unifying certified training with compression; objective over a compression set is well motivated.
- Theory for quantization: a clean reduction from quantization to weight-bounded perturbations via AWP with a formal upper-bound guarantee.
- Consistent empirical gains under compression: across pruning and quantization, CACTUS improves certified accuracy versus robust baselines; integration with multiple certified-training losses shows method generality.
- Ablations beyond the core table: AWP radius sweep, compression-set selection strategies, variance across seeds, extreme sparsity, more bit-widths, additional architectures, and datasets.

### Weaknesses
- Scope of headline comparisons: By design, CACTUS is strongest when compressed; for $\delta$=0 or unquantized, SABR typically wins. This is expected but should be emphasized alongside deployment guidance.

- Condition discrepancy: Main text, Theorem 4.1 states $q_{step}\leq \eta$ while Appendix Theorem D.1 states $q_{step}\leq 2\eta$. The bound is fine, yet the precise requirement should be consistently stated.

- Compute cost: Training time overhead is non-trivial; although addressed in Appendix E, a compact cost-vs-benefit summary in the main paper would help.

### Questions
- Uncompressed use case: For δ=0 or no quantization, is there a recommended $\lambda$ schedule or training variant that narrows the remaining gap to SABR, or is the intended operating point strictly under compression
- The appendix covers 0.9–0.99 sparsity and multiple bit-widths. Could the main paper include a compact figure summarizing these extremes to highlight how CACTUS compares against baselines, even under extreme sparsity?
- Could there be more clarity provided on the aforementioned "Condition discrepancy" (Weaknesses)?
- Maybe consider citing [1], as it directly studies robustness effects under compression and would contextualize CACTUS relative to prior compression-robustness efforts for robustness other than adversarial and certified robustness.
- Would it be possible to add something like a "Table of Contents" in the Appendix, before the content starts to clearly list all the additional studies, experiments, and other contents in the Appendix, since including a portion of this information in the "Reproducibility Statement" does not clearly state all the experiments in the Appendix. 

References:

[1] Hoffmann, J., et al. "Towards improving robustness of compressed CNNs." ICML Workshop on Uncertainty and Robustness in Deep Learning (UDL). 2021.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CACTUS (Compression-Aware Certified Training Using Network Sets), a framework designed to jointly train neural networks to be both certifiably robust and compressible through pruning and quantization. The approach integrates a joint objective that combines certified robustness and compression losses across a set of compressed model variants. To approximate the effects of quantization in a differentiable manner, CACTUS employs Adversarial Weight Perturbation (AWP), supported by Theorem 4.1, which formally links AWP bounds to quantization error. Empirical evaluations on MNIST, CIFAR-10, and TinyImageNet demonstrate that CACTUS achieves higher certified accuracy compared to post-hoc compression baselines such as SABR, HYDRA, NRSLoss, and QA-IBP. The authors claim that CACTUS represents the first unified framework that integrates compression and certified robustness within a single training process.

### Strengths
Deploying robust models on resource-limited devices is an important and timely research direction.

The joint training objective is clearly defined and implemented. The use of compression sets and curriculum-based loss weighting is technically reasonable.

Experiments are carefully executed and include ablations (AWP radius, compression-set size). CACTUS consistently outperforms sequential baselines in certified accuracy under compression.

The paper is well written, equations are clean, and implementation details are fully specified.

### Weaknesses
The work overlooks Gui et al. (2019), "Model Compression with Adversarial Robustness: A Unified Optimization Framework", which already introduced a unified optimization framework combining model compression (pruning and quantization) with adversarial training. While ATMC focused on empirical rather than certified robustness, the underlying idea (joint optimization of robustness and compression) is the same. A clearer connection to this prior line of work would strengthen the paper’s positioning and clarify its contribution.

Results are restricted to small and mid-scale datasets (MNIST, CIFAR-10 and TinyImageNet). 

Although the paper briefly reports results for joint pruning and quantization (Table 5), these results are weaker and not deeply analyzed. A deeper discussion on why multi-objective CACTUS performs suboptimally would be valuable.

### Questions
How does CACTUS differ formally from ATMC beyond using certified losses instead of adversarial training losses?

Could the CACTUS framework be applied to text or speech models, where quantization effects differ significantly?

Would combining ATMC’s constrained optimization with certified bounds yield similar or better results?

### Soundness
3

### Presentation
2

### Contribution
2
