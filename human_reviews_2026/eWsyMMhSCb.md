# PrunedLoRA: Robust Gradient-Based Structured Pruning for Low-rank Adaptation in Fine-tuning

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Low-rank adaptation (LoRA) has become a widely used paradigm for parameter-efficient fine-tuning of large language models, yet its representational capacity often lags behind full fine-tuning. Within the context of LoRA, a key open question is how to obtain expressive low-rank adapters from over-parameterized spaces. We propose PrunedLoRA, a new framework that leverages structured pruning to obtain highly representative low-rank adapters from an over-parameterized initialization. Unlike prior approaches that impose a fixed low-rank budget, PrunedLoRA dynamically prunes less important components during fine-tuning and prevents their reactivation, enabling flexible and adaptive rank allocation. For structured pruning, by minimizing the pruning error for overall loss, we provide fine-grained pruning and recovery updates in a gradient-based pruning strategy with grounded interpretation. We provide the first theoretical analysis of the robustness of structured pruning and provably show that under the impact of weight perturbation, gradient-based pruning is more robust than activation-based pruning with respect to overall loss. Empirically, PrunedLoRA consistently outperforms LoRA and its variants across supervised fine-tuning tasks in mathematical reasoning, code generation, and natural language understanding, and it also demonstrates advantages over existing structured pruning methods across diverse sparsity levels.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PrunedLoRA, a method that starts from a high-rank LoRA adapter and progressively prunes it using gradient- and Hessian-based criteria to achieve a compact final rank. The goal is to leverage higher expressivity during training while maintaining LoRA’s efficiency at inference. Experiments on math, code, and GLUE benchmarks show consistent gains over LoRA, DoRA, and AdaLoRA and other structured pruning methods, especially when starting from large initial ranks.

### Strengths
- The major idea in the paper is intuitive and clearly explained.


- The method is sound and evaluated on multiple tasks.


- The paper is clearly written and easy to follow.

### Weaknesses
`Peak memory will be significantly higher than LoRA, with no measurements reported.`

The peak memory usage in the paper will be significantly higher than pure LoRA (and other baselines at the base rank 64). For instance, when PrunedLoRA starts with rank 512, the memory required for parameters, gradients, and optimizer states (all of which occupy a large portion of memory) will increase roughly 8 times. The paper completely neglects mentioning any memory comparisons, which is very important in this case, since peak memory usage is a key metric for parameter-efficient fine-tuning.


`Missing comparisons to strong high-expressivity PEFT baselines.`

The paper does not compare against strong and relevant baselines that focus on high-expressivity and high effective rank PEFT methods, such as HiRA (https://openreview.net/forum?id=TwJrTz9cRS), MoRA (https://arxiv.org/abs/2405.12130), and ABBA (https://arxiv.org/abs/2505.14238). These methods are designed to achieve similar goals while maintaining efficiency, and excluding them makes it difficult to assess how much improvement actually comes from the proposed approach.


`Misaligned focus on inference efficiency instead of training-time memory.`

The authors mention that they optimize for the final inference efficiency of LoRA. This seems misplaced, PEFT methods generally optimize for training-time memory budgets, not for the small differences in storage caused by lower-rank adapters. Disk space is far cheaper than compute memory, and in many cases, adapters are even merged during inference, making storage savings largely irrelevant in practice.


`Can you compare against pure high-rank LoRA?`

It would be very helpful if the authors showed performance comparisons against pure LoRA trained at higher ranks. This would help characterize how close the pruning method comes to pure high-rank LoRA, which should typically serve as the upper-bound performance baseline (“skyline”).

### Questions
Please refer to weaknesses.

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
4

### Summary
The paper proposes the PrunedLoRA framework, which starts from a high-rank (over-parameterized) LoRA initialization and dynamically prunes it during fine-tuning using a novel, gradient-based structured pruning strategy . The main contributions include: (1) The PrunedLoRA framework; (2) A theoretical analysis demonstrating that gradient-based pruning is more robust to weight perturbations than activation-based pruning; (3) An interpretation of a second-order pruning metric ("saliency"). Experimental results show that PrunedLoRA (especially with a high initial rank, e.g., r=512) matches or exceeds the performance of FFT on math, code, and NLU tasks, and outperforms standard LoRA and other pruning baselines.

### Strengths
1. Strong Empirical Performance: On key benchmarks, PrunedLoRA (with r=512) matches or even exceeds the performance of Full Fine-Tuning, while significantly outperforming standard LoRA and other pruning baselines (Table 1, 3).
2. Novel Theoretical Analysis: The paper is the first to theoretically demonstrate (Sec 3.2, Prop. 1 & 2) that the proposed gradient-based pruning method has stronger robustness (error is independent of module magnitude) under weight perturbations compared to activation-based methods.

### Weaknesses
1. Insufficient Justification for Hessian Approximation: Approximating the Hessian in Eq. 5 is a very strong assumption, but the paper addresses it with only a single sentence, without discussing its impact on the "optimality" of the derivation.
2. Gap Between Theory and Practice: The theoretical advantages of robustness claimed in Proposition 1 & 2 (e.g., "magnitude dependency") are not directly verified experimentally, weakening the link to the empirical performance gains.
3. Missing Computational Cost Analysis: The paper fails to clearly isolate the cost of the pruning algorithm itself (Table 2) and ignores the significant O(r^3) cost from the high initial rank r in its complexity analysis. The pruning schedule for the best-performing models is also missing.

### Questions
1. Your theoretical analysis (Prop. 1 & 2) presents a novel claim that gradient-based pruning is more robust than activation-based pruning. However, the paper lacks direct empirical validation of this specific claim. Could you provide a more convincing argument or (if feasible) preliminary experimental evidence to demonstrate that this specific theoretical robustness, and not just the general concept of "pruning from over-parameterization," is what drives PrunedLoRA's superior performance over baselines like SparseGPT?
2. You report in Table 2 that PrunedLoRA (r=512) takes ~53 minutes longer than standard LoRA. However, it is unclear how much of this 53-minute overhead comes from  the inherent cost of performing forward/backward passes at a higher rank (r=512) versus the pruning steps themselves. Could you provide a more detailed cost breakdown, specifically stating the total wall-clock time (or percentage of total time) consumed by the pruning algorithm (Alg. 1) itself during the r=512 run?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
PrunedLoRA initializes LoRA with a high rank, then applies gradient/Hessian-guided structured pruning (column/row) with a closed-form compensation step to mitigate pruning-induced loss increase. The paper also provides theory suggesting gradient-based pruning is more loss-robust than activation-based criteria, and it reports consistent gains over LoRA/DoRA/AdaLoRA and structured-pruning baselines on GSM8K, HumanEval, and a GLUE subset.

### Strengths
1. **Clear and practical recipe**. High-rank initialization followed by structured pruning with second-order (gradient–Hessian) signals and closed-form updates is well-motivated and easy to implement conceptually.
2. **Consistent empirical improvements**. Across GSM8K, HumanEval, and GLUE (subset), PrunedLoRA narrows the gap to full FT and outperforms strong PEFT and pruning baselines under various sparsity/rank regimes

### Weaknesses
1. Motivation not fully compelling. The observation that "higher rank helps does not by itself necessitate pruning inside LoRA. Additionally, what concrete advantages does this scheme offer over alternatives? For instance, why not adopt AdaLoRA’s rank allocation, or directly learn a structurally sparse $\Delta W$? The paper should articulate a clearer conceptual and empirical rationale relative to these design points.

2. Narrow rank regime. Experiments primarily use relatively large ranks ($r\in[64,256]$ for initialization and prune to 64), whereas practical deployments often push to very small ranks ($r=1-8$). The original LoRA paper reports competitive results at $r=1-8$ on some tasks; please include systematic results and stability curves for $r\in\{1,2,4,8\}$. (Figures do explore small targets in ablations, but a consolidated, head-to-head comparison versus baselines at these tiny ranks would strengthen the case.) 

3. Training cost vs. inference savings trade-off is under-argued.
The claimed inference-time storage benefit comes at the price of higher training overhead (start high-rank + second-order mask search + compensation). In many practical settings, training memory/compute, not inference storage, is the binding constraint.

4. Other experimental settings. See Questions  below.

### Questions
1. **Learning-rate search range**. In the appendix you state: "we perform grid search over $\{1e-5, 5e-6, 1e-6 \}$" This range is extremely low for LoRA; in practice, many LoRA setups peak around $[5e-5, 2e-4]$ (See [1]). Using such low LRs raises fairness concerns, especially because PrunedLoRA may benefit from larger effective updates early on due to higher initial rank. Please expand the grid to include standard LoRA ranges and report the best.

2. **Scaling factor $\frac{\alpha}{r}$**. LoRA updates are $W=W_0 +\frac{\alpha}{r} BA$. You mention grid search over learning rates and scaling factors for all methods, but I didn't find the description about scaling factors in Appendix C.1.

3. Is sequence length 128 enough for math or code tasks? And why you use a such small training batch size? Usually the value would be 32 in [1][2].


[1] Zhang, Y., Liu, F., & Chen, Y. (2025). LoRA-One: One-Step Full Gradient Could Suffice for Fine-Tuning Large Language Models, Provably and Efficiently. arXiv preprint arXiv:2502.01235.

[2] Wang, S., Yu, L., & Li, J. (2024). Lora-ga: Low-rank adaptation with gradient approximation. Advances in Neural Information Processing Systems, 37, 54905-54931.

### Soundness
2

### Presentation
3

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
In this paper, the author proposed PrunedLoRA, a gradient-based structured pruning framework for obtaining efficient low-rank adapters from over-parameterized spaces. And they provide a theoretically grounded strategy for structured adapter compression.
Experiments over NLU and NLG tasks validate PrunedLoRA‘s effectiveness.

### Strengths
-	The paper is well-written and the figures are clear.
-	The gradient-based pruning theory is well-established and interesting.

### Weaknesses
-	My primary concern is the reliability of the experimental results. I noticed significant differences in the experimental setup compared to prior works [1,2,3]. The authors should clarify why their setup and results diverge so substantially.
    - For instance, the sequence length T=128 is unusually short—typically, T=1024 is used.
    -	The results in Table 1 also appear much weaker than those in [1], where the rank was only 8, far lower than the pre-pruning rank of 128 and post-pruning rank of 64 in Table 1.
    -	Work [1] trained for only 3,000 steps, whereas this study trained for 5,000 steps. Logically, the same method should yield significantly better results than [1].
-	Additional ablation studies are needed:
    -	MT-Bench results should be included in Table 1.
    -	Comparisons with newer baselines like LoRA-GA, LoRA-Pro, AltLoRA, and LoRA+ are missing—the current baselines are outdated.
    -	Need to include the results of Full Fine-Tuning using Algorithm 2.
    -	Table 2 should also report GPU memory usage during training.


[1] AltLoRA: Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections. In ArXiv 25.

[2] LoRA-Pro: Are Low-Rank Adapters Properly Optimized?. In ICLR 25.

[3] LoRA-GA: Low-Rank Adaptation with Gradient Approximation. In NeurIPS 24.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
