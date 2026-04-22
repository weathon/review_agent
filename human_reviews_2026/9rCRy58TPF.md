# SPICE: Submodular Penalized Information–Conflict Selection for Efficient Large Language Model Training

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Information-based data selection for instruction tuning is compelling: maximizing the log-determinant of the Fisher information yields a monotone submodular objective, enabling greedy algorithms to achieve a $(1-1/e)$ approximation under a cardinality budget. In practice, however, we identify alleviating gradient conflicts, misalignment between per-sample gradients, is a key factor that slows down the decay of marginal log-determinant information gains, thereby preventing significant loss of information. We formalize this via an $\varepsilon$-decomposition that quantifies the deviation from ideal submodularity as a function of conflict statistics, yielding data-dependent approximation factors that tighten as conflicts diminish. Guided by this analysis, we propose SPICE, a conflict-aware selector that maximizes information while penalizing misalignment, and that supports early stopping and proxy models for efficiency. Empirically, SPICE selects subsets with higher log-determinant information than original criteria, and these informational gains translate into performance improvements: across 8 benchmarks with LLaMA2-7B and Qwen2-7B, SPICE uses only 10% of the data, yet matches or exceeds 6 methods including full-data tuning. This achieves performance improvements with substantially lower training cost.
Code is available at https://github.com/Chang-pw/SPICE#.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem of selecting a small subset of instruction-tuning data for large language models in order to fine-tune efficiently. The authors observe that while maximizing the log-determinant of the empirical Fisher information matrix yields a submodular objective, in practice marginal gains collapse quickly due to gradient misalignment among samples. They formalize this by decomposing the marginal information gain into a base term and an interaction term, and conclude that controlling gradient conflict is key to sustaining information gain. Based on this, they propose SPICE: a selection algorithm that (1) uses a scoring function that subtracts a conflict penalty from the marginal information gain, (2) optionally stops early once the marginal gain falls below a threshold, and (3) uses a proxy (smaller) model to compute the gradients efficiently. They empirically show that on multiple benchmarks, using only ~10% of data, SPICE matches or exceeds full-data fine-tuning and outperforms several baselines, while reducing computation cost.

### Strengths
The method addresses both effectiveness (maintaining or improving performance with fewer training samples) and efficiency (using proxy model selection & early stopping) — a nice combination.

Empirical results are broad (multiple benchmarks, models, tasks) and show impressive savings (≈10% data) with no performance loss and even gains in some cases.

The algorithm is extensible: the idea of “penalize conflict” could be applied in other data-selection or multi-task contexts.

### Weaknesses
The current scope of experiments is limited to ~7 B-parameter models and instruction-tuning; extension to larger models (>30 B), multimodal tasks, or RLHF settings remains to be seen.

The proxy-to-target model transfer is shown only within same architecture family; cross-architecture transfer (e.g., completely different model family) may degrade and is less explored.

The cost comparison, while present, could be strengthened with more granular breakdowns (selection cost vs fine-tune cost) across all baselines under identical hardware settings.

The penalty on “conflict” implicitly biases toward samples aligned with current gradient direction—there is a risk that samples with contradictory but important signals might be under-selected; more analysis of diversity vs conflict trade-offs would help.

### Questions
How sensitive is SPICE to the choice of proxy model? If the proxy model differs in architecture or domain from the fine-tune target, how does performance vary?

In domains with heterogeneous instruction types (e.g., chat, coding, planning) where gradient directions may naturally differ, how does the conflict penalty trade off between “reducing harm” vs “reducing diversity”? Have you analysed domain-coverage of the selected subset?

Could you provide more detailed hardware/GPU-hour breakdowns (selection + fine-tune) for each baseline method (e.g., LESS, SelectIT, FisherSFT) under identical hardware, to strengthen the cost-efficiency claim?

Have you tested SPICE on larger models (>30 B) or other modalities (vision+language) or RLHF settings? If not yet, what do you foresee as the main challenge in scaling?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a submodular framework for data-efficient language model fine-tuning, introducing a conflict-aware selection mechanism that balances information gain and gradient disagreement. The method improves greedy subset selection efficiency and achieves competitive results.

### Strengths
1. Novel insight into the fast decay of marginal contribution to enhance greedy submodular optimization.
2. Comprehensive experiments and ablations supporting the method’s effectiveness.
3. Decent performance on various benchmarks.

### Weaknesses
1. High computational cost due to gradient retrieval for each selection step.
2. Possible misalignment between the theoretical motivation and empirical design; the overall selection pipeline remains somewhat unclear (see questions).
3. Limited baseline comparisons (see questions).

### Questions
1. *Theorem 1 (rows 167–169):* Why do large perturbations lead to faster decay? Shouldn’t it be the difference between successive perturbations, not the absolute magnitude of a perturbation, that drives faster decay?
2. *Definition 4:* Corollary 1 penalizes both similar and opposite gradients via squared inner products, while Definition 4 only penalizes opposite ones. Why are similar gradients (redundancy) ignored, given that the theory penalizes both?
3. *Pipeline clarity:*
    - Section 4.3 (row 346): When stating “at each iteration, we select one sample using our conflict-aware greedy algorithm,” does ‘one sample’ refer to a single example or a mini-batch of k samples? If it refers to a single example, does the model get updated after each selection (get updated after seeing a new example) when T=1?
    - Section 5 (row 372): How is the 120-sample candidate pool formed? Is it randomly drawn from D with size k×T?
    - Is the proxy model updated after each cycle?
4. *Baselines and related work:*
    - If the proxy model is periodically updated and selection occurs within a randomly sampled “candidate pool”, the setup seems closer to online batch selection, making comparisons to FisherSFT, LESS, or IFD, a non-periodic selection mechanism, potentially unfair. It remains unclear whether SPICE’s performance gains stem from the periodic schedule or the proposed selection mechanism.
    - Representation-based selection methods [1] and other recent instruction-tuning data selection works [2-4] are not discussed.
5. *Complexity analysis:* Could the authors provide an explicit asymptotic analysis of time complexity? Algorithm 1 appears to require gradient computation over the entire dataset D for each selection, which seems computationally expensive.

[1] Ivison, H., Zhang, M., Brahman, F., Koh, P. W., & Dasigi, P. (2025). *Large-Scale Data Selection for Instruction Tuning*. arXiv preprint arXiv:2503.01807.

[2] Liu, Z., Karbasi, A., & Rekatsinas, T. (2024). *TSDS: Data Selection for Task-Specific Model Finetuning*. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)*.

[3] Wang, J., Lin, X., Qiao, R., Koh, P. W., Foo, C.-S., & Low, B. K. H. (2025). *NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric*. In *Forty-second International Conference on Machine Learning (ICML 2025)*.

[4] Chen, Y., Li, Y., Hu, K., Ma, Z., Ye, H., & Chen, K. (2025). *MIG: Automatic Data Selection for Instruction Tuning by Maximizing Information Gain in Semantic Space*. In *Findings of the Association for Computational Linguistics: ACL 2025*.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies selecting instruction-tuning datasets by proposing to avoid gradient conflicts. The authors develop an epsilon-decomposition that splits the Fisher gain into a baseline and a perturbation term and showed that the perturbation is upper bounded by squared gradient inner products. The authors proposed SPICE, a greedy data selection algorithm that scores a candidate example by the marginal gain as well as a gradient penalty term. Empirically, the authors demonstrated that at 10% data matches or outperforms full-data SFT and several baselines while reducing selection/training cost.

### Strengths
- This paper is very well written - clear and flows well from theory to an a practical algorithm inspired by the theory. The experiments seemed pretty complete as well.
- The proposed SPICE selection algorithm is simple and intuitive as well
- The experiments compared several baselines on multiple benchmarks. The gain is pretty consistent.

### Weaknesses
- It would be interesting if the authors can demonstrate whether the finding can be extended to larger corpus / base LMs as behaviour might change as we scale up the model size.
- It would be nice if the authors could provide us with qualitative examples to better understand what constitute examples that has low/high gradient conflict. Is there some intuition as to what they might imply to the data.

### Questions
n/a

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
The paper proposes SPICE, a conflict-aware data selection method for instruction tuning. It starts from the submodular log-det(Fisher) objective, shows that marginal information gains decay faster when gradient conflicts are high, and formalizes this using an ε-decomposition → curvature analysis. SPICE scores each sample by (Fisher marginal) − λ·conflict (conflict = negative cosine to the running mean gradient), supports early stopping, and allows proxy models for efficiency. On LLaMA2-7B and Qwen2-7B, using ~10% of data, SPICE matches or exceeds full-data and several selectors (LESS, Fisher, SelectIT, IFD) across 8 benchmarks.

### Strengths
Clear theory–practice link: ε-decomposition → curvature explains greedy degradation under gradient conflicts.

Simple, practical selector: Fisher-marginal − λ·conflict with early stopping and proxy models; easy to drop into pipelines.

Solid empirical sweep: two bases (Qwen2-7B, LLaMA2-7B), 8 benchmarks, cost/ablation studies; strong gains on IFEval/MMLU at ~10% data.

### Weaknesses
Assumption fragility: bounds rely on α‖F‖<1 and AdaFisher approximations; reported violation rates in higher-conflict regimes weaken guarantees.

Limited baselines: several strong recent selectors and tiny-data LoRA baselines are missing; Random is competitive in places.

Conflict proxy is heuristic (−cosine to mean gradient); sensitivity to optimizer/state/batch is underexplored, and cross-architecture transfer is weak.

### Questions
Benchmark coverage. Please add stronger baselines, e.g., the method in arXiv:2402.02318 and a LoRA-only tiny-data baseline (e.g., 0.5–2% data) to show SPICE’s advantage at very small budgets.

Tiny-data behavior. How does SPICE compare to straightforward LoRA with small training sets across tasks? Any regime where plain LoRA beats SPICE-selected 5–10%?

### Soundness
3

### Presentation
3

### Contribution
3
