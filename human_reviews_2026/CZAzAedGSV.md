# SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
As large language models (LLMs) continue to scale up, their performance on various downstream tasks has significantly improved. However, evaluating their capabilities has become increasingly expensive, as performing inference on a large number of benchmark samples incurs high computational costs.
In this paper, we revisit the model-item performance matrix and show that it exhibits sparsity, that representative items can be selected as anchors, and that the task of efficient benchmarking can be formulated as a sparse optimization problem. 
Based on these insights, we propose SparseEval, a method that, for the first time, adopts gradient descent to optimize anchor weights and employs an iterative refinement strategy for anchor selection. We utilize the representation capacity of MLP to handle sparse optimization and propose the Anchor Importance Score and Candidate Importance Score to evaluate the value of each item for task-aware refinement. Extensive experiments demonstrate the low estimation error and high Kendall’s $\tau$ of our method across a variety of benchmarks, showcasing its superior robustness and practicality in real-world scenarios. Code is available at https://github.com/taolinzhang/SparseEval.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a framework for faster LLM evaluations that formalizes anchor selection and weighting as an optimization problem. The anchor weights can be learned via an MLP using gradient descent, and the anchors are iteratively updated via candidate importance score and anchor importance score. Across six benchmarks, the SparseEval achieves very low error in terms of MAE and high rank fidelity in terms of Kendall's $\tau$ with very few items (roughly 100) and a large set of models (roughly 5000) from the Open-LLM leaderboard.

### Strengths
1. Experiments are comprehensive, results are strong. The study aggregates per-item accuracy matrices over ~5,000 models and evaluates on six widely used datasets, expanding the model coverage far beyond prior efficient-evaluation work (e.g., TinyBenchmarks’ ~300 models). The scaling is nice to see. On this large setting, SparseEval consistently beats baselines (Anchor Points, gp-IRT, TailoredBench) across anchor budgets (20–100). For example, at 100 anchors the method reaches MAE below 1% with $\tau>0.90$ on multiple datasets; the paper highlights that competing methods frequently exceed 2% MAE in the same regime. The error-trend plots (e.g., ARC) show a clear and stable gap over baselines as anchors increase, while the ablation contrasting linear weighting vs MLP underscores that deeper architectures materially improve sparse estimation—precisely the regime SparseEval targets. The data-efficiency ablation is notable: even with 10% training data, the method maintains ≈1% MAE on HellaSwag/MMLU, supporting claims of robustness under scarce supervision.

2. The CIS/AIS refinement loop is intuitive and simple: at its core, the algorithm is greedy by swapping out the anchor with the smallest AIS and in the candidate with the largest CIS. CIS selects candidates whose response patterns align with residual structure; AIS quantifies an anchor’s gradient contribution to error reduction.

### Weaknesses
1. Experiments only report MAE and Kendall's $\tau$. Many modern evals use calibration metrics, long-form judgments, or pairwise win-rates. A brief experiment or discussion on extending the loss/aggregator to non-decomposable metrics (e.g., Brier/F1, win-rates) would broaden applicability.

2. Robustness of anchors transfer to newer set of models. The study is comprehensive in-distribution over a very large model set. It would be useful to show that anchors selected on one model cohort transfer to a different cohort or to newer models (e.g., hold-out families), even if with modest degradation.

3. Interpretability of learned weights: The work nicely shows negative weights expand the search space relative to cluster averages; a short qualitative analysis of which items receive high magnitude weights (e.g., skill categories in MMLU) would provide insight for benchmark curation.

### Questions
Typos:

* Line 180, $S'=S\odot ({\bf 1}_m^W\top)$ -> $S'=S\odot ({\bf 1}_mW^\top)$.

* Line 185, standard notation for $\ell_1$ norm is $\\|\cdot\\|_1$ rather than $\|\cdot\|_1$.

* Inconsistency on the use of $k$ and k for $k$-means.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper frames large-model benchmarking as a sparse optimization problem: select a small set of anchor items from the full benchmark and learn anchor weights via gradient descent to approximate full-benchmark scores. To improve representativeness, the authors propose iterative anchor refinement using Anchor Importance Score (AIS) and Candidate Importance Score (CIS) to quantify item value; they also use an MLP as a nonlinear aggregator to realize sparse reconstruction. Experiments reportedly achieve low estimation error and high Kendall’s τ ranking consistency across multiple benchmarks, demonstrating substantial cost savings without changing conclusions.

### Strengths
The paper offers a clear modeling perspective by framing evaluation sparsity through the model by item score matrix and performing sparse selection with learned weights directly aligned to overall scores and rankings. It introduces task aware anchor optimization via AIS and CIS to iteratively refine anchors beyond one shot selection, and employs a nonlinear MLP aggregator to capture inter item structure and model family differences. Empirically, it foregrounds practical gains, reporting low error and high Kendall's tau across benchmarks, consistent with the aim of reducing cost without altering rankings. Methodologically, it departs from IRT and heuristic clustering and external assessor approaches by explicitly learning anchor weights to reconstruct aggregate benchmark scores.

### Weaknesses
1. Do your learned anchors (and the associated aggregation/weights) directly generalize to unseen models? In other words, can we train anchors once and apply them to new model releases without retraining?
2. What is the end-to-end cost of discovering/learning the anchor set (including any proxy-model runs, scoring, and training)? If this cost is substantial, and the resulting anchors/weights only work well for a single model, then the approach may not be worthwhile. Please quantify the cost and clarify the cross-model reusability.
3. I’d like to see visualizations of the item/question space that justify the claimed sparsity (e.g., clustering/affinity structures). Which questions are similar, and why? Please provide empirical evidence (e.g., similarity heatmaps, spectral clustering plots, cluster exemplars) to explain why certain items are considered redundant or representative.

### Questions
Same as weakness

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
This paper introduces SparseEval, a novel method to efficiently evaluate large language models by framing the problem as sparse optimization. The method uses a proxy MLP model to learn optimized weights for a small subset of "anchor" test items via gradient descent. A task-aware refinement strategy iteratively improves the anchor set by swapping items based on new "Anchor Importance" and "Candidate Importance" scores. Experiments demonstrate that SparseEval accurately estimates model performance with significantly reduced computational cost, outperforming previous methods in estimation error and ranking correlation.

### Strengths
* This paper demonstrates the sparsity in the dataset, which enables the prediction of model performance using a small amount of anchor data.

* The paper proposes SparseEval, which trains an MLP to make predictions based on the performance of existing models on both anchor data and the full dataset.

* Experimental results show that this method can accurately select models that are representative of the entire dataset.

### Weaknesses
* Given a new dataset, this method may require testing many models and performing training, which can be costly to train in practice.
* The approach lacks generalization to stronger models and other architectures. Under the current training setup, it remains unclear whether the architecture can generalize to more powerful models or to new architectures such as linear attention or MoE models.
* The method lacks evaluation of its adaptability under long-chain-of-thought (long-CoT) conditions, such as on benchmarks like AIME or GPQA. Under long-CoT settings, model outputs tend to fluctuate significantly, so it is uncertain whether this method would still be effective.
* Training the MLP requires a relatively large amount of data. When only a few existing models are available, the performance may be suboptimal.

### Questions
* Could you provide generalization tests of the proposed method, including evaluations on stronger models and models with different architectures?
* Could you provide an analysis of the method’s adaptability on Long-CoT data?
* Could you further explain the sparsity assumption? From Figure 3.1, it appears that the similarity between different clusters of data is also quite high.
* When testing different training checkpoints of the same model, the model’s performance tends to fluctuate — can this method accurately predict the precise performance in such cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SparseEval, a sparse-optimization framework for efficient evaluation of LLMs. It formalizes benchmark sparsity by modeling the model–item score matrix and selecting a small set of representative “anchors.” The method optimizes anchor weights via gradient descent with an MLP predictor and refines anchors using AIS and CIS. Experiments on six LLM benchmarks show that SparseEval achieves <2% MAE and τ > 0.9 with only 100 samples, outperforming prior IRT- and clustering-based approaches.

### Strengths
Casting efficient LLM evaluation as a sparse optimization problem is conceptually clean and original.

The gradient-based anchor refinement (AIS/CIS) is simple, intuitive, and empirically strong.

Results across multiple datasets and ablations convincingly show superior efficiency and robustness.

The approach can substantially reduce evaluation cost in large-scale LLM benchmarking.

### Weaknesses
The proposed method relies on access to a large model–item performance matrix for training and anchor refinement. Its effectiveness when only a small number of model evaluations are available, or when evaluating a completely new task with limited historical data, has not been examined. This restricts its practical usability in real-world cold-start settings.

While the paper frames efficient evaluation as a sparse optimization problem, the theoretical analysis remains shallow. The notion of sparsity is mainly supported by empirical observations rather than formal proofs or quantitative measures, and there is no theoretical guarantee relating the number of anchors to estimation error.

Typos:
Line 410: demostrating → demonstrating 
Line 485: filed → field

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3
