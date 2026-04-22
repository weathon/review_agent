# A New Class of Benchmarks for Federated Multi-Objective Learning

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Federated Learning allows effective machine learning in distribution without exposing the underlying training data. An emerging direction of research is the combination of Federated Learning with multi-objective methods, capturing the complexities of real-world problems by parameterizing training across multiple metrics of success even where such metrics conflict. The evaluation of novel methods requires suitable benchmarks. In Federated Learning, benchmarks are commonly transferred from centralised settings without modification. In this work, we show that this practice is not always sufficient: in one natural setting, where federated clients have heterogeneous preferences over multiple objectives, the most commonly used class of benchmarks can be solved easily even by baseline algorithms, in apparent contrast to the difficulty of the problem in the non-federated setting. Following this insight, we introduce a different, more challenging class of benchmarking problem, derived from the field of fair machine learning. These benchmarks are adaptable, easy to implement, permit different model architectures and different (numbers of) objectives, include a range of different well-established datasets and do not require special adaptation of the federated algorithm. Finally, we demonstrate the versatility and applicability of the proposed benchmarks in a range of configurations on state-of-the-art algorithms, showing to a range of common Federated Learning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues that common multi-task learning (MTL) benchmarks (e.g., Multi-MNIST) are unsuitable for evaluating Federated Multi-Objective Learning (FMOL), since they lack genuine objective conflict under federation. The authors demonstrate this empirically, claiming that even simple FedAvg outperforms centralized multi-objective baselines. They then propose a new benchmark class based on fairness metrics (Demographic Parity, Equality of Opportunity, Equalized Odds), formulating differentiable approximations via tanh relaxations. Experiments with FedProx, CFL, FedCMOO, and FedPref across several fairness datasets aim to show that these fairness-based tasks better expose conflicting objectives in federated settings.

### Strengths
* highlights mismatch between centralized MOO/MTL and FMOL evaluation.
* Fairness metrics offer genuine conflicts; easy to port across models/objectives.
* Multiple FMOL baselines evaluated; implementation appears feasible to reproduce in principle.
* Both homogeneous and heterogeneous preference settings considered; includes 10- and 50-client studies.

### Weaknesses
Novelty/contribution:
* Incremental: fairness-as-objectives and fair-FL have prior art; no new algorithm or theory.
* No principled benchmark design framework (e.g., standardized partitioning, heterogeneity scales, or difficulty metrics); largely a collection of existing datasets plus a known relaxation.
* Central claim that MTL lacks inherent conflict under federation is not rigorously established (no gradient-conflict measures, task interference analysis, or counterexamples).

Soundness/experimental rigor:
* “FedAvg > centralized MOO” on Multi-MNIST not convincingly controlled: missing exact settings (model capacity, optimizer/epochs, learning rates), random seeds, CIs, and tuning parity; no ablations to isolate causes (architecture sharing, exploration from preference heterogeneity, etc.).
* Hypervolume reported without explicit reference point; fairness inversion (1 = fair) risks confusion; no statistical testing/variability reporting for HV or Pareto sets.
* Sensitivity to client partitioning and class/sensitive-attribute imbalance is acknowledged but not resolved; filtering of “near-perfect fairness” runs is ad hoc without robustness checks.
* Tanh-relaxation constant c and regularization mix (e.g., +0.1·BCE) lack principled selection; no stability/sensitivity results.

Presentation/clarity/grammar:
* Notation inconsistencies (e.g., y-hat placement), and a few typos (“is a is a well-known”); some equations/typesetting for fairness metrics are unclear and could confuse summation indices.
* Figures (especially the Multi-MNIST comparison) lack complete experimental context in captions; axes and normalization choices not fully explained; no error bars or CIs.
* Some references misdated (e.g., ProPublica “Machine Bias” typically 2016); fairness survey dates inconsistent; ensure consistent venue/year formatting.
* Appendices are long on grids/tables but short on interpretive diagnostics (e.g., why DDP harder than EO, when/why algorithms fail).

Scope/coverage:
* Evaluation limited to small tabular fairness datasets; no vision/NLP or larger-scale/federated-realism tasks.
* No alternative multi-objective families beyond fairness (e.g., calibration–accuracy, robustness–accuracy, latency–accuracy).

### Questions
1. How do you directly measure “conflict” in MTL vs FMOL (e.g., gradient cosine similarity across tasks, interference metrics, or conflict indices)?
1. What exact training budgets and seeds were used in Fig. 1; were centralized baselines tuned with the same search space/epochs/optimizers?
1. What is the hypervolume reference point per task/metric; how do results vary with that choice?
1. How is the tanh relaxation parameter c selected; show sensitivity and stability across datasets and objectives.
1. How robust are results to client partition strategies that preserve (label, sensitive) joint distributions; can you report replicated runs with CIs?
1. Can you add a principled difficulty measure (e.g., front curvature, dominated volume gap, gradient-conflict statistics) for the proposed benchmarks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the lack of suitable benchmarks for FMOL. The authors argue that existing multi-task benchmarks (e.g., Multi-MNIST, CelebA) often fail to exhibit genuine conflicts among objectives in federated contexts, as federated averaging can unintentionally eliminate objective trade-offs. To address this gap, the paper introduces a new class of benchmarks based on fairness-aware learning problems, where fairness metrics (e.g., demographic parity, equality of opportunity) serve as conflicting objectives against accuracy.

The authors propose a differentiable relaxation of fairness metrics (using tanh-based smoothing), allowing straightforward incorporation into stochastic optimization. They demonstrate this benchmark framework on several datasets (Adult, Law School, Credit Default), using multiple federated algorithms (FedProx, CFL, FedCMOO, FedPref) under both homogeneous and heterogeneous client preference settings. The experiments show that these fairness-based benchmarks yield nontrivial Pareto trade-offs and are more diagnostically rich than standard multi-task benchmarks.

### Strengths
The paper convincingly demonstrates that standard multi-task datasets (e.g., Multi-MNIST) may not produce meaningful trade-offs under federated aggregation, especially when clients have heterogeneous objectives. The motivating experiment is compelling and identifies an overlooked problem in current FMOL evaluation practices.
The paper provides nuanced observations — for example, that client preference heterogeneity might enhance parameter-space exploration — and discusses practical issues such as dataset imbalance and partitioning effects. 
The authors evaluate four distinct FMOL algorithms under both homogeneous and heterogeneous preference settings, providing systematic comparisons. The analysis includes Pareto front visualizations and hypervolume metrics across multiple datasets and fairness metrics.

### Weaknesses
While the motivation is well-argued empirically, the paper lacks formal justification for why fairness-based conflicts are representative of general multi-objective tension in federated learning. A theoretical characterization of conflict strength or Pareto diversity across benchmark types would have strengthened the claim.

Only three datasets (Adult, Law School, Default) are used in the main experiments. Although Table 1 lists more possible datasets, it remains unclear how well the approach generalizes to non-tabular modalities (e.g., images, text), where fairness labels and sensitive attributes behave differently.

The paper does not report whether fairness-based trade-offs are preserved or amplified by federation compared to centralized multi-objective training. This comparison would help confirm that the federated context indeed adds complexity.

Since fairness datasets involve sensitive demographic data, the paper could have addressed privacy-preserving concerns or how benchmark construction avoids ethical pitfalls. This is relevant for federated settings.

### Questions
why certain algorithms (e.g., FedPref vs. CFL) succeed or fail on the new benchmarks 


Since different fairness metrics are often incompatible, how sensitive are the benchmark results to the choice of metric (DDP vs. DEO)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the underexplored problem of benchmarking Federated Multi-Objective Learning. The authors argue that widely used multi-task benchmarks such as Multi-MNIST fail to exhibit genuine objective conflicts once deployed in a federated setting, leading to artificially easy problems. To remedy this, they propose a new class of fairness-based benchmarks, derived from established group fairness metrics. These benchmarks combine fairness and accuracy objectives, creating natural trade-offs. The authors evaluate several algorithms under homogeneous and heterogeneous preference settings.

### Strengths
The paper makes an interesting observation: existing multi-task benchmarks collapse under federated training, masking genuine multi-objective conflicts. This is an insightful finding that challenges standard evaluation practice. The proposed fairness-based benchmarks are well motivated by the inherent trade-off between fairness and accuracy. The experiments are competently executed across several algorithms and preference settings, offering initial evidence that the new benchmarks better reflect real multi-objective challenges. The discussion of homogeneous vs. heterogeneous preferences is also interesting.

### Weaknesses
While the paper presents a very interesting observation in Section 3 that federated training on Multi-MNIST under heterogeneous preferences may remove apparent task conflicts, but the analysis remains speculative. The experiment is limited because the Multi-MNIST setup is too simplistic to convincingly support the broader claim. More realistic examples or, ideally, a theoretical analysis explaining why preference heterogeneity mitigates conflicts in federation would greatly strengthen the argument.

Moreover, as a benchmark paper, some datasets used (Adult, Law School, Credit Default) are rather dated and small in scale. It is therefore unclear whether the proposed benchmarks generalize to realistic, large-scale FMOL settings.

Finally, the experimental comparison is quite narrow: only four algorithms are evaluated, and none of the more recent state-of-the-art multi-objective optimization methods are included. This limited scope weakens the empirical validation of the proposed benchmarks’ difficulty and generality.

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
