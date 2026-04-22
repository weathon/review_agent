# Continual Parameter-Efficient Adaptation for Rehearsal-Free Graph Class-Incremental Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Graph Class-Incremental Learning (GCIL) seeks to learn novel classes sequentially while preserving knowledge acquired from previously seen classes. However, to tackle the pervasive challenge of catastrophic forgetting, recent GCIL methods often train separate classifiers from scratch for each task, which is redundant in design and computationally expensive. Moreover, isolating streaming data in different tasks hampers knowledge transfer across tasks. To address these dilemmas, we first propose Graph2Hyper, a parameter-efficient framework that utilizes a hypernetwork to generate task-specific classifiers on the fly based solely on the input graph of the current task. Concretely, the hypernetwork is composed of just two linear layers: a frozen, task-shared layer that preserves cross-task knowledge, and a trainable, task-specific layer that captures the unique characteristics of each task. To distinguish between different tasks in the incremental learning process, task-prototypes are extracted via pooling over global node representations, which capture task-specific contextual knowledge. To further model the association between tasks and corresponding classes, we construct class-prototypes with dynamic task-level bias through a learnable mapping function. By encoding class-level discrimination while retaining task-level context, the hypernetwork enables continual forget-free adaptation to new classes without the need for prototype rehearsal. Extensive experiments on four benchmark datasets demonstrate that Graph2Hyper achieves promising performance with superior parameter efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper discusses rehearsal-free GCIL, which means that new classes come in over time, there is no way to save old data, and there are no task IDs at test time. The key challenges are accurate task identification without labels, preserving intra-task class separability, preventing catastrophic forgetting, and keeping adaptation parameter-efficient. The authors propose Graph2Hyper: (1) task prototypes built from propagation-pooled graph representations enable nearest-prototype task inference; (2) class prototypes with a learned dynamic bias enhance within-task discrimination by pulling same-class nodes together and pushing others apart; and (3) a lightweight hypernetwork generates task-specific classification heads from shared parameters, yielding efficient cross-task knowledge transfer. They provide theory showing the hypernetwork induces functional interpolation between old and new heads and tightens risk bounds under transferable structure. Experiments on CoraFull, OGB-Arxiv, Reddit, and Amazon Products show higher average accuracy and zero forgetting compared with strong GCIL baselines.

### Strengths
S1: Coherent “task→class” prototype hierarchy plus a lightweight hypernetwork yields rehearsal-free, parameter-efficient GCIL with strong cross-task transfer.

S2: Empirical results are robust across four benchmarks, align with the theory (functional interpolation / tighter risk bound), and ablations verify each component’s necessity.

### Weaknesses
W1: The experimental scope is narrow. Tasks always contain two classes, so the setting does not test more realistic cases such as three to five classes per step, uneven class counts, or class imbalance.

W2: Harder data conditions are under explored. The method is not evaluated with overlapping task distributions, open set or label noise, or clear inter task domain shift, and there is no report of task identification accuracy with a confusion matrix or failure case analysis.

W3: Efficiency evidence is incomplete. The paper does not provide wall clock time, memory use, FLOPs, or latency under identical hardware and batch size, and it lacks sensitivity sweeps of propagation order K, prototype and dictionary sizes, hypernetwork width, or alternative encoder strategies such as partial fine tuning or adapters.

### Questions
see the weaknesses above

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
4

### Summary
The paper proposes Graph2Hyper, a parameter-efficient framework for rehearsal-free graph class-incremental learning (GCIL). It uses a lightweight hypernetwork with a frozen shared layer and a trainable task-specific layer to dynamically generate classifiers based on the current task’s input graph. Task-prototypes and class-prototypes are introduced to capture task-specific context and enhance intra-task discrimination. The framework enables continual, forget-free adaptation without rehearsal and demonstrates competitive performance on four benchmarks.

### Strengths
1. The paper addresses a meaningful and timely problem in rehearsal-free graph class-incremental learning (GCIL).
2. The method achieves forget-free performance (AF = 0) across multiple benchmarks, matching or slightly improving over strong baselines.

### Weaknesses
1. The empirical gains over the most relevant baseline (TPP) are marginal (typically within 1–2%), which raises doubts about the practical significance of the proposed modifications. Moreover, the sensitivity analysis shows fluctuations of a similar magnitude (1–2%), making it difficult to conclusively attribute the improvements to the proposed method rather than parameter variance.
2. The general pipeline (frozen backbone + classifer-level adaptation) is conceptually close to TPP.
3. Theoretical presentation has incomplete references and unclear derivations (e.g., line 293 “Appendix XXX”, line 1042 “Lemma XXXXX”), which undermines rigor and readability.
4. The claim of “parameter efficiency” is not quantitatively substantiated — the paper lacks comparisons on parameter count, runtime, or memory cost relative to TPP or other baselines. And only aggregated metrics (AA/AF) are reported; there are no per-task results or visualization of learning dynamics to illustrate continual behavior.

### Questions
In Table 1, Graph2Hyper slightly outperforms the Oracle model on the Reddit dataset (99.5% ± 0.1 vs. 99.5%). Could the authors clarify how this is possible?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of rehearsal-free Graph Class-Incremental Learning (GCIL). To tackle catastrophic forgetting and parameter redundancy, it proposes the Graph2Hyper framework. This parameter-efficient framework's core contribution is the use of a partially-frozen, lightweight hypernetwork to dynamically generate task-specific classifiers. This hypernetwork comprises a frozen, shared layer for preserving cross-task knowledge and a trainable, task-specific layer for capturing task-specific characteristics. Leveraging a hierarchical prototype design—including task-prototypes to distinguish between tasks and class-prototypes with dynamic bias to enhance intra-task class discrimination—the method achieves high parameter efficiency, zero forgetting, and state-of-the-art average accuracy on multiple benchmark datasets. However, the method's novelty appears limited when compared to the closest baseline, TPP, as it adopts TPP's core settings. Furthermore, the practical connection between the paper's complex theoretical analysis and the final model's specific design is not sufficiently clear.

### Strengths
1. The method achieves zero average forgetting and surpasses all SOTA baselines, including TPP, in Average Accuracy across all four benchmark datasets.
2. The core contribution is a lightweight hypernetwork that dynamically generates classifiers, conceptually avoiding the significant parameter overhead of storing independent heads for each task.
3. The method employs a logically sound hierarchical prototype mechanism. Task-Prototypes achieve coarse-grained inter-task separation, while Class-Prototypes enhance fine-grained intra-task discrimination.

### Weaknesses
1. The proposed method heavily relies on the core configurations of TPP, such as using the same pretraining and frozen GNN encoder strategy. This dependence limits the method’s conceptual independence and raises questions about its novelty beyond TPP.

2. Although the key innovation lies in adopting a hypernetwork instead of TPP’s independent classifier, the paper does not provide sufficient empirical comparisons. For instance, the main experimental table (Table 1) omits direct comparisons in terms of the number of trainable parameters and training time between the proposed approach and TPP.

3. The paper presents complex theoretical analyses (e.g., Theorems 4.2–4.6); however, these results do not clearly motivate or justify the final architectural design choices of the hypernetwork. The link between theory and practical implementation remains unclear.

4. All experiments are conducted using SGC as the backbone. It remains unclear whether the proposed method would maintain its performance and parameter-efficiency advantages when employing more powerful GNN encoders.

5. “Superior parameter efficiency” is claimed as a core contribution of the paper, yet this claim is primarily supported by theoretical analysis (Appendix B.2) rather than empirical evidence. The main results (Table 1) do not include direct quantitative comparisons of total trainable parameters with key baselines such as TPP.

6. In the ablation study on the Class-Prototype module, the authors remove the entire module, making it impossible to isolate the contribution of its internal components—particularly the dynamic bias term (Eq. 3). A more fine-grained ablation would be needed to clarify its specific impact.

### Questions
See weakness

### Soundness
2

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
The paper tackles rehearsal-free Graph Class-Incremental Learning (GCIL) by proposing Graph2Hyper, a parameter-efficient framework that avoids catastrophic forgetting without storing past data. The method builds a coarse-to-fine prototype hierarchy—a task prototype to identify the task at test time, and class prototypes (as dynamic biases) to sharpen intra-task decision boundaries. A lightweight hypernetwork then generates the task-specific classifier head “on the fly,” sharing part of its parameters across tasks to transfer knowledge. A theoretical analysis argues that the hypernetwork’s adaptation is equivalent to an affine interpolation between the previous and current heads and yields a tighter generalization risk bound. Experiments on four benchmarks (CoraFull, Arxiv, Reddit, Products) show state-of-the-art accuracy with zero forgetting and strong parameter efficiency.

### Strengths
1. Clean, modular architecture that is rehearsal-free and forget-free:
The method pairs a coarse-to-fine prototype hierarchy (task prototypes for routing; class prototypes with dynamic, task-level bias for within-task separation) with a lightweight hypernetwork that generates task-specific heads while sharing part of its parameters across tasks. This yields continual adaptation without prototype rehearsal and targets the GCIL setting directly.

2. Clear, informative framework figure that clarifies the entire pipeline:
Figure 1 cleanly lays out both training—(1) task-prototype construction, (2) class-prototype dynamic bias, (3) hypernetwork-based head generation—and inference—prototype-dictionary matching for task ID and on-the-fly head generation—making the approach easy to follow and reproduce.

3. Theory that explains why the hypernetwork sharing works:
The paper proves a necessary/sufficient equivalence: when the adapted head lies in the affine span of the previous and independently trained heads, the model is functionally equivalent to their interpolation; for general heads a first-order equivalence holds with a bounded second-order remainder, which leads to a risk-transfer bound (exact for linear heads). This gives a principled account of knowledge transfer and expected risk behavior.

### Weaknesses
1. Scope & fairness of the “forget-free” claim:
The paper inherits the TPP setting and predicts a task ID at inference by matching a test graph-level prototype against a dictionary; this is what enables AF=0 across datasets (Table 1), not only the proposed hypernetwork. This assumption (graph-level, transductive task identification) is stronger than many GCIL protocols where test samples arrive mixed and task IDs are unknown without access to a full test graph. Baseline numbers are mostly taken “from published works,” so some methods may not benefit from the same task-ID prediction pipeline or frozen encoder, making comparisons optimistic for Graph2Hyper. Consider adding results under a stricter GCIL setting (no graph-level task matching, or noisy task-ID prediction) and re-running baselines in the same evaluation harness.

2. Fragility to task-ID errors (and task-ID leakage); need robustness analysis:
The approach implicitly assumes access to a per-task test graph and near-perfect task-ID prediction via prototype matching—a local-testing setup that can leak the task ID and inflate results. In this light, the ablation “w/o Task-P” may collapse not only because the model is weak, but because it removes the (potentially leaky) routing path the full system relies on. Please (i) report AA/AF as a function of task-ID accuracy (e.g., inject controlled noise into dictionary matching), (ii) evaluate under global testing where nodes from different tasks are mixed and no per-task test graph is available, and (iii) clarify whether the method requires test-graph access or supports node-level inference in mixed graphs. This will disentangle genuine robustness from gains driven by task-ID leakage or overly strong task-separation assumptions.

3. Minor correctness & polish issues:
- Table 1 misspells Arxiv as “Arixv.”
- The paper claims “code available after acceptance”; consider a (sanitized) artifact now to aid reproducibility.
- In Table 1, you mark both AA and AF with “↑” even though AF is a forgetting measure; you later explain positive AF for replay methods, but a clearer caption would reduce confusion.

### Questions
Q1. Inference assumptions:
Do your test-time assumptions require a per-task test graph and transductive access for task-ID prediction? State the exact inference setting in the main text. Add a schematic of the test-time pipeline and explicitly note whether per-task test graphs are required.

Q2. Robustness to task-ID errors:
How exactly is the task ID predicted (prototype construction, number of exemplars, normalization, update policy)? Document the full matcher recipe (feature normalization, distance metric, thresholds), publish the code snippet, and report task-ID accuracy alongside AA/AF.

Q3. Scope of the “forget-free” claim:
Under which evaluation settings does AF≈0 hold, and does it remain near zero when task IDs are unknown or imperfect? Reframe the claim with scope (“under local testing with accurate task-ID prediction”) and report AF under (i) global testing and (ii) noisy task-ID conditions.

### Soundness
3

### Presentation
3

### Contribution
3
