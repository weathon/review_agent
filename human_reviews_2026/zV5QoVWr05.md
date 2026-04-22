# Towards the Explainability of Temporal Graph Networks via Memory Backtracking

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Temporal graphs are ubiquitous in real-world applications such as social networks and finance, where Temporal Graph Networks (TGNs) achieve superior predictive accuracy. Understanding which historical events drive specific
model predictions enhances trustworthiness of TGNs.  Existing explanation methods for TGNs overlook the memory module, the core component that records and updates node histories, leaving unexplored how past events shape memory dynamics and influence the current predictions. To address this challenge, we propose a framework that attributes TGNs predictions through the topology attribution tree and memory backtracking tree. The topology attribution tree captures neighbor influence, including the impact of their memory vectors. Then, we use the memory backtracking tree to quantify how historical events shape memory evolution. Our method satisfies a conservation principle, ensuring that the total contribution of events equals the model’s logits. Finally, we introduce optimization objectives to map logits to probabilities. Experiments on seven temporal graph datasets, spanning node property prediction and link prediction tasks, show that our method provides faithful explanations and consistently outperforms four state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies explainability for Temporal Graph Networks. It introduces MemExplainer that attributes a target prediction to recent interactions and to stored node memories. The approach builds a topology attribution tree and then backtracks memory updates to earlier events. The authors claim a conservation-style property linking the final logit to the sum of selected event contributions. Experiments are shown across several temporal-graph tasks and datasets using figures to suggest improved fidelity.

### Strengths
1. The work targets a real need for explanations that account for temporal memory in graph models.

2. The pipeline is clear at a high level, with a split between topology attribution and memory backtracking and an emphasis on contribution conservation.

3. The evaluation spans multiple datasets and both link prediction and node-property prediction, which increases practical relevance.

### Weaknesses
1. The evaluation relies on model-internal fidelity measures that compare outputs before and after event selection. These measures can reward matching model behavior rather than faithfulness to the underlying phenomenon. There is no human study, no causal validation, and no task-utility assessment that would confirm usefulness beyond reproducing the model.

2. Key ablations and robustness checks are missing. The paper does not ablate the memory-backtracking component or the topology-attribution component, so their individual contributions are not isolated. And there is little analysis of sensitivity to timestamp noise, to alternative memory-update modules, or to choices that control recent-event sampling and backtracking depth. Multi-seed statistics and significance testing are not reported.

3. Computational cost is not reported. The method builds and traverses attribution structures over time, yet there are no numbers for run-time or peak-memory, or how these scale with graph size, event density, number of layers, or backtracking depth.

4. Baseline and metric coverage could be stronger. Only a small set of explainers is considered, and some are adapted from static-graph settings.

5. Lack of reproducibility details. Key implementation choices are not clearly described, including how node memories are built and updated, when event selection stops, and what optimizer is used to pick the event set. The paper does not mention code release or random seeds.

### Questions
1. What are the typical run-time and memory costs per instance, and how do they change with graph size, number of sampled recent events, number of layers, and backtracking depth?

2. Can you provide ablations that remove the memory-backtracking stage, remove the topology-attribution stage, and replace the event-selection objective with a simple top-$k$ rule to quantify each component’s contribution?

3. How sensitive are the explanations to timestamp perturbations, to noisy features, and to alternative memory-update modules? Do the explanations transfer across different TGN variants?

4. Do you have validation beyond model matching, such as human judgments or interventional tests that remove or inject events to confirm that selected events are truly explanatory?

5. How is the optimization over the selected event set solved in practice?

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
This paper proposes an explanation method for attributing the output of temporal graph neural networks to the important historical events. Specifically, the proposed method first applies the LRP on the embedding module to compute the topology attribution tree, which estimates the node memories’ contribution to node embeddings. Then, the proposed method further propagates it to the historical events by applying LRP on the memory updating module. The important events are finally selected by solving an optimization problem based on the obtained event contribution matrix.

### Strengths
- The ideas of applying the two-step LRP to estimate the importance of historical events and formulating the selection problem as an optimization variant are natural.
- The paper is generally well-written and easy to follow
- The proposed method achieves better performance than the compared baselines.

### Weaknesses
- The proposed method appears computationally intensive, as it requires building a topology attribution/memory backtracking tree and solving an optimization problem for each prediction. The authors should provide a clear complexity analysis and compare it with baseline methods. Additionally, a runtime comparison is needed to demonstrate practical efficiency.
- Some parts of the paper need further clarification. 1)How is Equation (13) converted into Equation (14)? 2) How is the optimization of Equation (14) actually solved?
- Minor issues. 1) It is better to denote the shape of the matrices/vectors when they first appear. 2) The superscript $t$ of $\mathcal N_{u_k}^t$ in Line 159 should be $n$.

### Questions
Please see the Weaknesses.

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
This paper addresses the challenge of explainability in Temporal Graph Networks (TGNs). It proposes a novel framework called MemExplainer to attribute the predictions made by TGNs by considering both spatial and temporal dynamics. The framework introduces two components: the topology attribution tree, which attributes predictions based on the contributions of recent events and their associated node memories, and the memory backtracking tree, which traces how past events influence memory evolution. The method ensures that the total contribution from events equals the model’s output logits, maintaining a conservation property. Extensive experiments on seven temporal graph datasets demonstrate that the proposed method outperforms several explainability approaches, showing improved fidelity and lower sparsity in the explanations.

### Strengths
1. This work proposes an innovative framework for explaining TGNs. The dual-attribution approach—using a topology attribution tree and a memory backtracking tree—gives a more comprehensive explanation of model predictions by accounting for both spatial and temporal influences. The conservation principle ensures that the sum of all event contributions exactly matches the model’s output logits, which helps guarantee the explanation’s faithfulness.
2. The method shows strong performance across multiple benchmarks. It consistently outperforms existing explainers such as TGNNExplainer and TempME on both node property prediction and link prediction tasks.
3. The paper is supported by solid theoretical foundations and practical design. For example, building both the spatial and temporal attribution components on Layer-wise Relevance Propagation (LRP) provides a consistent and theoretically justified framework, enhancing both reliability and interpretability.

### Weaknesses
1. The paper's exploration of explainability for memory-based dynamic graph methods is commendable. However, it is questionable whether such methods can provide meaningful explanations on datasets with very high repetition rates, such as Wikipedia, Reddit, etc. This issue has been noted in recent research. For example, [1] points out a key limitation of these datasets: models can achieve good performance simply by remembering whether an edge has appeared before, making the learning task too easy. The strong performance of the EdgeBank model from [3]—which requires no training—on these datasets further supports this concern. Similarly, [2] analyzes attention weights in Transformer-based models and finds that models mainly rely on high-frequency, repetitive edges. This makes the explanations straightforward and intuitively clear, but also suggests they may not capture deeper reasoning for datasets with low repetition rate. Therefore, when evaluating dynamic graph models, using event repetition rate as a criterion for dataset selection might lead to more meaningful research.
2. Although the method performs well on several datasets, its computational cost—especially from the memory backtracking process—could be an issue for large graphs with long temporal histories. The approach involves recursive backtracking and memory updates, which may become slow on very large-scale data. In addition, while the optimization for selecting important events is defined in Equation 14, the paper lacks details on how this optimization is implemented efficiently in practice, especially as the number of events increases. More information about convergence behavior, runtime, and computational trade-offs would help readers assess its practicality.
3. While the framework is technically sound, the resulting explanations may not be easy to interpret. The tree structure (topology and memory) could be hard to understand without more explanation or visual examples. The paper would benefit from including concrete case studies or qualitative examples to show how specific predictions (on both low/high repetition rate) are explained and how users might apply these explanations in real-world scenarios.

Reference

[1] TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics https://arxiv.org/abs/2502.02975v3

[2] TIDFormer: Exploiting Temporal and Interactive Dynamics Makes A Great Dynamic Graph Transformer https://arxiv.org/abs/2506.00431

[3] Towards Better Evaluation for Dynamic Link Prediction https://arxiv.org/abs/2207.10128

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MemExplainer, an explainability framework for Temporal Graph Networks (TGNs). It decomposes the model's prediction into contributions from historical interaction events through a two-stage attribution process: the Topology Attribution Tree and the Memory Backtracking Tree.

### Strengths
Timely and meaningful problem: Explaining TGNs is crucial in domains such as finance, recommendation, and fraud detection; focusing on memory influence is insightful.

Comprehensive method: The topology + memory two-step structure is well-described with algorithms and clear relevance propagation logic.

### Weaknesses
Unclear optimization in Eq. (14): The binary event-selection problem lacks details—whether solved exactly, heuristically, or via continuous relaxation—and its time complexity. Clarify the algorithm and provide runtime statistics.

Scalability and resource cost: Memory backtracking may explode for large event histories. The paper lacks complexity or runtime/memory analysis. Please add empirical resource tables or propose pruning strategies.

Baseline implementation details: How were static explainers (GNNExplainer, PGExplainer) adapted to TGNs? What hyperparameters and seeds were used for TGNNExplainer and TempME？

Robustness / statistical significance: Report averages ± std or confidence intervals for repeated runs, since some improvements are small.

Model-component assumptions: The derivations assume a GRU updater. Discuss applicability to other TGNs (e.g., attention-based or transformer-style updaters) and whether the LRP rules transfer directly.

Limited human-interpretability evaluation: Fidelity metrics show quantitative preservation, but user-level interpretability is not assessed.

### Questions
How exactly is Eq. (14) optimized? What algorithm and complexity are used, and what are the runtime/memory statistics across datasets?

Please provide empirical runtime and memory data for memory-backtracking with different depths L.

Can the approach generalize to non-GRU TGNs (e.g., attention updaters)? What modifications are required?

### Soundness
3

### Presentation
3

### Contribution
3
