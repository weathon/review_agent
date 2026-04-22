# Never Skip a Batch: Continuous Training of Temporal GNNs via Adaptive Pseudo-Supervision

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Temporal Graph Networks (TGNs), while being accurate, face significant training inefficiencies due to irregular supervision signals in dynamic graphs, which induce sparse gradient updates. We first theoretically establish that aggregating historical node interactions into pseudo-labels reduces gradient variance, accelerating convergence. Building on this analysis, we propose History-Averaged Labels (HAL), a method that dynamically enriches training batches with pseudo-targets derived from historical label distributions. HAL ensures continuous parameter updates without architectural modifications by converting idle computation into productive learning steps. Experiments on the Temporal Graph Benchmark (TGB) validate our findings and an assumption about slow change of user preferences: HAL accelerates TGNv2 training by up to 13$\times $ while maintaining competitive performance. Thus, this work offers an efficient, lightweight, architecture-agnostic, and theoretically motivated solution to label sparsity in temporal graph learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Node-level labels for node property prediction tasks in the TGB benchmark appear irregularly, resulting in a large proportion of training batches that lack ground-truth labels. Existing training procedures typically only perform memory state updates for these unlabeled batches, without updating model weights, leading to sparse gradient updates and consequently slow convergence.

This work introduces History-Averaged Labels (HAL), which augments batches with few or no true labels by generating pseudo-targets from historical data, thereby converting idle computation into productive gradient-based learning steps.

### Strengths
- The paper proposes a simple yet effective solution to the critical problem of sparse and irregular training signals in the node property prediction task in temporal graph learning on the TGB benchmark.

- The proposed History-Averaged Labels (HAL) method is architecture-agnostic, requiring no changes to the underlying model and no additional training parameters, making it broadly applicable and practical for diverse models.

- The paper provides a rigorous theoretical analysis, demonstrating that HAL reduces the variance of gradient estimates and accelerates the convergence of stochastic gradient descent.

- Empirical experiments on two advanced models (TGNv2 and DyRepv2) across multiple TGB node property prediction datasets show substantial improvements in convergence speed, with no degradation in model performance.

### Weaknesses
**W1.** The authors claim that the proposed method, HLA, is architecture-agnostic. However, this claim is not sufficiently substantiated, as the evaluation is limited to only two models, TGNv2 and DyRepv2. To convincingly demonstrate architecture independence, the authors should validate HLA on a broader range of temporal graph models, such as DyGFormer[1], TPNet[2] , TGAT[3], GraphMixer[5] and AGCRN[6]. Including results on these diverse architectures would provide stronger empirical evidence for the architecture-agnostic property of HLA.

**W2.** Lines 151–152 state
> Forward pass: A graph neural network (GNN) processes the subgraph to compute context-aware node embeddings.

While this description applies to a subset of temporal graph network (TGN) models that explicitly employ GNN modules for structural encoding, it does not generalize to a broader class of models, such as DyGFormer[1], DyGMamba[4], GraphMixer[5]. Given that the proposed method claims to be architecture-agnostic, this explanation should be reframed in a more general context, avoiding focusing on specific to GNN-based variants of TGNNs and better reflecting the diversity of temporal graph architectures.

**W3.** The overall readability of the paper could be significantly improved by rearranging and rewriting certain sections.

- Lines 98–107, which contain critical information about the problem formulation, should be separated into a section or subsection to clearly and explicitly define the problem statement and the task addressed by this work.

- The term “unsupervised batch” used in Line 132 to describe batches without training signals is misleading. In standard terminology, “unsupervised” implies learning from unlabeled data without explicit output labels, which is not synonymous with simply lacking training signals in batches. Clarification or alternative wording would prevent confusion.

- Several important terminologies, such as “Default” in Table 1, are introduced without clear definitions or explanations.

- Some statements are ambiguously phrased. For instance, in the sentence “As long as it is unbiased, we can estimate its variance,” the pronoun “it” lacks a clear reference, which makes the meaning uncertain.

- There is inconsistency in dataset naming across the manuscript. Datasets are referred to as “genre,” “reddit,” and “token” (e.g., Lines 379–380 and Table 1), while elsewhere the names “tgbn-trade,” “tgbn-genre,” “tgbn-reddit,” and “tgbn-token” are used.

**W4.** Reproducibility: The paper does not provide the source code or specify the hyperparameter settings (e.g., learning rate, weight decay, etc.) used in the experiments, raising concerns about the reproducibility of the work.

**W5.** Missing Citation: The datasets used in this work were not properly cited. I recommend that the authors include the appropriate references for each dataset, as listed in the [TGB datasets documentation](https://tgb.complexdatalab.com/docs/nodeprop/), to ensure proper attribution.

---

[1] Yu, Le, et al. "Towards better dynamic graph learning: New architecture and unified library." *Advances in Neural Information Processing Systems* 36 (2023): 67686-67700.

[2] Lu, Xiaodong, et al. "Improving temporal link prediction via temporal walk matrix projection." *Advances in Neural Information Processing Systems* 37 (2024): 141153-141182.

[3] Xu, Da, et al. "Inductive representation learning on temporal graphs." *arXiv preprint arXiv:2002.07962* (2020).

[4] Ding, Zifeng, et al. "Dygmamba: Efficiently modeling long-term temporal dependency on continuous-time dynamic graphs with state space models." *arXiv preprint arXiv:2408.04713* (2024).

[5] Cong, Weilin, et al. "Do we really need complicated model architectures for temporal networks?." *arXiv preprint arXiv:2302.11636* (2023).

[6] Bai, Lei, et al. "Adaptive graph convolutional recurrent network for traffic forecasting." *Advances in neural information processing systems* 33 (2020): 17804-17815.

### Questions
- Lines 176–177: What is the definition of $\alpha$, and what is its range? Additionally, which distribution mean are the authors referring to in this context?

- Line 169: The authors mention “non-zero for almost all nodes and batches.” Why “almost”? In what cases would the pseudo-label be zero?

- Lines 172–173: The authors state that pseudo-targets are computed only for nodes participating in the current batch $B_t$. Are nodes involved in edge events also considered, or only those involved in node events?

- TGBN-Token: Why does the performance peak appear around both window sizes 2–4 and 10–12, unlike TGBN-Trade and TGBN-Genre, which favour either short or long windows? Why does TGBB-token favour both?

- Figure 4: This figure shows the correlation between NDCG@10 and the moving average window size for the TGNv2 model. Is the same trend observed for DyRepv2?

- It is unclear why refining the training regime by retaining only the last 5% of chronologically ordered edges in the original training set is necessary. Could the authors elaborate on this choice? If all models can converge within a single epoch, why is HLA needed?

- In the Order Importance Check with Target Shuffling experiment, the setup of the ablation study is not clearly explained. How are the targets shuffled? Are targets shuffled between nodes, or is the order of targets for each node shuffled?

- Table 1: Why is $X = 4$ used for TGBN-Genre and TGBN-Reddit, while $X = 2$ is used for TGBN-Token?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the inefficiency of training Temporal Graph Networks under sparse supervision. The authors propose a lightweight pseudo-labeling approach called History-Averaged Labels (HAL), which continuously generates supervision signals for unlabeled batches by aggregating historical labels. Three variants are introduced: (1) Historical Average (HA), (2) Moving Average (MA), and (3) Persistent Forecast (PF).

The paper provides a theoretical analysis showing that label aggregation reduces gradient variance in stochastic gradient descent, leading to faster convergence by a factor of approximately min(h, k) (history length and number of classes). Empirical results on four datasets from the Temporal Graph Benchmark (TGB) demonstrate up to 13× faster training of TGNv2 and DyRepv2 without performance degradation, measured by NDCG@10.

### Strengths
1) The paper Introduces a simple yet novel approach to handle sparse supervision in temporal GNNs via History-Averaged Labels, enabling continuous training even on unlabeled batches.

2) It adapts historical averaging concepts from time-series forecasting to pseudo-labeling in dynamic graphs.

3) The method is easy to integrate into existing models without architectural changes.

4) Well-written, logically organized, and supported by informative figures and ablation studies.

5) Addresses an important and practical bottleneck in a growing area of research.

### Weaknesses
1) The experiments are restricted to only two architectures of similar type (TGNv2 and DyRepv2), which limits the evidence for generality. Broader testing across diverse temporal GNN frameworks would better support the “architecture-agnostic” claim.

2) The fact that the method achieves strong performance using only 5% of the training data is encouraging and highlights its data efficiency. However, the presentation of this result is somewhat confusing and potentially misleading, as it is framed as a “13× faster convergence” rather than as an advantage in low-supervision settings. Clarifying that the speedup reflects data efficiency rather than pure computational acceleration would make the contribution more transparent and credible. This clarification should be explained from the beginning.


3) The method critically relies on the assumption that node dynamics evolve slowly over time, yet this is never verified empirically. Without evidence that the assumption holds in real-world datasets, the general applicability remains uncertain.

4) The paper does not compare HAL against standard pseudo-labeling or semi-supervised learning approaches. This makes it difficult to assess how much of the observed benefit comes from the temporal design versus general pseudo-supervision effects.

### Questions
1) The theoretical section assumes convexity and independence between pseudo-labels and true labels. Could the authors clarify how these assumptions hold (or are approximated) in the context of deep non-convex models such as TGNs? A brief discussion on the limitations of the proof or its empirical verification would be useful.

2) The paper assumes that node preferences evolve slowly over time. Have the authors measured or quantified this stability in the datasets (e.g., via label autocorrelation or distribution drift)? Providing such evidence would strengthen the motivation for HAL.

3) The experiments only consider two architectures (TGNv2 and DyRepv2), both memory-based. Could the authors explain why other models—such as TGAT, CAW, or GraphMixer—were excluded? This would help readers assess the claimed architecture-agnostic property.

4) The method performs well using only 5% of the training data, which is a notable result. However, it is presented as a “13× faster convergence” rather than as a demonstration of data efficiency. Could the authors clarify this framing and make it explicit from the start that the reported speedup is achieved in a reduced-data regime? This clarification should be explained from the beginning.

5) HAL is conceptually related to pseudo-labeling approaches (e.g., self-training, label smoothing). Why were such baselines not included? A discussion or small-scale comparison would help position the method more precisely within the broader literature.

[minor] 

6) Figure 1 illustrates only the Moving Average variant. Could the authors expand this figure or add supplementary visualizations for the Historical Average and Persistent Forecast variants to clarify their operational differences?

7) The absence of code or pseudo-code limits reproducibility. Do the authors plan to release an anonymized implementation for review purposes? If not, could they explain the motivation?

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
4

### Summary
This paper, “Never Skip a Batch: Continuous Training of Temporal GNNs via Adaptive Pseudo-Supervision,” introduces a pseudo-labeling strategy for temporal graph neural networks (TGNv2, DyRep). The idea is to generate pseudo-supervision signals for unlabeled batches using historical labels, enabling continuous training when ground-truth labels are sparse. Theoretical analysis claims reduced gradient variance and faster SGD convergence, while experiments on four Temporal Graph Benchmark (TGB) datasets report significant speedups with similar accuracy.
The topic is relevant and the problem is practically important. However, the contribution is relatively incremental, as the proposed method mostly modifies existing TGN-based architectures by adding three pseudo-labeling variants (HAL, MA, PF). The motivation and experimental analysis require clearer support.

### Strengths
1.	Practical and timely problem. Temporal GNNs indeed suffer from sparse supervision, where many batches lack labels. Addressing this inefficiency is valuable for real-world streaming systems.
2.	Implementation simplicity. The proposed pseudo-labeling (HA/MA/PF) is easy to integrate into existing TGN pipelines, which enhances reproducibility.
3.	Initial theoretical analysis. The paper attempts to formalize the benefit of pseudo-labels through reduced gradient variance, offering a conceptual link between theory and empirical gains.
4.	Empirical evidence of faster convergence. Reported 2–13× training speedups are promising, though validation is limited.

### Weaknesses
1.	Limited novelty. The contribution lies mainly in adding pseudo-label updates to existing temporal GNNs (TGNv2, DyRep). No new architecture, optimization mechanism, or learning paradigm is introduced. And the proposed pseudo-labeling methods are kind similar to the “Moving Average” method mentioned in paper Temporal Graph Benchmark for Machine Learning on Temporal Graphs.
2.	Pseudo-label initialization unclear. When there is insufficient history (early timesteps), how are pseudo-labels initialized?
3.	Lack of quantitative motivation. The claim that “most batches contain sparse labels” is not empirically demonstrated. A motivating figure or table showing label density over time would make the motivation concrete.
4.	Possible conflict with the prediction goal. The task predicts a node’s current-time preference, yet HAL aggregates historical labels (shown in Figure 1). For fast-changing dynamics, this could blur the current signal. 
5.	Insufficient baselines. The comparison includes only “Default,” “Default-X,” and HAL variants. Stronger baselines (DyGFormer, NAVIS etc) exist within the TGB framework. Without them, it is unclear whether HAL provides consistent benefits. 
6.  Evaluation protocol. The experiments only use the most recent 5 % of interactions.  Would HAL still help on full datasets? Given that HAL relies on historical averaging, using a short time window may implicitly favor the method by limiting temporal drift and ensuring that historical pseudo-labels remain close to current preferences. Would HAL’s benefit diminish or vanish if the full dataset were used?  If the performance improvement disappears in that setting, it would imply that temporal GNNs like TGNv2 and DyRep already learn adequately. This would substantially narrow the scope of the proposed contribution.
7.	Presentation and citation issues. Missing Related Work section and lack of direct comparisons with pseudo-labeling, self-training, and temporal KG methods. Citations are incomplete and incorrectly formatted. The “General Pipeline” figure oversimplifies TGN frameworks; not all temporal GNNs follow this structure. Proper citations are required.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the problem of sparse supervision in Temporal Graph Networks (TGNs), where only a small fraction of node interactions are labeled. It proposes a history-based pseudo-labeling method that generates pseudo-targets for unlabeled batches using historical information, mainly through a Moving Average (MA) strategy. The approach allows continuous training rather than skipping unlabeled steps. The authors provide theoretical proof of faster SGD convergence and show experiments on four Temporal Graph Benchmark datasets, achieving 2–13× faster training while maintaining or improving accuracy.

### Strengths
1. The paper proposes a simple but effective pseudo-labeling method based on exponential moving averages of past labels. This approach reduces label sparsity and allows continuous training on temporal graphs.

2. The authors provide a clear theoretical analysis proving faster SGD convergence under historical label aggregation, with a quantified improvement factor of min(h, k). This adds rigor and supports the method’s validity.

3. The approach is implemented on both TGNv2 and a modified DyRep v2, showing up to 13× faster training without loss of accuracy across four benchmark datasets. This demonstrates the method’s practicality and generality.

### Weaknesses
1. In Section 2.2, the paper states: “For each batch Bt we compute pseudo-targets only for nodes v participating in Bt.” It is unclear what “unlabeled” means for these nodes — are they naturally without supervision at this timestep, or is this due to missing ground truth? If it is the former, using historical pseudo-labels might distort the temporal dynamics of infrequent or slowly changing nodes, whose past labels may no longer represent their current state. Could this affect model stability or prediction accuracy in such cases?

2. For new or long-inactive nodes, the MA and PF strategies cannot rely on any past labels. How are these cases handled — ignored, initialized uniformly, or inferred in some other way? 

3. Storing historical pseudo-labels for all nodes may introduce additional memory and synchronization overhead. As training progresses, this cache could grow substantially. It would be helpful to include an analysis or discussion of the memory and runtime impact of maintaining this history.

### Questions
1. Could using historical pseudo-labels distort the behavior of infrequent or slowly changing nodes?

2. How are new or long-inactive nodes handled when no past labels are available?

3. Does maintaining historical pseudo-labels for all nodes add noticeable memory or runtime overhead?

### Soundness
3

### Presentation
3

### Contribution
2
