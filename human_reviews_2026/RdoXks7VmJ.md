# Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Existing Temporal Knowledge Graph (TKG) representation learning approaches focus on modeling entity or relation correlations. However, since TKG datasets are constructed from events, which inherently contain heterogeneous causalities, focusing solely on entity or relation level correlations is inadequate for event prediction in TKGs. Although a TKG structural causal model can be established as a theoretical framework for event level causality disentangling, practical disentanglement is non-trivial due to the lack of explicit supervision signals. To this end, we propose a Heterogeneous Event causality Disentangling Representation learning Approach (HEDRA) for TKG reasoning, which is the first work that focuses on disentangling heterogeneous causalities at the event level in TKGs. Specifically, a counterfactual detector module is proposed to disentangle non-causality by leveraging event importance and distributional discrepancies of event representations. Moreover, an Instrumental Variable (IV)-guided disentangling module is proposed to disentangle spurious causality by constructing IVs, which can produce robust event representations against spurious causality through multi-view causality subgraphs. Finally, an evolutionary orthogonal module is proposed to separate dynamic causality from static causality for event prediction. Comprehensive experiments on five real-world datasets demonstrate that HEDRA achieves the state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces HEDRA, the first representation learning approach dedicated to disentangling heterogeneous event-level causalities in Temporal Knowledge Graphs, employing counterfactual detection, instrumental variable-guided modules, and evolutionary orthogonal mechanisms to separate non-causality, spurious causality, and dynamic causality, respectively; extensive experiments on five real-world datasets demonstrate HEDRA’s state-of-the-art performance in TKG reasoning.

### Strengths
1. The manuscript is well-written and structurally coherent. Despite the extensive use of diverse symbols, each is accompanied by a clear and explicit explanation.

2. The paper is grounded in solid theoretical foundations, effectively integrating causal learning theory into the domain of temporal knowledge graphs.

3. The experimental evaluation presented in the paper is comprehensive and rigorous, providing substantial evidence for the effectiveness of the proposed method.

### Weaknesses
1. This manuscript introduces numerous modules, which inevitably leads to increased computational time complexity (as demonstrated in Appendix C.3 and Table 8), potentially limiting the applicability of this approach to larger-scale temporal knowledge graphs.

2. This paper introduces numerous hyperparameters ($\alpha_{attn}$, $\beta_{KL}$); however, it lacks corresponding sensitivity analysis. If these parameters are all set to fixed values (0.5), then incorporating such an extensive array of hyperparameters becomes unnecessary.

3. In Table 1, the experimental results on ICEWS18 show a notable decrease in Hits@10 performance despite achieving superior outcomes on other metrics. The underlying reasons for this discrepancy warrant further clarification.

### Questions
See Weakness

### Soundness
3

### Presentation
4

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
Existing Temporal Knowledge Graph (TKG) methods mainly focus on modeling entity-level correlations, learning the relationships among entities through graph structures (nodes and edges) or higher-order structures such as hypergraphs and clusters. However, since events themselves exhibit diversity and causal heterogeneity, relying solely on entity correlations is insufficient to capture event-level causal mechanisms, thereby limiting the reliability of event prediction. HEDRA is the first TKG representation learning approach that achieves heterogeneous causality disentanglement at the event level. By incorporating counterfactual detection, instrumental variable guidance, and dynamic orthogonal decomposition, HEDRA effectively distinguishes non-causality, spurious causality, static causality, and dynamic causality, significantly enhancing both the performance and interpretability of event prediction tasks.

### Strengths
1. HEDRA is the first to elevate the research focus to the event-level causal relationships.
2. The paper theoretically formalizes four types of event-level causality, providing a solid theoretical foundation for future studies on temporal knowledge graphs based on causal reasoning.

### Weaknesses
1. The paper does not include a comparison with traditional methods based on GCN and GRU, such as TiRGN[1].
2. It also lacks comparisons with recent large-model approaches specifically designed for event prediction, such as MIRAI[2].
3. The anonymous GitHub repository is empty.
4. The notation is abundant and overly complex, which slightly reduces readability.
5. The text above the figure, such as in Figure 1, is too small.

[1] Li Y, Sun S, Zhao J. Tirgn: Time-guided recurrent graph network with local-global historical patterns for temporal knowledge graph reasoning[C]//IJCAI. 2022: 2152-2158.

[2] Ye C, Hu Z, Deng Y, et al. Mirai: Evaluating llm agents for event forecasting[J]. arXiv preprint arXiv:2407.01231, 2024.

### Questions
1. Why does the paper not include comparisons with TiRGN and large-model-based approaches?
2. In the introduction, the paper states:“Other approaches introduce derived structures, e.g., entity groups, hypergraphs, and evolutionary clusters, to capture high-order correlations among entities that are not directly connected (Zhang et al., 2022; Tang & Chen, 2024; Tang et al., 2024; Chen & Chen, 2024).” However, Tang & Chen (2024) proposed DHyper, which does not merely focus on entity-level correlations but jointly models high-order dependencies among both entities and relations.Why, then, is DHyper categorized under methods that primarily model entity correlations? Moreover, does its joint modeling capability potentially challenge the claimed novelty of this paper’s approach?

### Soundness
2

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
This paper proposes HEDRA, the first framework to disentangle heterogeneous event-level causalities in Temporal Knowledge Graphs. By developing a structural causal model and three specialized modules—counterfactual detection, IV-guided disentanglement, and evolutionary orthogonalization—it effectively separates static, dynamic, non-, and spurious causalities. Comprehensive experiments on five benchmarks demonstrate state-of-the-art performance, with significant improvements over existing methods, while maintaining practical computational efficiency.

### Strengths
Proposes the first framework for event-level causal disentangling in temporal knowledge graphs.

Comprehensive experiments on five datasets show consistent SOTA performance.

### Weaknesses
Ablation results show only minor drops.

The method introduces multiple complex modules, resulting in high training complexity and long runtime, especially on large-scale datasets such as GDELT.

It is unclear whether the proposed method can represent or generalize to unseen entities or relations.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
HEDRA proposes an event-level structural causal model for temporal knowledge graphs that separates four kinds of signals: non-causality, spurious causality, static causality, and dynamic causality. Backdoor adjustment is used as the conceptual guide for blocking confounding paths between causality and prediction. The framework instantiates three coordinated modules. First, a counterfactual detector uses an attention-style event importance term and a KL divergence term to construct a non-causality mask and trains with a contrastive loss. Second, an instrumental-variable guided module computes an IV score per candidate edge, selects top-quantile edges as genuine, builds multi-view subgraphs, and applies a robustness loss with alignment and separation terms. Third, an evolutionary orthogonal module projects event representations into static and dynamic components with an orthogonality step and an evolutionary loss that encourages temporal dependence of the dynamic component and temporal stability of the static component. Experiments on ICEWS14, ICEWS18, GDELT, WIKI, and YAGO show consistent gains and report average improvements over the runner-up on MRR and Hits. The paper includes a complexity analysis and a per-epoch runtime comparison on ICEWS14, and provides sensitivity studies that emphasize the effect of the neighbor count used to build the candidate graph.

### Strengths
1.	Event-level causal formalization
The SCM clearly separates non-causality and spurious causality from the static and dynamic components that should drive prediction, and articulates backdoor paths that motivate the design of the modules. 
2.	Concrete module designs with explicit masks and losses
The counterfactual detector fuses importance and distributional discrepancy to form a non-causality mask and optimizes a contrastive objective. The IV-guided module defines an IV score from observable features, selects top-quantile genuine edges, builds three propagation views, and regularizes with alignment and separation. The evolutionary module performs orthogonalization and trains with a two-part evolutionary loss. Each step has precise formulas. 
3.	Comprehensive empirical coverage
Results on five datasets with detailed tables, significance marks on ICEWS14 and ICEWS18, and a case study support the claims. The sensitivity section highlights that performance depends strongly on the neighbor count while being relatively insensitive to the history window length. 
4.	Transparent reporting of one-dataset runtime and end-to-end complexity
The complexity analysis enumerates per-module terms, and the runtime table compares HEDRA against two strong baselines on ICEWS14. 
5.	Clear writing and figures
The framework figure and mask definitions make the pipeline easy to follow, and the derivation for backdoor adjustment is provided in the appendix.

### Weaknesses
1.	IV validity and diagnostics are under-specified
The IV-guided module defines an IV score and uses an α-quantile rule to select genuine edges, then relies on robustness via multi-view propagation. The paper does not state explicit conditions under which the constructed IVs satisfy the usual exclusion and independence intuitions, nor does it include simple falsification checks such as placebo variables or outcome-on-IV tests. Adding a short subsection or appendix that names the identifying conditions and reports basic diagnostics would make the IV part more convincing. 
2.	Runtime and memory profiling are limited to one dataset
Table 8 reports per-epoch training and inference times on ICEWS14 only. Given the large E-squared and D-squared terms summarized in the complexity section, a resource profile on GDELT and WIKI or YAGO would help readers judge practical scalability. Module-wise wall-clock, peak memory, and parameter counts would be particularly useful. 
3.	Mask thresholds and calibration are not analyzed
The non-causality mask combines attention weights and KL divergence with fixed balancing coefficients and a small constant for stability. The IV mask then selects the top-α edges as genuine by a quantile rule. The paper does not present precision–recall calibration for the presumed genuine set under different α values or compare fixed quantiles with learned thresholds. Sensitivity and calibration of these gates would clarify robustness. 
4.	Per-loss training dynamics and attribution are not shown
The robustness loss has alignment and separation terms, and the evolutionary loss has dynamic and static terms, yet the paper does not provide per-term ablations or training curves that illustrate convergence and interactions with the main prediction loss. Adding curves for the contrastive, robustness, evolutionary, and total losses would strengthen the optimization narrative. 
5.	Definition and parity of evaluation protocols could be clearer in the main text
The use of MRR and Hits is standard, and the tables mark significance on ICEWS14 and ICEWS18. It would help to define time-aware evaluation choices in the main body and to extend significance reporting beyond ICEWS to all tables with multi-run results, so that parity with baselines is fully transparent. 
6.	GDELT comparison is affected by timeouts and OOM
The GDELT table shows several baselines timing out or running out of memory, which complicates margin interpretation. A breakdown by relation family and time granularity, plus a note on candidate-graph size and subgraph convolution memory use at fifteen-minute intervals, would contextualize the reported gains.

### Questions
1.	Instrumental variable assumptions and checks
What empirical evidence can you provide that the candidate IVs used to build Π satisfy the intended exclusion and independence intuitions in practice, and what failure modes arise when these conditions are weakened. A brief diagnostic report would be valuable. 
2.	Comprehensive resource profile
Can you report parameter counts, FLOPs if available, wall-clock time, GPU hours and peak memory for each dataset, and position HEDRA on a latency–MRR Pareto against at least two strong baselines. The ICEWS14 table is helpful, and similar measurements on GDELT would be informative. 
3.	Threshold sensitivity and calibration
How sensitive are results to the quantile level α in the IV mask and to the stability constant and temperature used in the non-causality mask. Do learned thresholds or adaptive calibration outperform fixed quantiles for precision of the presumed genuine set. 
4.	Training dynamics and loss interactions
Would you add training curves for the contrastive loss, the robustness loss with alignment and separation, the evolutionary loss, and the total objective, to document stability and to show how mass shifts from spurious edges to genuine ones during training. 
5.	Explanation of the small negative improvement in one metric
Table 1 shows a very small negative change on Hits at ten for ICEWS18 in the improvement row. Can you analyze which queries contribute to this dip and whether it is linked to rank-tail behavior or to the size of the candidate graph on this dataset.

### Soundness
3

### Presentation
3

### Contribution
4
