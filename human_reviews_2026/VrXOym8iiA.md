# Efficiently Solving the TSP in One-Shot Without Post-Processing

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Neural approaches relying on autoregressive (AR) solution construction for solving the Travelling Salesman Problem (TSP) — a cornerstone challenge in Combinatorial Optimization (CO) — can be computationally expensive during both training and inference, as each step of the solution construction requires a pass through the neural network. While non-autoregressive (NAR) methods promise efficient solution generation in just a single pass through the network, prior works have often required post-processing search techniques such as Monte Carlo Tree Search (MCTS), active search, beam search and 2-opt to achieve near-optimal performance. However, these techniques not only inflate inference times but have also drawn criticism for potentially overshadowing the model's intrinsic capability and raised questions about the contribution of the neural component in the solution generation pipeline.

To address these concerns, we demonstrate that high-quality solutions can be generated in one-shot using the NAR approach by producing highly accurate heatmaps without requiring sophisticated post-processing techniques. During inference, we decode the tour by selecting the top two edges per node based on the heatmap — a process which can be performed in parallel. Unlike most prior NAR approaches, which depend on costly expert solutions for supervised learning, our approach is trained with Self-Improvement Learning (SIL) which is entirely unsupervised. Trained as a unified model across graph sizes, our approach achieves near-optimal results. In addition, we introduce new metrics quantifying edge prediction precision and tour validity in the NAR setting, which emphasize the model's raw capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a one-shot, non-autoregressive TSP approach that predicts a dense edge-score matrix and constructs tours by selecting the top-2 incident edges per node in parallel, avoiding search-based post-processing. It trains a single model across sizes via unsupervised self-improvement and introduces NAR-specific metrics (edge precision, tour validity); experiments report near-optimal tours from a single forward pass.

### Strengths
The paper’s motivation is clear: it addresses the confound where search/post-processing can dominate neural TSP results and latency, and instead commits to a one-shot, no-post-processing setup. This goal is stated upfront and carried consistently through the method and evaluation, making the claims easy to verify. 

The pipeline—predict an edge-score matrix, then decode with a simple per-node rule—is clearly written and easy to follow.

Finally, the reported metrics are aligned with the premise (e.g., edge precision, tour validity), which helps interpret outcomes as model ability rather than search effects.

### Weaknesses
1. The empirical results feels partial. Several experiment tables have missing entries; key baselines (Att-GCN, DIMES, DIFUSCO, SoftDist) only appear on TSP-500. This weakens claims about broad superiority.

2. SIL is positioned as label-free, but so are standard RL and unsupervised surrogates. It will be better to make it clear why SIL is chosed over other training methods, such as a small ablation that pits SIL against REINFORCE/rollout and a simple unsupervised baseline, with convergence speed, stability, and final gap, would clarify the trade-offs.

3. All experiments are on uniform Euclidean instances. Generalization remains unclear under distribution shift. It would strengthen the paper to add clustered, anisotropic, and mixed distributions (see the variety used in [1]) and to report cross-distribution transfer.

4. Finally, some design choices (notably the value of K) are justified qualitatively (“learns faster with more diverse batches”) but lack evidence. A brief appendix study sweeping K—with validity rate, gap distributions, and training dynamics—would ground the choice and reveal any speed–quality trade-offs.

[1] Fang et al., “INViT: a generalizable routing problem solver with invariant nested view transformer.”

### Questions
1. Could you elaborate on the rationale behind the Sequential Edge Selection(Algorithm 1) design? As presented, it remains a heuristic, and the underlying mechanism is not fully clear. In particular, it would help to explain the guiding principles for the selection order, tie-breaking, and interactions with the predicted scores, and why this specific procedure is preferable to other reasonable choices.

2. Could you more clearly explain the intended meaning of the new metrics (Optimal Edges, Valid Tours, Optimal Tours) and how they relate to final tour length? From the reported results, their values are not in a simple positive/negative correlation relationship with one another, nor with the final performance across different decoding methods. It would be helpful to articulate the expected relationships and the role of the decoder in mediating these metrics and the final tour cost.

### Soundness
2

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
4

### Summary
The paper proposes a new neural non-autoregressive approach for solving TSPs. Specifically, the proposed approach is utilizing data augmentation and self-improvement learning scheme to train the model without requiring expert solutions. The emphasis of the paper is on one-shot generation without utilizing post-processing techniques such as beam search, 2-opt, MCTS, etc. Experimental results show the model outperforms neural baselines on TSP problems of sizes ranging from 20 to 500.

### Strengths
- Novel neural NAR approach for TSP, in particular it proposes a new data augmentation and self-improvement training scheme.
- The experiments show the proposed approach outperforms the baselines in terms of performance gap.

### Weaknesses
- Limited technical novelty: despite improved performance, the paper makes a limited technical contributions. It seems the main innovation behind the proposed approach is the self-improvement procedure and the data augmentation.
- The experimental results have several major concerns:
	* As the paper did not run the baselines and only used results reported in previous papers, the main table of results is missing 80% of the results for three of the baselines including the strongest baseline of DIFUSCO. It is therefore not clear that the proposed model is the best across all problem sizes.
	* When focusing on the last column (TSP500) which is the only one with full results, the observed gains in standard greedy search compared to DIFUSCO are limited, and when considering the best setting for the proposed approach (SIL+SE), both LEHD and BQ-NCO are outperforming the proposed approach with LEHD seems faster (not clear why it is written as 0.3m instead in seconds other approaches) and BQ-NCO a bit slower. In the only setting considered that includes results for DIFUSCO (TSP500), it also outperforms the proposed approach in terms of # of optimal edges and valid tours
	* Results for DIFUSCO for both greedy and sampling are identical however in the original paper sampling performed better on TSP-500
	* Results are restricted to TSP and do not consider other popular problems such as CVRP, MIS, OP, etc. It is not clear how the observed patterns generalize beyond TSP.
	* No analysis on out-of-distribution performance (either sizes that were not considered in training or different data distributions considered in previous works).
	* It is not clear why SIL+SE is ~20 times slower than SIL+GE - what is the source of this slow down?
- This is not a critical point, but I am not sure I am convinced that it makes sense to exclude simple heuristics like 2-opt, while including constraints on the degree and merging connected components as part of the decoding.

### Questions
See my questions and concerns under weaknesses above

### Soundness
2

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
This paper proposes a non-autoregressive (NAR) neural approach to solving the Traveling Salesman Problem (TSP) in a single forward pass without relying on post-processing techniques such as beam search, MCTS, or 2-opt. The model outputs an edge probability heatmap from which the final tour is constructed by selecting the top two edges per node. The method is trained using Self-Improvement Learning (SIL), an unsupervised framework where the model iteratively improves through self-generated pseudo-labels, avoiding the need for expert solutions or reinforcement learning. It also introduces new evaluation metrics to assess the model’s raw predictive capability without post-processing. Experiments on standard TSP benchmarks show that the approach achieves near-optimal results on smaller graphs and competitive results on larger instances, while being significantly faster.

### Strengths
1. The one-shot, non-autoregressive design drastically reduces inference time compared to AR or diffusion-based methods.

2. Achieves near-optimal tours on small to medium graphs; valid and optimal tours are significantly higher than prior NAR methods.

### Weaknesses
1. Performance drops notably for large TSPs (e.g., 500 nodes), suggesting limitations in model capacity or generalization.
2. While the approach is efficient, its novelty primarily lies in combining existing ideas rather than introducing a fundamentally new model.
3. This paper claims about outperforming others “without post-processing” could be better qualified, as some baselines use additional refinement by design.

### Questions
1. How does SIL compare quantitatively to reinforcement learning or supervised approaches in terms of convergence speed and stability?
2. Could enforcing global connectivity constraints during decoding further improve valid tour rates?

### Soundness
2

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
This paper presents a non-autoregressive (NAR) approach for solving TSP in one-shot without post-processing. The key innovation is a two-selection scheme that extracts valid tours from heatmaps by selecting top two edges per node. Training uses Self-Improvement Learning, an unsupervised method with self-generated pseudo-labels. On TSP-20, it achieves 98.23% valid tours and 0.011% gap, outperforming Att-GCN. A unified model handles N=20 to N=500.

### Strengths
The two-selection scheme achieves 98.23% valid tours on TSP-20 versus Att-GCN's 72.49%, with mathematical derivation ensuring two edges per node. SIL training eliminates expert solution dependence while achieving near-optimal performance. Three new metrics—Optimal Edges, Valid Tours, Optimal Tours—provide fairer NAR assessment; prior methods achieve 0% valid tours while this maintains 70.24% on TSP-100. Unified cross-scale model handles N=20-500.

### Weaknesses
Performance degrades severely on large instances: TSP-500 shows only 0.78% valid tours and 6.034% gap versus LEHD's 1.560%, with no N≥1000 evaluation. Valid tour percentage drops dramatically, meaning 99.22% of TSP-500 instances fail. The paper lacks failure mode analysis and recovery mechanisms. While claiming "post-processing-free," tour completion remains necessary for large-scale deployment. Comparisons focus on NAR methods; broader AR comparisons beyond POMO would strengthen evaluation.

### Questions
Can the method scale to N≥1000, and what prevents evaluation at this scale?
What specific failure modes cause the dramatic valid tour drop on large instances?

### Soundness
3

### Presentation
4

### Contribution
2
