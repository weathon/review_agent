# Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Learning-based methods for routing have gained significant attention in recent years, both in single-objective and multi-objective contexts. Yet, existing methods are unsuitable for routing on multigraphs, which feature multiple edges with distinct attributes between node pairs, despite their strong relevance in real-world scenarios. In this paper, we propose two graph neural network-based methods to address multi-objective routing on multigraphs. Our first approach operates directly on the multigraph by autoregressively selecting edges until a tour is completed. The second model, which is more scalable, first simplifies the multigraph via a learned pruning strategy and then performs autoregressive routing on the resulting simple graph. We evaluate both models empirically, across a wide range of problems and graph distributions, and demonstrate their competitive performance compared to strong heuristics and neural baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces GNN-based Multigraph Solvers (GMS)—two novel neural architectures for multi-objective (MO) routing on multigraphs, a setting previously unaddressed in neural combinatorial optimization. The first variant (GMS-EB) operates directly on the multigraph using edge-based autoregressive decoding, while the second (GMS-DH) prunes the multigraph via a learned edge-selection mechanism before performing node-based decoding. Both models employ preference-conditioned policies for MO optimization and are trained with a REINFORCE-style algorithm adapted for Chebyshev scalarization. Extensive experiments across multiple problem types (MOTSP, MOCVRP, and MGMOTSP) and graph distributions show that GMS achieves competitive or superior performance compared to strong neural and heuristic baselines, including LKH, MOEA/D, and a MatNet-based model.

### Strengths
**Originality**

- This is the first neural approach to handle multi-objective routing on multigraphs, an important and realistic extension beyond simple graphs. The paper clearly articulates why existing models (e.g., transformer-based TSP solvers) fail in this setting.

- The dual-decoder architecture (GMS-DH) is elegant, combining non-autoregressive edge pruning and autoregressive routing / node selection in a unified framework. The preference-conditioned hypernetwork design for multi-objective learning is well-justified and grounded in prior work (Lin et al., 2022; Navon et al., 2021).


**Quality & Clarity**
- The paper does an excellent job explaining why existing methods fail—namely, that standard encoders cannot represent multigraphs and node-based decoders are insufficient when edge choice is also required. The connection between multi-objective problems and the need for multigraphs is natural and well-argued.
- clear writing and figures describing the details of the neural policy
- The inclusion of training algorithms, detailed appendices, and reproducibility statement (code included) is commendable.


**Significance**
- Strong Empirical Validation: The experiments are comprehensive and well-designed.

- Strong Baselines: The comparison against state-of-the-art heuristics like LKH3 and a purpose-built neural baseline (MBM)  provides a robust benchmark.

- Crucial Ablation (GMS-DH PP): The "GMS-DH PP" variant, which uses a simple pre-pruning heuristic instead of the learned selection head, performs very poorly (Table 1). This convincingly demonstrates that the learned pruning mechanism of GMS-DH is far superior to a naive adaptation.

### Weaknesses
Both proposed variants face notable scalability issues. GMS-DH performs poorly on larger instances with many edges, suggesting that its hierarchical design does not scale effectively. Meanwhile, GMS-EB is computationally impractical for large graphs due to its excessive training and inference time, scaling poorly with the number of edges. Notably, the paper lacks a discussion or outlook on these scalability limitations and provides no guidance or future directions for addressing them.

### Questions
Why does performance for GMS-DH degrade on distributions with more edges (FLEX10, FIX10)? How do you think this can be mitigated in future work (without the excessive runtimes of GMS-EB)?

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
The paper presents GNN-based Multigraph Solver (GMS), the first neural architecture designed for multigraph and capable of handling multi objective routing problems on asymmetric graphs. Specifically, the paper considers two variants: one that is edge-based and one that is node-based and follows an edge pruning stage that converts the multigraph into a simple graph. Experimental results show the approach outperform the baselines for multi-objective TSP and multi objective CVRP problems, particularly on asymmetric graphs and multi graphs.

### Strengths
**Strengths:**
- The paper focuses on a relatively unexplored setting: multi-graph multi-objective neural routing optimization
- The proposed dual-head GMS seems novel
- Experiments show clear gains over the baselines

### Weaknesses
**Weaknesses:**
- The paper claims that existing techniques are not suitable for asymmetric and multigraph but this claim is not sufficiently substantiated. While approaches that rely solely on selecting the order of nodes are not sufficient, several recent approaches focus instead on selecting edges instead (for example, DIFUSCO and GREAT, both are cited in the paper). These approaches, at least in principle, can be used on asymmetric/multigraphs, and the paper does not compare to them, or explain why they are not suitable. There is no clear explanation why using them in a multigraph is not trivial, or what is the different between such approaches and the proposed edge-based approach (GMS-EB).

- The technical novelty needs to be more clearly specified in the paper: while it is clear that the combined setting of multigraph and multi objective has not been explored, the two are not intertwined in the proposed technical approach (that involves GMS, as well as a preference-condition MLP, and multi objective RL training) and the paper would benefit from making clear claims on the what is novel in each of the components, as well as the combined setting.

- I did not fully understand the claim on the scaling of edge-based GMS: why does it scale as O(MN^4)? If the auto-regressive decoder selects the next edge in the tour then it sounds like there are only MN edges to select from at each step and not MN^2?

- Missing multi-objective neural baselines: there are various works on multi-objective TSP that should be considered as baselines in the MO experiments; following is a subset of recent approaches. Most are not mentioned and all are not compared to:
	* Lin et al., 2022 that is mentioned in the paper is not being used.
	* Wu, Y., Fan, M., Cao, Z., Gao, R., Hou, Y., & Sartoretti, G. (2024, May). Collaborative deep reinforcement learning for solving multi-objective vehicle routing problems. In 23rd International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2024 (pp. 1956-1965). International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS).
	* Li, S., Wang, F., He, Q., & Wang, X. (2023). Deep reinforcement learning for multi-objective combinatorial optimization: A case study on multi-objective traveling salesman problem. Swarm and Evolutionary Computation, 83, 101398.
	* Wu, R., Wang, R., Hao, J., Wu, Q., Wang, P., & Niyato, D. (2024). Multiobjective vehicle routing optimization with time windows: A hybrid approach using deep reinforcement learning and nsga-ii. IEEE Transactions on Intelligent Transportation Systems.
	* Fan, M., Zhou, J., Zhang, Y., Wu, Y., Chen, J., & Sartoretti, G. A. (2025). Preference-Driven Multi-Objective Combinatorial Optimization with Conditional Computation. arXiv preprint arXiv:2506.08898.

- Not clear why Table 2 and Table 3 do not include all baselines

Minor: please fix the backward opening quote marks in all quotations in the paper.

### Questions
See my concerns and questions above

### Soundness
2

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
This work represents an extension of some NCO models to multi-objective optimization on multigraphs. In particular, the authors propose a novel method for selecting edges from the multigraph, rather than only nodes, which is typically the case in the neural combinatorial optimization (NCO) domain.

Two different autoregressive approaches are presented: the first operates directly on multigraphs, while the second performs pruning on the multigraph before executing routing on the simplified graph. The proposed method is evaluated on several benchmarks, including the multi-objective (MO) TSP, MO-CVRP, multigraph (MG) MO-TSP, and MG-MO-TSP with time windows.

### Strengths
1. The paper is well organized and written in a clear manner.
2. The proposed incorporation of edge selection into the decoding process represents a potentially novel contribution to the NCO domain.
3. The inclusion of two complementary models, one prioritizing speed and the other performance, is a reasonable design choice. However, the experimental results do not provide sufficient justification for maintaining both variants.

### Weaknesses
1. Contribution claims are overstated. The authors state that existing methods rely on transformers and can handle only problems defined on simple graphs, not beyond routing problems in the Euclidean settings. However, there are several works capable of encoding complex graphs, and the authors even cite some of them, although they claim that, to the best of their knowledge, such methods do not exist. Two of the cited works, (Kwon et al., 2021) and (Drakulic et al., 2025), can encode multigraphs and MatNet is even used in this work.

2. Related to the above, the novelty is limited. The main contribution lies in the implementation of the idea of autoregressively selecting edges from the hypergraph.

3. Limited evaluation. Although this work focuses on routing on multigraphs - which is its main contribution - it is evaluated on only one problem type (with two variants) of MGMOTSP. As the authors state, there is no prior work or benchmark datasets for this type of problem, which makes the list of baselines very limited. In this setting, it is difficult to assess the quality of the proposed solution. Relatedly, it is unclear what the broader impact of this work would be on the NCO community, which so far has not shown interest in studying these types of problems.

### Questions
1. I do not fully understand the experimental setup for testing MOTSP and MOCVRP with the Edge-based GMS. It seems that, during decoding, these problems do not require selecting edges from the hypergraph, only nodes. Could you please clarify how exactly this model is applied?

2. Is it possible to solve MOTSP and MOCVRP using classical NCO solvers such as the “vanilla” GREAT, MatNet, or GOAL models, adapted to the multi-objective setting? If so, I would like to see them included in Table 1 as baselines.

3. You provide a neural baseline based on the MatNet architecture, which is very nice, and demonstrate that your method is plug-and-play and compatible with other architectures. I would, however, like to see the results of the same experiments built upon other existing models that handle multigraphs.

4. Your experiments show that the edge-based model is only slightly better than the second model, but much slower, which raises questions about its purpose. It is approximately 10× slower while providing at most a 1% improvement in terms of the gap, which is quite negligible. Could it be that the selected benchmarks are too easy for both models, and that this is why the first model cannot demonstrate its full potential? Could you generate some more challenging problem instances and run tests of them?

5. The same question as above applies to GMS-DH PP — what is its purpose? For a small improvement in running time, the performance drops significantly.

### Soundness
3

### Presentation
4

### Contribution
3
