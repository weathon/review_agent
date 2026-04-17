# Edge-wise topological divergence gaps: guiding search in combinatorial optimization

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
We introduce a topological feedback mechanism for the Travelling Salesman Problem (TSP) by analyzing the divergence between a tour and the minimum spanning tree (MST). Our key contribution is a canonical decomposition theorem, expressing the tour-MST gap as edge-wise topology-divergence gaps from the RTD-Lite barcode. Based on this, we develop a topological guidance for 2-opt and 3-opt heuristics that increases their performance.
We carry out experiments with fine-optimization of tours obtained from heatmap-based methods, TSPLIB, and random instances.
Experiments demonstrate the topology-guided optimization results in better performance and faster convergence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a topological framework for guiding search in combinatorial optimization, focusing on the Traveling Salesman Problem (TSP). The authors derive a canonical edge-wise decomposition between a TSP tour and its Minimum Spanning Tree (MST), expressing their cost gap as a sum of nonnegative “topological divergence” terms computed via the RTD-Lite barcode (a persistent homology construct). These divergence scores are then used to guide local search (2-opt/3-opt) and to shape reward signals in reinforcement learning (DQN-based) solvers.

### Strengths
The paper provides a fresh perspective on classical TSP analysis by introducing a canonical edge-wise decomposition that connects persistent homology with the MST–tour relationship. Unlike prior MST-based bounds or α-scores, this formulation produces a per-edge attribution of suboptimality derived from topological structure. This is a genuine conceptual contribution.

### Weaknesses
1.  Limited practical significance. As far as i understand: The proposed improvements are demonstrated only on simple 2-opt and 3-opt local search operators and a small DQN agent, Modern TSP solvers  employ far more complex and adaptive search strategies, making it unclear whether RTDL guidance would meaningfully improve real or industrial optimization pipelines, and also in data-driven fields, the search used in ATT-GCN or UTSP is much powerful than 2-opt, 3-opt.
2. MST-based reasoning is not new. The use of MSTs as lower bounds or structural references for TSP has a long history. The novelty here lies in the topological reinterpretation—the decomposition into edge-level bar lengths—not in the MST connection itself. This should be emphasized to avoid overstating originality.
3. Scalability and efficiency. RTDL computation introduces roughly 2–3× runtime overhead in some experiments.

**The most confusing aspect of the paper is the reinforcement learning component.**
The authors employ a basic Deep Q-Network (DQN) framework, which is now largely outdated for combinatorial optimization. Modern RL-based TSP solvers — such as POMO [Kwon et al., 2020], Attention Model [Kool et al., 2018] already achieve sub-1% optimality gaps on TSP-100 with inference times that are orders of magnitude faster. The paper does not compare against these established methods, making it unclear whether the proposed topological reward provides any meaningful improvement in a contemporary RL setting. Moreover, it is not well-motivated why the authors combine RL with simple 2-opt/3-opt post-processing rather than adopting the well-validated RL frameworks from prior work (e.g., POMO or the Attention Model), which already incorporate learned search or sampling strategies.
As a result, the RL results feel disconnected from current practice and do not convincingly demonstrate that the proposed topological feedback advances the state of the art in learning-based TSP solving.

### Questions
Could the RTDL-guided approach be integrated into deeper or variable-depth search frameworks?

How sensitive is the decomposition to small perturbations in edge weights (e.g., noisy or approximate distances)?

Can the topological divergence signal be used as a differentiable loss or regularizer in end-to-end neural solvers?

why we should use RL + 2/3 opt instead of using the RL style used in other works [Kwon et al., 2020] or [Kool 2018]?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a topological feedback mechanism for improving combinatorial optimization, specifically the Travelling Salesman Problem (TSP). The core idea is to measure the divergence between a candidate tour and the instance’s minimum spanning tree (MST) using edge-wise topological divergence gaps derived from the RTD-Lite barcode. Extensive experiments on TSPLIB, random Euclidean TSPs, and heatmap-based neural TSP solvers demonstrate improved performance—shorter tours, faster convergence, and greater stability—compared to standard 2-opt, 3-opt, and RL baselines.

### Strengths
1. The decomposition theorem provides a mathematically grounded link between graph topology and optimization performance.
2. Edge-wise topological divergence gives interpretable insight into why certain edges are suboptimal.

### Weaknesses
1. Although the authors claim broader applicability to combinatorial optimization, all experiments focus solely on TSP.
2. Although an ablation is included, the effect of RTDL frequency, edge selection granularity, and reward scaling could be analyzed more systematically.

### Questions
1. Does the bijection between tour and MST edges always hold uniquely? What happens for degenerate weight cases?
2. Could RTDL feedback be integrated during neural tour construction rather than post-hoc?

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
5

### Summary
This submission deals with different heuristics for approximating optimal TSP tours. The novel contribution is to assign to each edge in a given (non-optimal) tour a value that intuitively measures the contribution of the edge to the non-optimality of the tour. Then these values are used to guide the local search heuristics 2-Opt and 3-Opt and reinforcement learning. The values are computed via RTD-barcodes, a topological method from existing literature to compare weighted graphs.

The focus of the results is on experimental evaluation of the heuristics and not on theoretical considerations. First, the authors consider 2-Opt and 3-Opt and use the topological values to prioritize which local improvements are performed (i.e., first edges with large values are tried to be swapped out). They show on several instances that these guided local search algorithms are superior to arbitrary local search algorithms that simply perform the first local improvement found. In particular, the quality of the resulting solutions improves slightly both for 2-Opt and 3-Opt and the running time improves for 3-Opt. Then the authors consider large-scale instances, where the initial solutions are computed via heatmap approaches. Also for these, the topologically guided version of 2-Opt is slightly better in terms of the resulting tour length and better in terms of running time. Finally, the authors also consider Q-Learning. They demonstrate that incorporating the topological values into the learning process leads to a speed up.

### Strengths
Local search and other heuristics for the TSP are an interesting area of research that has been an active field for decades. Hence, the line of research is interesting and well-motivated. Using a topological measure to speed up these heuristics is interesting. It is also interesting that this measure can be used in different heuristics (local search and Q-learning).

### Weaknesses
Please see the text in the field "Questions" below for a detailed discussion of the weaknesses. In summary, the quality of the writing could be improved and more experiments are necessary in my opinion.

### Questions
- The proof of Theorem 1 is fairly confusing and could be written more precisely. In particular, it could be elaborated more why the constructed mapping is injective.
- I find the paragraph in lines 151 to 161 quite unclear: RTDL barcodes are briefly mentioned and I get that for any two undirected unweighted graphs with a bijection of the nodes such a barcode can be obtained via Algorithm 1 from (Tulchinskii et al., 2025). Then the barcode for the current tour T_tour and the complete graph G is computed. But how is this information used? Further below, the penalties are defined as p(e)=w(e)-w(psi(e)), which is related to Theorem 1 but, as far as I understand the notation, unrelated to the barcode computed before. Which values are assigned to the edges? If only the values p(e) are used then I do not understand the role of the barcodes at all because these values depend only on the procedure in Theorem 1.
- Throughout the paper it is said that random instances are generated but it is not specified what distributions are used.
- Is there any intuition why setting p(e_max) to the minimum of the positive p(e) values leads empirically to the best results? On the first glance, this is not the most intuitive choice.
- Some more experimental comparisons would be interesting/necessary. In particular, the following: 
  1) How do the proposed algorithms (like 2-Opt+RTDL) perform compared with state-of-the-art heuristics like Lin-Kernighan?
  2) Is there already literature on the effect of which local improvements are chosen in 2-Opt or 3-Opt? At least variants that find the first improvement have been compared with variants that find the largest improvement (see references below).
  3) The alpha-score of LKH-3 is mentioned in the submission. Couldn't one use this score directly to guide the local search instead of the values used in this submission? How does this compare to the method proposed in the submission?
  
Pierre Hansen, Nenad Mladenović,
First vs. best improvement: An empirical study,
Discrete Applied Mathematics,
Volume 154, Issue 5,
2006,

Daniel Aloise, Robin Moine, Celso C. Ribeiro, Jonathan Jalbert,
First-improvement or best-improvement? An in-depth local search computational study to elucidate a dominance claim,
European Journal of Operational Research,
Volume 326, Issue 3,
2025,

Typos:
Line 60: deliver -> delivers

Line 139: denotes -> denote

Line 203: if -> is

Line 237: either leave out "a" or tours -> tour

Line 241: note -> noted

Line 432: signal -> signals

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
2

### Summary
The paper defines an edge-wise topological divergence score that measures how much each tour edge deviates from the structure of the minimum spanning tree (MST). By prioritizing edges with higher divergence during local search and using the same scores as a shaping signal in reinforcement learning, the method achieves faster convergence and produces shorter tours across classical, neural, and RL-based TSP solvers.

### Strengths
- The proposed method can be integrated with numerical and learning based solvers
- Experimental results show that the proposed method provides faster convergence and shorter tours.

### Weaknesses
- The core optimization idea is not fundamentally new. The paper’s edge badness score closely parallels the α-score in LKH-3, which already uses MST/1-tree structure to rank edges for removal.
- The paper doesn't compare to state of the art TSP solvers like Concorde, LKH.
- The RL experiments are limited to relatively small problem sizes (100 nodes). It is unclear whether the RTDL-shaped reward remains stable and beneficial for larger tours.

### Questions
- What conceptual advantage does the topological interpretation provide over the classical α-score heuristic beyond reinterpretation?

### Soundness
3

### Presentation
3

### Contribution
3
