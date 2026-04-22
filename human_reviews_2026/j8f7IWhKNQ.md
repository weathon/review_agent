# Unsupervised Graph Neural Networks for Solving Combinatorial Optimization Problems by Iterative Solution Refinement

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Combinatorial optimization (CO) problems are crucial in various scientific and industrial applications. Recently, Graph Neural Networks (GNNs) have been proposed as a framework for addressing NP-hard CO problems, demonstrating high performance and nearly linear scalability. Current approaches utilize GNNs to directly predict solutions based on standard node features. However, such methods are prone to overfitting and tend to converge to suboptimal local minima of the energy landscape, resulting in low quality solutions. We introduce a novel optimization method leveraging the power of GNNs to efficiently process CO problems with Quadratic Unconstrained Binary Optimization (QUBO) formulation. Instead of predicting a solution from a fixed static set of node features, the method implies iterative improvement of the current solution with GNNs, using the predictions obtained at each step as new properties of graph vertices. We also propose certain modifications to the GNN architecture and a set of additional static properties that can further improve quality. The performance of the proposed algorithm has been evaluated on the canonical CO benchmark datasets. Results of experiments show that our method drastically surpasses existing learning-based approaches and is comparable to the state-of-the-art conventional heuristics, improving their scalability on large instances.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel GNN for solving several graph CO problems, max-cut, coloring, and max-independent set, reformulated as QUBO optimization problems. The key algorithmic novelty is the use of the iterative framework in the message passing layers. Results show a moderate improvement relative to baselines.

[Would have selected 5 if available. I am persuadable if you address my two concerns.]

### Strengths
Inclusion of dynamic features is novel

QUBO and the CO problems are well explained, and well motivated

Improvements over a fairly good set of baselines

Good ablation study

### Weaknesses
The iterative refinement step should be explained more clearly since this is the key algorithmic development of the paper

There is no discussion of computational cost. If the iterative method significantly increases the cost of the method, this may undercut the entire story. After all, slow exact solver exist. The name of the game is to get fast, pretty-good solvers.

Minor:

In Section 2.1, there appear to be some citep vs citet issues. (Check throughout.)
For undirected graphs it is better to write \{i,j\} rather than (i,j). (The latter is for digraphs.)
Extra space in |V | in Table 1 (twice).
Some P's and d's are incorrectly not in "math mode" in Appendix A.3

### Questions
Is there a good way to systematically reframe CO problems as QUBO problems (or to tell if this is possible for a given CO problem)? Or must this require "clever case-by-case thinking?"

### Soundness
3

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
The paper proposes an instance-wise, unsupervised GNN framework for QUBO-formulated combinatorial optimization. The main novelty is iterative refinement: the previous iteration’s final hidden state is fed to the first layer at the next iteration. Benchmarks are reported on Max-Cut, Graph Coloring, and MIS. The paper claims better performance than other learning-based methods and than SOTA heuristics on larger instances.

### Strengths
The manuscript appears to have undergone substantial editing phases.
The introduction of the method is clear. Benchmarks are broad across three CO tasks with comparisons against numerous other approaches. A simple modification to a conventional GNN yields a significant effect, which is appealing.
The claim that it works better on larger instances is interesting if substantiated.

### Weaknesses
There is an accumulation of benchmark results with different solver sets and unclear reasoning for the choice of baselines.
This makes it hard to understand the true potential of the method (see questions for details).

### Questions
1) In principle, iterating may require backpropagation through time when differentiating the loss. Could the author clarify: is the previous state explicitly detached and is no BPTT used? If so, it is important to state this clearly or justify why there is no need to consider BPTT.

2) Why does Table 1 not show results for stronger heuristics such as EO (if EO is already re-implemented)? The baseline heuristics in this table are quite weak (MFA and Greedy).

3) For Table 2: the apparent win is driven by a single instance (G70). On G14, G15, G22, G49, G50, and G55, the proposed method is at best comparable and often below BLS/TSHEA. Please explain why G70 improves while G55 does not. Is this due to graph-type differences (e.g., degree or density) or just size? It would help to report per-instance runtime for all methods on the same hardware.

4) The same comments apply to Tables 3 and 4.

5) The paper hints that the advantage occurs at large problem size. It would be clearer to show scaling versus problem size on an artificial ensemble of instances using a time-to-target metric, comparing to SOTA or strong heuristics. This would make a much stronger case that the proposed method works better on larger graphs.

6) It is stated that “Difusco and T2TCO could not scale to the ER-9000–11000 dataset,” but the proposed method could. Is this due to memory cost? If so, it would be interesting to provide more information about the memory cost of the proposed method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes QIGNN, a novel unsupervised Graph Neural Network (GNN) framework for solving Combinatorial Optimization (CO) problems. To address the issue of existing GNNs getting trapped in local optima by predicting solutions in a single pass from fixed node features, it introduces an iterative refinement mechanism that feeds the predictions from the previous iteration back as dynamic node features for the subsequent step. The authors claim their method significantly outperforms prior learning-based approaches on Max-Cut, Graph Coloring, and Maximum Independent Set (MIS) problems, and that on large-scale instances, it matches or surpasses state-of-the-art classical heuristics while being more computationally efficient.

### Strengths
1, The core idea of iterative refinement is not, in itself, a new concept within the optimization field. However, its specific application within an end-to-end, unsupervised GNN framework for QUBO problems—by feeding the entire output from a previous step back as a dynamic input feature for the next—demonstrates a reasonable degree of originality. Other architectural elements (e.g., the parallel SAGEConv layers in Section 3.2) are more akin to a clever combination of existing techniques. Overall, while not paradigm-shifting, the contribution is a practical idea that effectively enhances existing GNN-based CO solvers.

2, The proposed QIGNN consistently outperforms state-of-the-art learning-based methods (e.g., PI-GNN, RUN-CSP, various SL/RL methods) across a wide range of benchmarks on three distinct NP-hard problems: Max-Cut, Graph Coloring, and MIS (Tables 1, 2, 3, 4, 5, 6). This trend is particularly pronounced on large-scale graph instances.

3, The paper clearly demonstrates that the iterative refinement mechanism is the key driver of its performance gains. The ablation study (Figure 7), which shows a sharp performance degradation on the Max-Cut problem when this mechanism is removed, strongly supports the validity of the proposed design.

### Weaknesses
1, The authors state, "Since QIGNN is sensitive to initialization, we run multiple seeds and report the best result" (lines 268-269). Reporting the single best score from multiple runs, rather than the mean and standard deviation, can significantly inflate the method's perceived performance. This is a major flaw that severely harms reproducibility. It is unclear whether the baselines were subjected to the same "best-of-N" evaluation protocol (e.g., RUN-CSP is noted as best-of-64, line 314), casting doubt on the fairness of the comparison.

2, The paper uses different sets of baselines for the Max-Cut (Tables 1, 2), Graph Coloring (Tables 3, 4), and MIS (Tables 5, 6) problems. While including problem-specific SOTA heuristics is appropriate, several modern GNN-based methods are only compared on a subset of problems, making it difficult to form a holistic judgment of the proposed method's superiority.

3, While the authors acknowledge the method's sensitivity to initialization (line 268), they provide no analysis as to why this sensitivity arises or whether the iterative design might amplify it. This leaves open questions regarding the method's stability and practical reliability.

### Questions
1, Regarding the evaluation protocol of reporting the single best result from multiple runs (lines 268-269), I have a question about the statistical representation of the performance. For methods that are sensitive to initialization, a conventional approach is often to define a single 'evaluation trial' as the process of executing N runs and selecting the best outcome. The final reported metrics would then be the mean and standard deviation over several such independent trials to reflect the expected performance and its variance. Could you elaborate on the rationale for reporting a single best-of-N result rather than the statistics over multiple trials? This clarification would help in assessing the practical reliability and expected performance of the proposed method.

2, In Table 2, QIGNN underperforms BLS/TSHEA on smaller Gset instances but outperforms them on the largest instance, G70. What is the intuition behind this superior scaling behavior? Please provide an analysis of how the proposed iterative refinement mechanism interacts with graph properties like size or density to become more effective on larger-scale problems.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces QIGNN, an unsupervised iterative optimization framework for solving COPs formulated as QUBO. Compared to PIGNN, the proposed method incorporates an iterative dynamic feedback mechanism that helps the model escape local minima. Experiments on classic benchmarks such as MaxCut, MIS, and Graph Coloring demonstrate that QIGNN achieves superior solution quality over several learning-based approaches, while maintaining high efficiency on large-scale graphs.

### Strengths
1. The authors identify the limitations of existing methods and propose an iterative dynamic feedback mechanism to address them.
2. Experiments on classic benchmarks demonstrate the effectiveness of QIGNN in terms of solution quality and computational efficiency.

### Weaknesses
1. Although the paper introduces a dynamic iterative mechanism to enhance performance, it lacks a clear analysis of the mechanism’s complexity in the embedding and feature spaces, which may lead to potential computational overhead.
2. The paper does not include sufficient competitive baselines to fully validate the performance of QIGNN. Moreover, since QIGNN is trained and tested on the same instances, the authors should report the overall training and testing complexity together for a fair evaluation.
3. While the paper targets general COPs, the experiments are limited to only three classic benchmarks, which restricts the demonstration of its general applicability.

### Questions
1. Could you include some competitive baselines to better illustrate the performance of QIGNN, and incorporate these methods into the related work section (e.g., [1–3])?
2. Could you train QIGNN on a set of instances and directly apply it to test instances drawn from the same distribution to evaluate its generalization ability?
3. Although COPs can theoretically be transformed into QUBO formulations, different COPs exhibit varying levels of difficulty during transformation. Could you present additional problems that can be converted into QUBO form, like routing, packing problems,  and apply QIGNN to them to further demonstrate its effectiveness? Moreover, please report the transformation complexity along with the training and testing complexities. In addition, transforming some COPs into QUBO formulations requires introducing auxiliary variables, which causes the dynamic mechanism to incur increasing computational overhead.
4. Could you provide a detailed analysis of the computational complexity of QIGNN and compare it with the baselines?
5. Could you provide a more detailed analysis of the hyperparameters? For instance, the penalty coefficient that governs the transformation from COPs to QUBO formulations directly influences the problem structure and solution quality.

[1] Learning to Solve Quadratic Unconstrained Binary Optimization in a Classification Way. NeurIPS 2024.
[2] Controlling Continuous Relaxation for Combinatorial Optimization. NeurIPS 2024.
[3] Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics. ICLR 2025.

### Soundness
2

### Presentation
2

### Contribution
1
