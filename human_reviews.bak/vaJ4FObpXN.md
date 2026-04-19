# Learning to Explore and Exploit with GNNs for Unsupervised Combinatorial Optimization

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6

## Abstract
Combinatorial optimization (CO) problems are pervasive
across various domains, but their NP-hard nature often necessitates problem-specific
heuristic algorithms. Recent advancements in deep learning have led to the development of learning-based heuristics, yet these approaches often struggle with limited search capabilities.
We introduce  Explore-and-Exploit GNN ($X^2$GNN, pronounced x-squared GNN), 
a novel unsupervised neural framework that combines exploration and exploitation for combinatorial search optimization:
i) Exploration - $X^2$GNN generates multiple  solutions simultaneously, promoting diversity in the search space; 
(ii) Exploitation - $X^2$GNN  employs neural stochastic iterative refinement to exploit partial existing solutions, guiding the search toward promising regions and helping escape local optima.
By balancing exploration and exploitation, $X^2$GNN achieves superior performance and generalization on several graph CO problems including Max Cut, Max Independent Set, and Max Clique. Notably, for large Max Clique problems, $X^2$GNN consistently generates solutions within 1.2\% of optimality, while other state-of-the-art learning-based approaches struggle to reach within 22\% of optimal. Moreover, $X^2$GNN consistently generates better solutions than Gurobi on large graphs for all three problems under reasonable time budgets. Furthermore, $X^2$GNN exhibits exceptional generalization capabilities. For the Maximum Independent Set problem, $X^2$GNN outperforms state-of-the-art methods even when trained on smaller or out-of-distribution graphs compared to the test set.  Our framework offers a more effective and flexible approach to neural combinatorial optimization, addressing a key challenge in the field and providing a promising direction for future research in learning-based heuristics for combinatorial optimization.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes GNN-based framework to solve several classic combinatorial optimization problems. The proposed approach behaves like a population-based heuristic method. Since extensive efforts have been devoted to the development of machine learning methods for addressing combinatorial optimization, I'm concerned about whether it can outperform the state-of-the-art algorithms.

### Strengths
- The proposed network generates $K$-coupled solutions and behaves like a population-based heuristic method. This is kind of novel.

### Weaknesses
I have a few concerns below.

- Line 245, the loss function includes constraint satisfaction and solution diversity. How to choose $\lambda_1$ and $\lambda_2$? Usually, the penalty method demonstrate very weak generalization capabilities. Hence, I personally think combining several terms in the loss function is not a good idea.

- For MCut and MIS comparison, a state-of-the-art algorithm [1] should be considered as baseline. This algorithm [1] is quite scalable and able to provide high-quality solutions to MCut and MIS.

- The proposed algorithm is not very scalable. For example, in Table 2 and 3, the computational time increases quickly with the problem size. MIS, MC and MCut are simple combinatorial optimization problems. Why not consider some large-sized instances (for example, Gset instances for MCut,  https://web.stanford.edu/~yyye/yyye/Gset)? How does the proposed algorithm perform?

[1] Schuetz, M.J., Brubaker, J.K. and Katzgraber, H.G., 2022. Combinatorial optimization with physics-inspired graph neural networks. Nature Machine Intelligence, 4(4), pp.367-377.

### Questions
See the weakness part.

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
4

### Summary
In this paper, the authors propose a framework that combines exploration and exploitation for combinatorial optimization (CO). The proposed framework explores the search space by generating a pool of solutions and exploits the promising ones through refinement. The model is based on Graph Isomorphism and Graph Attention Network, it outputs soft solutions that are heuristicly converted to hard solutions. The framework is applied and tested on three graph CO problems: the Maximum Clique Problem, the Maximum Independent Set Problem, and the Maximum Cut Problem.

### Strengths
- The idea of using K-coupled solutions for exploration and Iterative Stochastic Refinement for exploitation looks original and promising.
- The model shows excellent results; the proposed method outperforms state-of-the-art learning-based approaches not only on the training distribution but also in terms of generalization to larger problem sizes.

### Weaknesses
- Main concern: reproducibility seems impossible. There are no details about the implementation, only a brief description of the architecture  ('2L layers' of GIN and GAT), with no further details. There is no mention at all of the hyperparameters and the training process.
- The proposed framework is tailored to a small subclass of CO problems. It can be applied to simple graphs, defined by their adjacency matrices, and to problems where solutions can be represented as binary decisions for each node. This makes the framework inapplicable to other classes of CO problems, such as routing or scheduling, as well as to any graph problems with node or edge features.
- Although the paper claims that the method promotes solution diversity by generating multiple solutions simultaneously, in practice, the pool contains only two solutions. The method struggles when more diverse solutions are provided.

### Questions
1. Is it possible to apply this approach to other combinatorial optimization (CO) problems, such as routing or scheduling? Or to graphs with node/edge features (e.g., the Maximum Weighted Independent Set/Clique)?

2. What is the motivation for using Graph Isomorphism Networks (GIN) and Graph Attention Networks (GAT) and constructing the multilayer graph in the way described? There is no theoretical or empirical discussion justifying this choice. The ablation study shows that using more than two coupled layers degrades performance, which is really surprising This may suggest that GAT struggles to propagate information effectively across more than two solutions and/or that the proposed simple multilayer graph, which connects only copies of the same node in G, is not powerful enough to represent relations between solutions. Did you try using a more sophisticated multilayer graph and/or a different method to aggregate the data between solutions?

3. Following this, the multilayer graph for 2-coupled solutions (as used in the experiments) is very simple - it has 2N nodes and N edges (one edge per pair of corresponding original nodes). GAT is designed to aggregate information from many neighboring nodes, so using it on such a simple graph (in effect, it computes attention between just two nodes) seems odd and possibly unnecessary. Wouldn't a simple MLP achieve the same result?

4. In the discussion of experiments, much emphasis is placed on comparing results based on running time, but no details are provided on how the experiments were conducted. Were the solvers and models run on the same hardware? Were they tested under the same conditions (e.g., serial or parallel execution)? Neural networks can often solve multiple instances in parallel batches on GPUs, which might not be the case for solvers executed on CPUs (which are inherently much slower than GPUs by design). Claims about running times are only comparable if all methods are tested under similar conditions; otherwise, the comparison could be confusing. E.g. claim No. 4 "We additionally allow solvers a 30-minute time limit, which is at least 24 times longer than our longest-running model." could be misleading. By checking results, Gurobi is in most cases much faster than the proposed method (e.g. in Table 1 Gurobi vs. longest-running model for RB250 is 0.31s vs. 1.41s). 

5. All CO problems have simple greedy heuristics, such as choosing the node with the smallest degree for the MIS problem. Did you attempt to exploit this for the initialization of node features (e.g., assigning lower probabilities to high-degree nodes since they are less likely to be part of the solution)? This approach might provide a better initialization than random and could lead to faster learning.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents an explore-and-exploit Graph Neural Network (GNN) framework for combinatorial optimization (CO) problems. The key idea involves generating multiple solutions simultaneously to facilitate exploration while employing neural stochastic iterative refinement for exploitation. This approach effectively balances exploration and exploitation, leading to high-quality performance. Experiments conducted on three CO problems—namely, the maximum independent set, maximum clique, and maximum cut—demonstrate that the proposed algorithm outperforms learning-based algorithms in the literature.

### Strengths
1. The proposed framework utilizes Graph Neural Networks and unsupervised learning to effectively balance exploration and exploitation for combinatorial optimization problems.

2. Empirical results demonstrate high-quality optimization performance and goode generalization capabilities compared to learning-based baselines.

### Weaknesses
1. The encoder uses multiple original graphs as input; however, the rationale for connecting identical vertices across these graphs with edges is unclear. Table 5 only presents comparison results for the MIS and MC problems, why are the results for MCut not included? The description of the Drop Value is unclear, it would be better to provide a more detailed comparison of the results. Additionally, the drop value for K=2 shows a significant difference only in the context of MIS.

2. There is a lack of an ablation study on the design of the total loss function. The total loss function includes includes objective quality, constraint satisfaction, and solution diversity. It would be useful to analyze the results when the loss function includes only one of these components, such as solely objective quality, as well as the combination of objective quality and constraint satisfaction, and the combination of objective quality and solution diversity. This comparative analysis could provide insights into the impact of each loss component on the overall performance.

3. The result comparisons for each CO problem contain too few types of benchmarks. MC and MIS are closely related problems, and the instances tested in the experiments should remain consistent. Additionally, it would be beneficial to include more results for RB graphs and ER graphs. For the Max Cut problem, providing more results for BA graphs would also be helpful. Furthermore, testing the proposed algorithm on DIMACS and COLOR02 instances would demonstrate its generalization capabilities.

4. It would be beneficial to explicitly state the limitations of the proposed approach, for example, the scalability issues.

### Questions
1. During the iterative refinement process, do the local optimal solutions occur at intermediate steps, or do they only manifest in the final iteration?
2. How are the values ​​of C and T determined for each CO problem?

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
The paper proposes an iterative neural approach to search for solutions to graph combinatorial optimization (CO) problems. The approach is based on two phases: the generation of a diverse pool of solutions and their iterative improvement. The training is unsupervised and relies on a composite loss that combines the continuous relaxation of the CO problem objective, a penalization of the constraint violation and a diversity-encouraging term. The approach is evaluated on three graph CO problems and shows a very good performance compared to learning and non-learning based methods as well as a strong generalization performance.

### Strengths
* Novel unsupervised framework to generate solutions to graph CO problems 
* The framework components, in particular the architecture and the loss are generic and should apply to a variety of CO problems defined on unweighted graphs. 
* Original way to deal with several solutions for a given instance by constructing a "K-coupled graph" that allows to capture the whole collection of solutions to input to the refinement step.
* Strong and consistent performance on the three problems and nice generalization to larger instances
* The paper cites and compares to a number of relevant baselines in the experiments

### Weaknesses
* While the paper explains well how the hyperparameters (K, C, T, $\phi$) control the exploration/exploitation trade-off, the paper does not provide clear guidelines on how to choose good values for these hyperparmeters, except a grid-search. 
    * In Sec 5.5, the paper claims L253 "Different search strategies are needed for MC and MIS due to the different feasible regions. MC requires exploration to avoid local optima, and MIS requires exploitation to improve solutions." I don't understand this argument, can the authors elaborate on this? 
    * In general, for all CO problems and search methods, there is a risk of getting trapped in a local minima and a need for solution improvement. I can't see how one can decide beforehand what is more important for a given problem, especially since it may depend on the instances. 

* Comparing the run times between learning-based approaches which usually run on GPUs and OR solvers which run on CPUs is always delicate to interpret and gives a partial view of the efficiency of the methods. While there is no straightforward way to make the comparison more fair, it should at least be acknowledged.
   * In addition, the paper does not provide information on the machines on which the experiments were done -- this is especially important to appreciate the claims on the run times.

* The main paper contributions are to compute meaningful output probabilities on the nodes but then only a simple rule or a greedy method is applied to construct a feasible solution (See paragraph Converting Soft Solutions to Hard Solutions). 
   * Using a threshold of 0.5 seems arbitrary to chose whether or not a node is part of the solution. Did the author try other values? How one can choose this threshold for a new problem?
   * Given the probabilities, more sophisticated search methods can be applied such as beam search, Monte Carlo tree search or a least stochastic sampling (similarly to what is done when the model outputs heatmaps for example in the cited DIFUSCO method). 
   * Evaluating the proposed approach in combination with a stronger search technique, like the above, would be interesting and strengthen the claims. 
   * The question being: is the proposed approach useful only when a simple rule is used to construct the solutions or is it also helpful when combined with more sophisticated search?

### Questions
* L258: the paper states that the training is done in two stages. Are they done sequentially or alternatively? The arrows in Figure 1 towards the "loss block" made it confusing to me.

* In the Ablation section, when evaluating the impact of K, what was the value of C? In particular, it's important to evaluate the effect of K=1 with a large C, to demonstrate the value of the K-coupled solutions. 

* Remarks:
  * L245, L252 it may be misleading to state that the diversity is "imposed" through a loss, "encouraged" would be more clear.
  * It would be helpful to give an explanation of the corresponding equations L249 and L254 
  * Since at training, T=1, the authors could get rid of the t index in the description to lighten the notations

### Soundness
3

### Presentation
3

### Contribution
2
