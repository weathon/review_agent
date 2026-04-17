# Learning with Local Search MCMC Layers

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Integrating combinatorial optimization layers into neural networks has recently
attracted significant research interest. However, many existing approaches lack theoretical guarantees or fail to perform adequately when relying on inexact solvers. This is a critical limitation, as many operations research problems are NP-hard, often necessitating the use of neighborhood-based local search heuristics. In this paper, we introduce a theoretically-principled approach for learning with such inexact solvers. Inspired by the connection between simulated annealing and Metropolis-Hastings, we propose to transform problem-specific neighborhood systems used in local search heuristics into proposal distributions, implementing MCMC on the combinatorial space of feasible solutions. This allows us to construct differentiable, stochastic combinatorial layers and associated loss functions. Replacing an exact solver by a local search strongly reduces the computational burden of learning on many applications. We demonstrate our approach on a dynamic vehicle routing problem with time windows, a multi-dimensional knapsack problem, and on binary vector and k-subset prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper leverages the link between local search heuristics and MCMC methods to integrate local search heuristics as differentiable, stochastic layers in neural networks. Fenchel Young loss functions are used as practical stochastic gradient algorithms for both conditional and unconditional settings, and principled convergence guarantees are provided for the algorithm. Empirical results on the dynamic VRP with time windows and predicting binary vectors further validate the effectiveness of the proposed algorithm.

### Strengths
1. This paper is based on the very neat idea of linking local search heuristics and MCMC methods to allow local search heuristics as differentiable layers. 

2. Principled theoretical analysis are provided, together with convincing empirical results, making the paper very solid.

### Weaknesses
While the dynamic VRPTW experiment (5.1) seems solid, the predicting binary vector experiment (5.2) seems quite synthetic. I wonder if the method proposed in this paper can perform competitively for other combinatorial optimization as well (e.g. scheduling, graph coloring, max cut etc).

### Questions
1. The authors mention in the future work they plan to extend their framework to large neighborhood search algorithms. I’m quite curious about how they may be able to do it & if the current proposed framework is flexible enough to extend to other iterative combinatorial optimization algorithms (e.g. genetic search, tabu search etc). 

2. Can the authors discuss more about the limitation of Alg. 2 (computing the single individual ratio instead of summing the forward probabilities for all move types and the reverse probabilities for all move types)? In what realistic situation could this fails?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to integrate neural network training directly into combinatorial optimization processes. Specifically, it introduces a method that enhances local search by incorporating a differentiable layer based on MCMC, and presents the corresponding theoretical background and training procedure. Experiments are conducted on two tasks—VRPTW and binary vector prediction—and in the case of VRPTW, the proposed method outperforms HGS within limited computational budgets.

### Strengths
- The paper proposes a novel approach to improving local search performance using a differentiable MCMC layer, which is technically interesting.

- It demonstrates the ability to solve problems with complex constraints, such as VRPTW, within a short amount of time.

### Weaknesses
- Similar to traditional heuristic-based local search methods, the performance of the proposed approach heavily depends on how well the local search component is designed. In this sense, the paper does not improve prior NCO methods to better handle complex constraints; rather, it proposes a new type of meta-heuristic that utilizes a differentiable MCMC layer. Considering the ongoing trend of incorporating neural networks more actively in combinatorial optimization, the long-term impact of this work may be somewhat limited.

- Experiments are restricted to VRPTW and binary vector prediction, and for VRPTW, there are no comparison baselines beyond HGS. Incorporating commonly evaluated tasks (e.g. CVRP) and comparing against prior NCO or heuristic methods would enable a more objective assessment of the proposed approach.

- While the method outperforms HGS under strict time limits for VRPTW, its performance becomes inferior when more runtime is allowed. This suggests that although it can quickly generate reasonably good solutions, it may struggle to further refine solutions near optimality or escape local optima compared to traditional meta-heuristic approaches.

- The overall presentation and method complexity may be challenging for NCO researchers to follow, potentially limiting the number of researchers who may be interested in or willing to adopt the approach.

### Questions
None

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
3

### Summary
This paper proposes a theoretically principled framework that integrates local search heuristics as differentiable layers in neural networks.
By interpreting neighborhood systems used in combinatorial heuristics as proposal distributions, the authors construct MCMC-based layers that remain differentiable even when relying on inexact solvers.
Theoretical analysis establishes connections to Fenchel–Young losses, convergence guarantees under stochastic gradient estimation, and asymptotic properties.
Experiments on the Dynamic Vehicle Routing Problem with Time Windows (DVRPTW) and on a synthetic binary vector prediction task demonstrate the feasibility of the method.

### Strengths
1. Well-motivated problem formulation: the paper clearly identifies the key limitation of existing differentiable combinatorial optimization approaches and formulates the challenge of learning with inexact local-search-based oracles in a principled manner.
2. Theoretical guarantees: the proposed method is supported by rigorous theoretical guarantees, including convergence analyses and connections to Fenchel–Young losses. This provides confidence in the soundness and reliability of the approach, even when the optimization oracle is approximate.

### Weaknesses
1. Limited coverage of application tasks: although the proposed approach is claimed to be general, experiments are restricted to DVRPTW and a toy binary vector prediction task. As a result, the empirical validation of generality remains narrow.
2. Sparse comparison with existing methods: Table 1 summarizes a broad landscape of differentiable combinatorial optimization methods, yet the experiments compare only against perturbed optimizers (Berthet et al., 2020). While the authors may argue that other approaches rely on exact oracles, this justification should be stated explicitly, and experimental evidence that those methods fail under inexact solvers would strengthen the argument.
3. Limited interpretability of short-time performance: Table 3 shows that the proposed method excels under very tight time budgets (1–100 ms). However, the paper does not clearly explain how this property translates to broader usefulness—e.g., whether such gains are relevant in other tasks or larger-scale industrial settings.

### Questions
1. Could the authors include other additional benchmark results (e.g., TSP, scheduling, or knapsack) to test generality of the proposed method?
2. Could the authors include other baselines (in Table 1) to test generality of the proposed method?
3. It would be valuable to discuss whether the observed short-time efficiency carries over to other tasks with combinatorial optimization layer.

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
The paper introduces Local Search Markov Chain Monte Carlo (MCMC) Layers, a theoretically grounded framework for integrating inexact combinatorial optimisation solvers within neural networks as differentiable, stochastic components. It reinterprets local search as a form of MCMC sampling, transforming neighborhood-based moves into proposal distributions over combinatorial solution spaces. This connection enables traditional non-differentiable optimisation procedures to act as learnable stochastic layers that can provide meaningful gradient estimates for end-to-end model training. The authors demonstrate the theoretical soundness of the framework and validate it empirically on binary vector prediction and a large-scale dynamic vehicle routing problem with time windows.

### Strengths
- The core idea of the paper is strong, as it successfully combines two well-established concepts, local search and Markov Chain Monte Carlo, into a coherent framework for differentiable combinatorial optimisation. As well, it is both general and flexible as it supports multiple neighborhood systems and extends beyond prior works.
- The work offers rigorous analysis and solid theoretical guarantees for the proposed framework.
- Despite the technical depth, the paper is clear, well structured, and easy to follow with good explanations and well-presented algorithms and propositions.

### Weaknesses
My main concerns are with the empirical section:
- The authors provide empirical results on two cases (dynamic vehicle routing and binary vector prediction), which was effective to show the strength of the claims but yet lack extra experiments to prove the generalisation of the framework to other combinatorial problems aside from routing.
- The evaluation comparison on dynamic vehicles tasks relies solely on the EURO Meets NeurIPS 2022 PC-HGS–based baseline, which, though strong, does limit the depth of the empirical evaluation.

### Questions
- As mentioned in the weaknesses section, the experiments primarily focus on binary vector prediction and dynamic vehicle routing. It would be helpful if the authors could discuss or test how the framework might generalise to other combinatorial problems.
- For dynamic vehicles routing, could the authors include additional or more recent differentiable combinatorial optimisation baselines?
- The paper mentions that meaningful gradients can be obtained even with a single MCMC iteration. Could the authors clarify how stable these gradients remain in practice when training end-to-end with few iterations?
- On Table 3, the proposed method performs better under shorter time budgets but converges to similar results as the baseline when given more time. Could the authors please provide more analysis or intuition on what drives this behaviour?

### Soundness
3

### Presentation
3

### Contribution
2
