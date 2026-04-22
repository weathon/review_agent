# Learning to Solve Orienteering Problem with Time Windows and Variable Profits

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
The orienteering problem with time windows and variable profits (OPTWVP) is common in many real-world applications and involves continuous time variables. Current approaches fail to develop an efficient solver for this orienteering problem variant with discrete and continuous variables. In this paper, we propose a learning-based two-stage DEcoupled discrete-Continuous optimization with Service-time-guided Trajectory (DeCoST), which aims to effectively decouple the discrete and continuous decision variables in the OPTWVP problem, while enabling efficient and learnable coordination between them. In the first stage, a parallel decoding structure is employed to predict the path and the initial service time allocation. The second stage optimizes the service times through a linear programming (LP) formulation and provides a long-horizon learning of structure estimation. We rigorously prove the global optimality of the second-stage solution. Experiments on OPTWVP instances demonstrate that DeCoST outperforms both state-of-the-art constructive solvers and the latest meta-heuristic algorithms in terms of solution quality and computational efficiency, achieving up to 6.6x inference speedup on instances with fewer than 500 nodes. Moreover, the proposed framework is compatible with various constructive solvers and consistently enhances the solution quality for OPTWVP.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper develops a two-stage algorithm for solving the OPTWVP, where the first stage determines the order nodes are visited and the second stage determines how much time to spend at each node.  The first stage is solved heuristically using learned policy and the second stage is solved exactly using a linear programming model. Paper develops solid experimental evidence that the approach works well in practice.

### Strengths
1.	Solid experimental evidence that the approach is faster than state of the art methods, including global optimization and local optimization.
2.	Relatively clear exposition of the main ideas in the core paper, backed up by details in the supplemental material

### Weaknesses
1.	The discussion on the benchmark problems is limited, which it makes it unclear how narrow the results are.  In other words, what is different about each problem in the training/testing data set (price, travel time, etc.). If the benchmarks all share very similar structure, the results become less compelling, but are very compelling if the problem data has a lot of variance with the static space of a <TW, n> class. 
2.	The practical motivation could be explicit and direct for how problems of interest map into the OPTWVP model.  It may not be readily obvious to portions of the ICLR community not versed in the orienteering/vehicle routing community

Minor comment (not a weakness) - marzal/sebastia reference appears twice in the reference list.

### Questions
1.	Since the second stage is a linear program, what is the advantage of using Algorithm 1 over an off-the-shelf linear programming solver, e.g. Gurobi?
2.	Can you provide more details on the OPTWVP benchmarks? Are these standard benchmarks akin to the Solomon ones mentioned later in the paper? If so, can you provide a reference for them? Or are they generated in some fashion?  If so, what parameters are varied and how?
3.	How does the model behave when presented with a completely new benchmark class, e.g. one that is not part of the training data.  In other words, if the model was only trained on n = 50 and n = 100, how do the results on n = 500 change?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the Orienteering Problem with Time Windows and Variable Profits (OPTWVP), a routing problem where the solution may serve any subset of nodes, subject to time-windows constraints on when each node can be served and a constraint on the duration of route, maximizing the variable profit collected from serving the nodes. The paper proposes a computational approach that consists of two stages: first, a learned policy constructs an initial solution (route and initial service times); second, a polynomial-time algorithm finds optimal service times (given the route). The policy is learned using REINFORCE with baseline, and it is based on a graph-transformer architecture. The polynomial-time algorithm for improving the solution is proven to be optimal. The proposed approach is evaluated on benchmark problem instances, demonstrating that it outperforms baselines.

### Strengths
* The polynomial-time algorithm for finding service times for a given route is a nice contribution, and it is proven to be optimal.
* The combination of the RL policy with the polynomial-time algorithm is beneficial (while there exist approaches that use traditional optimization methods to improve solutions provided by NCO for similar problems, there is novelty in the specific combination, which decouples the routing from the scheduling).
* The numerical results are promising.

### Weaknesses
* The key contribution of the paper seems to be polynomial-time algorithm for finding service time. While this is an interesting contribution, ICLR might not be the best venue for publishing it since it is not a learning-based approach. A broader AI conference might be a better fit.
* The significance of this contribution is limited by the fact that the problem can also be solved by a straightforward LP, which also takes polynomial time. So this contribution really only seems important to communities that deeply care about this optimization problem.
* The novelty in the combination of the RL policy with the polynomial-time algorithm seems to be OPTWVP-specific (generally, the idea of combining NCO with traditional optimization is not novel, but specific combination based on fixing the route is), which might be of limited interest to ICLR.
* The discussion on lines 72 to 76 seems misleading. The references argue that transformers are ill-suited for continuous time series. However, this paper considers a discrete set of variables (scheduled times) that may have continuous values, which is very different, and which should not be an issue for transformers. The paper does not make it clear what the supposed issue with "hybrid decision variables" is. In the end, the proposed RL-based approach is a fairly straightforward one (graph transformer with discrete and continuous action heads); so there does not seems to be a "hybrid decision" challenge.

The presentation of the paper needs to be revised to improve its readability. The following is a non-exhaustive list of major and minor issues:
* The paper should provide at least an informal definition of OPTWVP in the introduction for readers who are not familiar with the problem. There is currently no definition or explanation.
* References should be \citep instead of \cite when the names of the authors are not used as subject or object.
* Figure 1(a) and its description are not particularly clear, e.g., it is not obvious how the problem is mapped to OPTWVP. Perhaps present a more straightforward, vehicle-routing example.
* The features and architectures of the learned policy are not described in the main text in adequate detail. Without reading the appendix, it is not clear how the Routing Decoder and STD work together. There is limited space in the main text, but these seem like important details.
* Notation is inconsistent. On line 199, $P$ and $C$ are introduced, but they are later typeset in calligraphic font $\mathcal{P}$ and $\mathcal{C}$. On line 204, subscript $5n$ is never explained (why 5? what is $n$?).
* The constraint on line 213 does not seem to make sense. What does the expectation of inequality mean? Later, it seems that the constraints are strict, so what is the point of the expectation?
* Also, most of the notation introduced in lines 199 to 214 is never used later, so what is the point of introducing them?
* On line 224, $G$ and $M$ are never introduced or explained. Typo for $\mathcal{G}$? Batch size $M$?
* Equation (1) would be simpler if it used the notation $R(\tau_i | G)$, which is introduced anyway.
* Lines 226 to 228 discuss coupling again. The importance of this is unclear; this "coupling" is just a mixed continuous-discrete action space.

### Questions
* Can you elaborate on the significance of the polynomial-time algorithm for finding service time considering that an LP can also solve this?
* What other routing problems could this algorithm be applied to? Or other contributions of the paper readily applicable to other routing problems?

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
This paper addresses the Orienteering Problem with Time Windows and Variable Profits (OPTWVP), which involves both discrete routing and continuous service-time decisions. The authors propose a two-stage DEcoupled discrete–Continuous optimization with Service-time-guided Trajectory (DeCoST) framework. In Stage 1, a parallel decoding structure predicts a route and initial service-time allocation; in Stage 2, service times are refined via a linear programming formulation, with theoretical guarantees of global optimality. The method reportedly achieves better solution quality and faster inference than state-of-the-art constructive and metaheuristic solvers across benchmark instances.

### Strengths
1. Clear and structured exposition: The paper is well written, easy to follow, and methodologically consistent. The decomposition into discrete and continuous components is intuitively appealing and technically well-motivated.
2. Solid engineering contribution: The combination of a neural constructive method with an LP-based continuous-time refinement is elegant and appears to yield computational gains.

### Weaknesses
1. Motivation for optimizing service times unclear: In standard formulations of the orienteering problem with time windows, service times are typically derived from route structure and scheduling constraints. It remains unclear why an explicit optimization of service times is required and how this impacts practical applicability. A full formal problem statement would clarify the modeling choices.

2. Limited methodological novelty: While the decoupling approach is well-implemented, the paradigm of combining discrete RL-style route construction with an optimization-based refinement layer is well established in recent literature (e.g., in ride-hailing and hybrid combinatorial optimization works). The contribution appears primarily incremental rather than conceptually new.

3. Missing related work: The paper overlooks several closely related lines of research, in particular the Neural Search and Decision Learning frameworks developed by Kevin Tierney and co-authors. Including and contrasting these would help position the contribution more accurately.

 4. Questionable experimental evidence.
 • Table 1 raises concerns: it is surprising that Gurobi fails to find feasible solutions within 24 hours — this requires verification or clarification.
 • Figure 3 lacks subcaptions, and key plots are underexplained.
 • The absence of comparisons on standard Solomon benchmarks limits interpretability of the numerical improvements.
 • It is unclear whether the reported gains (up to 6.6×) persist across varying problem sizes and constraint tightness.

 5. Marginal theoretical contribution: Theorem 1 and Algorithm 1 formalize a well-known fact — that continuous variables over a fixed route can be optimized efficiently via LP. The inclusion adds little beyond formal completeness and could be shortened.

 6. Overstated generality: While the authors claim compatibility with various solvers and broad applicability, evidence for transferability beyond OPTWVP is not provided.

### Questions
1. Is it necessary to tailor algorithm 1? A standard continuous optimization problem should also be solvable in polynomial time given a fixed visit sequence?
 2. Can the authors provide stronger justification or empirical evidence for Gurobi’s failure to find feasible solutions?
 3. Why is a comaprison to Gurobi skipped for the solomon benchmark?
 4. What are the key differences between DeCoST and hybrid approaches in prior works (e.g., neural combinatorial optimization combined with LP or MILP refinement)?
 5. Would the method scale similarly for larger instances or tighter time windows?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces DeCoST, a two-stage framework designed to solve hybrid combinatorial optimization problems that combine discrete routing and continuous time-dependent decisions. Specifically,  the discrete components refer to the routing decisions (i.e choosing which locations to visit and in what order) while the continuous components concern the time variables, such as how long to spend at each location and when to start service within the allowed time windows. The DeCoST algorithm first determines the discrete route and then optimizes the continuous service times through a linear program to maximize the overall reward. In the first stage, DeCoST employs a parallel decoder to generate a feasible route and an initial estimate of service times, enhanced by spatial encoding and feasibility masks that respect time-window constraints. A profit-weighted time allocation ratio (pTAR) is introduced in order to learn the trade-off between travel time and service time allocation for the initial service time assignment. In the second stage, given the fixed route, a linear programming (LP)-based Service Time Optimization (STO) algorithm computes globally optimal service times. 
Experiments show that DeCoST consistently outperforms both heuristic and neural combinatorial optimization (NCO) baselines. Overall, DeCoST offers a theoretically grounded, efficient, and generalizable approach to hybrid discrete-continuous optimization, bridging exact optimization and neural learning methods.

### Strengths
The authors propose a novel well-designed two-stage framework that cleanly separates discrete and continuous decision-making for the orienteering problem with time windows and variable profits. It also combines rigorous theoretical guarantees via the LP formulation in the second stage with extensive experiments supporting the results.

### Weaknesses
One weakness of the paper is that, unlike the second stage, the first-stage routing policy has no theoretical guarantee of optimality. It relies on reinforcement learning, which may converge to only locally optimal routes, so the overall solution quality depends on the effectiveness of this learned policy without any formal performance bound.

### Questions
Is it possible to provide any form of theoretical guarantee or performance bound for the first-stage routing policy?

### Soundness
3

### Presentation
3

### Contribution
2
