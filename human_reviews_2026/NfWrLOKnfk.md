# Learning from Algorithm Feedback: One-Shot SAT Solver Guidance with GNNs

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Boolean Satisfiability (SAT) solvers are foundational to computer science, yet their performance typically hinges on hand-crafted heuristics. This work introduces Reinforcement Learning from Algorithm Feedback (RLAF) as a paradigm for learning to guide SAT solver branching heuristics with Graph Neural Networks (GNNs). Central to our approach is a novel and generic mechanism for injecting inferred variable weights and polarities into the branching heuristics of existing SAT solvers. In a single forward pass, a GNN assigns these parameters to all variables. Casting this one-shot guidance as a reinforcement learning problem lets us train the GNN with off-the-shelf policy-gradient methods, such as GRPO, directly using the solver's computational cost as the sole reward signal. Extensive evaluations demonstrate that RLAF-trained policies significantly reduce the mean solve times of different base solvers across diverse SAT problem distributions, achieving more than a 2x speedup in some cases, while generalizing effectively to larger and harder problems after training. Notably, these policies consistently outperform expert-supervised approaches based on learning handcrafted weighting heuristics, offering a promising path towards data-driven heuristic design in combinatorial optimization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Reinforcement Learning from Algorithm Feedback (RLAF), a paradigm for training Graph Neural Networks to guide SAT solver branching heuristics through a one-shot mechanism where a single GNN forward pass assigns multiplicative variable weights w(x) and polarities p(x) that persistently influence all branching decisions. The approach modifies existing solvers by scaling their native scoring functions with learned weights, formulating weight selection as a single-step MDP trained with GRPO policy-gradient methods using only solver cost as reward, eliminating expert supervision or predefined properties like UNSAT cores. Training on easy instances with 2000 GRPO iterations sampling M=40 parameterizations across N=100 instances, the policies generalize to larger problems, achieving 69% runtime reduction on 400-variable 3SAT, 2× speedup on 600-vertex graph coloring, and 3× improvement on cryptographic instances. The approach consistently outperforms supervised baselines predicting handcrafted variable properties, with learned weights from different solvers showing high correlation (r=0.73-0.85), suggesting solver-agnostic structural understanding. The one-shot design incurs minimal overhead (0.02-0.1s) versus prior RL methods requiring one GNN pass per decision, enabling practical guidance throughout trajectories exceeding 100K decisions.

### Strengths
1. The paper is well-written with good illustration and easy to follow. The paper provides clear algorithmic descriptions (Algorithms 1-3) showing precisely how variable weights are injected into branching heuristics, with Figure 2 effectively illustrating the complete pipeline from graph representation through GNN processing to solver guidance. The background section carefully explains both SAT solving fundamentals and RL formulations, making the paper accessible to readers from either community. Experimental setup is thoroughly documented with comprehensive hyperparameter tables and detailed data generation procedures in the appendix.
2. The learned policies achieve substantial speedups across diverse problem distributions, including 69% runtime reduction on satisfiable 3SAT(400) instances and more than 2× acceleration on graph coloring and cryptographic problems, while generalizing to instances significantly larger than those seen during training. RLAF consistently outperforms supervised baselines that predict handcrafted heuristics like UNSAT cores and backbones, demonstrating that direct RL optimization yields more effective guidance. The one-shot design incurs negligible overhead (0.02-0.1 seconds) compared to solver runtime, making the approach practically viable.

### Weaknesses
This paper is in general a good paper to me, but I am not an expert in this field. Here I point out some suggestions which I hope could strengthen the paper further.

While the paper demonstrates strong results on three well-studied problem classes, the evaluation could be strengthened by including more diverse and realistic industrial SAT benchmarks from domains like hardware verification, software testing, or planning problems from SAT competitions. 

For example, the choice of LogNormal distribution for variable weights (Equation 2) with fixed standard deviation σ_w=0.1 is mentioned to perform best in "preliminary experiments," but no systematic comparison with alternative distributions (e.g., Gamma, truncated Normal) or analysis of sensitivity to σ_w is provided.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces Reinforcement Learning from Algorithm Feedback (RLAF), a paradigm for learning to guide SAT solver branching heuristics using Graph Neural Networks (GNNs). 

The authors demonstrate that RLAF-trained policies can reduce the mean solve time of various base solvers across a diverse set of SAT problems.

Many branching heuristics-based methods are implemented by starting with selecting a variable 
\hat{x} = argmax_x Score(x) 
that maximizes some function Score. 

The authors modify this to incorporate additional variable weights w for the given input formula:
\hat{x} = argmax_x w(x) · Score(x)

This way, we can inject prior knowledge of variable importance into the solver’s branching decisions.
In addition to the weights w(x), polarity p(x) can also be assigned to each variable x. When x is chosen as a decision variable, the polarity determines which value is assigned to x first.

The authors map input formula Phi to a graph representation (“Literal-Clause Graph”) that captures the instance’s structure. This graph is processed by a GNN that extracts structural information.
The output of the GNN [mu(x), rho(x)] is used to parameterize variable-wise weight and polarity distributions.
The models are trained by RL: the input formula Phi is the state, and a variable parameterization is the action. 
Once the action is taken, the environment transitions immediately to a terminal state, providing a reward R(Phi, W) = −Cost(Phi, W).

### Strengths
•	The paper is well written.
•	It investigates how RL-trained Graph Neural Networks (GNNs) can improve the branching heuristics of SAT solvers.
•	The authors modify existing DPLL-based backtracking SAT solvers to incorporate external variable weights into their branching heuristics.
•	The results demonstrate that, after training, the learned policies generalize well to significantly larger and more challenging problems.
•	I particularly appreciate the inclusion of the graph coloring and cryptographic experiments.

### Weaknesses
The work is demonstrated only on small-scale examples, with all training performed on machines equipped with a single multi-core CPU and one GPU. I wish the paper included more numerical experiments on larger-scale problems.

### Questions
•	How would the results change if other RL methods were used instead of GRPO?
•	How does the training time scale with problem size? How much faster is training on a GPU compared to a CPU?
•	What are the main challenges (if any) in developing a distributed version?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a novel method to combine (graph) neural networks and
a SAT solver. They use the network to predict an "importance" of each
variable, and then combine this value with SAT solver's inner variable
evaluation. The approach uses curriculum learning to train the NN with
RL (GRPO) on easy instance in one of three categories, and show time
improvement on harder problems, compared to the base solver as well as
other approaches.

### Strengths
* The use of a NN once before the solver doesn't slow down the solver,
   and directly relates to SOTA methods for SAT solving.
* The article is well written, including discussing the downsides

### Weaknesses
* Using RL forces the network to be trained only on small SAT instances,
   so it was not attempted on serious problems.

### Questions
-

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Reinforcement Learning from Algorithm Feedback (RLAF), a novel paradigm for accelerating SAT solvers by training a Graph Neural Network (GNN) to guide their branching heuristics. Departing from traditional approaches that rely on expert-crafted rules or supervised labels, RLAF formulates solver guidance as a one-shot reinforcement learning problem. Instead of intervening at each decision step, the GNN performs a single pass at the beginning of the search to assign an "importance score" (weight) and a "starting guess" (polarity) to each variable.
These predictions are integrated directly into the branching heuristics of existing solvers. The GNN policy is trained end-to-end using policy-gradient methods, such as GRPO, with the objective of minimizing the solver’s computational cost. The solver's own performance serves as the reward signal, creating a self-improving feedback loop that obviates the need for expert supervision. The experimental evaluation demonstrates consistent speedups across random 3-SAT, graph coloring, and cryptographic SAT problems. Notably, models trained on smaller instances generalize effectively to significantly larger and more difficult unseen problems.

### Strengths
-	The RLAF paradigm is a novel and effective concept. It learns by using the solver's own performance, i.e., its computational cost, as a reward signal. This help in eliminating the need for costly human labelling or hand-curated features (GNN can train itself).
-	The GNN runs just once per problem to generate a set of hints (weights and polarities) that guide the entire search. It resolves the computational bottleneck of prior RL/GNN methods that required per-decision GNN calls, which often made them slower than the original solver .
-	The proposed mechanism for feeding variable weights and polarities into existing branching heuristics is generic, which allows the learned policies to be applied to different base solvers.
-	Strong Empirical Results: The experimental evaluation is comprehensive and interesting. The authors demonstrate substantial speedups across 3 distinct SAT problem distributions and two different base solvers. It also generalizes well to larger and harder unseen instances. RLAF-trained policies also consistently outperforms over supervised approaches.
-	Clarity and Well-Structured Paper: The paper is clearly structured and well-written. Figures (especially Figs. 2–5) effectively convey key ideas. Mathematical formulations (e.g., the RL setup, Eqns. 6–10) are precise. Overall, its easy to follow.
-	Detailed appendices include architecture, solver modification details, and training algorithms. Reproducibility is well-handled.

### Weaknesses
The primary limitations of the work, which the authors are commendably upfront about, relate to the scalability of training and the depth of the methodological analysis. The most significant constraint is the computationally intensive nature of the RL training loop. This currently confines the approach to smaller problem instances, leaving its effectiveness on industrial-scale SAT problems as a crucial direction for future work.

Furthermore, the analysis would benefit from comprehensive ablation studies to better understand the sources of the performance gains. For instance, disentangling the separate impacts of weight and polarity guidance would clarify the mechanics of the learned heuristic. 

A sensitivity analysis regarding GNN depth and RL hyperparameters would also be a valuable addition, providing a more complete picture of the model's robustness and helping to guide future work in this direction.

### Questions
1.	How stable is RLAF training across random seeds? Does GRPO converge consistently?

2.	Did the authors attempt fine-tuning across mixed problem distributions (e.g., 3SAT + 3COL)? Given the "solver-agnostic structural properties" observed, could the authors comment on whether transfer learning (pre-training a GNN policy on a general SAT solver/distribution and then fine-tuning for specific instances or solvers) is a promising avenue for reducing the training cost and improving generalization further?

### Soundness
4

### Presentation
4

### Contribution
4
