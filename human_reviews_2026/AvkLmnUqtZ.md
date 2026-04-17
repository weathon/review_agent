# $\psi$DAG: Projected Stochastic Approximation Iteration for Linear DAG Structure Learning

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Learning the structure of Directed Acyclic Graphs (DAGs) presents a significant challenge due to the vast combinatorial search space of possible graphs, which scales exponentially with the number of nodes. Recent advancements have redefined this problem as a continuous optimization task by incorporating differentiable acyclicity constraints. These methods commonly rely on algebraic characterizations of DAGs, such as matrix exponentials, to enable the use of gradient-based optimization techniques. Despite these innovations, existing methods often face optimization difficulties due to the highly non-convex nature of DAG constraints and the per-iteration computational complexity. In this work, we present a novel framework for learning DAGs, employing a Stochastic Approximation approach integrated with Stochastic Gradient Descent (SGD)-based optimization techniques. Our framework introduces new projection methods tailored to efficiently enforce DAG constraints, ensuring that the algorithm converges to a feasible local minimum. With its low iteration complexity, the proposed method is well-suited for handling large-scale problems with improved computational efficiency. We demonstrate the effectiveness and scalability of our framework through comprehensive experimental evaluations, which confirm its superior performance across various settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ψDAG, a framework for Directed Acyclic Graph (DAG) structure learning based on Stochastic Approximation (SA) principles combined with projected stochastic gradient methods. The authors reformulate the DAG learning problem as a stochastic optimization task and introduce projection-based steps to enforce acyclicity efficiently. They claim scalability to large graphs (up to 10,000 nodes) and show empirical results comparing ψDAG with NOTEARS, GOLEM, NOCURL, and DAGMA.

### Strengths
1- The theoretical presentation is mostly sound; I did not find explicit errors in the mathematical derivations or proofs.


2- The idea of combining stochastic approximation (SA) with projection-based DAG constraints is straightforward and could be computationally appealing.


3- The authors provide comparisons with several standard baselines, including NOTEARS, GOLEM, and DAGMA.


4- The proposed algorithm is simple and might be useful in certain large-scale linear SEM settings.

### Weaknesses
1- The paper presents two main ideas:
(1) a new formulation of a stochastic loss function (Eq. 9) and an equivalent version based on the adjacency matrix (Eq. 10), and
(2) a strategy to reduce runtime by first learning the graph skeleton through a stochastic optimization procedure, then estimating a variable ordering via a heuristic projection function, and finally constructing the best DAG consistent with that ordering.

Regarding the first idea, the method largely reuses standard stochastic approximation updates and applies them to DAG learning with only minor modifications compared to existing approaches such as NOTEARS. For the second idea, the proposed projection mechanism is purely heuristic and lacks both theoretical justification and novelty relative to prior constrained optimization methods. Moreover, the paper overlooks several related approaches, such as BOSS, which first determines a variable ordering and then performs score-based structure learning with a BIC score and greedy search, as well as various hybrid methods that use constraint-based algorithms to obtain an initial DAG or skeleton for subsequent optimization. These omissions weaken the claimed novelty and contextual positioning of the work.

2- The reported numerical results are unconvincing. The claimed scalability (10,000 nodes) is not backed by verifiable or reproducible evidence. The GitHub link to the implementation (https://anonymous.4open.science/r/psiDAG-8F42) appears to be non-functional, which undermines reproducibility and transparency. 

3- The Sachs protein network results are particularly unconvincing: ψDAG reports a Structural Hamming Distance (SHD) of 14, which is not competitive. Simple score-based or constraint-based methods, such as PC or Hill-Climbing (HC), can achieve lower SHD on this dataset.

4- The paper suffers from weak writing quality, with numerous typographical errors and noticeable inconsistencies and discontinuities between paragraphs.

5- The method is limited to linear Bayesian (SEM) models only. This is a strong limitation, especially since many recent DAG-learning frameworks (e.g., nonlinear NOTEARS, DAG-GNN) address nonlinear dependencies.

### Questions
1- Can the proposed method be extended to handle categorical variables in the model, or is it limited to continuous (linear Gaussian) settings?

2- Is there any theoretical justification or proof showing that the proposed projection function can reliably recover the correct variable ordering?

3- Could the authors evaluate the method on additional benchmark datasets, such as ALARM, Link, and Munin, available at https://www.bnlearn.com/bnrepository/?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ψDAG, a novel framework for learning Directed Acyclic Graph (DAG) structures based on Stochastic Approximation (SA) integrated with SGD-based optimization. The key contributions include: (1) reformulating the discrete DAG learning problem as a stochastic optimization problem (Eq. 9), (2) introducing a three-stage algorithmic framework alternating between unconstrained optimization, projection onto DAG space, and constrained optimization, and (3) demonstrating scalability to graphs with up to 10,000 nodes.

### Strengths
1. The method successfully handles graphs with up to 10,000 nodes, significantly outperforming baselines (GOLEM/NOCURL fail beyond 3,000 nodes, DAGMA beyond 5,000).
2. The paper includes extensive experiments across multiple dimensions: graph types (ER, SF), densities (k=2,4,6), sizes (d=10 to 10,000), and noise distributions (Gaussian, Exponential, Gumbel), with both equal and non-equal variance settings.
3. The proposed projection method (Algorithm 3) has O(d²) complexity and avoids expensive matrix exponentials or log-determinant computations required by prior methods.

### Weaknesses
1. Theorem 8 claims convergence to a local minimum, but the proof is informal and hand-wavy. The two-case analysis doesn't rigorously establish convergence, and there's no guarantee the method won't cycle between subspaces indefinitely.
2. Section 3.1 states "which implies that the minimizer of (9) recovers the true DAG" but provides no rigorous proof. The algebraic manipulation ||x - W^⊤x|| = ||(I-W)(I-W*^⊤)^{-1}N_i|| doesn't obviously imply W=W* is the unique minimizer.
3. Unlike recent DAG learning theory (e.g., Gao et al. 2022b, Deng et al. 2023b), this paper provides no sample complexity bounds or finite-sample convergence rates.
4. Algorithm 3's greedy heuristic has no theoretical analysis. Why should minimizing row/column norms find a good topological ordering?
5. DAGMA "fails to converge" in numerous settings (protein dataset, r>15, d≥5000 for ER2). This is highly unusual given DAGMA's reported robustness in the original paper. Have implementations been verified against original codebases?
6. The paper uses a non-standard convergence criterion (f(x_k) - f(x*) ≤ 0.1·f(x*)) which requires knowing f(x*). How is this computed? Different methods may have different sensitivities to this threshold.
7. Only one small real dataset (d=11, n=853) is tested. For a method claiming scalability, evaluation on larger real networks is essential.
8. Why alternate between unconstrained optimization, projection, and constrained optimization? Why not just project once? The paper provides no theoretical or empirical justification for this design choice.
9. How are τ₁ and τ₂ chosen? How many outer iterations K are needed? What initialization is used? 
10. Lemmas 2, 5, 6 are basic set theory facts that add little value. The claim that D is a conic set (Lemma 2) is trivial since scaling edge weights doesn't create cycles.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the problem of learning graphical structures (Directed Acyclic Graphs; DAGs) from data, specifically targeting the typical linear model structure learning frameworks used in existing methods like NOTEARS (Zheng et al., NeurIPS 2018), GOLEM (Ng et al., NeurIPS 2020), and NOCURL (Yu et al., ICML 2021). The paper proposes a new algorithm called 𝜑-DAG. The key idea is a three-stage optimization process: unconstrained optimization → projection onto the DAG space → optimization that preserves the vertex order. Stochastic gradient methods are applied in both the first and third stages. This approach reduces the search space from all possible DAGs, an exponentially large space, to the space of topological orderings, enabling a more efficient algorithm suited for large-scale problems. Empirical results show that 𝜑-DAG outperforms existing methods like NOTEARS, GOLEM, and NOCURL in comparative experiments.

### Strengths
- This paper presents a very interesting and robust algorithm that tackles one of the key challenges in DAG structure learning, i.e. how to satisfy the strict DAG constraint, which has been a major difficulty for existing optimization methods. The proposed approach decomposes the problem into largely independent subproblems: optimization → projection onto the constraint-satisfying solution space → (weakly-)constrained optimization, forming a three-stage framework.
- In the final step (step 3) of this three-stage process, optimization must be performed while preserving the topological order of vertices determined in step 2. To handle this, the paper introduces a valid method based on computing the transitive closures and applying masking to enforce the order constraints within optimization.

- Previous methods that needed to handle strict constraints over an exponentially large DAG search space, but the proposed method with Algorithm 3 used in step 2 can now reduce the problem to enforcing strict constraints over node orderings in step 3. This effectively narrow down the search from the exponentially combinatorial space of DAGs to the permutation space of node orderings, which is smaller, and leads to a more efficient and logically grounded solution in the proposed method.

- Experimental results demonstrate empirical superiority in both accuracy and computational efficiency compared to representative existing methods, including NOTEARS (Zheng et al., NeurIPS 2018), GOLEM (Ng et al., NeurIPS 2020), and NOCURL (Yu et al., ICML 2021).
- Each of the points is thoroughly explained in the appendix, which is more than great.

### Weaknesses
- Since the idea of using stochastic gradient descent and the idea of using projection methods seem largely independent, an ablation study analyzing the contribution of each would make the work more informative. For example, in non-convex hard-constraint optimization problems like those with L0-norm penalties, projected gradient methods are a traditional approach. However, it’s well known that even simple gradient descent combined with projection often faces challenges in terms of convergence guarantees and optimality. These issues typically require additional techniques or relaxations, and simply replacing the gradient method with a stochastic version likely doesn’t resolve them on its own. 

- The SI provides detailed explanations, but a clearer discussion in the main text about how existing methods handle hard DAG constraints and how the proposed method takes a different approach would help readers better understand the contributions of this work. 

- The rationale for using stochastic gradient methods from the perspective of Stochastic Approximation (SA) vs. Sample Average Approximation (SAA) is valid as described, but a bit misleading. In practice, the difference between the sample average and the expectation is often handled with some form of regularization in SAA. So, while adopting SA may offer benefits in terms of computational efficiency or convergence stability, the current explanation suggesting it directly improves approximation accuracy may be a bit confusing. That said, recent work has shown that stochastic gradient methods can offer implicit regularization in complex optimization landscapes, so this could be useful to clarify the benefit with a more careful explanation.

- It seems the formulation reuses a standard setup, but since the objective function implicitly becomes quadratic, it would be helpful to include a brief explanation. When the noise term ( N ) is Gaussian, a quadratic objective is appropriate. However, for cases like exponential or Gumbel noise, as tested in the experiments, it’s not immediately clear whether the quadratic objective is still valid. One possible reason for the proposed method’s stability might be that, while the DAG constraint is complex, the error term’s quadratic form provides favorable properties, and this could be indirectly contributing to its effectiveness.

### Questions
I'm not a researcher in this specific area, so I'd like to ask a few clarification questions:

- From a general optimization design perspective, is the main takeaway that, in the case of DAG constraints, methods that explicitly account for graph structure are more effective, meaning that standard approaches like Projected Gradient Methods or Proximal Gradient Methods with convex relaxations are not sufficient?

- On p.15 of the appendix, are the objective functions in the existing methods optimized using techniques other than stochastic gradient methods? Since the objective function and the optimization strategy are conceptually separate, it seems that one could, in principle, apply stochastic gradient methods to equations (11), (12), and (13) by handling constraints via Lagrangian multipliers and using proximal methods for the L1 terms. Was this tested? Or is there some technical barrier that makes introducing stochastic gradient methods into this problem particularly challenging? A clearer explanation of this point would be appreciated.

- Both the existing formulations and the proposed method use a quadratic loss term, but is there no assumption of Gaussian noise? As shown on p.7, the noise ( N ) is tested not only with Gaussian noise, but also with Exponential and Gumbel noise. In those cases, wouldn’t a linear loss be more appropriate for Exponential noise, and a logistic loss for Gumbel noise? Wouldn't this part affect the entire paper?

- In lines 290–293 on page 6, it says, “if we know the true topological ordering ord(G∗), then we can recover the true DAG W∗ with high accuracy.” However, in practice, we don’t actually know the true topological ordering, and we can't guarantee that the node ordering obtained in Step 2 is the true one. So, should we understand this not as a theoretical guarantee of finding the exact solution, but rather as a claim that the search space has been reduced from the combinatorial DAG space to the permutation space of node orderings?

- Since the true topological ordering generally can't be identified, that means even when using the proposed method, if the ordering obtained in Step 2 isn't the true one, the solution won't converge to the correct one, as we can see in Figure 2, right? I’d appreciate it if you could provide some clarification, as the takeaway in Section 4.1 wasn’t entirely clear to me.

### Soundness
4

### Presentation
3

### Contribution
3
