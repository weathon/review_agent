# Chance-constrained Flow Matching for High-Fidelity Constraint-aware Generation

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Generative models excel at synthesizing high-fidelity samples from complex data distributions, but they often violate hard constraints arising from physical laws or task specifications. A common remedy is to project intermediate samples onto the feasible set; however, repeated projection can distort the learned distribution and induce a mismatch with the data manifold. Thus, recent multi-stage procedures attempt to defer projection to clean samples during sampling, but they increase algorithmic complexity and accumulate errors across steps. This paper addresses these challenges by proposing a novel training-free method, Chance-constrained Flow Matching (CCFM), that integrates stochastic optimization into the sampling process, enabling effective enforcement of hard constraints while maintaining high-fidelity sample generation. Importantly, CCFM guarantees feasibility in the same manner as conventional repeated projection, yet, despite operating directly on noisy intermediate samples, it is theoretically equivalent to projecting onto the feasible set defined by clean samples. This yields a sampler that mitigates distributional distortion. Empirical experiments show that CCFM outperforms current state-of-the-art constrained generative models in modeling complex physical systems governed by partial differential equations and molecular docking problems, delivering higher feasibility and fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Chance-constrained Flow Matching algorithm(CCFM)  aimed for a generation of samples given a certain constraint condition.  CCFM introduces the constraint indirectly by applying the chance-modulated constraint, which is then shown to be mathematically convertible to deterministic constraint in the form of euclidean projection.    By repeatedly applying this constraint projection at the time of the inference, the algorithm produces samples  that satisfy the constraint. The paper shows the efficacy of the method  on PDE and Molecular Docking.

### Strengths
###1 
By "regularizing" the constraint via  probabilistic formulation, the method outperforms other generative models with constraint. 

###2 
The converstion of the probabilstic constraint to determistic projections are mathematically justified, along with the guarantee that the intermediate constraint would also make the finally generative samples feasible.

### Weaknesses
While the paper tackles an important problem and supports the algorithm is several theoretical guarantees,  the paper seems to lack several components necessary for clarity. Please see the Questions section. 

Minor points: In (6), $min_x$ instead of $min_y$?   In (11), is the constraint defined interms of $x_t'$?

### Questions
###1 What is the mathematical goal of the model? What is CCFM guaranteed to generate? For a moment I thought the goal was to generate $P_{target}(\cdot | constraint)$, but I guess that the model generates $\pi_{constraint}$ #  $P_{target}$, the pushforward of the Constraint Projection operator to the target distribution?   Is there any theoretical guarantee for what CCFM is producing at the end? 

###2 The probabistic parameter $\alpha$, after all, seems to be acting as the regularizer of the projection operator ; as the regularization strength, its scheduling seems vital in the training.  As a method, both a good heuristics and theoretical support seems warranted.  

###3 The Projection operator seems to require "if" statement for each sample in most of the applications discussed here.  Wouldn't this pose a computational risk? It is being claimed that CCFM achieves faster runtime than PCFM, what is the essential computational gain? 

###4 In section 4,  CCFM is motivated by the problem of previous works applying repeated projections to the noisy samples---however, the methods suffering from this problem do not seem to be compared against in the experimental section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper extends recent repeated projection schemes for training-free constrained generation via projection, to a chance-constrained stochastic optimization framework. This formal viewpoint leads to a practically-relevant algorithm with theoretical guarantees for relatively idealistic settings and positive experimental results on practically relevant problems with general complex constraint sets.

### Strengths
- The paper provides a new rigorous viewpoint on constrained generation by transporting standard understanding in stochastic optimization theory. I regard this connection and formalized problem as highly principled and valuable.

- The problem treated (i.e., constrained generation for flows/diffusion) is highly practically relevant and it is still arguably an open problem.

- The paper is well-structured, and well written.

- Both the theoretical analysis presented within sec. 4.3, and experiments within sec. 5 seem mostly convincing.

### Weaknesses
- (main concern) Sec. 4.2 explains that the presented opt. problem with probabilistic constraints is intractable and Prop. 1 reformulates it so that it becomes tractable via a deterministic reformulation. But it seems to me that Prop. 1 only tackles the case of linear/quadratic constraints, which arguably never capture typical constraints. I could not understand how the (arguably most common) case of non-convex constraints is managed. As of now, this seems to render the presented chance-constrained algorithmic machinery very limited.

- (main concern) The paper seems to propose an improved repeated-projection approach.  Nonetheless, reading the experiments sec. I could not clearly understand where to find a comparison with current repeated-projection schemes which do not leverage the chance-constrained formulation to determine the intermediate constraints. Where is it? Is it missing? I believe this experimental comparison would be essential to assess the gain of the core contribution of the paper, which is arguably the chance-constrained-based method as a way to improve repeated-projection schemes.

- I have fundamental doubts about the 'entire' repeated-projection approach. It seems to me that constraints are typically defined at the data level (e.g., g(x_0)). Due to the Markovian structure of the diffusion/flow process, a large amount of works [e.g., 1] interpret the sampling process as an MDP. This makes it possible to reduce constrained generation to constrained planning, a standard problem in RL/control, e.g. as done in [2] for the fine-tuning case. This standard RL/control viewpoint renders it possible to automatically derive intermediate deterministic (value) functions which maximization would lead to reward maximization at the last time-step. While the standard projection onto the constraint set C in previous repeated-projection works seems to me to make very little sense, this work arguably determines reasonable constraints for intermediate steps via the linear optimal transport viewpoint. But these constraints are probabilistic (and hence leading to intractable opt. problems), while methods exploiting the dynamic programming structure of the problem seem to lead to formulations corresponding to standard planning/RL tasks, which are arguably easier. Moreover, due to the control theoretic interpretation of classifier guidance [see 3], this logic extends also to classifier guidance given a proper weighting parameter of the 'penalty' term (which can be computed e.g. via Augmented Lagrangian schemes as in [2]). What is the authors opinion about this point?

- (writing, minor) I found the structure of presentation quite confusing. In particular, it seems to me that the chance-constrained viewpoint is leveraged as a way to develop algorithmic machinery to tackle a deterministic constraint problem. I would suggest to first introduce the formal problem on the data-level (i.e., last time-step constraint) and only afterwards introduce the chance-constrained framework implying probabilistic constraints at intermediate steps. Currently, I understood only late in the text that the data-level objective is deterministic rather than probabilistic. Since also a data-level probabilistic objective would make sense, I found this presentation quite misleading.

**References**

[1] Training Free Guided Flow Matching with Optimal Control, 2024.

[2] Constrained Molecular Generation via Sequential Flow Model Fine-Tuning, 2025.

[3] Variational Control for Guidance in Diffusion Models, 2025.

### Questions
I am happy to change my score if convincing answers to the posed questions are provided.

- Sec. 2 mentions that Classifier guidance (CG) schemes cannot provide feasibility guarantees. Could the authors clarify (formally) what they refer to with feasibility guarantee and why CG schemes cannot achieve it? 

- How is optimization carried out for non-linear/quadratic constraints not leading to deterministic reformulations of intermediate probabilistic opt. problems?

- See questions within the Weaknesses sec. of the review.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces Chance-Constrained Flow Matching (CCFM), a training-free modification of flow matching designed to enforce hard constraints during sampling. The method reformulates the standard projection step as a chance-constrained program, allowing constraints to be satisfied with high probability on noisy intermediate states while preserving fidelity to the clean target distribution. Theoretical results claim equivalence between intermediate and final projections under an optimal transport path assumption, and tractable Gaussian surrogates are derived for linear and quadratic constraints. Experiments are presented on two scientific domains: molecular docking and PDE solution generation, showing improved feasibility and efficiency over existing baselines.

### Strengths
### **Strengths**

- The idea of combining flow matching with chance-constrained programming is conceptually interesting and introduces a probabilistic treatment of feasibility within generative flows, which is novel and potentially useful for scientific modeling.
- The theoretical exposition is clear and internally consistent under its assumptions, and the connection between intermediate and final feasible sets is elegant (Theorem 4).
- The method is simple to implement, training-free, and shows promising qualitative results on molecular docking and PDE benchmarks, indicating some practical value.

### Weaknesses
### **Weaknesses**

- The experimental evaluation is limited in scope and depth. Only two scientific domains are tested, each with narrow baselines and minimal ablation. The probability scheduler, risk level, and theoretical assumptions (convexity, linearity) are not systematically validated. Since ICLR emphasizes experimental rigor and breadth, the current empirical evidence is insufficient to support the paper’s general claims.
- The claimed equivalence between noisy and clean projections relies on restrictive assumptions that are unlikely to hold in realistic nonconvex cases like docking, yet no empirical study quantifies the resulting approximation error or its impact on feasibility.
- The docking and PDE comparisons lack strong baselines and statistical rigor. Metrics are presented without confidence intervals or robustness analyses, and the reported efficiency gains are small or inconsistent across tasks, making it difficult to assess the true advantage of CCFM.

### Questions
### **Questions**

Besides the implied ones based on the weaknesses above:

- Do the authors plan to test on other classes of constraints to demonstrate broader generality? 
- Can the authors provide sensitivity analyses for the probability scheduler and risk level across both domains?

### Soundness
2

### Presentation
3

### Contribution
2
