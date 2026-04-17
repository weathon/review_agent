# Constraint-Aware Discrete Black-Box Optimization Using Tensor Decomposition

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Discrete black-box optimization has been addressed using approaches such as Sequential Model-Based Optimization (SMBO), which aim to improve sample efficiency by fitting surrogate models that approximate a costly objective function over a discrete search space.
In many real-world problems, the set of feasible inputs such as valid parameter configurations in engineering design is often known in advance.
However, existing surrogate modeling techniques generally fail to capture feasibility constraints associated with such inputs.
In this paper, we propose a surrogate modeling approach based on tensor decomposition that captures the structure of discrete search spaces while directly integrating feasibility information.
To implement this approach, we formulate surrogate model training as a constrained polynomial optimization problem and solve a relaxed version of it.
Our experiments on both synthetic and real-world benchmarks, including a pressure vessel design task, demonstrate that the proposed method improves sample efficiency by effectively guiding the search away from infeasible regions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CA-TD, a constraint-aware approach to discrete black-box optimization that explicitly incorporates feasibility information into a tensor-decomposition-based surrogate model. By leveraging structured tensor representations and penalty-based constraint handling, the method demonstrates competitive sample efficiency across a range of discretized benchmark problems, including synthetic functions and combinatorial optimization tasks.

### Strengths
1. It introduces CA-TD, a constraint-aware tensor decomposition framework that explicitly embeds feasibility constraints into the surrogate model.
2. The paper is well-organized and clearly written.
3. The authors provide transparent discussion of practical considerations and thoughtful directions for future work.

### Weaknesses
1. Lack novelty: The paper’s approach to constraints—assigning infeasible points poor objective value and including a penalty term in the loss—is essentially the standard practice in many constrained black-box optimization settings. In fact, several commonly used benchmark problems are constructed exactly this way.
2. The empirical evaluation is narrow: the paper only considers four small-scale benchmark problems and omits comparisons with several established baselines for constrained discrete optimization.
3. The penalty-based constraint handling (assigning poor values to infeasible points and adding a loss penalty) is a standard technique used widely in both continuous and discrete optimization. Why develop and evaluate the method only in the discrete setting?
4. It’s unclear whether the points proposed by CA-TD during optimization are guaranteed to be feasible. If not, what fraction of the evaluated points actually satisfy the constraints? 
5. As noted in Section 3.4, CA-TD trains M separate surrogate models. How does this affect training time compared to baselines?
6. The paper does not release code.
7. There are minor typos. For example, on line 72, “known a priori” should be “known as a prior”.

### Questions
See Weaknesses.

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
3

### Summary
This paper proposes Constraint-Aware Tensor Decomposition (CA-TD) for discrete black-box optimization (BBO) under known input constraints. The method aims to integrate constraint information directly into the surrogate model rather than only in the acquisition function. The surrogate model is constructed using Tensor-Train (TT) decomposition and trained either by (i) solving a constrained polynomial optimization problem (HSDP) or (ii) a Penalty with Gradient-based optimization (PGRAD).
Experiments on small-scale benchmarks (Ackley, Warcraft, and Diabetes) illustrate that CA-TD achieves faster convergence and better sample efficiency than several baselines such as GP, TPE, PROTES, and NN+MILP.

### Strengths
- This paper introduces constraint integration into surrogate training, which is a promising direction to tackle constrained black-box optimization problems.
- The proposed approach, CA-TD, achieves faster convergence and better performance than other baselines in a wide range of small tasks.

### Weaknesses
- The assumption that infeasible inputs yield objective values $≥ \tau$ (Eq. 4, L183–188), where $\tau$ is the maximum objective value observed among feasible inputs, might be problematic. This treats infeasibility as high-cost supervision, which is often incorrect and misleading. The surrogate thus learns from false target values, biasing it toward suboptimal or distorted landscapes. This issue could even be aggravated when the penalty coefficient λ is large (as shown in Appendix D.1) or the optimization operates in a higher-dimensional space. Therefore, I think this assumption undermines the validity of the proposed approach.
- All experiments are conducted on extremely small discrete domains (e.g., Ackley functions on 3×3 or 5×5 grids, and the Diabetes dataset with only $5^8$). These settings fall far short of realistic discrete or combinatorial optimization scenarios, and thus fail to demonstrate genuine scalability. Consequently, the paper’s claims regarding “sample efficiency” and “scalability” are not substantiated by the presented evidence. 
- The presentation quality requires improvement; for instance, Table 2 is excessively large and extends beyond the page layout.

### Questions
- It is unclear whether $\tau$ and $y^*$ (line 231) refer to the same quantity or represent distinct variables. Please clarify their relationship to avoid confusion.
- The symbol T in Table 1 is not defined. It should be explicitly explained in the table caption to ensure clarity and self-containment.
- Minor points: The GitHub link in line 89 is empty.

Please also address these points mentioned in the Weaknesses section.

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
This paper proposes a method called Constraint-Aware Tensor Decomposition (CA-TD) for solving discrete optimization problems with constraints.
The idea is to make the surrogate model itself aware of constraints during training, instead of ignoring infeasible points until later.
The authors use a Tensor-Train (TT) model to represent the function efficiently and add a penalty loss that pushes the model to predict bad (high) values for infeasible points.
They also discuss two versions:

- HSDP, which gives theoretical guarantees using semidefinite programming but is slow.

- PGRAD, which uses gradient descent with a soft penalty and is fast and practical.

The experiments show that CA-TD works better than standard methods like GP, TPE, and PROTES on several benchmark problems.

### Strengths
- Clear motivation: infeasible regions waste time in discrete optimization, and the method directly handles this.

- Novel idea: combines tensor decomposition with constraint-aware learning.

- PGRAD is simple and can be trained with standard gradient-based tools.

### Weaknesses
- The PGRAD version has no formal theoretical guarantee of constraint satisfaction — it works empirically but not proven mathematically.

- The mathematical explanation of POP, SDP, and PGRAD is quite dense and difficult to follow for readers without background in polynomial optimization or semidefinite programming.

- The paper tests CA-TD on five benchmarks and four baselines, which is good but still limited for ICLR standards. Larger or more diverse benchmarks (e.g., higher-dimensional discrete tasks) would make the claims stronger.

### Questions
1- How often does PGRAD violate constraints in practice? Can you report the fraction or percentage of infeasible predictions during training?

2- How sensitive is the approach to the choice of the penalty parameter 
𝜆 ? Did you tune it separately for each benchmark, or was it fixed across experiments?

3- Is the feasibility threshold 𝜏 updated dynamically based on the current feasible region, or is it fixed once per task?

4- Could the proposed CA-TD framework be extended to handle mixed discrete-continuous optimization problems in the future?

### Soundness
3

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
3

### Summary
This paper addresses the problem of incorporating known input constraints into surrogate models for discrete black-box optimization (BBO). The authors propose CA-TD, a method that integrates feasibility information directly into the training of a low-rank tensor decomposition surrogate model, specifically using the Tensor-Train (TT) format. The core idea is to formulate the surrogate learning as a constrained Polynomial Optimization Problem (POP), solved either via (computationally heavy) hierarchical SDP relaxation (HSDP) or, more scalably, a gradient-based method with a penalty term (PGRAD). The paper demonstrates that this constraint-aware surrogate modeling improves sample efficiency over methods that handle constraints only at the acquisition function stage for a particular number of problem with small search space.

### Strengths
- The core idea of "constraint-aware" surrogate modeling is rather novel.
Previous work (e.g., NN+MILP) handle constraints in the acquisition step. But method PROTES, with which the comparison is made, also constructs ONE tensor during sampling, combining both the data tensor and the constraint tensor into a single object. In this sense, something similar has already been encountered. 

- The experimental section is a major strength. The authors take several benchmarks, including a diverse set of tasks (Ackley, Pressure Vessel, Warcraft, Diabetes) that test the method under various conditions. The comparison is extensive, pitting CA-TD against relevant baselines (GP, TPE, NN+MILP, PROTES) in both constrained and unconstrained variants.

- The paper is generally well-written, and the methodology is clearly explained.

### Weaknesses
- The main weakness is  in scalability to high dimensions (The Elephant in the Room): The paper correctly identifies the memory bottleneck of dense tensor representations as the primary limitation of the approach. While PGRAD improves computational scalability over HSDP, the fundamental issue of dimensionality (the "curse of dimensionality") remains largely unaddressed. The method is demonstrated on search spaces with a moderate number of dimensions (e.g., up to 8 for Diabetes). A more substantive discussion or preliminary experiments on potential pathways to mitigate this would significantly strengthen the paper.

- The additional comparative study in Appendix E is valuable but highlights a concern. The authors had to use "scaled-down versions" of the original NN+MILP tasks to make the comparison feasible. This inevitably raises the question: Is CA-TD's competitive performance a result of its intrinsic merit, or is it simply better suited to these smaller-scale problems? It would be interesting to see a direct comparison on at least one of the larger-scale problems from the NN+MILP paper, even if it requires significant engineering effort or approximation. 

- While the POP formulation is elegant, the paper lacks a theoretical analysis of the PGRAD relaxation. For instance, under what conditions does the penalty method recover the solution of the constrained problem? A discussion on the approximation guarantees or the impact of the relaxation on the surrogate's behavior in the feasible vs. infeasible regions would add considerable depth.

### Questions
- please, see the questions in Weakness

- could the authors elaborate on concrete strategies for scaling to higher-dimensional spaces (e.g., > 20 dimensions)? For example, have you considered or experimented with sparse tensor formats, or is this a fundamental limitation of the current CA-TD framework? 

- In the NN+MILP comparison (Appendix E), CA-TD wins on some tasks and loses on others. Can you provide an intuition for the problem characteristics (e.g., structure of the feasible set, smoothness of the objective) that make CA-TD particularly well-suited or ill-suited? 

- The threshold $\tau$ is set to the maximum observed feasible value. Was there any exploration of more adaptive or probabilistic strategies for setting τ? A poorly chosen τ could potentially misguide the surrogate, especially in early stages.

### Soundness
2

### Presentation
3

### Contribution
2
