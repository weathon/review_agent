# Efficient MIP-LP Gap Mitigation for Predict+Optimize

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
The Predict+Optimize (P+O) paradigm seeks to train prediction models for unknown parameters in optimization problems, with the goal of yielding good optimization solutions downstream. Prior works have proposed strategies for gradient computation in neural network training, when the downstream optimization is a linear program (LP). Yet, in face of mixed-integer linear programs (MIP), much prior work simply relax the MIP into an LP, resulting in sub-optimally trained predictors. The issue is particularly stark in the recent Two-Stage Predict+Optimize framework, where even the MIP constraints can contain uncertainty.

In this work, we propose a (shockingly) simple and fast approach for addressing the MIP-LP gap, and show that it yields essentially the same or more accuracy gains over a much slower method adapted from prior work. Concretely, for the latter, we adapt the approach of MIPaaL (Ferber et al. (2020)) and introduce cutting planes into the LP relaxation, before using LP-based gradient computation methods. Such adaptation is slow and requires some work for the new Two-Stage P+O setting, given the constantly-changing constraint predictions during training. We instead propose and advocate for a far simpler method: replace the relaxed-LP optimum in the LP-based gradient computation with the actual true MIP optimum, avoiding the repeated use of (slow) cutting plane MIP solvers in the slow method.
    
Experimental results on 3 benchmarks show that this simple strategy yields the same or more accuracy gain over the much slower cutting plane approach, and the conjunctive use of the two methods yields only minor further gains at the expense of vastly increased training time, sometimes by a whole order of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper examines how to address the MIP–LP gap in Predict+Optimize training, where integer constraints are typically relaxed. The authors propose a simple but effective fix: instead of using the LP relaxation solution when computing gradients, they directly use the true MIP optimum from neural solvers.

### Strengths
- The main idea is refreshingly simple. Instead of building a complex surrogate or adding a long chain of approximations, the authors just use the true MIP solution inside the training loop. The simplicity itself makes the message clear: sometimes the straightforward fix can go a long way.
- The paper reads clearly. The authors explain the setup step by step, which is not common in papers that integrate optimization and learning.
- The experiments are well organized and cover three different benchmarks. They show that this simple change can achieve accuracy on par with or slightly better than a more complex MIPaaL-style approach.

### Weaknesses
- The main concern is the limited novelty. The method is very close to MIPaaL and differs mostly in implementation. The paper does not introduce a new theoretical idea or learning principle.
- The use of the true MIP optimum is treated as a direct replacement, but the paper does not analyze why this replacement could contribute to more ideal gradients. The derivation in Section 3.2 depends on KKT conditions that are not valid for discrete problems, but this issue is not discussed.
- The use of KKT-based differentiation for a discrete problem is mathematically questionable. The KKT system assumes smoothness and convexity, which do not hold when integrality constraints are present. The paper admits this implicitly but does not discuss what kind of approximation the resulting gradient represents. This omission weakens the credibility of the method’s foundation.
- The method depends on solving a full MIP at every iteration. Although branch-and-bound solvers are faster than cutting-plane solvers, the runtime can still grow rapidly with problem size or solution variance. The paper does not analyze how training scales with instance complexity, nor does it discuss how solver randomness or multiple optimal solutions might affect gradient consistency.

### Questions
- When the discrete optimum changes between iterations, how is gradient discontinuity handled? Are there cases where training fails to converge?
- Since the gradient no longer comes from a differentiable mapping, what prevents the optimizer from following unstable directions?
- How does the proposed method handle multiple optimal MIP solutions? Which one is used for gradient computation, and does this choice affect the results?
- Can the authors clarify whether the gradients computed with integer solutions have any interpretation as subgradients of a convex envelope?
- How sensitive is the training process to solver tolerances and time limits? Could these parameters change the gradient signal significantly?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers differentiable optimization for MIPs with both a first stage decision and then a recourse step (to repair feasibility violations at some cost), and where predictions enter into the constraints as well as the objective. They propose that instead of adding cutting planes to tighten the LP relaxation before differentiation, we can instead use a simpler straight-through style approach of substituting the exact MIP optimum into the LP before differentiating.

### Strengths
Predict + optimize with uncertainty is a challenging setting that most previous work avoids (preferring to handle only uncertainty in the objective). The proposed approach is very simple to implement and benefits from the (many) settings where MIPs are solvable efficiently via other strategies like branch and bound but not via cutting planes. It seems like a compelling drop-in replacement.

### Weaknesses
The experimental validation is not particularly thorough, particularly since 2 of the 3 settings use the same prediction task (which is from a separate domain unrelated to the optimization problem). The paper would be more convincing with more substantively different data distributions. The experiments are also all done with respect to a single, somewhat strange, architecture (5 layer neural network with 16 neurons per layer). Since predict+optimize methods often struggle with difficult training dynamics, it would be worthwhile to see if the findings generalize to other architectures.

### Questions
A broader set of experimental results are the biggest place for potential improvement.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies decision-focused learning for MILP and addresses the mismatch between training on continuous LP relaxations and testing on discrete MIP problems. Thus, The authors adopt a two-stage predict–then–optimize formulation, where an initial decision based on predicted parameters is later adjusted once the true parameters—appearing in both the objective and the constraints—are revealed, with a linear penalty capturing the cost of revision.

Two training strategies are proposed: adding cutting planes to tighten LP relaxations, and using the true MIP optimum instead of the LP solution during gradient computation.

Experiments on Weighted Set Multi-Cover, 0–1 Knapsack, and Nurse Rostering demonstrate that the latter approach achieves comparable decision quality to more complex combinations while being significantly faster, providing a practical balance between accuracy and efficiency.

### Strengths
- **Originality:** The proposed strategy, which leverages non-stationary KKT information from integer optima as surrogate gradients, is simple yet novel in combining exact discrete solutions with differentiable LP-based learning.
- **Quality:** The experimental evaluation covers three benchmark problems of different structure and scale. Results consistently show that using the true MIP optimum improves decision quality over standard LP-based training. The comparisons against cutting-plane variants are fair and well-motivated.
- **Clarity:**  The paper is clearly written and easy to follow, with a logical presentation of the two-stage framework and experimental methodology.
- **Significance:** The work addresses a practically important limitation in the gap between LP relaxation and MILP. The proposed approach offers a pragmatic balance between accuracy and efficiency and could influence future work on learning-integrated optimization for MIPs and related combinatorial problems.

### Weaknesses
- **Missing related work:** The paper frames the problem as “predict + optimize” but overlooks a large body of existing research under the umbrella of decision-focused learning (DFL). Several prior works have already considered mixed-integer problems, such as SPO+ [1] and PFYL [2]. While many of these methods focus on predicting objective coefficients rather than constraint parameters, their formulations and theoretical insights remain directly relevant to the proposed approach. The absence of this discussion creates the impression that the contribution is more novel than it actually is, and the paper would benefit from explicitly positioning itself within the established DFL literature.
- **Lack of theoretical justification:** The proposed “MIP Optimum Replacement” strategy injects integer solutions into the KKT system to compute surrogate gradients. However, this process is inherently non-stationary and discontinuous, since integer optima do not satisfy the first-order optimality conditions of the continuous relaxation. The paper provides no theoretical justification, convergence argument, or stability analysis to support the validity of such gradients. This concern becomes even more critical when the LP relaxation and MIP solution differ substantially, as the resulting gradients may no longer reflect meaningful descent directions and could introduce significant bias or instability during training.
- **Strong Assumption Under Ad-hoc penalty:** The paper addresses infeasible predictions through an ad-hoc linear adjustment penalty proportional to the cost coefficients. This relies on a strong assumption that incorrect decisions can be revised and that the cost of revision scales directly with the original objective weights. While computationally convenient, such a design lacks theoretical or economic justification and is not a standard practice in optimization or decision analysis.
- **Limited experimental scale and generality:** The evaluation is restricted to small, toy-sized benchmarks with only tens of variables and constraints. For instance, the Knapsack and Nurse Rostering problems involve fewer than 50 decision variables, which limits the relevance of the reported improvements to realistic settings. As a result, it remains unclear whether the proposed method would remain stable or computationally viable on larger or more structured mixed-integer problems.

[1] Mandi, J., Stuckey, P. J., & Guns, T. (2020, April). Smart predict-and-optimize for hard combinatorial optimization problems. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 02, pp. 1603-1610).

[2] Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J. P., & Bach, F. (2020). Learning with differentiable pertubed optimizers. Advances in neural information processing systems, 33, 9508-9519.

### Questions
1. The proposed method uses integer optima within the KKT system to compute gradients, which are inherently non-stationary. Have the authors analyzed whether these surrogate gradients provide unbiased or stable updates in expectation?
2. Intuitively, the effectiveness of the MIP Optimum Replacement strategy should depend strongly on the tightness of the LP relaxation. Have the authors evaluated how the method behaves when the relaxation is loose and the integrality gap between the LP and MIP optima is large? In such cases, could the substituted KKT gradients become less informative or even detrimental to training performance?
3. The experiments are conducted on small-scale instances with fewer than 50 variables. Could the authors comment on the computational scalability of their approach for larger problems?
4. The paper employs CPLEX for cutting-plane experiments and Gurobi for MIP optimization. Have the authors verified that solver-specific behaviors do not bias the comparison in runtime or solution quality?

### Soundness
2

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
This paper proposes new methodologies for solving MIP in the predict+optimize paradigm beyond directly solving the relaxed continuous version of the optimization problem. 

Specifically, the authors first proposed a cutting-plane-based method that aims to solve the MIP directly but has high time complexity. To address this complexity issues, they also proposed a faster method that aims to solve the MIP objective directly. Additionally, the authors use numerical results to illustrate the performance and interpret some intuitions.

### Strengths
1. The paper is very clear in its setup, contributions, and results.
2. The paper proposed many new methods of solving MIP in the Predict+Optimize framework, instead of solving the relaxed problem directly.

### Weaknesses
1. The exact methodology of choosing cutting planes corresponding to Section 3.1, as well as the branch-and-bound methodology, seems not to be included in the paper. Could the authors elaborate more on that?

2. It seems the paper is purely experimental. Is there any theoretical guarantee on the performance? I am asking this question because both (1) the choice of cutting plane and (2) branch-and-bound choices may depend on the assumption of approximation in Stage 1, so any theoretical results will be greatly appreciated.

3. If this paper wants to purely focus on experimental results, only two problem instances might not be strong enough to answer the listed questions in the paper.

### Questions
1. Conceptually, what are the advantages of using the cutting-plane-based method?

2. If the paper advocates for the faster method, I wonder why the authors still propose the slow one and use it as a benchmark, given it is not an SOTA algorithm.

3. Please also see my questions in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
