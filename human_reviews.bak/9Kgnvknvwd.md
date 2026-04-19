# A First-Order Multi-Gradient Algorithm for Multi-Objective Bi-Level Optimization

- Decision: Reject
- Scores: 6, 5, 3, 6

## Abstract
In this paper, we study the Multi-Objective Bi-Level Optimization (MOBLO) problem, where the upper-level subproblem is a multi-objective optimization problem and the lower-level subproblem is for scalar optimization. Existing gradient-based MOBLO algorithms need to compute the Hessian matrix, causing the computational inefficient problem. To address this, we propose an efficient first-order multi-gradient method for MOBLO, called FORUM. Specifically, we reformulate MOBLO problems as a constrained multi-objective optimization (MOO) problem via the value-function approach. Then we propose a novel multi-gradient aggregation method to solve the challenging constrained MOO problem. Theoretically, we provide the complexity analysis to show the efficiency of the proposed method and a non-asymptotic convergence result. Empirically, extensive experiments demonstrate the effectiveness and efficiency of the proposed FORUM method in different learning problems. In particular, it achieves state-of-the-art performance on three multi-task learning benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the Multi-Objective Bi-Level Optimization (MOBLO) is studied, and an efficient first-order multi-gradient method for MOBLO, called FORUM, is proposed. The proposed method first reformulates MOBLO as an equivalent constrained multi-objective problem, then a novel multi-gradient aggregation method to solve the constrained multi-objective problem.

### Strengths
1. The proposed method combines the value-function-based approach and multi-gradient method, which is novel in the multiobjective bilevel optimization problems.

2. The writing of this work is good, and the logic of the proposed method is clear.

### Weaknesses
I believe this work is solid and good, however, I have some concerns as follows.

1. In the proposed method, an additional optimization problem is required to solve every iteration, i.e., Eq. (11). Thus the proposed method seems inefficient since it is a nested-loop algorithm.

2. I suggest the authors add a table to compare the differences between the proposed methods and existing MOBLO methods (i.e., MOML and MoCo) to clearly show the advantages of the proposed method.

### Questions
See Weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a first-order multi-gradient method tailored for the multi-objective bi-level optimization (MOBLO) problem. More precisely, the authors first reformulate the MOBLO problem as an equivalent single-level constrained multi-objective optimization problem using a value-function-based approach. Subsequently, they integrate the BOME method, designed for single-objective bi-level optimization, with the MGDA for multi-objective optimization (MOO) to address this equivalent problem. The authors provide convergence analysis and numerical results.

### Strengths
The topic of multi-objective bi-level optimization is important. 

Convergence analysis is provided for the proposed method under some assumptions, e.g., the lower level problem is strongly convex.

Numerical validation is presented for the proposed method.

### Weaknesses
1. The proposed algorithm is a straightforward combination of two existing methods, and the accompanying analysis appears rather standard. Consequently, the technical innovation compared to prior work upon which this study is based is limited.

2. The convergence result of the proposed method lacks persuasiveness. As pointed out by the authors, the constraint $q(z) \le 0$ in the reformulated problem (3) is ill-posed, rendering the KKT stationary condition not a necessary condition for problem (3) solutions. Therefore, the utilization of the $\mathcal{K}(z)$ measure for the convergence of the proposed method in the main convergence theorem (Theorem 4.3) is inappropriate. In contrast, the MoCo method by Fernando et al. (2023) employs hyper-gradients under the same strong convexity in the LL problem to characterize convergence and establish convergence to Pareto stationarity.

3. The analysis of the proposed algorithm is confined to a deterministic setting, which may restrict its applicability given that the motivating applications are mostly in the stochastic setting.

4. The strong convexity of the LL problem appears to play a pivotal role in the convergence analysis, which, in turn, constrains the applicability of the proposed method. Notably, the MOML method introduced by Ye et al. (2022) does not necessitate such an assumption.

### Questions
My questions are listed in the “Weaknesses” part.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studied a multi-objective optimization problem where each objective has a bilevel optimization structure. Each upper-level objective is evaluated at the minimum of the same lower-level problem. The authors utilize the idea from value-function-based method to solving the bilevel problem as well as a momentum update idea in Zhou et al., 2022 in the multi-objective algorithm. A convergence analysis is provided for the derived algorithm. Several experiments on hyper-cleaning, multi-task learning over Office-31, NYUv2, QM9 datasets are provided.

### Strengths
1.	Overall, the studied problem is timely. Bilevel optimization and multi-objective optimization have wide applications in practice. This work uses a first-order idea from bilevel literature in MOO, and seems to work well in experiments.

2.	Experiments seem to support that the proposed method can work well in some datasets.

### Weaknesses
1.	However, I have quite a few concerns regarding the analysis and the novelty. First, to deal with the reformulated constraint in the value-function based method, the authors use a dot product between $d$ and the gradient of the constraint and make it less or equal to $-\phi_k$. However, how to pick this $\phi_k$ in practice and in theory? Also, the final convergence criterion seems questionable. For example, the authors use the measure of KKT stationary condition to as convergence criterion. How does this condition correlate with Parato stationarity? How fast does the parameter $v_k$ decrease to 0 in this criterion? All these questions are not well explained in this work. 
2.	The algorithm has some unclear parts. For example, the authors use the idea of momentum update on $\lambda$ update. This step is originally proposed in Zhou et al., 2022 (the authors should mention about this). However, there is characterization on the distance between the true variable $\lambda_k$ and the surrogate $\titilde \lambda_k$? This is important, because what you need to use is $\lambda_k$ rather than surrogate $\titilde \lambda_k$ in the algorithm. 
3.	The algorithm is deterministic without data sampling, but in the experiments it seems data sampling is used. I am wondering if it is possible to extend the algorithm and analysis to the more practice stochastic setting? If not, what are the challenges? Some recent progresses on stochastic MOO may be helpful here (some of them are missing in this work). 

[1] Suyun Liu and Luis Nunes Vicente. The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning. 

[2] Lisha Chen, Heshan Fernando, Yiming Ying, and Tianyi Chen. Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance.

[3] Heshan Devaka Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen. Mitigating gradient bias in multi-objective learning: A provably convergent approach.

[4] Peiyao Xiao, Hao Ban, and Kaiyi Ji. Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms.

4.	The analysis assumes the upper-level function $F_i$ is bounded. However, this has not been made in the aforementioned [1,2,3,4] works. More clarifications should be provided. If there are any special challenges making this assumption necessary?  Also, the assumption $q(z_k)<B$ is a strong assumption that has not been made in previous bilevel and MOO literatures, because bounded function value and strong-convexity cannot be made simultaneously.

### Questions
Overall, this work studied an interesting and important problem. However, it has quite a few questions and problems to be solved. However, I am open to increase my score given the authors’ response. See my questions in the weakness part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel first-order multi-gradient algorithm called FORUM for solving multi-objective bi-level optimization problems. The proposed method achieves state-of-the-art performance on three multi-task learning benchmark datasets. The paper also provides a reformulation of the MOBLO problem as a constrained multi-objective optimization problem using the value-function-based approach. The proposed method is evaluated through empirical experiments, which demonstrate its effectiveness and efficiency.

### Strengths
1. Novelty: The paper proposes a new method, FORUM, for solving multi-objective bi-level optimization problems that is based on a first-order multi-gradient algorithm. This is a novel approach that addresses the computational inefficiency of existing gradient-based methods that require computing the Hessian matrix.

2. Efficiency: The proposed FORUM algorithm is shown to be more efficient than existing methods based on complexity analysis. The paper provides a theoretical analysis of the algorithm's complexity and a non-asymptotic convergence result. Empirical experiments also demonstrate the efficiency of the proposed method in different learning problems.

3. Effectiveness: The proposed FORUM algorithm achieves state-of-the-art performance on three multi-task learning benchmark datasets. The paper provides extensive experimental results that demonstrate the effectiveness of the proposed method in comparison to other state-of-the-art algorithms.

### Weaknesses
1. Limited scope: The paper only evaluates the proposed FORUM algorithm on two learning problems, i.e., multi-objective data hyper-cleaning and multi-task learning on three benchmark datasets. The generalizability of the proposed method to other learning problems is not thoroughly explored.

2. Lack of comparison with non-gradient-based methods: The paper only compares the proposed FORUM algorithm with existing gradient-based methods, such as MOML and MoCo. It would be interesting to see how the proposed method compares to non-gradient-based methods, such as evolutionary algorithms or swarm intelligence.

3. Lack of implementation details: The paper does not provide detailed implementation information about the proposed FORUM algorithm, such as the specific hyperparameters used in the experiments. This makes it difficult for other researchers to reproduce the results and compare the proposed method with their own algorithms.

### Questions
Regarding the use of approximation methods in the paper, I have a question for the authors. While the paper proposes an approximation method to compute ω∗(α) and approximates the constraint function q(z) using eq(z) = f(z)−f(α, ˜ωT ), it is not clear how the approximation errors affect the performance of the proposed FORUM algorithm. Could you please provide more insights into the impact of the approximation errors on the convergence and efficiency of the proposed method?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
