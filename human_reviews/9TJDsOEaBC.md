# Bayesian Vector Optimization with Gaussian Processes

- Decision: Reject
- Scores: 5, 5, 5, 6

## Abstract
Learning problems in which multiple conflicting objectives must be considered simultaneously often arise in various fields, including engineering, drug design, and environmental management. Traditional methods of multi-objective optimization, such as scalarization and identification of the Pareto set under componentwise order, have limitations in incorporating objective preferences and exploring the solution space accordingly. While vector optimization offers improved flexibility and adaptability via specifying partial orders based on ordering cones, current techniques designed for sequential experiments suffer from high sample complexity, which makes them unfit for large-scale learning problems. To address this issue, we propose VOGP, an ($\epsilon,\delta$)-PAC adaptive elimination algorithm that performs vector optimization using Gaussian processes. VOGP allows users to convey objective preferences through ordering cones while performing efficient sampling by exploiting the smoothness of the objective function, resulting in a more effective optimization process that requires fewer evaluations. We first establish provable theoretical guarantees for VOGP, and then derive information gain based and kernel specific sample complexity bounds. VOGP demonstrates strong empirical results on both real-world and synthetic datasets, outperforming previous work in sequential vector optimization and its special case multi-objective optimization. This work highlights the potential of VOGP as a powerful preference-driven method for addressing complex sequential vector optimization problems.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes to use Gaussian processes (GPs) to identify the optimal set under a given preference cone, reducing the sample complexity of such approaches. It relies on the uncertainty quantification features of GPs to filter points that are likely to be dominated as well as optimal ones. The query is at the least certain design, so a fully exploratory scheme. A theoretical analysis is provided as well as some empirical results on data sets.

### Strengths
- The use of preference cones is less common in the multi-objective Bayesian optimization community.
- Existing theoretical results are extended to ordering cones.

### Weaknesses
- There are a lot of existing works on preference learning with BO.
- It is not clear how to use cones for a practitioner.
- Only discrete input spaces are considered.
- The links to a similar method, PAL, are not detailed enough. An empirical comparison with this method is needed.
- The empirical results are not reproducible.

### Questions
Related references:
- Picheny, V. (2015). Multiobjective optimization using Gaussian process emulators via stepwise uncertainty reduction. Statistics and Computing, 25(6), 1265-1280.
- Emmerich, M. T., Deutz, A. H., & Klinkenberg, J. W. (2011, June). Hypervolume-based expected improvement: Monotonicity properties and exact computation. In 2011 IEEE Congress of Evolutionary Computation (CEC) (pp. 2147-2154). IEEE.
- Yang, K., Li, L., Deutz, A., Back, T., & Emmerich, M. (2016, August). Preference-based multiobjective optimization using truncated expected hypervolume improvement. In 2016 12th International Conference on Natural Computation, Fuzzy Systems and Knowledge Discovery (ICNC-FSKD) (pp. 276-281). IEEE.
- Lepird, J. R., Owen, M. P., & Kochenderfer, M. J. (2015). Bayesian preference elicitation for multiobjective engineering design optimization. Journal of Aerospace Information Systems, 12(10), 634-645.
- Khan, F. A., Dietrich, J. P., & Wirth, C. (2022). Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge. arXiv preprint arXiv:2208.10300.
- Garnett, R. (2023). Bayesian optimization. Cambridge University Press.
- Svenson, J., & Santner, T. (2016). Multiobjective optimization of expensive-to-evaluate deterministic computer simulator models. Computational Statistics & Data Analysis, 94, 250-264.
- Ignatenko, T., Kondrashov, K., Cox, M., & de Vries, B. (2021). On Preference Learning Based on Sequential Bayesian Optimization with Pairwise Comparison. arXiv preprint arXiv:2103.13192.
- Ungredda, J., & Branke, J. (2023, July). When to Elicit Preferences in Multi-Objective Bayesian Optimization. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation (pp. 1997-2003).
- Taylor, K., Ha, H., Li, M., Chan, J., & Li, X. (2021, June). Bayesian preference learning for interactive multi-objective optimisation. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 466-475).
- Jussi Hakanen and Joshua D Knowles. On using decision maker preferences with ParEGO.
In International Conference on Evolutionary Multi-Criterion Optimization, pages 282–297.
Springer, 2017.
- Barracosa, B., Bect, J., Baraffe, H. D., Morin, J., Fournel, J., & Vazquez, E. (2022). Bayesian multi-objective optimization for stochastic simulators: an extension of the Pareto Active Learning method. arXiv preprint arXiv:2207.03842.

Page 1: It is unclear what the inclusion relation is between Pareto optimal solution and cone order optimal solution (“However, this approach can be restrictive, as it only permits a certain set of trade-offs between objectives.” and then “preference cones provide a way to bias the search toward certain regions of the Pareto front.”

Introductive agricultural example: perhaps you could complement Figure 1 with an actual Pareto front to better illustrate the interest. You could add the pessimistic Pareto front defined later. Also in Figure 1, cones are parameterized with angles but later on with a matrix. 

Can you describe the convex optimization problem used in the pessimistic Pareto front construction?

Discuss the relation with epsilon-PAL. It should be added to the empirical comparison.

It is unclear where the cone properties appear in the theoretical results.

Could you provide timings? Progress curves rather than just fixed snapshots? Random search should be added as a baseline.

As I understand, the noise hyperparameter is not learned by the GP? What is multi-output covariance kernel used? Is it the same across all compared methods? Is the learning strategy shared among all methods (e.g., fixed hyperparameters)?
State of the art EHVI is proposed by Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization. Advances in Neural Information Processing Systems, 33, 9851-9864.
Could you add an example with more than 2 objectives? Too many details are missing to reproduce the experiments: code used, number of samples, etc.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Vector optimization formulates a multi-objective optimization problem in a way that expresses a user's preference to trade off the different objectives.
The study leverages the machinery of Bayesian optimization (BayesOpt) for the task of vector optimization, where the objective functions are assumed to be black boxes.
The authors propose a sample-efficient policy that explores the solution space using as few queries as possible.
Using the smoothness assumption made by the Gaussian process (GP), the paper also proves a PAC-learning guarantee for the proposed algorithm.
The experiments show that this algorithm is more effective at maximizing the success rate (according to the PAC criterion) than a wide range of baselines, while keeping the number of queries low.

### Strengths
The problem studied in this paper is motivated well.
Vector optimization seems like an elegant way for a user to flexibly express their preference over multiple objectives.
Using BayesOpt to tackle this problem when objectives are expensive to query seems like a natural solution.
The proposed algorithm is proven to have good theoretical guarantee, where the algorithm will return an approximate Pareto-optimal (in the context of vector optimization) with high probability in at most some specified number of queries.
The experiments are convincing in showing that the proposed method is competitive against many baselines in the multi-objective optimization literature.

### Weaknesses
I find the background and problem definition a bit hard to follow.
The authors can consider prioritizing intuitive understanding over exposition of all the math.
The same goes for the algorithm itself; perhaps add a diagram on the procedure the policy goes through in Algorithm 1.

The paper does a good job comparing the proposed algorithm against state-of-the-art multi-objective BayesOpt policies.
However, from what I understand, only one algorithm from the vector optimization literature, Naïve Elimination, is included.
It could be worth including other (possibly sample-inefficient) algorithms for a more complete comparison.

The experiments are set up in a way that the GP always has access to the correct hyperparameters (obtained via training on the entire data set).
In many real-life settings, we don't have access to the correct hyperparameters, or even good priors for them.
The paper could benefit from studying the effects of the GP having the wrong hyperparameters on the performance of the algorithm.

### Questions
- In Definition 1, my understanding is that the second condition (ii) specifies that $x \in P$ is not dominated by another by more than $2 \epsilon$.
What does the first condition (i) say?
Perhaps the the background of vector optimization could benefit from more descriptive discussions.
- The authors noted that once a point is added to the Pareto set, it will not be removed.
This doesn't match my intuition well; isn't it possible that as we learn more about the objectives, we realize that some of the points already added aren't non-dominated?
- Could the authors comment on the possible difficulties of extending the proposed algorithm to continuous setting?
I imagine the challenge lies in the discarding and Pareto identification phases.
Can we try to discard and identify dominated and non-dominated regions, respectively?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a generalization of multi-objective Bayesian optimization.
Specifically, the partial ordering is defined by a convex cone.
The partial ordering induced by coordinate-wise comparisons used in the literature of multi-objective Bayesian optimization is a special case where the convex cone is the nonnegative orthant.
Next, the paper proposes an $(\epsilon, \delta)$-probably approximately  correct algorithm, which finds an approximate Pareto set with high probability.
Theoretically, the authors present an upper bound on the number of iterations finding an $(\epsilon, \delta)$-PAC Pareto set.
Empirical evaluations on a few low dimensional functions demonstrates its superior sample complexity compared to the naive elimination (Ararat and Tekin, 2023), a recent algorithm proposed in the setting of stochastic bandits.

### Strengths
- This paper introduces to the BO community the concept of partial ordering induced by a convex cone, which I think is a useful generalization and may be beneficial to certain BO applications.
- A theoretical analysis on the sample complexity is presented, which shows that the algorithm finds an $\epsilon$-approximate Pareto set with high probability $1 - \delta$.

### Weaknesses
- The experiments are done on very small datasets. The largest dimension is $4$ and all tasks have two objectives.
- At this point, the method in the paper is restricted to discrete domains $\mathcal{X}$, which limits the application of the method: the constant $\beta_t$ in the algorithm depends on the cardinality of $\mathcal{X}$, the theorem statements assume finite domains, and the evaluation metrics needs a finite cardinality $|\mathcal{X}|$ as well. I would assume this is fixable by extending the results in the paper, but additional empirical evaluations are required.
- The naive elimination is a theoretical construct in the bandit setting, which does not exploit the correlation in the GP model. A more meaningful baseline for comparison is other multi-objective BO algorithms. For example, as the angle $\theta$ changes, how do the Pareto precision and Pareto recall change comparing with regular multi-objective BO algorithms (which are designed specifically for a particular definition of Pareto optimality)? Another helpful experiment is to plot metrics w.r.t. the number of queries. This allows us to visually check the convergence. From the current tables, it is hard to tell if they are fully converged or not.
- The experimental setup is non-standard. For example, "for each dataset, we learn the kernel hyperparameters by training on the entire dataset". However, the hyperparameters in BO are typically learned as more queries are added to the training data. Have the author tried the latter more commonly used setting?

Minor comments:
- The following notations need more explicit definitions in the main text: the hyperrectangles $R_t(x)$, their diameters $\omega_t(x)$, the information gain $\gamma_t$ and the constant $\eta$.
- In the definition of the polyhedral cone, it should be $C = \\{\mathbf{x} \in \mathbb{R}^M: W \mathbf{x} \geq 0\\}$ and $W$ should be $N \times M$. The cone is defined in the output space, not in the domain.

### Questions
- Theorem 1 and Theorem 2 need to add an extra technical assumption on the cone $C$. Otherwise the bounds may be vacuous in certain cases. For example, $C = \\{(x_1, x_2) \in \mathbb{R}^2: x_1 = x_2\\}$ is a well-defined polyhedral ordering cone. However $d(1) = \infty$ in this case and thus both bounds become vacuous.
- Can you share more intuition on Definition 4? My intuition is that $\mathbf{u}^*$ points to the "center" of the cone.
- In line 3 of Algorithm 4, why $+ C$ and $- C$ are on both sides of the intersection? Shouldn't $(\mathbf{R}_t(\mathbf{x}) + \epsilon \mathbf{u}^* + C) \cap \mathbf{R}_t(\mathbf{x}^\prime)$ be already sufficient?
- The evaluation metrics PA, PR and PP need the ground truth Pareto set $P^*$. How are the ground truth Pareto sets computed?
- Branin and Currin are continuous functions defined on continuous domains. Does the experiment in Table 1 discretize their domains?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes vector optimization for multiple-objective optimization using confidence intervals built from Gaussian Processes. The proposed method called VOGP allows users to convey objective preferences through ordering cones while performing efficient sampling by exploiting the smoothness of the objective function, resulting in a more effective optimization process that requires fewer evaluations. Both theoretical guarantee and experimental results are demonstrated to consolidate the claims.

### Strengths
proposes a vector optimization based on Gaussian processes.

### Weaknesses
There might be more applications illustrated.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
