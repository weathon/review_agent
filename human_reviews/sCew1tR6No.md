# Partial Gromov-Wasserstein Metric

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6, 6

## Abstract
The Gromov-Wasserstein (GW) distance has gained increasing interest in the machine learning community in recent years, as it allows for the comparison of measures in different metric spaces. To overcome the limitations imposed by the equal mass requirements of the classical GW problem, researchers have begun exploring its application in unbalanced settings. However, Unbalanced GW (UGW) can only be regarded as a discrepancy rather than a rigorous metric/distance between two metric measure spaces (mm-spaces). In this paper, we propose a particular case of the UGW problem, termed Partial Gromov-Wasserstein (PGW). We establish that PGW is a well-defined metric between mm-spaces and discuss its theoretical properties, including the existence of a minimizer for the PGW problem and the relationship between PGW and GW, among others. We then propose two variants of the Frank-Wolfe algorithm for solving the PGW problem and show that they are mathematically and computationally equivalent. Moreover, based on our PGW metric, we introduce the analogous concept of barycenters for mm-spaces. Finally, we validate the effectiveness of our PGW metric and related solvers in applications such as shape matching, shape retrieval, and shape interpolation, comparing them against existing baselines. Our code is available at https://github.com/mint-vu/PGW_Metric.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Partial Gromov-Wasserstein (PGW), a particular case of UGW, and prove that it is a metric between mm-spaces. The authors propose Frank-Wolfe-based algorithms with convergence guarantees for solving the PGW problem and extend the concept of barycenters to mm-spaces. PGW’s effectiveness is demonstrated through applications like shape matching, retrieval, and interpolation, showing improved performance over existing baselines.

### Strengths
1. This paper is well-written and easy to follow.
  
2. The paper rigorously proves that PGW, a specific case of UGW with the TV norm as the chosen $f$-divergence, defines a valid metric between mm-spaces, contributing to the theoretical understanding of these transport frameworks.
  
3. The authors propose using the Frank-Wolfe algorithm to solve the PGW problem and provide a convergence guarantee, enhancing the computational tractability of the method.

### Weaknesses
1. Limited Experimental Scope: The experimental evaluation in this paper could be expanded. It might be helpful to include experiments on PU learning (positive-unlabeled learning), similar to those conducted in the UGW paper, for a more comprehensive comparison.
  
2. Hyperparameter Tuning in Baselines: For the toy examples and point cloud matching experiments, the baselines' performance might improve with more thorough hyperparameter tuning since authors use fixed values mentioned in Appendix N. Given that UGW has been shown to be robust to outliers both theoretically and empirically (as demonstrated in the original UGW paper and also [A]), better-tuned baselines could potentially mitigate the impact of outliers more effectively in these experiments.

3. Minor Issues: Several broken references and citations are present in the paper, such as those on lines 165-166, 894, and 1025. Addressing these would improve the paper's readability and professionalism.


[A] Tran, Quang Huy, et al. "Unbalanced co-optimal transport." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 37. No. 8. 2023.

### Questions
1. Could the authors incorporate other experiments, such as PU learning (positive-unlabeled learning), to provide a broader evaluation?
  
2. Could the authors report the results of varying hyperparameters in the baselines to examine their impact on performance?
  
3. While the KL divergence in UGW is not a rigorous metric, it has desirable properties such as robustness to outliers. Could the authors explain why the TV norm used in PGW results in better performance in the numerical experiments compared to UGW? Alternatively, is the improved performance due to the use of the Frank-Wolfe (FW) algorithm instead of Sinkhorn?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The Gromov-Wasserstein (GW) distance is a metric on probability measures supported on different metric spaces. This paper extends this notion to more general Radon measures, not necessarily probability measures, termed partial Gromov-Wasserstein (PGW) distance. The authors show that PGW is indeed a metric. Two Frank-Wolfe (FW) algorithms are proposed to compute PGW, and three examples are provided to illustrate the performance of PGW.

### Strengths
1, The main contribution is to establish the metric property of partial GW. 

2, Simulations in subsection 6.2 clearly demonstrate improved performance of PGW compared to MPGW.

2, The simulation examples are interesting and illustrative.

### Weaknesses
1, The current manuscript has many typos. For example, the citations on lines 165 and 177 are not displaying properly.

### Questions
1, Please read the paper thoroughly and correct any other typos.

2, Is the proposed Algorithm 1 the same as Algorithm 1 in the paper "Partial Optimal Transport with Applications on Positive-Unlabeled Learning"? What are the novelties in the optimization algorithms?

### Soundness
4

### Presentation
3

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
This paper introduces the Partial Gromov-Wasserstein (PGW) metric, a novel adaptation of the Gromov-Wasserstein (GW) distance tailored for metric measure spaces. Unlike traditional GW, which requires equal mass in spaces for meaningful comparison, PGW accommodates unbalanced settings through the incorporation of a total variation penalty. The authors present theoretical results, including proof of PGW as a metric, and develop two Frank-Wolfe solvers for computational efficiency. Experiments on shape matching, retrieval, and interpolation demonstrate PGW's robustness to some extent.

### Strengths
- The paper rigorously proves that PGW qualifies as a metric.
- The two Frank-Wolfe solvers for PGW demonstrate effective performance, making the method feasible for practical applications.
- Experimental results indicate that PGW outperforms traditional GW in handling unbalanced data with outliers, enhancing its utility in real-world tasks.

### Weaknesses
- Comparisons with alternative unbalanced OT metrics are somewhat limited, which leaves the practical performance of PGW against state-of-the-art methods less explored.
- The robustness parameter $\lambda$ requires careful tuning, which could pose a challenge for practical applications.
- The computational efficiency of PGW may still be restrictive in high-dimensional settings or for very large datasets.

### Questions
- The baseline selection of the experiment is too limited to demonstrate the effectiveness of your methods. Why don't you compare the methods [1,2,3] based on robust OT? Please add corresponding experiments.
- - [1] Nietert S, Goldfeld Z, Cummings R. Outlier-robust optimal transport: Duality, structure, and statistical analysis[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2022: 11691-11719.
- - [2] Wang X, Huang J, Yang Q, et al. On Robust Wasserstein Barycenter: The Model and Algorithm[C]//Proceedings of the 2024 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2024: 235-243.
- - [3] Le K, Nguyen H, Nguyen Q M, et al. On robust optimal transport: Computational complexity and barycenter computation[J]. Advances in Neural Information Processing Systems, 2021, 34: 21947-21959.
- In Wasserstein barycenter's experiment for robustness, why only show the interpolation between two shapes? Can you show me how the algorithm computes the average shape of several shapes?
- It seems that all the experiments you provide can be replaced by the equivalent of robust OT, which cannot truly reflect the advantages of robust Gromov-Wasserstein barycenter.
- Is the parameter $\lambda$ easy to search? Different proportions of noise seem to require different $\lambda$s. Does this hinder the practicality of this method? It seems that the specific $\lambda$ search process and the influence of $\lambda$ on the experimental effect are not given in the experiment.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The Gromov-Wasserstein (GW) distance has drawn a lot of attention in recent years for the comparison of measures in different metric spaces. This paper focuses on the Unbalanced Gromov-Wasserstein (UGW) problem, which they called Partial Gromov-Wasserstein (PGW). They showed PGW is a well-defined metric between metric measuring spaces and discussed some theoretical properties. They designed a Frank-Wolfe algorithm-based method to solve PGW. Convergence analysis is provided. Numerically, they tested it on some datasets with existing baselines.

### Strengths
Pros:

They propose PGW, a total variation version of the UGW problem with theoretical properties analysis.

The experiment is showing their good performance in PGW.

### Weaknesses
Cons:

1. The main concern is the novelty and significance since the problem itself is not well-generalizable to other problems. Meanwhile, some techniques are adopted from existing works. For example:

a. What is the unique technique of your method design compared to the Frank-Wolfe solver presented in (Chapel et al., 2020) except their problem setting is UGW?

b. How is the convergence analysis partially harder than the existing method?

c. Since the algorithm only guarantees stationary points (this is fine, the reviewer does not require a global optimality), how can we convinced it works well in more real scenarios? Is there any more evidence?



2. For the FW algorithm in the optimization problem in line 331, it’s the main subroutine in each iteration, how is the complexity of solving it? This is important in the analysis since FW usually could be dramatically slowed down by the subroutine.


3. Minor issue. There are some citation errors in lines 165 and 177.

### Questions
Please address the question in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper improves unbalanced approaches to the optimal transport (OT) problem. 
The objective is to compare probability measures between different metric spaces (GW distances).
To overcome the non-convexity of the  formulation, which usually relies on the Frank-Wolfe (FW) 
algorithm. 

The authors propose a new metric rooted in the so called Partial Gromov-Wasserstein (PGW) problem, 
and the main novelty is to use a total variation penalty as well as to provide efficient solvers. 

The paper commences by introducing OT and its partial extension POT.  Then, it comes out the regularization
term which in principle is the Kullback-Leibler divergence which is replaced by Total Variation (TV). 
Then, the FW algorithm is adopted. 

The toy experiment: 
"TOY EXAMPLE: SHAPE MATCHING WITH OUTLIERS
We use the moon dataset and synthetic 2D/3D spherical data in this experiment."

should be explained earlier in the paper to motivate the reader. Formal language should not be 
an enemy of intuition.

Pros: 
- Nice formal language. 

Cons:
- Not intuitive for general readers. 
- The contribution is relevant (avoid strong outliers) 

Questions:
- Is there any formalism for Bregman divergences?

### Strengths
Nice formalization.

### Weaknesses
Very hard to read even for an experienced reviewer.

### Questions
* Adequacy to Bregman divergences?
* Put synthetic examples in the beigning as a motivation and smooth the formal language.

### Soundness
3

### Presentation
2

### Contribution
3
