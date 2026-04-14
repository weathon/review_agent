# Private Stochastic Optimization for Achieving Second-Order Stationary Points

- Decision: Reject
- Scores: 5, 8, 8, 6

## Abstract
This paper addresses the challenge of achieving second-order stationary points (SOSP) in differentially private stochastic non-convex optimization. We identify two key limitations in the state-of-the-art: (i) inaccurate error rates caused by the omission of gradient variance in saddle point escape analysis, resulting in inappropriate parameter choices and overly optimistic performance estimates, and (ii) inefficiencies in private SOSP selection via the AboveThreshold algorithm, particularly in distributed learning settings, where perturbing and sharing Hessian matrices introduces significant additional noise. To overcome these challenges, we revisit perturbed stochastic gradient descent (SGD) with Gaussian noise and propose a new framework that leverages general gradient oracles. This framework introduces a novel criterion based on model drift distance, ensuring provable saddle point escape and efficient convergence to approximate local minima with low iteration complexity. Using an adaptive SPIDER as the gradient oracle, we establish a new DP algorithm that corrects existing error rates. Furthermore, we extend our approach to a distributed adaptive SPIDER, applying our framework to distributed learning scenarios and providing the first theoretical results on achieving SOSP under differential privacy in distributed environments with heterogeneous data. Finally, we analyze the limitations of the AboveThreshold algorithm for private model selection in distributed learning and show that as model dimensions increase, the selection process introduces additional errors, further demonstrating the superiority of our proposed framework.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies the convergence to a SOSP while preserving differential privacy. The paper points out an issue in the analysis of Liu et al. (2024) - incorrect handling of the stochastic noise - and shows how to fix it. It then presents a differentially private algorithm in the distributed settings.

### Strengths
A clear set of results on an interesting theoretically and well-motivated and studied problem.

### Weaknesses
Major concerns:

-- Please make it clear where you use non-constant learning rate. You don’t mention it in the main body and even at the crucial point - which seems to be Equation (41). This is a major concern since this is the part which, as you claim, is incorrect in Liu et al.

-- I believe the discussion “Gaps in Private SOSP selection” in Section 3 is not exactly fair.
1) First, one can compute the smallest Hessian eigenvalues efficiently - by using Hessian-vector products, which can be computed efficiently in neural networks.
2) Second, not sure what you mean by “requires additional data” - Liu et al. give bounds in Lemmas 4.1 and 4.5, and no additional data seems to be required.
3) Liu et al. didn’t try to guarantee privacy in the distributed settings, so I’m not sure one can call it a gap. That said, I’m not sure that even their approach has many concerns. First, one can approximate the gradient norm using sketches. Second, as outlined above, one don’t need to perturb Hessians, just the vectors resulting from the Hessian-vector products. Since this needs to be computed for at most logarithmic number of candidates, I expect the overall increase in the amount of noise to be negligible compared with the entire training procedure.

-- You want to find an α-SOSP for the population risk, while the proof seems to only cover the empirical risk (i.e. on the sampled dataset).

-- The choice of b1 and b2 depends on η and χ. On the other hand, η and χ depend on b1 and b2. Please eliminate the cyclic dependency, i.e. express b1 and b2 in terms of problem parameters. Please also express σ and r in the terms of problem parameters. Due to the cyclic dependency, combined with complicated expressions for b1 and b2, I couldn’t verify whether Theorem 2 is correct.

-- In Section 4, I don’t see why you can’t just use the analysis of Jin et al. black-box. The choice of parameters is the same, and the only change I see is how you check whether the point is a saddle point, by checking whether the point moved sufficiently far; this technique is, however, covered in their earlier paper “How to Escape Saddle Points Efficiently”.

### Questions
-- For all statements (at least for major theorems and lemmas), please:
1) State all assumptions and parameters precisely (you can state assumptions and refer to them later). To give an example, I spent quite some time looking for definition of U.
2) Refer to the algorithms they correspond to.
3) Refer to where they are proved in the appendix.

-- I personally believe the notation for the escaping radius is not great. I suggest using a more self-explanatory notation, such as R.

-- It’s not clear to me that the choice α = Θ(ψ) is the optimal choice. For example, in analysis of DP-SGD, alpha is basically independent of ψ (that is, you can make ψ and the number of iterations T arbitrarily large, without changing α), so it’s surprising to me that you have a hard dependence of α on ψ.

-- I think the paper would benefit from experimental evaluation.

-- You assume that each individual function (i.e. f(..., z)) is Lipschitz, smooth, and has Lipschitz Hessian. Do you need all of these assumptions for the individual functions?

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
This paper studies a DP stochastic non-convex optimization problem with the focus on convergence to the second-order stationary points (SOSP), which is a new formulation and more stringent than previous focus first-order stationary points (FOSP). By revisiting the perturbed SGD with Gaussian noises, the paper corrects the error rate analysis in Liu et al. 2024 who ignored the statistic gradient variance, and improve the computational cost as well as the statistics error in the distributed learning setting in the previous literature Liu et al. 2024.

### Strengths
The paper is technically solid and well-written. The insight that model parameters move sufficiently far (over a threshold) leads to the novel algorithmic design as well as the theoretical analysis. The comparisons with the Liu et al. 2024 is solid and comprehensive. The extension to the distributed learning setting with improved computational cost and error guarantee showcase the power of the proposed algorithmic design and analysis.

### Weaknesses
A modest size of empirical experiments could help with the illustration of the practical value of the proposed novel algorithm.

### Questions
None.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a new framework for achieving SOSPs in DP non-convex optimization. The authors improved over the current state-of-the-art methods by better accounting over gradient variance and error rates based on a perturbed SGD . The framework is also extended to distributed environments.

### Strengths
The authors improve upon the state-of-the-art in the challenging area of private stochastic optimization for general non-convex functions.

The paper is a collection of elegant and technically solid results that advance the field through sophisticated analytical techniques.

The proposed algorithm represents a significant innovation compared to previous results.

### Weaknesses
For a reader who is not fluent with the literature, the paper is not easy to follow. Several insights are left unexplained. The paper is quite dense without enough provided intuition on the proposed algorithm. However, this is somehow understandable due to the page limit.

### Questions
Compared to "Private (Stochastic) Non-Convex Optimization Revisited: Second-Order Stationary Points and Excess Risks," the algorithm got rid of the internal state of "Frozen," which seems to be important. Could the authors further explain this insight?

The gap between the current result and the lower bound is still large (rate of n); any comments on this?

### Soundness
3

### Presentation
3

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
This paper addresses the problem of achieving second-order stationary points (SOSP) in differentially private (DP) stochastic non-convex optimization. The authors identify two key limitations in the state-of-the-art Liu et al. (2024): (i) inaccurate error rates caused by the omission of gradient variance in saddle point escape analysis (ii) inefficiencies in private SOSP selection via the AboveThreshold algorithm.

The authors propose a new framework that leverages general gradient oracles to overcome these challenges.
The paper establishes a new DP algorithm that corrects existing error rates, and extends the approach to distributed settings.

### Strengths
- The paper is well structured and easy to follow. It clearly outlines the gap in state-of-the-art (SOTA) and the contributions.
- The paper points out an error in the analysis of SOTA and establishes the correct rate. The proposed DP-SPIDER seems simpler than the SOTA.
- The authors also consider the distributed settings and demonstrate the benefits of their proposed approach over the AboveThreshold used in SOTA.

### Weaknesses
The paper fixes the error in SOTA and establishes a correct rate. It would be nice if the authors could discuss if they think their results can be improved and how they compare with the results in SOTA if an easy fix is possible.

### Questions
Is the error in Liu et al. (2024) easy to fix? If so, what would be the correct error rate and how does it compare to your results?

### Soundness
3

### Presentation
3

### Contribution
4
