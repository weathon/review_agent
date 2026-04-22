# Sharper Analysis of Single-Loop Methods for Bilevel Optimization

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Bilevel optimization underpins many machine learning applications, including hyperparameter optimization, meta-learning, neural architecture search, and reinforcement learning. While hypergradient-based methods have advanced significantly, a gap persists between theoretical guarantees—typically derived for multi-loop algorithms—and practical single-loop implementations required for efficiency. This work narrows that gap by establishing sharper convergence results for single-loop approximate implicit differentiation (AID) and iterative differentiation (ITD) methods. For AID, we improve the convergence rate from $\mathcal{O}(\kappa^6/K)$ to $\mathcal{O}(\kappa^5/K)$, where $\kappa$ is the condition number of the inner-level problem. For ITD, we prove that the asymptotic error is $\mathcal{O}(\kappa^2)$, exactly matching the known lower bound and improving upon the previous $\mathcal{O}(\kappa^3)$ guarantee. We further validate the refined analyses by the experiments on synthetic bilevel optimization tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents significant theoretical advancements in understanding single-loop methods for bilevel optimization, a fundamental problem in machine learning applications like hyperparameter tuning and meta-learning. The authors focus on two popular algorithms—Approximate Implicit Differentiation (AID) and Iterative Differentiation (ITD)—and introduce a novel analytical framework called Decoupled Norm Analysis (DNA) to overcome limitations in existing convergence proofs. Their key contributions include improving AID's convergence rate from $\mathcal{O}(k^6/K)$ to $\mathcal{O}(k^5/K)$ and establishing that ITD's asymptotic error is $\mathcal{O}(k^2)$, which matches the theoretical lower bound and resolves a previous gap in the literature.

### Strengths
The paper achieves remarkable progress in tightening convergence bounds for both AID and ITD methods. The reduction in AID's condition number dependence from κ⁶ to κ⁵ and the establishment of ITD's optimality at O(κ²) represent meaningful advances that significantly narrow the theory-practice gap in bilevel optimization.

The proposed DNA approach is a methodological highlight. By avoiding premature norm squaring and carefully handling error propagation, the authors develop a more refined analysis technique that prevents the overestimation common in prior work. This framework could inspire similar improvements in other optimization domains.

### Weaknesses
The requirement that ∇²ₓᵧg(z) and ∇²ᵧᵧg(z) are ρ-Lipschitz continuous is quite restrictive. Many practical bilevel problems in deep learning (e.g., neural architecture search) involve non-smooth or highly non-linear objectives where this assumption may not hold. The paper could better discuss the implications of this limitation and potential relaxations.

While the analysis assumes strong convexity (Assumption 1), many real-world applications like MAML involve non-convex inner problems. The paper acknowledges this in passing but doesn't explore how the results might extend to these settings, limiting immediate practical applicability.

The complex expression for LΦ appears without an intuitive explanation. For example, the term ρL²M/µ³ seems particularly large, yet its necessity isn't justified. More discussion on how these constants affect practical performance would be valuable.

Validation is limited to synthetic problems, leaving questions about performance on real-world tasks (e.g., hyperparameter optimization). Including even one practical benchmark would strengthen the paper's impact.

The related work mentions competing single-loop methods like MEHA and F3SA, but experiments don't compare against them. This makes it difficult to assess whether the theoretical improvements translate to practical advantages.

### Questions
See the above weakness.

My main concerns lie in the assumptions 2-4, which may not be satisfied in BO scenarios. The upper problem is usually nonconvex and even non-smooth, which brings the gap between your algorithm and theoretical findings.

And the baselines on theory seem limited. More findings on first-order or single-loop algorithms are required.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper makes a significant theoretical contribution to bilevel optimization by providing a sharper, more faithful analysis of single-loop methods—bridging the gap between theory and practical algorithms. The Decoupled Norm Analysis (DNA) framework is elegant and potentially influential for future work on large-scale bilevel learning and reinforcement learning formulations.

### Strengths
1. This paper improves AID from $\mathcal{O}(\kappa^6 / K)$ to $\mathcal{O}(\kappa^5 / K)$ and pins ITD's inherent error exactly at $\mathcal{O}(\kappa^2)$, narrowing the gap between practice (one inner step) and prior multi-loop-learning analyses.

2. This work uses the new Decoupled Norm Analysis (DNA) to explain why earlier bounds were pessimistic and provides a reusable template that can tighten $\kappa$-dependence for related algorithms.

3. On synthetic bilevel problems, the new bounds track the true hypergradient closely; for ITD, they nearly coincide with the lower bound up to constants, reinforcing the order-tight claim.

### Weaknesses
The improvement presented in this paper is quite limited. It focuses solely on the convergence analysis of single-loop AID and single-loop ITD methods, without providing evidence that the analysis extends to other mainstream approaches—such as multi-loop AID/ITD frameworks or other single-loop methods like SOBA, or first-order methods.

This raises a major concern that the overall impact of this work may be quite restricted.

### Questions
My questions are from the weakness above. This work focuses on the convergence analysis improvement. However, the proposed analysis only applied to single-loop AID and the single-loop ITD method. 

1. Can the proposed analysis improve the convergence rate of multi-loop AID/ITD?

2. Can the proposed analysis improve the convergence rate of other bilevel solvers like SOBA or F2SA?

3.  Can the proposed analysis improve the convergence rate in a stochastic setting?

4. Does the analysis ask for anything more than the original AID/ITD analysis, like assumptions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes new theoretical technique to improve the convergence rate of existing bilevel optimization algorithms AID and ITD. The author further validate the theory by synthetic bilevel optimization experiments.

### Strengths
1. The paper is very clean and clear to read and understand.
2. The figure is helpful to understand the theorem and theoretical technique is novel.

### Weaknesses
1. It is good to improve the convergence rate in existing algorithms AID and ITD. However, I do not know if the proposed technique can be extended into other works or domains. The generalizability of new technique has not been fully studied and discussed in this paper. The author is encouraged to provide such study and discussion for better understanding for new technical analysis.

2. The main contribution of this paper lies in the theory and the experiments is only used to illustrate the effectiveness of the theory. Usually, such theoretical paper also provides some insights for the practical experiments and present some experimental results. The author is encouraged to include and discuss the potential experimental insights and results.

3. There exist other bilevel optimization algorithms. The author is encouraged to further discuss the applicability of the proposed technique into other bilevel algorithms. The applicability of this technique in bilevel optimization domain has not fully studied.

### Questions
Please check the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
