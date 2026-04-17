# Privacy Amplification by Iteration with Projected Alternating Direction Method

- Decision: Reject
- Scores: 4, 6, 4, 4, 2

## Abstract
Alternating direction method of multipliers (ADMM) is a common approach for privacy amplification and utility guarantees in various machine learning tasks, especially those require cooperation between private and public users (or servers). However, this approach cannot achieve exact feasibility constraint throughout the learning process, and even has a large feasibility gap at the early iterative stage, which cannot handle the small-sampled situations. To solve these problems, we propose a projected alternating direction method that achieves exact feasibility and enables each user to supervise the objective value throughout the learning process. Moreover, it allows both Gaussian and Laplace noise for variable masking and privacy amplification. Third, it does not require the Markov operator condition or double-iterations to achieve one-step privacy and utility guarantees. Fourth, it achieves the same order of privacy-utility tradeoff as that of the existing ADMM methods. In summary, the proposed methodology requires fewer conditions but solves more general privacy amplification problems and enjoys more favorable properties than the existing ADMM methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PADM, a method that solves the minimization of a sum of convex functions under linear constraints without using dual variables, and its differential privacy (DP) variant, Private PADM. While it is common to solve the dual problem with ADMM, a feasibility gap for the constraint term often remains in practice. In contrast, the paper optimizes the primal problem augmented with a (squared-norm) regularization on the linear constraint, using auxiliary variables $(u,v)$. It also provides utility and privacy analyses for Private PADM. Through several numerical experiments, the authors empirically show that both the feasibility gap and the objective value remain small across multiple privacy levels $\epsilon$.

### Strengths
**(S1) Clear and practical motivation.**
Rather than solving the dual problem, the method solves the primal problem with regularizations using primary $(x,y)$ and their auxiliary variables $(u,v)$ to suppress the feasibility gap—a choice that can be effective in practice.

**(S2) Theoretical contributions.**
The fixed-point condition analysis for PADM and the utility/privacy analyses for Private PADM are carefully presented and provide solid theoretical value.

### Weaknesses
**(W1) Derivation of PADM.**
This may be a minor point, but the derivation could be made clearer. Following the organization of Ryu et al. (2015) [*1], could PADM be derived more naturally from a fixed-point condition? As you know, ADMM can be viewed as a solver for the dual problem $\min_{\zeta} F^{\ast}(-A^\top \zeta) + g^{\ast}(-B^\top \zeta) - c^\top \zeta$, whose fixed-point condition is $0 \in T_1(\zeta) + T_2(\zeta)$, where $T_1(\zeta) = -A \partial F^{\ast}(-A^\top \zeta)$ and $T_2(\zeta) = -B \partial g^{\ast}(-B^\top \zeta) - c$. Reformulating this via Douglas-Rachford splitting yields the (standard) ADMM updates. By analogy, can the fixed-point condition for the cost inside the $\arg \min$ of Eq. (15) be reformulated to yield PADM’s updates?

[*1] E. K. Ryu et al., “A PRIMER ON MONOTONE OPERATOR METHODS,” Appl. Comput. Math., 2015.


**(W2) Limited theoretical evidence for feasibility-gap suppression.**
PADM solves the regularized loss-sum minimization in Eq. (15) via the auxiliary variables $(u,v)$, and a small feasibility gap is shown empirically (e.g., Figure 1). However, the theoretical reason why a penalty on the constraint term should suppress the feasibility gap remains underdeveloped. Providing an explicit analysis would strengthen the claim.

**(W3) Insufficient theoretical comparison to existing DP methods.**
Private ADMM (Cyffers et al., 2023) is used as a baseline for private PADM. Could you provide a theoretical comparison of utility gaps and/or convergence rates to argue for the advantages of the proposed method?

**(W4) Experimental concerns.**

While the core contributions are theoretical, several issues remain on the experimental side:

**Objective in Figure 2.** How exactly is the vertical-axis “objective function” computed for a constrained loss problem? How are constraints handled in this metric? (You mention “final objective function values with the same regularization parameters…,” but the precise evaluation protocol was unclear to me.)

**Hyperparameter tuning.** Did you tune per-method parameters (e.g., learning rate)? How should the regularization weight $\beta$ be determined in practice?

**Baselines.** The following studies may be additional baselines:

[*2] Z. Huang et al., “DP-ADMM: ADMM-based distributed learning with differential privacy,” IEEE TIFS, 2019.

[*3] T. Fukami et al., “DP-norm: Differential privacy primal-dual algorithm for decentralized federated learning,” IEEE TIFS, 2024.

**More practical benchmarks.** For example, fine-tuning only the last layer of DNNs (e.g., language models) yields a convex objective; applying PADM/private PADM in such federated learning (fine-tuning tasks) would increase practical relevance.

### Questions
(Q1) I could not follow the derivation of Eq. (16). In particular, why does the manipulation lead to adding $z/2$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a differentially private Projected Alternating Direction Method (PADM) to address the limitation of existing ADMM-based approaches that cannot ensure exact feasibility throughout the learning process. It supports both Gaussian and Laplace noise for privacy, and the authors prove that the proposed operator is non-expansive, leading to convergence guarantees and formal privacy amplification by iteration. Experimental results validate the effectiveness.

### Strengths
1. The core insight of achieving exact feasibility via projection in the input space seems interesting. 
2. Thorough privacy analyses for both the Gaussian mechanism and the Laplace mechanism, which are commonly used differential privacy mechanisms, are provided.
3. The paper is generally well organized.

### Weaknesses
While the paper is highly theoretical, there are some concerns.

1. It is assumed that the matrices $A$ and $B$ are full-row-ranked. Some discussions about the practical implications of this assumption would be helpful to improve the paper. 
2. The utility guarantee in Theorem 3.4 is quite complicated. Some insights and discussions would improve the readability. 
3. The experiments are conducted on simple problems and datasets, and results on more complicated tasks are needed. Moreover, it is not clear what practical applications/scenarios will benefit from the proposed method that enables exact feasibility.
4. For the baselines in the experiments, it is not clear how the "feasibility gap" is handled, especially when the baseline algorithms output an infeasible point.
5. For the real-world experiment in Section A.13, the authors use the first-order Taylor approximation to derive the closed-form of the operator. Would such an approximation hurt the exact feasibility?

### Questions
1. Could you please add some discussions about the practical implications of the assumption about $A$ and $B$?
2. Could you please clarify the practical applications or scenarios that will benefit from the exact feasibility?
3. For the baselines in the experiments, how is the "feasibility gap" handled?
4. Could you please comment on the impact of the first-order Taylor approximation in Section A.13?

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
3

### Summary
This paper proposes a novel projected alternating direction method (PADM) to improve privacy amplification compared to the existing ADMM. The new method aims to eliminate the feasibility gap that exists in ADMM. Moreover, the privacy-utility tradeoff is in the same or better order compared to the existing methods.

### Strengths
+ The motivation of this paper is strong and clear.

+ The proposed PADM is applicable to privacy amplification by both iteration and subsampling. 

+ Concrete theoretical proofs are provided. 

+ The proposed PADM can guarantee privacy preservation with both Gaussian noise and Laplace noise to achieve RDP and DP.

### Weaknesses
- While this paper provides a strong theoretical development of PADM, the experiment only focuses on two relatively small-scale problems, synthetic data with linear regression task and real-world data with logistic regression task. 
- Although the results verify the effectiveness of the proposed method and show improvement, it would be better to show the potential in other tasks like federated settings. 
- In addition, there may also be some discussions about the impact of the regularization parameter beta.

### Questions
- Does the proposed algorithm work on other more generalized datasets or FL tasks?
- Is it possible to provide some discussions about the impact of the regularization parameter beta?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a projected alternating direction method of multipliers (ADMM) approach that solves two issues in existing results: 1) conventional ADMM cannot satisfy the exact feasibility constraint during algorithmic iterations; and 2) the resulting feasibility gap may amplify the effect of differential-privacy noises on optimization accuracy, leading to a poor privacy-utility tradeoff. In my opinion, the main idea of this paper is to execute a projection operator at each iteration to enforce the feasibility constraint, thereby mitigating the feasibility gap throughout the optimization process. The main concerns are given in the weaknesses below.

### Strengths
This paper eliminates the need for dual variable updates and avoids the assumptions of smoothness and strong convexity used in the existing ADMM-based approaches.

### Weaknesses
1. **Weak practicability:** The authors claim that the proposed approach does not require the differentiability of $F(\boldsymbol{x})$ and $g(\boldsymbol{y})$. However, Steps 1 and 2 in Eq. (10) involve solving optimization subproblems of $F$ and $g$, which rely on computing their proximal mappings. If $F$ and $g$ do not have closed-form or easily computable proximal operators, these subproblems become computationally infeasible, making the proposed approach impractical for real-world applications. In addition, the proposed algorithm requires both $A$ and $B$ to be full row rank. How this condition can be ensured in practical applications should be further clarified.

2. **Non-rigorous Rényi DP analysis:** Theorem 3.5 and Theorem 3.6 derive the Rényi DP results under the implicit assumption that the algorithmic outputs are independent. However, in PADM, $u_{(k)}$  is recursively defined, and the noise $z_{(k)}$ is injected into correlated variables. This dependency means that the RDP composition is not simply additive. A rigorous privacy analysis should account for the iterative dynamics of Algorithm 1, rather than treating each iteration as an independent process.

3. **Finite incremental contribution compared with Chan et al. (2024):** Compared with Chan et al. (2024), this paper has two differences: 1) this paper introduces a projection operator $\Pi_{\mathcal{W}}$ to enable the feasibility constraint. Nevertheless, since $\mathcal{W}$ is convex, the projection operator $\Pi_{\mathcal{W}}$ is firmly non-expansive.
When combined with existing results in Chan et al. (2024), this naturally leads to a 
non-expansive operator $\mathcal{T}_{F}$ and ensures convergence of the proposed algorithm. Therefore, the theoretical analysis is a direct extension of existing results, lacking novel insights. 2) this paper does not require  $F$ and $g$ to be differentiable. However, as noted in Weakness 1, this holds only when $F$ and $g$ have closed-form or easily computable proximal operators. In summary, the authors should more explicitly clarify the fundamental differences between this work and Chan et al. (2024).

### Questions
See the weaknesses above. In addition, I have the following questions:

1. In Eq. (3), why $F$ and $\boldsymbol{x}$ 
are considered private, whereas $g$ and $\boldsymbol{y}$ are assumed to be public?

2. What is the relationship between the set $\mathcal{C}$ in Eq. (1) and the feasible set (*w.r.t* $\boldsymbol{x}$) in Eq. (3)? If the constraint set $\mathcal{C}$ in Eq. (1) is a subset of that in Eq. (3), does Eq. (4) still hold under this condition?

3. Does the non-expansive property of $\psi$ require the smoothness of $\nabla f(\cdot;\mathcal{D})$? In general, Lipschitz continuity of the gradient should be sufficient.

4. The conditions in Proposition 3.2 contradict the earlier statement that the proposed method "does not require differentiability of $F$ and $g$". In fact, if $f$ is not differentiable, then Proposition 3.2 no longer holds, which invalidates the sensitivity bound required for Condition 2. Consequently, the results in Theorem 3.5, Theorem 3.6, and Theorem 3.7 become questionable. The authors should clarify this inconsistency and explain whether differentiability is actually required for theoretical guarantees.

5. The experimental setup is overly simple. Even in the so-called real-world experiment, the logistic regression function considered in Eq. (77) is a simple form of  $\ln(1+e^{a_k})$. I understand that this choice is made to ensure that the operator $\mathcal{P}_{\beta,\cdots}$ has a closed-form solution (as mentioned in Weakness 1). However, this also indirectly demonstrates that applying the proposed method to practical problems would be challenging.

If the authors could address Weaknesses 1 and 2, as well as Problem 4 (which are my main concerns), I would consider raising my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a projected alternating direction method that achieves exact feasibility and allows each user to monitor the objective value throughout the learning process. However, I find that the motivation and presentation should be improved. The novelty of the paper appears limited — it mainly introduces a projection step into ADMM and leverages existing privacy amplification tools.

### Strengths
This paper proposes a projected alternating direction method that achieves exact feasibility and allows each user to monitor the objective value throughout the learning process.

### Weaknesses
1.	The motivation for studying problem (3) is not clearly explained. What are the real-world applications of this formulation? In particular, the assumption that both F and g are convex further restricts the applicability.

2.	Many assumptions are presented without sufficient justification. For example, the convergence of ADMM with non-convex losses has already been studied in the literature, it is unclear why this paper focuses solely on the convex case. The authors should include a more comprehensive literature review on non-convex ADMM and discuss why the convex case is of particular interest here.

3.	The main idea of adding a projection step to ensure feasibility is straightforward and natural. The authors should carefully clarify whether they are the first to introduce projection into ADMM and compare their approach with existing works, such as “Projected Alternating Direction Method of Multipliers for Hybrid Systems.”

4.	What is the convergence rate of the proposed algorithm with respect to iteration? The authors should discuss this in detail and provide comparisons with existing results. Moreover, how does the privacy guarantee depend on the number of iterations?

5.	Does Proposition 3.1 strictly ensure $\tau \leq 1$ and satisfy the assumptions required in Theorem 3.4? It seems that additional strong convexity assumptions might be necessary. The authors should discuss this issue more carefully.

6.	Why does the algorithm require visiting each sample only once? What is the result if each sample is revisited multiple times, as in the general case?

### Questions
Why is the noise added outside the projection step in equation (5)? This design may lead to infeasible $x_k$.

### Soundness
2

### Presentation
2

### Contribution
1
