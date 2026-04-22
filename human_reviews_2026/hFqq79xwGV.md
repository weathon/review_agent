# Finite‑Time Bounds for Distributionally Robust TD Learning with Linear Function Approximation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Distributionally robust reinforcement learning (DRRL) focuses on designing policies that achieve good performance under model uncertainties. In particular, we are interested in maximizing the worst-case long-term discounted reward, where the data for RL comes from a nominal model while the deployed environment can deviate from the nominal model within a prescribed uncertainty set. Existing convergence guarantees for robust temporal‑difference (TD) learning for policy evaluation are limited to tabular MDPs or are dependent on restrictive discount‑factor assumptions when function approximation is used. We present the first robust TD learning with linear function approximation, where robustness is measured with respect to the total‑variation distance uncertainty set. Additionally, our algorithm is both model-free and does not require generative access to the MDP. Our algorithm combines a two‑time‑scale stochastic‑approximation update with an outer‑loop target‑network update. We establish an $\tilde{O}(1/\epsilon^{2})$ sample complexity to obtain an $\epsilon$-accurate value estimate. Our results close a key gap between the empirical success of robust RL algorithms and the non-asymptotic guarantees enjoyed by their non-robust counterparts. The key ideas in the paper also extend in a relatively straightforward fashion to robust Q-learning with function approximation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work provides a non-generative robust TD algorithm and establishes its finite-time convergence bounds. It analyzes the required sample size under both total variation and Wasserstein distances. Notably, the convergence of the proposed robust TD algorithm does not rely on strict contraction assumptions. Overall, this work introduces a novel algorithm and offers a fundamental understanding of distributionally robust reinforcement learning (DRRL).

### Strengths
1.  This work provides the robust TD algorithm, and firstly does not rely on extra assumptions within the TV and Wasserstein uncertainty set. 

2. This work provides solid theoretical result.

### Weaknesses
1. The work approximates the Lagrange multipliers using linear function approximation. This raises a concern regarding whether the assumption is reasonable and whether the resulting approximation error can be sufficiently small, given that the Lagrange multipliers induce a nonlinear correction within the Wasserstein uncertainty set.

2. Is the parameter $\mu$ sufficiently small? The final result contains the term $\epsilon_{\text{approx}} / \mu$, which suggests potential instability or sensitivity to the choice of $\mu$.

3. There appears to be a typo in the definition of $G$ — the last line should use $G$ instead of $g$.

4. The definitions and assumptions in equations (19), (20), (22), and (23) in the appendix seem unreasonable or inconsistent with the main formulation. Further clarification or justification would be helpful.

### Questions
Please check the Weaknesses part.

### Soundness
3

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
4

### Summary
This paper derives a finite-time bound for distributionally robust TD-learning with linear function approximation. It primarily focus on the robustness with respect to the transition probability; that is, it pre-assumes the transition probability falls into a given set, named uncertainty set, and it considers the worst-case value function among the given uncertainty set. The linear feature is adopted to approximate the Q-function. The goal is to learn the robust Q-function under this setting. A general assumption is given to characterize the interested uncertainty set; two well-known uncertainty sets, wasserstein and total varioation uncertainty sets satisfy the given condition. Then the convergence guarantee is given based on the scheduler.

### Strengths
This paper is addressing an important problem in robust RL; that is, how to accurately and efficiently solve the robust value function. The presented results indicate that under the linear approximation (with some traditional requirements), the robust TD-learning will finally learn the robust value function in the desired complexity. 

 The theoretical analysis sounds to me as it matches the best-known complexity for the non-robust RL literature under the approperiate parameter setting. All related assumptions to guarantee the convergence are also used in non-robust RL literature. Taking the target network  also introduces a new technique in robust RL.

### Weaknesses
1. Turnning the optimization problem into a distributionally robust optimization problem and applies the dual form is not new. Assumption 1 simply says that this optimization problem can be solvable in the desired complexity. It typically cannot be considered as a novel contribution. 

2. The presentation is not very clear:
    1. Many notations are used without defining them or being defined somewhere hard to find. For example, $\mu$ and $C_e$ in Line 352. 
    2. And the complete form of $C^\star$, $C_1$, and $C_2$ are not very helpful here. And the author doesn't elaborate anything on them. 
    3. In Line 390, "it requires c to be chosen sufficiently large". But it is unclear why it is an issue; is there any problem for using  $1/\mu$?
    4. Some statements do not have explainations; see my questions below. 

3. No experiments validating the necessasity of the two-time scale. 

4. Writing issues:  The author commonly omitts punctuation marks in formulas.  Many LaTeX issues.

### Questions
1. Can the proposed Assumption 1 motivate other uncertainty sets beyond the total variation and Wasserstein? As those distances have been well-studied (e.g. the total variation has been covered by the IPM from *Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation* under linear function approximation), it is hard to find how useful the proposed concept is; it just sounds like for convenience of analysis.

2. The author commented that "Prior work Zhou etal.(2023) circumvents this by imposing restrictive assumptions on $\gamma$". I didn't see their restrictive assumptions. Can the author further clarify it?

3. How should I understand the complexity calculated in Corollary 1? Has the author included the proof somewhere?

4. Can author include some experiments? It is hard to tell if the two-time scale is necessary or more efficient. It would be better to
    1. Compare it with the single-time scale baseines.
    2. Compare with robust TD-learning in IPM uncertainty sets used in  Zhou et al.(2023)  *Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation*.
    3. Validate the compatibility of this approach with policy gradient or other policy gradient based algorthms.

### Soundness
3

### Presentation
2

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
The paper provides finite-time convergence guarantees for distributionally robust TD learning with linear function approximation. It analyzes a model-free, two time scale scheme. The paper derives a non-asymptotic error bound showing $O(1/\epsilon^2)$ sample complexity.

### Strengths
+ The paper provides a finite sample analysis of distributionally robust policy evaluation with linear approximation.
+ The paper is rigorous and mathematically sound (although I did not check all the proofs) giving finite-time analysis of distributionally robust TD with linear FA.
+ Useful theoretical work for robust RL.

### Weaknesses
- The results are restricted to linear function approximation under assumptions, which limits applicability to large-scale or nonlinear deep RL. 
- The analysis seem to require very restrictive and strong assumptions, e.g., Assumption 2 requiring that the policy induces an irreducible and aperiodic Markov chain under P0 (hence, mixing assumptions), bounded features, exact projection operator, etc. These conditions are generally unrealistic in RL environments which limits applicability. Also the constants hidden in $\tilde{O}$ may grow poorly with the problem parameters.
- I am not sure what the key technical novelty in analysis is in relation to prior robust TD or adversarial contamination analyses. 
- I did not see clear guidance on how to select the ambiguity radius or choose between different uncertainty sets (TV and Wasserstein) from data.
- There is no experimental validation. Some empirical tests on benchmarks would better demonstrate the utility of the bounds.

### Questions
1) What specific innovations allow finite time bounds compared to earlier analyses?
2) How tight are the obtained rates? Can you show matching lower bounds? 
3) Can the approach extend beyond linear FA?
4) Are the assumptions made in the paper necessary? Can some of the assumptions be relaxed or dispensed with?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses temporal-difference (TD) learning with linear function approximation for robust discounted Markov decision processes (MDP) under Total Variation (TV) and Wasserstein uncertainty in transition dynamics. Namely, the authors employ a two tier approach with varied time scales - one to address the inner optimization problem of the dual variables for each state-action pair and another to effectively parameterize the dual variables - which ultimately allow for the derivation of a sample complexity of $\tilde{O}(1/\epsilon^2)$ for uncertainty sets satisfying specific conditions.

### Strengths
- The problem being approached is highly relevant and motivated with respect to recent advances in theoretical robust RL.
- The use of a target network mechanism to overcome the projection mismatch arising from linear function approximation to facilitate stable convergence in finite-time enables the algorithm to alleviate the non-contractive nature of the projected robust Bellman operator $\Pi\mathcal{T}^\pi_r$ is fascinating. By having this occur at a "slower" rate, the algorithm can efficiently address the inner optimization problem of finding the worst-case distribution.
- Through the above, the author's were able to derive the final sample complexity in Corollary 1, which aligns with results in the non-robust case. This theoretical work then makes progress on closing the sim-to-real gap.

### Weaknesses
- While useful, this work has limited scope. Specifically, in practice one often wants to find some optimal policy $\pi^{*}$, not just evaluating some arbitrary policy, thus limiting the practical significance of the work. The authors briefly discuss a Q-learning extension, however, no formal justification is made, making it's contribution indirect.
- Perhaps more pressing than this is that I believe that there is an error in the proof of Lemma 2. Specifically, Part 3 of Assumption 1 requires an unbiased estimator for the objective function $F(\lambda^a_s)$, meaning that $\mathbb{E}[\sigma]=F(\lambda^a_s).$ The target function is then $F(\lambda^a_s)=\mathbb{E}[\min\{V(X),\lambda^a_s\}]-\delta\lambda^a_s$. On line 10 of Algorithm 1, you use equation 20 to find $\sigma(\cdot;\cdot,\cdot).$ However, by optimizing for $a=\lambda^a_s\in\\{-\frac{1}{1-\gamma},\frac{1}{1-\gamma}\\}$ as in equation 18, equation 20 would only hold if $\lambda^a_s=1$. But from line 122, $0<\gamma<1$ which would imply that the R.H.S of equation 20 should be $\min(V(S'),\lambda^a_s)-\delta\lambda^a_s$. As written currently, we can see that $bias=\mathbb{E}[\sigma]-F(\lambda^a_s)=\delta\lambda^a_s-\delta$. Putting this aside for a moment, how applicable is your algorithm in practice with the underlying assumptions?
- There is not any empirical validation of the proposed algorithm.
- The claim on lines 115-116 is incorrect, see [1].
- Significant number of typos and inconsistent notation. See below for suggestions on actionable edits.

[1] Zachary Roch, George Atia, and Yue Wang. A Reduction Framework for Distributionally Robust Reinforcement Learning under Average Reward. 2025.

### Questions
- $\Delta_\mathcal{S}$ on line 121 is not defined. Also, use different notation to mean the same thing on lines 235, 246, and elsewhere.
- Notation for the states and actions are not consistent/clear depending on it's respective use. i.e. use of $s, s', S', S_t$.
- Reuse of $r$ for both the reward function and to denote a robust MDP, robust value function, etc.
- Need citations for the $(s,a)$-rectangularity assumption, i.e. [2,3].
- Increased clarity by showing $\forall s\in\mathcal{S}$ and $\forall(s,a)\in\mathcal{S}\times\mathcal{A}$ when formally writing equations like on line 158 versus equation 50.
- $M$ used in Assumption 1 without definition.
- $X$ used without defining several places, i.e. line 239.
- $d$ used on line 268 before being defined. What is the difference between $\lambda_d$ and $d_\lambda$ as seen on lines 268 and 270, respectively?
- $\mathcal{M}_\nu$ used in equation 6 before being defined.
- Are $\theta^t_k$ and $\theta_{t,k}$ referring to the same thing?
- If $C_{mix}$ is the robust mixing time, it should be formally defined and discussed.
- $c$ is not defined in the algorithm. Similarly $C_e$ on line 352.
- Lemma 1 should come before Theorem 1. There is also not a clear distinction in the wording of these.
- "The noise term $n^\theta_{k+1}$ collects all remaining terms" on line 423 is not precise.
- Typo of "State" instead of "stae" on line 057, line 428, "this", on line 465, "introduction", in the uncertainty set on line 643, at the end of line 953, and a missing space on line 1492.
- Use of $W_\ell$ when discussing TV in the appendix.
- Reuse of notation starting on line 646 where $a\in[m,M].$
- Extra line in the equation on line 799.
- Incomplete sentence on line 819 and on line 1497.
- Is $\lambda(s,a)$ on line 825 the same as $\lambda^a_s$?
- $B_\nu$ is used in the appendix in equation 25 before being defined.
- Period on the wrong line in equation 26 and line 998.
- Does MDS on line 1026 refer to a Martingale Difference Sequence?
- What does the subscript of $\cdot_{op}$ refer to in the notations section in the appendix? Also, the notation section should go before the main proofs and where the notation is subsequently used.
- No clear distinction when a proof ends.

[2] Iyengar, G. N. Robust dynamic programming. 2005.

[3] Nilim, A. and El Ghaoui, L. Robust control of Markov decision processes with uncertain transition matrices.

### Soundness
1

### Presentation
1

### Contribution
2
