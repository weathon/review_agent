# Understanding the theoretical properties of projected Bellman equation, linear Q-learning, and approximate value iteration

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
In this paper, we study the theoretical properties of the projected Bellman equation (PBE) and two algorithms to solve this equation: linear Q-learning and approximate value iteration (AVI). We consider two sufficient conditions for the existence of a solution to PBE : strictly negatively row dominating diagonal (SNRDD) assumption and a condition motivated by the convergence of AVI. The SNRDD assumption also ensures the convergence of linear Q-learning, and its relationship with the convergence of AVI is examined. Lastly, several interesting observations on the solution of PBE are provided when using $\epsilon$-greedy policy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the projected Bellman equation (PBE) and two algorithms used to solve it: linear Q-learning and approximate value iteration (AVI). 

It proposes (i) an SNRDD (strictly negatively row-dominating diagonal) condition as a sufficient condition for existence/uniqueness of PBE solutions and convergence of linear Q-learning, (ii) a second sufficient condition motivated by AVI (two infinity-norm contraction conditions), and (iii) examples highlighting pathological behaviors under $\epsilon$-greedy policies. 

The contributions are summarized in the introduction (existence/uniqueness under SNRDD; a second AVI-motivated condition; convergence proofs; examples where AVI converges but linear Q-learning does not, and vice-versa; $\epsilon$-greedy pathologies).

### Strengths
1. **Clear examples illustrating ε-greedy pathologies.**

Section 6 presents counterexamples showing how changing ε can lead to the existence of no, one, or multiple PBE solutions. This illustrates that discontinuous policies can undermine fixed-point guarantees even under SNRDD conditions.

2. **Comparative visualization of algorithm behavior.**

The examples and figures demonstrate cases where AVI converges but linear Q-learning diverges, and vice versa. These visual results help highlight subtle differences between projection-based and iterative schemes.

3. **Technically correct foundational results.**

The use of fixed-point theorems and local Lipschitz arguments in proving existence (Theorem 3.2) is mathematically sound.

### Weaknesses
1. **Lack of practical motivation for SNRDD and AVI conditions.**

   The paper repeatedly states that the theoretical properties of PBE, linear Q-learning, and AVI are “not well understood” (line 45) but does not clarify *why* such understanding is important. It remains unclear what practical benefit arises from identifying the SNRDD or AVI contraction conditions. Do they guide algorithm design, help diagnose convergence failure, or provide insight into stability under function approximation? Without this link, the results feel detached from practical reinforcement learning. For instance, while SNRDD guarantees uniqueness of the PBE solution (Theorem 3.2), the paper never demonstrates how this insight could be used to construct or modify algorithms in practice.

2. **No discussion of necessity or minimality of conditions (4), (6), and (7).**

   The analysis provides only *global sufficient* conditions for convergence, SNRDD in Eq. (4) and the AVI-motivated inequalities in (6)–(7). However, these are imposed uniformly for all $\theta$ in $\mathcal{D}$, which may be stronger than necessary. For example, Lemma 3.6 follows almost directly from the definition in (4), suggesting that the assumptions are conservative rather than tight. The paper never investigates whether local conditions near a fixed point would suffice, nor does it analyze the borderline cases when these inequalities fail. This omission leaves unclear what the true boundary of convergence is and whether the results could be sharpened.

3. **Lack of empirical or verifiable interpretation.**

   While the paper introduces two mathematical conditions (SNRDD and AVI contraction), it does not discuss how practitioners could verify them in realistic reinforcement learning settings. For instance, Eq. (4) requires checking a row-wise dominance property of a matrix involving the unknown transition structure and feature representation, which is something infeasible to compute in practice. The paper provides no numerical examples demonstrating whether these conditions approximately hold, nor any heuristic indicators that could help identify when a system might violate them.

4. **Unclear takeaway from ε-greedy pathologies (Section 6).**

   Section 6 presents examples where $\epsilon$-greedy policies lead to multiple or nonexistent PBE solutions as $\epsilon$ varies, illustrating discontinuities in the induced operator. While these examples are interesting, the paper does not extract a clear lesson: should $\epsilon$-greedy be avoided, or can the results motivate a modification (e.g., replacing it with softmax or continuous exploration)? Moreover, the examples are disconnected from the main theory: the paper does not explain whether SNRDD or AVI contraction fails in those cases or whether these examples reveal a fundamental limitation of the proposed conditions.

5. **Weak connection to prior convergence literature and unclear technical novelty.**

   The paper positions itself as deepening the theoretical understanding of PBE, yet many results (e.g., Lemma 3.6 and Proposition 3.13) appear to follow directly from standard definitions or fixed-point arguments. It is unclear what the main technical difficulty is or how it compares to earlier analyses such as Li et al. (2024) (*Operations Research*), Tsitsiklis and Van Roy (1997), or Baird (1995). Those works already explore convergence, bias, and sample complexity of Q-learning with general function approximation. In contrast, this paper’s focus on the *existence* of a fixed point lacks discussion on how such existence results affect convergence guarantees, estimation bias, or sample efficiency. Without this comparison, it is hard to see what new theoretical challenge the paper truly addresses.

### Questions
1. Beyond “not well understood,” what practical benefit arises from identifying SNRDD or AVI contraction conditions?

2. Are conditions (4)/(6)/(7) necessary near fixed points, or can weaker conditions suffice?

3. How could practitioners empirically verify these conditions?

4. What actionable takeaway should one draw from ε-greedy pathologies (Section 6)?

5. How do these results extend or differ from prior convergence analyses (e.g., Meyn 2024; Li et al. 2024)? What is the technical difficulty?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the projected Bellman equation (PBE) and examines its theoretical connections to linear Q-learning and approximate value iteration (AVI). It establishes sufficient conditions for the existence and uniqueness of PBE solutions—most notably through a strictly negatively row-dominant diagonal (SNRDD) property and an additional condition motivated by AVI. The authors analyze convergence of both AVI and linear Q-learning under these assumptions using contraction mappings and fixed-point arguments. They also present illustrative examples, including cases where convergence fails or yields suboptimal solutions under ε-greedy policies.

### Strengths
The paper tackles a fundamental and underexplored theoretical topic: when and why the projected Bellman equation admits a unique solution and how that affects linear Q-learning and AVI. The formal analysis seems to be technically sound, and the examples highlighting pathological convergence behaviors are informative. The paper is well-written and easy to follow in general.

### Weaknesses
The existence and uniqueness of PBE solutions under diagonal dominance (SNRDD) are not particularly surprising—such conditions are standard in numerical linear algebra and reinforcement learning theory. Consequently, Section 3, while technically correct, feels incremental and could be condensed substantially. Additionally, some of the technical assumptions, such as requiring relatively large regularization constants for certain guarantees, seem restrictive and analytically convenient rather than broadly insightful. These conditions limit the perceived generality and practical relevance of the results.

Further, given that MDP and RL is a very classical and well-sturdied field, many related works seems missing, to name a few: Convergence of Q-learning with Linear Function Approximation (Melo, Meyn & Ribeiro, 2008), and An Analysis of Linear Models and Value-Function Approximation (Parr et al., 2008). A lack of comparison and acknowledgement of previous works makes it difficult to evaluate how this paper advances beyond well-established results. It remains unclear which aspects are novel relative to known stochastic approximation and ODE-based analyses. The manuscript would benefit from clearer statements of what is newly proven here versus what is already established in the literature.

### Questions
See the Weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the projected Bellman equation (PBE) under linear function approximation and analyzes two solvers—linear Q-learning and approximate value iteration (AVI). It introduces a sufficient condition based on “strictly negatively row-dominating diagonals” (SNRDD) to guarantee existence/uniqueness of PBE solutions, connects that condition to AVI-style contraction conditions, and gives convergence proofs for several Q-learning variants via contraction/ODE arguments. It also presents examples showing (i) divergence/convergence mismatches between AVI and linear Q-learning and (ii) pathological behaviors under $\epsilon$-greedy policies (solution multiplicity, non-existence, and emergence of optimal yet unattainable solutions).

### Strengths
Strengths

Clear new sufficient condition (SNRDD) for PBE solvability that applies across tabular, linear, and regularized settings, and accommodates off-policy cases and (locally) Lipschitz policies. This is positioned as broader/different from prior on-policy or tamed-Gibbs results. 

Unifying convergence view for Q-learning variants (tabular asynchronous, linear, and regularized) via contraction theory/ODE analysis—reducing auxiliary assumptions (e.g., feature positivity/orthogonality; no target network/projection required for their regularized variant). 
Explicit comparison to AVI-motivated conditions and a formal relationship result (Proposition 3.13) clarifying when the SNRDD-based criterion aligns with an AVI-style norm bound. 

Insightful counter-examples: concrete constructions where AVI converges but linear Q-learning oscillates and vice-versa (Figure 2), and where Q-learning converges to a unique but sub-optimal fixed point. These help map the frontier between the two methods.

### Weaknesses
Scope restricted to linear function approximation: While the linear regime is foundational, many modern RL systems are nonlinear; the paper stops short of indicating which parts of the analysis might transfer (even qualitatively) to nonlinear function classes. (Authors briefly note this as future work.)

### Questions
Nonlinear approximation outlook: Which pieces of your ODE/contraction proof strategy seem most likely to carry over to nonlinear function classes (e.g., local SNRDD-like Jacobian conditions), and what are the main obstacles?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the existence and uniqueness of solutions for two projected Bellman-equation-based algorithms: linear Q-learning and AVI. The main conditions used in the paper are SNRDD, a matrix condition for determining the stability of dynamical systems, for linear Q-learning, and a matrix norm condition for AVI. The paper also provides a convergence analysis using ODE-based stochastic approximation (Borkar and Meyn, 2000) for the linear Q-learning algorithm, leveraging the SNRDD condition.

### Strengths
- The authors present a unifying tool (SNRDD) for analyzing the convergence of tabular, linear, and regularized linear Q-learning.
- An interesting contrast is provided, with conditions showing when AVI converges while linear Q-learning does not, and vice versa, as well as a condition under which both converge (Proposition 3.13).
- An extensive appendix with theoretical rigor is provided that helps guide readers through the definitions and results.

### Weaknesses
- My main concern is the novelty of the results provided. The paper centers on the SNRDD condition, which was already used by (Lim and Lee, 2024) for a similar purpose in regularized linear Q-learning. Although (Lim and Lee, 2024) required two additional conditions (e.g., orthogonal and non-negative features), as the authors indicated in Remark 4.5, I believe this is a bit incremental.
- The fixed behavior policy condition, although assumed in related works, is restrictive in my opinion. The authors mention that a replay buffer can be considered a fixed distribution, but the standard use of a replay buffer involves continual updates with recent experience. Therefore, I am not convinced that a replay buffer argument applies here.

### Questions
- Why is the fixed-behavior policy $\beta_\theta$ denoted by $\theta$ in Section 2.2?
- In the e-greedy case, how do the existence of an optimal solution and Q-learning’s ability to converge to it relate to the SNRDD condition?
- What are the implications of the conditions for the existence and uniqueness of the PBE solution (SNRDD and the AVI condition)? For example, can we make design choices in linear Q-learning based on the SNRDD condition to ensure convergence?
- (minor) In Definition 3.1 (SNRDD), is there a missing absolute value on the entries $A_{ij}$?

### Soundness
3

### Presentation
3

### Contribution
2
