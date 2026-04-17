# Counterfactual Regret Minimization for Sequential Equilibrium

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Computing the Nash equilibrium (NE) in the imperfect-information two-player zero-sum sequential game is an important problem. Finding some refinements of the Nash equilibrium is important because the Nash equilibrium may take sub-optimal actions in states that can not be reached in equilibrium. In this work, we improve the framework of the counterfactual regret minimization (CFR) algorithm, proving that our algorithm can converge to the refinements of the Nash equilibrium under some assumptions. The extensive-form perfect equilibrium (EFPE) and the sequential equilibrium (SE) are two refinements of the Nash equilibrium, they improve on this shortcoming of the Nash equilibrium by assuming that players make mistakes. Most current sequential equilibrium and extensive-form perfect equilibrium computing algorithms are not iterative algorithms and need to solve linear programs, which are ineffective on large-scale games. Our method gives a local perturbation in all the states in the game and gives a suitable perturbation descent method. We compare our Sequential Perturbed Counterfactual Regret Minimization (SPCFR) algorithm with CFR variants and the approximate EFPE computing algorithm, perturbed CFR. Experimental results show that our method outperforms existing CFR-based methods on popular games, including Kuhn Poker, Leduc Hold'em, and GoofSpiel.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Sequential Perturbed Counterfactual Regret Minimization (SPCFR) algorithm, an iterative algorithm designed to compute refinements of the Nash Equilibrium (NE), specifically the Sequential Equilibrium (SE), in two-player zero-sum imperfect-information games. The experiments show that SPCFR converges close to the best CFR variant in terms of exploitability, and can also approach SE well compared to Perturbed CFR.

### Strengths
- Important Problem: The paper tackles the important and challenging problem of computing equilibrium refinements beyond the standard NE. While NE strategies can prescribe suboptimal actions in parts of the game tree that are unreachable in equilibrium, refinements like SE provide a more robust solution concept.
- Novel Approach: Existing algorithms typically employ global local perturbation, while this paper use perturbation. The paper provides a convincing argument that a global perturbation that decays over time would lead to impractically slow convergence as the game tree depth increases, as the reach probability of deep nodes would diminish too quickly. This provides a strong rationale for the proposed local perturbation.

### Weaknesses
- The primary weakness of this paper is its lack of clarity. Upon my initial review, I assumed that this paper aims to addressed the problem of learning an EFPE. However, this paper only demonstrates the convergence to an SE.

- Symbol definitions are unclear. For instance,  $a$, $b$, and $\sigma$ in Equation (12) lack the definitions.

- The symbol definitions differ significantly from those in most CFR papers and the closely related work--Farina et al. [2017], which increases the difficulty for readers.

### Questions
- In the experiments, how were the hyperparameters $a$ and $b$ for the perturbation schedule in Equation 12 selected for each game?

- Why you do not compare with PCFR and PCFR+ (Predictive CFR and Predictive CFR+)? I notice that you use “PCFR” to denote Perturbed PCFR. However, “PCFR” usually denotes Predictive PCFR.

- Why not conduct comparisons on larger games, such as HUNL subgames?

- Why is there no established convergence theory indicating that SPCFR+ converges to SE? Is it due to that some assumptions cannot be satisfied?

### Soundness
2

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
This paper introduces the Sequential Perturbed Counterfactual Regret Minimization (SPCFR) algorithm, which extends the CFR framework to converge toward sequential equilibrium (SE) in imperfect-information sequential games. The key innovation is the use of local, decreasing perturbations at each information set, rather than a fixed global perturbation. The authors provide theoretical guarantees on regret bounds and convergence to SE under certain assumptions, and demonstrate empirically that SPCFR outperforms existing CFR variants and perturbed CFR in terms of exploitability and maximum information set regret in games like Kuhn Poker, Leduc Poker, and GoofSpiel.

### Strengths
-	The idea of using local, decreasing perturbations is novel and well-motivated, addressing limitations of fixed global perturbation methods.

-	Theoretical analysis is provided, including regret bounds and convergence proofs under specific assumptions.

-	Empirical evaluation uses standard metrics and games, showing consistent improvements over relevant baselines.

### Weaknesses
-	The theoretical convergence guarantee relies on the strong assumption that the limit of the strategy sequence exists, which is not sufficiently justified or discussed.

-	Empirical evaluation is limited to small-scale games, with no demonstration of scalability to larger or more complex domains.

-	Performance gains in some environments (e.g., Leduc Poker) are modest, raising questions about the practical significance of the improvements.

-	Lack of ablation studies or sensitivity analysis for key parameters (e.g., a and b) weakens the reproducibility and practical utility of the method.

-	The comparison with existing methods is incomplete, omitting recent non-CFR-based approaches for equilibrium refinement.

### Questions
-	Could the local perturbation scheme be integrated with sampling-based CFR variants (e.g., MCCFR) to improve scalability? If so, what would be the expected trade-offs?

-	Under what practical conditions does the assumption of a strategy sequence limit hold? Are there common game structures or settings where this assumption is likely to fail?

-	Have the authors considered evaluating SPCFR in larger games?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Sequential Perturbed Counterfactual Regret Minimization (SPCFR), an improved CFR-based algorithm for computing sequential equilibria in imperfect-information games. By introducing local perturbations that decay adaptively during training, the method overcomes the limitations of fixed-perturbation approaches like Perturbed CFR. The authors provide theoretical guarantees that SPCFR converges to sequential equilibrium under certain conditions and demonstrate experimentally on Kuhn Poker, Leduc Hold’em, and GoofSpiel that it achieves comparable exploitability to the best CFR variants while significantly reducing information-set regret.

### Strengths
The main strength of this paper lies in its meaningful motivation and direction. Extending CFR toward computing sequential equilibria addresses an important yet relatively underexplored problem. The proposed idea of using locally adaptive perturbations to refine equilibrium computation is conceptually appealing and shows potential for further development.

### Weaknesses
1. The front matter (Introduction, Related Work, Preliminaries) feels disproportionately long for a 9-page submission—the paper does not present its core contribution until the end of page 5. It would help to surface the main ideas earlier (e.g., add a brief Motivation section) and consider moving parts of Related Work to the appendix.
2. The figures need substantial improvement. 
  - In Figs. 1–2, text is too small, and the lines in Fig. 2 lack clarity. 
  - Using a log scale for the top row in Fig. 1 could make trends clearer. 
  - In Fig. 1, “Goofspiel” appears inconsistently (capitalization) and I am unsure what the “3” in the title denotes.
  - The sharp jumps in the Leduc plot (especially the lower-right panel of Fig. 1) likely indicate an implementation issue; this deserves investigation rather than being described as mere “fluctuation.” 
  - In addition, the parameters “$a$” and “$b$” for SPCFR are not explained in the text (they only appear in Algorithm 1).
3. I do not fully understand the necessity of fixing the functional form $\delta_T(I)=\frac{1}{a+b\,Q(I)}$. Theoretically, $\delta_T(I)$ only needs to be sufficiently small; a simpler or smaller schedule might suffice. Also, if the pseudocode is literal, then $Q(I)$ may change across different nodes within the same infoset in single iteration, implying $\delta_T(I)$ is not consistent within that iteration—please clarify whether this is intended.
4. The paper mentions Liar’s Dice (line 435) but does not include it in the experimental evaluation.
5. The experimental scope is relatively limited (which may be due to hardware or the full-traversal design). It would help to state explicitly in the main text why only a small set of environments is used, or to scale up (e.g., larger-deck Kuhn/Leduc/Goofspiel). Including more recent baselines such as PCFR [1] would also strengthen the comparisons.
6. The presentation could be more intuitive. Since the target equilibrium accounts for dominated strategies, it would be informative to show a case study—for example, in Leduc—where a player deviates into an off-path/dominated strategy and compare SPCFR’s response to CFR/DCFR.

---

[1] Farina G, Kroer C, Sandholm T. Faster game solving via predictive blackwell approachability: Connecting regret matching and mirror descent[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(6): 5363-5371.

### Questions
Refer to the previous section

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Sequential Perturbed Counterfactual Regret Minimization (SPCFR) algorithm, an iterative method designed to compute the sequential equilibrium (SE) in two-player zero-sum games. The core problem is that NE, which standard algorithms like CFR compute, can prescribe suboptimal actions in parts of the game tree that are unreachable under the equilibrium policy. SPCFR addresses this by building on the idea of a perturbed game, where all actions have a non-zero probability of being played. The key contributions are a method that uses a local perturbation at each information set, and a specific decreasing schedule for this perturbation that is dependent on the cumulative reach probability of that set. The authors provide a theoretical analysis suggesting the algorithm converges to an SE and present experimental results on several small games (Kuhn Poker, Leduc Hold'em, GoofSpiel) showing its performance relative to CFR variants and a fixed-perturbation baseline.

### Strengths
1. The paper tackles an important and well-motivated problem. The shortcomings of NE in extensive-form games are well-known, and developing scalable, iterative algorithms for computing refinements like SE is a valuable research direction.

2. The core idea of using a local perturbation that anneals based on the information set's visit coun is clever and intuitive. It elegantly adapts the exploration to parts of the game tree as they are encountered, which is a more principled approach than a global, fixed perturbation.

3. The inclusion of theoretical analysis (Theorems 1 and 2) to justify the algorithm's convergence properties is a significant strength. Providing proofs for convergence to SE, even under specific assumptions, lends substantial credibility to the proposed method.

### Weaknesses
1. The main claim of the paper is that SPCFR is an effective iterative method for computing SE. However, the experiments primarily compare against algorithms that compute NE (CFR, CFR+, DCFR). The most critical comparison would be against other iterative algorithms that aim for equilibrium refinements, such as the OOMD-based approach by Bernasconi et al. (2024) mentioned in the related work. Without this, it is difficult to assess the practical advantages of SPCFR over the state-of-the-art in this specific subfield. The comparison in Figure 2 is against a "perturbed CFR", but this seems to be a baseline implemented by the authors rather than a well-established algorithm from prior work.
 
2. The motivation for an iterative approach to SE is its potential scalability to large games where methods based on solving linear programs are infeasible. However, the experiments are conducted on very small benchmark games (Kuhn Poker, Leduc Hold'em, 3-card GoofSpiel). These games are not large enough to convincingly demonstrate the scalability and practical necessity of the proposed algorithm. A stronger case would be made by evaluating SPCFR on a game that is known to be challenging for non-iterative solvers.

3. Clarity on Theoretical Assumptions: Theorem 2 provides a necessary and sufficient condition for convergence that requires the cumulative reach probability of every node h to tend to infinity. This is a very strong condition. It is not immediately obvious how the proposed perturbation schedule guarantees this for nodes that are deep in the tree or are only reachable via multiple "mistakes". A more detailed discussion of how this assumption is met in practice would strengthen the paper.

### Questions
1. Could the authors elaborate on why other iterative methods for computing equilibrium refinements, such as the OOMD algorithm, were not included as experimental baselines? A direct comparison would be very informative.

2. The paper's motivation rests heavily on the need for scalability. Could you discuss the feasibility of applying SPCFR to a larger game, and why such an experiment was not included in the current version?

3. Regarding the condition in Theorem 2, how sensitive is the practical satisfaction of this condition to the choice of hyperparameters $a$ and $b$ in the perturbation schedule? Does a poor choice risk non-convergence for certain information sets?

### Soundness
2

### Presentation
2

### Contribution
2
