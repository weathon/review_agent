## Summary

This paper establishes CLS-completeness for finding Nash equilibria in two-team zero-sum polymatrix games with independent adversaries, resolving an open question from Cai & Daskalakis (2011). The authors prove CLS-hardness via a novel reduction from MinQuadKKT using a "COPY" gadget to eliminate quadratic terms, and establish CLS-membership through a duality-based reformulation that avoids the Moreau envelope machinery used in prior work.

## Strengths

- **Resolves a recognized open problem:** The paper closes the complexity gap for two-team polymatrix games with independent adversaries, a question explicitly left open by Cai & Daskalakis (2011) who only resolved the three-team case (PPAD-complete). Theorem 3.1 (CLS-hardness) and Theorem 4.1 (CLS-membership) together establish tight CLS-completeness for this setting.

- **Novel COPY gadget construction:** The Stage I reduction (Section 3.1, Lemma 3.1) introduces a clever multilinear gadget $\text{COPY}(x_i, x'_i, y_i) := (x'_i - x_i \cdot (1 - 2\eta) - \eta) \cdot (y_i - 1/2)$ that forces variable duplication at KKT points, enabling the elimination of quadratic terms while preserving hardness. This technique may have applications beyond this specific problem.

- **Simplified membership proof:** The CLS-membership proof (Section 4) uses LP duality to reformulate the minmax problem as a minimization problem (Equation 6), avoiding the Moreau envelope machinery required in Anagnostides et al. (2023). This yields a more direct proof specific to the polymatrix structure.

- **Hardness for bilinear objectives:** The paper establishes CLS-hardness for finding stationary points of bilinear polynomial minmax objectives, improving upon Fearnley et al. (2024) who required quadratic (non-bilinear) polynomials.

## Weaknesses

### Fatal

None.

### Major

None.

### Minor

- **Symmetry construction could be more explicit:** The reduction in Section 3.2 constructs coordination game matrices $A^{a_i, a_j}$ from coefficients $\gamma_{ij}$ of the multilinear polynomial $M$. While Equation (2) explicitly sets $A^{a_i, a_j} = A^{a_j, a_i}$ (ensuring symmetry by construction), the paper does not explicitly note that this satisfies Definition 2's coordination game requirement $A^{i,i'} = (A^{i',i})^\top$. A brief remark clarifying that the construction inherently produces symmetric matrices would prevent reader confusion about whether symmetrization of the original quadratic coefficients is needed.

- **Membership proof relies heavily on appendix:** Theorem 4.1 (CLS-membership) depends entirely on Claim 4.1, whose proof is deferred to Appendix A.2. The main text provides no intuition for why KKT points of the dual minimization problem (Equation 6) correspond to Nash equilibria of the original game. Given that the equivalence between min-max stationary points and min-min dual stationary points is non-trivial, a 1-paragraph sketch in the main body explaining the key insight would improve verifiability.

- **Scope framing in Abstract could be clearer:** The Abstract states "Our main contribution is to prove that the two-team version remains hard" before qualifying that the tight bound applies to "the setting where one of the teams consists of multiple independent adversaries." While Section 5 and Theorem 1.1 correctly delineate that the general two-team case (with internal edges in Team Y) remains open (CLS-hard vs PPAD gap), earlier qualification in the Abstract would better manage reader expectations about which subclass is fully resolved.

### Trivial

None.

## Nice-to-Haves

- **Discussion of single-adversary case:** Section 5 lists the single-adversary case as an open problem. It would be helpful to briefly clarify why the main reduction (which produces $n$ adversaries) does not directly resolve the single-adversary setting—specifically, the dimensionality mismatch between $n$ binary players versus 1 player with $2^n$ actions.

- **Comparison to gradient-based approaches:** The paper mentions that gradient-based approaches yield $O(\text{poly}(1/\varepsilon))$ algorithms (adapted from Anagnostides et al., 2023). A brief remark on the practical implications of CLS-hardness (no $O(\text{poly}(\log(1/\varepsilon)))$ algorithm expected) versus what polynomial-time approximation schemes can achieve would strengthen the connection to multiagent learning applications.

## Removed Points

The following points from the Harsh Critic were flagged for removal with justification:

1. **"Methodological Gap: Unstated Symmetrization in Hardness Reduction"** — This criticism misunderstands the construction. Equation (2) explicitly defines $A^{a_i, a_j} = A^{a_j, a_i} = \begin{bmatrix} -\gamma_{ij} & 0 \\ 0 & 0 \end{bmatrix}$, which by construction satisfies the coordination game symmetry requirement $A^{i,i'} = (A^{i',i})^\top$. The paper builds symmetric matrices directly from the multilinear coefficients; no separate symmetrization step is needed. However, making this explicit would improve clarity, so this is retained as a Minor weakness about presentation rather than correctness.

2. **"Evidential: Lack of Main-Text Justification for Membership Proof"** — Deferring proofs to appendix is standard practice in theoretical computer science papers. The paper correctly states Claim 4.1 and notes the proof is in Appendix A.2. However, providing intuition in the main text would improve accessibility, so this is retained as a Minor weakness about exposition rather than a fundamental flaw.

3. **"Methodological Gap: Scope of 'Two-Team' Complexity Resolution"** — The paper does acknowledge the gap for the general two-team case in Section 5 and qualifies the main result in the Abstract and Theorem 1.1 to specify "independent adversaries." This is not an overclaim but rather a precise delineation of what is resolved versus what remains open. The framing could be slightly clearer in the Abstract, so this is retained as a Minor weakness about presentation.

## Novel Insights

The paper's primary novel contribution is the COPY gadget reduction that eliminates quadratic terms while preserving CLS-hardness, enabling the hardness result to apply to bilinear (rather than general quadratic) minmax objectives. This bridges the gap between optimization complexity and game-theoretic equilibrium computation in a technically elegant way. The duality-based membership proof is also a simplification over prior work, though it builds on known techniques rather than introducing fundamentally new machinery.

## Suggestions

1. Add a sentence after Equation (2) in Section 3.2 explicitly noting that the construction $A^{a_i, a_j} = A^{a_j, a_i}$ inherently satisfies Definition 2's coordination game symmetry requirement.

2. Include a 2-3 sentence intuition in Section 4 explaining why KKT points of the dual minimization problem (Equation 6) enforce the best-response conditions for both teams, before deferring the formal proof to the appendix.

3. Consider qualifying the Abstract's hardness claim earlier (e.g., "we prove CLS-hardness for two-team polymatrix games, and show this bound is tight when adversaries are independent") to immediately clarify the scope of the completeness result.

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| EOV1q1U23N.md (Regret Matching in Potential Games) | 7.00 | Resolves 2-decade open problem with convergence proofs; similar theoretical depth. This paper is comparable in resolving an open complexity question. |
| 7qNbWQTV26.md (DP Equilibrium in Polymatrix Games) | 6.00 | Proves hardness/impossibility results for polymatrix games; similar domain but this paper has tighter completeness characterization. |
| XXNexSaay2.md (Equilibrium beyond Unilateral Deviation) | 5.50 | Proves fixed-parameter lower bounds; this paper has cleaner completeness result (hardness + membership). |
| aUAMKEFwdo.md (Decomposition of Differentiable Games) | 4.00 | Theoretical game decomposition with restrictive assumptions; this paper has fewer limiting assumptions and resolves a more clearly stated open problem. |
| jh2Ffu97vS.md (Imperfect-Recall Games via SOS) | 5.00 | Theoretical framework with limited practical demonstration; this paper is more self-contained. |
| y8N45EEW05.md (Parameterized Hardness of NN Verification) | 6.50 | Proves W[1]-hardness results; similar theoretical complexity contribution. |

This paper resolves a recognized open problem (Cai & Daskalakis 2011) with sound proofs and no fundamental gaps. The weaknesses are presentational rather than substantive. Compared to EOV1q1U23N.md (7.00) which resolved a similar long-standing open problem about regret matching convergence, this paper is comparable in contribution quality. It exceeds the 6.00-6.50 range papers in clarity of the open problem resolved and tightness of the characterization (CLS-completeness rather than just hardness).

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>