## Summary

This paper proposes a novel identifiability theory for latent variables in nonlinear causal models using only single-domain observational data. The core contributions are: (1) a formal reduction that transforms general structural causal models (SCMs) into equivalent powerset bipartite graph SCMs (PBG-SCMs), making the identification problem tractable; (2) a new *minimality* condition for shared latent variables that prevents information "plundering" from individual latents; and (3) proofs of identifiability (up to invertible transformations) under invertibility, independence, and minimality assumptions, first for a basis model and then for general PBG-SCMs. Synthetic experiments demonstrate that algorithmic mechanisms corresponding to each theoretical condition are necessary for successful latent recovery.

## Strengths

- **Novel single-domain identifiability theory with a clean minimality condition.** The paper introduces a theoretically well-motivated minimality assumption (Assumption 1.iii and 2.iii) that fills a gap in the identifiability literature. Theorem 1 provides a modular identifiability result for the basis model, and Theorem 2 extends it to general PBG-SCMs. This is a genuine conceptual contribution, as prior block-wise identifiability work typically requires multi-domain data, interventional data, or strong parametric restrictions (Section 2, Related Work).
- **Elegant SCM-to-PBG reduction.** The reduction process (Section 4, Definition 4.1) is a neat formalization that cleanly encodes which exogenous variables can be distinguished from observations. It allows the theory to apply to a broad class of causal graphs without restricting to additive or compositional forms.
- **Clear correspondence between theory and experiments.** The experiments explicitly map each theoretical condition to an algorithmic mechanism: autoencoders for invertibility, CLUB for independence, and dimension matching for minimality (Section 6.2). Table 1 shows that only when all three conditions are enforced (AE+CLUB, $d_z=5$) does identification achieve $R^2 > 0.95$, with severe degradation when either independence or minimality is violated.
- **Evaluation on challenging globally-but-not-locally invertible synthetic data.** The "Split" and "Fusion" datasets (Section 6.1) are constructed so that latent variables cannot be recovered from individual observed variables, yet the method succeeds, supporting the claim that global invertibility is sufficient.

## Weaknesses

### Fatal
None.

### Major
- **Practical enforcement of minimality relies on oracle knowledge of intrinsic dimensions, which the paper underplays.** Corollary 5.1 shows that setting the learned dimension equal to the intrinsic dimension is a sufficient substitute for minimality, and the Limitations section honestly states that algorithms "still need pre-known knowledge of the intrinsic dimension of latent variables." However, the Abstract and Introduction frame the assumptions as "mild" and claim the theory is broadly applicable, while the related work section asserts minimality is "much easier to be satisfied in general scenarios" than acquiring multi-domain data. In practice, knowing the exact intrinsic dimension of every shared latent is a strong oracle assumption that limits the practical applicability of the theory. The claims of mildness should be more heavily qualified.
- **Experiments demonstrate recovery, not identifiability (uniqueness).** The theory concludes that latent variables are uniquely determined up to invertible transformations under the stated conditions. The experiments show that a particular optimization procedure can find latent variables with high $R^2$ when endowed with the correct dimensions and independence loss. While this validates the algorithmic sufficiency of the conditions, $R^2$ does not test whether a *different* set of latents could also fit the observational distribution. No control experiment probes uniqueness (e.g., multiple random initializations with overcomplete latent spaces), which weakens the empirical support for the paper's central theoretical claim.

### Minor
- **The claim that minimality is "much easier to be satisfied in general scenarios" is not justified.** Section 2 asserts this without argument. For many real-world settings, acquiring multi-domain or interventional data may be easier than knowing the exact intrinsic dimension of latent variables a priori. The paper should either justify this claim or remove it.
- **The main text proof sketch for Theorem 2 is extremely terse.** Section 5.2 provides only a two-sentence sketch for the paper's main general identifiability result. While the detailed proof is in Appendix A.7, the main text offers little intuition for how the iterative decomposition preserves the required conditions through intermediate concatenated blocks. A slightly more detailed sketch or an explicit statement of the inductive invariant would help readers understand the proof strategy.

### Trivial
- **Figure 1 caption appears to mislabel reduced latent variable types.** The caption labels $u'_3$ as "unobserved endogenous" and $u'_1$ as "observed endogenous," but Section 4 defines the reduced latent variables $U'$ as concatenations of exogenous variables (with $S' = O$ being the observed endogenous variables). This is inconsistent with the text.

## Nice-to-Haves
- Include contemporary single-domain nonlinear ICA or disentanglement baselines (e.g., iVAE-style objectives) on the same synthetic data to contextualize the $R^2$ gains.
- Add a uniqueness probe: run multiple random initializations with overparameterized latent spaces to test whether all solutions achieving good reconstruction lie in the same equivalence class.
- Provide latent scatter plots for the general PBG-SCM experiments to make the claimed equivalence visually concrete.
- Derive a regularization term or architectural constraint that enforces minimality without oracle dimensions—this would significantly strengthen the practical value of the theory.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that Theorem 2 has a fatal proof gap in grouped-block minimality:** The critic argues that hierarchical minimality of individual latents does not obviously imply minimality of concatenated blocks during the iterative decomposition, rendering the proof unsubstantiated. However, the paper explicitly states that a "detailed constructive proof can be found in Appendix A.7." Per review guidelines, weaknesses about missing proofs in the appendix must be removed because the parser strips those sections; they exist in the original submission. Without access to Appendix A.7, the claim of a fatal gap is unverifiable.
- **Claim that minimality is "functionally equivalent" to assuming oracle knowledge of latent dimensions:** This is an overstatement. Corollary 5.1 establishes that matching the intrinsic dimension is a *sufficient* substitute for minimality, not that minimality itself is definitionally equivalent to known dimensions. Minimality remains a structural information-theoretic condition. That said, the practical limitation—that the paper's only demonstrated operationalization relies on oracle dimensions—is real and retained above.

## Novel Insights

The paper's honest acknowledgment that its algorithms require "pre-known knowledge of the intrinsic dimension" (Limitations) highlights an important tension in single-domain identifiability research: dimension estimation and identifiability are often treated as separate problems, but this work suggests they may be deeply coupled. A genuinely novel direction for the field would be to develop algorithms that jointly estimate intrinsic dimensions and enforce minimality, rather than treating dimension as an oracle input. The SCM-to-PBG reduction also provides a useful conceptual framework that could be applied beyond identifiability to other causal inference problems involving latent variables.

## Suggestions
- Prominently qualify the "mild assumptions" claim in the Abstract and Introduction by noting that the practical enforcement of minimality currently requires oracle knowledge of latent dimensions.
- Expand the proof sketch in Section 5.2 by at least one paragraph to explain how the iterative basis-model construction preserves minimality over concatenated blocks, or reference the specific lemma in the appendix that handles this step.
- If possible, include an experiment where latent dimensions are unknown and must be inferred or regularized, even if performance degrades, to better characterize the practical limits of the theory.

## Score and Decision

**Calibration comparison:**
- **High anchor** `/home/wg25r/review_agent/human_reviews/hrqNOxpItr.md` (avg 8.00, Oral): Strong identifiability theory with rigorous main-text proofs and extensive experiments on synthetic and real benchmarks. The paper under review is below this level due to its terse main-text proof sketch, synthetic-only experiments, and reliance on oracle dimensions.
- **High anchor** `/home/wg25r/review_agent/human_reviews/p60Y6o85Cj.md` (avg 6.60, Poster): Content-style identifiability with the key advantage of not requiring known latent dimensions. The paper under review has a cleaner theoretical framework but is weaker practically because it requires oracle dimensions.
- **Medium-accept anchor** `/home/wg25r/review_agent/human_reviews/ehr4oTe6XI.md` (avg 5.50, Poster): Disentanglement via optimal transport; reviewers found the motivation unclear but empirical results reasonable. The paper under review has a clearer theoretical story but more limited experiments.
- **Medium-reject anchor** `/home/wg25r/review_agent/human_reviews/5tSLtvkHCh.md` (avg 5.50, Reject): Temporal identifiability under non-invertible generation; rejected due to math mistakes, a wrong lemma, and poor terminology. The paper under review does not suffer from these correctness issues.
- **Low anchor** `/home/wg25r/review_agent/human_reviews/0sO2euxhUQ.md` (avg 4.00, Reject): Learning latent SCMs with no identifiability guarantee, linear Gaussian assumptions, and very limited experiments. The paper under review is clearly above this level because it provides genuine identifiability theorems without requiring known interventions.

The paper under review sits between the medium-accept and medium-reject bands. It has a real theoretical contribution (the minimality condition and SCM reduction) and honest limitations, but its practical applicability is constrained by the oracle-dimension requirement and its experiments do not test uniqueness. It is comparable to or slightly weaker than `p60Y6o85Cj.md` (6.60) because that paper's primary contribution was removing the known-dimension requirement. It is stronger than `5tSLtvkHCh.md` (5.50, Reject) because it contains no apparent mathematical errors. I position it at the lower end of the borderline band.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>