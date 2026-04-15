---

## Summary

This paper proposes a novel single-domain identifiability theory for latent variables in nonlinear causal models. The core contributions are: (1) a *SCM reduction procedure* that converts any SCM (with no directed paths among observed variables) into an equivalent *Powerset Bipartite Graph SCM* (PBG-SCM), where latent variables correspond to topology-indexed concatenations of exogenous sources; and (2) identifiability theorems for latent variables in PBG-SCMs under three conditions—global invertibility, mutual independence, and a novel *minimality* condition that prevents shared latent variables from absorbing information from private ones. Synthetic experiments validate that violations of each assumption degrade recovery performance.

---

## Claims and Support

**Claim 1: Any SCM with no directed paths among observed variables can be reduced to an equivalent PBG-SCM.**
*Partially supported.* The reduction procedure is clearly described and the conceptual argument is sound: exogenous variables are clustered by observed-descendant topology; unobserved endogenous variables are implicitly composed away; and the resulting graph is bipartite with powerset indexing. The paper correctly notes (Section 4) that "equivalent" means sharing the same marginal distribution on observed variables *and* on the concatenated exogenous variables — and this scoping is repeated in the conclusion. However, no formal theorem establishing this equivalence appears in the main text; the claim that substituting away unobserved endogenous variables preserves the structural equations correctly is stated but not proved. The detail is likely in Appendix A.7, but the main paper is thin on this.

**Claim 2: Latent variables in the basis model are identifiable under Assumptions 1(i–iii).**
*Supported, with important caveats.* Theorem 1 is a valid conditional statement: if both models are minimal (in the sense of Assumption 1iii), invertible, and independent, and produce the same distribution, they are equivalent. This is not circular—Proposition 5.1 provides an independent operational characterization of minimality via intrinsic dimension, and Corollary 5.1 shows minimality is automatically satisfied when the learned latent dimension equals IDim(z). The theorem is mathematically non-trivial. The concern is practical: minimality cannot be verified from data without knowing IDim(z) in advance, which the paper acknowledges only in the limitations section.

**Claim 3: All latent variables in a general complete PBG-SCM are identifiable under Assumption 2.**
*Partially supported.* Theorem 2 is stated and a proof sketch given; the detailed constructive proof is deferred to Appendix A.7 (removed from review). The hierarchical minimality condition in Assumption 2(iii) is more complex than in the basis case (it conditions on "lower" variables already being equivalent), and the main text gives insufficient justification for why this is the right generalization or when it holds. The sketch is plausible but unverifiable from the main paper alone.

**Claim 4: The proposed assumptions are mild/practical.**
*Weakly supported.* Invertibility and independence are indeed standard. However, the paper repeatedly calls minimality "mild" (e.g., "Such assumptions are quite mild" in Section 5.2) despite the fact that it is defined by quantifying over all alternative observationally equivalent models—a condition not verifiable from data. This overstatement is acknowledged only in the Limitations paragraph. The characterization is misleading relative to the paper's body.

**Claim 5: Experiments demonstrate necessity of each mechanism.**
*Partially supported.* The experiments show that using oracle dimension + CLUB substantially improves R² over ablations (Table 1, Fig. 4a), which is consistent with the theory. However, the "necessity" language is too strong: the experiments demonstrate that specific implementation choices matter for these synthetic tasks, not that the theoretical assumptions are individually necessary in a mathematical sense.

---

## Strengths

- **Novel minimality condition**: The key insight—that shared latents can "plunder" information from private latents unless constrained—is genuinely novel and cleanly formalized. Proposition 5.1 provides a concrete decomposition showing exactly how non-minimality manifests (z absorbs components of s₁ and s₂), and Corollary 5.1 operationalizes it via intrinsic dimension. This is a real conceptual contribution that explains a blind spot in the existing literature (most works implicitly set correct dimensions, masking the issue).

- **PBG reduction framework**: The SCM-to-PBG reduction is elegant: it provides a canonical form for reasoning about latent-sharing patterns and makes precise what is and is not identifiable from a general SCM (topology-indexed concatenations of exogenous clusters). This is a clean abstraction connecting general causal models to the bipartite structure for which the theory applies.

- **Single-domain identifiability**: Achieving block-wise identifiability with only single-domain observational data—without auxiliary variables, domain indices, or interventions—is a meaningful advance over much of the prior literature, which requires such extras.

- **Constructive proof and algorithm alignment**: The iterative basis-model construction for Theorem 2's proof directly informs the experimental algorithm (Fig. 4b), and the iterative identification results show high R² at each stage, providing tangible support for the constructive argument.

- **Nontrivial synthetic benchmarks**: The *Split* and *Fusion* datasets are globally but not locally invertible, probing genuinely harder regimes beyond simple concatenation.

---

## Weaknesses

### Fatal
None.

### Major

- **Minimality and hierarchical minimality are practically unenforceable without oracle dimension knowledge.** Both Assumption 1(iii) and 2(iii) are defined by quantifying over all alternative observationally equivalent models—this is a strong uniqueness-type condition. While theoretically valid (and explicitly connected to intrinsic dimension via Corollary 5.1), there is no method proposed or analyzed for estimating IDim from data. The paper's own Limitations section concedes: "the succeeded algorithms in our experiments still need pre-known knowledge of the intrinsic dimension of latent variables." The abstract and body text characterize this as "mild" and broadly practical, which is a significant overstatement. The practical gap between the theory and any real deployment is thus large and unaddressed.

- **Reduction and Theorem 2 lack formal proofs in the main text.** For a theory paper, leaving both the SCM reduction equivalence and the full proof of the general identifiability theorem to an appendix (whose contents are not visible here) is a meaningful gap. The informal statement of reduction equivalence (Section 4) does not verify that substituting away unobserved endogenous variables yields well-defined structural equations for all PBG-SCMs, and the two-step proof sketch for Theorem 2 does not address index overlap, termination, or correctness of the iterative decomposition. The general theorem carries the paper's broadest claim and deserves a more complete argument.

- **All experiments use oracle intrinsic dimensions, directly undermining the practical-contribution framing.** Correct dimensions are supplied as $d_z = 5$ (ground truth IDim(z)) and $d_s = 2$ in all success cases. This means the "minimality condition" is never enforced by the algorithm itself—it is enforced by the experimenter's privileged knowledge. This makes it impossible to evaluate whether the theory's key new assumption offers any algorithmic advantage over just knowing the correct dimensions, which is already standard practice in the literature.

### Minor

- **Hierarchical minimality (Assumption 2iii) is notably more complex and opaque than the basis-model minimality.** The condition fixes "lower" latent variables up to equivalence while shrinking "upper" ones, which interleaves the identifiability of different layers. The remark in Section 5.2 offers only intuition; there is no worked example or proof that this condition is implied by (or implies) some simpler structural property. This makes it harder to evaluate its satisfiability or to know when it may fail.

- **Experiments cover only size-2 (basis) and size-3 (general) PBG-SCMs.** Since the latent variable count grows as $2^n - 1$ and the constructive proof requires $O(2^n)$ basis models, scaling behavior is entirely unknown. Even $n = 4$ (15 latents, 4 observed) would test the combinatorial feasibility of the iterative approach.

- **The abstract and introduction overstate mildness.** Phrases like "mild conditions," "much easier to be satisfied in general scenarios," and "broad applicability" are not substantiated. A recalibrated framing—acknowledging minimality as a structural property of the model class whose practical verification requires IDim—would improve precision without weakening the core contribution.

### Trivial

- The R² metric for independence is not defined in the main text ("Detailed definition can be found in Appendix A.8"), making it difficult for readers to interpret Table 1 and Fig. 3/4 without consulting the appendix.

---

## Nice-to-Haves

- **Experiments with unknown intrinsic dimension**: A setting where $d_z$ is over-specified and an auxiliary penalty or sparsity mechanism is used (rather than oracle knowledge) would directly validate the practical relevance of minimality as a theoretical concept.
- **Scaling experiments**: Testing $n = 4$ or larger PBG-SCMs would evaluate whether the exponential latent growth creates practical obstacles for the constructive identification algorithm.
- **Semi-synthetic or real data**: Even approximate validation on a standard disentanglement benchmark (e.g., dSprites or Causal3DIdent with appropriate SCM structure) would substantially strengthen the practical impact claim.
- **Analysis of information loss in the SCM reduction**: Characterizing how different original SCMs can be while mapping to the same PBG-SCM would clarify what is genuinely lost in the reduction and when two original SCMs are indistinguishable by the theory.
- **Connection to minimal sufficient statistics / information bottleneck**: Grounding the minimality condition in established information-theoretic concepts could clarify its relationship to other regularity assumptions in the field.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**[R1] "The minimality assumption is tautological / the theorem is near-circular."** (Harsh Critic, Critical Issue 1.) Verified against text: Assumption 1(iii) says there is no *alternative model* with z' ≺ z producing the same distribution. Theorem 1 says if *two* models both satisfy this (plus invertibility and independence) and match on observations, they must be equivalent. This is not circular—Proposition 5.1 independently characterizes non-minimality structurally (z absorbs components of s₁, s₂), and Corollary 5.1 provides an operational substitute. The theorem is a genuine conditional uniqueness result, not a restatement of identifiability. **Kept the practical enforceability concern but removed the tautology framing.**

**[R2] "The reduction only identifies topology-based concatenations, not latent variables of the general SCM—this is misrepresented."** (Harsh Critic, Critical Issue 2.) Verified against text: the paper explicitly states in Section 4 that identified objects are "the concatenation of a set of exogenous variables in original SCM," and the conclusion says "we can guarantee the identification of (concatenations of) original exogenous variables in the finest grain." The framing is consistent; this is not a hidden overclaim. **Removed as a stand-alone weakness.**

**[R3] "No comparison with multi-domain identifiability baselines (iVAE, etc.)."** (Spark reviewer, Human Finder.) The paper's setting is single-domain; comparing to multi-domain methods is not a fair evaluation of single-domain algorithms. Additionally, multi-domain methods require domain index inputs unavailable in this setting. **Removed per the hard rule on unfair comparisons unfavorable to the baseline and per scope rules.**

**[R4] "No mention of missing related works X, Y, Z."** Per instructions, no missing-related-work criticisms are included as external sources cannot be verified.

**[R5] Pure reproducibility requests** (seeds, full training logs, hyperparameter tables in appendix). Removed per hard rule on trivial reproducibility concerns.

---

## Novel Insights

The most genuinely novel observation in this paper is the explicit characterization of *minimality* as a previously overlooked precondition for latent variable identifiability in shared-latent settings. Prior work implicitly enforced it by fixing known dimensions; this paper is the first to name it, prove that its violation leads to a specific structural failure mode (information plundering, Proposition 5.1), and show it is equivalent to correct intrinsic dimension setting (Corollary 5.1). This reframing clarifies why existing identifiability experiments—which typically supply the true latent dimensionality—succeed, and motivates a harder evaluation protocol: unknown intrinsic dimension. This is a clean insight that the community should absorb even if the paper's own experiments do not fully exploit it.

---

## Suggestions

1. **Formalize the SCM reduction as a theorem** (main text, not appendix): state exactly what is preserved (observational distribution + concatenated exogenous marginals), and provide a brief proof or tight proof sketch addressing the composition of unobserved endogenous variables.

2. **Recalibrate the "mildness" claim**: replace "mild" with "structural conditions on the true model class" and explicitly acknowledge in the abstract/introduction that minimality requires knowing IDim(z) in practice.

3. **Run at least one experiment without oracle dimension knowledge**: use dimension-selection techniques (e.g., model selection by validation loss, sparse bottleneck regularization) to test whether minimality can be enforced approximately without privileged information.

4. **Expand the main-paper proof of Theorem 2**: even a 1–2 paragraph sketch with the key induction step and a concrete size-3 example would substantially increase confidence in the general result.

5. **Define the R² independence metric inline** in the main text, not only in the appendix, given that it is a central evaluation tool.

6. **Test $n = 4$ or larger**: report at least qualitative findings on whether the iterative basis-model identification remains computationally feasible and empirically successful as the number of latent variables grows exponentially.

---

## Score and Decision

**Originality:** Good. The PBG reduction framework and the explicit minimality condition are genuine contributions not found in prior single-domain identifiability work.

**Importance of research question:** High. Single-domain latent variable identifiability without auxiliary information is a significant open problem.

**Claim support:** Moderate. The theorems are structurally sound but the main proof for the general case is a sketch, and the framing overstates practical mildness.

**Experimental soundness:** Weak. All experiments are synthetic, use oracle dimensions, and test only $n \leq 3$. The key practical gap (unknown intrinsic dimension) is never addressed empirically.

**Clarity of writing:** Moderate. The basis model section is clear; the hierarchical minimality condition and the reduction formalism are harder to parse.

**Value to community:** Moderate-good. The minimality insight is a genuine conceptual advance; the PBG framework is a clean abstraction. The theory-practice gap limits immediate impact.

The paper makes real theoretical contributions but has a significant gap between its stated practical ambitions and what is actually demonstrated. The proof of the general theorem is incomplete in the main paper, the mildness claims are overstated, and the experiments rely entirely on privileged information the theory is supposed to avoid. These are substantial but not fatal deficiencies—they call for revision, not rejection of the core ideas.

**Score: 5.0** — Borderline / weak reject. The paper has genuine merit and novel ideas, but in its current form the framing, proof completeness, and experimental depth do not meet the bar for acceptance.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>