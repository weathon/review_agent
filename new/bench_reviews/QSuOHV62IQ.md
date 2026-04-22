## Summary
The paper develops an identifiability theory for latent variables in nonlinear structural causal models with single-domain observational data. It reduces any SCM without directed edges among observed variables to an equivalent powerset bipartite graph SCM (PBG-SCM) and proves that, under global invertibility, independence, and a novel (hierarchical) minimality assumption, all latent blocks in a complete PBG-SCM are identifiable up to invertible transformations. Synthetic experiments instantiate these conditions via autoencoders and independence regularization to illustrate when identification succeeds or fails.

## Strengths
- Clear formalization of the identifiability problem, including precise notions of “subvariable,” equivalence via invertible transformations (Defs. 3.1–3.2), and an SCM-based setup that connects to existing causal representation learning work.
- A concrete SCM reduction procedure (Sec. 4) that clusters exogenous variables by their observed descendant sets, removes unobserved endogenous variables, and yields a PBG-SCM whose exogenous blocks correspond to concatenations of exogenous variables in the original SCM, preserving the marginal distributions on observed and concatenated exogenous variables (Eq. in lines 77–80).
- Introduction of a minimality / hierarchical minimality assumption with a clear mathematical definition (Assumptions 1(iii), 2(iii)) and its link to intrinsic dimension (Def. 5.1, Prop. 5.1, Cor. 5.1), which crisply captures the phenomenon of shared latents “stealing” information from private ones.
- Two-tiered identifiability results: a basis model theorem for the 2-observation case (Thm. 1) and a constructive extension to general complete PBG-SCMs (Thm. 2 with the sketch in lines 152–152 and Fig. 4b), providing an explicit hierarchical disentangling strategy.
- Experiments are carefully aligned with the theory: synthetic generators are constructed to satisfy invertibility and minimality (Sec. 6.1–6.2, 6.3), CLUB is used to promote independence, and ablations systematically remove independence and/or minimality (via overcomplete dimensions) to show significant drops in R² (Tab. 1, Figs. 3–4).

## Weaknesses

### Fatal
None.

### Major
- **Minimality is an inherently structural, non-operational assumption.**  
  Minimality (Assumption 1(iii), 2(iii)) is defined via the non-existence of *any* alternative model with the same observed distribution and smaller shared variable (e.g., “there does not exist a model … such that … and \(z' \prec z\)” at lines 118–119; similarly for hierarchical minimality at 148). This is a property of the full model class, not something testable from data, and in practice is indistinguishable from the identifiability property one is trying to prove. The only operational handle provided is Cor. 5.1, which reduces minimality to setting latent dimension equal to the (unknown) intrinsic dimension of the ground-truth z. In the experiments (Sec. 6.2, 6.3) minimality is enforced by hard-coding \(d_z\) and \(d_s\) to match simulator ground truth (lines 198–199, 216–218). Consequently, the main theorems characterize identifiability under an assumption that cannot be checked or enforced in realistic settings, and the empirical section presupposes exactly the oracle knowledge the theory is supposed to help with.

- **Dependence on knowing true intrinsic dimensions undercuts practical impact.**  
  While Def. 5.1 and Prop. 5.1 are conceptually clean, the way they are used (lines 130–132) essentially assumes prior knowledge of the intrinsic dimension of each shared latent: “if we know the intrinsic dimension of latent variables in advance … the minimality condition will automatically be satisfied by setting latent dimensions the same as ground truth.” The limitations section acknowledges that “the succeeded algorithms … still need pre-known knowledge of the intrinsic dimension of latent variables” (lines 224–224), but the main text repeatedly describes the assumptions as “mild” and of “broad applicability” (lines 35–39, 158–159, 220–222). In domains where intrinsic dimensionality is unknown and arguably part of the problem, the theory offers no mechanism to infer or constrain it, sharply limiting practical relevance.

- **Global invertibility of the full latent-to-observation mapping is strong and under-discussed.**  
  Assumptions 1(i) and 2(i) require a differentiable, globally invertible function \(g : \mathcal{S}\to\mathcal{V}\) with a differentiable inverse (lines 116–117, 144–145). The authors’ synthetic generators are deliberately made invertible by stacking full-rank MLP layers (lines 172–172, 174–174), and they note that some constructions are globally but not locally invertible (line 172). However, there is no substantive discussion of how restrictive global invertibility is in realistic problems where information loss, dimensionality mismatch, or many-to-one mappings are common. The abstract and conclusion still frame the assumptions as “mild” (lines 15, 35, 158–159, 220–222) without qualifying this strength. For much of the intended application space (e.g., images from complex scenes) this assumption will not hold, so the claims about broad applicability are overstated.

- **Experiments validate a narrow, idealized setting and do not probe robustness to assumption violations.**  
  All experiments use low-dimensional Gaussian latents passed through carefully constructed invertible MLPs, with independence and minimality enforced by design (Sec. 6.1–6.3). The ablations vary only (i) whether CLUB is used and (ii) whether the latent dimension is overcomplete (Tab. 1, Fig. 3, Fig. 4a). There are no experiments where independence is slightly violated, the generative map is only approximately invertible or noisy, or latent dimensions are under-specified. Nor are there tests on SCMs that are not PBG-SCMs to empirically illustrate the impact of the reduction. As a result, the empirical evidence supports only very modest, stylized claims (“in these well-specified invertible toy models, smaller latent dimension plus an independence penalty helps”), but does not substantiate the stronger narrative that the proposed assumptions are necessary/sufficient in any robust, practically relevant sense.

### Minor
- **SCM reduction and its interpretation of “equivalence” are somewhat conceptually slippery.**  
  The SCM reduction (lines 77–81) clusters exogenous variables by identical observed descendant sets, removes those without observed descendants, and discards all unobserved endogenous variables, yielding a PBG-SCM where each new latent corresponds to a concatenation of exogenous variables with the same descendant set. The text claims that the reduced SCM is “equivalent” in terms of the joint distributions on observed and concatenated exogenous variables (line 77), and that “if latent variables in a PBG-SCM can be identified, then the concatenated … exogenous variables in original SCM can also be identified” (lines 92–92). While correct by construction, this choice of equivalence means that any finer-grained distinctions among exogenous variables sharing descendants are declared in principle unidentifiable and collapsed at the definition stage. The paper states that it does not aim at structure identifiability (line 51–52), but later phrases like “identification of (concatenations of) original exogenous variables in the finest grain” (line 158) could mislead some readers into over-interpreting what is actually being identified.

- **Hierarchical minimality is technically opaque and its necessity is not explored.**  
  Assumption 2(iii) uses a condition involving the bitwise relation \(k \& i = 0\) (line 148) to formalize “upper” and “lower” variables in the PBG lattice, and the remark (lines 156–157) gives an intuitive explanation. However, the condition is quite technical, and the main text does not provide examples showing that it is close to necessary or that weaker variants would fail. This leaves some ambiguity about how tight or ad hoc this assumption is.

- **Scope limitation on graphs with no directed path among observed variables is under-motivated.**  
  The model assumes “there are no directed path between observed variables” (lines 65–65; reiterated in the conclusion and limitations). While common in some identifiability works, this excludes many natural settings (e.g., time series with observed dynamics). The paper acknowledges this only tersely in the limitations (line 224), without discussing how central this restriction is to the reduction and identifiability arguments.

- **Experiments are narrow in data type and baselines.**  
  All datasets involve low-dimensional Gaussian latent vectors and fully connected MLPs; there is no exploration of more realistic synthetic image-like data or non-Gaussian noise. Baselines are essentially AE vs AE+CLUB and correct vs overcomplete dimensions (Tabs. 1, Fig. 4a); there is no comparison to other single-domain identifiability frameworks or alternative independence mechanisms. This is consistent with a theory-focused paper but somewhat limits empirical insight.

### Trivial
- The explanation of the bitwise index/j notation and descendant relationships (e.g., in Def. 4.1 and Eq. (2)) could be made more accessible with a worked numeric example, as the current description may be hard to parse for some readers.
- High-level intuition for the constructive procedure in Thm. 2 is mostly deferred to Fig. 4b and the appendix; a more explicit walk-through for a small n in the main text could help.

## Nice-to-Haves
- Provide an operational proxy or estimator for minimality, e.g., via intrinsic dimension estimation, penalizing redundant information in shared variables, or regularizers that discourage shared variables from capturing idiosyncratic information—while being clear about the gap between such proxies and the ideal assumption.
- Add experiments probing near-violations of assumptions: small correlations between latent blocks, slightly non-invertible or noisy generators, under-specified latent dimensionality, and SCMs that are not PBG-SCMs prior to reduction.
- Offer more nuanced discussion of where global invertibility and exact independence might plausibly hold (e.g., invertible flow-based latent models) versus where they are likely violated, and position the work accordingly.
- Clarify the conceptual status of SCM reduction with respect to causal abstraction notions (intervention equivalence, etc.), even if only qualitatively.

## Removed Points
These points are flagged to be removed, treat them with caution.

- Critique that the reduction “wipes out any notion of structural causal identification” and “trivializes part of the causal question” by defining the unit of identification (Harsh Critic point 3). The paper is explicit that it does not aim to identify structure (lines 51–52) and that the reduced model is only required to preserve observed and concatenated-exogenous distributions (lines 77–81). Within this stated scope, the reduction is not misleading; it is a design choice.
- Claims that Thm. 1/2 proofs are too high-level or potentially incorrect due to missing main-text details. The paper consistently defers detailed proofs to the appendix (e.g., lines 152–152) and provides reasonable sketches; without appendix access we cannot substantiate allegations of mathematical errors.
- Concerns about missing measurability or continuity conditions in Defs. 3.1–3.2. While they could be more explicit, the paper consistently assumes differentiable mappings where needed (e.g., Assumptions 1(i), 2(i)), and this level of rigor is typical for the venue.
- Complaints about missing related work or specific alternative methods to compare against. Under the given instructions, we cannot verify missing references, and the existing related work section is reasonably comprehensive for this context.

## Novel Insights
None beyond the paper’s own contributions; the synthesized review primarily clarifies the practical limitations of the minimality and invertibility assumptions and the narrowness of the empirical setting.

## Suggestions
- Reframe the main claims to more clearly emphasize that this is a *conditional* identifiability result for a stylized but mathematically clean setting, rather than a broadly applicable recipe for real-world disentanglement.
- Expand the discussion around minimality and intrinsic dimension to (i) clearly label minimality as a non-testable regularity assumption, and (ii) outline possible algorithmic proxies and their limitations.
- Add at least one experiment that systematically violates each assumption slightly (independence, invertibility, minimality) to empirically characterize failure modes and robustness.
- Discuss in more depth when global invertibility might reasonably hold (e.g., models built from invertible flows) and when it is unrealistic, tempering the “mild” and “broad applicability” language.
- If space permits, provide a small, fully worked example (with explicit dimensions and functions) of the SCM reduction and the stepwise basis-model identification procedure for n=3 to improve accessibility.

### Overall Evaluation on Key Axes
- **Originality:** Moderately strong; the PBG-SCM reduction plus minimality-based identifiability for single-domain data is a nontrivial synthesis, though built on known ingredients (invertibility, independence).
- **Importance of question:** High; block-wise identifiability from single-domain data is a central open problem in disentangled/causal representation learning.
- **Support for claims:** Solid within the stylized assumptions, but the crucial minimality and intrinsic-dimension knowledge assumptions are not operationally addressed, limiting practical implications.
- **Soundness of experiments:** Methodologically sound but restricted to highly idealized, assumption-satisfying toy settings with limited robustness analysis.
- **Clarity:** Generally clear and well-structured; some technical definitions (hierarchical minimality, bitwise conditions) could use more intuition.
- **Value to community:** As a theoretical exploration of what identifiability would require under PBG-SCM structure and minimality, it is interesting; as a guide for practical identifiability in realistic settings, its impact is more limited.

## Score and Decision

### Calibration Anchors Used
- **Medium band (4–6):**  
  - `/home/wg25r/review_agent/human_reviews/5tSLtvkHCh.md` (avg 5.5, Reject): temporal causal identifiability with non-invertible generation; mixed reviews citing strong theory but execution/presentation issues. The current paper has somewhat clearer presentation and fewer outright math concerns but similar reliance on strong structural assumptions and synthetic experiments; overall comparable quality.  
  - `/home/wg25r/review_agent/human_reviews/0sO2euxhUQ.md` (avg 4.0, Reject) and `/home/wg25r/review_agent/human_reviews/kkQSwtx0p3.md` (avg 5.25, Reject): latent causal models with identifiability claims but limited empirical reach or over-strong assumptions. Our paper fits in this spectrum, probably around the middle-to-upper part.

- **High band (>7):**  
  - `/home/wg25r/review_agent/human_reviews/3cuJwmPxXj.md` (avg 8.0, Accept): strong, rigorously presented identifiability plus clear downstream task (intervention extrapolation) and well-argued assumptions. The current paper is less tightly connected to concrete downstream benefits and relies more on non-operational assumptions (minimality, known intrinsic dimension), so it is clearly weaker.  
  - `/home/wg25r/review_agent/human_reviews/2efNHgYRvM.md` (avg 8.0, Accept) and `/home/wg25r/review_agent/human_reviews/hrqNOxpItr.md` (avg 8.0, Accept): similarly strong identifiability works with well-justified modeling assumptions and clearer empirical stories. Our paper does not reach this level.

- **Low band (<3):**  
  - `/home/wg25r/review_agent/human_reviews/KpSNPeRuTf.md` (avg 2.5, Reject): adds a simple sparsity regularizer for counterfactual estimation with unclear causal guarantees and weak novelty. The current paper is significantly more substantial and rigorous than this low anchor: it has a nontrivial theory, carefully matched experiments, and a coherent narrative.

Relative to these anchors, the submission is meaningfully better than the low-quality causal/representation papers (avg <3), comparable to mid-range identifiability works that were rejected (avg ~5), and clearly below the strong accepted identifiability papers (~8). Given the conceptual interest but practical limitations of the assumptions, a calibrated score near the mid-range, slightly above the weaker mid-band examples, is appropriate.

**Final score:** 5.5  
**Decision:** Reject (the work is theoretically interesting but rests on non-operational minimality and intrinsic-dimension assumptions, and the empirical support remains too stylized for ICLR acceptance).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>