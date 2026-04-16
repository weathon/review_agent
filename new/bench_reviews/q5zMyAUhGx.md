## Summary
This paper gives learning-theoretic generalization bounds for KAN-style compositional networks in two settings: (i) layers whose activations are represented as linear combinations of basis functions, and (ii) layers modeled as low-rank RKHS-valued maps. The strongest contribution is the basis-function analysis, which develops a covering-number bound leading to excess-risk/generalization guarantees that can handle unbounded regression-type losses via truncation, and it is supplemented by a small empirical study relating a derived complexity proxy to excess loss during training.

## Strengths
- **Meaningful theory contribution for a timely architecture.** The paper addresses an important gap: KANs have attracted substantial interest, and the paper provides one of the first systematic generalization analyses for this family.
- **Coherent and technically nontrivial basis-function analysis.** Section 2.2 progresses cleanly from a layerwise covering-number decomposition (Proposition 1) to a basis-function entropy bound via Maurey sparsification (Proposition 2) and then to network-level complexity and risk bounds (Theorems 1–3). This appears to be a genuine extension of classical MLP covering-number machinery to learnable edge functions.
- **Broader-than-usual loss treatment.** Theorem 3 explicitly removes the bounded-loss requirement used in Theorem 2 by a truncation argument, allowing the theory to cover common regression losses such as squared, pinball, and Huber losses. This is a substantive advance over bounded-loss-only statements.
- **Interpretability of the resulting complexity measure.** The main complexity term is expressed in terms of layerwise coefficient norms and Lipschitz constants, which makes the bound reasonably interpretable and potentially useful for model design.
- **The paper is generally clear and mathematically organized.** The architectural definition, assumptions, and theorem flow are mostly easy to follow, and the paper is careful to note in Remark 2 that Assumption 2 is more general than the canonical additive edge-wise KAN form.

## Weaknesses

###: Fatal
None.

### Major:
- **The empirical section does not support the paper’s stronger “practical relevance” claims about the bounds.**  
  The paper repeatedly claims that the numerical study “demonstrate[s] the practical relevance of these bounds” (Abstract, Section 1.1, Section 3), but Section 3 does not evaluate the actual bound from Theorem 3/Corollary 1 or even an unnormalized instantiation of it. Instead, it studies a simplified proxy under the special case \( \Psi_i(0)=0 \), namely a quantity “proportional to” the complexity term, and then **normalizes it post hoc** so “the maximum value of the complexity measure is equal to the last value of the excess loss.” This only shows qualitative co-movement of a rescaled proxy with excess loss over epochs. It does **not** establish that the bound is numerically informative, non-vacuous, or predictive across settings. Given how prominently “practical relevance” is advertised, this overreach matters.
- **The low-rank RKHS part is only loosely tied to the KAN architecture as defined in the paper.**  
  The defining KAN structure in Eq. (1) is a matrix of **univariate edge functions** summed at nodes. The paper already broadens beyond this in Assumption 2, explicitly acknowledging in Remark 2 that the class considered is “more general than the additive structure in (1).” In Section 2.3, this broadening becomes more pronounced: the function class is a generic low-rank vector-valued RKHS composition class \(\Psi_l \in \mathcal A_{r_l}(R_l)\), not a class that enforces or exploits the characteristic edge-wise univariate KAN structure. This does not make the mathematics invalid, but it does mean the low-rank results are better described as bounds for a broader compositional function class inspired by or containing KAN-like models, rather than a tightly KAN-specific analysis.
- **The “no combinatorial dependence outside logarithmic factors” framing is somewhat overstated.**  
  The statement is literally true for the explicit dependence in Theorem 1, but the paper presents it in a way that may suggest width/basis-count insensitivity. In reality, the main complexity term \(\tilde\alpha\) depends on \(B_i\), \(C_i\), and especially products of Lipschitz constants \(\rho_j\), and these can themselves scale adversely with architecture size and parameterization. So the practical interpretation is weaker than the headline wording suggests.

### Minor
- **Potentially poor depth dependence is not discussed enough.**  
  The complexity term contains products of layerwise Lipschitz constants, e.g. \(\prod \rho_j\), so the bound can deteriorate rapidly with depth when these constants exceed 1. This is a common issue in norm-based deep-network bounds, so it is not unique to this paper, but it should be discussed more directly, especially because the paper emphasizes architectural guidance.
- **The practical estimation and meaning of the controlling constants are not sufficiently developed.**  
  The utility of the theory depends on quantities such as \(B_l\), \(\rho_l\), and in Theorem 3 also \(C', C'', s, s'\). The paper defines them clearly but gives little guidance on how tractable or stable these are in realistic trained KANs. The empirical section estimates \(\rho_j\) using upper bounds from Remark 5, but there is no discussion of how loose those estimates may be.
- **The condition in the RKHS result deserves more interpretation.**  
  Theorem 4 assumes \(\tilde d := \max_i d_i > \nu\). Since \(\nu\) is the Matérn/Sobolev smoothness parameter, the paper should say more about when this condition is natural or restrictive in practice.

### Trivial
- The regularization claim is best viewed as a forward-looking hypothesis rather than an experimentally supported conclusion. The paper itself partly acknowledges this in Section 4, so the main fix is just to tone down the earlier wording.

## Nice-to-Haves
- Report the **actual numerical value** of the generalization bound or excess-risk upper bound, not only a normalized complexity proxy.
- Add quantitative analyses: correlations, multiple seeds, and tests across architectures/hyperparameters.
- Include a sharper main-text comparison to the corresponding MLP-style bounds, especially clarifying whether KAN structure yields any real theoretical advantage beyond accommodating learnable activation functions.
- If keeping the RKHS section, explicitly reframe it as analysis for a broader class containing KAN-like networks, unless the authors can impose a genuinely KAN-structured RKHS formulation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“No comparison with MLP baselines” as an unfairness complaint.**  
  The paper is primarily a theory paper; while an MLP comparison could strengthen the empirical narrative, the absence of such a baseline is not by itself a core flaw. I retain only the milder version that the main text could compare more explicitly to MLP bounds.
- **Missing implementation minutiae for reproducing the complexity estimates (e.g., spline degree/grid details).**  
  These are reproducibility nitpicks rather than substantive reviewing points under the stated rules.
- **General complaint that the proof strategy is “just” adapted from prior MLP covering-number arguments.**  
  The paper explicitly states this lineage, and extending those tools to KAN-style learnable edge functions is still a legitimate contribution. This is not a reason to discount the work, though the paper could articulate the novelty more crisply.
- **Pure scope-creep requests for lower bounds.**  
  The paper itself notes lower bounds as future work. Their absence is not a defect severe enough to count as a main weakness here.

## Novel Insights
The paper’s strongest contribution is narrower than its framing suggests: the basis-function setting is a real KAN-relevant learning-theory result, while the RKHS low-rank section reads more like a general compositional-function extension than a KAN-specific theorem. This split matters for evaluation: the submission is not fatally flawed, but its best case is as a solid partial theory paper whose empirical and low-rank components are currently more suggestive than definitive. In other words, the work is strongest when read as “generalization analysis for KAN-motivated compositional models, with a particularly convincing basis-function specialization,” rather than as a comprehensive, practically validated theory of KAN generalization.

## Suggestions
- **Tone down the empirical claims** in the abstract, contributions, and conclusion unless the authors add experiments evaluating the actual bound values.
- **Show unnormalized bounds/proxies** and compare them directly to observed excess risk; this is the most important missing empirical check.
- **Clarify the scope of Section 2.3**: either reformulate it explicitly as a broader function-class result containing KAN-like models, or strengthen the connection to the edge-wise univariate KAN architecture.
- **Discuss depth dependence and the role of \(\rho_l\)** more candidly, including whether trained KANs typically keep these constants moderate.
- **Improve the KAN-vs-MLP comparison in the main text**, not necessarily experimentally, but by explaining more concretely how the bound’s dependence differs and what is genuinely gained by the KAN parameterization.

Originality is **good**: extending covering-number analyses to KAN-style learnable edge functions is nontrivial and timely. The research question is **important**, since KANs have seen growing interest and currently lack much theory. The core claims are **partly supported**: the main basis-function theory is solid and meaningful, but the practical-utility claims are overextended relative to the experiments. Experimental soundness is **limited but not disastrous** for a theory paper; the issue is not that experiments are required to be larger, but that the current ones do not validate what the paper says they validate. Clarity is **generally good**, with the main conceptual ambiguity being the mismatch between the KAN-specific framing and the broader function classes actually analyzed, especially in Section 2.3. Overall community value is **positive**: this is a useful theory contribution, but not yet a fully convincing one in its current framing.

## Score and Decision
**Calibration anchors used:**
- **/home/wg25r/review_agent/human_reviews/hiHZVUIYik.md** — accepted theory paper on generalization bounds despite vacuous/limited empirical sharpness, because the technical contribution was strong and clearly presented. This submission is somewhat **below** that anchor because its empirical claims are overstated and one part of the theory is less tightly aligned with the advertised architecture.
- **/home/wg25r/review_agent/human_reviews/Y7lc4aZ4iP.md** — rejected paper where reviewers felt the significance and claims around the bounds were not convincing, with depth-dependence concerns and limited validation. The present paper is **above** that anchor because the basis-function contribution appears more coherent and substantively useful, and the paper is better scoped mathematically.
- **/home/wg25r/review_agent/human_reviews/NkmJotfL42.md** — accepted paper with meaningful theoretical insight despite some overclaiming/presentation issues. This submission is **weaker** than that anchor in practical validation, but comparable in being a legitimate theory contribution with some framing excesses.
- **/home/wg25r/review_agent/human_reviews/oV72wHuRNy.md** — rejected paper where the theory was not presented convincingly enough and impact was limited. The present paper is **stronger**: clearer, more timely, and with a more concrete positive contribution.

Relative to these anchors, I view this paper as **borderline but slightly above reject**: the basis-function theory is real and publishable, but the paper should be more careful about what exactly is proved for KANs and what the experiments actually show.

**Score: 6.0 / 10**  
**Decision: Weak Accept / Accept if the venue values timely partial theory contributions; otherwise borderline.** For a binary decision here, I lean **Accept**.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>