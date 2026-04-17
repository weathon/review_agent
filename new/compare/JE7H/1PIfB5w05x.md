---
job_id: 2ae6ffad-b2fc-4d7b-92d4-f7a563f0085a
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 1PIfB5w05x.pdf
paper: Price of Quality: Sufficient Conditions for Sparse Recovery Using Mixed-Quality Data
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper studies sparse recovery thresholds and Lasso under heterogeneous noise, which fits squarely within learning theory, optimization, and high‑dimensional statistics for ML.

## Minimum Quality
Pass ✅. The paper has Abstract, Introduction, methodological and theoretical sections (sampling complexity, algorithmic recovery), and a Conclusion. While there are no experiments, the work is positioned as purely theoretical; the main results are rigorous theorems with detailed proofs and quantitative guarantees, which serve as the “results” section. I see no fatal mathematical flaw or obvious re‑use of existing results without added contribution.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I did not find any hidden instructions, prompt injections, or other manipulative content in the provided text or figures.

---

# Expected Review Outcome:

## Summary

The paper studies sparse support recovery in a linear model where observations come from two sources with different noise variances (“high‑quality” and “low‑quality” samples).  

On the information‑theoretic side, it derives sufficient conditions on the pair of sample sizes \((n_1, n_2)\) for support recovery in agnostic and informed settings, and interprets these via a “Price of Quality” that quantifies how many low‑quality samples can replace one high‑quality sample.  

On the algorithmic side, it extends Wainwright’s Lasso support‑recovery analysis to the heterogeneous, agnostic setting and shows that the Lasso threshold depends only on total sample size and the average noise level, making high‑ and low‑quality samples interchangeable for algorithmic recovery.

## Strengths

1. **Well‑posed and timely problem.** The work formalizes a clean mixed‑quality data model in a canonical sparse regression setting, with two noise levels and both agnostic and informed decoders. This closely matches emerging practice (e.g., combining human and weak labels) but in a mathematically tractable form.

2. **Clear separation between information‑theoretic and algorithmic regimes.** The paper revisits the classic gap between \(n_{\text{INF}}\) and \(n_{\text{ALG}}\) under heterogeneous noise, and articulates how these thresholds respond very differently to changes in data quality. The “Price of Quality” construction is a useful conceptual device for this comparison.

3. **Non‑trivial theoretical extensions.**  
   - Theorem 1 and Theorem 2 extend union‑bound / Chernoff techniques from homogeneous to heterogeneous noise, yielding explicit sufficient conditions (Eqs. (9) and (16)) that reduce to the known thresholds when \(\sigma_1^2 = \sigma_2^2\).  
   - Theorem 3 requires reworking Wainwright’s Lasso analysis because \(\Sigma\) is no longer scalar; the use of QR decomposition and Haar properties of the orthogonal group (see the argument around Eq. (49) and Lemmas D.1–D.6) is technically competent and interesting on its own.

4. **Insightful characterization of “Price of Quality”.** The asymptotics of \(\gamma\) in different SNR regimes (Eqs. (13), (14), (19)–(21)) are quite interpretable. For example, in the agnostic setting, Eq. (14) shows \(\gamma \simeq 2 - \sigma_1^2/\sigma_2^2 < 2\) in the low‑SNR\(_2\) regime, whereas in the informed setting \(\gamma\) can diverge as in Eq. (20). This directly formalizes an intuition that variance‑aware weighting is much more sensitive to quality differences than black‑box algorithms.

5. **Algorithmic robustness result.** Theorem 3 shows that, in the agnostic setting, the Lasso’s sample size threshold \(n_{\text{ALG}} = 2s\log(p-s)+s+1\) is unaffected by heteroscedasticity; only the choice of \(\lambda_p\) changes, and it depends on the noise only through the average \(\sigma_{\text{avg}}^2\) (Eq. (6) and condition (28)). This is a surprisingly strong robustness statement and is clearly spelled out in Section 4.

6. **Mathematical care and completeness of proofs.** The paper provides detailed proofs in Appendices A–E. For example, Appendix A derives the heterogeneous Chernoff exponent carefully, with a transparent discussion of why solving the cubic (Eq. (37)) would tighten the bound but is avoided for tractability. Appendix D’s probabilistic control of \(U_i\) and \(V_j\) in Eq. (43)–(44) is also quite thorough.

7. **Potentially reusable technical tools.** The treatment of \(X_S^T \Sigma^2 X_S\) via Haar‑distributed \(Q\) and block‑diagonal \(\Sigma\) (Section D.2 and Lemma D.5) could be reused by others analyzing heteroscedastic linear models, beyond sparse recovery.

## Weaknesses

1. **No empirical illustration whatsoever.**  
   The main claims are entirely theoretical; there is not a single numerical experiment or even a synthetic simulation to illustrate the Price of Quality or the heterogeneous Lasso performance.  
   - For instance, the paper makes fairly strong, practically relevant statements such as “one high‑quality sample is never worth more than two low‑quality samples” (Section 3.1, Eq. (14)) and “high‑quality and low‑quality samples contribute equally to the Lasso threshold” (Section 4). A small simulation study, plotting recovery probability against \((n_1, n_2)\) and SNRs, would both validate the asymptotics and demonstrate how tight the sufficient conditions are at moderate dimensions.  
   - This is a missed opportunity to show that the asymptotic formulas match finite‑sample behavior, especially for practitioners considering mixed human/LLM labels.

2. **Agnostic information‑theoretic condition likely quite loose and somewhat opaque.**  
   The authors themselves acknowledge in Remark 3.2 that Theorem 1 is not expected to be tight, because they relax the cubic optimality condition (Eq. (37)) to a fixed \(\theta\) (middle of feasibility interval). However, the extent of looseness is never quantified.  
   - Eq. (9) assigns the exponent for low‑quality samples entirely in terms of \(\sigma_2^2\), ignoring \(\sigma_1^2\) in the \(n_2\) term, which already hints at potential sub‑optimality from the perspective of combining information from both groups.  
   - It would significantly strengthen the contribution to either (i) show analytically that optimizing (37) reduces to the homogeneous threshold when \(\sigma_1^2 = \sigma_2^2\) (as claimed in spirit) and characterize at least the leading‑order improvement in \(\gamma\), or (ii) empirically demonstrate how conservative Eq. (9) is. Right now, the reader is left with a sufficient condition of unclear tightness.

3. **Limited scope of algorithmic analysis (no informed Lasso or other algorithms).**  
   The introduction frames two settings (agnostic vs. informed), and the information‑theoretic analysis cleanly covers both. However, Section 4 only studies the Lasso in the agnostic setting.  
   - Remark 4.2 correctly explains why extending the Wainwright‑style argument to \(\Sigma^{-1}\)-weighted Lasso is nontrivial, but given how central the “informed vs. agnostic” dichotomy is to the story, the absence of any algorithmic analysis or even partial results in the informed setting is a notable gap.  
   - At minimum, I would expect some discussion of alternative algorithmic tools (e.g., generalized Lasso with known weights, or pre‑whitened design) and conjectured thresholds, or a partial negative result about difficulties in controlling \((X_S^T \Sigma^{-2} X_S)^{-1}\).

4. **Assumptions are quite strong and realism is only lightly discussed.**  
   - The Gaussian design \(X_{ij} \sim \mathcal{N}(0, 1)\), exact sparsity, and additive Gaussian noise are standard and acceptable for a first pass. However, many mixed‑quality data scenarios involve heavy‑tailed or correlated features, and label noise that is not additive Gaussian. The brief remark that “results naturally extend to sub‑Gaussian errors” (Page 3) is not substantiated with precise conditions or proof sketches.  
   - Remark 4.1 notes that correlated designs are not treated, but there is no discussion of which parts of the Lasso proof actually break for non‑identity covariance. Since Wainwright (2009) already handles correlated designs under RE‑type conditions, it would be useful to at least state how the heterogeneous noise interacts with such conditions and whether it breaks concentration or only complicates constants.

5. **No figures summarizing main phenomena; existing figures are opaque.**  
   The only visual material is the algebraic expansions shown in **Figure img‑0** and **Figure img‑1** in Appendix D.3, which are snapshots of intermediate steps in the derivation of Eq. (51).  
   - These figures are not referenced in the text and essentially duplicate the already heavy Einstein‑notation expansions on Pages 50–51. They do not help the reader understand the main narrative.  
   - In contrast, the core conceptual contribution is the behavior of the Price of Quality \(\gamma\) as a function of \(\text{SNR}_1,\text{SNR}_2\) (Eqs. (13), (14), (19)–(21)) and of the sample sizes. A simple phase diagram or contour plot of \(\gamma(s,\sigma_1^2,\sigma_2^2)\) and/or recovery regions in the \((n_1, n_2)\)-plane would make the theory much more interpretable. Right now, the reader must decode all behavior from logarithmic asymptotics.

6. **Some notational and algebraic inconsistencies in the appendices.**  
   There are several misprints that, while likely fixable, make the technical arguments harder to follow and could hide subtle algebraic issues:  
   - In Proposition A.1 (Page 14), the bound is stated with a term \(\frac{\delta(2\sigma_2^2 - \sigma_1^2)s}{2\sigma_2^2}\), but in Eq. (32) and later in Eq. (12) the denominator oscillates between \(2\sigma_2^2\) and \(2\sigma_2^4\). Eq. (14) also contains a \(\sigma_4^2\) that is presumably \(\sigma_2^4\). These inconsistencies propagate into the definition of \(\gamma\) and the approximations in Eqs. (13)–(14).  
   - The probability bounds in Theorem 2’s statement (Eq. (17)) contain “\(\}^{p\rightarrow+\infty}1\)” inline, which appears to be a malformed convergence notation.  
   These likely do not invalidate the main asymptotics, but they should be systematically cleaned up, and the derivation of Eq. (12) from Proposition A.1 re‑checked.

7. **No quantitative comparison to homogeneous benchmarks beyond asymptotics.**  
   The paper frequently claims that thresholds “match” or are “the same as” the homogeneous case. For example, Section 4 states that the Lasso threshold behaves like the homogeneous noise case with variance \(\sigma_{\text{avg}}^2\). While the asymptotic form of \(n_{\text{ALG}}\) is indeed identical, there is no discussion of constants or lower‑order terms that might be affected by heterogeneity. Since the analysis in Appendix D tracks many constants (e.g., in Lemma D.2 and D.5), it seems at least possible to comment on whether heterogeneity changes the second‑order regime.

8. **Limited discussion of practical implications and how to act on the results.**  
   The conclusion reiterates high‑level messages (e.g., “whenever possible, quantify uncertainty in the annotations and rescale the loss accordingly”), but does not give concrete guidance. For example:  
   - In the agnostic case, what rule of thumb on sample accounting should a practitioner use if they know only approximate noise variances?  
   - Is the statement that “one high‑quality sample is never worth more than two low‑quality samples” robust to model mis‑specification, or is it strictly an artifact of the binary‑signal Gaussian model?  
   A more careful discussion of robustness and what could plausibly hold beyond the stylized setting would make the work more actionable.

## Potentially Missing Related Work

I did not find citations or discussion of the following directly related papers, which should be considered for inclusion:

1. **Boufounos et al., “Sparse Recovery from Combined Fusion Frame Measurements,” 2009.**  
   This work studies sparse recovery from multiple measurement systems (fusion frames), dealing with heterogeneous measurement structures. The conceptual similarity to combining different “quality” sources makes it relevant. It should be discussed in the related work on mixed measurements and compared conceptually to the current \((n_1, n_2)\) trade‑off (likely in Section 1.1.1 or the beginning of Section 3).

2. **Wang, Wang, Xu, “On recovery of block-sparse signals via mixed \(\ell_2/\ell_q\) (0 < q ≤ 1) norm minimization,” 2013.**  
   This paper studies alternative regularizers for structured sparsity. While focused on block sparsity, it offers another angle on how different “groups” of measurements or coefficients are combined. It would fit into the discussion of algorithmic recovery methods in Section 4.

3. **Karahanoglu et al., “A mixed integer linear programming formulation for the sparse recovery problem in compressed sensing,” 2013.**  
   This provides an exact (but intractable) sparse recovery algorithm. Since the current paper contrasts information‑theoretic vs polynomial‑time regimes, it would be natural to mention such exact formulations as an alternative to the combinatorial MLE minimization in Eq. (8).

4. **Vu et al., “Adaptive matching pursuit for sparse signal recovery,” 2015.**  
   Adaptive greedy algorithms are another algorithmic family for sparse recovery. Including them in related work would help contextualize why the paper focuses on Lasso as the canonical polynomial‑time method and whether the robustness phenomenon might extend to matching pursuit.

5. **Yang et al., “A sparse recovery model with fast decoupled solution for distribution state estimation and its performance analysis,” 2019.**  
   This is an applied sparse recovery method with an efficient algorithm. While in a specific domain (distribution state estimation), it is an example of mixing structure and algorithmic efficiency; it could be cited as an application domain where heterogeneous measurement quality arises.

6. **Wu et al., “Sparse Parabolic Radon Transform with Nonconvex Mixed Regularization for Multiple Attenuation,” 2023.**  
   This work employs nonconvex mixed regularization in a sparse transform setting. It could be mentioned as a complementary algorithmic approach for sparse recovery under complex noise models, in Section 4’s discussion of alternative estimators.

7. **Yang et al., “Seismic reflectivity inversion with mixed L1-L2 norm regularization,” 2025.**  
   This is another example of mixed regularization for sparse inverse problems. While domain‑specific, it reinforces that combining different penalty structures (or measurement qualities) is an active theme and should be acknowledged.

8. **Blelly et al., “Sparse data inpainting for the recovery of Galactic-binary gravitational wave signals from gapped data,” 2021.**  
   This addresses sparse recovery from irregular / gapped data. The setting is different, but the idea of compensating for low‑quality or missing observations is related in spirit and could enrich the application discussion in Section 1.1.2.

These works do not undermine the core contributions but their inclusion would improve positioning and signal awareness of broader sparse recovery literature.

## Questions

1. **Tightness of the agnostic bound.**  
   Can the authors provide at least a partial quantitative assessment (theoretical or empirical) of how loose the sufficient condition (9) is compared to the best Chernoff bound obtained by solving the cubic (37)? Even a comparison in the homogeneous case would help calibrate the conservatism introduced by fixing \(\theta^\star = 1/(4\sigma_2^2)\).

2. **Informed algorithmic recovery.**  
   While fully extending Theorem 3 to the informed setting might be difficult, could the authors comment more concretely on whether they expect the Lasso threshold to still depend only on \(n\) and \(\sigma_{\text{avg}}^2\) when using a weighted loss \(\|\Sigma^{-1}(Y - X\beta)\|_2^2\)? Are there heuristic or partial arguments (e.g., under diagonal covariance of \(X\)) suggesting that the algorithmic Price of Quality in the informed case would differ qualitatively from the agnostic one?

3. **Robustness to non‑Gaussian noise and design.**  
   The paper states that “results naturally extend to sub‑Gaussian errors” but does not elaborate. Which parts of the proofs of Theorems 1–3 rely crucially on Gaussianity (e.g., exact chi‑square mgfs, Wishart inverses) versus mere sub‑Gaussian tails? Could the authors give precise moment or concentration assumptions under which their main theorems remain valid?

4. **Clarification of discrepancies in denominators and typos.**  
   Can the authors clarify the correct form of the Chernoff bound in Proposition A.1 and how it leads exactly to Eq. (12)? In particular, what is the correct denominator in the first logarithm, and is \(\sigma_4^2\) in Eq. (14) just a typo? Since these constants feed into the Price of Quality, it would be good to see a clean, consistent derivation.

5. **Practical interpretation of “one high‑quality sample is never worth more than two low‑quality samples”.**  
   This statement (Section 3.1, following Eq. (14)) is intuitively appealing but is very specific to the sufficient condition and the binary Gaussian model. Could the authors clarify in the camera‑ready that this should be interpreted as a property of the particular bound, not a universal impossibility result? Or, if they believe the factor 2 is close to fundamental, can they argue why?

6. **Why no simple plots or simulations?**  
   Given that the proofs are already fully worked out, is there a particular reason no simulations were included? If the authors have any preliminary experiments, even 1‑D phase diagrams for small \(p,s\), sharing them (or adding them to an appendix) would strongly improve the paper’s accessibility.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The main theorems and proofs appear technically sound and carefully developed; remaining issues are mostly about tightness and clarity rather than correctness.

## Presentation Rating

2: fair.  
The exposition is mathematically detailed but dense; important conceptual insights are buried in algebra, there are several notational inconsistencies, and the only figures are algebraic expansions that do not aid intuition.

## Contribution Rating

3: good.  
The paper addresses a relevant and nontrivial extension of sparse recovery theory to mixed‑quality data, introduces a useful “Price of Quality” perspective, and provides a technically solid Lasso analysis; the lack of experiments and partial coverage of the informed algorithmic setting prevent a higher score.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The theoretical contributions are meaningful and technically solid, particularly the characterization of the Price of Quality and the robustness result for Lasso under heterogeneous noise. However, the paper is purely asymptotic, lacks any empirical illustration or intuitive figures, and leaves the informed algorithmic side largely untouched despite emphasizing that dichotomy. With some polishing, additional numerical evidence, and tightened or better‑calibrated bounds, it would make a strong contribution.

## Reviewer Confidence

4: confident.  
I am familiar with sparse recovery literature and closely inspected the main derivations, though I did not re‑derive all intricate Haar‑measure and inverse‑Wishart calculations in the appendices.