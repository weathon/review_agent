## Summary
This paper proposes a highly ambitious causal identification framework for **functional longitudinal data**, where treatment, confounders, and outcomes are continuous-time stochastic processes on path space, potentially with censoring and death. The main theoretical contribution is a partition-to-limit construction intended to extend g-computation, IPW, and doubly robust identification to this infinite-dimensional setting, together with a density-style “almost nonparametric” result under piecewise continuous path spaces.

## Strengths
- **Targets a genuinely hard and underdeveloped problem setting.** The paper is not merely extending a standard discrete-time setup by minor notation changes; it explicitly tackles continuous-time, infinite-dimensional longitudinal data with treatment-confounder feedback, which is a real conceptual gap relative to standard longitudinal causal inference.
- **The estimand in Eq. (1) is broad and well-motivated.** It accommodates survival-type functionals, path functionals, and stochastic intervention regimes, allowing outcomes such as survival probability, restricted mean survival time, endpoint outcomes, and weighted path averages under stochastic treatment measures.
- **The partition-to-limit intervention strategy is conceptually meaningful.** Section 3.2’s construction—intervening on finite partitions and then letting mesh go to zero—is the natural way to formalize continuous-time interventions on path space, and if fully justified, could be a useful contribution.
- **Section 3.4 contains the paper’s most distinctive technical claim.** Theorem 4 does not establish full nonparametricity, but the density result over observed-data laws induced by full-data laws satisfying the assumptions is interesting and more substantial than generic rhetoric about being “assumption-lean.”
- **The paper is explicit about scope.** It clearly states that estimation and inference are left for future work, so the paper should be judged primarily as an identification/theory paper rather than an estimation paper.

## Weaknesses

###: Fatal
- **The empirical section does not validate the paper’s central causal identification claim.**  
  The main simulation explicitly removes the very difficulties the paper claims to solve: “we consider a simple setting where there is no mortality or censoring (\(T=C\equiv\infty\)), or other measured confounding process, except for the outcome process itself.” More importantly, Step 3 simulates treatment paths **directly from the target intervention law \(\mathbb{G}\)**, then simulates outcomes under that law, and finally averages them. This checks Monte Carlo approximation under the interventional distribution, not identification from observed confounded data. There is therefore no empirical evidence that the proposed assumptions or formulas recover counterfactual targets from an observational law with treatment-confounder feedback, censoring, or death. For a paper whose headline contribution is a new identification framework, this is a serious mismatch between claims and evidence.

### Major:
- **The main text overclaims “nonparametric” identification relative to what is actually stated in Section 3.4.**  
  The paper repeatedly describes the framework as nonparametric and as imposing “no restrictions on the observed data,” but Section 3.4 itself states: “Technically, we have not achieved full nonparametric paradigm,” and Theorem 4 is proved only when “the path space consists of all piece-wise continuous processes.” That is a qualified density result in a restricted path space, not literal unrestricted nonparametric identification. This is not a trivial wording issue because the nonparametric framing is central to the paper’s positioning.
- **The IPW and especially DR “generalizations” are stated at too abstract a level to be convincing as substantive results in the current presentation.**  
  Theorem 2 defines the IPW object abstractly via the Radon–Nikodym derivative \(d\mathbb P_{\mathbb G}/d\mathbb P\), but the paper does not provide a concrete representation of the weighting process for functional treatment paths or discuss conditions under which this derivative has a tractable form. Theorem 3 is weaker still: the key object \(\Xi(H,Q)\) is only defined as a limit “whenever it exists,” along with an additional limit-expectation interchange assumption (Eq. 22). Those are precisely the hard analytic points. As written, the DR result reads more like a formal conditional identity than an established, usable continuous-time DR theorem.
- **The core identification assumptions are not unpacked enough for claims of this breadth.**  
  Assumption 1 is the conceptual centerpiece, but it is highly nonstandard in appearance: it is formulated through total-variation closeness of conditional laws over infinitesimal windows, uniformly over treatment paths. The paper gives some intuition, but not enough explanation of how Eq. (9) serves as the continuous-time analogue of sequential exchangeability/no unmeasured confounding, nor enough main-text guidance on why it is sufficient for the partition-invariant limit in Proposition 1. Since Proposition 1 is the bridge from finite partition interventions to the target law, the lack of accessible proof architecture in the main text leaves the strongest claims under-supported.
- **The strongest wording about “resolving” uncountably infinite treatment-confounder feedback is too strong for the level of justification shown in the main paper.**  
  The paper does present a plausible mathematical route—intervene on finer and finer partitions and take the limit—but in the visible paper the crucial convergence and partition-independence arguments are asserted rather than made transparent. “Proposes a framework to address” would be better aligned with the evidence shown than “resolves.”

### Minor
- **The simulation section is also internally disconnected from several main-theory components.**  
  The paper claims support for handling mortality, censoring, and functional treatment-confounder feedback, but the main simulation removes all of these. Even if one accepts the paper as primarily theoretical, the current experiment functions only as a numerical sanity check for discretizing a path integral under a Gaussian process intervention.
- **Practical interpretability of the assumptions is limited.**  
  Assumptions 1, 2, and 4 are mathematically stated, but the paper gives little guidance on when they would plausibly hold in real monitoring settings such as ICU or CGM data, or what concrete failure modes would look like.
- **There are some notation/presentation issues that make the theory harder to audit.**  
  For example, the filtration definition appears to repeat \(\mathbb 1(X\le s)\), and Section 3.3 introduces \(\mathcal G_t\) in Definitions 1–2 without a clear earlier definition in the main text. These do not by themselves invalidate the claims, but they compound the difficulty of checking already-abstract arguments.

### Trivial
- None.

## Nice-to-Haves
- A simulation from an **observational** continuous-time data-generating process with genuine treatment-confounder feedback, plus censoring/death, would substantially improve the paper.
- A short main-text proof sketch for Proposition 1 and clearer sufficient conditions for Theorems 2–3 would make the theory much easier to evaluate.
- Reframing the contribution around an **“almost nonparametric” density result on path space** would make the paper’s strongest distinctive point clearer and more credible.
- A finite-grid comparison to standard discrete-time approximations would help readers understand what is gained by the continuous-time formulation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Requests for additional related work comparisons.** I do not include missing-related-work criticisms because external coverage cannot be verified here, and the paper already cites multiple strands of relevant work.
- **Complaints about lack of baseline comparisons where the paper provides no estimator.** Since this is explicitly scoped as an identification paper without an implemented estimation framework, demanding broad empirical baselines is partly scope creep. The real issue is not missing baselines per se; it is that the current simulation does not test identification at all.
- **Pure reproducibility or formatting nitpicks.** Typos, parser-related sign inconsistencies, and missing trivial implementation details are not central here, especially given the user’s warning about PDF extraction artifacts.
- **Criticism that the paper should include a real-data application.** This would improve impact, but for a theory/identification submission it is not by itself a core flaw.
- **The claim that the paper lacks practical implementation guidance because it omits estimation.** The paper explicitly states estimation/inference are out of scope. The substantive issue is instead whether the identification theory itself is convincingly established.

## Novel Insights
The paper’s most credible and potentially publishable angle is **not** the full trio of g-computation/IPW/DR generalizations, but rather the combination of (i) a path-space intervention construction via shrinking partitions and (ii) the density-style result in Theorem 4 showing that the identifying assumptions do not carve out a tiny observed-data model class, at least within piecewise continuous path spaces. If the paper were refocused around that narrower contribution, with toned-down claims about “nonparametricity” and much stronger exposition of Proposition 1, it would read as an ambitious theory paper with a specific conceptual advance rather than an overextended framework paper whose strongest components are unevenly supported.

## Suggestions
- Rework Section 4 so that it actually evaluates **identification from observational data**: generate treatment paths from a history-dependent observational mechanism, include time-varying confounding, and compare the identified target under \(\mathbb G\) to known truth.
- Tone down the framing from “nonparametric” to the more accurate statement already present in Section 3.4: an **almost nonparametric** or density-based result under piecewise continuous path spaces.
- Add a concise but real proof sketch in the main text for Proposition 1, explicitly explaining why Assumptions 1 and 2 yield partition-independent convergence in total variation.
- Strengthen Theorem 2 with concrete sufficient conditions or a more explicit pathwise characterization of \(d\mathbb P_{\mathbb G}/d\mathbb P\).
- Strengthen Theorem 3 by stating verifiable sufficient conditions for existence of \(\Xi(H,Q)\) and for Eq. (22), or otherwise weaken the DR claim to a formal abstract characterization.
- Clarify the relationship between Assumption 1 and standard sequential exchangeability/sequential randomization in continuous time.
- Clean up notation in Section 3.3, especially the filtration/process definitions around \(\mathcal G_t\), to make the claims more auditable.

## Score and Decision
**Novelty:** high. The target problem is genuinely difficult and underexplored, and the partition-to-limit path-space intervention idea is interesting.  
**Technical soundness:** mixed to weak in current presentation. The paper may contain substantial theory, but the main text does not support the strongest claims clearly enough, especially for IPW/DR.  
**Empirical support:** weak. The only experiment does not test causal identification and therefore does not substantiate the paper’s main contribution.  
**Significance:** potentially high if the theory is fully worked out, but current evidence does not justify that impact claim.  
**Clarity:** below the bar for a paper this abstract and technical; too many crucial points are asserted at a high level.

**Calibration against similar human-reviewed papers:**
- Compared with **0mtz0pet1z (Incremental Causal Effect for Time to Treatment Initialization, Accept Poster, scores 6/6/6/5)**: that paper combined identification with an estimator, simulations, and a real-data study. This submission is more ambitious theoretically but much less convincing empirically, and lacks comparable end-to-end support.
- Compared with **9vTAkJ9Tik (Doubly robust identification from multiple environments, Accept Poster, scores 6/8/8/6)**: that paper also made identification-heavy claims but supported them with stronger empirical validation across synthetic, semi-synthetic, and real data. The present paper is weaker on support and clarity.
- Compared with **jZffxvubJ9 (Treatment Rule Optimization Under Counterfactual Temporal Point Processes, Reject, scores 3/3/5/3)**: both papers are ambitious continuous-time causal works where reviewers were unconvinced by the gap between formal claims and demonstrated evidence. This submission is stronger conceptually and less obviously unsound than that reject, but shares the problem that the empirical section does not really validate the headline claim.
- Compared with **wsn1lPgDvU (Stable Estimation of Survival Causal Effects, Reject, scores 5/6/3)**: that paper had stronger estimation/experiment grounding even if presentation was uneven. This submission has broader theoretical ambition but less practical substantiation.
- Compared with **UWdPsY7agk (Efficient Causal Decision Making with One-sided Feedback, Accept Poster, scores 6/8/6/6)**: that paper paired identification theory with semiparametric efficiency and empirical evidence. This submission falls short of that bar.

Overall, this looks like a **promising but not yet acceptance-ready theory paper**. The problem is important and the ideas are interesting, but the current submission overstates what is established, and the empirical section fails to validate the core claim. For ICLR’s bar, I would reject in its current form, while acknowledging that a more focused and better-supported revision could become strong.

**Score: 4.4**

MY FINAL SCORE: <pineapple>4.4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>