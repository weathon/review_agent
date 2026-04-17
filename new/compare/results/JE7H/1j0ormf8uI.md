---
job_id: 558fe176-4c92-4389-80e6-27af62bf1aa2
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 1j0ormf8uI.pdf
paper: Conformalized Survival Counterfactual Prediction for General Right-Censored Data
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.  
The submission includes abstract, introduction, related work, method, theory, experiments (synthetic + real), discussion, and appendices.

## Topic Compatibility
Pass ✅.  
The work is on conformal prediction, causal counterfactual inference, and survival analysis for right‑censored data, squarely within ML / UQ / causal reasoning and healthcare applications.

## Minimum Quality
Pass ✅.  
The paper is technically nontrivial, presents new methodology and theorems, has reasonably detailed experiments, and is written in clear English. No fatal flaws, missing core sections, or obvious misuse of data are apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No signs of prompt injection, hidden instructions, or other manipulative content targeting automated reviewers are present.

---

# Expected Review Outcome:

## Summary

The paper proposes a conformal prediction framework for constructing lower prediction bounds (LPBs) for counterfactual survival times under different treatments in the presence of general right censoring. The key idea is to use a counterfactual censored quantile regressor and a reweighting scheme that maps the target coverage event to a weighted conformal problem defined on uncensored treated units, leading to marginal coverage guarantees. The authors establish finite‑sample and asymptotic guarantees, including a doubly robust property with respect to the censoring/propensity weight model and the quantile regression model, and empirically evaluate the method on synthetic data and a real lung cancer dataset.

## Strengths

1. **Addresses a real gap: exact marginal coverage for counterfactual survival under general censoring.**  
   The paper targets the combination of three difficult aspects: (i) counterfactual outcomes, (ii) right‑censoring beyond Type‑I, and (iii) exact marginal coverage instead of PAC‑type guarantees. Existing work such as Candès et al. (2023), Gui et al. (2024), and Davidov et al. (2025) cover only subsets of this triangle. The proposed reweighting + conformal scheme provides a clean route to exact marginal coverage under strong ignorability and overlap.

2. **Conceptually simple but technically nontrivial reduction to weighted conformal prediction.**  
   Section 4.1 shows how the target coverage probability  
   \[
   \mathbb{P}(T(w)\ge \hat L_{N,n}^{(w)}(X))
   \]
   is upper‑bounded by a weighted expectation over uncensored treated units using the Radon–Nikodym derivative \(\omega(x) = d\mathbb{P}_X / d\mathbb{P}_{X|W=w,e=1}\) (Eq. (3)). This reduces counterfactual survival calibration to weighted conformal prediction under covariate shift (Lei & Candès, 2021). The derivation in Eq. (1) / Page 5 is one of the more interesting technical pieces, even though it could be written more clearly.

3. **Finite‑sample coverage bound explicitly quantifying density‑ratio error.**  
   Theorem 4.1 (Page 6) gives a non‑asymptotic guarantee
   \[
   \mathbb{P}(T(w)\ge \hat L_{N,n}^{(w)}(X)) 
   \ge 1-\alpha - \tfrac12 \mathbb{E}_{X|W=w,e=1}|\mathfrak D(X)-\omega(X)|
   \]
   after normalizing \(\mathfrak{D}\). This is a nice sharpening compared to purely asymptotic statements; the dependence on the L1 error of the density‑ratio estimate is explicit and useful for reasoning about how weight estimation quality affects coverage.

4. **Doubly robust asymptotic guarantee.**  
   Theorem 4.2 (Page 7) and its underlying Corollary B.4 show that asymptotically, marginal coverage holds if either the weight function \(\gamma(x)\) (equivalently \(\omega(x)\)) is consistently estimated (Assumption A1), or the conditional quantiles are well estimated under Assumption A2. While the conditions are strong, this is a nontrivial extension of doubly robust ideas into a conformal counterfactual survival setting.

5. **Reasonably thorough synthetic evaluation that stresses several relevant axes.**  
   Figure 1 evaluates coverage and LPB across six synthetic settings that vary dimensionality, censoring mechanisms, and treatment/censoring rates (Tables 2–3), comparing “Uncab”, “Naive”, “Focus”, “Fused” and the proposed method. The plots show that the proposed method generally hits coverage closest to the nominal 0.9 line while achieving competitive or larger relative LPBs than other valid methods, especially against “Focus” and “Fused” (which only have PAC‑type guarantees). Figure 3 further shows robustness under injected extreme outliers, where the baselines’ coverage visibly deteriorates while the proposed method remains around the nominal line.

6. **Real clinical application with some interpretable patterns.**  
   On the 541‑patient lung cancer dataset, Figure 4 shows that the method produces near‑90% coverage across different radiochemotherapy regimens and provides LPBs that align with known clinical insights (e.g., higher LPB for VMAT vs. IMRT, and for regimens with induction or concurrent chemo). Figure 5 further explores LPB adaptiveness across clinical and radiomic covariates (stage, T/N stage, KPS, Max3D‑Diameter, Voxel‑Volume), showing expected monotonic trends, which suggests the LPB is capturing meaningful prognostic structure rather than being purely noise.

7. **Sensitivity and robustness analyses.**  
   The appendices include several useful checks: effect of sample size on coverage / LPB (Figure 6), adaptiveness on synthetic covariates (Figure 7), dependence on \(p(W=w,e=1)\) (Figure 8), different quantile regression models (Figure 9) and different classifiers for the weight function (Figure 12). Table 7 gives runtime comparisons among calibration schemes. Collectively, these support the claim that the method is not overly brittle to specific implementation choices.

8. **Algorithmic clarity.**  
   Algorithm 1 on Page 4 gives a succinct description of the procedure, including how weights \(\bar p_i(x)\) and \(\bar p_\infty(x)\) from Eq. (2) are used to compute the weighted conformal quantile. This is quite helpful for readers who know weighted conformal prediction but are less familiar with survival settings.

## Weaknesses

1. **Key derivations around Eq. (1) are opaque and somewhat sloppy, obscuring the main identification argument.**  
   The chain of equalities and inequalities that leads from the marginal miscoverage \(\alpha\) to the weighted expectation in Eq. (1) (Page 5) is central, but the notation and conditioning are messy:
   - The line labeled “(i)” writes a conditional probability \(\mathbb{P}(T \le \bar q_\alpha^{(w)}(x) - c_{1-\alpha}^{(w)}(\tau) | X=x, W=w)\) inside an outer \(\mathbb{E}_X[\cdot]\) without clearly justifying why integrating over \(X\) w.r.t. \(\mathbb{P}_X\) is appropriate given that only units with \(W=w, e=1\) are observed. The step from the unconditional \(\mathbb{P}_{X,T(w)}\) to this conditional form should be spelled out formally.
   - The use of “\(\stackrel{(iii)}{\leq}\)” and “(iv)” is hard to parse: inequality (iii) supposedly follows from Lemma A.1, but that lemma operates on events of the form \(\{T\le \bar q_\alpha^{(w)}(x)-c_{1-\alpha}^{(w)}\}\) and \(e=1\), and the exact substitution of its bound into the sequence is not explained.  
   - The use of both \(\hat T\) and \(\widetilde T\) and the notational redefinition \(\gamma(x) = p(W=w,e=1|x)\) (and later \(\bar\gamma\), \(\mathfrak D\)) in the same equation line is confusing. For a central result like this, a cleaner derivation that keeps track of all conditioning and uses consistent symbols for observed vs. potential outcomes would significantly improve clarity and reduce the risk of subtle mistakes.

2. **Assumptions for the doubly robust property (Theorem 4.2 / Assumption A2) are quite strong and not well‑argued in terms of realism.**  
   A2(i) requires the conditional density \(\mathbb{P}(T(w)=t|X=x)\) to be uniformly bounded away from zero and infinity on an interval \([q_\alpha^{(w)}(x)-r, q_\alpha^{(w)}(x)+r]\) for all \((x,t)\). In survival problems with heavy tails or multimodality, this is nontrivial and likely to be violated. A2(ii) requires the quantile estimation error \(\mathcal{E}_N(X)\) and the inverse‑probability weights to satisfy moment conditions like Eq. (5), including  
   \[
   \lim_{N\to\infty}\frac{\mathcal{E}_N(X)}{\bar\gamma_N(X)} = \lim_{N\to\infty}\frac{\mathcal{E}_N(X)}{\gamma(X)}
   \]
   which is quite opaque and not interpreted at all in the main text. These conditions are central to the doubly robust claim, but the paper does not discuss their plausibility in typical clinical right‑censoring setups or provide any empirical checks (e.g., plots of estimated density ratios, histograms of 1/γ). As a result, the doubly robust label feels more formal than practically meaningful.

3. **Limited discussion and empirical analysis of weight estimation quality, even though it directly enters the coverage bound.**  
   Theorem 4.1 shows that the coverage shortfall is proportional to \(\mathbb{E}_{X|W=w,e=1}|\mathfrak D(X)-\omega(X)|\). However:
   - In experiments, the weight function \(\gamma(x)\) is always estimated using a Random Forest classifier with fairly ad‑hoc hyperparameters (Appendix D), and there is no assessment of how accurate the resulting \(\widehat{\omega}\) is, nor whether alternative estimators (e.g., logistic regression, boosted trees) would substantially change coverage and LPB. Figure 12 somewhat checks robustness across classifiers, but only at the level of final coverage/LPB rather than directly probing the error term in (4).
   - The synthetic setups in Table 3 have reasonably large \(p(W=w,e=1)\) (e.g., 32.8–60.6%), which is a relatively benign regime for weight estimation. It would be more convincing to see an explicit stress test where \(p(W=w,e=1)\) is pushed closer to zero, not just varied moderately as in Figure 8, and to observe how coverage deteriorates in line with the bound.
   - In the real data, treatment groups like consolidation chemo (Table 4) have moderate prevalence, but the paper does not report diagnostics for γ(x) estimation (e.g., proportion of extremely large weights, effective sample size). This matters because conformal calibration under extreme weights can be numerically unstable even if the theory allows it.

4. **Handling of censoring mechanism and potential informative censoring is under‑discussed.**  
   Assumption 3.1 requires \(\{T(1),T(0)\}\perp\!\!\!\perp(W,C) \mid X\), i.e., both treatment and censoring are ignorable given observed covariates. In realistic oncology datasets, censoring can depend on unobserved frailty or post‑baseline dynamics (e.g., toxicity), so this is nontrivial. The paper mentions in Remark 3.2 that independence of T and C has been assumed in Kalbfleisch & Prentice (2002), but does not discuss how violations would affect the proposed calibration. For example, the conversion of miscoverage into expectations over \((W=w,e=1)\) in Eq. (1) directly uses this assumption. Some sensitivity analysis (even a simple simulation with informative censoring) or at least a qualitative discussion would be important, especially since the claimed advantage over PAC‑type methods is precisely better behavior in rare/extreme cases, where informative censoring is more likely.

5. **Exposition issues: notation overload and inconsistent symbols make the paper harder to read than necessary.**  
   A few concrete examples:
   - The same symbol \(e\) is used for the failure indicator \(e = 1\{T<C\}\) throughout, but in Assumption A2 and Theorem 4.2 there is also \(\epsilon\) (or \(\varepsilon\)) used ambiguously, and in several places the subscript “\(\epsilon=1\)” appears where it should clearly be \(e=1\). This is minor but pervasive.
   - The use of \(\overline{q}_\tau^{(w)}\), \(\widehat{q}_\tau^{(w)}\), and \(q_\tau^{(w)}\) is not fully consistent, especially around the definition of the non‑conformity scores \(V_i^{(w)} = \overline{q}_\tau^{(w)}(X_i) - \widetilde{T}_i\) vs. later steps that talk about “true” vs “estimated” quantiles.
   - Equation (3) defines \(\omega(x) = d\mathbb{P}_X/d\mathbb{P}_{X|W=w,e=1}\), while elsewhere \(\gamma(x) = p(W=w,e=1|x)\) and \(\mathfrak D(x) = 1/\bar\gamma(x)\) are introduced. The interplay between these quantities is easy to check algebraically, but the text does not show this explicitly; remarking that \(\omega(x)\propto 1/\gamma(x)\) from Bayes’ rule early on would help anchor the reader.
   - The statement of Theorem 4.1 uses \(\mathfrak D(x)\) and then the narrative paragraph immediately below refers to “\((\varpi_N(x)-\omega(x))\)”, which is a different symbol; this kind of notational drift is distracting.

6. **Experimental baselines do not include other recent conformal survival or doubly robust approaches that look highly relevant.**  
   The comparisons focus on Uncab, Naive, Focus, and Fused (the Davidov/Gui family) and do not consider:
   - Methods that do doubly robust conformal survival under right censoring (e.g., Sesia & Svetnik, 2025),
   - Weighted conformal survival under covariate shift (e.g., Shin et al., 2025),
   - Conformal prediction for counterfactual outcomes under runtime confounding (e.g., Barnatchez et al., 2026).  
   While some of these are more general or focus on different aspects (e.g., runtime confounding), they seem technically close enough that at least a discussion and possibly a small synthetic comparison would be appropriate. Their absence weakens the positioning of both the theoretical and empirical contributions.

7. **Clinical evaluation is interesting but relatively thin as a causal study.**  
   The real‑data section essentially treats the method as an LPB‑producing black box and then narratively aligns LPB differences with prior clinical knowledge (e.g., VMAT vs. IMRT). However:
   - There is no attempt to validate counterfactual predictions via, e.g., comparing LPB ranking with some external risk score or with time‑dependent ROC curves of observed outcomes.
   - The treatment assignment model and potential confounding structure in the hospital cohort are not described beyond listing covariates. Without even basic balance diagnostics or propensity summaries, it is hard to judge whether the identification assumptions are remotely plausible.  
   Given that the main conceptual advance is about *counterfactual* survival LPBs, it would be good to at least show that the method yields sensible pairwise treatment comparisons for matched or stratified subgroups.

8. **LPB optimization over \(\tau\) is underdeveloped and might be data‑hungry.**  
   Section 4.1 suggests choosing \(\tau^*(x) = \arg\max_\tau (\bar q_\tau^{(w)}(x) - c_{1-\alpha}^{(w)}(\tau)(x))\) per test point. In practice, this still requires estimating \(c_{1-\alpha}^{(w)}(\tau)\) for a grid of \(\tau\)’s from a finite calibration set that is already filtered to \(W=w, e=1\). Table 1 and Figure 11 show that optimal \(\tau^*\) is often close to \(\alpha\) and LPBs are similar, suggesting limited benefit relative to simply fixing \(\tau=\alpha\). However, the algorithmic and statistical cost of this additional optimization (multiple quantile levels) is not quantified, and there is no guidance on a principled grid of \(\tau\). A more thorough discussion or ablation (e.g., how often \(\tau^*\) lands at boundaries, variance of LPBs across \(\tau\)) would help.

9. **Minor but non‑negligible issues with clarity and organization.**  
   - Some proof sections (e.g., Appendix B.1) are extremely long and rely on deep probabilistic inequalities (Rosenthal, von Bahr–Esseen) without giving intuition for key intermediate bounds like Eq. (27) or (30). For readers not already fluent in this literature, this significantly raises the barrier to assessing the correctness.
   - The introduction mixes causal‑effect language (“treatment effect on survival time”) with LPB construction, but the paper never actually defines a conformal LPB for the *difference* \(T(1)-T(0)\) or any contrast. All results are per‑treatment. This is fine, but the framing could be tightened to avoid over‑promising.

Overall, I do not see a fatal flaw, but the combination of strong/opaque assumptions and somewhat under‑analyzed weight estimation keeps the work at “good but not outstanding.”

## Potentially Missing Related Work

1. **Sesia & Svetnik, “Doubly Robust Conformalized Survival Analysis with Right-Censored Data”, 2025.**  
   - Relevance: Develops conformal survival methods with a doubly robust structure for right‑censored data, very close to this paper’s focus on doubly robust LPBs under censoring.  
   - Suggested addition: Discuss in Section 2 (Related Work) alongside Meixide et al. (2024) and Qin et al. (2025), and compare the structure of their doubly robust score with Theorem 4.2 / Corollary B.4, clarifying similarities and differences in assumptions and what “double robustness” means in each context.

2. **Shin et al., “Weighted Conformal Prediction for Survival Analysis under Covariate Shift”, 2025.**  
   - Relevance: Uses weighted conformal techniques specifically for survival analysis with covariate shift, which is conceptually close to the weighting by \(\omega(x)\) in Eq. (3) and Algorithm 1.  
   - Suggested addition: Cite in Section 4.1 where weighted conformal prediction is introduced, and contrast their covariate‑shift setting with the treatment‑ and censoring‑induced shift considered here; clarify why your particular construction of \(\omega(x)\) is appropriate for potential outcomes.

3. **Barnatchez, Josey, Nethery, “Debiased Machine Learning for Conformal Prediction of Counterfactual Outcomes Under Runtime Confounding”, 2026.**  
   - Relevance: Provides conformal prediction intervals for counterfactual outcomes with debiased machine learning and runtime confounding; relevant to the general theme of counterfactual conformal prediction and doubly robust ideas.  
   - Suggested addition: Mention in Section 2 in the sub‑paragraph on conformal counterfactual inference, after Lei & Candès (2021) and Jin et al. (2023), and discuss how their debiasing compares to your density‑ratio weighting, and to what extent their framework can or cannot handle censoring.

## Questions

1. **Clarification of the key inequality in Eq. (1).**  
   Could you provide a cleaner, self‑contained derivation of the inequality chain labeled (i)–(iv) on Page 5, explicitly showing each conditioning step and exactly how Lemma A.1 is used? In particular, please clarify the transition from  
   \[
   \mathbb{E}_X[\mathbb{P}(T(w)\le \bar q_\alpha^{(w)}(x)-c_{1-\alpha}^{(w)}(\tau)\mid X=x)]
   \]
   to the weighted expectation over \(W=w,e=1\) units, and the precise role of ignorability and independence of \(T\) and \(C\).

2. **Plausibility and interpretation of Assumption A2(ii).**  
   Can you give more intuition for the condition
   \[
   \lim_{N\to\infty}\left[\frac{\mathcal{E}_N(X)}{\bar\gamma_N(X)}\right] = \lim_{N\to\infty}\left[\frac{\mathcal{E}_N(X)}{\gamma(X)}\right], 
   \]
   and for the constraint on \(\mathbb{E}[1/\bar\gamma(X)^{1+\delta}]\) in Eq. (5)? Under what kinds of data‑generating processes and estimators would you expect these to hold, especially in high‑dimensional clinical settings?

3. **Behavior under informative censoring.**  
   Have you explored, even in simulation, what happens if \(T(w)\) and \(C\) are dependent given \(X\) (e.g., via a shared frailty term)? Would you expect the coverage guarantee to degrade gradually or catastrophically, and could a modified weighting scheme (e.g., joint modeling of censoring) mitigate this?

4. **More direct evaluation of density‑ratio estimation.**  
   For the synthetic settings where the joint \((X,W,e)\) is known, can you compute \(\omega(x)\) analytically and report the empirical \(L_1\) error \(\mathbb{E}_{X|W=w,e=1}|\hat\omega(X)-\omega(X)|\) alongside coverage? This would make the finite‑sample bound (4) more interpretable and help readers understand the gap between theory and practice.

5. **On the necessity and benefit of optimizing over \(\tau\).**  
   Given Table 1 and Figure 11 suggest that \(\tau^*\) is often close to \(\alpha\) and gives only modest LPB gains, is the per‑test‑point optimization actually necessary? Would a simpler strategy, like fixing \(\tau=\alpha\) or using a small fixed set of nearby quantiles, suffice in practice and reduce variance in \(c_{1-\alpha}^{(w)}(\tau)\) estimation?

6. **Treatment effect comparisons.**  
   Since the introduction motivates “LPB for treatment effect on survival time”, is there a principled way within your framework to construct an LPB (or at least a conservative bound) for \(T(1)-T(0)\) or for a contrast in survival time across treatments? If not, could you clarify in the introduction that the scope is per‑treatment counterfactual LPBs, and discuss whether extensions to contrasts are feasible?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3 good — Theoretical development is substantial and largely sound, with explicit finite‑sample and asymptotic guarantees, but relies on strong and somewhat opaque assumptions (especially for double robustness) and lacks deeper empirical probing of weight estimation error.

## Presentation Rating

2 fair — The main ideas are present and the structure is standard, but notation is overloaded, several key derivations (e.g., Eq. (1)) are hard to follow, and some theorem conditions are stated without adequate intuition.

## Contribution Rating

3 good — The paper makes a meaningful contribution by bringing exact marginal conformal coverage to counterfactual survival with general censoring and by providing a nontrivial theoretical analysis; not a complete reinvention of the area, but clearly beyond incremental.

## Overall Rating

6 Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The work fills a real methodological gap, has solid theory and reasonably thorough experiments, and is likely of interest to the conformal / causal survival community. However, the reliance on strong assumptions, limited analysis of weight estimation, and some clarity issues prevent a stronger recommendation.

## Reviewer Confidence

4 confident — I am comfortable with conformal prediction and causal survival literature, have gone through the main proofs and experiments in detail, and am reasonably confident in this assessment, while acknowledging that the more technical parts of Appendix B could conceal subtleties.