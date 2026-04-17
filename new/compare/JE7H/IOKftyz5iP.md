---
job_id: b4eeb7f2-6fe4-46d9-b71b-b85dc5356008
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: IOKftyz5iP.pdf
paper: Adaptive World Models for Data-Efficient Learning
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a structured latent “world model” with modular dynamics, uncertainty quantification, and theoretical generalization bounds, evaluated on synthetic AR processes and a real tabular prediction task. This squarely fits ICLR topics (representation learning, generative models, causality, uncertainty, and learning theory).

## Minimum Quality
Pass ✅.  
All standard sections (Abstract, Introduction, Related Work, Method/theory, Experiments, Results, Discussion/interpretation) are present. The work is technically nontrivial and clearly within modern ML, with both theory and experiments. While I identify significant weaknesses later, they are not of the “fatal / desk-reject” type.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden or explicit instructions intended to manipulate automated reviewing systems within the paper content.

---

# Expected Review Outcome:

## Summary

The paper proposes Adaptive World Models for Data-Efficient Learning (AWML), a framework combining modular latent “world models,” counterfactual generation via recombining learned modules, and an uncertainty-based accept–reject filter for synthetic data.  

On the theoretical side, the authors derive excess-risk bounds for: (i) structured hypothesis classes, (ii) modular “data amplification” by recombination (Theorem 3.5), and (iii) uncertainty-thresholded acceptance of synthetic samples (Theorems 3.8–3.11 and Corollary 3.13).  

Empirically, they validate parts of the theory on a synthetic AR(1) modular system and on a real low-label tabular dataset (Uganda LSMS 2019 electrification), reporting modest improvements in RMSE and substantial AUC gains under limited labels after applying AWML.

---

## Strengths

1. **Clear, unified theoretical narrative around augmentation bias and variance.**  
   Sections 3 and Appendix A give a reasonably coherent chain from standard learning-theoretic tools (Rademacher complexity, covering numbers) to modular product-TV bounds (Lemma 3.2), generator-induced risk shift (Lemma 3.3), and the final modular amplification result (Theorem 3.5). The explicit decomposition in Equation (5) and Corollary 3.11 into a variance term  
   \[
   C\sqrt{\frac{\log\mathcal N(\mathcal H,\varepsilon)+\log(1/\beta)}{N_\text{eff}}}
   \]
   plus additive bias terms (e.g., \(2D\), \(2(1-\alpha)(Q(U>u)+u)\)) is conceptually clean and makes the bias–variance trade-off legible for practitioners.

2. **Use of modular latent dynamics with explicit TV bookkeeping.**  
   The modular factorization in Equation (2) together with Lemma 3.2 is a nice and fairly tight way to propagate per-module total variation deviations \(\delta_m\) into an aggregate bound  
   \[
   D = 1 - \prod_m (1-\delta_m) \le M\delta.
   \]  
   The small-deviation scaling discussion in Appendix A.3 is helpful. This is one of the few works that explicitly treats “world model recombination” as a compositional probabilistic object and then carries TV bounds through to downstream risk.

3. **Concrete excess-risk guarantees for accepted augmentation.**  
   The uncertainty-filtering piece around Assumption 3.6 and Theorem 3.8 is interesting: it formalizes the intuition that an uncertainty score \(U(\tau)\) acting as an upper bound on a per-sample discrepancy \(d(\tau)\) lets you control deployment-level TV and hence excess risk. Theorem 3.10 and Corollary 3.11 then connect this to empirical mixtures \(P_{\text{aug}} = \alpha\widehat P_N + (1-\alpha)\widehat Q_{u,B}\). The deployment-level story is clearer than in many augmentation papers that only provide generator-level metrics.

4. **Figures that directly visualize the theoretical predictions.**  
   - **Figure 1 (top-left)** plots test RMSE vs \(\log_{10} N_\text{eff}\) for Ridge and MLP, and the fitted slopes are reported as close to \(-1/2\). This explicitly connects to the \(N_\text{eff}^{-1/2}\) term in Lemma 3.4 and Theorem 3.5; visually, the linear trend in the log–log plot roughly matches the theory.  
   - **Figure 1 (top-right)** shows empirical augmentation bias vs \(\sum_m \delta_m\), with points staying below the line corresponding to \(2D\). This is a nice sanity check that the product-TV bound is not wildly off in the synthetic regime.

5. **Uncertainty-filtering diagnostics are explicitly exposed.**  
   Section 4.3 and **Figure 2** provide a reasonably rich set of diagnostics: the acceptance curve (Panel A), a reliability diagram (Panel B), uncertainty histogram of factual vs synthetic (Panel C), and ROC curves before and after augmentation (Panel D). This is more than the usual “we used uncertainty” black box. Table 3 also reports a “TV bound” diagnostic, \(L_{95}\), and clamp fraction, which reflects awareness of numerical instability in calibration.

6. **Empirical benefit on a challenging low-label real dataset.**  
   On the Uganda LSMS electrification task, AWML shows consistent AUC gains in low-label settings, e.g. Table 3 reports an AUC improvement from 0.8797 to 0.9402 at \(n=25\) labels, with accepted synthetic count \(B=1110\). Section 4.2 asserts that these gains are robust across seeds and that bias remains below a curve \(2Q(U>u)+2u\) in calibrated regimes. Even if not exhaustively benchmarked, this demonstrates that the method can matter in a realistic, messy, non-image domain.

7. **Appendix contains nontrivial practical estimation procedures.**  
   Appendix A.7 outlines concrete (if somewhat high-level) procedures for estimating per-module TV deviations \(\delta_m\) via classifier-based density-ratio estimation, estimating the calibration slope \(L\), and adjusting for dependence in \(N_\text{eff}\). These are at least plausible recipe-level instantiations of the theory, rather than leaving all constants as unmeasurable abstractions.

---

## Weaknesses

1. **Conceptual novelty is moderate; many theoretical pieces are standard and the “framework” feels like a repackaging.**  
   The key results use: standard Rademacher bounds (Theorem 3.1), a classical product-TV argument (Lemma 3.2), and uniform convergence over synthetic data (Lemma 3.4). The modular recombination in Theorem 3.5 is mathematically straightforward given these ingredients; the only special structure is that the generator is factored as products of per-module conditionals. Similarly, Assumption 3.6 and Theorem 3.8 reduce to “uncertainty upper-bounds a discrepancy that controls TV,” which is conceptually aligned with well-known density-ratio / importance-weighting reasoning. What is missing is a sharper, clearly new insight beyond “use compositionality, measure TV, and threshold by uncertainty.” Relative to existing world-model and data-efficiency work, this looks more like a theoretically decorated design pattern than a qualitatively new algorithmic concept.

2. **Strong and somewhat opaque assumptions for the certified-acceptance theory.**  
   - **Assumption 3.6** posits a nonnegative discrepancy \(d\) such that for all \(|f|\le 1\),
     \[
     |\mathbb E_P[f] - \mathbb E_Q[f]| \le \mathbb E_Q[d],
     \]
     and that the uncertainty score satisfies \(U(\tau)\ge d(\tau)\) almost surely. This is extremely strong: it essentially demands that \(U\) upper-bound the per-sample contribution to the worst-case risk shift for *all* bounded test functions, not just the loss you care about. The paper treats this as if it were achievable by “conformal scores or ensemble variance,” but provides no rigorous link between any concrete estimator (Section A.7.2) and this assumption.  
   - Equation before Theorem 3.8 in the abstract mentions a form \(|q(\tau)-p(\tau)| \le L U(\tau)\), but in the main text this morphs into Assumption 3.6’s expectation bound. The relationship between these formulations, the constant \(L\), and the empirical “TV bound” in Table 3 is not cleanly spelled out, so it is hard to judge whether the reported diagnostics truly certify anything.  
   - The pointwise calibration constant \(L\) appears in the abstract and later in Appendix A.4 but its estimation relies on high-dimensional density-ratio estimation, which is itself brittle. Appendix A.7.2 sketches an estimator for \(L\) that bootstraps off a classifier and uses trimmed ratios, but there is no finite-sample guarantee that the resulting \(L_{\text{UB}}\) actually preserves the inequality required for Corollary 3.13. This disconnect undermines the claimed “certified” nature.

3. **The modular world-model learning procedure is under-specified and lacks empirical stress tests.**  
   The central story relies on learning a modular latent representation \(z_t = (z_t^{(1)}, \dots, z_t^{(M)})\) whose transitions approximately satisfy Equation (2). However:
   - The main text gives almost no detail on *how* this modular structure is actually induced or validated in the LSMS experiment. There is mention of “modular neural blocks and neural-operator components” around Equation (3), but in the real dataset experiment (Section 4.2) the model used is “an ensemble of twenty small MLPs” with pseudo-labels and variance-based uncertainty. It is unclear whether any genuine modular latent world model is used there at all, or whether the “world model” part simply degenerates into a standard ensemble classifier with a latent representation.  
   - There is no analysis of whether the factorized assumption (Equation (2)) is even approximately satisfied in LSMS; e.g. no empirical estimates of per-module TV deviations \(\hat\delta_m\) on real data in the main text. Table 1 reports a “generator TV bias D < 0.25” as a “typical value,” but no per-dataset or per-run numbers are shown, and no methodology is spelled out in the main body. This makes it hard to believe that modular recombination is doing anything structurally meaningful on LSMS as opposed to generic perturbations.  
   - The synthetic experiment uses truly independent AR(1) modules, i.e. a very favorable case where Equation (2) is exactly true. There is no experiment where the factorization is violated in a controlled way and the consequences for bias, as predicted by Theorem 3.5, are rigorously quantified.

4. **Experimental evaluation is limited and baselines are weak or under-specified.**  
   - **Synthetic AR(1):** The main table (Table 2) reports a *single-seed* improvement for Ridge (RMSE 0.227 → 0.219) and MLP (0.253 → 0.233). These gains are numerically small. Appendix Table 4 shows more numbers, but the main text does not convey effect sizes relative to simple baselines like training directly on more real data or using standard regularization. Figure 1’s top-left panel does show a log–log slope near \(-1/2\), but the error bars look quite large, and no statistical test is given on the slope estimate itself. For a core theoretical claim (scaling with \(N_\text{eff}^{-1/2}\)), this is thin.  
   - **LSMS dataset:** Only a single tabular dataset is used. Baselines in Section 4.2 are: logistic regression, small MLP, a self-supervised autoencoder, and a pool-based uncertainty sampler. There is no comparison against any competing *world-model-based* or causal/structured augmentation method, or even standard tabular semi-supervised or label-propagation baselines. Given that the domain is static household data, it is unclear why a world model is intrinsically needed; a strong tabular model (e.g., gradient boosting, random forest with careful calibration) might achieve similar or better AUC with simpler calibration-based augmentation or even classical resampling.  
   - The self-supervised and active learning baselines are described very briefly (“same features and label splits,” “uncertainty sampling”), with no architectural or hyperparameter details in the main text, making it difficult to assess whether they are reasonably strong.  
   - Table 3 shows that for \(n=100\), the TV bound diagnostic jumps as high as 0.24556, while AUC still improves. If the theory is taken seriously, such a large diagnostic bias should call into question the reliability of the augmented model. The text acknowledges that diagnostics “highlight runs where the assumptions behind Theorem 3.8 may fail,” but then still reports the associated performance improvements without a principled decision rule for rejecting such runs.

5. **Lack of clarity about data usage and calibration procedure relative to theory.**  
   The theoretical developments repeatedly require i.i.d. samples and clean separation between training, calibration, and validation sets. In practice, however, Section 4.2 and Appendix B.1 suggest a somewhat ad hoc use of a validation split for both threshold selection and calibrator fitting. It is not clear:
   - whether the same labels are used both to determine the threshold \(u\) and to fit the final classifier, which could create subtle selection bias relative to Corollary 3.11’s use of independent \(N\) factual samples and \(B\) synthetic samples;  
   - how much unlabeled data is available and whether it is being used consistently across AWML and baselines;  
   - how “isotonic calibration,” “Platt,” and “temperature scaling” in Table 11–13 relate mathematically to Assumption 3.6. For instance, Figure 2B’s reliability diagram is extremely coarse: a few bins, with some bins showing severe over- or under-confidence, yet the text claims “calibration diagnostics are stable.” This seems optimistic.

6. **Some mathematical claims are loose or potentially misleading.**  
   - Theorem 3.12 states a classical submodular greedy bound but writes the approximation factor as \(1 - 1/\varepsilon\) instead of \(1 - 1/e\); this is likely a typo but appears in the main text and risks confusion.  
   - Corollary 3.13 imports a transfer bound (Theorem A.4) with constants \(C_1, C_2\) and a term \(\mathcal E_{\text{target}}\) but does not clearly define \(\mathcal E_{\text{target}}\) in the main text or reconcile what “error on the target environment” means for the non-RL LSMS task. This corollary starts to look like a generic laundry-list bound rather than something tightly tied to the experimental setting.  
   - The abstract claims a pointwise calibration bound \(|q(\tau) - p(\tau)| \le L U(\tau)\) that “yields the deployment-level control \(\mathrm{TV}(P_{\text{aug}}, P) \le \frac{B}{N+B} L u + \varepsilon.\)” However, this exact inequality does not appear in the main body; the closest is Theorem 3.10 / Corollary 3.11 with a term \(2(1-\alpha)(Q(U>u)+u)\). The mismatch between these formulations is never resolved.

7. **Positioning with respect to current world-model literature is incomplete.**  
   The related work section mentions classical world models, neural operators, and causal representation learning, but omits several directly relevant and recent works on adaptive, sample-efficient world models and world-model-based RL:
   - Recent adaptive world-model frameworks (e.g., adaptive latent world models under non-stationarity, time-aware world models with explicit temporal adaptation) are not discussed.  
   - Sample-efficient world-model papers using transformers or non-curated data are missing from the comparison.  
   - Empirical evaluation does not include any of these methods as baselines or even conceptual comparisons.  
   As a result, it is hard to see where AWML stands in the crowded space of “adaptive world models” and “data-efficient world modeling.”

8. **Figures sometimes tell a more nuanced story than the text admits.**  
   - In **Figure 1 (bottom-left)**, the ablation on module count \(M\) shows gains but also suggests diminishing returns and potentially aggressively rising variance for high \(M\); the shaded confidence intervals widen notably. The main text glosses over these variance issues, focusing only on mean improvements.  
   - **Figure 2D** shows an ROC curve for a representative LSMS run where baseline AUC is already very high (0.954), and the final model reaches 0.997. This near-ceiling performance suggests that minor perturbations in calibration or thresholding could heavily influence reported gains, yet no sensitivity analysis is presented beyond limited robustness in Appendix B.1.

9. **Scope-creep: AWML is pitched as a very general framework, but experiments cover only narrow regimes.**  
   The method is framed as a general approach for “low-resource languages, clinical cohorts, sparse Earth observations,” etc., but the only non-synthetic evidence is a single tabular classification dataset. There is no demonstration on any temporal or control task where world models are truly needed (e.g., RL, time-series forecasting, PDE systems). Given the strong “world model” branding and the emphasis on modular latent dynamics and neural operators, this gap between ambition and evidence is substantial.

---

## Potentially Missing Related Work

The following directly related works are not cited in the paper and should be discussed:

1. **Gao, S., Zhou, S., Du, Y. (2025). “AdaWorld: Learning Adaptable World Models with Latent Actions.”**  
   - Relevance: Proposes adaptable world models with latent actions explicitly aimed at data-efficient adaptation across heterogeneous environments, which aligns closely with AWML’s focus on “adaptive transfer across environments” (Section 1, 2).  
   - Where to add: Discuss in Section 1.1 “Latent dynamics and world models” and in the positioning paragraph; clarify how AWML’s modular counterfactual recombination differs from or complements AdaWorld’s latent-action adaptation.

2. **Gospodinov, E., Shaj, V., Becker, P. (2024). “Adaptive World Models: Learning Behaviors by Latent Imagination Under Non-Stationarity.”**  
   - Relevance: Deals with adaptive world models for non-stationary environments using latent imagination, directly overlapping with AWML’s stated goal of robust transfer across a family of environments \(\mathcal E\).  
   - Where to add: Section 1.1 “Latent dynamics and world models” and possibly Section 3.13’s transfer discussion as a conceptual comparator.

3. **Dedieu, A., Ortiz, J., Lou, X. (2025). “Improving Transformer World Models for Data-Efficient RL.”**  
   - Relevance: Focuses on data-efficient world-model learning with transformers, similar in spirit to AWML’s data-efficiency goals but with a different architectural bias.  
   - Where to add: Compare in Section 1.1 and discuss in Section 4 how AWML’s theoretical guarantees might (or might not) apply to transformer-based world models.

4. **Gumbsch, C., Sajid, N., Martius, G. (2024). “Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics.”**  
   - Relevance: Introduces hierarchical world models with modular temporal abstractions, conceptually close to AWML’s modular latent blocks and recombination.  
   - Where to add: Section 1.1 and the “Modularity and recombination” subsection; clarify whether AWML’s modules are compatible with hierarchical temporal abstraction.

5. **Nhu, A. N., Son, S., Lin, M. (2025). “Time-Aware World Model for Adaptive Prediction and Control.”**  
   - Relevance: Addresses adaptive world modeling with explicit temporal structure for prediction and control, relevant to AWML’s claim of leveraging operator-structured transitions (Equation (3)).  
   - Where to add: Section 1.1 and possibly the structured transition parameterization subsection.

6. **Ying, L., Collins, K. M., Sharma, P. (2025). “Assessing Adaptive World Models in Machines with Novel Games.”**  
   - Relevance: Evaluates adaptive world models in novel task regimes, which could provide benchmark perspectives on “adaptive transfer across environments.”  
   - Where to add: Related work and the concluding discussion, as a reference point for empirical evaluation norms.

7. **Micheli, V., Alonso, E., Fleuret, F. (2022). “Transformers are Sample-Efficient World Models.”**  
   - Relevance: A prominent work on sample-efficient world models using discrete autoencoders and transformers; directly competes with AWML on the data-efficiency axis.  
   - Where to add: Section 1.1 and the “Positioning” subsection; at minimum, compare conceptual strengths (structured priors vs autoregressive modeling) and discuss why AWML is not benchmarked against such a method.

8. **Zhao, Y., Scannell, A. (2026). “Efficient Reinforcement Learning by Guiding World Models with Non-Curated Data.”**  
   - Relevance: Uses non-curated data with world models for sample-efficient RL, similar in spirit to AWML’s augmentation-with-uncertainty idea.  
   - Where to add: Section 1.1 and the data-efficient learning subsection; contrast AWML’s certified filtering with their way of handling non-curated data.

9. **Song, Y., Jin, J., Zhuang, T. (2025). “Distill Models by Aptitude: Efficient Reasoning Capability Distillation via Adaptive Data Curation and Overthinking Mitigation.”**  
   - Relevance: Introduces adaptive data curation and overthinking mitigation for efficient distillation, conceptually overlapping with AWML’s idea of selectively accepting synthetic data based on uncertainty.  
   - Where to add: Data-efficient learning and uncertainty / calibration-related text; relate AWML’s accept–reject rule to adaptive data selection in distillation.

10. **Nhu, A. N., Son, S., Lin, M. (2024). “Time-aware World Model: Adaptive Learning of Task Dynamics.”**  
    - Relevance: An earlier version of time-aware world models for adaptive learning of task dynamics, again tightly aligned with AWML’s adaptive transfer claims.  
    - Where to add: Merge discussion with the 2025 follow-up in the related work section and explicitly state how AWML differs in theoretical or algorithmic aspects.

---

## Questions

1. **Concrete learning procedure for modular latents on LSMS.**  
   - How exactly are the latent modules \(z^{(m)}\) defined and trained on the LSMS dataset? Are you actually using a latent world model there, or is AWML instantiated purely with an ensemble classifier and feature-space recombination? Please provide an explicit architecture diagram or equation-level description for the LSMS “world model” and clarify which parts correspond to modules \(m\).

2. **Quantitative estimates of \(\delta_m\) and \(D\) on real data.**  
   - Table 1 states that \(D < 0.25\) “typically,” but the main text never shows the distribution of per-module \(\widehat\delta_m\) or aggregate \(D\) on LSMS. Can you report, for LSMS, the estimated \(\widehat\delta_m\) and \(D\) per seed, along with confidence intervals, and overlay these on the empirical bias curve in Figure 1 (top-right) or a new figure? This would substantiate the claim that modular recombination is safe in practice.

3. **Validating Assumption 3.6 and the calibration constant \(L\).**  
   - The validity of Theorem 3.8 hinges on the existence of a discrepancy \(d\) and a constant \(L\) (in the abstract) such that \(U(\tau)\) upper-bounds the relevant density difference or expectation discrepancy. Can you provide any empirical evidence (e.g., scatter plots of \(|\widehat q(\tau)-\widehat p(\tau)|\) vs \(U(\tau)\)) that your estimated \(L\) in Appendix A.7.2 actually satisfies  
     \[
     |q(\tau)-p(\tau)| \le L U(\tau)
     \]  
     or the weaker Assumption 3.6 variant with high probability? Right now, this connection is purely asserted.

4. **Clarifying data splits and prevention of selection bias.**  
   - In Section 4.2 and Appendix B.1, is the same labeled validation set used to (i) fit the calibration mapping (isotonic/Platt/temp), (ii) choose the threshold \(u\) by grid search over validation AUC, and (iii) select the final model? If so, does this violate the independence assumptions underpinning Theorem 3.10 / Corollary 3.11? Please explain the exact splitting protocol and whether any nested cross-validation or holdout was used.

5. **Comparison to strong tabular baselines.**  
   - Have you tried gradient-boosted trees, random forests with proper calibration, or modern tabular neural architectures (e.g., FT-Transformer) on LSMS, both with and without simple augmentation schemes (e.g., SMOTE, mixup on numeric features)? How do your AUCs and diagnostic TV bounds compare to these baselines? This would help distinguish AWML’s benefits from those of straightforward tabular methods.

6. **Instantiating AWML on a genuine sequential / control task.**  
   - Given the world-model framing, do you have any preliminary or supplementary results on RL or time-series prediction tasks (even small-scale) where the dynamics are nontrivial and modular recombination generates entire trajectories rather than pseudo-tabular examples? Even one such experiment could materially strengthen the case that AWML is more than a tabular data curation heuristic.

7. **Clarifying Theorem 3.12 and Corollary 3.13.**  
   - Please check the approximation factor in Theorem 3.12 and correct it if it was intended to be the classical \(1 - 1/e\).  
   - For Corollary 3.13, please define \(\mathcal E_{\text{target}}\) explicitly in the main text and explain which experiments (if any) are meant to instantiate this unified bound.

Providing clear answers or additional experiments addressing these points could substantially change my view, especially if you can robustly connect the estimated diagnostics (TV bounds, \(L\), \(\delta_m\)) to the theoretical assumptions and broaden the empirical evidence beyond LSMS.

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A. The work uses a public household survey dataset with no apparent new data collection or sensitive subgroup analysis. There is no direct optimization over individuals or deployment into high-stakes real-world systems discussed.

---

## Soundness Rating

2: fair.  
The math at the level of lemmas and covering-number bounds is generally standard and appears correct, but key assumptions (factorized modular dynamics, Assumption 3.6, existence/estimation of \(L\)) are very strong and not convincingly tied to the actual instantiated models. Experimental evidence is limited and not sufficient to fully validate the “certified” claims.

---

## Presentation Rating

3: good.  
The paper is readable and well organized, with equations clearly numbered and figures/tables (e.g., Figure 1, Figure 2, Tables 2–3) integrated into the narrative. However, some definitions are under-specified (e.g., exact world model used for LSMS, explicit definition of \(\mathcal E_{\text{target}}\)), and there are minor inconsistencies and typos in theorems and abstracted bounds.

---

## Contribution Rating

2: fair.  
The contribution primarily lies in bundling known tools (modular TV bookkeeping, uniform convergence, uncertainty-based filtering) into a single framework and applying it to a modest synthetic setup and a single real dataset. While the unifying perspective and emphasis on diagnostics are useful, the lack of stronger empirical validation and incomplete positioning within the modern world-model literature limit the impact.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper provides a coherent and mostly sound theoretical treatment of modular world-model-based augmentation with uncertainty filtering, and the synthetic plus LSMS experiments offer some evidence that the ideas can be beneficial. However, key assumptions behind the “certified” guarantees are very strong and not empirically validated, the modular world-model component is under-specified on real data, and the empirical evaluation is narrow with relatively weak baselines and no genuine sequential/control tasks. With clearer empirical links between diagnostics and theory, stronger baselines, and better positioning versus recent adaptive world-model work, this could become a solid contribution; in its current form, it falls slightly short of ICLR’s standards.

---

## Reviewer Confidence

4: confident.  
I am reasonably familiar with learning theory, uncertainty calibration, and world-model literature, and I carefully checked the main math and experimental setup, though I did not fully re-derive all appendix proofs. My main uncertainties concern unobserved implementation details and potential additional experiments that might exist but are not in the main paper.