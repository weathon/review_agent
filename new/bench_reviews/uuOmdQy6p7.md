Now I have read the full paper and calibrated against relevant anchors. Let me construct the final review.

---

## Summary
This paper introduces CEMA, a black-box adversarial attack framework for multi-model multi-task learning (MTL) systems. It converts the multi-task attack into a single-task binary classification problem by clustering concatenated input-output embeddings, trains substitute models on the resulting pseudo-labels, and selects adversarial examples based on transferability across an ensemble of bootstrapped substitutes. The method is evaluated on text classification and translation tasks, including attacks against commercial translation APIs (Baidu, Ali Translate), reporting high attack success rates (60–80% ASR) with very low per-instance query costs (0.045–0.05).

## Strengths
- **Novel and realistic threat model**: The problem formulation targets black-box, multi-model MTL systems without shared parameters—a setting largely ignored by prior white-box multi-task attacks. Evaluating against commercial APIs (Baidu, Ali Translate) demonstrates practical relevance beyond open-weight models (Section 3, Table 2).

- **Plug-and-play cluster-substitute design**: Converting multi-task objectives into a single binary classification task via clustering (Section 4.2, Algorithm 1) is a clean engineering simplification. The ablation that increasing from 1 to 3 attack methods raises average ASR by 30.39% (Table 3) confirms the ensemble benefit.

- **Zero-shot cross-distribution robustness**: Using Emotion auxiliary data to attack SST5 (and vice versa) still yields 64–66% ASR despite distribution mismatch (Table 6), suggesting the clustering approach captures generalizable structure rather than memorizing training distributions.

## Weaknesses

### Fatal
None.

### Major

- **Fundamentally unfair query accounting invalidates the primary efficiency claim**: The paper reports CEMA's query cost as 0.045–0.05 per text by dividing 100 upfront auxiliary queries by the total number of test instances (2,000–2,210), while baselines are restricted to 30 queries *per instance* (Section 5.1). This comparison is structurally asymmetric: CEMA spends its entire 100-query budget building a substitute model that can then attack the full test set at essentially zero marginal cost, while each baseline is capped at 30 queries per test instance — meaning baselines effectively receive up to 66,000+ total queries across the test set. This makes the "few-shot" headline claim misleading. A fair comparison would either (a) cap CEMA at the same *per-instance* budget, (b) give all methods the same *total* budget (e.g., 100 queries distributed across test instances), or (c) acknowledge that CEMA operates under a "model-once, attack-many" paradigm distinct from per-instance optimization and present appropriate baselines for that regime.

- **Mechanism–objective mismatch for translation tasks with no bridging analysis**: The substitute models are binary classifiers trained on cluster labels derived from concatenated input-output embeddings (Eq. 1, Section 4.2). Adversarial perturbations are then optimized to flip these binary labels (using classification attack methods like BAE, TextBugger), with the expectation that this degrades BLEU scores for autoregressive translation models. For classification tasks, the cluster-label flip may correlate with the victim's classification output. For translation, however, there is no analysis — theoretical or empirical — explaining why crossing an artificial binary decision boundary in a static embedding space systematically disrupts sequence-to-sequence generation. The paper presents BLEU reductions as successful attacks (Table 1) without establishing a mechanistic link between the optimization target (cluster label flip) and the evaluation metric (BLEU degradation). This is a gap in understanding: the results are empirically real but the method's mechanism for translation remains unexplained.

### Minor

- **Ensemble transferability selection operates on highly correlated substitutes**: All *w* substitute models are trained on 80% bootstrap splits of the same 100 auxiliary data points with identical binary cluster labels. The transferability criterion (Eq. 6) selects candidates that flip the most substitutes, but these substitutes approximate essentially the same arbitrary cluster boundary. The paper provides no diversity/correlation metrics, no out-of-distribution validation, and no analysis whether ensemble agreement actually predicts victim-model transfer. Maximizing agreement across correlated models identifies inputs near the synthetic boundary, but claims this generalizes to the victim's true task boundaries without evidence.

### Trivial
None.

## Nice-to-Haves
- Visualizing the geometric relationship between the artificial 2-cluster boundary and true victim decision boundaries would strengthen interpretability.
- Reporting variance/standard deviation across bootstrap substitute iterations would establish result stability.
- Evaluating how attack performance scales with smaller auxiliary budgets (20–50 samples) would test the true few-shot limits.

## Removed Points
These points are flagged to be removed, treat them with caution:
- *"The probability derivation (Eqs. 2–5) is elementary and adds no novel theoretical insight."* — This is a presentation critique, not a methodological error. The derivation is indeed basic probability, but it correctly supports the ensemble argument. Downgraded from a weakness to a note that the theoretical contribution is limited.
- *Reviewer concerns about "missing related works" on multi-task attacks.* — Per the rules, do not flag missing references.
- *"The claim that CEMA is 'the first to extend text adversarial attacks to the multi-task setting' is overstated."* — The paper defines its scope as black-box multi-model MTL; this claim, while broad, is within the paper's framing.
- *"Suboptimal figure choices / notation inconsistencies."* — Formatting and visual presentation issues are parser artifacts or minor style notes, not substantive weaknesses.
- *"No variance or standard deviation reported."* — Moved to Nice-to-Have; single-run reporting is common in adversarial attack literature.
- *"Missing appendix proofs / absent references."* — Per rules, parser strips these sections; do not flag.

## Novel Insights
The core insight — that clustering concatenated input-output embeddings creates a discriminative binary boundary whose adversarial disruption transfers across multiple tasks — is genuinely interesting. If verified, this suggests that multi-task model vulnerability can be reduced to a simpler structural property of the joint representation space, rather than requiring per-task surrogate models. The observation that zero-shot auxiliary data (from a different but related distribution) still yields ~65% ASR hints that the cluster boundary captures something domain-general. However, the methodological gaps (query accounting, mechanism analysis, ensemble correlation) mean this insight remains suggestive rather than demonstrated. A more carefully controlled evaluation protocol, with a fair query budget and mechanistic analysis of the cluster-to-translation link, could make this a strong contribution.

## Suggestions
1. **Restructure the evaluation to use per-instance or equal-total query budgets.** Report CEMA's performance when limited to a per-instance budget matching the baselines, or report baseline performance when limited to a total of 100 queries distributed across the test set. This would establish whether CEMA's advantage survives a fair comparison.
2. **Add mechanistic analysis for translation tasks.** Provide a case study or analysis (e.g., token-level perturbation analysis, attention map comparison, or semantic drift metrics) showing *how* perturbations that flip cluster labels affect translation output.
3. **Quantify substitute ensemble diversity.** Report prediction agreement (e.g., pairwise correlation or disagreement rate) across the *w* bootstrapped substitutes, and correlate ensemble success rate with actual victim-model attack success to validate the transferability criterion.
4. **Clarify the query cost framing.** Distinguish between upfront model-building queries and per-instance attack queries, and position CEMA explicitly within a "surrogate-once" paradigm rather than claiming per-instance few-shot parity with traditional methods.

## Calibration and Scoring
I compared against:
- **High-scoring anchor (8,8,6,8)**: tIBAOcAvn4 — strong theory + thorough experiments with fair baseline comparison on hard-label attacks. This paper exceeds CEMA in methodological rigor.
- **Mid-scoring anchor (6,6,6,6)**: asR9FVd4eL — accepted poster with some missing baselines and demonstrative results but solid core methodology and clear claims. CEMA falls below this because its evaluation protocol has a structural asymmetry (unlike asR9FVd4eL, where the methodology itself was sound).
- **Low-scoring anchors**: 4NtrMSkvOy (3,3,3,3) — transfer-based attack with insufficient experiments and unvalidated intuition; ByAhXwV4bH (3,8,3,3,3) — rejected for serious methodology flaw in experimental design. CEMA is somewhat better than these (results are empirically real, problem formulation is novel), but shares the pattern of evaluation methodology undermining confidence in claims.

CEMA sits below the 5–6 borderline because its primary claimed contribution (query-efficient few-shot attack) is supported by an evaluation protocol that structurally favors CEMA over baselines. The mechanism-objective gap for translation tasks and the correlated-ensemble issue further weaken the paper's evidential grounding. It is above the 3-reject range because the core idea (cluster-substitute for multi-task attack) is novel and the empirical results on commercial APIs are real.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>