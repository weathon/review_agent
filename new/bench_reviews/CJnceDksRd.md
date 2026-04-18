Now let me look at the referenced human review files for calibration.Now I have enough context to write the final consolidated review. Let me analyze everything carefully.---

## Summary

DRL (Decomposed Representation Learning) proposes a one-class tabular anomaly detection framework that maps normal sample representations into a latent space constrained to be decomposable as weighted linear combinations of fixed, randomly generated orthogonal basis vectors. A "separation constraint" further pushes normal sample weight vectors apart to amplify the discrepancy between normal and anomalous patterns at inference time. The method is evaluated on 40 datasets against 16 baselines, with Wilcoxon signed-rank significance testing.

---

## Strengths

- **Unusually broad and rigorous empirical evaluation.** 40 tabular datasets, 16 competing methods (classical and deep), two metrics (AUC-PR, AUC-ROC), Wilcoxon signed-rank tests, 10 independent runs per experiment, and a 9.8%+ absolute AUC-PR gain that is statistically significant at α=0.05 across every baseline. This is substantially above the standard for this subfield.
- **Comprehensive ablation study.** Table 2 presents 10 ablation variants covering all training loss combinations, different weight learner architectures (least squares vs. learned), using MCM's representations and architecture as swapouts (Variants G, H), and two different inference-time scoring strategies (Variants I, J). Table 1 probes five orthogonalization strategies plus a learnable-basis variant. Table 3 varies three distance metrics for each loss. Very few papers probe their design space this thoroughly.
- **Clear motivation with concrete visualization.** The data entanglement issue is convincingly demonstrated via t-SNE across multiple datasets (Fig. 1, 6, Appendix Fig. 7), and the separation constraint's effect is quantified by distance-ratio boxplots (Fig. 5), showing that DRL's latent space achieves a substantially higher normal-to-anomalous distance ratio than baselines.
- **Simple and reproducible method.** The architecture is two-layer MLPs throughout, with publicly released code. The fixed random basis vectors require no special initialization procedure.
- **Practical advantages over strong baselines.** Unlike NPT-AD, DRL does not require the entire training set at inference, making it scalable. Computational details are provided in Appendix A.5.

---

## Weaknesses

### Fatal
None.

### Major

- **Theoretical gap between propositions and the implemented optimization objective.** Propositions 1 and 2 (Section 3.2.2) state that the expected discrepancy between normal and anomalous weights is amplified by *increasing Var(‖w_n‖₂)* among normal weights. The paper then asserts "By sufficiently separating the normal weights w_n, we can promote increased variance in their L2 norms," but this logical step — that maximizing pairwise L2/cosine distances between simplex-constrained weight vectors increases the *variance of their norms* — is neither proved nor empirically verified. On the probability simplex, pairwise distance maximization tends to push weights toward corners (near one-hot vectors), but this does not formally guarantee norm variance increases as claimed. The paper offers no proof, no empirical measurement of Var(‖w_n‖₂) before and after training, and no intermediate proposition linking Eq. (4) to the Var(‖w_n‖₂) quantity in the propositions. Since the separation constraint is described as a core contribution and Table 2 shows it contributes significantly (AUC-PR 0.708→0.734, AUC-ROC 0.812→0.857), this formal disconnect between the stated theoretical justification and the actual optimization is a genuine weakness. This does not mean the mechanism is wrong, but the theoretical narrative as presented is incomplete.

- **Tension between "shared patterns" framing and the separation objective.** The paper motivates decomposition by the assumption that normal samples "share common statistical information" (Section 3.2.1). Yet the separation constraint (Eq. 4) pushes every pair of normal weight vectors maximally apart. If K=5 (fixed for all 40 datasets), the simplex has only five corners; with large N, many samples must share similar weight vectors by a pigeonhole argument, and the separation pressure may simply cluster samples to a small number of corners rather than achieving continuous diversity. The paper does not analyze the distribution of w values (e.g., entropy, clustering structure), nor discuss how these two objectives (shared basis + maximal separation) are simultaneously satisfiable. Appendix A.7 mentions weight visualization, but the main text leaves this tension unaddressed.

### Minor

- **Fixed K=5 across all 40 datasets with limited in-text justification.** The dimensionality of the basis subspace is a central architectural choice: too small a K collapses the representation, too large and the constraint is trivially satisfied. The paper uses K=5 universally across datasets that vary from low-dimensional (few features) to high-dimensional (hundreds of features). Sensitivity analysis is deferred to Appendix A.6 without a principled guideline in the main text for how K should scale with data complexity. This matters for practitioners and for understanding why the method is robust.

- **Missing sensitivity to basis vector random seed.** The paper averages over 10 independent runs to reduce randomness, but these runs apparently share the same random basis B (the paper says "which are fixed during the whole training procedure" and does not indicate seed-level variability for B). Table 1 shows that the *orthogonalization method* barely matters, but it does not test whether different random seeds for B produce meaningfully different results. This is a reproducibility question that is distinct from the orthogonalization ablation — two different random seeds with the same Gram-Schmidt process could yield very different bases. This is especially important since the central narrative claims the bases capture underlying data structure.

- **Counterintuitive result for Variant I not fully explained.** Variant I (decomposition + alignment loss at inference) achieves AUC-ROC 0.8286, while DRL using only decomposition loss achieves 0.8574 — a meaningful drop from combining two signals. The paper's explanation ("decomposition loss is more effective") is consistent with the observation-space entanglement narrative but does not explain why adding an informative alignment score *hurts* rather than simply being neutral. A brief mechanistic analysis would help practitioners understand when to use each score.

- **Narrative overclaims the role of fixed random bases.** The abstract and Introduction repeatedly state that fixed random orthogonal bases "capture the underlying shared normal patterns." However, since f and φ are jointly trained around fixed B, the system learns to make h decomposable onto whatever B happens to be — the actual "patterns" reside in the learned network weights, not in B. Table 1 confirms that all orthogonalization methods yield nearly identical results, which is strong evidence that the specific geometry of B is largely irrelevant. The paper would be more accurately framed as "using a fixed random orthogonal subspace as a regularization constraint on the learned representation," which is an honest and still-interesting contribution, but a weaker claim than "the basis vectors encapsulate the global structure of normal data."

- **Contamination robustness analysis is in the appendix.** In practice the one-class assumption is often violated. Appendix A.8 addresses this, but given that it is a primary real-world concern for any one-class method, moving at least a summary to the main paper would strengthen the practical claims.

### Trivial

- **Eq. (6) uses nested min operators in a non-standard way.** Writing `min_{θ_f,θ_φ,θ_g} L_all = min_{θ_f,θ_φ} L_decomp + λ₁ min_{θ_φ} L_sep + ...` with different parameter sets in the inner mins is confusing notation. In practice the joint gradient is computed via backpropagation; the notation should reflect this or be explained.

---

## Nice-to-Haves

- **Analysis of w distribution.** Visualizing the actual distribution of weight vectors w for normal vs. anomalous test samples (e.g., entropy, clustering structure, how close they are to simplex corners) would directly test the theoretical claims about Var(‖w_n‖₂) and confirm that the separation mechanism works as intended.
- **Effect of batch size on the separation constraint.** The mini-batch implementation of Eq. (4) approximates a population-level objective. It would be useful to report how performance varies with batch size to confirm the per-batch approximation is stable.
- **Per-dataset breakdown for failure modes.** The boxplots in Fig. 3 show considerable variance. Identifying a few representative datasets where DRL underperforms and analyzing what data properties (dimensionality, anomaly rate, feature heterogeneity) correlate with weaker results would improve practical guidance.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Claim that KNN/LUNAR/GMM are absent from baselines (Human Finder Weakness 5):** Factually incorrect. The paper explicitly includes KNN (Ramaswamy et al., 2000), OCSVM, LOF, IForest, PCA, and ECOD as classical baselines (Section 5.1, line 149). Removed as a misread.
- **Harsh critic "objective degeneracy" (collapse to simplex corners) as a *fatal* issue:** The decomposition loss and alignment loss act as countervailing forces; the method is empirically validated across 40 datasets with consistent gains. The concern is a plausible failure mode worth mentioning (captured in the Major weakness on tension), but framing it as an undisclosed structural flaw overstates it. The harsh framing is removed; the underlying concern is incorporated into Major weaknesses.
- **Eq. (6) nested-min as a "fundamental" methodological gap:** This is a presentation/notation issue, not a methodological flaw. Moved to Trivial.
- **Criticism that MCM and NPT-AD baselines are under-tuned (Harsh critic Issue 3):** The paper follows the same evaluation protocol as MCM and NPT-AD (Yin et al., 2024; Thimonier et al., 2024), uses the same architectures and settings as those papers report, and the asymmetry disfavors DRL's claimed novel components rather than its total capacity. This does not meet the threshold for a meaningful fairness concern and is removed per the rule on unfair-comparison criticisms that disfavor the authors.
- **Missing related work on contrastive/displacement-based methods (Human Finder, Spark):** Cannot confirm specific references without external access; removed per the no-missing-related-work rule.
- **Reproducibility concern about hyperparameter grid search per dataset:** The paper states consistent hyperparameters across all datasets ("The DRL architecture remains consistent across all datasets"), making this a non-issue. Removed.
- **Dataset selection criteria "not justified" (Human Finder Weakness 9):** The paper explicitly states "following previous works (Yin et al., 2024; Thimonier et al., 2024)" as the selection rationale, which is an accepted justification. Removed.

---

## Novel Insights

The most genuinely insightful observation across all reviewers — largely underemphasized in the paper itself — is the conceptual tension between the "shared patterns" narrative and the separation objective. If normal samples truly share statistical patterns, one might expect their weight vectors to cluster in a few regions of the simplex. But the separation loss explicitly prevents clustering, instead spreading weights across the simplex. This creates a hidden duality: the basis vectors act as shared "normal modes," while the weight vectors serve as individual *signatures* that distinguish each normal sample from every other. The separation constraint, viewed through this lens, is actually amplifying *individual* normal variation rather than capturing *group* normal structure — and it is precisely this diversity among normal samples that renders anomalous samples illegible to the decomposition. This reframing, if explicitly articulated, would make the theoretical story considerably more coherent and arguably more interesting than the current topic-modeling analogy, which implies normal samples should cluster around a few dominant "topics."

---

## Suggestions

1. **Close the theoretical gap in Section 3.2.2.** Either prove (or empirically demonstrate on a subset of datasets) that the separation loss in Eq. (4) under simplex constraints increases Var(‖w_n‖₂), or revise Propositions 1–2 to directly invoke pairwise distance between weights as the amplification mechanism, bypassing the norm-variance intermediate step.

2. **Report Var(‖w_n‖₂) and entropy of w before and after training.** A simple experiment measuring whether the separation constraint achieves its stated intermediate goal (increased norm variance) would directly validate the theoretical narrative.

3. **Add a multi-seed experiment for B.** Fixing B is a design choice; confirming via a 5–10 seed experiment that performance variation across random B seeds is small would close the reproducibility question efficiently.

4. **Reframe the narrative on random bases.** Describe them explicitly as a fixed regularization subspace that constrains the representation rather than as bases that "encapsulate the global structure of normal data." This is more accurate and equally compelling.

5. **Move contamination robustness results (Appendix A.8) to the main paper**, at minimum as a brief summary paragraph, since one-class AD contamination is a standard concern reviewers and practitioners will immediately raise.

6. **Explain the Variant I (combined-score) result more carefully.** A brief mechanistic explanation — e.g., that the alignment loss introduces observation-space noise that degrades the latent-space signal — would preempt reader confusion.

---

## Score and Decision

**Calibration:**

- **MCM** (tabular AD, masked cell modeling, accepted poster): scores 6, 8, 6 (avg ~6.7). Strong empirical results, similar benchmark size, accepted. DRL has a larger benchmark and more ablations but somewhat weaker theoretical grounding.
- **PTAD** (tabular AD with orthogonal basis + prototypes, withdrawn/reject): scores 3, 5, 3, 6 (avg ~4.25). Weaker empirical setup (20 datasets, 3 runs, no statistical tests, reproducibility issues). DRL is substantially stronger than PTAD on all these dimensions.
- **AnoLLM** (tabular AD with LLMs, accepted poster): scores 5, 8, 8, 6 (avg ~6.75). Strong results on mixed-type data, but narrower benchmark than DRL.
- **NCSN for tabular AD** (score-based method, withdrawn): scores 6, 6, 5, 6 (avg ~5.75). Extensive experiments (57 datasets), similar theoretical novelty level, but withdrawn due to limited novelty concerns.

**Assessment:** DRL is clearly stronger than PTAD (the most similar work) on every experimental dimension. It is comparable to MCM and AnoLLM in experimental rigor, with a broader benchmark. The two Major weaknesses (theory-practice gap and shared-patterns tension) are genuine but not fatal — they concern the framing and formal justification of a mechanism that demonstrably works. The method is simple, reproducible, and delivers statistically significant improvements across a large benchmark. This profile places DRL near the MCM score range (6–7), with the theoretical gaps pulling it toward the lower end.

**Axes summary:**
- *Originality*: Moderate — decomposition + simplex weights + random fixed bases is a novel combination for tabular AD, though each component individually draws on established ideas.
- *Importance of research question*: High — tabular anomaly detection is practically important and under-served by deep methods.
- *Support for claims*: Moderate — empirical claims are well-supported; theoretical claims have a meaningful gap.
- *Soundness of experiments*: High — unusually thorough.
- *Clarity of writing*: Good overall, with the Eq. 6 notation and the shared-patterns/separation tension as clarity gaps.
- *Value to research community*: Good — broad benchmark + code release + comprehensive ablations make this a useful reference paper.

**Score: 6.0** — Marginally above the acceptance threshold. The experimental contribution is solid and well above average for the field; the theoretical claims need tightening but do not invalidate the method. Comparable to MCM's lower-bound reviewer scores. The paper is publishable but would benefit from the revisions described above.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>