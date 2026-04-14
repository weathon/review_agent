## Summary
This paper proposes the Binary Alignment Network (BiAN) for unsupervised domain-adaptive object counting. The core insight is that standard DA methods incorrectly treat object density — a task-relevant quantity — as a domain-invariant nuisance to be aligned away, thereby destroying the very information needed for counting. BiAN addresses this by partitioning images into foreground/background regions (conditioned on pseudo-density-map predictions) and aligning each partition independently, while a Condition-consistent Mechanism (CM) encourages consistency between partial and full density predictions. The method is accompanied by a theoretical analysis adapting the Zhao et al. (2019) joint-error lower bound to the conditional alignment setting, and is evaluated across eight counting dataset combinations spanning crowd and cell counting.

---

## Strengths

- **Sharply identified and non-trivial problem formulation.** The observation that conventional DA violates its own task-irrelevance assumption in counting (density is both domain-variant and task-relevant) is precise, well-motivated, and the contrast with CODA's conflicted treatment of density is convincingly argued in the introduction and Figure 1.

- **Empirical breadth across modalities and domain types.** Evaluating on eight combinations spanning crowd counting (JHU-Crowd++, ShanghaiTech) and cell counting (VGG→ADI, VGG→DCC) with diverse shift types (weather, scene type, cell morphology) provides genuine coverage, not cherry-picking.

- **Ablation isolating component contributions.** Table 4 cleanly shows the incremental value of (a) conditional vs. unconditional alignment and (b) the CM module, especially in the high-shift VGG→ADI and SHB→SHA conditions, making the source of gains interpretable.

- **Condition-consistent Mechanism as a self-supervised stabilizer.** The CM loss (Eqs. 3–4) is a concrete and novel mechanism for enforcing partition consistency, addressing the pseudo-label circular-dependency problem in a principled way rather than ignoring it.

---

## Weaknesses

### Fatal
*None confirmed as definitely fatal, but one result approaches this threshold — see Major #1.*

### Major

- **The SHB→SHA result (MAE=42.3) is extraordinary and unexplained.** BiAN — a DA method using *no SHA training labels* — achieves MAE=42.3 on SHA, while STEERER (Han et al., 2023), fully supervised and trained directly on SHA, achieves MAE=54.5. A DA model without target labels outperforming a supervised model trained on the target domain by 22% would be unprecedented in the counting literature. No discussion or analysis is provided. This single result demands a careful accounting: is the SHA test/train split identical across methods? Are the comparison numbers taken from published papers or re-run? Is there any inadvertent data leakage from target samples into pseudo-label generation? Without this explanation, the paper's central empirical claim is unverifiable and raises serious concerns about the validity of the experimental protocol.

- **GCC→UCF appears in the ablation (Table 4) but is absent from the main comparison tables (Tables 1–3).** The paper claims "eight dataset combinations" in the abstract and Section 4.1, yet GCC→UCF only surfaces in the ablation. This unexplained omission is suspicious — if results on this combination were unfavorable, their absence from the primary comparison would be selective reporting. The authors must include GCC→UCF in the main comparison or provide a principled justification for its exclusion.

- **Loss function formulation (Eqs. 6–7) is non-standard and likely erroneous in notation.** Equations 6 and 7 present the loss as a literal ratio of prediction losses over discriminator losses. While the paper offers a partial justification ("reversed NLL loss, maintaining $\mathcal{L}_{source}$ positive"), dividing prediction MSE by a discriminator NLL is dimensionally inconsistent, numerically sensitive to the magnitude of the denominator, and unlike any established adversarial-DA loss in the literature. If this is intended as a sum (numerator terms plus denominator terms), the notation must be corrected immediately as it is misleading. Additionally, Eq. 6 includes $\mathcal{L}_p(\hat{y}_s^b, y_s)$ — background features compared against the *full* density map — while the same equation also includes $\mathcal{L}_p(\hat{y}_s^b, \mathbf{0})$ (background against zero). These two objectives directly conflict: one penalizes the background for not being zero, the other for not matching the full density. This appears to be either a notation error (likely the first should be $\hat{y}_s^f$) or a conceptual inconsistency that undermines training correctness.

- **Missing CODA baseline.** The introduction constructs its motivation almost entirely around CODA (Li et al., 2019) as the representative failure case of prior DA counting methods. Yet CODA does not appear in any of Tables 1–3. This is the most important single comparison the paper owes the reader. Its omission makes it impossible to verify the paper's central empirical claim of superiority over task-specific counting DA.

- **Theoretical framework has a fundamental mismatch with the implementation.** Lemma 2, Lemma 3, and Theorem 4 all explicitly assume "discrete label space $\mathcal{Y}$" and treat the label set as the condition set $\mathcal{C}$. Object counting is a regression task — the label space is continuous (or high-count integer). More critically, the actual implementation uses $\mathcal{C} = \{f, b\}$ (foreground/background), which is not the label set $\mathcal{Y}$. The theory therefore does not formally analyze what the method actually does. The proof that conditional alignment on $\mathcal{Y}$ achieves $d_{JS}(D, D') = d_{JS}(\mathcal{Y}, \mathcal{Y}')$ (Theorem 4), while internally coherent, concerns a scenario that does not correspond to the implementation, and the Theorem 4 result substituted into Theorem 1 reduces the lower bound to zero — showing only that perfect label-conditional alignment is error-optimal, which is near-tautological. The theoretical advantage of the specific binary spatial conditioning used in practice is never formally established.

### Minor

- **Mask generation underdefined.** Section 3.2 states the mask "can be generated from the predicted points of objects in $\hat{y}$ by extending range" without defining what "extending range" means (Gaussian kernel? fixed pixel dilation? learned morphological operation?). This is the most critical missing implementation detail for reproducibility.

- **No oracle upper bound.** No fully supervised target-domain model (trained on target labels) is reported as an upper bound for any of the eight dataset combinations. This is standard in DA papers and is essential for calibrating how much of the performance gap remains.

- **Ablation covers only 4 of 8 combinations.** Table 4 omits the JHU-Crowd++ four-way combinations and SHA→SHB. Since the paper's claims apply to all eight combinations, partial ablation leaves the contribution verification incomplete.

- **Hyperparameter $\alpha$ (CM weight) has no sensitivity analysis.** Given the unusual ratio-based loss structure, the balance between CM, prediction, and adversarial terms is likely non-trivial, and no sweep is presented.

### Tiny

- Section 4.2 contains a duplicated sentence: "These findings indicate that BiAN effectively adapts to cross-scene crowd counting scenarios." appears in consecutive paragraphs.

- The term "Binary" in BiAN is first explained only implicitly; a brief clarification in the introduction of what "binary" refers to (the binary foreground/background partition) would help first-time readers.

---

## Nice-to-Haves

- **Oracle mask ablation:** Replace pseudo-label-derived masks with ground-truth point annotations to establish the performance ceiling and quantify how much pseudo-label noise limits the method.
- **Circular dependency analysis:** Plot mask quality (e.g., IoU with ground truth foreground) and counting accuracy jointly as training progresses to confirm the co-evolution is stable and not self-reinforcing.
- **Feature-space t-SNE or conditional distribution metrics:** Visualize whether $z^f$ and $z^b$ actually separate object and background information or merely partition spatially.
- **Computational overhead:** A brief table comparing inference time and parameter count vs. baselines is useful for practitioners.
- **Backbone generalization:** Evaluating BiAN with one alternative backbone (e.g., CSRNet) would address whether gains are specific to SAU-Net.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "BiAN outperforms DG methods — comparison conflation."** The paper explicitly labels DA vs. DG columns in Table 1 and never hides the methodological difference. The comparison is transparent and informative; this is not a genuine flaw.

- **Harsh critic: "Missing appendices."** The uploaded paper explicitly states "Rest of paper (reference and Appendix) is removed." The appendices exist in the full submission; their absence from this review copy does not imply they are missing from the paper.

- **Harsh critic: "Claim of theoretically demonstrated superior adaptability is oversold."** While the theoretical analysis has a discrete-vs-continuous mismatch (retained as a major weakness), the general direction of the claim — that conditional alignment provides a tighter lower bound framework — is directionally correct even if the proof does not fully cover the implementation.

- **Harsh critic: "Theorem 4 is trivially obvious."** The result that perfect label-conditional alignment achieves the minimum possible lower bound (zero extra contribution from feature misalignment) is not obviously trivial — it frames conditional alignment as achieving the theoretical optimum under that framework. The real problem is the discrete assumption mismatch, not triviality.

- **Statistical significance / error bars for main tables.** Single-run evaluation is the norm for large-scale crowd counting benchmarks (consistent with how all baselines are reported), so requiring multi-run statistics would be holding the paper to non-standard expectations. However, for the extraordinary SHB→SHA result specifically, a second run would be worth reporting.

- **Harsh critic: "Repeated sentence" and venue tag complaints.** Pure style/formatting nitpick.

---

## Novel Insights

The most substantive novel observation, partially identified across the reviews, is the following: the standard domain-adaptive assumption ("domain shift is task-irrelevant") is provably self-undermining for any task where density, magnitude, or count constitutes both the target label *and* a dimension of domain variation. BiAN's framing of this as a "task-relevant factor $z_{task}$" that must be preserved rather than aligned is a principled diagnostic that likely generalizes beyond crowd counting to any regression-based DA problem (depth estimation, temporal event counting, object size estimation). The Condition-consistent Mechanism, which enforces that $f(g(x)) \approx \text{concat}(f(g(x^f)), f(g(x^b)))$, is a novel self-supervised constraint that implicitly regularizes the pseudo-label quality without requiring an external teacher — a useful design pattern for bootstrapped DA. These conceptual contributions are more significant than the paper's current presentation emphasizes, and would be worth expanding if the empirical issues are resolved.

---

## Suggestions

1. **Explain or re-examine the SHB→SHA result immediately and prominently.** Verify that the STEERER comparison uses the same train/test protocol, and if the result holds, provide a hypothesis (e.g., SHA training set is small/noisy enough that SHB provides complementary signal). This is the single most important revision.

2. **Add CODA to all main tables** to close the gap between the paper's motivation and its empirical evaluation.

3. **Rewrite Eqs. 6–7 as unambiguous sums** (if that is the intent) or provide a detailed justification for the ratio formulation including convergence analysis. Clarify the $\hat{y}_s^b$ vs. $\hat{y}_s^f$ discrepancy.

4. **Revise the theoretical section** to either (a) adopt a continuous-regression-compatible framework or (b) explicitly scope the theory to a discrete approximation of density levels and discuss the gap to the binary-partition implementation. At minimum, add a paragraph bridging Theorem 4's label-set conditioning to the actual {f, b} binary partition.

5. **Add GCC→UCF to the main comparison tables** or explain in Section 4.1 why it is reserved for ablation only.

6. **Specify mask generation precisely** (kernel type, radius, threshold) in either Section 3.2 or as a dedicated implementation detail.

---

**Evaluation summary:**
- *Novelty:* Moderately high — the conditional alignment framing for counting DA is a genuine conceptual contribution.
- *Technical soundness:* Concerning — the loss formulation is unclear or erroneous, and theoretical assumptions do not match the implementation.
- *Empirical support:* Mixed — broad coverage across eight combinations is a strength, but the SHB→SHA result is unexplained and potentially unreliable, and a key motivated baseline (CODA) is absent.
- *Significance:* Moderate — if the empirical claims stand up to scrutiny, the method is practically useful; if the SHB→SHA result reflects protocol issues, significance is substantially reduced.
- *Clarity:* Below standard for ICLR — loss function notation is misleading, mask generation is underdefined, and the theory-implementation connection is not established.

MY FINAL SCORE: <pineapple>4.3</pineapple>