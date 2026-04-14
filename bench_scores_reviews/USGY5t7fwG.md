## Summary
BiAN (Binary Alignment Network) addresses a fundamental mismatch between standard domain adaptation (DA) assumptions and the counting task: traditional DA treats all domain shifts as task-irrelevant and aligns entire feature distributions, but in counting, object density changes are task-relevant and should not be suppressed. BiAN segments images into foreground (objects) and background partitions, performs conditional adversarial alignment on each partition independently, and introduces a Condition-consistent Mechanism (CM) that enforces spatial additivity of predictions across partitions as a self-supervised signal. Experiments across eight dataset combinations in crowd and cell counting show strong improvements over prior DA and DG methods, occasionally surpassing source-only supervised baselines.

---

## Strengths

- **Precisely identified structural flaw in prior work.** The paper articulates a concrete contradiction in existing counting-specific DA methods (e.g., CODA): they acknowledge dynamic density yet still align the density distribution domain-invariantly—directly contradicting their own motivation. This is a sharply reasoned problem formulation, not merely a "new application of DA to counting."

- **Strong empirical results with a well-designed ablation.** Table 2 (SHB→SHA: BiAN 42.3 MAE vs. 110.2 for the next-best DA method CGNN-DA; SHA→SHB: 5.7 vs. 5.8 for fully supervised STEERER) and Table 4's ablation together demonstrate both that BiAN substantially outperforms prior DA approaches and that both the conditional alignment and CM components contribute meaningfully—CM alone yields −12.9 MAE improvement on SHB→SHA.

- **Binary partition + CM as a practically novel coupling.** While conditional adversarial alignment exists in general DA literature, the specific design choice of using predicted object locations as a binary spatial condition, combined with a decomposability loss (CM) as a self-supervised consistency signal to refine noisy pseudo-labels, constitutes a non-trivial application-specific architecture that is validated empirically.

- **Theory motivates the direction even if not the exact implementation.** Theorem 4's result—that feature divergence equals label divergence under conditional alignment—provides a principled argument for why aligning on subsets can be preferable to global alignment, even if the formal connection to binary spatial partitioning is left implicit.

---

## Weaknesses

### Fatal
None identified.

### Major

- **CODA is the primary competing method but is absent from all tables.** The introduction devotes a full paragraph to critiquing CODA (Li et al., 2019) as the method that most directly attempts to solve the same problem, and the paper's core contribution is framed as improving over CODA's specific failure mode. Yet CODA does not appear in Table 1, 2, or 3. Without a direct empirical comparison, the claim that BiAN solves a problem CODA fails at is unsubstantiated. This must be rectified.

- **Theory conditions on label space; implementation conditions on binary spatial partitions—the gap is never bridged.** Theorem 4 is stated for the case where the condition set C is the label space Y (discrete class labels). BiAN's actual implementation uses binary foreground/background spatial masks, which is a categorically different conditioning. Lemma 2 and Theorem 4 together say that conditioning on labels theoretically reduces the joint error bound, but the paper contains no argument—even informal—for why binary spatial partitioning yields the same or analogous benefit. This is not a minor notational issue; the central theoretical claim of the paper does not formally support its own method as implemented.

- **Loss function notation in Eqs. 6–7 is genuinely ambiguous and likely contains a typo.** The fraction bar between supervised and domain-alignment losses is unexplained and misleading—neither a ratio nor a standard combined loss notation. More concretely, Eq. 6 contains the term `L_p(ŷ_s^b, y_s)` which compares the *background-only* prediction `ŷ_s^b` against the *full* ground-truth density map `y_s`. This appears to be a typo (likely intended as `ŷ_s^f`) or a design choice that contradicts intuition and is never explained. This is not a parser artifact; it is a substantive ambiguity in the core training objective.

- **Mask generation ("extending range") is a critical implementation detail left unspecified.** The paper states: "the mask can be generated from the predicted points of objects in ŷ by extending range." The radius of dilation, thresholding strategy, and whether this is fixed or learned are nowhere specified. Because the entire conditional alignment depends on the quality of these masks, this omission prevents reproducibility and makes it impossible to assess a primary source of variance in results.

### Minor

- **No analysis of pseudo-label quality or "cold-start" robustness.** Target domain masks are derived from the model's own predictions, creating a feedback loop: noisy early predictions produce incorrect masks, which degrade conditional alignment, which degrades predictions. The CM module is described as mitigating this but no experiment (e.g., starting from a weaker initialization, or injecting artificial noise into masks) demonstrates graceful degradation or establishes the method's robustness floor.

- **Ablation covers 4 of the 8 claimed dataset combinations.** The abstract and Section 4.1 claim experiments on eight dataset combinations, but Table 4 contains only four. This inconsistency weakens the claim that the method's effectiveness generalizes across all tested scenarios.

- **Theoretical contribution is limited and Lemma 2 borders on tautological.** The key lower-bound (Theorem 1) is reproduced directly from Zhao et al. (2019). Lemma 2—"if the label set is used as condition set C, then d_C(Y, Y') = 0"—is nearly definitional: within a shared label class, both domains trivially share the same labeling function, so conditional divergence is zero by construction. The non-trivial step is Theorem 4, but its connection to binary spatial partitioning is absent. The theoretical section should either be scoped down honestly or extended to cover the actual implementation.

- **No limitations section.** The paper does not acknowledge any of its limitations—neither the potential failure of the foreground/background split in extremely dense scenes (acknowledged obliquely in the ablation discussion but not formalized), nor the computational overhead, nor the reliance on pseudo-label quality.

### Tiny

- **Language error in contribution bullets.** "The existing DA methods contempt the dynamic density" (Introduction) should be "neglect" or "ignore." Minor but affects readability.
- **No sensitivity analysis for α** (the CM loss weight in Eq. 5). A single-sentence justification or brief sweep would suffice.

---

## Nice-to-Haves

- **Density distribution preservation metrics.** The paper's core claim is that BiAN preserves task-relevant density information while baselines destroy it. Adding a distributional metric (e.g., KL divergence between predicted and ground-truth count distributions per dataset) would provide direct empirical evidence for the mechanism, not just downstream MAE.

- **Visualization of conditional feature separation.** A t-SNE or activation map comparison showing that object features are aligned across domains while background/density features remain domain-distinct would make the paper's mechanism visually compelling.

- **Oracle upper bound.** Including a fully supervised result trained on the target domain for each combination would allow readers to assess how close BiAN is to the performance ceiling.

- **Backbone generalizability check.** BiAN is built on SAU-Net; a brief experiment with one alternative backbone would increase confidence in the method's general applicability.

- **Failure case analysis.** Showing examples where the foreground/background segmentation degrades—e.g., extremely dense crowds—and the resulting impact on counting would strengthen the paper's empirical honesty.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **No statistical significance testing / confidence intervals (Harsh Critic).** Single-run evaluation is the standard practice in crowd and cell counting benchmarks. Requesting multi-run statistics or confidence intervals imposes a non-standard norm for this community and is therefore removed.

- **Mixing DA and DG methods in Table 1 (Harsh Critic).** The table explicitly labels every method with DA/DG checkmarks, making the distinction transparent. Including DG baselines gives readers a broader context. This is a common practice and not a flaw.

- **Dense crowd failure mode as a fatal/major concern (Harsh Critic).** The empirical results (BiAN achieves 42.3 MAE on SHB→SHA, a high-density dataset, competing with supervised methods) do not support the hypothesis that dense scenes catastrophically degrade the method. The ablation section acknowledges "severe overlap situations" in crowd counting and frames CM as addressing it. The concern is not empirically demonstrated as a failure.

- **Unfair mixing of supervised upper-bound with DA baselines (Harsh Critic's framing of Table 2 comparison).** Comparing against source-only supervised baselines in the same table is standard practice to show the DA performance gap; it is not misleading.

- **Claim about SAU-Net backbone limiting results (Harsh Critic).** Evaluating on one well-established backbone is standard; demanding multi-backbone evaluation is not a standard ICLR requirement for a method paper.

---

## Novel Insights

The most genuinely novel analytical observation in the reviews—confirmed by the paper—is the precise contradiction within prior counting-specific DA methods: CODA and related methods acknowledge that density is task-relevant but then proceed to *align* the density distribution anyway, which directly violates their own stated premise. This is sharper than the generic claim that "standard DA doesn't work for counting"; it identifies an internal inconsistency in the prior literature's own framework. The theoretical observation that conditioning on the label space reduces the feature divergence to the label divergence (Theorem 4) is also interesting, though its connection to spatial binary partitioning remains an open formal gap that, if bridged, could constitute a meaningful theoretical contribution to conditional DA theory beyond counting.

---

## Suggestions

1. **Add CODA to Tables 1–3.** This is the single most important fix. Run CODA on all experiment splits and include it as a direct comparison. The narrative of the paper depends on this comparison being visible.

2. **Fix or clarify Eqs. 6–7.** Replace the fraction bar with explicit summation notation. Confirm whether `L_p(ŷ_s^b, y_s)` is intentional or a typo for `L_p(ŷ_s^f, y_s)`, and justify whichever is correct.

3. **Specify mask generation precisely.** State the dilation radius, threshold value, and whether these are dataset-specific hyperparameters or fixed constants. Include a brief sensitivity analysis or at minimum a table in the appendix.

4. **Bridge theory to implementation.** Either (a) add a corollary or remark showing that binary spatial partitioning is a valid instantiation of the label-conditioned theory, or (b) explicitly scope the theory as motivational and reformulate the contribution list accordingly. As it stands, Theorem 4 and the implementation are in different conceptual universes.

5. **Expand ablation to cover all 8 dataset combinations**, or explicitly explain why 4 are representative and move the remaining results to the appendix with a cross-reference.

6. **Add a limitations section** acknowledging at minimum: pseudo-label noise sensitivity, potential issues in extreme-density scenes, and the theory-implementation gap.

---

**Evaluation:**
- **Novelty:** Moderate-to-high — the problem identification is sharp; the technical mechanism is incremental relative to conditional DA literature but well-targeted.
- **Technical soundness:** Moderate — the empirical results are compelling but the loss function ambiguity, the unspecified mask generation, and the theory-practice gap introduce genuine uncertainty about the method as described.
- **Empirical support:** Moderate-to-high — broad coverage across 8 combinations, strong ablation for a DA paper, but CODA's absence from tables is a notable gap.
- **Significance:** High for the counting community, where domain shift is a practical bottleneck.
- **Clarity:** Below standard for ICLR — the method section has multiple ambiguities that prevent reproducibility without the appendix, and the loss formulation in Eqs. 6–7 requires correction.