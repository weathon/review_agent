Now let me read the remaining sections of the paper to verify remaining claims.---

## Summary

Patch-Aware Prompting (PAP) is a modular framework that integrates local patch-level information into vision-language model prompt tuning across three branches: (1) a patch-consistency loss with intra-view and inter-view alignment to frozen CLIP patch features, (2) view-tailored text prompts conditioned on Voronoi-clustered patch tokens with inter-view text consistency loss, and (3) patch-enhanced vision features feeding into KL-divergence-regularized logit consistency. PAP is applied on top of PromptSRC and DePT and evaluated on base-to-novel generalization, cross-dataset, and domain generalization benchmarks across 11 datasets, consistently improving over its host methods.

---

## Claims and Support

**Claim 1 – Patch-level information improves generalization over prior global-consistency methods.**
*Partially supported.* Tables 1–3 show consistent directional improvements over PromptSRC and DePT (e.g., +1.08% HM on base-to-novel average over PromptSRC, +0.64% on cross-dataset, +0.31% on domain generalization). However, PAP bundles several simultaneous changes (two-view training, patch losses, clustered conditioning, ConvProj, adapters, logit KL), and ablation tables 4–6 as rendered cannot be cleanly read due to notation/presentation issues (see Weaknesses). The attribution of gains to patch-level information specifically rather than the full bundle is only partially isolable from the current evidence.

**Claim 2 – PAP is modular and can be applied to existing methods.**
*Partially supported.* Table 11 demonstrates improvements on CoCoOp and CoPrompt (+0.85% HM and +0.84% HM respectively), and main results use PromptSRC and DePT as hosts. However, as verified from Table 13, PAP increases learnable parameters from 0.46M to 4.89M (~10×) and roughly doubles training time (6:06 → 13:47). The "modular" framing is overstated relative to a lightweight plug-in.

**Claim 3 – The patch consistency loss prevents catastrophic forgetting and improves vision representations.**
*Inadequately specified.* The text clearly states the intent: align prompted patches with zero-shot CLIP patches. However, Equation (5) as written reads $\sum_{i=1}^M (1 - \text{sim}(\tilde{\mathbf{P}}_{\text{an}}^i - \tilde{\mathbf{P}}_{\text{an}}^i))$, which compares a quantity with itself, yielding a degenerate result. Additionally, Section 3.2 introduces a genuine notation collision: both zero-shot and prompted anchor patch features are labeled $\mathbf{P}_{\text{an}}$. The intent is recoverable from context but the formal specification is ambiguous and must be corrected. Functional ablation support for intra vs. inter loss is present in Table 6 (numbers differ across rows) but unverifiable due to rendering of check marks.

**Claim 4 – Patch-conditioned, view-tailored text prompts outperform global conditioning.**
*Supported with caveats.* Table 7 isolates three conditioning approaches and shows patch-cluster conditioning outperforms CoCoop global conditioning and cross-attention alternatives. Table 8 shows Voronoi clustering outperforms KMeans and EM. These are the paper's cleanest ablations, though "Ours" in Table 7 is the full system, not purely a conditioning swap.

**Claim 5 – PAP achieves SOTA or superior performance across benchmarks.**
*Supported in direction, overstated in framing.* Improvements are consistent but often modest (0.3–1.1% range). No variance estimates are reported, which matters when claiming superiority at fine-grained margins, especially on domain generalization (+0.31% over PromptSRC).

**Claim 6 – Extensive ablations validate the effectiveness of each component.**
*Inadequately presented.* Tables 4–6 have identical check-mark patterns across all rows in the extracted text (rendering artifact or genuine table error). While the differing numerical values confirm that different configurations were run, the ablation tables as submitted are not interpretable without knowing which components are active in each row. This substantially weakens the empirical case for each component's individual contribution.

---

## Strengths

- **Breadth of evaluation:** Three distinct evaluation protocols (base-to-novel, cross-dataset, domain generalization), 11 datasets, and application to four host methods (PromptSRC, DePT, CoCoOp, CoPrompt). The empirical improvements are consistent in direction, not cherry-picked.
- **Novel perspective:** PAP is the first work to integrate patch-level consistency simultaneously across the vision encoder, text prompts, and logit regularization in the prompt-tuning framework. While individual components have precedents (CoCoop-style conditioning, CoPrompt-style inter-view consistency), the cross-branch patch-level integration is genuinely new.
- **Clean isolation of text prompt conditioning (Tables 7–8):** The comparison of CoCoop global conditioning vs. attention-based vs. Voronoi patch clustering, and the comparison of KMeans/EM/Voronoi, provide the paper's most interpretable ablations. The Voronoi approach shows a clear advantage on novel classes.
- **Transparency about efficiency costs:** Table 13 directly reports parameter counts, memory, and training time, including the cost increase. The paper is not hiding this trade-off.
- **Modular design demonstrated empirically:** Table 11 shows gains when PAP is added to CoCoOp and CoPrompt, validating that the framework generalizes beyond the primary test beds.

---

## Weaknesses

### Fatal
*None that irrevocably undermine the core contribution—the method works empirically.*

### Major

- **Equation (5) notation error / inconsistent symbols in Section 3.2** — The intra-view patch consistency loss is written as $\sum_{i=1}^M (1 - \text{sim}(\tilde{\mathbf{P}}_{\text{an}}^i - \tilde{\mathbf{P}}_{\text{an}}^i))$, which is identical on both sides and evaluates to zero. The surrounding text clearly states the intent is to compare prompted patches with zero-shot patches, but the equation as written contradicts this. This is compounded by a genuine notation collision in Section 3.2 where both zero-shot and prompted anchor patch features are labeled $\mathbf{P}_{\text{an}}$ in the same paragraph. Since the patch consistency loss is the paper's primary technical contribution, this must be corrected with unambiguous notation. (Verified from extracted text lines 137–143.)

- **Ablation tables 4–6 are unreadable as presented** — All rows in Tables 4, 5, and 6 display identical check marks, making it impossible to determine which components or losses are active in each row. While the varying numerical values confirm different configurations were evaluated, a reader cannot verify what is being ablated. This directly undermines the paper's claim of "extensive ablation studies confirming each component." Whether this is a PDF rendering issue or a genuine table error, it must be fixed. (Verified from extracted text lines 296–321.)

- **Efficiency framing is misleading** — The paper describes the parameter increase as "minimal" and "slight" (Section 4.5, p. 9). Table 13 shows: PromptSRC has 0.46M learnable parameters; PAP+PromptSRC has 4.89M (~10× increase). Training time roughly doubles (6:06 → 13:47). In the prompt-tuning community where parameter efficiency is a central motivation, framing a 10× increase in learnable parameters as negligible is inaccurate and misleading. This doesn't disqualify the method but must be stated honestly.

### Minor

- **Per-dataset hyperparameter tuning** — The paper states "we set $\lambda_p, \lambda_t, \lambda_l$ to 1.0, 0.1, 1.0 respectively as default but modify it for individual dataset when required." With three loss weights plus $\alpha$ potentially tuned per-dataset, the generality of reported improvements is uncertain. A single fixed hyperparameter setting or a sensitivity analysis should be provided.

- **No statistical significance estimates** — Many improvements are in the 0.3–1.1% range. Without multi-seed variance or confidence intervals, especially for domain generalization improvements (+0.31%), it is difficult to assess which gains are robust. While single-run evaluation is common in this field, the modest magnitudes make this matter more here.

- **Voronoi clustering implementation is underspecified** — Voronoi diagrams partition geometric space; applying them to high-dimensional feature vectors requires explaining how seeds are initialized and how cluster membership is computed. Table 8 shows empirical superiority over KMeans and EM, but the operational definition of "Voronoi\_Clustering($\bar{P}$)" in Eq. (9) needs to be explicitly described in the methodology or appendix.

### Trivial
- Equation (13) has a mismatched parenthesis in the extracted text (an extra closing parenthesis), likely a typo that should be corrected.

---

## Nice-to-Haves

- **Multi-seed runs** on at least the average metrics in Tables 1–3 to confirm robustness.
- **Experiments with ViT-L/14 or larger backbones** — All results use ViT-B/16; whether patch-level gains persist at larger backbone scale is an open question.
- **Visualization of Voronoi patch clusters** — Overlaying cluster assignments on images would verify whether they capture semantically meaningful local regions (object parts, textures, etc.) and strengthen the narrative.
- **Feature-space analysis** — t-SNE/UMAP showing prompted vs. zero-shot patch representations with and without PAP would directly validate the "prevents catastrophic forgetting" claim rather than inferring it from accuracy changes alone.
- **Single fixed hyperparameter ablation** — A table showing PAP with fixed $\lambda_p=1.0, \lambda_t=0.1, \lambda_l=1.0$ across all datasets (without per-dataset tuning) would strengthen the generality claim.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "Experiments do not cleanly isolate patch-level semantics as the driver of gains vs. extra compute/capacity"** — *Partially removed as standalone weakness.* The concern is valid in principle, but this is standard for empirical systems papers that add bundled components. The paper does provide multi-component ablations (Tables 4–6); the issue is their readability, which is already captured in the major weakness about ablation table presentation. The broader epistemological concern about multi-component attribution is a weaker version of a common critique applicable to nearly all prompt-tuning papers and has been absorbed into the framing discussion.

- **Harsh Critic and Spark: "No comparison with Long et al. (2024)"** — *Removed.* The paper explicitly addresses this in Section 2: "Independently developed, Long et al. (2024) uses clustered patch tokens for text prompts but lacks inter-view consistency and patch integration into predictions, underperforming compared to PromptSRC." The paper cannot be penalized for not including a comparison with a concurrent method it already acknowledges as weaker than its baseline (PromptSRC). The text comparison is reasonable.

- **Human Finder: "Incremental novelty of individual components"** — *Removed/weakened.* Each of the three consistency mechanisms has precedents, but the patch-level application and cross-branch integration are a genuine contribution. Per the comparable papers' acceptance (CoPrompt, DeKg), incrementality at this level is not grounds for rejection in this community. The paper could argue its novelty more forcefully, but the critique as framed (citing other papers' reviewer comments as authority) is rhetorical rather than substantive.

- **Multiple reviewers: "No comparison with PromptKD / CasPL"** — *Removed.* Per review guidelines, missing related works comparisons are not flagged. Furthermore, PromptKD and CasPL use larger teacher models (CLIP-L/14), making them an unfair asymmetric comparison where the baseline is advantaged—this asymmetry explicitly favors the baseline, not PAP.

- **Harsh Critic: "Causal claim that Eq. (5) breaks the method specification"** — *Partially downgraded from fatal to major.* The equation has a genuine notation problem, but the surrounding text makes the intent unambiguous, and the method demonstrably produces consistent empirical improvements across 11 datasets and 4 host methods. The specification error must be corrected but does not invalidate the contribution.

---

## Novel Insights

The most genuinely novel insight in this paper is the asymmetric inter-view patch matching strategy in Eq. (6): the augmented view's prompted patch features are matched to the **zero-shot** anchor patch features (rather than the prompted anchor patches) when computing the inter-view consistency target. The paper argues—and the Tables 6 and 9 ablations partially support—that this prevents collapse (all prompted patches collapsing to a single anchor patch), a failure mode that would not arise in purely global feature consistency methods. This is a subtle but practically important design choice that has implications for any local-feature consistency approach in prompt tuning. The use of frozen CLIP features as matching anchors to prevent degenerate solutions generalizes naturally beyond this specific framework.

---

## Suggestions

1. **Fix Equation (5) and notation throughout Section 3.2.** Introduce unambiguous notation: e.g., use superscript "$zs$" for zero-shot features and "$pt$" for prompted features consistently. The intra-view loss should read $\sum_{i=1}^M (1 - \text{sim}(\tilde{\mathbf{P}}^{pt,i}_{\text{an}}, \mathbf{P}^{zs,i}_{\text{an}}))$ or equivalent.

2. **Reconstruct Tables 4, 5, 6** so each row clearly shows which components/losses are enabled (✓) or disabled (✗). At minimum, add a caption explaining each row's configuration in text if tables are not fixed.

3. **Rewrite Section 4.5 efficiency discussion** to accurately characterize the ~10× parameter increase as a real cost. Provide a cost-benefit discussion: e.g., "PAP achieves +1.08% HM improvement at 10× the learnable parameters and 2× training time relative to PromptSRC; a subset of components (patch loss only, or text conditioning only) may provide a better trade-off for resource-constrained settings."

4. **Add Voronoi implementation details** in the appendix: seed initialization strategy, how cluster membership is assigned in high-dimensional feature space, and whether clusters are recomputed per image or per batch.

5. **Report fixed-hyperparameter results** for all 11 datasets in Table 1 without per-dataset tuning to validate generality. The gap between tuned and fixed results (if any) should be disclosed.

---

## Score and Decision

**Originality:** Moderate-to-good. Patch-level consistency in prompt tuning is genuinely novel; the individual components have precedents but the cross-branch integration and asymmetric zero-shot-matching strategy are original.

**Importance of research question:** High. Preventing overfitting and preserving CLIP generalization under prompt tuning is a central open problem with direct practical implications.

**Claims vs. support:** Fair. Empirical claims are well-supported directionally across broad benchmarks. The mechanistic claim (patch-level features specifically drive gains) is plausible but not cleanly isolated. Core technical specification (Eq. 5) needs correction.

**Soundness of experiments:** Adequate. Coverage is broad and multi-method. Ablation presentation is the main weakness. No variance reporting.

**Clarity of writing:** Mixed. Related work, experimental setup, and results sections are clear. Section 3.2 notation is genuinely confusing and must be corrected.

**Value to the research community:** Positive. The framework is practically useful, consistently improves strong baselines, and introduces a perspective (local patch-level consistency) that can inform follow-up work.

**Overall:** This is a solid empirical contribution with a real idea, consistent results, and sufficiently broad evaluation to be publishable. The primary revision requirements are notational (Eq. 5 and Section 3.2 symbols) and presentational (ablation tables 4–6 and efficiency discussion). These are correctable issues that do not undermine the paper's actual contribution. The work is of comparable quality to accepted papers in the same sub-area (CoPrompt, DeKg).

**Score: 6.0** — Weak Accept. The paper makes a genuine and useful contribution to prompt tuning for VLMs. Acceptance is contingent on correcting the Eq. (5) notation error, reconstructing the ablation tables so they are interpretable, and honestly recharacterizing the efficiency trade-off.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>