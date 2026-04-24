Now let me do calibration searches.Now I have sufficient information to write the consolidated review. Let me synthesize everything.Now I have all the information needed to write the final consolidated review.

---

## Summary

PAP (Patch-Aware Prompting) is a modular prompt-tuning framework for vision-language models (CLIP) that incorporates patch-level information at three stages: (1) an intra/inter-view patch consistency loss aligning prompted patch features with zero-shot anchor patches, (2) view-tailored text prompts conditioned on Voronoi-clustered patch features, and (3) patch-enhanced vision features with inter-view KL-divergence logit regularization. Evaluated on top of PromptSRC and DePT across base-to-novel generalization (11 datasets), cross-dataset evaluation, and domain generalization, PAP consistently outperforms baselines with an average HM gain of ~1% over PromptSRC.

---

## Strengths

- **Consistent multi-benchmark improvements (Tables 1–3):** PAP+PromptSRC achieves 81.05% average HM across 11 datasets (+1.08% over PromptSRC's 79.97%), with gains in all 11 individual datasets. The cross-dataset evaluation (Table 2) shows +0.64% average over PromptSRC, and domain generalization (Table 3) shows +0.31% average — all consistent rather than cherry-picked.

- **Voronoi-based view-tailored text prompts are novel and well-validated (Tables 7–8):** Using patch-cluster biases per prompt vector (Eqs. 9–10) outperforms both CoCoop-style global conditioning (79.45 vs. 81.05 HM) and cross-attention conditioning (80.24 vs. 81.05 HM). Voronoi clustering outperforms KMeans (79.51) and EM (79.22) on novel classes, providing genuine empirical support.

- **Modular applicability demonstrated across multiple base methods (Table 11):** PAP improves CoCoop (+0.85% HM), CoPrompt (+0.84% HM), PromptSRC, and DePT, demonstrating broad applicability rather than tuning specifically for one baseline.

- **Comprehensive ablation landscape (Tables 4–12):** Design choices are systematically studied: component ablations (Table 4), loss ablations (Table 5), intra/inter patch loss (Table 6), conditioning type (Table 7), clustering method (Table 8), projection and adapter design (Table 9), crop strategy (Table 10), and augmentation strategy (Table 12). This is above the norm for the prompt tuning literature.

- **Intra+inter patch loss both necessary (Table 6):** Removing either the intra-view or inter-view patch loss degrades novel class accuracy (75.71–76.02 vs. 77.41), confirming both are independently necessary — not just the presence of an augmented view.

---

## Weaknesses

### Fatal
None.

### Major

- **Multi-view consistency confound — the central attribution claim is unestablished.** Every component of PAP (patch loss, view-tailored text, inter-view logit KL) requires the presence of the augmented view. The paper never ablates a control that adds only the augmented view with *global-level* inter-view consistency (as in PromptSRC applied between two views) but without any patch-specific component. The missing ablation is: *PromptSRC + augmented view + global inter-view losses (no patches)*. Table 5 partially mitigates this concern — removing all PAP losses while keeping the augmented view yields HM ≈ 79.98, nearly identical to PromptSRC (79.97), suggesting the augmented view alone contributes nothing — but the specific comparison of patch-level vs. global-level inter-view consistency is still absent. Until this ablation exists, the paper cannot establish that *patches specifically* (rather than a finer-grained inter-view consistency of any kind) drive the gains. This is the paper's core scientific claim.

- **Undisclosed per-dataset hyperparameter modifications undermine reproducibility and evaluation integrity.** The paper explicitly states (Section 4): *"We set λp, λt, λl to 1.0, 0.1, 1.0 respectively as default but modify it for individual dataset when required."* Neither the datasets receiving non-default values nor the modified values themselves are disclosed. With performance gains in the range of 0.3–1.5% on individual datasets, undisclosed tuning across an 11-dataset benchmark is a meaningful concern. The practice is disclosed in principle but not in specifics, making full reproduction impossible.

### Minor

- **Notation inconsistency between zero-shot and prompted anchor features (Section 3.2).** The paper uses the same variable name `P_an` for both zero-shot anchor patches (introduced at the start of 3.2) and for prompted anchor patches (re-introduced later in the same section). This collision makes it ambiguous what the intra-view loss in Eq. 5 is actually comparing — the description says "prompted vs. zero-shot outputs," but the notation does not cleanly reflect this.

- **No runtime comparison against a longer-trained PromptSRC baseline.** Training time roughly doubles vs. PromptSRC (13:47 vs. 6:06 min, Table 13). The paper does not compare against running PromptSRC for twice as many epochs or with stronger augmentation, leaving open whether the additional compute budget alone could explain the gains.

### Trivial
None beyond parser artifacts removed below.

---

## Nice-to-Haves

- **Multi-seed variance on core results.** The prompt tuning literature (CoOp, CoPrompt) is known for seed sensitivity. Reporting mean ± std across 3 seeds for base-to-novel generalization would strengthen confidence in ~1% gains.

- **Patch cluster visualization.** Showing Voronoi clusters on representative images from diverse datasets (especially EuroSAT, DTD, FGVC Aircraft) would verify that clustering captures semantically meaningful local structures.

- **Global inter-view control ablation.** Adding a PromptSRC + augmented view + global consistency baseline (without patches) would cleanly establish the patch-specific contribution.

---

## Removed Points

*These points were removed per the review rules — treat with caution.*

- **Eq. 5 self-similarity error (Harsh Critic).** The equation appears to show `sim(P̃_an^i, P̃_an^i)` (two identical arguments), which the critic argues makes the loss identically zero. Verified: this is a parser artifact — the separator between arguments was misread as a minus sign. The description clearly intends comparison between prompted and zero-shot patches. Removed per the rule against formatting artifact criticisms.

- **Tables 4 and 5 are unreadable due to formatting failure (Harsh Critic).** Verified: all cells in these tables show ✓ for all configurations because the parser converted ✗ marks to ✓. This is definitively a parser artifact — the numbers vary meaningfully across rows, so the original table clearly had distinct on/off patterns. Removed per the formatting artifact rule.

- **ConvProj spatial reshaping details omitted (Harsh Critic).** Applying 3×3 convolution to ViT patch token sequences requires 2D reshaping (e.g., 14×14). Implementation detail likely in the stripped appendix. Removed per the rule on appendix-deferred implementation details.

- **No theoretical justification for ConvProj over MLP (Harsh Critic).** Table 9 compares ConvProj vs. Adapter empirically; demanding further theoretical justification is outside the empirical scope of this systems paper.

- **Strength Finder: "represents the first integration of such semantics in this context."** This is a generic novelty framing from the abstract, not a concrete empirical strength. Removed per rule on generic strengths.

---

## Novel Insights

The paper's most actionable insight, not fully articulated, is that the augmented view alone contributes negligibly (ΔHM ≈ 0.01% over PromptSRC per Table 5's first row), while adding the patch losses yields a substantial +1.07% HM gain. This suggests patch-level regularization — not multi-view training per se — is the operative mechanism. This finding would be stronger if reported explicitly with the global inter-view control. A secondary insight is that Voronoi clustering substantially outperforms KMeans and EM for this task (+1.5–2.4% novel accuracy), a result that is somewhat surprising and may inform future prompt conditioning design.

---

## Suggestions

1. **Add the critical control ablation**: PromptSRC + augmented view + PromptSRC's existing global SCL losses applied between two views (no patch components). This single row in Table 5 would either confirm or refute the patch-attribution claim.
2. **Fully disclose per-dataset hyperparameter values**: Add a table in the appendix listing which datasets received modified λ values and the specific values used. This is a transparency fix, not a methodology change.
3. **Fix the notation conflict**: Use distinct symbols (e.g., P̄_an for zero-shot, P̃_an for prompted) consistently throughout Section 3.2 and all equations.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison to PAP |
|---|---|---|---|
| wsRXwlwx4w (CoPrompt) | 5.75 | Accept (poster) | Direct predecessor; PAP adds more components and novelty but has additional confound concern |
| Ew3VifXaxZ (Local-Prompt) | 6.00 | Accept (poster) | Comparable scope: local features for VLM prompting; no confound issue |
| jzzEHTBFOT (C-TPT) | 6.00 | Accept (poster) | Test-time prompt tuning for CLIP; similar benchmark coverage; no undisclosed tuning |
| 2x1U8a3s7G (Prompt Diffusion) | 5.00 | Reject | Weaker baselines, less comprehensive ablation than PAP |
| dO06t9iVO3 (MoA Domain Gen) | 3.00 | Reject | Weak results, unfair baselines, missing ablation — much weaker than PAP |

PAP's breadth and consistency (3 benchmarks × 11+ datasets) and comprehensive ablation place it above CoPrompt (5.75) in contribution, but the missing multi-view control ablation and undisclosed per-dataset tuning are genuine methodological concerns absent in Local-Prompt and C-TPT (both at 6.0). The paper falls between these anchor bands. The multi-view confound is partially mitigated by Table 5 but not definitively resolved. Given that: (a) the contribution is real and consistent; (b) the ablation is comprehensive; (c) the novelty (Voronoi-conditioned prompts, patch inter-view loss) is concrete; but (d) the paper cannot fully attribute gains to patches vs. multi-view training, and (e) reproducibility is limited by undisclosed per-dataset tuning — I score this at **5.5**, borderline reject, centered between the rejected ~5.0 and accepted ~6.0 anchors.

**Originality:** Good — patch-level integration for prompt tuning is novel; Voronoi conditioning is a concrete mechanistic contribution  
**Importance:** Moderate — incremental advance in an active area  
**Claim support:** Partially supported — consistent empirical gains, but central attribution (patches vs. views) unresolved  
**Experimental soundness:** Adequate — broad benchmarks, comprehensive ablation, but missing the key control  
**Clarity:** Good — writing is clear; notation inconsistency in Section 3.2  
**Community value:** Moderate — modular framework applicable to multiple methods is useful; unresolved confound limits scientific impact

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>