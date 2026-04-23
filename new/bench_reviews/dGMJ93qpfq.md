Now I have all the information needed to write the final review. Let me synthesize everything.

## Summary

The paper proposes Patch-Aware Prompting (PAP), a modular framework that integrates patch-level information into prompt tuning for vision-language models across three mechanisms: (1) patch consistency loss with intra-view and inter-view alignment, (2) view-tailored text prompts conditioned on Voronoi-clustered patch features, and (3) patch-enhanced vision features with inter-view logit regularization. PAP is designed as a plug-in add-on to existing methods (PromptSRC, DePT, CoCoOp, CoPrompt) and demonstrates consistent improvements across base-to-novel generalization (11 datasets), cross-dataset evaluation (10 target datasets), and domain generalization (4 ImageNet variants).

## Strengths

- **Modular and broadly applicable design**: PAP improves four different base methods—PromptSRC (+1.08% HM avg), DePT (+1.09% HM avg), CoCoOp (+0.85% HM), and CoPrompt (+0.84% HM) —demonstrating that the approach is not overfit to a specific architecture. Table 11 provides evidence for this versatility.

- **Extensive evaluation suite**: The paper evaluates on 11 datasets for base-to-novel generalization, 10 target datasets for cross-dataset evaluation, and 4 domain shift datasets, providing strong empirical breadth. Improvements are consistent across nearly all settings (Tables 1–3).

- **Comprehensive ablation of design choices**: Tables 4–12 systematically evaluate individual components (Table 4), loss terms (Table 5), patch loss variants (Table 6), conditioning techniques (Table 7), clustering methods (Table 8), projection/adapter choices (Table 9), crop configurations (Table 10), and augmentation strategies (Table 12), providing useful guidance for practitioners.

- **Voronoi clustering outperforms alternatives by a clear margin**: Table 8 shows Voronoi (HM 81.05) substantially outperforms KMeans (79.51) and EM (79.22), with the gap most pronounced on novel classes (+2.44% over KMeans), supporting the design choice.

- **Transparent computational cost reporting**: Table 13 reports learnable parameters (4.89M vs 0.46M for PromptSRC), GPU memory, and training time, allowing readers to assess the cost-benefit tradeoff.

## Weaknesses

### Fatal
None.

### Major

- **Core claim about patch-level information is confounded by simultaneous introduction of multiple components**: The paper's central thesis is that patch-level information improves prompt tuning generalization. However, PAP simultaneously introduces (i) a second augmented view, (ii) three new loss terms, (iii) a convolution projection layer, (iv) a text adapter, and (v) Voronoi clustering. The ablation in Table 4 decomposes the method into three coarse blocks (patch loss, view-tailored text, enhanced vision features), but each block inherently combines patch-level processing with multi-view augmentation. The critical missing baseline is: **PromptSRC + second augmented view + inter-view KL/logit consistency + inter-view ℓ₁ feature consistency, but without any patch-level components**. If this simpler multi-view baseline matches PAP's performance, the core claim about patch-level information collapses. The current evidence cannot rule this out. While the consistency of improvements across datasets and base methods makes it unlikely that all gains come from multi-view augmentation alone, the absence of this baseline significantly weakens the attribution.

- **The "first integration" claim in the abstract is overreaching**: The abstract states the method represents "the first integration of such semantics in this context," but the related work section (Section 2) explicitly acknowledges Long et al. (2024), which uses clustered patch tokens for text prompts. The distinction drawn—that Long et al. lacks inter-view consistency and patch integration into predictions—is a difference in scope, not a difference in kind that justifies a "first" claim. Additionally, self-supervised frameworks cited by the paper itself (Yao et al., 2021; Yun et al., 2022) also integrate patch-level information. The claim should be qualified (e.g., "the first to integrate patch-level semantics across vision, text, and prediction branches simultaneously in prompt tuning").

### Minor

- **Dataset-specific hyperparameter tuning without transparency**: Section 4 states "We set λp, λt, λl to 1.0, 0.1, 1.0 respectively as default but modify it for individual dataset when required," but no information is given about which datasets required tuning, what values were used, or sensitivity to these choices. Given that some improvements are sub-1% (e.g., ImageNet HM: 74.01→74.33, Food101 HM: 91.10→91.34), reporting per-dataset hyperparameters and sensitivity analysis would strengthen confidence that gains are methodological rather than tuning artifacts.

- **Notation ambiguity in equations**: The paper uses the same notation (e.g., $\mathbf{P}_{\text{an}}$) for both zero-shot and prompted patch features in Section 3.2, and Eq. 5 appears to use identical symbols for both arguments of the similarity function, which would yield zero loss. While the text description makes the intended meaning clear (alignment between prompted and zero-shot patches), the notation is confusing and could mislead readers.

- **Modest improvements on several individual datasets**: While average improvements are meaningful (+1.08% HM over PromptSRC), many individual datasets show sub-1% gains (ImageNet +0.32%, Food101 +0.34%, OxfordPets +0.43%, SUN397 +0.59%). Without variance reporting, it is hard to assess the statistical significance of these smaller improvements.

- **Significant increase in computational cost**: Table 13 shows PAP increases learnable parameters ~10× (0.46M→4.89M vs. PromptSRC) and training time ~2× (6:06→13:47 min). The cost-benefit ratio deserves more discussion, especially for the smaller improvements.

### Trivial

- The stop-gradient design in Eqs. 11 and 13 (applied to the anchor view) is justified briefly as encouraging "the augmented view to align more closely with the anchor," but no comparison with symmetric alignment is provided.

## Nice-to-Haves

- A "multi-view PromptSRC without patches" baseline, as discussed in the Major weakness section, would decisively establish whether patch-level information drives the gains.
- Visualization of Voronoi clusters on example images to demonstrate that clusters capture meaningful semantic regions (rather than being a procedural step with incidental benefits).
- Analysis of failure modes (e.g., EuroSAT shows high base but relatively low novel performance despite patch-level information).
- Variance reporting across multiple seeds for key results.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "Cannot attribute improvements to patch-level information specifically [Evidential]"** — Partially removed. The core concern about the missing "multi-view without patches" baseline is valid and kept as a Major weakness. However, the claim that "the evidence cannot rule this out" is too strong — the consistent improvements across multiple base methods and the per-component ablations (Table 4, each component contributing) provide some evidence even if not definitive. Downgraded from Fatal to Major.

- **Harsh critic: Eq. 5 notation error as "fatal" issue** — Kept as Minor, not Fatal. While the notation is genuinely confusing (same symbol for both prompted and zero-shot patches), the text description makes the meaning unambiguous. This is a presentation issue, not a methodological one.

- **Harsh critic: ConvProj not motivated** — Removed. Table 9 provides ablation evidence for the projection/adapter choices, and the text describes its role in obtaining "better feature representations." The design choice is supported empirically even if the motivation could be more explicit.

- **Harsh critic: Cross-view matching using zero-shot features introduces mismatch** — Removed as a standalone weakness; kept only as a trivial note about stop-gradient. The paper explicitly argues that zero-shot matching "prevents the model from finding an easier learning path, such as having all prompted patches match closely with a single target patch" (line 151), which is a reasonable design rationale.

- **Harsh critic: Improvements over CoCoOp/CoPrompt are modest (Table 11)** — Removed. The harsh critic treats +0.85 HM over CoCoOp and +0.84 HM over CoPrompt as "modest," but these are meaningful improvements given the difficulty of the base-to-novel generalization task and the fact that PAP is an add-on module.

- **Harsh critic: Cross-dataset improvements even smaller on average (+0.64%)** — Partially removed. The average improvement is indeed smaller in cross-dataset evaluation, but this is expected since the model is trained on ImageNet and evaluated on out-of-distribution datasets. The consistency of improvements matters more than absolute magnitude here.

- **Harsh critic: Training time roughly doubles and parameters ~10×** — Kept as Minor, not Major. The paper transparently reports these costs (Table 13), and the absolute training time (13:47 min) is still practical. The parameter increase is mostly from the adapter and ConvProj, not from fundamental architectural changes.

- **Harsh critic: No standard deviations or confidence intervals** — Kept as Minor, not Major. This is standard practice in the prompt tuning literature (most papers in this area don't report variance). The concern is valid for sub-1% improvements but is a community norm issue.

- **Strength finder: "First integration of patch-level semantics into prompt tuning for VLMs"** — Removed as a strength. This claim is contested by Long et al. (2024) and self-supervised frameworks, as discussed in the Major weakness section. A strength cannot stand when it directly conflicts with a verified Major weakness.

- **Strength finder: "Non-trivial cross-view patch matching design"** — Partially kept. The design choice is interesting but the claim that it prevents "trivial collapse" is asserted rather than empirically demonstrated (e.g., by showing what happens with prompted-prompted matching). Moved to nice-to-have for further validation.

## Novel Insights

The paper reveals an interesting asymmetry in how patch-level information can be leveraged across the three branches of a VLM pipeline: (1) in the vision branch, patches serve as regularization targets against zero-shot features; (2) in the text branch, clustered patches serve as initialization biases for view-specific prompts; (3) in the prediction branch, averaged patch projections augment class token features. This three-pronged integration pattern—where the same patch information plays qualitatively different roles (regularization target, initialization bias, feature augmentation) across branches—is the paper's most insightful design contribution, even if the empirical attribution to patch-level information specifically remains confounded.

## Suggestions

- Run the critical "multi-view PromptSRC without patches" baseline: take PromptSRC, add a second augmented view, apply inter-view KL on logits and ℓ₁ on global features between views, but use no patch loss, no patch-conditioned text, and no patch-enhanced vision features. This single experiment would either validate or substantially undermine the core claim.
- Soften the "first integration" claim to something like "the first framework to systematically integrate patch-level semantics across vision, text, and prediction branches."
- Report per-dataset λ values in a supplementary table and provide sensitivity analysis for at least one representative dataset.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Decision | Relation to PAP |
|-------|-----------|----------|-----------------|
| CoPrompt (wsRXwlwx4w) | 5.75 | Accept poster | Same domain, multi-component consistency method, questioned novelty but accepted |
| Local-Prompt (Ew3VifXaxZ) | 6.0 | Accept poster | Also leverages local information in VLMs, cleaner design |
| LogicAI-PT (BlzBcWYmdB) | 5.0 | Reject | Multi-component VLM prompt tuning, overclaimed novelty |
| CLIPSelf (DjzvJCRsVf) | 7.0 | Accept spotlight | Patch-level VLM adaptation, much stronger contribution |
| MVMP (j1FLTvgyAh) | 2.5 | Reject | Multi-prompt VLM method, overclaimed, negligible improvements |
| PLPP (2VAi5F9BOJ) | 2.5 | Reject | Overclaimed "first work" in prompt tuning |

PAP is stronger than LogicAI-PT (more extensive experiments, broader applicability) but weaker than CoPrompt (confounded ablation undermines core claim). The confounded ablation—where the contribution of patch-level information cannot be isolated from multi-view augmentation—is a significant methodological gap that places PAP below the acceptance threshold compared to CoPrompt, which had cleaner attribution of its contributions. PAP is clearly above the weak rejected papers (MVMP, PLPP). The paper makes genuine contributions (consistent improvements, modular design, extensive evaluation) but the core claim is not convincingly established.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>