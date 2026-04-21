Now I have a thorough understanding of the paper. Let me consolidate the review carefully.

## Summary

The paper introduces a "second-order lens" for interpreting individual neurons in CLIP-ViT by tracing how each neuron's activation propagates through subsequent attention head OV (output-value) pathways to the final output representation. The key empirical findings are that second-order effects concentrate in late layers, are sparse (<2% of images per neuron), and can be approximately captured by a single direction in CLIP's joint text-image space, enabling sparse text decomposition that reveals polysemantic behavior. Two applications are demonstrated: generating "semantic" adversarial examples using polysemantic spurious correlations, and zero-shot segmentation via neuron activation aggregation.

## Strengths

- **Clear methodological motivation and improvement over alternatives (Table 1):** The paper convincingly shows that second-order effects overcome significant limitations of direct effects (negligible for MLP neurons) and indirect effects (obscured by self-repair). Table 1 demonstrates that mean-ablating second-order effects of layer 9 neurons drops accuracy to 29.6% vs. 52.3% for indirect effects, and the first PC explains 48.2% of variance vs. only 11.0% — a substantial improvement in interpretability per component.

- **Principled sparse decomposition (Section 4, Figure 4):** The use of orthogonal matching pursuit over text embeddings to decompose each neuron's rank-1 direction is a clean and well-motivated approach. Figure 4 shows convergence of reconstruction accuracy across different text pools at large set sizes, supporting the robustness of the decomposition.

- **Novel adversarial application exploiting polysemy (Section 5.1, Table 3):** Using polysemantic neuron structure to identify spurious cross-class correlations and generating adversarial images via an LLM + text-to-image pipeline is genuinely creative. Table 3 shows the method substantially outperforms all baselines (e.g., 22.7% vs. 6.3% for indirect effects on dog→deer), and the baselines use the same generation pipeline, controlling for distribution shift.

- **Well-calibrated selectivity and rank-1 characterization (Section 3.3, Figure 3):** The empirical demonstration that second-order effects are sparse (significant for <2% of images) and that rank-1 reconstruction preserves classification accuracy nearly perfectly ("rec. from PC #1" line in Figure 3) provides a solid foundation for the decomposition approach.

## Weaknesses

### Fatal
None.

### Major

- **Rank-1 approximation discards majority of per-neuron variance with no individual-neuron fidelity validation.** Table 1 reports that the first PC explains only 48.2% of variance in second-order effects — meaning more than half is discarded. The validation in Figure 3 (replacing all neurons simultaneously with rank-1 approximations) shows this preserves classification accuracy, but collective error cancellation can mask individual approximation failures. Since all interpretability claims (Table 2 descriptions, Figure 5 polysemanticity, adversarial generation) depend on individual neurons' rank-1 directions being faithful, the absence of per-neuron variance-explained distributions or per-neuron reconstruction fidelity metrics leaves a gap. A neuron whose effect is poorly approximated by one direction would yield a misleading text decomposition, and the paper provides no way to identify such neurons. The collective accuracy preservation is necessary but not sufficient for interpretability claims.

- **The second-order lens systematically excludes QK effects, which means it cannot detect neurons whose primary functional role is modifying attention patterns.** The paper acknowledges this in Section 6 ("we ignored the effect of neurons on consecutive queries and keys"), but the framing throughout — including the abstract's claim to "interpret the function of individual neurons" and Section 3.2's presentation — implies a more complete characterization than delivered. While the title and lens name are appropriately scoped ("second-order effects"), key passages overstate the coverage. For instance, the abstract says analyzing other effects "fails to capture the neurons' function in CLIP," implicitly positioning the second-order lens as the lens that *does* capture function — yet neurons that primarily reroute attention via QK pathways would be invisible to this method. This is a scope limitation rather than a methodological error, but the framing should be more precise about what functional roles are and are not captured.

### Minor

- **The adversarial evaluation is narrowly scoped (binary CIFAR-10 tasks, ~100 images per task).** While the proof-of-concept is compelling and baselines are well-controlled, the scale leaves open whether the approach works for multi-class or real-world classification settings. The paper acknowledges pipeline failure modes (Section 6), and the binary setting is reasonable for initial demonstration, but generalization remains unvalidated.

- **The segmentation improvement over TextSpan (the authors' own prior method) is marginal.** Table 4 shows gains of +1.6 pixel accuracy, +0.9 mIoU, and +0.8 mAP over TextSpan, with no variance reported. Other baselines are not competitive (dating from 2016–2020). This application is a supplementary contribution and does not affect the core interpretability claims, but the limited gains weaken the practical utility argument.

- **Some common-word decompositions in Table 2 appear semantically noisy.** For instance, Neuron #4's common-word decomposition includes "closings" and "advent" alongside "snowy" and "frost," while Neuron #391 includes "swelling" alongside "woodworking" and "cedar." While polysemy can produce unexpected associations, some entries read like embedding geometry artifacts rather than genuine semantic features. The class-description pool produces cleaner results, and the paper could benefit from analyzing the gap between decomposition pools.

### Trivial
None.

## Nice-to-Haves

- Per-neuron variance-explained distributions (a histogram across neurons of how much variance PC1 captures) would directly address the major fidelity concern and help readers understand which neurons' descriptions are trustworthy.
- A control experiment where random (non-neuron-derived) descriptions are fed through the same text-to-image pipeline, with the same filtering, would further isolate the contribution of spurious cue identification from distribution shift.
- Failure case analysis for the text decomposition — showing neurons where the decomposition is clearly wrong — would establish the method's limits and build trust.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Claim that "second-order effect" is just a first-order Taylor expansion or that the naming is misleading.** The method traces signal through nonlinear attention heads' OV pathways, which is genuinely different from a first-order expansion. The paper names this "second-order" by analogy to the "logit lens" (first-order), and explicitly defines it as flow through subsequent attention heads. The naming is internally consistent and the derivation is clearly presented.

- **Claim that the small adversarial sample size (~100 images) produces unreliable results.** While small, the effect sizes are large (e.g., 22.7% vs. next-best 6.3%) and three baselines use the same pipeline, making the relative comparison meaningful. Scale is a legitimate concern for generalization but not for validity of the reported results.

- **Claim that generated images introduce a distribution shift confound that invalidates adversarial results.** All four conditions (second-order, indirect, similar words, random) use the same text-to-image generation pipeline, so distribution shift affects all equally. The comparison isolates the contribution of neuron-derived cues.

- **Demand for multi-class adversarial evaluation or larger-scale validation as a missing experiment.** This is scope creep — the paper demonstrates proof of concept for binary tasks and explicitly acknowledges limitations in Section 6.

- **Criticism that manual filtering exclusion rate is not reported.** The filtering is a small quality-control step to remove obviously invalid images (containing wrong class objects), and is applied uniformly across all conditions. Not a substantive methodological concern.

- **Request for causal validation of text decompositions (activating identified concepts to increase neuron effects).** This would be a nice addition but goes beyond the paper's scope, which focuses on correlational validation (top-activating images match described concepts) and functional validation (accuracy preservation under replacement).

- **Criticism of the 5000 ImageNet training images used for mean-ablation and the unspecified significance threshold for <2%.** The sparsity finding is empirically demonstrated in Figure 3 through ablation experiments that show clear separation between "large norm" and "small norm" images. The threshold emerges from the data rather than being an arbitrary choice.

- **Formatting and presentation nitpicks.** Removed per instructions.

## Novel Insights

The key insight that elevates this work is the empirical finding that second-order effects through OV pathways are simultaneously low-rank (≈rank-1), sparse (significant for <2% of images), and highly ablation-impactful — properties that do not hold for indirect effects due to self-repair. This coincidence of properties makes neuron interpretation tractable in a way that direct or indirect effects do not permit, and the paper exploits all three properties to build a coherent analysis pipeline. The adversarial application cleverly inverts the interpretability framework: rather than using decomposition to understand what a neuron does, it uses polysemantic decomposition to discover what a neuron does *accidentally* (spurious cross-class correlations), turning an interpretability insight into an attack vector. None beyond the paper's own contributions.

## Suggestions

- Report per-neuron PC1 variance-explained statistics (mean, median, distribution) to justify or qualify the rank-1 approximation at the individual level, addressing the most impactful factual gap.
- Temper the abstract framing from "interpreting the function of individual neurons" to "interpreting the second-order effects of individual neurons" or similar, to match the paper's actual scope and acknowledged limitations.
- Add a brief discussion of which neurons' common-word decompositions are less reliable and why (e.g., embedding geometry artifacts), to set expectations for the method's limitations alongside its strengths.

<context>
**Original reviewer signal:** Harsh Critic considers the paper promising but overclaiming, with core interpretability claims not adequately established due to incomplete lens (QK exclusion) and lack of per-neuron rank-1 fidelity validation. Strength Finder emphasizes the methodological innovation over prior methods (Table 1), the novel adversarial application (Table 3), and rigorous characterization of second-order effects (Figure 3). Disagreement centers on whether the 48.2% variance-explained gap is fatal to interpretability claims (Harsh) vs. functionally acceptable given collective validation (Strength Finder).

**What was dropped and why:**
- QK exclusion as *fatal flaw*: Downgraded to major. The paper explicitly acknowledges this as future work in Section 6, titles itself "second-order effects" (not "neuron function"), and the abstract scope is clear. It's a scope limitation with honest disclosure, not a hidden overclaim, though some abstract language ("interpret the function") slightly overpromises.
- Distribution shift confound in adversarial evaluation: Dropped because all four conditions (second-order, indirect, similar words, random) use the same text-to-image pipeline, making comparison valid.
- Small adversarial sample size/missing filtering rate: Downgraded to minor. Effect sizes are large and baselines are controlled; reporting the filtering rate would be nice but doesn't threaten validity.
- "First-order Taylor expansion" characterization: Dropped as factually wrong. The method traces signal through nonlinear attention heads, which is genuinely different from first-order expansion.
- Demand for multi-class adversarial settings: Removed as scope creep.
- Per-neuron causal validation: Moved to nice-to-have.

**Cross-checks performed:**
- Verified Eq. 5 and Section 3.2: The derivation correctly traces neuron contribution through OV matrices of subsequent attention heads, with attention weights held fixed. This is the stated scope.
- Verified Table 1: 48.2% variance confirmed. The comparison with indirect (11.0%) shows genuine improvement.
- Verified Figure 3: "Rec. from PC #1" line shows negligible accuracy drop from baseline, confirming collective functional validation.
- Verified adversarial baselines (Section 5.1): All four baselines use same pipeline; the random neuron control achieves ~1%, confirming spurious cues are method's contribution.
- Verified Section 6 limitations: QK exclusion and neuron-neuron interactions explicitly identified as future work.

**Severity read:** The surviving major weaknesses are real but not fatal: (1) the per-neuron rank-1 fidelity gap, while supported by collective validation, lacks individual-neuron evidence crucial for interpretability claims; (2) the QK pathway exclusion is honestly scoped but abstract language slightly overclaims coverage. The minor weaknesses (narrow adversarial setting, marginal segmentation gains, noisy common-word decompositions) are genuine but don't threaten core claims. No single weakness invalidates the paper's contribution.

**Anything else load-bearing:** The paper is from the same group as TextSpan (Gandelsman et al., 2024), making the segmentation comparison an incremental improvement over their own prior work — this should not be weighted as a major independent contribution. The primary novelty lies in the second-order lens itself and the adversarial application, not the segmentation results.
</context>