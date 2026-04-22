Now I have all the information needed to write the final review. Let me synthesize everything.

## Summary

This paper addresses the "copy-paste artifact" in identity-consistent image generation, where models over-replicate reference faces rather than preserving identity across natural variations. The authors make three contributions: (1) MultiID-2M, a large-scale paired multi-identity dataset with ~500k labeled images; (2) MultiID-Bench, a benchmark introducing a Copy-Paste (CP) metric that measures relative angular bias toward the reference vs. ground truth; and (3) WithAnyone, a FLUX-based model using paired training, a GT-aligned ID loss, and an ID contrastive loss with extended negatives to mitigate copy-paste while maintaining high identity similarity.

## Strengths

- **Formalization and metric for the copy-paste artifact.** The paper identifies a concrete failure mode that prior work largely overlooked and provides a principled metric for it (Eq. 2). Figure 2's 3D density plot powerfully shows that models like InstantID produce face-similarity distributions sharply peaked near 1.0, while real photo pairs of the same person have much broader distributions (scores of 0.77, 0.46, 0.46, 0.30 for the same person under natural variation).

- **Breaking the fidelity–copy-paste trade-off.** Figure 5 is the paper's most striking evidence—all other evaluated methods lie approximately along a regression curve between Sim(GT) and Copy-Paste, while WithAnyone is positioned clearly in the upper-right region (high similarity, low copy-paste). This demonstrates the combination transcends what prior methods achieve rather than merely trading one metric for another.

- **Paired training paradigm that directly targets copy-paste.** The Phase 3 paired tuning strategy—using a different image of the same identity as reference versus target—directly breaks the reconstruction shortcut. Table 3 ablation confirms this: removing Phase 3 increases CP from 0.161 to 0.239 while Sim(GT) remains essentially unchanged (0.406 vs. 0.405), proving the CP reduction comes at no cost to ground-truth identity similarity.

- **GT-aligned ID loss enabling supervision across all noise levels.** Using ground-truth landmarks rather than noisy predicted landmarks for ArcFace embedding extraction is a simple but effective idea. Figure 7 shows this yields consistently lower ID loss across all noise levels (0.2–0.8) and higher-variance, more informative gradients at high noise. Table 3 confirms the practical impact: removing GT-Align drops Sim(GT) from 0.405 to 0.385 and worsens CP from 0.161 to 0.175.

- **Sim(GT) as a more faithful identity evaluation metric.** The insight that Sim(Ref) inadvertently rewards copy-paste is well-argued. Table 1 shows the GT row has Sim(Ref) = 0.521, yet InstantID achieves 0.734—exceeding what natural variation produces—while having lower Sim(GT) (0.464 vs. WithAnyone's 0.460).

- **Comprehensive baseline comparison spanning 14+ methods** on both single- and multi-person subsets (Tables 1–2), providing a broad and useful characterization of the current landscape.

- **Fully open-sourced project** including dataset, benchmark, and model.

## Weaknesses

### Fatal
None.

### Major

- **CP metric's per-method filtering makes cross-method comparisons potentially unfair.** Tables 1–2 state that "for Copy-Paste ranking, only cases with Sim(GT) > 0.40 [or 0.35] are considered." This threshold is applied per-case, meaning different methods are evaluated on *different subsets* of test cases. A method with poor identity preservation that barely crosses the threshold on a few easy cases may have its CP computed on a highly selective (and possibly easier) subset, while a stronger method is evaluated on a broader distribution. The paper does not report how many cases pass the filter for each method, nor does it compute CP on a common subset. This directly affects whether the headline CP comparison is fair. While the Figure 5 scatter plot provides a more holistic view that mitigates this concern, the per-method CP rankings in Tables 1–2 are the primary quantitative claims and they rely on potentially non-comparable subsets. Reporting per-method case counts through the filter and/or a common-subset analysis would substantially strengthen the evaluation.

- **The ID contrastive loss with extended negatives increases CP, which the paper's narrative does not acknowledge.** Table 3 shows that removing extended negatives (w/o Ext. Neg.) reduces CP from 0.161 to 0.074—a 55% reduction—while also reducing Sim(GT) from 0.405 to 0.368. This means the contrastive loss with extended negatives *increases* copy-paste tendency even as it improves identity fidelity. Yet the abstract states the contrastive loss "leverages paired data to balance fidelity with diversity," and Section 5's preamble claims the training strategies "suppress trivial copy-paste artifacts." The actual mechanism is: paired training (Phase 3) reduces CP; the contrastive loss improves identity fidelity at the cost of somewhat increased CP. The paper does not explicitly acknowledge this trade-off *within* the method itself. While the ablation table transparently shows the numbers, the textual narrative gives the misleading impression that all components conspire to reduce copy-paste. This should be corrected to give an honest account of what each component contributes.

### Minor

- **The "FFHQ only" ablation conflates dataset quality with the paired-training paradigm.** Replacing MultiID-2M with FFHQ removes both the dataset's diversity/quality *and* the paired training signal simultaneously. It cannot distinguish whether improvements come from having paired data at all (which could potentially come from simpler sources) or from MultiID-2M's specific scale and diversity. However, this is partially mitigated by the separate "w/o Phase 3" ablation that isolates the paired training effect on the same dataset, so the conflation is not complete.

- **The Eq. 5 loss is labeled "InfoNCE" but omits the positive term from the denominator.** Standard InfoNCE includes exp(cos(g,t)/τ) in both numerator and denominator. The paper's formulation omits it from the denominator, making it closer to a margin-based objective. With M=4096 negatives the practical difference is small, but calling it "follows the InfoNCE (Oord et al., 2018) formulation" is technically incorrect. The deviation should be acknowledged or justified.

- **User study figure (Fig. 8) uses different method names** ("Cure" instead of "WithAnyone," "iDetch" instead of "InstantID," "Uniformal" instead of "UniPortrait") that don't match the rest of the paper, creating potential confusion about whether the evaluation setup differs.

### Trivial
None.

## Nice-to-Haves

- Ablating paired training on a simpler paired dataset (e.g., CelebA with identity grouping) to isolate the specific contribution of MultiID-2M beyond just having paired data.
- Analysis of the CP metric's sensitivity to θ_{tr} magnitude, particularly when reference and GT are very close or very far in embedding space.
- Explicit discussion of the contrastive-loss/CP trade-off, including analysis of when the net effect is positive.
- Reporting per-method case counts passing the Sim(GT) filter in Tables 1–2.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **GPT-4o prior knowledge flagging.** The harsh critic suggests GPT-4o results should be excluded due to prior knowledge of identities. The paper already clearly flags this issue in the Table 2 caption ("GPT exhibits prior knowledge of identities from TV series in subsets with more than two IDs, leading to abnormally high similarity scores"). Including it with an explicit caveat is standard practice; exclusion would reduce transparency. The flag is visible in the table caption.

- **Missing hyperparameters/training details.** The critic flags missing learning rates, batch sizes, optimizer, etc. These are standardly deferred to appendices in this venue. The paper mentions key details (20k/40k steps, λ values of 0.1). Per the rules, this is a nitpick about trivial implementation details impractical to include in the main text.

- **Identity matching threshold of 0.4 concern.** The critic questions whether cosine similarity 0.4 for ArcFace is too low, potentially creating false-positive identity matches. While a valid concern about data quality, this is speculative—the paper could have quality analysis in the appendix (which we cannot see). The threshold is clearly stated.

- **ε value unspecified.** The paper mentions ε is "a small constant for numerical stability" without giving its value. This is a very minor detail that doesn't affect the metric's interpretation.

- **Dummy prompt in Phase 1 not ablated.** This is a nice-to-have ablation, not a weakness. The paper provides sufficient ablations on the main components.

- **User study sample size (10 participants).** While 10 participants is modest, user studies in this area are often similarly sized. The paper reports the study as supplementary evidence, and the quantitative metrics are the primary evaluation. Flagging lack of confidence intervals is reasonable but this is more of a nice-to-have than a substantive weakness for this type of paper.

- **Missing failure cases.** A reasonable suggestion but not a core weakness; most papers in this area present primarily successful outputs.

- **Missing related works.** Per the rules, I cannot confirm the existence of suggested missing references.

## Novel Insights

The most insightful observation across the reviews is that this paper's ablation table (Table 3) reveals an internal tension within the proposed method: the contrastive loss with extended negatives and the paired training phase push in opposite directions on the CP metric. The paired training is the primary mechanism for CP reduction, while the contrastive loss provides identity fidelity at the cost of increased CP. The net result (the full model) achieves a favorable trade-off, but this is a *balance* of competing forces rather than a *conspiracy* of complementary ones. Acknowledging this honestly would strengthen the paper's credibility and provide clearer guidance for future work on how to tune this balance.

## Suggestions

- Report the number of test cases passing the Sim(GT) filter for each method in Tables 1–2, and optionally compute CP on the common intersection of cases that pass the filter for all methods. This single addition would substantially address the major evaluation concern.
- Revise the narrative around the contrastive loss to acknowledge that it primarily strengthens identity fidelity while somewhat increasing CP, and that the overall system's low CP is achieved primarily through paired training. This honest framing would not diminish the paper's contribution.

## Score and Decision

**Calibration comparison:**

| Anchor Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| DisenBooth | /home/wg25r/review_agent/human_reviews/FlhjUkC7vH.md | 7.5 | Similar identity-preservation topic with disentanglement; this paper is more comprehensive (dataset+benchmark+method) but has evaluation methodology concerns that DisenBooth doesn't |
| RB-Modulation | /home/wg25r/review_agent/human_reviews/bnINPG5A32.md | 8.0 | Addresses content leakage in diffusion; more theoretically grounded but narrower scope than this paper |
| Bright Ending Attention | /home/wg25r/review_agent/human_reviews/p4cLtzk4oe.md | 7.33 | Novel memorization observation with practical method; comparable novelty in problem identification |
| DreamBench++ | /home/wg25r/review_agent/human_reviews/4GSOESJrk6.md | 6.0 | Benchmark for personalized generation; this paper goes further with method+dataset, but DreamBench++ had cleaner evaluation |
| RetriBooru | /home/wg25r/review_agent/human_reviews/IjVCcykKdr.md | 4.5 | Addresses same fundamental issue (reference leakage/copy-paste) in anime domain; this paper is much more comprehensive and well-executed |
| Unsupervised FL Face Recognition | /home/wg25r/review_agent/human_reviews/XH3OiIhtvf.md | 2.0 | Genuinely weak paper; this paper is clearly far above this level |
| Memorization Detection/Mitigation | /home/wg25r/review_agent/human_reviews/84n3UwkH7b.md | 8.0 | Directly addresses copy-paste replication in diffusion from memorization angle; comparable significance but cleaner methodology |

This paper sits above the medium anchors (DreamBench++ at 6.0, RetriBooru at 4.5) due to its comprehensive contributions and strong core results. It is somewhat below the high anchors (DisenBooth 7.5, Bright Ending 7.33) because those papers had cleaner evaluation methodology and more precise claims, while this paper has the CP filtering concern and the partial mischaracterization of the contrastive loss's role. The core finding (Figure 5—breaking the fidelity–copy-paste trade-off) is compelling and the paired training insight is genuine. The weaknesses are addressable but represent real gaps in the current presentation.

**Originality:** High. The copy-paste formalization and CP metric are novel and fill a gap in the literature.

**Importance of research question:** High. Identity-consistent generation is a major research direction, and the copy-paste artifact is a real practical problem.

**Claims support:** Moderate-to-good. The main claim (breaking the trade-off) is well-supported by Figure 5 and ablations, but the CP filtering issue weakens the precision of cross-method CP comparisons, and the component-level claims are partially misleading.

**Soundness of experiments:** Good overall, with the noted evaluation methodology gap.

**Clarity of writing:** Good. The paper is well-structured with clear motivation.

**Value to community:** High. The dataset, benchmark, and metric are valuable community resources.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>