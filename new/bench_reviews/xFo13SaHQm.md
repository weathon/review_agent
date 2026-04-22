Now I have all the data I need. Let me compile the final review.

## Summary

WithAnyone tackles the copy-paste artifact in identity-consistent image generation—where models replicate reference images rather than producing identity-preserving but visually distinct outputs. The paper contributes (1) MultiID-2M, a large-scale paired multi-identity dataset; (2) MultiID-Bench, a benchmark with Sim(GT) and a Copy-Paste metric (M_CP); and (3) a four-phase training pipeline centered on paired tuning and an ID contrastive loss with extended negatives, built on FLUX. Experiments across 12 baselines demonstrate that WithAnyone substantially reduces copy-paste while maintaining competitive identity similarity, effectively breaking the fidelity–copy-paste trade-off observed in prior methods.

## Strengths

- **The copy-paste problem formulation is important and previously under-formalized.** The observation (Fig. 2) that reconstruction-based training combined with Sim(Ref) evaluation creates a perverse incentive toward literal copying is clearly articulated. This is a genuine conceptual contribution that could shift how the community thinks about ID-consistent generation evaluation.

- **The paired training paradigm (Phase 3) is effective and well-motivated.** Replacing same-image reconstruction with paired (reference, target) pairs from the same identity directly addresses the root cause of copy-paste. Table 3 provides direct evidence: removing Phase 3 increases CP from 0.161 to 0.239 (~48% increase) while Sim(GT) barely changes (0.405→0.406), confirming paired training is the primary anti-copy-paste mechanism.

- **MultiID-2M fills a critical data gap.** The dataset provides ~400 paired reference images per identity across ~3k identities in group-photo contexts (§3). Table 3 ablation confirms its necessity: training on FFHQ alone collapses performance to Sim(GT) = 0.224 vs. 0.405 with the full dataset.

- **The GT-aligned ID loss is a practical, effective technique.** Using GT landmarks for ArcFace extraction on generated images (Eq. 4) avoids unreliable landmark detection from noisy diffusion outputs and enables applying the ID loss across all noise levels with negligible overhead. Fig. 7 provides concrete evidence of improved denoising error and more informative gradients.

- **Breaking the fidelity–copy-paste trade-off is convincingly demonstrated in Figure 5.** Most baselines lie along a regression curve where higher similarity correlates with stronger copy-paste; WithAnyone visibly deviates from this curve, achieving competitive Sim(GT) (0.460) with a CP score (0.144) less than half that of comparably faithful methods like InstantID (0.337) and PuLID (0.315).

- **Comprehensive baseline comparison spanning 12 methods.** Evaluation covers both general customization models (OmniGen, FLUX.1 Kontext, GPT-4o, etc.) and face-specific methods (InstantID, PuLID, UniPortrait, etc.) on both single-person and multi-person subsets.

## Weaknesses

### Fatal
None.

### Major

- **Factual error: WithAnyone does not achieve the highest Sim(GT).** The paper states "achieving the highest face similarity with regard to GT while maintaining a markedly lower copy-paste score" (§6.1), and the abstract claims "state-of-the-art identity similarity (with regard to target image)." However, Table 1 shows InstantID at Sim(GT) = 0.464 vs. WithAnyone at 0.460. The claim is numerically false. While the difference is small (0.004) and WithAnyone achieves this with far lower CP (0.144 vs. 0.337), the specific claim about "highest" Sim(GT) is incorrect and should be revised. The broader claim about breaking the trade-off is valid, but the overstatement undermines trust.

- **Misleading attribution of the anti-copy-paste mechanism.** The paper frames the ID contrastive loss with extended negatives as part of the solution to copy-paste (§5: "This rich, identity-labeled supervision not only substantially improves identity fidelity but also suppresses trivial copy–paste artifacts"). However, Table 3 shows that removing extended negatives *decreases* CP from 0.161 to 0.074—meaning the contrastive loss actually *increases* copy-paste. It does improve identity fidelity (Sim(GT): 0.368→0.405), so it serves a valuable purpose, but it does not suppress copy-paste. The anti-copy-paste effect comes almost entirely from Phase 3 (paired tuning). This misattribution matters because it misleads future work about which mechanism is responsible for mitigating copy-paste. The paper should honestly acknowledge that the contrastive loss trades off some copy-paste for identity fidelity, and credit paired training as the primary anti-copy-paste intervention.

- **M_CP metric validation is insufficient for a metric central to the paper's framing.** The Copy-Paste metric operates in ArcFace embedding space, which is designed for identity discrimination rather than measuring appearance-level copying. While the metric has a sensible geometric interpretation (Eq. 2), its validity as a proxy for perceptual copy-paste rests on a 10-participant user study (§6.3) with no reported correlation coefficient, confidence intervals, significance tests, or inter-rater agreement. The paper states only "a moderate positive correlation" without quantifying it. For a metric that underpins the benchmark and the paper's core narrative, this validation is too thin. Notably, Sim(Ref) for GT = 0.521 (Table 1) shows substantial within-identity variation in ArcFace space, meaning M_CP captures residual appearance signal that ArcFace failed to remove—useful, but not necessarily a principled measure of visual copying.

### Minor

- **Unexplained discrepancy between Table 1 and Table 3 results.** WithAnyone's Sim(GT) is 0.460 in Table 1 but 0.405 in Table 3 (Full Setting). This non-trivial gap suggests a different evaluation protocol or checkpoint is used, but no explanation is provided. Clarifying this would strengthen the ablation's credibility.

- **Sim(GT) as primary metric has an unacknowledged limitation.** Evaluating against a specific ground-truth image penalizes valid generations that don't match that particular rendering. A good model given "a woman smiling" and a neutral reference could legitimately produce many different smiles; only one matches the GT. This is not a fatal flaw—the metric is clearly better than Sim(Ref) for the stated purpose—but the limitation deserves acknowledgment.

- **The 4-phase pipeline transitions are underspecified.** Phase 1 ends when the model achieves "satisfactory identity similarity" (§5.2) with no quantitative threshold given. Phase 4 uses a "curated high-quality subset" with no selection criteria described. These choices could significantly affect results and should be clarified.

### Trivial
None.

## Nice-to-Haves

- A larger user study (≥30 participants) with reported correlation coefficient, confidence intervals, and inter-rater agreement (e.g., Krippendorff's α) to properly validate M_CP's perceptual relevance.

- An ablation that removes Phase 3 while keeping the contrastive loss, to cleanly isolate the contrastive loss's effect on copy-paste independent of paired training.

- Analysis of failure cases where M_CP disagrees with visual copy-paste judgment, which would reveal the metric's blind spots and build community trust.

- Evaluation on non-celebrity identities to test generalization beyond well-represented face distributions.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic's "ArcFace paradox" argument that M_CP is fundamentally flawed**: The theoretical argument that if ArcFace were perfect at identity invariance, M_CP would be uninformative is intellectually interesting but not practically decisive. The paper's own data shows Sim(Ref) for GT = 0.521, confirming ArcFace retains substantial within-identity variation. The metric clearly captures something useful; the issue is insufficient validation, not fundamental invalidity.

- **CC-licensing plausibility concern**: The harsh critic questions whether ~1M celebrity reference images could be obtained under CC licenses. This approaches questioning the dataset's existence/availability, which is disallowed. The paper provides a detailed ethics statement describing their collection process; we take this at face value per the rules.

- **Missing related works**: Per rules, we do not flag missing citations since we cannot verify their existence.

- **Formatting/notation issues**: Per rules, parser artifacts and formatting nitpicks are removed.

- **Reproducibility concerns about hyperparameters and training details**: Per rules, trivial implementation details and large artifacts impractical to include are not flagged.

- **Strength Finder's claim that ID contrastive loss with extended negatives helps break the fidelity-CP trade-off**: This conflicts with the verified Major weakness that the contrastive loss increases CP. The loss improves identity fidelity, which is a genuine strength, but it does not contribute to reducing copy-paste as the Strength Finder implies. Moved here to avoid contradiction.

## Novel Insights

The ablation table reveals an important and counterintuitive finding that the paper itself does not adequately highlight: the ID contrastive loss and the anti-copy-paste objective are partially antagonistic. Strengthening identity discrimination (via extended negatives) inadvertently pulls the generated image closer to the reference's appearance, increasing copy-paste. This suggests that future work on mitigating copy-paste while maintaining identity fidelity may need to explicitly decouple identity-level and appearance-level representations, rather than relying on a single embedding space for both objectives.

## Suggestions

- Correct the claim about "highest face similarity with regard to GT" to accurately reflect that WithAnyone achieves *competitive* Sim(GT) (comparable to InstantID's 0.464) with *substantially lower* CP (0.144 vs. 0.337). The genuine contribution is breaking the trade-off, not achieving the single highest Sim(GT).

- Revise §5 and the abstract to clearly attribute the anti-copy-paste effect to paired training (Phase 3), and acknowledge that the contrastive loss improves identity fidelity at the cost of increased copy-paste tendency. This honest attribution would actually strengthen the paper by surfacing the antagonism between the two objectives.

- Report the correlation coefficient, p-value, and inter-rater agreement for the M_CP user study, even if the sample size remains small.

- Explain the Table 1 vs. Table 3 discrepancy (Sim(GT) = 0.460 vs. 0.405).

## Calibration

**Anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| SHINE | /home/wg25r/review_agent/human_reviews_2026/DcVg87ibK9.md | 7.33 | Training-free image composition with artifact suppression. Cleaner execution, no factual errors, comprehensive experiments. WithAnyone is below this due to overclaiming and misleading narrative. |
| Phantom-Data | /home/wg25r/review_agent/human_reviews_2026/IjqKXnzUXx.md | 6.00 | Also tackles copy-paste via cross-pair dataset construction. Similar data-contribution profile. WithAnyone has comparable strengths but additional issues (factual error, misattribution). Roughly comparable. |
| OmniPortrait | /home/wg25r/review_agent/human_reviews_2026/DVmR3Ij0ap.md | 5.50 | Face identity customization with outdated baselines and incomplete evaluation. WithAnyone is clearly above this — better problem formulation, more comprehensive evaluation, genuine mechanism insight. |
| MOSAIC | /home/wg25r/review_agent/human_reviews_2026/7AH0y1OtnC.md | 4.50 | Multi-subject generation with missing baselines, limited evaluation. WithAnyone is above this — more baselines, clearer problem formulation, stronger empirical evidence. |
| FaceID-6M | /home/wg25r/review_agent/human_reviews_2026/yTq81RcKaw.md | 3.50 | Face ID dataset that overclaims slight gains as "comparable/slightly better." WithAnyone has more methodological contribution and more comprehensive evaluation. Clearly above. |
| TIIF-Bench | /home/wg25r/review_agent/human_reviews_2026/vm8lNKLfuo.md | 2.50 | T2I benchmark overclaiming novelty, missing critical related work. WithAnyone is clearly above — genuine contributions, less severe issues. |
| ZX6XEfBidf | /home/wg25r/review_agent/human_reviews_2026/ZX6XEfBidf.md | 2.00 | Image generation theory with incorrect theorems. Far below WithAnyone in every dimension. |

WithAnyone sits above OmniPortrait (5.5) and MOSAIC (4.5) due to its genuine problem formulation, dataset contribution, and the compelling Figure 5 trade-off result. It sits below Phantom-Data (6.0) and SHINE (7.33) due to the factual error about highest Sim(GT), the misleading attribution of the anti-copy-paste mechanism, and weak metric validation. The paper's core contributions are real and valuable, but the overclaiming and narrative misattribution are significant. A score of 6.0 reflects a paper with genuine contributions that need honest revision of claims, comparable to Phantom-Data which also tackles a similar problem with similar strengths and was accepted as poster.

## Score and Decision

**Originality**: Good — The copy-paste formalization and the paired-training insight are novel contributions. The contrastive loss and GT-aligned ID loss are incremental but effective.

**Importance of research question**: Good — Copy-paste is a real, widespread problem in ID-consistent generation that the community needs to address.

**Claims support**: Moderate — The core trade-off-breaking result (Fig. 5) is well-supported, but specific claims about "highest" Sim(GT) are false, and the attribution of anti-copy-paste effects to the contrastive loss is contradicted by the ablation.

**Soundness of experiments**: Moderate-to-Good — Comprehensive baselines and ablations, but the metric validation is weak and the Table 1/Table 3 discrepancy is unexplained.

**Clarity**: Good — Well-structured and clearly written.

**Value to community**: Good — The dataset, benchmark, and paired-training insight are valuable resources.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>