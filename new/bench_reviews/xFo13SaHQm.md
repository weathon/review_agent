## Summary

The paper addresses the copy-paste artifact in identity-consistent image generation—where models replicate reference images rather than preserving identity across natural variations. It contributes (1) MultiID-2M, a large-scale paired multi-identity dataset of 500k labeled group photos with ~1M reference images across ~3k identities; (2) MultiID-Bench, a benchmark with a novel copy-paste metric M_CP and Sim(GT) as primary identity measure; and (3) WithAnyone, a FLUX-based model combining GT-aligned ID loss, ID contrastive loss with extended negatives, and a four-phase paired training pipeline to break the fidelity–copy-paste trade-off.

## Strengths

- **Copy-paste problem formalization and M_CP metric (Eq. 2)**: The identification and principled quantification of copy-paste as a distinct failure mode is a valuable conceptual contribution. M_CP captures the relative bias of a generated image toward the reference versus ground truth, normalized by their angular distance. Fig. 5 validates the metric by revealing a clear trade-off curve across 14 baselines that WithAnyone breaks—this is the single most compelling empirical result in the paper.

- **MultiID-2M is a substantial data contribution**: 500k labeled group photos with per-identity reference banks of ~400 images each fills a genuine gap. The ablation (Table 3, "FFHQ only" row) confirms the dataset's importance: Sim(GT) drops from 0.405 to 0.224 without it. The open-source release makes this particularly valuable for the community.

- **Paired training (Phase 3) is a simple and effective mechanism**: Replacing 50% of training samples with paired instances—where reference and target are different images of the same identity—directly targets the reconstruction shortcut. Table 3 shows Phase 3 reduces CP from 0.239 to 0.161 with virtually no cost to Sim(GT) (0.406 → 0.405). This is a clean, intuitive, and well-validated contribution.

- **GT-aligned ID loss (Eq. 4)**: Using GT landmarks for ArcFace extraction from generated images avoids noisy landmark detection and enables ID supervision across all noise levels with negligible overhead. Fig. 7 demonstrates the mechanism (lower denoising error at low noise, higher-variance gradients at high noise), and Table 3 shows removing it drops Sim(GT) from 0.405 to 0.385.

- **Comprehensive baseline evaluation**: Tables 1 and 2 evaluate 14 methods spanning general customization models and face-specific methods on both single- and multi-person subsets. Fig. 5 provides the most thorough landscape of the fidelity–copy-paste trade-off in the field to date.

- **Open-sourced project**: Full release of dataset, benchmark, and model enhances reproducibility and community value, especially given the scarcity of large-scale paired multi-ID datasets.

## Weaknesses

### Fatal
None.

### Major

- **Ablation contradicts the narrative around the contrastive loss's role in copy-paste mitigation**: Table 3 shows that removing extended negatives (reducing pool from 4096 to 63) *decreases* CP from 0.161 to 0.074—meaning the contrastive loss with extended negatives actually *increases* copy-paste artifacts while improving Sim(GT) from 0.368 to 0.405. The real copy-paste reduction comes from Phase 3 paired training (CP drops from 0.239 to 0.161). Yet the paper's framing explicitly links the contrastive loss to copy-paste mitigation (e.g., abstract: "a contrastive identity loss that leverages paired data to balance fidelity with diversity"; Section 5 intro: "This rich, identity-labeled supervision not only substantially improves identity fidelity but also suppresses trivial copy–paste artifacts"). The contrastive loss is a genuine contribution to identity fidelity, but the paper should honestly acknowledge the trade-off rather than implying all components work toward both goals. Section 6.3's ablation discussion only mentions "the effectiveness of ID contrastive loss is greatly reduced" without noting the CP increase, leaving the reader with a misleading impression.

- **Overclaimed "highest face similarity with regard to GT"**: Section 6.1 states WithAnyone achieves "the highest face similarity with regard to GT," but Table 1 shows InstantID has Sim(GT)=0.464 versus WithAnyone's 0.460 on the single-person subset. WithAnyone's genuine and important advantage is its dramatically lower CP (0.144 vs. 0.337) with comparable Sim(GT—this should be the honest framing rather than claiming it leads on both metrics.

### Minor

- **Ablation conducted in a different setting than main results**: The ablation (Table 3) achieves Sim(GT)=0.405 for the "Full Setting," while the main results (Table 1) report Sim(GT)=0.460. This disconnect (likely due to different model configurations or training runs) makes it difficult to generalize the ablation findings to the reported model—particularly the CP/Sim(GT) trade-offs.

- **Sim(GT) conflates identity preservation with pose/expression matching**: The GT row in Table 1 shows Sim(Ref)=0.521 for the ground truth, confirming ArcFace assigns the same person in a different photo only ~0.52 cosine similarity. This means Sim(GT) does not purely measure identity—it measures similarity to a specific instantiation of that identity with specific pose/expression/lighting. The paper's insight that Sim(GT) is better than Sim(Ref) for evaluating copy-paste is sound, but the limitation should be acknowledged and the metric's achievable range (~0.52, not 1.0) should contextualize reported scores.

- **User study is underpowered (N=10)**: Ten participants ranking 230 image groups provides limited statistical power. The paper claims statistical analysis is in Appendix H, but no confidence intervals or significance tests appear in the main text. Given that the quantitative Sim(GT) improvements over some baselines are modest (0.460 vs. 0.458 for UMO), the perceptual claims need stronger support.

### Trivial
None notable beyond what is already captured above.

## Nice-to-Haves

- Ablation of the extended negatives pool size between 63 and 4096 to understand the scaling behavior of the identity-fidelity vs. copy-paste trade-off.
- A simple baseline using paired training (Phase 3) alone, without the contrastive loss or GT-aligned loss, to isolate the contribution of paired data from the loss innovations.
- Analysis of how many test cases are excluded by the Sim(GT) threshold filtering (>0.40 for single-person, >0.35 for multi-person) and whether excluded cases share common characteristics.
- Normalization of Sim(GT) scores as a fraction of the achievable range (using the GT-Ref similarity ceiling of ~0.52) to give readers better calibrated expectations.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **Ethics/licensing concerns about CC verification**: Questioning whether CC-licensed celebrity photos are genuinely CC-licensed is a knowledge-gap concern, not a paper error. Per rules, if the paper cites it, it exists; the ethics statement describes a reasonable process.

- **Anonymization contradiction with celebrity name queries**: The critic notes the dataset uses celebrity name queries during collection but claims "no personal names" during training. These are different stages—collection vs. training—and not contradictory.

- **Missing hyperparameter ablation**: Demanding ablation of Phase step counts (20k, 40k) or the 50% paired sample ratio falls under reproducibility nitpicks per rules.

- **Missing appendix content**: Criticisms about absent statistical analysis in Appendix H or missing ablation details in Appendix G are removed per rules—the parser strips appendices from all papers.

- **Missing related works**: Per rules, cannot confirm existence of uncited works.

- **Formatting/style nitpicks**: Removed per rules.

- **CP metric normalization instability**: While mathematically valid, this is a minor design observation that doesn't threaten the benchmark's validity—especially since the threshold filtering already addresses edge cases.

- **Strength finder's claim that "WithAnyone achieves the highest Sim(GT) among all 14 methods (0.460)"**: This is factually incorrect per Table 1 (InstantID=0.464) and conflicts with a verified Major weakness, so it is removed from Strengths.

## Novel Insights

The ablation's revelation that the contrastive loss with extended negatives *increases* copy-paste while improving identity fidelity reveals an inherent tension in ID-consistent generation: stronger identity discrimination signals pull generation toward the reference, worsening copy-paste. This suggests that the fidelity–copy-paste trade-off exists not just across models (as Fig. 5 shows) but also *within* a single model's components—copy-paste mitigation and identity reinforcement can be at odds, and the paired training strategy is the key mechanism that resolves this tension rather than the contrastive loss.

## Suggestions

- Honestly reframe the contributions: the contrastive loss with extended negatives is a contribution to *identity fidelity*, not to copy-paste mitigation. The copy-paste reduction is primarily achieved by Phase 3 paired training. Acknowledging this trade-off would strengthen rather than weaken the paper, as it provides deeper insight into when and why copy-paste arises.

- Correct the "highest face similarity with regard to GT" claim in Section 6.1 to accurately reflect that WithAnyone achieves *comparable* Sim(GT) to the best methods (0.460 vs. InstantID's 0.464) while achieving dramatically lower CP—the trade-off breaking is the genuine and impressive contribution.

- Consider reporting Sim(GT) as a fraction of the achievable range (e.g., 0.460/0.521 = 0.883 of the GT-Ref ceiling) to give readers calibrated context for what these scores mean.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| HyperHuman (large-scale dataset + method, poster) | 7.50 | Stronger methodology, no narrative issues; WithAnyone has weaker methodological honesty |
| MGFR (face ID preservation + dataset, spotlight) | 7.33 | Cleaner narrative, stronger per-component validation; WithAnyone has ablation contradictions |
| SemVarBench (novel metric + benchmark, poster) | 6.00 | Similar metric+benchmark pattern; WithAnyone adds model contribution but has narrative issues |
| DreamBench++ (benchmark + dataset + metric, poster) | 6.00 | Similar pattern; WithAnyone has stronger dataset scale but weaker claims accuracy |
| MoLE (~1M dataset + LoRA MoE, reject) | 5.25 | Weaker methodology and evaluation; WithAnyone is clearly stronger |
| KAN See Your Face (ablation contradicts narrative, reject) | 3.00 | Similar ablation-narrature mismatch but much weaker overall; WithAnyone has real contributions and results |

WithAnyone is clearly above the reject-line papers (KAN, MoLE) and above borderline papers with only benchmark contributions (SemVarBench, DreamBench++) due to its combined dataset+benchmark+model contribution. However, it falls below the strong accept papers (MGFR, HyperHuman) due to the narrative misalignment between ablation findings and claims, and the overclaimed "highest Sim(GT)" result. The sweet spot is around 6.0—solid contributions with honest issues that need addressing but don't invalidate the work.

## Score and Decision

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>