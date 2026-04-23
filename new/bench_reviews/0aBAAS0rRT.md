## Summary

SigMap proposes a multimodal foundation model for cross-scenario wireless localization that combines cycle-adaptive masked pre-training (adapting mask patterns to CSI periodicity to prevent shortcut learning) with a map-as-prompt framework (encoding 3D geographic information via a GCN into soft prompts for parameter-efficient fine-tuning). The model achieves substantial improvements over baselines on DeepMIMO, with the most notable result being a 34.4% MAE reduction over LWLM in single-BS NLoS localization and strong few-shot transfer to unseen environments (DeepMIMO O2 and WAIR-D).

## Strengths

- **Substantial performance improvements with map integration**: SigMap with map achieves 1.564m MAE in single-BS NLoS (Table 1), a 34.4% reduction over LWLM (2.382m), and more than doubles CDF@1m (60.5% vs. 25.3%). In multi-BS (Table 2), 0.673m MAE with 84.5% CDF@1m. These are meaningful improvements on a challenging task.

- **Map-as-prompt is a well-motivated and effective mechanism**: The geographic prompt (Algorithm 1, 2-layer GCN over Delaunay graph) improves single-BS MAE from 2.275m to 1.564m (Table 4, 31.2% reduction) while adding only 0.085M trainable parameters (Table 5). The ablation from 3D to 2D maps (only 8% MAE degradation, Table 4) demonstrates robustness and suggests practical deployment paths.

- **Strong few-shot transfer to unseen environments**: On DeepMIMO O2 and WAIR-D (100 real-world city scenes), SigMap with map outperforms LWLM by 53.2% and 44.3% in MAE respectively (Section 4.5), while updating only ~0.7% of parameters. These are the most compelling results in the paper, demonstrating that learned representations transfer effectively.

- **Cycle-adaptive masking is empirically validated**: Table 3 shows adaptive masking (0.673m MAE, 84.5% CDF@1m) outperforms grid masking (0.770m) and strip masking (0.753m), confirming the value of disrupting periodic shortcuts.

## Weaknesses

### Fatal

None.

### Major

- **Misleading "zero-shot generalization" claim contradicted by the paper's own experiments**: The abstract states the model exhibits "strong zero-shot generalization in unseen environments," and Section 1.2 lists "Parameter-Efficient Generalization" with "strong zero-shot generalization to unseen environments." However, Section 4.5 explicitly describes using "approximately 100 instances per scenario" to fine-tune task heads for the generalization experiments, and even labels the setup as "few-shot learning." Fine-tuning on labeled target data is few-shot transfer, not zero-shot. This is not a minor terminological slip—zero-shot generalization is a central headline claim that the paper's own experimental design contradicts. The claim should be corrected to "few-shot" throughout.

- **NLoS-aware attention mechanism (Eq. 11) absent from methodology**: Section 4.2 introduces an "NLoS-aware attention mechanism" with learned weights $W_{\text{NLoS}}$ (Eq. 11) and describes it as "the key advantage" behind single-BS performance. This mechanism does not appear in the Methodology section (Sections 3.1–3.5), Algorithm 1, or any formal model specification. If it is part of the evaluated model, the methodology is incomplete and the system is not reproducible as described. If it is not part of the model, then the results discussion attributes performance to a mechanism that doesn't exist in the system. Either way, this is a structural gap between what is claimed to work and what is actually specified.

- **Main headline results are on the same distribution used for self-supervised pre-training**: Section 4.1 states that DeepMIMO O1 3p5 is used "for both pre-training and fine-tuning." Tables 1–4 (the primary results) are therefore on the same data distribution the SSL backbone has already seen. While pre-train-then-fine-tune on the same distribution is standard SSL protocol, it undermines the paper's framing as a "foundation model" with "cross-scenario" capability. The genuine generalization test is Section 4.5 (few-shot on unseen scenarios), but those results are presented secondarily. The main results say little about whether the learned representations actually transfer, and the "foundation model" framing is overclaimed without held-out-scenario main evaluations.

### Minor

- **Missing directly relevant SSL baselines**: The introduction discusses LWM (Alikhani et al., 2024), CrowdBERT (Han et al., 2024), and signal-guided masked autoencoders (Wang et al., 2025) as directly comparable SSL-based approaches for wireless representation learning and localization, yet none appear in the experimental comparisons. The only SSL baselines are SWiT and LWLM. Including at least one more directly comparable masked modeling baseline would strengthen the SOTA claims.

- **Inconsistent parameter efficiency figures**: Section 4.5 states "0.4% of parameters" are updated during fine-tuning, while Section 4.6 states "0.7% of the total parameters are activated." Table 5 reports 0.085M fine-tuning parameters against a total of ~11.8M, which yields ~0.72%, matching the 0.7% figure. The 0.4% figure in Section 4.5 appears to be incorrect.

- **Inconsistency in WAIR-D MAE value**: Section 4.5 text states SIGMAP reaches "1.580 m on WAIR-D Scenario-2," but the corresponding table reports 1.880 m. This 0.3m discrepancy should be corrected.

- **Per-station MLP heads scale with base station count**: Equation 10 uses per-station MLP heads $W_{\text{multi}}^{(t)}$, meaning task-specific parameters scale linearly with the number of base stations. In scenarios with many base stations, this partially undermines the "parameter-efficient" framing, though the backbone remains frozen.

### Trivial

- The WAIR-D result text (1.580m) vs. table (1.880m) discrepancy may be a simple typo.

## Nice-to-Haves

- True zero-shot evaluation on unseen scenarios (no target-sample fine-tuning at all) would substantiate the original claim and distinguish the backbone's transfer quality from the task head's adaptation.
- Probing experiments to verify that cycle-adaptive masking actually prevents periodic shortcut learning (rather than just improving end-to-end performance) would strengthen the causal mechanism behind the improvement.
- Attention map visualizations comparing prompt-present vs. prompt-absent conditions would validate the "interpretable fusion" claim.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Claim that existing methods "fail to capture rich spatial-topological relationships" is unsupported (Section 1.1)**: This is a standard research-gap framing. The paper cites specific limitations of prior methods in the surrounding text. Removed as overly demanding of gap-justification evidence.

- **Cycle-adaptive masking detection algorithm not fully specified**: The paper states "we compute shift patterns using cross-correlation analysis" (Section 3.3) and the formula depends on $d_{\text{final}}$. While the exact cross-correlation computation details are not spelled out, this level of detail is typically deferred to implementation or appendix. Removed as a reproducibility nitpick per the soft rules.

- **Delaunay graph could be large for dense urban scenes; 2 GCN layers limit receptive field**: This is speculative about a failure mode not demonstrated empirically. The method works well in the reported experiments. Removed as a generic concern not grounded in observed failures.

- **200-epoch pre-training on 6×A800 GPUs (36 hours) is substantial cost**: This is standard for foundation model pre-training and not unusually large. Removed as a generic computational cost concern.

- **RMSE vs. MAE inconsistency in Table 3 (strip masking has lowest RMSE but worst CDF@1m)**: This reflects different error distributions across masking strategies, which is expected when MAE and RMSE weight tail errors differently. While interesting, this does not undermine the paper's conclusions and is not analyzed in a misleading way. Removed as minor observation not rising to the level of a weakness.

- **2D map only 8% degradation questions whether 3D is necessary**: The paper itself discusses this finding and notes it "suggests an immediate upgrade path" for practical deployment. This is an interesting result, not a weakness—it shows the prompt mechanism is robust to geometric simplification. Removed as mischaracterized strength.

- **Fine-tuning GNN on ~100 samples raises overfitting concerns**: The paper shows the model works well in this regime. While overfitting is always a concern with small data, there's no evidence it's a problem here (the model generalizes to 100 diverse city scenes). Removed as speculative concern not supported by evidence.

## Novel Insights

The most insightful observation emerging from the review is the tension between the paper's two most compelling contributions: the cycle-adaptive masking (which is a purely signal-driven innovation) and the map-as-prompt mechanism (which is a purely geometry-driven innovation). The ablation on 2D vs. 3D maps revealing only 8% degradation suggests that the bulk of the performance gain comes from topological/LoS information rather than detailed 3D geometry. This raises a subtle question: does the cycle-adaptive masking mainly help the backbone learn LoS-related features that the map prompt then exploits, or does it independently learn multipath representations? Disentangling these two pathways—through probing experiments or by testing each component in isolation across LoS vs. NLoS regimes—would clarify the distinct value of each contribution.

## Suggestions

- Correct all instances of "zero-shot generalization" to "few-shot generalization" or "few-shot transfer" in the abstract, contributions, and throughout the paper. This is the single most important revision.
- Move the NLoS-aware attention mechanism (Eq. 11) into the Methodology section (Section 3.5, alongside the task-specific adaptation heads) with full specification, or clarify whether it is part of the evaluated model or an interpretive analysis. Without this, readers cannot understand or reproduce the single-BS results.
- Consider adding one more held-out-scenario main evaluation (e.g., pre-train on O1, evaluate primary metrics on O2 or WAIR-D) to properly substantiate the "foundation model" and "cross-scenario" framing.
- Fix the WAIR-D MAE inconsistency (1.580 in text vs. 1.880 in table) and the 0.4% vs. 0.7% parameter efficiency inconsistency.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| NavFoM (kkBOIsrCXh) | 8.0 | Accept (Poster) | Strong foundation model paper with no methodology gaps. SigMap is well below this — has misleading claims and missing method components. |
| X-VLA (kt51kZH4aG) | 7.33 | Accept (Poster) | Soft-prompt cross-embodiment transfer, clean methodology. SigMap has comparable innovation but much weaker execution. |
| Aurora (VVJ6Ck9JBl) | 6.0 | Accept (Poster) | Similar zero-shot/few-shot confusion, but Aurora has 3 benchmarks, 20 datasets, and far more thorough evaluation. SigMap is below this. |
| AEMP (3AbyfpQgR2) | 5.33 | Withdrawn | Directly comparable wireless localization + masked pre-training paper. SigMap has more innovations and stronger results but also more serious issues (false zero-shot claim, missing NLoS mechanism). |
| AWDPO (KJkC2pwSXy) | 4.5 | Reject | Zero-shot claim contradicted by few-shot fine-tuning, very similar to SigMap's core issue. SigMap has stronger empirical results but also the NLoS mechanism gap. |
| GeoDiffusion (w7xpNeFIbb) | 4.0 | Reject | Key mechanism in results but missing from methodology, directly analogous to SigMap's Eq. 11 problem. SigMap is comparable. |
| SNOV (hdcZGX3saZ) | 2.0 | Reject | Genuinely poor paper with multiple methodological gaps. SigMap is clearly above this. |

SigMap sits in the 4.0–4.5 range. The false zero-shot claim and the NLoS attention mechanism gap are significant issues that undermine the paper's core framing. However, the map-as-prompt mechanism and cycle-adaptive masking are genuine contributions with strong empirical support, and the few-shot generalization results are impressive even when properly labeled. The paper is comparable to GeoDiffusion (4.0) and AWDPO (4.5), both rejected for similar severity issues, but slightly above GeoDiffusion because SigMap's methodology is mostly complete (only one component missing) and its empirical results are stronger.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>