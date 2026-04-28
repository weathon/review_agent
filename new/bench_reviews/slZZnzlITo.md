Now I have sufficient information to write the final review. Let me compile everything based on the paper text I've read.

---

## Summary

This paper proposes Multimodal Open Set Recognition (MMOSR), extending the OSR task from single-modal to multimodal settings. The authors empirically identify "fusion degradation"—the phenomenon where naively applying OSR regularization on top of multimodal fusion degrades performance by over-compressing fused representations—and propose the Multimodal Representation Reactivation Network (MRN), which uses bidirectional cross-attention (mutually enhanced fusion) and a Mixture-of-Experts module (adaptive fusion) to obtain comprehensive multimodal representations for simultaneous closed-set classification and unknown rejection.

---

## Strengths

- **Genuine task formalization with empirical motivation.** Section 3.1 provides a clear formal definition of MMOSR. Table 1's direct comparison of Single-modal OSR / Fusion / Fusion-OSR concretely demonstrates that naively combining the two paradigms is insufficient: on Food-101/20, Fusion-OSR drops AUROC from 91.33 to 86.43 and ACC from 91.37 to 85.52 relative to plain Fusion, a nontrivial and non-obvious finding.

- **Multi-dataset, multi-modality experimental scope.** The paper spans four structurally distinct datasets (Food-101 image-text, Flower-102 image-text, CREMA-D audio-visual, SUN RGB-D RGB-depth), which is broader than typical OSR benchmark suites and supports the claim that the problem is practically motivated in robotics and sensor fusion.

- **MRN consistently improves over competing fusion baselines when combined with OSR backbones.** In Table 2, ARPL-MRN and CSRR-MRN are the best-performing methods in their respective groups across all four datasets; on Flower-102 ARPL-MRN gains +5.97 OSCR over ARPL-ADD/CAT/GQA. This indicates the MRN fusion architecture generalizes as a backbone.

- **Ablation validating cross-attention directions.** Table 4 shows that adding both C₁ and C₂ cross-attention modules progressively improves AUROC from 89.93 to 92.16 on Food-101, providing concrete evidence for the mutually enhanced fusion design.

- **Hyperparameter robustness.** Figures 4 and 5 demonstrate stable performance across expert counts E=10–20 and selected expert counts K=2–7, indicating the model does not require careful tuning.

---

## Weaknesses

### Fatal
None.

### Major

- **Diagnostic experiment uses a single OSR method to establish a universal claim.** Table 1 employs only OpenAUC to demonstrate fusion degradation. The conclusion—that "simply combining MM and OSR cannot perform satisfactorily" as a *fundamental* challenge—may be specific to OpenAUC's reciprocal-point objective rather than a universal property. Different OSR methods (ARPL, CSRR, ASH) impose structurally different constraints; the paper does not test whether all of them cause degradation when applied over a fused representation. This limits the generalizability of the core problem framing.

- **MRN fails on CREMA-D without explanation.** Table 2 shows MRN (AUROC 66.78, OSCR 57.32) loses to MLA (67.83, 57.50) on CREMA-D—the only dataset where MRN fails standalone. The paper acknowledges this loss in the gain rows (1.05↓, 0.18↓) but provides no discussion. If MRN's claimed mechanism of reactivating suppressed representations is general, this failure on audio-visual data undermines the generality claim and suggests the gains may be modality-type-specific (image-text-friendly). This is a meaningful gap for a paper positioning itself as a general MMOSR solution.

- **Ablation does not isolate the MoE contribution.** Table 4 ablates C₁ and C₂ cross-attention modules but the "base" condition (no C₁, no C₂) in the ablation *already includes* adaptive fusion (MoE). No row in Table 4 removes the MoE module. Consequently, the paper cannot attribute how much of the overall gain comes from the MoE versus the cross-attention. This is a significant gap given that the MoE is a core architectural contribution.

- **No variance across random splits.** The paper explicitly states that known classes are "randomly selected" (Section 5.1), but all results are single-run. Gains of 0.01 OSCR on SUN RGB-D and 0.18↓ on CREMA-D are not interpretable without standard deviations. This is particularly important when the paper's strongest claim rests on marginal improvements in two out of four primary datasets.

### Minor

- **The implicit argument connecting the diagnosed problem and the proposed solution is never stated.** The paper diagnoses fusion degradation as caused by OSR regularization over-compressing fused representations, then proposes a method (MRN, Equation 3) that uses only cross-entropy + load balancing—no OSR-specific regularization. The implicit claim is that richer representations survive the compressive effect of OSR losses more robustly, but this is never articulated. The paper should make this argument explicit, especially given that ARPL-MRN and CSRR-MRN show the architecture is compatible with and improves under OSR regularization.

- **Threshold τ calibration is not specified.** Section 4.3 states τ is set "to ensure 95% of the known samples are correctly classified" but does not say whether this is calibrated on a validation set, training set, or test known samples. If test known samples are used, AUROC/OSCR evaluations may be inflated.

- **The CLIP/CoOp/MaPLe comparison asymmetry is disclosed but the framing oversells it.** Table 3 clearly labels these as "zero-shot / 16-shot fine-tune" while MRN is fully supervised on all known classes. The paper's claim that MRN "outperforms pre-trained multimodal models" (Section 5.2, observation 3) is literally accurate but misleading—the result primarily demonstrates task difficulty for pretrained models, not MRN's superiority under equal conditions. This should be framed more carefully.

### Trivial

- **Possible OSCR > AUROC anomaly in Table 3.** TMC at 50/51 shows AUROC 78.87 but OSCR 84.42. Whether the paper's definition of OSCR permits this is not stated; the relationship between the two metrics should be clarified.

---

## Nice-to-Haves

- Reproduce the Table 1 diagnostic with at least ARPL and CSRR in addition to OpenAUC to establish whether fusion degradation is indeed a universal challenge or an artifact of one specific OSR loss function.
- Add a row in Table 4 that removes the MoE adaptive fusion while keeping both C₁ and C₂, so the contribution of the two modules can be independently verified.
- Report results over 3–5 random known/unknown class splits with mean ± standard deviation, especially for near-tie cases (SUN RGB-D, CREMA-D).
- Discuss the CREMA-D failure: is audio-visual cross-attention less effective? Does the audio encoder (ResNet34 on spectrogram) introduce modality imbalance that cross-attention cannot compensate?

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Cross-attention formulation (Equation 1) is standard and the claim it 'reactivates suppressed representations' is mechanistically unsupported."** Partly valid as a precision issue, but cross-attention is explicitly the *application* to this problem, not claimed as a novel formulation per se, and the ablation does demonstrate its effect. The "reactivation" language is intuitive framing, not a falsifiable mechanistic claim. Kept only as a note in Minor rather than a standalone weakness.

- **Harsh Critic: ResNet34 + Bi-LSTM encoder choice in 2024/2025.** Section 5.2 explicitly states all baselines use the same encoders ("same encoder"), making the comparison fair within the paper's scope. Criticizing the encoder choice in isolation, separate from the CLIP asymmetry (which is already kept), is a separate scope concern.

- **Harsh Critic: "MRN is a better-engineered fusion architecture evaluated under an OSR protocol — it does not modify, mitigate, or redesign OSR regularization."** Partially valid but the paper's main claim is that better fusion architecture survives OSR regularization better; this is a legitimate engineering response to the identified problem. The concern has been reframed and included under Minor.

- **Harsh Critic: The 5.23% headline is cherry-picked.** Technically valid observation, but "up to X%" phrasing is standard. Included implicitly in the Major weakness about inconsistent gains and lack of variance.

- **Any missing-related-work criticism** — removed per hard rules.

- **Strength Finder: "Task formalization with clear problem definition"** — generic strength about problem importance; absorbed into the verified strength on task formalization with empirical motivation.

---

## Novel Insights

The most genuinely novel observation synthesized from all reviews is the structural tension in the paper's logic: OSR regularization (specifically objective-function-level compactness constraints) is diagnosed as the proximate cause of fusion degradation, yet the remedy is a better fusion architecture with no corresponding change to OSR regularization. This reveals an underexplored design question: *can OSR losses be reformulated to be representation-preserving in multimodal settings?* The paper implicitly sidesteps this by showing that richer representations tolerate existing OSR objectives better, but the inverse—designing OSR losses that are compatible with multimodal diversity—remains entirely open and is arguably more principled. This is a meaningful direction the paper opens without fully exploiting.

---

## Suggestions

1. Explicitly state and defend the causal argument linking the proposed solution to the diagnosed problem: richer representations are more robust to compressive OSR regularization, as evidenced by ARPL-MRN and CSRR-MRN consistently outperforming simpler fusion baselines even under OSR regularization.
2. Add an MoE-ablation row in Table 4 (C₁+C₂ without MoE) and extend the ablation to CREMA-D and SUN RGB-D to give a complete picture.
3. Run all four primary benchmarks over 3–5 random splits and report mean ± std, particularly for SUN RGB-D and CREMA-D where gains are negligible or negative for standalone MRN.
4. Reproduce Table 1 diagnostic with at least one additional OSR method (ARPL or CSRR) to support the universal-challenge claim.
5. Address CREMA-D failure explicitly: ablate whether audio encoding quality or audio-visual cross-attention quality is the bottleneck.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Synergistic OSR (SCAD) | `i4kDKfllrz.md` | 3.50 | OSR paper, rejected for unfair baselines and missing ablations — similar pattern; this paper is more original (new task) but has comparable experimental gaps. |
| SMILE Audio-Visual Multimodal | `74IIsh2kM6.md` | 5.00 | Multimodal fusion with incomplete ablations, rejected at 5.0 — comparable structural weakness profile to this paper. |
| Energy-based OOD (Regularizing Energy) | `Lbx9zdURxe.md` | 6.00 | OOD detection with solid experiments and analysis, accepted — this paper has more novel task framing but weaker experimental rigor. |
| Neural Collapse OOD Detection | `mUXdysoxEP.md` | 6.75 | OOD detection with clear theoretical grounding and strong comprehensive experiments, accepted — significantly better experimental completeness than this paper. |

**Assessment against anchors:** The paper's closest structural match is SMILE (5.0) — a multimodal fusion paper that introduces a useful combination but lacks critical ablations and has incomplete analysis. SCAD (3.5) was rejected for unfair comparisons and weak ablation; this paper is meaningfully better due to task novelty and multi-dataset scope, so 3.5 is too low. Papers scoring 6+ (Lbx9zdURxe, mUXdysoxEP) exhibit much cleaner mechanism-to-experiment alignment and complete ablations. The incompleteness of MoE ablation, single-method diagnostic for a universal claim, unexplained CREMA-D failure, and missing variance reporting collectively position this paper below the acceptance threshold in this calibration cluster.

**Final Score: 4.5 — Borderline Reject.** The task proposal and multi-dataset empirical study are genuine contributions, but the experimental evidence does not yet support the strength of the claims made.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>