Now I have everything I need to write the final review. Let me synthesize the findings.

---

## Summary

This paper proposes PKA (Patch-wise and Keyword-Aware Attention), a framework for efficient multi-condition control in Diffusion Transformers. It introduces two specialized attention modules—Position-Aligned Attention (PAA) for spatial conditions and Keyword-Scoped Attention (KSA) for subject-driven conditions—motivated by an empirical analysis showing that attention in concatenate-and-attend systems is largely sparse and condition-type-specific. Complemented by an early-timestep training strategy, PKA claims up to 10× inference speedup and 5.12× VRAM reduction while matching or improving generation quality over published baselines.

---

## Strengths

- **Principled empirical motivation (Figures 2–3):** The paper demonstrates that spatial-condition attention matrices are diagonal-dominant, while subject-driven attention is keyword-localized. These observations directly justify the PAA and KSA designs rather than relying on generic sparsity assumptions.

- **Efficient PAA formulation (Eq. 2, Section 3.2.1):** Reducing cross-condition attention from O(N²) to O(N) via one-to-one positional correspondence is theoretically clean and structurally justified by the observed redundancy pattern.

- **Condition Cache mechanism (Section 3.2, Figure 4a):** By constraining condition tokens to self-attend within their own modality, KV projections for all conditions can be computed once and reused across all denoising steps. This is a low-overhead but practically significant optimization.

- **Strong quantitative gains on most metrics (Table 1, Figures 7–8):** PKA achieves the best FID (52.99, 62.08, 53.01) and SSIM across all three tasks, with 3.90×–10× latency improvement and 2.46×–5.12× VRAM reduction. Even with the caveats noted below, the efficiency gains over naive full-attention systems are plausible and well-documented.

- **Generalizes across condition-type combinations (Table 1):** Evaluation on Subject-Canny, Subject-Depth, and Canny-Depth demonstrates the framework handles diverse multi-condition task configurations.

---

## Weaknesses

### Fatal
None.

### Major

- **Uncontrolled training data alignment in Table 1 — potentially inflates quality results.** Section 4.1 states that PKA fine-tunes FLUX.1 with LoRA for 20,000 iterations on a curated subset of Subject200K, and the test set is derived from "a partition" of the same dataset. The paper does not state whether OminiControl2 and UniCombine are re-trained under the same protocol, or evaluated using their published checkpoints trained on their respective data pipelines. If the baselines use their published checkpoints while PKA is trained specifically on Subject200K-sourced data that shares the test distribution, the FID improvements of ~10–19 points across all tasks may reflect data alignment rather than architectural superiority. This is the most important unresolved issue: without controlling for training data, the quality comparison in Table 1 cannot be cleanly interpreted. The authors should clarify the training protocol used for each baseline and ideally retrain them on the same data subset under the same LoRA regime.

- **Mischaracterization of the Subject-Canny F1 result.** Section 4.2.3 calls the F1 gap on Subject-Canny a "narrow margin," but the actual numbers—PKA: 0.414 vs. UniCombine: 0.551—represent a 25% relative regression in edge controllability, the largest directional gap in the entire table. This is not a narrow margin. For a method that centralizes controllability as a design goal, an honest accounting of this gap is necessary. The paper should explain whether this drop is attributable to PAA, KSA, the cache design, or training, and whether it is acceptable given the quality gains.

### Minor

- **Condition Cache severs condition-to-image attention with no ablation (Section 3.2).** Restricting conditions to self-attend only means conditions cannot adapt to the evolving noisy image state across the denoising trajectory. This is a potentially consequential design choice, but no ablation isolates its effect from the PAA/KSA sparsity benefits. An ablation comparing "conditions attending to image" vs. "conditions self-only" would clarify how much quality is sacrificed by the cache constraint.

- **Test set size not reported (Section 4.1).** FID scores are sensitive to sample count, and "a partition of the curated Subject200K subset" provides no basis for interpreting or comparing FID values. The exact test set size should be stated.

- **KSA temporal consistency assumption not validated (Section 3.2.2).** The mask M^t computed at timestep t is reused at t+1. While motivated by Zhou et al.'s temporal consistency findings, the paper does not characterize how frequently this approximation degrades (e.g., in complex or multi-subject scenes) or what the quality impact is when it does. The ablation in Figure 10 uses a single qualitative example.

- **Early-timestep sampling ablation is qualitative only (Section 3.3, Figure 11).** The perturbation analysis in Figure 5 is interesting, but the ablation of µ is limited to three qualitative examples with no FID, SSIM, or controllability metrics as a function of µ. Quantitative support is needed to establish that the benefit is consistent and meaningful.

### Trivial

- The PAA efficiency discussion notes reduction from O(N²) to O(N) for the PAA sub-block but does not quantify what fraction of total compute this sub-block represents relative to the remaining full self-attention over text and image tokens.

---

## Nice-to-Haves

- **Multi-condition scaling experiment (4+ conditions).** The efficiency claim peaks at high condition counts (Figure 7 reports 10× speedup at scale), but all quantitative experiments use exactly two conditions. An experiment with 4–6 conditions would directly validate the asymptotic efficiency claims, which are the most commercially significant aspect of the work.

- **Failure case analysis.** PAA assumes positional alignment (invalid if the spatial condition is not co-registered) and KSA assumes identifiable keywords. Showing representative failure modes would help scope the method's applicability.

- **Quantitative µ/δ sensitivity analysis for early-timestep sampling.** A table showing quality and controllability metrics across a range of µ values would strengthen the training strategy claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic, Issue 3 (full formulation):** The observation that conditions cannot attend to image tokens is real and kept as a Minor weakness, but the critic's framing—that this "invalidates" quality comparisons—is too strong. The design choice is a deliberate structural decision enabling the Condition Cache, not an uncontrolled confounder.

- **Harsh Critic, Section 3.2.1 (O(N) fraction of total compute):** Moved to Trivial; this is a presentation note rather than a substantive flaw.

- **Strength Finder, "Graceful and tunable trade-off in KSA" (Figure 10):** This strength is not dropped outright but is weakened: the supporting evidence is one qualitative example, which is insufficient to call KSA "robust to its hyperparameter."

---

## Novel Insights

The most technically novel observation in the paper is that attention sparsity in multi-condition DiTs is *condition-type-specific* rather than generic: spatial conditions produce diagonal-dominant attention (amenable to positional alignment), while subject conditions produce semantically localized attention (amenable to keyword masking). This motivates designing *separate, structurally distinct* attention modules for each condition type, rather than applying a single pruning or approximation scheme uniformly. The Condition Cache, enabled by forcing conditions into self-attention-only groups, is a clean structural consequence of this decomposition and could generalize to any system where condition tokens do not require dynamic interaction with the image state during denoising.

---

## Suggestions

1. **Clarify baseline training setup explicitly** — add a table or footnote specifying whether each baseline is (a) the published checkpoint, (b) re-trained from scratch on Subject200K, or (c) fine-tuned from the published checkpoint on Subject200K. If they are published checkpoints, re-run at least one baseline with identical LoRA training on the same data for an apples-to-apples comparison.
2. **Correct the characterization of the Subject-Canny F1 result** in Section 4.2.3; call it "a meaningful controllability trade-off" and analyze its cause.
3. **Add an ablation** with condition tokens allowed to attend to image tokens (no cache) but with PAA/KSA sparsity retained, to isolate the cache's quality cost.
4. **Report test set size** alongside all quantitative results.
5. **Include a multi-condition scaling experiment** (≥4 conditions) to validate the headlining 10× speedup claim with direct experimental evidence.

---

## Calibration

| Paper | Path | Avg Score | Relation to PKA |
|---|---|---|---|
| PT-T2I/V (sparse token attention for DiTs) | lTrrnNdkOX.md | 6.4 | Most topically similar high-scoring anchor; also addresses DiT attention efficiency with condition-specific analysis. PKA's contributions are comparably principled but has the unresolved fair-comparison issue. |
| Deep Compression Autoencoder | wH8XXUOUZU.md | 6.8 | Diffusion efficiency, strong empirical results; higher score reflects cleaner experimental design. |
| Würstchen | gU58d5QeGv.md | 8.0 | Oral, very strong contribution with broader impact; PKA is narrower in scope. |
| LinFusion (linear attention for DiTs) | D2as3jDmRA.md | 6.25 | Attention approximation paper rejected despite technical merit; reviewer concerns about missing baselines resemble PKA's situation. |
| SparseDM (sparse masks for diffusion) | 3kADTLbKmm.md | 4.0 | Sparse diffusion attention with weaker validation; PKA's motivation and evaluation are stronger. |
| Attention optimization without clear validation | vnp2LtLlQg.md | 3.0 | Weak anchor; PKA's evidence base is substantially better. |
| DeeDiff (early exiting for diffusion) | 3xHbRLymyZ.md | 4.5 | Medium-range anchor; diffusion efficiency without principled condition-specific analysis. |

PKA sits between the medium anchors (~4.5) and the PT-T2I/V high anchor (6.4). The contribution is genuine and principled, the empirical motivation is strong, and the efficiency gains are credible. However, the fair-comparison concern (Issue 1) is a real methodological gap that PT-T2I/V does not share, and the mischaracterized metric (Issue 2) undermines trust in the narrative. This places PKA in the 5.0–5.5 range — above the medium anchors (which lack principled design) but below the high anchors (which have cleaner experimental protocols).

## Score and Decision

The paper presents a genuinely motivated and architecturally principled contribution to efficient multi-condition DiT control. The empirical analysis (Figures 2–3), the structural PAA/KSA decomposition, and the efficiency results are legitimate strengths. However, the unresolved training data alignment concern in Table 1 is a substantive methodological gap, and the paper's own characterization of the Subject-Canny F1 regression as "narrow" is factually inaccurate. These are not fatal—the efficiency story is independently credible—but they prevent full confidence in the quality improvement claims. Calibrating against PT-T2I/V (6.4, topically closest strong anchor) and accounting for PKA's weaker experimental controls, I place this paper at **5.5**.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>