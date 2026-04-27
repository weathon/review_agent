Now let me search for calibration papers to score this work properly.Now I have sufficient information to write the final review. Let me synthesize everything.

---

## Summary

PKA (Patch-wise and Keyword-Aware Attention) addresses the quadratic cost of "concatenate-and-attend" in multi-condition Diffusion Transformers. It proposes two sparse attention modules: Position-Aligned Attention (PAA), which restricts spatial conditioning to a strict one-to-one token alignment, and Keyword-Scoped Attention (KSA), which uses a text-keyword-derived mask to confine subject-condition attention to semantically relevant image regions. These are complemented by an early-timestep training sampling strategy. The paper reports up to 10× inference speedup and 5.12× VRAM reduction versus a full-attention baseline, with Table 1 showing competitive or improved generation quality metrics.

---

## Strengths

- **Empirically grounded motivation for sparsity**: Figure 2 shows a clear diagonal structure in spatial-condition attention maps, and Figure 3 shows sparse keyword-localized activations for subject conditions. These are not hand-waving; they directly motivate the PAA and KSA designs with concrete evidence.

- **Substantial and scalable efficiency gains**: Figures 7 and 8 document 3.90×–10× inference speedup and 2.46×–5.12× VRAM reduction over UniCombine as condition count grows from 4 to 16, with per-condition breakdowns. The speedup is practically meaningful, not just a theoretical reduction.

- **Condition Cache as an elegant architectural synergy**: Because condition tokens (SP, SJ) perform only self-attention among themselves, their K/V projections are independent of the noisy image state across denoising steps. This enables a lossless KV cache (computed only at step 1) without approximation—a clean consequence of the decomposed attention design.

- **PAA outperforms sliding-window alternatives even on efficiency**: Figure 9 shows PAA achieves lower latency (13.63s) and VRAM (237MB) than all SWA variants tested (best SWA: 14.00s, 276MB), confirming the 1-to-1 positional alignment is a stronger inductive bias than windowed approximations.

- **Perturbation analysis backs the timestep sampling design**: Figure 5 provides explicit quantitative evidence that perturbing early (high-t) steps causes far greater SSIM degradation than late-step perturbations—a clean empirical motivation for the early-biased Logit-N(μ > 0, δ > 1) schedule.

---

## Weaknesses

### Fatal
None.

### Major

- **Training comparison confound in Table 1**: Section 4.1 states that PKA fine-tunes FLUX.1 via LoRA for 20,000 iterations on an author-curated subset of Subject200K. However, the paper does not state that OminiControl2 and UniCombine were retrained by the authors under identical conditions (same data split, iterations, optimizer). The phrase "We employ OminiControl2 and UniCombine as baselines" most naturally means released weights are used. If so, Table 1 conflates attention architecture with training differences—including the authors' early-timestep sampling strategy, which §4.3.3 independently shows improves generation quality. The claim that PKA "maintains or improves generative quality" relative to full attention is therefore not isolated to the architectural choice. An ablation that holds training fixed and varies only the attention module is needed to support this claim.

- **PAA ablation provides no quantitative quality metrics**: The PAA ablation (Figure 9) shows only qualitative images and latency/VRAM numbers—no FID, SSIM, F1, or MSE. PAA is one of the two core architectural contributions; it eliminates all non-local cross-attention for spatial conditions, replacing it with a strict one-to-one mapping. For a change this dramatic, the absence of quantitative quality evaluation means the efficiency–quality tradeoff for PAA specifically cannot be measured. The claim that PAA achieves "high-quality spatial control at lower cost" rests solely on cherry-picked qualitative comparison.

### Minor

- **Headline 10× speedup applies to an unvalidated condition count**: All quality evaluations (Table 1, Figure 6) use 2-condition tasks. The headline 10× figure appears in the abstract, introduction, and conclusion but applies only to the 16-condition setting. Figure 7 does show curves for 1–16 conditions, but neither efficiency nor quality results are reported jointly for the same condition count. The practical 2-condition speedup (readable from Figure 7 but not stated) should be reported alongside Table 1 so readers can assess efficiency and quality in the same experimental regime.

- **F1 drop on Subject-Canny mischaracterized**: Table 1 shows PKA achieves F1 = 0.414 vs. UniCombine's 0.551 on edge controllability for the Subject-Canny task—a 25% relative drop. The paper calls this "a narrow margin on the Subject-Canny task" (§4.2.3). This characterization is inaccurate and obscures a genuine trade-off that a practitioner needs to understand when choosing PKA over a full-attention model. This gap should be reported honestly, contextualized, and investigated rather than minimized.

- **Early-timestep sampling ablation is qualitative only**: Figure 11 compares (μ, δ) configurations via visual snapshots at different iterations but provides no quantitative convergence metrics (FID, SSIM, CLIP-I at convergence). Since this strategy also influences the final quality reported in Table 1, a quantitative ablation would help decompose its contribution from the PKA attention modules.

### Trivial

- The Condition Cache validity (K/V frozen from the noisiest step and reused) relies on the structural choice that condition tokens never attend to image tokens. This is correct and important, but is not explicitly articulated—a brief justification in §3.2 would preempt likely reviewer confusion.

---

## Nice-to-Haves

- A component-wise ablation table with quantitative metrics for (1) early-timestep sampling only, (2) PAA only, (3) KSA only, (4) all combined would cleanly decompose the quality and efficiency contributions of each module.
- An ablation comparing KSA with fresh mask at each step vs. the temporally reused mask from step t, especially for early denoising stages where temporal consistency is weakest—given that early timesteps are claimed to be the most conditioning-critical.
- Failure mode visualization for KSA at high thresholds (ε ≥ 0.6) on subjects that occupy small or low-salience areas, to calibrate graceful degradation claims.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they were found to not hold up to verification.*

- **"Reproducibility: hyperparameter details undisclosed"** — The paper specifies LoRA, Prodigy optimizer, 20k iterations, batch size 1, gradient accumulation 4, ε = 0.2 by default. Removed as a nitpick.
- **"PAA incompatible with non-aligned conditions"** — PAA is explicitly designed only for spatial-aligned conditions (Canny, Depth), not all condition types. The paper clearly scopes its application. Not a weakness.
- **"Test set fairness: Subject200K overlap with OminiControl baseline"** — Subject200K is the established benchmark for this task; the authors use a curated subset distinct from full training. This is standard practice and not a flaw.
- **"Temporal consistency weakest in early denoising invalidates KSA"** — This is a plausible but unverified concern; temporal consistency in the spatial layout of generated content holds reasonably even early in denoising. Moved to nice-to-have ablation.

---

## Novel Insights

The observation that condition-type specificity enables architectural specialization—not just pruning—is the most transferable insight here. PAA and KSA succeed not because they impose generic sparsity but because they encode modality-specific spatial priors (aligned positional correspondence for structural conditions; keyword-activated region scoping for semantic conditions). The Condition Cache is a direct consequence of this modularity: because conditions never read from image tokens, their representations are image-step-agnostic and can be frozen. This principle—that correct decomposition of cross-modal attention by condition type enables both efficiency and structural inductive bias—is a useful design principle beyond this specific setting.

---

## Suggestions

1. Retrain OminiControl2 and UniCombine (or at least one baseline) on the same data split with the same iteration budget, and compare against PKA using identical training procedure. If only the attention module differs, this directly validates the architectural contribution.
2. Add quantitative FID/SSIM/F1 rows to the PAA ablation (Figure 9) alongside the latency/VRAM numbers—this is a small ask that would substantially strengthen the core claim.
3. Report 2-condition inference time and VRAM alongside Table 1 so efficiency and quality results share the same experimental regime.
4. Revise §4.2.3 to report the Subject-Canny F1 drop as the trade-off it is, and provide a brief analysis of when this matters.

---

## Calibration

**Anchor papers reviewed:**
- `/home/wg25r/review_agent/human_reviews/lTrrnNdkOX.md` (PT-T2I/V, avg 6.4): Directly topically similar—sparse attention for DiTs, motivated by redundancy analysis, accepted as poster. Stronger than PKA in breadth of evaluation (T2I, T2V, T2MV) with cleaner baseline comparisons. PKA's confounded training comparison and missing quantitative ablation are weaker than PT-T2I/V's issues.
- `/home/wg25r/review_agent/human_reviews/OqTVwjLlRI.md` (S2-Attention, avg 4.25): Sparse attention paper with missing quality-at-scale validation and weak downstream evidence, rejected. PKA is clearly stronger: better motivated, more complete empirical package, practical domain.
- `/home/wg25r/review_agent/human_reviews/Jt1gGIumJo.md` (Highlight Diffusion, avg 3.0): Training-free attention acceleration for T2I, much weaker paper, rejected. PKA is substantially stronger with trained architecture, full evaluation suite, and real speedups.
- `/home/wg25r/review_agent/human_reviews/pDI03iK5Bf.md` (Δ-DiT, avg 5.5): DiT acceleration via denoising property alignment, borderline reject. Comparable in scope to PKA; similar issue of qualitative ablations without tight quantitative decomposition.
- `/home/wg25r/review_agent/human_reviews/eC2ICbECNM.md` (Ctrl-U, avg 6.0): Conditional image generation, accepted. Cleaner experimental design with controlled ablations; PKA's confounded comparison is a notable gap relative to that standard.

PKA sits between the Δ-DiT (~5.5) and PT-T2I/V (~6.4) anchors. The core mechanisms are well-motivated and the efficiency gains are real, but the confounded training comparison—the central evidentiary issue for the quality claim—and the missing quantitative PAA ablation pull the score below PT-T2I/V. The paper is stronger than Δ-DiT due to clearer motivation and better-characterized efficiency results, but not cleanly.

**Final assessment:** The efficiency contribution is solid and the method is well-motivated. The quality claims are the weak point—confounded with training differences, and lacking quantitative ablation for one of the two core modules. The paper is borderline accept/reject with a lean toward reject in current form due to the training confound, which is fixable but not cosmetic.

## Score and Decision

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>