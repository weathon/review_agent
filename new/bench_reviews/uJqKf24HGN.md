## Summary

The paper proposes UniCon, a control adapter architecture for large-scale diffusion models that replaces the bidirectional (ControlNet-style) interaction pattern with a unidirectional information flow: the frozen diffusion model feeds intermediate features to a fully trainable adapter, which directly outputs the denoised latent $z_{t-1}$. This design eliminates gradient computation for the diffusion backbone, reducing training VRAM by approximately half and achieving 2.3× training speedup. The method is validated across both U-Net (SD2.1) and Transformer (PixArt-α/DiT) backbones across five conditioning tasks, showing consistent improvements over ControlNet and T2I-Adapter in controllability and generation quality.

## Strengths

- **Clear, practical efficiency contribution with measured gains.** Figure 6 demonstrates that UniCon nearly halves gradient-storage VRAM (DiT: 18 GB adapter alone vs. 16 GB additional from Diff. Model in ControlNet) and achieves ~2.3× training speedup by eliminating backward passes through the frozen backbone. This is a concrete engineering contribution for scaling adapters to large diffusion models.

- **Architecture-agnostic design validated on both U-Net and Transformer backbones.** Table 2 shows UniCon outperforms ControlNet across all 20 reported metrics on DiT and most metrics on SD U-Net. The method does not assume an encoder-decoder structure, making it naturally compatible with transformer-based diffusion models where encoder/decoder boundaries are ambiguous.

- **Systematic ablation of design choices.** Table 1a provides a useful decomposition of encoder vs. decoder control, revealing that decoder-side control improves controllability while encoder-side control improves generation quality—explaining why encoder-only ControlNets struggle on pixel-precise tasks like super-resolution. Table 1b validates the ZeroFT connector against ZeroMLP and ShareAttn.

- **UniCon-Half provides partial parameter-controlled comparison.** The paper acknowledges the parameter scaling issue and includes UniCon-Half (half parameters) baselines in Table 2, which still outperform same-parameter ControlNet on SR (PSNR: 35.64 vs. 34.82 on DiT) and deblur tasks.

## Weaknesses

### Fatal
None

### Major

- **Primary comparison (UniCon Full vs. ControlNet) conflates architectural benefits with capacity scaling.** Standard ControlNet copies only the encoder portion of the backbone (~50% of parameters), while UniCon Full copies the entire diffusion model (~2× parameters). The headline Table 2 comparisons (e.g., DiT-Canny SSIM 0.4748 vs. 0.5458) thus mix architectural advantage with raw capacity increase. While UniCon-Half partially addresses this, the paper's main narrative—"unidirectional information flow yields superior controllability"—should isolate the architecture from the parameter count. This gap means the architectural contribution cannot be fully separated from the capacity benefit in the primary reported results.

- **Ablation data reveals a fidelity-control trade-off for the unidirectional design, but the narrative presents unidirectional flow as universally superior.** Table 1c shows that for the Canny task, Full unidirectional UniCon (FID 55.22, Clip-Score 0.7612) actually **underperforms** Skip-Layer bidirectional (FID 49.78, Clip-Score 0.7776) in image quality and text consistency. The paper acknowledges this only for Skip-Layer ("not suitable for UniCon") but never provides the critical Full-bidirectional Canny comparison to complete the picture. The claim that unidirectional flow "substantially enhances performance, improving controllability and generative quality in both high-level and low-level tasks" is overstated given this trade-off.

### Minor

- **Direct latent ($z_{t-1}$) prediction formulation lacks theoretical justification or alternative comparisons.** Standard diffusion training predicts noise residuals $\epsilon$ or denoised images $x_0$ to maintain trajectory stability across denoising steps. UniCon instead predicts the full next-step latent $z_{t-1}$ directly through the trainable adapter layers (Section 3, line 92). The paper provides no ablation comparing $z_{t-1}$ prediction against $\epsilon$ or $x_0$ targets, no analysis of error accumulation across timesteps, and no discussion of how this choice interacts with different noise schedulers or step counts. This makes the empirical success somewhat opaque and potentially brittle to alternative diffusion configurations.

- **SUPIR-UniCon scaling claim is purely qualitative with no quantitative metrics or training resource documentation.** Figure 8 shows visually appealing restoration crops from a model trained on SD3 (8B parameters), but the paper reports no FID, LPIPS, or controllability metrics, no training VRAM/time costs, and no dataset details. Given that a full-copy UniCon adapter for an 8B-parameter model would require substantial compute, the absence of resource documentation makes it impossible to assess the practical feasibility of this scaling demonstration.

- **The Canny bidirectional comparison in Table 1c is structurally incomplete.** For the Canny task, the unidirectional Full adapter (SSIM 0.5343, FID 55.22) is compared against Skip-Layer bidirectional (SSIM 0.4983, FID 49.78) and Decoder bidirectional (SSIM 0.5131, FID 59.32). A Full bidirectional baseline—which would be the most direct comparison—is absent for Canny (it exists only for SR), leaving the reader to infer rather than observe whether the unidirectional design wins when the parameter count is held equal.

### Trivial
None

## Nice-to-Haves

- **Provide a trajectory stability analysis for the $z_{t-1}$ prediction.** Quantifying how prediction error accumulates (or doesn't) across denoising timesteps would clarify whether UniCon generalizes beyond the specific DDPM/DDIM schedule used. This is not essential to the core claim but would strengthen the methodological foundation.

- **Include failure cases that show when unidirectional control breaks down.** Showing instances where the adapter fails to preserve fine-grained structure would reveal the limits of the approach and help the community understand the applicability boundary.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"SR PSNR gains shrink from +10.9 dB to +0.82 dB; Canny SSIM gains drop from +0.11 to +0.07"** — The harsh critic's numbers are incorrect. From Table 2, DiT SR PSNR is 34.82 (ControlNet) vs. 37.34 (UniCon Full), a gain of +2.52 dB, not +10.9 dB. The critic appears to have used fabricated or misread values. The underlying concern about capacity scaling is valid but the specific claims are inaccurate.

- **"Conflates gradient elimination with architectural novelty"** — Freezing backbones is standard PEFT practice, but UniCon's structural change (adapter as sole output pathway, no residual injection, full-model copy) is genuinely different from LoRA or ControlNet. The gradient-backprop elimination is a consequence of this structural change, not merely a rebranding of frozen-backbone training.

- **"ZeroFT tensor dimensions and normalization schemes are omitted"** — While the exact dimensions are not spelled out in the main text, Figure 2(c) clearly shows the connector structure (ZeroConv/MLP → element-wise multiply → ZeroConv/MLP with skip), which is sufficient for replication given the reference architectures (PixArt-α, SD2.1) with known block dimensions. This is too granular for a conference submission.

- **"LAION crop/resize domain shift" and "duplicated/mislabeled rows in Table 1c"** — The 512×512 cropping from higher-resolution LAION images is a standard preprocessing choice consistent with prior work. Table 1c's SR rows include both Full bidirectional (✗) and Full unidirectional (✓), which is a correct comparison for SR. There is no evidence of duplication.

- **"SUPIR-UniCon requires massive compute contradicting low-resource narrative"** — The paper explicitly positions SUPIR-UniCon as a qualitative demonstration of UniCon's broad applicability (Section 4.3, Figure 8). The efficiency savings are in training overhead, not eliminating the base model cost—which is inherent to any adapter. This is not contradictory.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Add a Full-bidirectional (same-architecture) Canny comparison to Table 1c.** This would either confirm the unidirectional advantage for Canny as well, or honestly reveal that the trade-off is task-dependent.

2. **Include an ablation of output targets** ($z_{t-1}$ vs. $\epsilon$ vs. $x_0$) to validate the direct latent prediction choice. Even a small-scale experiment on one task would significantly strengthen the methodological grounding.

3. **Report quantitative metrics for the SUPIR-UniCon model.** FID, LPIPS, and training VRAM/time costs would make the scaling demonstration substantially more informative and allow the community to judge practical feasibility.

4. **Consider separating the "same-parameter" narrative from the "same-resources" narrative** in Table 2, perhaps with a dedicated column or clearer framing, to avoid confusing readers about what is being compared.

## Score and Decision

I calibrated this paper against:
- **Higher-scoring anchor**: *AnimateDiff* (Fx2SbBgcte, scores 6/8/8/6, avg 7.0) — a clean adapter contribution with strong empirical results across community models. UniCon is comparable in its adapter novelty and multi-backbone validation but has a more notable methodological gap ($z_{t-1}$ prediction unvalidated).
- **Mid-range anchor**: *EfficientDM* (UmMa3UNDAz, scores 6/6/8/6, avg 6.5) — strong efficiency results with incomplete analysis. UniCon is similar: clear efficiency contribution but incomplete baseline isolation and ablation.
- **Borderline anchor**: *LoRA-X* (6cQ6cBqzV3, scores 3/6/6/8) — strong results but incomplete comparisons. This paper is closer to this tier due to the parameter-confounding issue; the partial UniCon-Half remedy is better than nothing but incomplete.
- **Lower anchor**: *X-PlugVid* (TTWxMAwS6n, scores 5/5/3, avg 4.3) — adapter with significant comparison flaws. UniCon is notably stronger than this in its experimental coverage.

**UniCon has genuine contributions**: the efficiency gains are real and well-measured, the architecture-agnostic demonstration is valuable, and the systematic ablation of connector designs and copy-target choices is useful. However, the parameter-confounding gap, the Canny fidelity trade-off, and the $z_{t-1}$ formulation validation gap prevent a clear accept. It sits solidly in the borderline range with meaningful strengths but unresolved questions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>