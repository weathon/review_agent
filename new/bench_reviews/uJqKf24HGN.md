Now let me search for calibration papers to calibrate my score.Now I have enough information to write the final review. Let me verify one more specific item before writing.Now I have all the information needed to write the final review.

---

## Summary

UniCon introduces a unidirectional information flow paradigm for training control adapters on large-scale diffusion models. Rather than injecting residuals back into a frozen diffusion model's intermediate layers (as ControlNet does), the UniCon adapter reads features from the frozen model's layers and directly produces the final denoised output — eliminating the need to compute or store gradients for the frozen backbone. This yields measurable efficiency gains (approximately half the VRAM, 2.3× training speed on DiT) while also enabling the training of larger adapters within the same resource budget. The method is validated on five conditional generation tasks across both U-Net (SD-2.1) and transformer (PixArt-α/DiT) architectures.

---

## Strengths

- **Principled efficiency gain with transparent accounting (Figure 6, Section 4.2):** Severing backpropagation through the frozen model is a clean architectural insight, not a trick. The per-component VRAM breakdown in Figure 6 shows gradients are the dominant cost, and UniCon's design eliminates them from the frozen model. The reported ~2× VRAM reduction and 2.3× speedup on DiT are consistent with this analysis.

- **Same-parameter comparison validates the architectural advantage (Figure 1c, Table 2 DiT-SR):** Figure 1c explicitly shows that at equal trainable parameter counts, UniCon still outperforms ControlNet on SR (FID 22.07 vs. 26.43), demonstrating the gain is not solely from having more parameters. The inclusion of UniCon-Half in Table 2 for DiT-SR further supports this: even with half the full UniCon parameters, performance exceeds the encoder-only ControlNet baseline (PSNR 35.64 vs 34.82, FID 22.07 vs 26.43).

- **Architecture-agnostic applicability (Table 2, Figure 2):** UniCon is validated on both U-Net and transformer-based DiT, overcoming ControlNet's structural assumption of an encoder–decoder split. Table 1a confirms that for DiT, using all network layers (Full/Skip-Layer) outperforms encoder-only control, motivating the Full-copy UniCon design.

- **Well-structured ablation study (Tables 1a–1c):** The five-variant ablation comparing encoder/decoder/skip-layer/full copying strategies, connector types (ZeroMLP, ShareAttn, ZeroFT), and bidirectional vs. unidirectional flow is systematic and informative. Figure 4's ablation showing that discarding the frozen model's decoder causes quality degradation meaningfully supports the design rationale.

- **Broad task coverage (Table 2):** Testing five condition types spanning high-level semantic control (Canny, Depth, Pose) and pixel-accurate low-level control (4× SR, deblur+SR) provides a wider empirical picture than most prior adapter work.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Missing same-parameter comparison for high-level DiT tasks (Canny, Depth, Pose) in Table 2.** For these tasks, the table compares UniCon-Full (a full copy of the DiT model, ~2× the parameters of ControlNet-Encoder) against the encoder-only ControlNet. While the paper's efficiency framing argues that UniCon-Full is cost-equivalent to ControlNet at training time, the reader cannot separate the architectural advantage from the parameter advantage without a UniCon-Half (or equivalent) row for Canny, Depth, and Pose. UniCon-Half is provided for SR only. This gap is acknowledged conceptually in Figure 1(c)/(d), but is not filled in the primary comparison table for the three high-level tasks, making the performance claims for those tasks inconclusive on an equal-parameter basis.

- **SUPIR-UniCon (SD3) result is purely qualitative (Section 4.3, Figure 8).** This is presented as UniCon's most compelling scaling demonstration — a next-generation 8B-parameter backbone (SD3), architecturally distinct from the DiT used in all quantitative experiments. Yet Figure 8 contains only cherry-picked image triples with no PSNR, LPIPS, MUSIQ, or FID scores and no comparison against the ControlNet-based SUPIR baseline at any parameter budget. Given that SD3 is both larger and structurally different, the qualitative examples cannot substitute for quantitative evaluation here.

### Minor

- **FID computed on only 1,000 test images (Section 4.1).** The standard practice for FID estimation is ≥10,000 images; variance at 1,000 samples is substantial. Several key comparisons in Table 2 show FID differences of 2–5 points (e.g., SD Depth: 54.30 vs 53.45), which may well be within sampling noise. No confidence intervals or bootstrap estimates are reported. This weakens the FID-based claims, though the directional trend is supported by the non-FID metrics.

- **Inconsistent memory-savings claim between Abstract and Introduction.** The Abstract states "reduces GPU memory usage by one-third" while the Introduction states "saves half the video memory (VRAM)." Figure 6 and Section 4.2 describe approximately halved gradient-related VRAM for DiT. These should be reconciled with a single precise statement tied to a specific configuration.

- **Canny FID trade-off not prominently acknowledged.** Table 1b/1c shows that for Canny, UniCon with ZeroFT achieves FID=52.31, which is slightly worse than bidirectional Skip-Layer ControlNet (FID=49.78), even though UniCon's SSIM is better (0.5426 vs 0.4983). The paper's text states that "employing the unidirectional information flow substantially enhances performance" without noting that this improvement is on controllability, and that FID is slightly lower for Canny specifically.

- **SD Depth comparison with T2I-Adapter not adequately analyzed (Table 2).** T2I-Adapter outperforms UniCon on Clip-IQA, MAN-IQA, and MUSIQ for SD Depth, while UniCon's MSE advantage is small (85.00 vs 87.72, ~3%). The paper briefly acknowledges this but offers no quantitative analysis of the quality–controllability trade-off.

### Trivial

- The training cost analysis (Figure 6) is conducted without gradient checkpointing on ControlNet. In practice, gradient checkpointing substantially reduces ControlNet's memory footprint. Noting this condition explicitly would make the efficiency comparison more precise.

---

## Nice-to-Haves

- Adding UniCon-Half to Table 2 for Canny, Depth, and Pose (DiT) would provide a definitive same-parameter comparison and substantially strengthen the core performance claim.
- Adding quantitative evaluation for SUPIR-UniCon on SD3 (even a small set: PSNR, LPIPS, MUSIQ vs. SUPIR-ControlNet) would transform Section 4.3 from illustrative into evidence.
- Recomputing FID on ≥5,000 or 10,000 test images would substantially improve confidence in the perceptual quality claims.
- A wall-clock convergence plot (training loss / validation metric vs. time) would reveal whether UniCon's 2.3× per-step speedup translates to fewer total hours to reach equivalent quality.
- A failure-case analysis for UniCon would be informative, particularly for cases where the frozen model's features conflict with the adapter's control signal.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "ControlNet-Full should be the primary baseline in Table 2."** The paper includes ControlNet-Full as an ablation in Table 1a, and the standard convention in the field is to compare against the canonical baseline (ControlNet with encoder copy). Using encoder-only ControlNet as the main baseline is the established norm, and the paper does compare against it consistently. This is standard practice, not an omission. Removed.

- **Harsh Critic: "Table 2, odd entry for DiT SR with identical values for ControlNet and UniCon."** The extracted line 213 (showing apparently identical metrics for both) is a parser artifact from the PDF extraction — the original table does not duplicate rows. Per the hard rules, parser errors are not paper errors. Removed.

- **Harsh Critic: "ZeroFT is overstated as a major contribution."** Table 1b shows ZeroFT improving Canny FID by ~3 points and SSIM by ~0.008 over ZeroMLP. The paper describes this as "superior" but does not claim it as the central contribution. The ZeroFT connector is presented as one component of the design. The criticism that its mechanistic explanation is absent is minor but not a meaningful weakness. Removed as a weakness; could be a trivial nitpick.

- **Strength Finder: "Practical demonstration on large-scale restoration (SUPIR-UniCon / Figure 8)."** This is listed as a strength, but as noted under Major Weaknesses, Figure 8 is purely qualitative. A qualitative demonstration without metrics cannot function as a genuine strength. Removed per the rule that strengths conflicting with verified weaknesses are dropped.

---

## Novel Insights

The central insight — that the computational cost asymmetry in bidirectional adapters (where the frozen backbone requires gradient storage proportional to its own size) can be eliminated by inverting the information flow direction so the adapter becomes a trainable decoder rather than a residual injector — is both simple and architecturally principled. The ablation showing that discarding the frozen decoder while letting the adapter produce the output (Figure 4, variant e) significantly degrades quality establishes that the frozen model's forward computation is informationally necessary even when its gradients are not, which is a non-obvious and practically important finding. This insight also explains why the method scales better to transformer architectures where encoder/decoder boundaries are artificial.

---

## Evaluation on Key Axes

- **Originality:** Moderate-to-high. The unidirectional flow framing is a clear departure from ControlNet's paradigm; the idea itself is simple but the execution (zero-initialized connectors, full-model copy, DiT compatibility) is well-thought-out.
- **Importance of research question:** High. Efficient adapter training for large diffusion models is directly relevant to the community scaling toward 8B+ parameter backbones.
- **Claims vs. support:** Moderate. Core efficiency claims are well-supported; performance claims for DiT high-level tasks lack same-parameter comparison, and SD3 scaling is unsupported quantitatively.
- **Soundness of experiments:** Moderate. Five tasks, two architectures, solid ablations. Weakened by small FID sample size and asymmetric parameter comparison for Table 2 headline tasks.
- **Clarity of writing:** Good overall, with one internal inconsistency (memory savings claim).
- **Value to research community:** Meaningful. The efficiency mechanism is replicable and the architecture-agnostic framing is practically valuable.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to UniCon |
|---|---|---|---|
| ELR-Diffusion (low anchor) | `edx7LTufJF.md` | 2.5 | Fundamentally weaker: duplicated sentences, missing baselines, flawed efficiency claim. UniCon is clearly above this. |
| DiffBody (low) | `R8t9Q3jmCQ.md` | 4.0 | Domain-restricted, narrow contribution, weaker ablations. UniCon is stronger. |
| 3D-Adapter (medium) | `C0HDYvGwol.md` | 5.6 | Rejected despite 5-6 range scores. Adapter for diffusion models, similar scope. Missing baselines and references were concerns. UniCon has cleaner experimental setup but similar scope. |
| SaRA (medium-high) | `wGVOxplEbf.md` | 6.2 | Accepted poster. Efficient diffusion fine-tuning with progressive sparse low-rank adaptation. Comparable ablation depth. UniCon's core insight is slightly cleaner; SaRA's core assumption has empirical weaknesses reviewers flagged. |
| Adding Conditional Control with RL (medium-high) | `svp1EBA6hA.md` | 6.5 | Accepted poster. Adds conditional control to diffusion, limited experimental breadth. UniCon has more thorough experiments. |
| Würstchen (high) | `gU58d5QeGv.md` | 8.0 | Oral. Highly novel compression-based diffusion architecture. Substantially more fundamental architectural innovation than UniCon. UniCon is below this level. |

**Positioning:** UniCon sits between the 3D-Adapter (5.6, rejected) and SaRA (6.2, accepted poster). The 3D-Adapter was rejected partly for missing baselines and citations (not applicable here); UniCon's comparable weaknesses (parameter asymmetry, qualitative SD3 only) are less severe than those that sank 3D-Adapter. The efficiency contribution and cross-architecture validation push it above the 3D-Adapter level. However, the lack of same-parameter comparisons for three out of five main DiT tasks and the purely qualitative SD3 result are genuine gaps that keep it below SaRA's cleanness. I place it at **5.5**, leaning toward a weak accept at the poster level contingent on the authors providing same-parameter comparisons for Canny/Depth/Pose in Table 2.

**MY FINAL SCORE: <pineapple>5.5</pineapple>**
**MY FINAL DECISION: <orange>Accept</orange>**