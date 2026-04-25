Now let me run calibration searches to properly score this paper.Now I have sufficient context to write the final review. Let me synthesize everything.

---

## Summary

UniCon proposes a unidirectional information-flow paradigm for training control adapters for large-scale diffusion models. Instead of the bidirectional interplay used in ControlNet (where gradients must flow through the frozen backbone), UniCon routes features one-way from the frozen diffusion model into a fully trainable adapter that directly outputs the denoised result. This eliminates the need to compute and store diffusion-model gradients, reducing VRAM and training time roughly by half. The design is validated on both DiT (transformer) and SD U-Net architectures across five conditional generation tasks.

---

## Strengths

- **Clean efficiency gains with detailed accounting (Figure 6):** VRAM breakdown into weight/activation/gradient/optimizer components and separate FP/BP timing is unusually thorough. For DiT Full adapters, UniCon reduces peak VRAM from ~34 GB to ~23 GB (dropping below the 24 GB single-GPU limit) and cuts training time roughly in half, explained transparently by eliminating backbone gradient computation.

- **Architecture-agnostic and genuinely novel design (Sections 1, 3, Figure 2):** The unidirectional paradigm cleanly generalises to both transformer (DiT) and U-Net diffusion models. This is a principled and concrete departure from prior work, not an incremental tweak.

- **Controlled ablation isolating the contribution of unidirectionality (Table 1c):** The paper compares Full-adapter bidirectional vs. unidirectional at identical parameter counts on DiT, showing PSNR improvement from 36.53 → 37.34 and FID from 23.04 → 20.34 for SR, and SSIM improvement on Canny. This is the cleanest evidence that the design principle itself — not just having more parameters — is responsible for gains.

- **Encoder-vs-decoder ablation reveals a useful empirical finding (Table 1a):** For DiT, encoder-focused control yields better image quality while decoder-focused control yields better controllability. The "full" design dominates, and skip-layer fails in the unidirectional setting. This is a practically valuable insight for the community.

- **ZeroFT connector validated against alternatives (Table 1b):** The new connector (element-wise multiplication + skip connection on top of ZeroMLP) consistently outperforms ZeroMLP and ShareAttn, with the superiority corroborated numerically.

- **DiT comparison is parameter-matched and consistent (Table 2, DiT section):** For DiT, ControlNet-Full and UniCon-Full are the same size. UniCon wins on all five tasks and all metrics except one Clip-Score tie, providing credible cross-architecture validation.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Parameter-count confound in the SD U-Net Canny and Depth comparisons (Table 2, SD section):** ControlNet for SD copies only the U-Net encoder, while UniCon copies the entire U-Net (encoder + decoder), giving UniCon roughly twice the trainable parameters. The paper includes UniCon-Half for SR (where it still beats ControlNet), but SD Canny and SD Depth show only the full-parameter UniCon against the half-parameter ControlNet. A reader cannot determine from Table 2 whether UniCon-Half would also outperform ControlNet on Canny and Depth — which would be required to fairly attribute the gains to the unidirectional design rather than to a larger model. The paper should have included UniCon-Half for all SD U-Net tasks, not just SR.

  *Why it matters:* this leaves the SD U-Net headline result partially unsupported. The claim "UniCon outperforms ControlNet" is fully supported for DiT (equal parameters) and for SD SR (UniCon-Half also wins), but is only weakly supported for SD Canny and SD Depth.

- **SUPIR-UniCon evaluation is qualitative only (Figure 8):** The paper's strongest scalability motivation is the ability to train adapters for 8-billion-parameter SD3 where ControlNet is impractical. Yet the SUPIR-UniCon results are limited to three example images (Figure 8) with no quantitative evaluation on any standard image restoration benchmark (e.g., RealSR, DRealSR). Given that SUPIR-UniCon is presented as the most compelling real-world application, the absence of any quantitative comparison with original SUPIR leaves the central scalability claim anecdotal.

### Minor

- **Efficiency claim wording is inconsistent across the paper:** The abstract states "reduces GPU memory usage by one-third," while Figure 1(c) states "ControlNet 2X, UniCon 1X" (implying a 50% reduction), and the main text (Section 4.2) says "saving nearly half the storage required for gradients." These three formulations cannot all be simultaneously correct. The discrepancy appears to arise from comparing different quantities (total VRAM vs. gradient-only VRAM vs. the overhead ratio), but it is never clarified. The paper should either be consistent or explicitly state which quantity each figure refers to.

- **The unidirectional paradigm fails for Skip-Layer adapters (Table 1c), and this constraint on applicability is understated:** Table 1c shows that applying unidirectional flow to the Skip-Layer design actually hurts FID (49.78 → 55.22 for Canny). This finding — that unidirectionality requires sufficient adapter capacity to generate output independently — is explained in one sentence and then dropped. This is a genuine scope limitation that deserves a short dedicated discussion.

### Trivial

*None that are not parser artifacts.*

---

## Nice-to-Haves

- **Quantitative comparison with memory-optimised ControlNet** (e.g., gradient checkpointing, ZeRO), to establish whether UniCon's efficiency advantage persists against a best-effort optimised baseline, not just the naive implementation.
- **Inference-time cost comparison.** UniCon runs the full diffusion backbone plus the full adapter at inference, doubling inference compute versus a plain diffusion model. A table of inference VRAM and latency (UniCon vs. ControlNet) would help practitioners assess deployment cost.
- **Failure case analysis.** Showing representative failure cases, especially for the unidirectional setting where the adapter must generate output without direct backbone fine-tuning, would strengthen the empirical characterisation.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Harsh Critic — "Table 2 contains a block of rows with identical results for ControlNet and UniCon" (lines 212–213):** This is a PDF parser artifact. The hard rules explicitly prohibit flagging formatting artifacts as paper errors; the submitted table does not have this issue.

2. **Harsh Critic — "ControlNet's memory cost assumes naive training; gradient checkpointing would close the gap":** This is a hypothetical baseline not present in the literature as a standard comparison point. Criticising the paper for not testing a non-standard optimised configuration falls under "requesting practices not standard in the field." Moved to Nice-to-Haves.

3. **Harsh Critic — "The ZeroFT connector improvement is marginal and should not be listed as a distinct contribution":** Table 1b does show consistent improvements (SSIM 0.5343 → 0.5426, FID 55.22 → 52.31). Whether these margins warrant being called a "contribution" is a matter of framing, not a falsifiable weakness. The connector is described as one component of UniCon, not a standalone contribution.

4. **Harsh Critic — "Why does unidirectionality help for full-network adapters but not skip-layer adapters? The skip-layer issue could be resolved.":** The paper provides a clear and intuitive explanation. Demanding a follow-up investigation of how to fix the skip-layer case is out-of-scope for the current submission.

5. **Harsh Critic — "Evaluation on 1,000 LAION images limits generalization claim":** All methods are evaluated identically on the same held-out split; relative comparisons are valid. The "broad application prospects" claim is illustrated by SUPIR-UniCon, not solely by this benchmark. This is a very generic concern that does not target a specific overclaim.

6. **Strength Finder — "SUPIR-UniCon produces visually compelling results":** Removed as a strength because the corresponding major weakness (no quantitative evaluation) undermines this strength — when a strength and weakness disagree, the weakness wins.

---

## Novel Insights

UniCon's key insight — that the frozen backbone need not receive gradients at all if the adapter itself serves as the final decoder, consuming backbone features as read-only feature maps — is both simple and underexplored relative to its payoff. The ablation in Table 1c demonstrates that this unidirectional constraint is not just an efficiency trick: it also improves generation quality when the adapter has sufficient capacity, possibly because the trainable decoder can adapt to condition signals without being bottlenecked by the frozen backbone's fixed processing pipeline. The finding that this benefit disappears for skip-layer adapters (insufficient capacity to generate independently) sharpens the principle: unidirectionality is beneficial if and only if the adapter can function as a self-sufficient decoder. This capacity threshold, while not deeply explored in the paper, is a practically actionable design criterion.

---

## Suggestions

1. Add UniCon-Half rows to SD U-Net Canny and Depth in Table 2, so every SD comparison has a parameter-matched reference point.
2. Resolve the one-third vs. one-half efficiency claim across abstract, Figure 1, and Section 4.2 — define precisely which quantity is being measured in each case.
3. Add a short quantitative evaluation of SUPIR-UniCon on one standard benchmark (e.g., RealSR) comparing PSNR/LPIPS against original SUPIR, even if preliminary.
4. Devote a short paragraph to when unidirectionality fails (skip-layer case) and what the minimum capacity requirement is for the adapter to function correctly.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg score | Relation to this paper |
|---|---|---|
| `wGVOxplEbf.md` (SaRA) | 6.20, Accept | Closest topically — efficient diffusion fine-tuning; comparable scope and thoroughness |
| `svp1EBA6hA.md` | 6.50, Accept | Conditional control for diffusion; similar validation depth |
| `kZvor5aaz7.md` | 6.25, Accept | Diffusion adapter for slot-based conditioning; similar contribution scale |
| `az5WtGe48n.md` | 5.20, Reject | Dense vision diffusion adaptation; medium quality; weaker ablations than UniCon |
| `bFMpmb8p3D.md` | 4.75, Withdrawn | Multi-task diffusion adapter; weaker baseline comparisons |
| `RFJGFrMvYj.md` | 1.50, Reject | Controlled generation — two-stage method; much weaker methodology |
| `yPxhj1FKhG.md` | 3.67, Withdrawn | Conditional control diffusion; limited evaluation |

UniCon's core contribution — unidirectional paradigm validated cleanly for DiT with parameter-matched ablations, concrete efficiency gains with detailed accounting, and multi-task evaluation — is clearly above the medium-band papers (5.0–5.33) and broadly comparable to the accept cluster (6.2–6.5). The principal drag is the parameter-count confound for SD U-Net Canny/Depth and the purely qualitative SUPIR-UniCon evaluation. These do not invalidate the core claim (the DiT results are clean and the efficiency is well-characterised) but do weaken some of the headline comparisons. Positioning the paper slightly below the accept cluster centre at **5.5** is appropriate.

**Originality:** Moderate-high. The unidirectional paradigm is a clean conceptual contribution, not merely an engineering optimisation.  
**Importance:** High. Efficient training of large-scale adapters is a pressing practical problem.  
**Claims vs. support:** Mostly well-supported for DiT; partially unsupported for SD (Canny/Depth comparisons).  
**Soundness:** Good; ablations are structured, metrics are appropriate, efficiency analysis is transparent.  
**Clarity:** Good overall; minor inconsistencies in efficiency wording.  
**Value to community:** Moderate-high, particularly for practitioners targeting transformer-based diffusion models.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>