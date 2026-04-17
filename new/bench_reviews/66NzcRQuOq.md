## Summary

This paper introduces pyramidal flow matching, an efficient video generative modeling framework that reinterprets the standard denoising trajectory as a series of pyramid stages operating at progressively higher resolutions. The method combines a spatial pyramid (where only the final stage operates at full resolution) with a temporal pyramid (compressing autoregressive history at lower resolutions), enabling end-to-end training with a single DiT rather than cascaded separate models. A renoising mechanism at inter-stage jump points is derived to maintain probability path continuity. The model generates 5–10 second videos at 768p/24fps using only 20.7k A100 GPU hours, achieving competitive VBench and EvalCrafter results among models trained on public data.

## Strengths

- **Addresses a genuinely important problem**: Efficient training of long, high-resolution video generators is a central challenge. The spatial-temporal pyramid approach targets both key redundancies (noisy early steps don't need full resolution; distant history frames don't need full resolution) with coherent solutions.

- **Competitive empirical performance**: Achieves VBench total score 81.72 and quality score 84.74 (best among public-data models, surpassing CogVideoX-5B which has 2.5× the parameters). EvalCrafter Final Sum Score of 244 also leads public-data models. User study shows preference over all open-source baselines and even some closed-source ones (Kling).

- **Concrete efficiency gains**: 20.7k A100 hours for 10s/241-frame/768p training is a real improvement. Token reduction from 119,040 to ≤15,360 is significant and enables full attention (rather than factorized) at these sequence lengths.

- **Unified training is a practical advance**: Unlike cascaded approaches requiring separate models, the single-DiT, single-objective formulation (Eq. 11) simplifies implementation and enables knowledge sharing across stages. The Patch n' Pack batching strategy naturally handles varying token counts.

- **Zero-shot image-to-video generation**: The autoregressive structure with causal attention naturally supports T2V and I2V without fine-tuning, which is a practical advantage demonstrated in Figure 6.

## Weaknesses

### Major

- **The "unified flow matching" construction is not rigorously established as a single coherent flow.** The paper frames pyramidal flow matching as "reinterpreting the original denoising trajectory as a series of pyramid stages" where "flows of different pyramid stages can be interlinked to maintain continuity" (Abstract). However, training defines per-stage conditional paths (Eqs. 7–10) with a straight-line velocity field within each stage, while inference composes ODE integration with renoising at stage boundaries (Eq. 15). The renoising derivation is only for nearest-neighbor upsampling with a specific covariance structure (Eq. 14), and the resulting inference process is not shown to correspond to the same conditional path family $p_t(\cdot|x_1)$ used in training. The method is therefore better described as multi-scale conditional training plus stitched sampling, not a single flow in the formal sense. This overclaims the theoretical contribution without invalidating the practical utility.

- **Ablation studies are insufficiently rigorous for supporting core claims.** The spatial pyramid ablation (Fig. 7) is image-only, at 50–60k steps, using FID as the sole metric—not at the video scale or benchmarks used in the main evaluation. The temporal pyramid ablation (Fig. 8) is entirely qualitative with no quantitative metrics, compared against an undertrained full-sequence baseline. There is no controlled ablation isolating spatial vs. temporal pyramid contributions at the full video training scale, no ablation on the number of stages $K$ (fixed at 3 throughout), and no ablation on the renoising scheme (Eq. 15). These gaps mean the paper cannot substantiate how much each component contributes to the final results or whether the renoising derivation matters versus simpler alternatives.

- **The semantic alignment gap is significant and the explanation is unvalidated.** VBench semantic score (69.62) is 7+ points below CogVideoX-5B (77.04) and 3+ points below VideoCrafter2 (73.42). EvalCrafter text-video alignment (57.01) is the lowest among all compared methods. The paper attributes this to "coarse-grained synthetic captions" (Sec. 4.3) but provides no experiment (e.g., retraining with better captions, or analyzing failure modes) to validate this claim. An alternative explanation—that pyramidal compression may lose fine-grained semantic detail—is not discussed.

- **Efficiency comparisons are imprecise and not apples-to-apples.** The comparison against Open-Sora 1.2 (20.7k A100 hours vs. 4.8k Ascend + 37.8k H100 hours) mixes GPU architectures, training regimes, and target resolutions/lengths. No breakdown of compute (VAE pretraining vs. DiT) is provided, and the 20.7k GPU hours figure is ambiguous about what it includes. The asymptotic scaling claim ($T^2N^2/16^K$) in Sec. 4.2 yields astronomically large ratios that do not match actual measured speedups, making it misleading.

### Minor

- **The user study lacks methodological detail**: 50 prompts, 20+ raters, but no information on randomization, blinding to model identity, or statistical significance. The 52.3% preference vs. CogVideoX-5B on aesthetics is within noise margin for this sample size.

- **Autoregressive error accumulation is not quantitatively analyzed**: The temporal pyramid relies on conditioning on lower-resolution history, and while corruptive noise training (Sec. 3.3) mitigates drift, no frame-wise quality degradation analysis is provided for 5s vs. 10s videos. VBench evaluation is on 5s clips; the 10s capability is only shown qualitatively.

- **The $16^K/T$ efficiency claim** (Sec. 3.3, end) is confusing: for $K=3$ and typical $T$, this suggests speedups of thousands, far exceeding what's observed. The derivation conflates best-case token reduction with actual FLOP savings and should be clarified.

### Trivial

- The $\oplus$ notation in Eq. 5 introduces an abstraction that is immediately replaced by the concrete formulation in Eq. 6; this could be streamlined.

## Nice-to-Haves

- Quantitative video ablations (VBench/EvalCrafter) for spatial vs. temporal pyramid separately and combined, at the full training scale
- Ablation on $K \in \{2, 3, 4\}$ to characterize the efficiency-quality tradeoff
- Ablation on the renoising scheme (Eq. 15 vs. simple upsampling without renoising vs. isotropic noise)
- Inference latency breakdown by pyramid stage and comparison with baselines at matched resolution/duration
- Confidence intervals on user study preferences

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **"The method is not end-to-end" / "requires stage-by-stage training"** (from Matryoshka DM reviewer analogy): This criticism applies to Matryoshka, not this paper. Pyramidal Flow Matching explicitly trains end-to-end with a unified objective (Eq. 11).

2. **"Reproducibility concerns" / "cannot independently verify"**: The paper provides open-sourced code and models, and training details are specified. This is a knowledge-gap criticism.

3. **"VAE trained on WebVid-10M only creates domain mismatch"**: Speculative concern not supported by evidence. The model achieves strong results despite this.

4. **"Comparison unfairly favors the proposed method"**: The efficiency comparison with Open-Sora actually uses more frames and higher resolution (241 vs. 97 frames), which would disadvantage the proposed method if anything. The asymmetric comparison favors the baseline, not the authors.

5. **"Missing related work on cascaded/diffusion pyramids"**: Per hard rules, do not flag missing related works.

6. **"No confidence intervals on benchmark metrics"**: Large-scale video generation benchmarks like VBench/EvalCrafter are standardly reported as single numbers; demanding confidence intervals is beyond community norms.

7. **"VAE reconstruction quality not reported"**: The VAE is a standard component (similar to MAGViT-v2) trained separately; demanding its separate evaluation is scope creep for a paper focused on the generative model.

8. **"FID vs. training steps is misleading; should report FID vs. wall-clock time"** (from Spark section): This is a reasonable suggestion but the number of tokens per step is transparent (stated in the paper), and Fig. 7 does show convergence curves. Reporting wall-clock time would be useful but is not a fundamental flaw since the tokens-per-step advantage of the pyramid method is exactly the claimed contribution.

## Novel Insights

The paper's most interesting conceptual contribution is the observation that flow matching's flexibility in choosing interpolation endpoints (unlike standard diffusion schedules) naturally supports cross-resolution interpolation—this is what enables the unified training objective. The temporal pyramid conditioning (using progressively compressed history) is also a clean idea that generalizes beyond this specific flow matching formulation. The coupling of shared noise between endpoints (Eqs. 9–10) to encourage trajectory straightness across resolution boundaries is a technically non-obvious design choice that deserves more discussion.

## Suggestions

- **Add a large-scale video ablation** with quantitative VBench metrics for: (a) spatial pyramid alone, (b) temporal pyramid alone, (c) both, and (d) no renoising vs. with renoising. This would substantiate the individual contributions and validate the renoising mechanism.

- **Temper the "unified flow matching" framing**: In the abstract and introduction, present the contribution as multi-scale flow matching with a unified training objective rather than claiming a single coherent flow across all stages, which the current derivation does not fully establish.

- **Provide per-component training time and inference latency**: Break down the 20.7k GPU hours into VAE pretraining, image pretraining, and video training. Report inference time per pyramid stage.

- **Add a K ablation**: Report VBench metrics for K=2 and K=4 alongside K=3 to characterize the efficiency-quality tradeoff frontier.

## Evaluation

**Originality**: The combination of spatial and temporal pyramids with a unified flow matching objective is genuinely novel. The piecewise flow construction and renoising scheme at resolution boundaries represent a creative adaptation of flow matching to multi-scale generation,even if the theoretical grounding is incomplete.

**Importance**: Efficient video generation is a high-impact research direction. Enabling 10s/768p/24fps generation with 20.7k A100 hours on public data is a meaningful practical contribution.

**Claims support**: The core claim that pyramidal flow matching enables efficient training with competitive quality is well-supported by VBench/EvalCrafter results and the user study. The claim that the method constitutes a principled, unified flow is less well-supported due to the heuristic nature of the renoising step and lack of proof that inference follows the training path.

**Soundness of experiments**: Main evaluation is solid (two benchmarks, user study), but ablations are thin—image-only spatial ablation, qualitative-only temporal ablation, no K variation, no renoising ablation.

**Clarity**: The paper is generally well-written. The flow matching formulation is clear, though some notation (Eq. 5's $\oplus$) is unnecessarily abstract.

**Community value**: Open-sourced model and code plus competitive results on public data make this highly valuable for the community.

## Score and Decision

**Calibration anchors:**
- Matryoshka Diffusion Models (multi-scale diffusion, similar ablation concerns, accepted poster, avg score ~6.25): This paper has stronger empirical results (VBench/EvalCrafter) but weaker theoretical grounding.
- CogVideoX (video generation, efficiency, accepted poster, avg ~6.8): This paper has comparable quality but less methodological novelty in the architecture itself; however, the pyramidal training efficiency contribution is distinct.
- CMD (efficient video diffusion, accepted poster, avg ~7): CMD had imprecise fairness concerns in its comparisons and limited ablations, similar to this paper.
- Relay Diffusion (unified cascade across resolutions, accepted spotlight, avg ~7): Relay had cleaner theoretical motivation but limited scale.
- Efficient-vDiT (rejected, avg ~5.75): Had efficiency claims without sufficient ablations.

This paper sits above the rejection line due to genuine algorithmic novelty, strong practical results, and significant community value (open-source, competitive on public data). However, the gaps in ablation rigor, the overclaiming of theoretical coherence, and the imprecise efficiency comparisons prevent a higher score. It is roughly on par with Matryoshka Diffusion Models (6-6.5 range).

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>