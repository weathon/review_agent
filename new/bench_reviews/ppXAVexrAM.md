Now I have a good picture of the paper and calibration papers. Let me now carefully verify the harsh critic's claims against the paper text and construct the final review.

Key claims to verify:
1. "Claim of superiority over SOTA is contradicted by the paper's own data" - Let me check. The paper's abstract says "comparable to state-of-the-art" while the Discussion says "out-performs state-of-the-art methods". Table 1 shows ARSS wins on PSNR and LPIPS vs SEVA on Re10K (19.02 vs 18.73 PSNR, 0.269 vs 0.349 LPIPS) and ACID (21.93 vs 21.77 PSNR, 0.265 vs 0.326 LPIPS), but loses on SSIM (0.624 vs 0.670; 0.623 vs 0.664) and FID on ACID (47.76 vs 33.16). The text acknowledges this: "although our method produces higher-fidelity novel views (e.g., +1.1% PSNR, –21% LPIPS), it can show minor geometric inconsistencies (e.g., –6.6% SSIM, +22% FID)." So the paper is fairly transparent about the tradeoffs in the results section, but the Discussion overclaims by saying "out-performs" when the results are mixed. This is a valid but moderate concern.

2. "The central motivation (causal generation for incremental trajectory extension) is never experimentally validated" - This is valid. The paper motivates AR over diffusion by saying it allows incremental extension and reuse, but no experiment tests this.

3. "No ablation on the camera autoencoder" - This is valid. The ablation only covers token permutation and tokenizer choice, not the camera autoencoder.

4. Claims about Table 2 missing FVD - Let me verify. Table 2 (permutation ablation) reports PSNR, SSIM, LPIPS, FID. It does NOT report FVD. Table 3 (tokenizer ablation) reports PSNR, SSIM, LPIPS, FVD (no FID). The absence of FVD in Table 2 is noted by the critic. This is a valid minor concern since FVD measures temporal consistency.

5. Error accumulation analysis based on 17 frames - Valid concern.

6. 256×256 resolution - Valid, though the paper acknowledges tokenizer limitations.

7. No inference speed comparison - Valid concern.

8. The Equation 6 notation error - This is a formatting/parser issue, remove it.

9. "FID is 44% worse" - On ACID, ARSS FID is 47.76 vs SEVA's 33.16. 47.76/33.16 = 1.44, so 44% higher. The paper calls this "+22% FID" which seems to be measuring (47.76-33.16)/47.76 ≈ 30% or perhaps uses a different calculation. Actually let me recalculate: (47.76-33.16)/33.16 = 0.44, so the FID is 44% worse. The paper says "+22% FID" for some reason. The percentage calculation is wrong - perhaps they were computing relative to a different baseline. Actually wait, the text says "–6.6% SSIM, +22% FID" - maybe they're comparing on Re10K: ARSS FID 47.60 vs SEVA 46.98 = +1.3% difference. On ACID: ARSS FID 47.76 vs SEVA 33.16 = 44% difference. The paper is perhaps citing Re10K numbers? Or maybe they cherry-picked. Actually for SSIM: Re10K 0.624 vs 0.670 = (0.670-0.624)/0.670 = 6.87%. For ACID: (0.664-0.623)/0.664 = 6.17%. So SSIM is ~6.6% lower on both. FID: Re10K (47.60-46.98)/46.98 = 1.3%, ACID (47.76-33.16)/33.16 = 44%. So "+22% FID" averages? Or they cherry picked? This is concerning.

Now let me calibrate the score. The paper proposes a reasonable first application of decoder-only AR models to NVS with camera control. It has:
- Clear novelty as first GPT-style AR model for NVS with camera control
- Reasonable architecture with three well-motivated modules
- Competitive but NOT superior results (wins on PSNR/LPIPS, loses on SSIM/FID)
- Overclaimed in Discussion as "outperforms" 
- Missing key ablations (camera autoencoder, no test of incremental generation advantage)
- Low resolution (256×256)
- Missing FVD from permutation ablation
- No inference speed comparison

Comparison with anchors:
- LVSM (7.67, Oral): Much stronger results (1.5-3.5 dB PSNR gain), more thorough evaluation
- NOVA (6.5, Poster): Strong AR visual generation with good results, clean ablations
- NVS-Solver (6.0, Poster): Training-free method, decent but limited
- 3D-free meets 3D priors (5.0, Reject): Overclaimed results, mixed metrics, poor writing
- Multi-view consistent generation (3.5, Reject): Limited evaluation, overclaimed

ARSS is somewhere between 3D-free meets 3D priors (5.0) and NOVA (6.5). It has real novelty (first AR model for NVS) and reasonable but mixed results. The overclaim issue is moderate (the abstract honestly says "comparable" but Discussion says "out-performs"). The missing ablations are a concern. The lack of testing the core motivation (incremental generation) is significant but the paper is still presenting a reasonable first exploration.

I'd place this around 5.0-5.5. It's a borderline paper with real novelty but significant gaps in evaluation completeness and overclaiming.</think>Now I have thoroughly read the paper and reviewed all relevant calibration anchors. Let me synthesize the final review.

## Summary

ARSS proposes the first GPT-style decoder-only autoregressive model for novel view synthesis from a single image with camera trajectory conditioning. The framework combines three modules—a video tokenizer for temporal consistency, a camera autoencoder that maps Plücker raymaps to latent tokens as 3D positional guidance, and an autoregressive transformer with a spatial-permutation strategy that preserves temporal causality while allowing bidirectional spatial context. Experiments on RealEstate10K, ACID, and DL3DV show results competitive with (but not uniformly superior to) diffusion-based SOTAs like SEVA.

## Strengths

- **Novel and well-motivated problem formulation.** Applying decoder-only autoregressive models to NVS with explicit camera control is genuinely underexplored. The paper correctly identifies key challenges (temporal consistency, 3D-aware conditioning, bidirectional spatial context) and addresses each with a dedicated module. To the authors' knowledge, this is the first GPT-style causal AR model applied to NVS with camera control, establishing a new paradigm distinct from joint denoising.

- **Strong LPIPS performance.** On both RealEstate10K (0.269 vs. 0.349) and ACID (0.265 vs. 0.326), ARSS substantially outperforms SEVA on LPIPS, indicating superior perceptual quality. It also achieves the best PSNR on both datasets (19.02 vs. 18.73; 21.93 vs. 21.77).

- **Permutation ablation provides useful insight.** The comparison of raster-scan vs. full-permutation vs. spatial-only permutation (Table 2) clearly demonstrates that preserving temporal order while permuting spatial order yields the best results (19.22 vs. 16.29 and 18.76 PSNR), validating the core design principle.

- **Video tokenizer ablation is convincing.** Replacing the video tokenizer with a per-frame VQ tokenizer drops PSNR from 19.22 to 15.69 and worsens FVD from 52.56 to 137.68, establishing that temporal compression is essential.

- **Geometrically grounded camera encoding.** The Plücker raymap autoencoder with orthogonality and normalization losses (Eq. 5) provides physically meaningful 3D positional guidance rather than relying on a single global conditioning token.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed superiority when results are mixed.** The Discussion states ARSS "out-performs state-of-the-art methods leveraging diffusion models and transformers," but Table 1 shows ARSS loses to SEVA on SSIM (0.624 vs. 0.670 on Re10K; 0.623 vs. 0.664 on ACID) and on FID on ACID (47.76 vs. 33.16—a substantial 44% gap). The quantitative discussion section acknowledges "minor geometric inconsistencies (–6.6% SSIM, +22% FID)," but the "minor" characterization understates the FID deficit on ACID, particularly since FID is one of the most standard distributional quality metrics for generative models. The paper's overall framing should be as competitive/parity, not outperforming. The abstract's more careful "comparable" language is more accurate.

- **Core motivation (incremental/causal generation advantage) is never experimentally validated.** The introduction's primary argument for preferring AR over diffusion is that causal generation enables imposing "a strictly causal structure along a camera path" and makes it possible "to incrementally extend and reuse existing generations when the trajectory changes." Yet all experiments evaluate fixed-length sequences generated in one go—the same joint-generation setting as diffusion baselines. No experiment tests trajectory extension, adaptation, or varying trajectory lengths. Without this, the paper shows an AR model that matches diffusion on diffusion's own terms, at 256×256 resolution, without demonstrated benefit from causality.

- **Key architectural contribution (camera autoencoder) lacks ablation.** The camera autoencoder is presented as one of three core modules, with specific design choices (Plücker ray encoding, geometric loss terms). Yet Section 4.3 ablates only token permutation strategy (Table 2) and tokenizer choice (Table 3). There is no ablation testing whether the camera autoencoder is needed vs. simpler alternatives (sinusoidal positional encoding, direct Plücker embedding without autoencoding), whether the geometric loss terms matter, or how accurately the autoencoder reconstructs Plücker coordinates. Since camera conditioning is what differentiates this from a generic AR video model, this gap leaves the core methodological contribution unevaluated.

### Minor

- **Missing FVD in permutation ablation (Table 2).** Table 2 reports PSNR, SSIM, LPIPS, and FID but omits FVD, the primary temporal-consistency metric—despite Table 3 including it. Since the permutation strategy is specifically designed to preserve temporal causality, evaluating its effect on temporal consistency is directly relevant, and its absence leaves the design principle empirically under-supported for temporal coherence.

- **Evaluation at 256×256 resolution with 17 frames.** This is relatively low by current NVS standards, which limits practical impact claims. The paper acknowledges tokenizer limitations in the Discussion, but does not discuss scalability to higher resolutions or longer trajectories.

- **No inference speed comparison.** Autoregressive generation at 256×256 with 17 frames of 32×32 latent tokens involves O(L×h×w) sequential steps, which is inherently slower than parallel denoising. The practical viability of the AR approach depends on this tradeoff, which is not quantified.

## Nice-to-Haves

- Test incremental trajectory extension (the core claimed advantage) by generating views for extended or modified trajectories and comparing against diffusion regeneration.
- Ablate the camera autoencoder design (comparing with simpler conditioning alternatives, evaluating reconstruction quality of the autoencoder itself).
- Report inference time comparisons with baselines.
- Evaluate at higher resolution (512×512) or longer trajectories (32+ frames) to test scalability and the claimed long-horizon advantage.

## Removed Points

- **Notation error in Equation 6 subscript.** Flagged by the harsh critic, but this is a formatting/display issue typical of PDF parsing artifacts, not an author error.
- **SEVA/ViewCrafter/RayZer excluded from DL3DV because of training data overlap.** The harsh critic raised this as making the zero-shot evaluation "less informative," but the exclusion is methodologically sound—comparing against methods that trained on the test data would be unfair, not the other way around. The baselines that remain (MotionCtrl, Genwarp, LVSM) still provide meaningful comparison.
- **Claim that "full perm" row (18.76 PSNR) vs "ours" (19.22) shows temporal ordering contributes marginally.** This is an interpretive point the harsh critic makes, but the 0.46 PSNR gap, combined with the larger raster-to-full-perm gap, actually does show meaningful improvement from preserving temporal order while enabling spatial permutation—both design choices contribute.
- **Strength claim about "lower error accumulation" from per-frame curves (Figure 6).** The harsh critic noted these lack confidence intervals. This is a standard visualization approach in the field; removing this strength entirely would be overly harsh. The curves do show a consistent trend across all three metrics.

## Novel Insights

The permutation ablation (Table 2) reveals an instructive decomposition: the raster-to-spatial-perm jump (16.29→ours 19.22, +2.93 PSNR) accounts for most of the improvement, while adding temporal permutation on top (full perm 18.76 vs. ours 19.22, +0.46) is more modest. This suggests that for the current sequence length (17 frames), spatial bidirectional context matters far more than temporal ordering—the very feature the paper claims as its primary motivation. This creates an interesting tension between the theoretical motivation and empirical evidence, suggesting that the value of causal temporal ordering may only emerge at longer horizons than tested (a hypothesis the paper should explicitly acknowledge).

## Suggestions

- Reframe claims from "outperforms" to "achieves competitive/parity results" in the Discussion and throughout, being explicit about the SSIM and FID tradeoffs rather than calling them "minor."
- Add an ablation comparing the camera autoencoder against simpler Plücker embedding alternatives and report reconstruction quality of the autoencoder.
- Add at least one experiment testing incremental generation (e.g., extending a trajectory by generating additional frames conditioned on previously generated ones) to validate the core AR motivation.
- Include FVD in the permutation ablation to directly measure the impact on temporal consistency.

## Score Calibration

I compared ARSS against the following anchors:

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LVSM (Oral) | QQBPWtvtcn.md | 7.67 | Stronger results (1.5-3.5 dB PSNR gains), more thorough ablations, both decoder-only architectures for NVS; ARSS is clearly weaker |
| NOVA (Poster) | JE9tCwe3lp.md | 6.50 | AR visual generation with cleaner methodology and better ablations; ARSS is weaker |
| NVS-Solver (Poster) | zDJf7fvdid.md | 6.0 | Simplier training-free approach to NVS with similar-level limitations; ARSS has more novelty but similar evaluation gaps |
| Diffusion² (Poster) | fectsEG2GU.md | 6.25 | Multi-view generation via composed diffusion with competitive results; ARSS has comparable methodology depth |
| 3D-free meets 3D priors (Reject) | VLuJL8cnGk.md | 5.0 | Overclaimed improvements, mixed metrics, poor writing; ARSS is better structured and more honest |
| Multi-view consistent generation (Reject) | Ns84n4NWh6.md | 3.5 | Very limited evaluation, cherry-picked datasets; ARSS is substantially stronger |

ARSS has real novelty (first AR model for camera-conditioned NVS) and a reasonable architectural contribution, but is hampered by mixed results relative to SEVA, an overclaim of superiority, and the absence of experiments testing its core motivation. It is above the clearly weak papers (~3.5-5.0) but below the solidly accepted ones (~6.0-6.5). The overclaim and missing causal-generation experiment push it below the borderline.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>