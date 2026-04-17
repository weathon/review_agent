## Summary

ARSS proposes a decoder-only autoregressive (GPT-style) model for novel view synthesis from a single image, conditioned on a predefined camera trajectory. The framework combines three modules: (1) a video tokenizer for temporally consistent discrete token sequences, (2) a camera autoencoder that converts Plücker raymaps into 3D positional guidance tokens, and (3) a hybrid permutation strategy that randomly shuffles spatial token order within frames while preserving temporal causality. Results on RealEstate10K, ACID, and DL3DV show competitive-but-mixed performance compared to diffusion-based and transformer-based baselines.

## Strengths

1. **Meaningful problem framing and first-of-kind system**: Applying decoder-only AR models to NVS with camera control is a genuinely novel direction. The paper is the first to assemble this particular pipeline, and the motivation for causal, sequential generation is clearly articulated.

2. **Strong and informative ablations on tokenization and permutation**: Table 2 clearly demonstrates the advantage of spatial-only permutation over raster (16.29→19.22 PSNR) and full permutation (18.76→19.22 PSNR). Table 3 shows dramatic FVD improvement with video tokenization (137.68→52.56). These ablations provide useful empirical evidence for the design choices.

3. **Competitive per-pixel and perceptual metrics on some benchmarks**: On RealEstate10K, ARSS achieves the best PSNR (19.02) and LPIPS (0.269) among compared methods, demonstrating that the AR paradigm can produce high-fidelity results in some dimensions.

4. **Principled 3D camera conditioning**: The Plücker raymap encoding with geometry-constrained loss (Eq. 5) is a well-motivated design, and pairing each visual token with a camera token is an elegant way to inject 3D awareness into the AR sequence.

5. **Error accumulation analysis**: The per-frame metric curves (Figure 6) provide useful insight into how quality degrades along the trajectory, which is particularly relevant for the AR generation paradigm.

## Weaknesses

### Major:

1. **Core motivation (causal/sequential world modeling) is empirically unvalidated**: The paper's central selling point — that AR models enable incremental extension, trajectory modification, and reuse of accumulated knowledge (stated explicitly in the Introduction: "incrementally extend and reuse existing generations when the trajectory changes") — receives zero experimental validation. All experiments use fixed 17-frame trajectories; no experiment tests trajectory extension, modification, or reuse. Without this, the paper's primary architectural advantage over diffusion baselines remains entirely theoretical, and the method functions as just another NVS model.

2. **Overclaimed "out-performs state-of-the-art" narrative**: The abstract states the method "out-performs current state-of-the-art methods," but Table 1 shows mixed results. SEVA outperforms ARSS on SSIM (0.670 vs. 0.624 on Re10K; 0.664 vs. 0.623 on ACID) and FID (46.98 vs. 47.60 on Re10K; 33.16 vs. 47.76 on ACID). The FID gap on ACID is substantial (33.16 vs. 47.76, a ~44% increase). The paper briefly acknowledges this in the text ("minor geometric inconsistencies") but the framing throughout ("out-performs," "state-of-the-art") is misleading given these tradeoffs.

3. **Camera autoencoder contribution is unverified**: One of the paper's two main conceptual contributions — the camera autoencoder producing Plücker-based 3D positional tokens — has no ablation. There is no experiment comparing against simpler alternatives (raw pose embeddings, 2D coordinate embeddings, or no camera tokens), no evaluation of how well the autoencoder reconstructs camera information, and no perturbation study showing sensitivity to camera conditioning. Without this, the actual impact of this module on the final results remains unclear.

4. **AR factorization itself is not cleanly validated**: The ablations compare only within the AR framework (different permutation strategies, different tokenizers). There is no comparison against a bidirectional (non-causal) transformer using the same video tokenizer, which would directly test whether the causal AR constraint helps or hurts. Since the spatial permutation strategy already partially breaks strict bidirectionality, the claim that "decoder-only AR" is the right paradigm for this task is not substantiated. The improvements could primarily come from the video tokenizer rather than the AR factorization itself.

### Minor:

- **No evaluation of inference efficiency**: AR next-token prediction over 5120 tokens per sequence is inherently slow. The paper mentions "parallel decoding" as an advantage of random spatial permutation (Section 3.2.3) but provides no runtime measurements, throughput comparisons, or even a concrete parallel decoding algorithm. This is a notable gap given that the claimed advantage over diffusion models includes efficiency-related considerations.

- **Low resolution (256×256)**: The method operates at 256×256, notably below many competing diffusion-based NVS methods. The authors acknowledge this but it limits practical applicability and comparability.

- **Limited novelty of individual components**: The video tokenizer is adopted from VidTok, the backbone from LlamaGen, and the spatial permutation from RandAR/Open-MAGVIT2. The contribution is in combining these for NVS with camera control, which is valid but incremental in terms of methodology.

### Trivial:

- The discussion section is brief and does not analyze when or why the method fails, despite the FID degradation on ACID.

## Nice-to-Haves

- **Trajectory extension experiment**: Demonstrating that ARSS can generate beyond 17 frames or modify trajectories mid-generation would substantiate the paper's main motivation and be its most distinctive experimental result.
- **Bidirectional transformer baseline with the same tokenizer**: This would cleanly test whether the AR factorization itself provides benefits beyond tokenization.
- **Camera autoencoder ablation**: Comparing with simpler conditioning schemes would establish whether the Plücker-based encoding is necessary.
- **Higher resolution evaluation or efficiency benchmarks**: Would clarify the practical tradeoffs of the AR approach.
- **Failure case analysis**: Given the SSIM/FID gaps versus SEVA, understanding geometric inconsistencies would be valuable.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unfair comparison because baselines use more data/compute (SEVA)":** The paper itself notes that "SEVA benefits from large-scale, high-resolution training data and heavy computational resources, whereas our approach attains competitive performance without such requirements." This is a factual observation about their respective scale, not an unfair comparison. However, it does mean the paper should frame results as "competitive despite less resources" rather than "out-performs."

- **"Missing baselines (ZeroNVS, other methods) on DL3DV":** Per the hard rules, we do not flag missing related works. The paper's omission of SEVA/ViewCrafter/RayZer on DL3DV is actually defensible since those models were trained on DL3DV data — it avoids an unfair comparison in the other direction. However, the paper should not use DL3DV results to claim superiority over those specific omitted methods.

- **"Results from original papers vs. reimplemented (fairness of baseline numbers)":** This is a reproducibility nitpick that falls under the hard rules for removal (undisclosed hyperparameters/benchmarking details). The standard practice in this field is to report numbers from original papers.

- **"Missing confidence intervals/standard deviations":** Single-run evaluation is the norm in this community for large-scale generation; requesting statistical significance is beyond standard practice.

## Novel Insights

The clearest novel observation from this work is that spatial-only token permutation with preserved temporal ordering significantly outperforms both raster-scan and fully-random permutation for multi-view AR generation (PSNR: 16.29→18.76→19.22). This validates the intuition that temporal causality is essential for view sequence coherence while bidirectional spatial context within frames should be exploited — a finding that generalizes beyond this specific architecture to any AR approach on multi-frame generation tasks.

## Suggestions

1. Revise all claims of "out-performs state-of-the-art" to "achieves competitive results" and clearly acknowledge the SSIM/FID tradeoffs against SEVA in the abstract and conclusions.
2. Add an experiment demonstrating trajectory extension (e.g., generating 17 frames, then continuing to 34 frames without re-encoding) — this is the single most important missing validation for the paper's motivating claim.
3. Ablate the camera autoencoder by replacing it with a simpler pose embedding (e.g., sinusoidal encoding of camera parameters) to isolate its contribution.
4. Add a bidirectional transformer baseline on the same video tokens to test whether the AR framework specifically adds value.
5. Report wall-clock inference time per sequence and compare against diffusion baselines.

## Score and Decision

**Calibration papers:**
- AR-1-to-3 (Reject, scores 5,5,5,5): Similar AR-for-NVS idea but weaker execution and evaluation. ARSS has better ablations and stronger baselines.
- Fluid (Accept Poster, scores 6,6,5,6): Limited novelty (combining existing techniques), solid empirical study but acknowledged as incremental. ARSS has a similar novelty profile.
- GST (Accept Poster, scores 8,6,8,3): Novel AR framework for spatial tasks with reasonable but not SOTA results.

ARSS's main issue is that its core motivation (causal sequential generation) is unvalidated, and its claims overclaim relative to mixed empirical results. The system is well-designed with informative ablations, but the AR paradigm's advantage over alternatives is not convincingly demonstrated. This is a promising direction that needs significant additional validation. Placing it below papers like GST (which validated its core claim) and Fluid (which had cleaner framing), at a level above AR-1-to-3 (which had worse execution).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>