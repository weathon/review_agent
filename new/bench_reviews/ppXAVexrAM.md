Now I have all the context I need. Let me write the final consolidated review.

## Summary
ARSS proposes a GPT-style decoder-only autoregressive model for novel view synthesis from a single image, conditioned on a predefined camera trajectory. The framework combines three components: (1) a video tokenizer (VidTok with FSQ) for temporally coherent discrete tokenization, (2) a camera autoencoder that maps Plücker raymaps into 3D positional guidance tokens paired with each visual token, and (3) a hybrid spatial-permutation/temporal-causal token ordering strategy that enables a unidirectional transformer to exploit bidirectional spatial context within frames while maintaining temporal causality across frames. Experiments on RealEstate10K, ACID, and DL3DV show competitive PSNR/LPIPS compared to diffusion-based and feed-forward baselines.

## Strengths

1. **Clean and well-motivated architectural design.** The decomposition into video tokenizer, camera autoencoder, and hybrid permutation strategy is conceptually coherent. Each component addresses a specific challenge (temporal consistency, 3D camera conditioning, bidirectional spatial context), and the integration is natural. The hybrid permutation preserving temporal order while randomizing spatial order (Eq. 6) is a thoughtful adaptation of RandAR-style ideas to the multi-view setting.

2. **Strong ablation evidence for core design choices.** Table 2 (PSNR 16.29 → 18.76 → 19.22 across raster/full-perm/spatial-only) and Table 3 (FVD 137.68 → 52.56 for video vs. image tokenizer) convincingly validate the two most architecturally distinctive choices. These are not marginal improvements; the permutation strategy gains ~3 dB PSNR and the video tokenizer cuts FVD by 62%.

3. **Competitive results under modest training budgets.** ARSS achieves the highest PSNR and best LPIPS on RealEstate10K and ACID, and competitive FVD, despite training on 8 GPUs for 100K iterations at 256×256 — substantially fewer resources than diffusion-based baselines like SEVA, which the paper explicitly acknowledges.

4. **Error accumulation analysis (Figure 6).** Per-frame metric curves showing flatter degradation over camera trajectory steps are a genuine strength, directly addressing the motivation for causal/sequential generation and differentiating ARSS from one-shot diffusion methods.

## Weaknesses

### Major

1. **Overclaiming relative to SOTA, especially SEVA.** The abstract says "comparable to state-of-the-art," but Section 5 states "outperforms state-of-the-art methods leveraging diffusion models and transformers." Table 1 tells a more mixed story: ARSS is better than SEVA on PSNR (19.02 vs. 18.73) and LPIPS (0.269 vs. 0.349) on Re10K, but worse on SSIM (0.624 vs. 0.670) and FID (47.60 vs. 46.98). On ACID, SEVA dominates SSIM (0.664 vs. 0.623) and FID (33.16 vs. 47.76). The paper acknowledges "minor geometric inconsistencies" but still frames ARSS as winning overall. Given that SSIM and FID are well-established proxies for structural fidelity and distributional quality, respectively, the claim of "outperforms" is not justified. The paper should clearly state the trade-offs rather than cherry-picking the favorable metrics.

2. **Missing ablation on the camera autoencoder.** The camera autoencoder with its Plücker raymap encoding and geometric loss (Eq. 5) is a central contribution, yet no experiment compares it against simpler alternatives (e.g., direct Plücker coordinate injection, learned per-frame pose embeddings, or a simple MLP projection). Without this ablation, it is unclear whether the architectural complexity of the autoencoder is necessary or whether much simpler camera conditioning would suffice. The loss weights λ₁–λ₄ and the autoencoder's training data/procedure are also not specified, making it impossible to assess this component's contribution.

3. **The causal/world-model advantage is claimed but not empirically demonstrated.** The introduction motivates AR modeling by arguing that "it is desirable to process observations in a sequential and causal manner" and that diffusion methods "cannot incrementally extend and reuse existing generations when the trajectory changes." However, no experiment demonstrates trajectory adaptation, incremental extension, or reuse of past generations — the core advantages claimed for the AR paradigm. The error accumulation analysis (Figure 6) only shows per-frame quality along a fixed-length trajectory, not any compositional or adaptive generation. Without validating the central motivational claim, the paper's framing as a "world model" approach is overstated.

4. **No efficiency analysis for an autoregressive approach.** Generating 5×32×32 = 5,120 tokens sequentially via next-token prediction is inherently slower than parallel denoising in diffusion models. The paper claims parallel decoding is possible (Section 3.2.3) but provides zero inference time, throughput, or latency measurements. This is critical context: if the AR approach is orders of magnitude slower than diffusion baselines, the competitive quality metrics become less compelling. The paper also does not disclose model parameters, layer count, or hardware-specific timing.

5. **Low resolution and limited training scope constrain generality claims.** All experiments are at 256×256 resolution on RealEstate10K and ACID, with 100K training iterations. The paper acknowledges this limitation in the Discussion, but the "outperforms state-of-the-art" claim is made without acknowledging that this comparison may be atypical since baselines like SEVA operate at higher resolutions. This asymmetry in training resources and resolution makes the comparison inherently limited.

### Minor

- **Training/inference asymmetry not fully described.** During training, the model uses teacher-forced ground-truth frames; during inference, it autoregressively generates from only the first frame. While this is standard practice, the exact training procedure (e.g., whether curriculum strategies or scheduled sampling are used) is not detailed, making it hard to assess how well the model handles the train-inference distribution shift that affects AR models.

- **The video tokenizer is used off-the-shelf (VidTok), which was designed for video rather than multi-view data.** The paper acknowledges this limitation but does not quantify it, e.g., by measuring reconstruction quality on multi-view sequences versus generic video. Understanding the tokenizer's upper bound on generation quality would contextualize the results.

### Trivial

- Minor notation issues in Eq. (7) and surrounding definitions due to parsing artifacts.

## Nice-to-Haves

- Demonstrating trajectory adaptation (e.g., extending or modifying a camera path mid-generation) would directly validate the key motivational claim about causal world modeling advantages.
- Reporting inference latency and comparing throughput with diffusion baselines would address a major practical concern for AR approaches.
- Ablating the camera autoencoder against simpler conditioning schemes (direct embedding, MLP) to isolate its contribution.
- Experiments at 512×512 or higher resolution to demonstrate scaling capability.
- 3D consistency metrics (e.g., depth error, reprojection error) beyond per-frame 2D metrics to better assess multi-view coherence.

## Removed Points

- **"Baselines not evaluated in standard configurations or retrained" (Harsh Critic #1):** The paper explicitly states that SEVA benefits from more resources and data, and that some baselines are excluded from DL3DV due to train/test contamination (appropriate). The training asymmetry favors the baselines (SEVA has more resources), not the authors' method, so this is not an unfair comparison against ARSS. Removed per the hard rule against criticizing unfair comparisons that favor baselines.

- **"Not clearly stated whether training uses single-image conditioning or multi-view" (Harsh Critic #2):** The paper clearly describes in Section 3.2.3 ("the first frame is the input, so the corresponding visual and camera tokens are always visible to the subsequent tokens") and Section 4.1 ("the first 32×32 tokens are the input tokens and their orders would not be permuted") that only the first frame is the conditioning input. Teacher-forcing during training is standard AR practice and does not constitute "multi-view conditioning" in the misleading sense.

- **"Missing related works" (Neutral Reviewer #5):** Per the hard rules, I do not flag missing citations as I cannot verify their existence. The paper cites relevant AR generation works (RandAR, LlamaGen, etc.) and diffusion NVS methods.

- **"No confidence intervals / variance across runs" (Harsh Critic):** Single-run evaluation without variance reporting is standard practice for large-scale generative models in this community. This is a nice-to-have, not a core flaw.

- **"Parallel decoding not evaluated" (Spark Reviewer):** The paper mentions this as an "advantage" in one sentence. It's fair to note as a limitation, but it is not a core claim of the paper — the main claim is about quality and causal structure. Moved to Nice-to-Have.

- **"Failure case analysis" (Neutral Reviewer):** While useful, the absence of failure case visualization is not uncommon in the field. Moved to Nice-to-Have.

- **"Missing baselines like ZeroNVS, Sin3DGM, feed-forward 3D reconstruction methods" (Human Finder):** The paper already compares against 6 baselines spanning both diffusion and non-diffusion methods. Adding more baselines is a Nice-to-Have, not a core flaw. The paper's comparison set is reasonable for its scope.

## Novel Insights

The most interesting empirical finding is how dramatically the token ordering matters: raster-scan ordering gets 16.29 PSNR while spatial-permutation-with-temporal-causality gets 19.22 — nearly a 3 dB jump. This suggests that for AR models applied to multi-view data, the ordering question is not a detail but the primary design lever, and that the bidirectional nature of spatial data in 2D frames must be explicitly addressed. The FVD gap between video (52.56) and image (137.68) tokenization also highlights that temporal consistency is largely a tokenization problem, not just a generation problem.

## Suggestions

1. **Retitle and reframe claims**: Change "outperforms state-of-the-art" to "achieves competitive results" and present PSNR/LPIPS advantages alongside SSIM/FID trade-offs transparently.
2. **Add a camera conditioning ablation**: Replace the camera autoencoder with a simple Plücker embedding or MLP projection to quantify its contribution.
3. **Report inference time**: Provide wall-clock timings for generating a full 16-frame sequence, compared against diffusion baselines.
4. **Demonstrate trajectory extension**: Generate a sequence, then extend it with additional cameras, to validate the causal/incremental generation motivation.
5. **Specify model parameters and training details**: Report total parameters, number of transformer layers/heads, and context length for reproducibility.

## Score and Decision

**Calibration comparison:**
- **AR-1-to-3** (most similar task: autoregressive NVS): All scores 5, withdrawn. Weaknesses included error accumulation concerns, limited evaluation, unclear advantages over baselines.
- **LVSM** (decoder-only transformer NVS): Scores 8,8,8,6,8,8, accepted as Oral. Strong quantitative results, clear scalability, but still had weaknesses in qualitative consistency and missing efficiency analysis.
- **NOVA** (autoregressive video generation): Scores 6,6,6,8, accepted as Poster. Novel AR video framework with moderate novelty, reasonable but mixed results.
- **GST** (autoregressive view prediction with camera tokenization): Scores 8,6,8,3, accepted as Poster. Novel camera tokenization direction, but limited evaluation scope.

ARSS sits between AR-1-to-3 (rejected) and NOVA/GST (accepted posters). Unlike AR-1-to-3, ARSS has stronger baselines (SEVA, etc.), better quantitative results, and meaningful ablations. Unlike NOVA/GST, ARSS does not clearly demonstrate the practical advantages it claims (causal world modeling, trajectory extension), and the overclaiming relative to SOTA is a substantive concern. The missing camera autoencoder ablation and efficiency analysis also limit the evidential completeness. However, the framework design is solid and the ablations that do exist are convincing.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>