Let me verify some specific claims from the paper text given in the prompt.

The paper text is in the user message. Let me check key points:

1. The "minor exception" claim - the paper says "with the minor exception of a narrow margin on the Subject-Canny task" regarding controllability. Let me verify: In Table 1, Subject-Canny, the F1 score is 0.414 for Ours vs 0.551 for UniCombine. This is a 25% relative drop, not really a "narrow margin."

2. KSA keyword selection - The paper says "the keyword set K typically contains just 1 to 2 tokens" but doesn't explain how keywords are selected.

3. Condition Cache - The paper states "The Key and Value projections for all condition tokens are computed only once in the first denoising step and are then cached and reused for all subsequent steps." This is a strong assumption that condition tokens are static.

4. Early-timestep sampling - Figure 11 only shows qualitative results.

5. PAA ablation (Figure 9) - Only shows visual results and efficiency, no quantitative quality metrics.

6. PixelPonder is cited in related work but not compared against.

7. The 10× speedup claim - the paper says "scenarios with a high number of conditions" but Table 1 only has 2-condition tasks.

Now let me synthesize my review.

**Calibration references:**
- PAB (Pyramid Attention Broadcast): scores 8/8/6/6 → poster, similar approach to efficient attention but in video, has strong speedups (10.5×), simpler method
- Efficient-vDiT: scores 6/5/6/6 → reject, similar sparse attention approach but has issues with ablation and comparison, overclaims
- ToCa (Token-wise Caching): scores 6/6/6/6 → poster, similar efficiency focus for DiTs
- Ctrl-Adapter: scores 8/6/6/8 → oral, multi-condition control for diffusion, much more thorough evaluation

This paper has:
- A real and important problem (quadratic cost of multi-condition attention in DiTs)
- Strong efficiency numbers (up to 10× speedup)
- But significant weaknesses: overclaimed "no quality loss", limited evaluation scope, structural rigidity of PAA, weak ablations, missing comparison with PixelPonder, conflated efficiency contributions

Compared to Efficient-vDiT (rejected, 5.75 avg), this paper has better motivation but similar issues with overclaiming and insufficient ablation. Compared to PAB (accepted poster, 7 avg), this paper has weaker evaluation and more structural assumptions. I'd place it around 5-5.5, in the borderline reject range, because while the direction is promising, the claims significantly outpace the evidence, and the core structural assumptions (PAA's strict 1:1, KSA's mask reuse, condition cache) are not rigorously validated.

Let me finalize the review now.</think>## Summary

The paper proposes PKA (Patch-wise and Keyword-Aware Attention), an efficient attention mechanism for multi-condition control in Diffusion Transformers (DiTs). PKA decomposes full attention into two specialized modules: Position-Aligned Attention (PAA) for spatial conditions, which restricts attention to one-to-one spatially aligned patches, and Keyword-Scoped Attention (KSA) for subject conditions, which confines attention to keyword-activated image regions via a reusable binary mask. Combined with a Condition Cache (reusing condition KV across denoising steps) and an early-timestep sampling strategy, the method achieves up to 10× inference speedup and 5.12× attention VRAM reduction.

## Strengths

- **Addresses a real and important bottleneck.** The quadratic cost of "concatenate-and-attend" in multi-condition DiTs is well-identified, and the empirical attention-pattern analysis (Figures 2, 3) provides clear motivation for the proposed sparsity decomposition.

- **Strong efficiency gains are convincingly demonstrated.** The speedup curves (Figure 7) and VRAM reduction (Figure 8) show clear and substantial benefits that scale with the number of conditions, directly tackling the core scalability problem.

- **Principled condition-type-specific decomposition.** Separating spatial-aligned (PAA) from subject-driven (KSA) attention is a natural and elegant design that leverages genuine structural differences in how conditions interact with image tokens. The Condition Cache is a simple but effective engineering contribution that leverages the static nature of condition tokens.

- **Competitive generation quality on most metrics.** Table 1 shows the method matches or exceeds baselines on FID, SSIM, CLIP-I, DINOv2, and depth MSE, demonstrating that the aggressive attention simplification does not catastrophically degrade quality.

## Weaknesses

### Major:

- **The "no quality loss" claim is overstated given the evidence.** The abstract and conclusion state quality is "maintained or improved," but on Subject-Canny controllability (F1), the method drops from 0.551 (UniCombine) to 0.414—a 25% relative regression. The paper dismisses this as "a minor exception," but this is a substantial gap for one of only three evaluated tasks. The claim that quality is maintained should be qualified. More broadly, evaluating on only three 2-condition tasks derived from a single curated dataset is narrow support for the paper's sweeping claims about "maintaining or improving generative quality."

- **PAA's strict one-to-one spatial alignment is a structural restriction with insufficient justification.** Equation 2 restricts each image token to attend only to the spatial condition token at the *exact same position*. While Figure 2 shows diagonal attention patterns, real spatial conditions (edges, poses, depth maps) often require local-but-not-identical context—e.g., at object boundaries, thin structures, or in the presence of any spatial misalignment between condition and image latents. The PAA ablation (Figure 9) reports only latency and VRAM, not quantitative quality metrics, and compares only against SWA (which it already outperforms in efficiency). No ablation tests PAA vs. a small local window (e.g., 3×3 neighboring patches) on quality, which is the critical question. Additionally, there is no evaluation on sparse layout conditions (bounding boxes, keypoints) where the one-to-one assumption would be most problematic.

- **KSA's assumptions about keyword-based mask generation and temporal mask reuse are weakly validated.** The method depends on (1) reliably selecting "keyword" tokens (the paper says "typically 1 to 2" but never specifies how they are selected—heuristically? via POS tagging? manually?), and (2) reusing a binary mask from step t at step t+1. Temporal consistency of attention masks across denoising steps is cited without quantitative evidence. No mask drift analysis is provided—the ablation (Figure 10) shows only 3 ε values on a single example. Furthermore, no ablation isolates KSA's effect on subject-consistency metrics (CLIP-I, DINOv2) vs. the other components.

- **Efficiency contributions are not disentangled.** The headline speedup numbers (up to 10×) conflate PAA, KSA, and Condition Cache. The paper does not provide an ablation showing how much of the speedup comes from the Condition Cache alone (which is architecturally independent and could be combined with full attention). Without isolating these contributions, it is unclear how much of the gain is due to the novel attention sparsity vs. the straightforward KV caching strategy.

### Minor:

- **Early-timestep sampling lacks quantitative validation.** Figure 11 shows only qualitative visual results for different μ values. No FID, controllability, or consistency metrics are reported for this component, making the claim that it "accelerates convergence and enhances control fidelity" unsupported beyond visual impression. The perturbation study (Figure 5) lacks methodological detail (protocol, sample count, which conditions were perturbed).

- **Experimental scope is narrow.** Only FLUX.1 is tested; all tasks use Canny and Depth as spatial conditions from a curated subset of Subject200K where every caption has a "descriptive keyword." The latter biases the evaluation in favor of KSA. No evaluation generalizes to other DiTs, different resolutions, sparse layout conditions, or out-of-domain prompts.

- **Missing details on KSA formulation.** The `Norm` operation in Eq. 3 is unspecified (softmax? min-max normalization?), and the thresholding scheme (per-token vs. global) is unclear. These implementation details affect reproducibility and the method's behavior.

## Nice-to-Haves

- Ablation testing PAA against a small local window (e.g., 3×3) to quantify the quality/efficiency tradeoff of strict one-to-one alignment.

- Quantitative metrics (FID, CLIP-I, F1, etc.) for the early-timestep sampling ablation across multiple μ and δ values.

- Evaluation on a 3+ condition task (e.g., Subject-Canny-Depth-to-Image) to validate the scalability claims that motivate the entire paper, since all quality evaluations only use 2 conditions.

- Comparison with PixelPonder (cited in Section 2.2 as directly relevant work on efficient multi-condition DiTs, but absent from experiments).

- Failure case analysis: show examples where PAA's strict alignment fails and where KSA's mask incorrectly excludes subject regions.

## Removed Points

- **PixelPonder comparison absence.** PixelPonder is cited in related work but not compared against in experiments. While this would strengthen the paper, PixelPonder addresses a different sub-problem (dynamic token pruning vs. condition-specific sparse attention), and the paper already compares against two relevant baselines in the same architectural family. Flagged as nice-to-have rather than a fatal omission.

- **Condition KV caching may introduce approximation errors.** The reviewer claimed this is a "strong assumption" that "could accumulate errors." However, the Condition Cache operates on condition tokens that perform self-attention only among themselves—they do not receive gradient signals from noisy image tokens in the original MMA either (condition tokens attend to all tokens including image tokens). The approximation introduced by caching is that condition KV projections don't update across steps; but since conditions are static inputs, this is architecturally sound. The weakness about not quantifying the cache's contribution is kept (it's about attribution, not validity).

- **No user study / perceptual evaluation.** Standard for this type of efficiency-focused systems paper; FID, CLIP, and SSIM are the community's accepted metrics. Demanding a user study is scope creep.

- **Missing details on LoRA rank, learning rate, train/test split sizes.** These are minor reproducibility details that are standard to omit from submissions and not central to the contribution.

- **No standard text-to-image benchmarks (COCO, etc.).** The paper specifically addresses multi-condition generation, and evaluating on standard T2I benchmarks would be out of scope.

- **Formatting nitpicks / minor citation style issues.** Removed per guidelines.

## Novel Insights

The paper's key insight—that multi-condition attention redundancy in DiTs is *conditionally typed* (spatial conditions exhibit diagonal locality, subject conditions exhibit keyword-driven sparsity)—is genuinely novel and well-motivated by the attention visualizations. This decomposition enables condition-type-specific sparsification that is far more aggressive than generic sparse attention mechanisms. However, the observation that this decomposition is *safe* (i.e., does not sacrifice quality) is only partially validated: the evidence works for some metrics and tasks but fails on others (F1 for Canny), and the structural assumptions underlying each sparsity pattern (perfect spatial alignment for PAA, keyword-triggered masks for KSA) have untested failure modes.

## Suggestions

- **Separate Condition Cache from PAA/KSA in experiments.** Report speedup and quality with Condition Cache + full attention vs. Condition Cache + PKA to properly attribute gains.

- **Add a PAA vs. local-window ablation with quality metrics.** Even a 3×1 or 3×3 local window comparison would clarify whether strict one-to-one alignment sacrifices spatial faithfulness (measured by F1/MSE) on tasks like Canny-to-Image.

- **Clarify keyword selection and evaluate KSA in isolation.** Explain how keywords are selected and report CLIP-I/DINOv2 with and without KSA to isolate its contribution.

- **Report quantitative results for the early-timestep sampling ablation** (FID, controllability, consistency at different μ values) rather than only qualitative figures.

- **Acknowledge the F1 regression on Subject-Canny** as a real trade-off and investigate whether PAA's restriction is the cause.

## Score and Decision

**Calibration:**
- PAB (Pyramid Attention Broadcast): scores 8/8/6/6 (avg 7), accepted poster. Similar efficient DiT attention with strong speedups (~10.5×), but training-free and more thoroughly evaluated across models.
- Efficient-vDiT: scores 6/5/6/6 (avg 5.75), rejected. Similar sparse attention for DiTs with notable speedups, but overclaimed and had insufficient ablation and missing baselines.
- ToCa (Token-wise Caching): scores 6/6/6/6 (avg 6), accepted poster. Efficient DiT method with moderate speedups (~2×), solid but incremental.
- Ctrl-Adapter: scores 8/6/6/8 (avg 7), accepted oral. Multi-condition control for diffusion models, extensive evaluation.

This paper has a stronger motivation than Efficient-vDiT (the condition-typed decomposition insight is genuinely novel), but shares similar issues: overclaimed "no quality loss" narrative, insufficient ablation, and missing key comparisons. The efficiency gains are real and substantial, but the evidence for "maintaining quality" is narrower than claimed, and the structural assumptions (PAA's one-to-one alignment, KSA's keyword masks, Condition Cache's step-independence) are not rigorously validated. This places it below PAB and ToCa (which have cleaner claims relative to their evidence) but above Efficient-vDiT (which had more fundamental flaws in its attribution of speedup contributions). The paper is a promising direction that needs more honest characterization of trade-offs and more thorough ablation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>