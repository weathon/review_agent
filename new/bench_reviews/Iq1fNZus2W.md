Now I have a thorough understanding of the paper. Let me compose the final review.

## Summary

The paper proposes PKA (Patch-wise and Keyword-Aware Attention), a decomposed attention framework for efficient multi-condition control in Diffusion Transformers (DiTs). PKA exploits observed sparsity in multi-condition attention patterns via two specialized modules: Position-Aligned Attention (PAA) that restricts spatial condition attention to one-to-one position-aligned pairs (O(N²)→O(N)), and Keyword-Scoped Attention (KSA) that confines subject-driven attention to keyword-relevant image regions via a learned mask. A Condition Cache reuses condition KV pairs across denoising steps, and an early-timestep sampling strategy concentrates training on steps where visual conditions matter most. The method achieves up to 10× inference speedup and 5.12× attention VRAM reduction over UniCombine's full attention while maintaining or improving generation quality.

## Strengths

- **Strong empirical motivation via attention sparsity analysis**: Figures 2 and 3 directly visualize that spatial conditions concentrate along the diagonal while subject conditions activate only keyword-relevant regions. This concrete sparsity analysis provides principled justification for the entire design philosophy and goes beyond merely asserting that full attention is expensive.

- **Substantial efficiency gains with quality preservation**: Table 1 shows PKA achieves best FID and SSIM across all three tasks and best subject consistency (CLIP-I, DINOv2) on subject-driven tasks, while Figures 7–8 demonstrate up to 10× speedup and 5.12× VRAM reduction. Achieving better quality at drastically lower cost is a compelling result.

- **Principled complexity reduction for spatial conditions**: PAA's O(N²)→O(N) reduction via one-to-one position-aligned computation directly exploits the diagonal structure in Figure 2. The ablation in Figure 9 confirms PAA (13.63s, 237MB) is more efficient than even the smallest sliding window (14.00s, 276MB).

- **KSA provides a tunable efficiency-quality knob**: Figure 10 systematically varies the mask threshold ε, showing graceful degradation (latency drops from 16.99s to 15.26s, VRAM from 368MB to 242MB at ε=0.4 with only subtle detail differences), demonstrating robustness rather than brittle failure.

- **Condition Cache is a practical and non-obvious design**: Restricting condition tokens to self-attention only (Figure 4b) enables KV caching across denoising steps (Figure 4a), a design made possible by the decomposed attention structure itself.

- **Perturbation analysis motivating early-timestep sampling**: Figure 5 cleanly shows high-to-low perturbation degrades SSIM much faster, empirically establishing that visual conditions matter most at early steps, which directly motivates the shifted logit-normal sampling validated in Figure 11.

## Weaknesses

### Fatal
None.

### Major

- **Headline 10× speedup claim is disconnected from quality validation**: The 10× speedup occurs at 16 conditions (Figure 7), but all quality evaluations (Table 1, Figure 6) are conducted with only 2–3 conditions (Subject-Canny, Subject-Depth, Canny-Depth). At 4 conditions, speedup drops to 3.9×. The paper never demonstrates that PKA generates acceptable images with 8 or 16 simultaneous conditions. The paper's most prominent efficiency claim is asserted in a regime whose output quality is entirely unverified, while the regime where quality is verified shows more modest gains. This disconnect matters because the "up to 10×" framing prominently headlines the abstract and contributions.

- **Ablation studies lack quantitative metrics**: All three ablation studies (Figures 9, 10, 11) are purely visual with no quantitative metrics (FID, CLIP-I, controllability scores). This is a significant gap: for PAA vs. full attention vs. sliding window (Figure 9), there is no quality measurement to complement the latency/VRAM numbers. For the early-timestep sampling (Figure 11), there is no quantitative comparison of different (μ, δ) settings at matched iteration counts. Without quantitative ablation metrics, it is impossible to assess whether the efficiency gains come at an unmeasured quality cost.

### Minor

- **Condition Cache changes the conditioning mechanism without explicit validation**: Section 3.2 states condition KV pairs "are computed only once in the first denoising step and are then cached and reused for all subsequent steps." In full attention (Eq. 1), condition representations evolve as the noisy image changes across steps. Caching KV pairs from step 0 effectively freezes condition representations. While this is a reasonable approximation given the architectural separation, the paper provides no direct comparison of cached vs. recomputed KV to quantify the approximation cost. The overall Table 1 results indirectly validate this choice, but an explicit cache-vs-recompute ablation would strengthen confidence.

- **PAA's one-to-one spatial alignment is a strong restriction**: Equation 2 constrains each noisy image token to attend only to the spatial condition token at the exact same coordinate, eliminating all non-local spatial conditioning. The justification is Figure 2's diagonal pattern, but even there there is visible off-diagonal activation, and no analysis is provided of how common near-diagonal patterns are across conditions, layers, or timesteps. The paper notes PAA "is both intuitive and efficient" but does not acknowledge this as a potentially lossy approximation for conditions that could benefit from spatially distant context.

- **KSA keyword selection is underspecified**: Equation 3 uses keyword tokens 𝕂 (typically 1–2 tokens) but does not specify how these are selected — manually per-prompt or automatically extracted. The training data is curated to "ensure each image caption contains a descriptive keyword" (Section 4.1), which suggests manual curation that may limit generalizability. Additionally, mask reuse from step t at step t+1 is justified by "temporal consistency" but no sensitivity analysis for mask freshness is provided.

- **Efficiency metrics are attention-module only**: Figures 7 and 8 measure only the attention module's VRAM and latency, not full-model metrics. The 10× and 5.12× figures likely translate to much smaller fractions of total model inference cost.

- **No variance or statistical significance reported**: Table 1 reports single numbers with no error bars, confidence intervals, or repeated trials, making it impossible to assess whether reported differences are statistically meaningful. The Subject-Canny task shows PKA losing on F1 (0.414 vs. 0.551) and CLIP-T (0.349 vs. 0.352), which the paper dismisses as "narrow margin" but which may be within noise.

### Trivial
None.

## Nice-to-Haves

- Quality evaluation at 4+ simultaneous conditions to validate the higher-condition-count efficiency claims.
- Quantitative ablation metrics (FID, CLIP-I, controllability) for PAA vs. full attention vs. sliding window.
- Direct comparison of cached vs. recomputed condition KV pairs to explicitly validate the Condition Cache.
- Failure case examples showing where PKA underperforms, helping readers understand practical limitations.
- Full-model (end-to-end) efficiency metrics including VAE encode/decode time and total GPU memory.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "PAA's softmax over a single key isn't really attention, it's learned gating"**: While technically correct that Eq. 2 computes a scalar softmax over a single key-value pair (making it functionally a feature modulation), this is a presentation/naming criticism rather than a substantive flaw. The paper clearly defines the operation and its O(N) complexity. Calling it "attention" is consistent with naming conventions for position-aligned operations in the literature.

- **Harsh Critic: "baseline fairness — different training for OminiControl2/UniCombine"**: The paper states they fine-tune with LoRA for 20K iterations on their curated Subject200K subset (Section 4.1) and explicitly says "To ensure a fair comparison." The training configuration for baselines is a valid concern but the paper provides the same base model and data; the asymmetry, if any, is minor and standard for the field. This is also a nitpick about reproducibility details that are handled similarly across the field.

- **Harsh Critic: "missing related works"**: Per the rules, no criticism about missing related works is included.

- **Harsh Critic: "formatting/notation nitpicks"**: Various formatting-related points (e.g., Eq. notation concerns) are removed per the rules.

- **Harsh Critic: "video generation claim is speculative"**: The conclusion's brief mention of extending to video is standard future work speculation, not a core claim being evaluated.

- **Strength Finder: "Clean and complete ablation suite"**: This strength is weakened by the verified weakness that all ablations are visual-only with no quantitative metrics. The ablation structure is good, but the execution is incomplete.

## Novel Insights

The paper reveals a useful structural insight: multi-condition attention sparsity manifests differently by condition type (diagonal for spatial, keyword-localized for subject), and each sparsity pattern requires a different efficient approximation strategy. This type-specific decomposition — rather than applying a single sparsification method uniformly — is a design principle that could generalize beyond diffusion models to any multi-modal transformer processing heterogeneous token types with distinct interaction patterns.

## Suggestions

- Add quantitative ablation tables with FID/CLIP-I/controllability metrics for each component, even if only on a subset of the evaluation tasks.
- Reframe the "up to 10×" claim more carefully — present it as "3.9× at 4 conditions (validated quality) and up to 10× at 16 conditions (efficiency-only)" to avoid the disconnect between quality and efficiency claims.
- Run a simple cache-vs-recompute sweep (e.g., recomputing every K steps) to quantify the Condition Cache approximation cost.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| **SANA** (N8Oj1XhtYZ) | 8.5 (Oral) | Linear DiT with deep compression AE; more complete system with stronger empirical validation across resolutions. PKA targets a narrower but important problem (multi-condition control) and has less comprehensive evaluation. Below this. |
| **PT-DiT** (lTrrnNdkOX) | 6.4 (Accept Poster) | Proxy-tokenized DiT reducing global attention redundancy; similar motivation (attention sparsity → efficiency), but evaluated on more tasks (T2I, T2V, T2MV) with quantitative ablations. PKA is comparable in motivation but weaker in ablation rigor. Below or similar. |
| **FasterCache** (W49UjcpGxx) | 5.5 (Accept Poster) | Caching features across denoising steps; similar cache-based efficiency idea but less novel than PKA's decomposed attention. PKA has stronger architectural novelty but weaker ablation support. |
| **LinFusion** (D2as3jDmRA) | 6.25 (Reject at this venue) | Linear attention replacement for SD; overclaimed but strong compatibility experiments. PKA's efficiency story is better grounded in sparsity analysis but also overclaims. |
| **SparseDM** (3kADTLbKmm) | 4.0 (Reject) | 50% MACs reduction but only 1.2× GPU speedup; efficiency claims underwhelming. PKA's efficiency gains are much more convincing (real latency speedup), but has the quality-at-high-condition-count gap. |
| **FlashSampling** (V4Xs283LHH) | 2.5 (Reject) | 384% speedup claim disconnected from practical impact; softmax isn't the bottleneck. PKA's efficiency claims are far better grounded — attention really IS the bottleneck for multi-condition DiTs. Well above this. |
| **Strided Transformers** (x7kyIVdtSz) | 2.33 (Reject) | Only theoretical speedup, no real-world evaluation. PKA measures real latency/VRAM, so well above this. |

PKA sits in a space where the core idea is well-motivated and the efficiency results are real, but the quality evaluation doesn't fully support the headline claims (especially the 10× speedup at untested condition counts) and ablations lack quantitative rigor. This is above the FlashSampling/Strided Transformer tier (disconnected speedup claims) and above SparseDM (efficiency claims don't translate). It's below SANA (complete system, comprehensive evaluation). It's somewhat below PT-DiT (similar novelty but better evaluated). The combination of real contributions (sparsity analysis, decomposed attention, practical efficiency gains) with the major evaluation gaps puts it in the borderline range, slightly below acceptance. The 3.9× validated speedup with quality preservation is a solid result, but the overclaiming on 10× and lack of quantitative ablations are meaningful concerns.

## Score and Decision

Score: 5.5  
Decision: Accept (Poster)

Rationale: The paper makes a real and practical contribution — decomposing multi-condition attention via PAA and KSA based on observed sparsity patterns is well-motivated and achieves genuine efficiency gains (3.9× at 4 conditions with quality preservation). The sparsity analysis (Figures 2-3) and the overall framework design are solid. However, the headline 10× claim is not quality-validated, ablations lack quantitative metrics, and several design choices (condition cache, PAA alignment restriction) lack explicit empirical validation. These are significant gaps but do not invalidate the core contribution. Compared to calibration anchors, PKA is below PT-DiT (6.4, which had quantitative ablations across more tasks) and above SparseDM (4.0, whose efficiency claims didn't translate). A score of 5.5 reflects a paper with genuine contributions that would benefit from more rigorous evaluation, positioned at the acceptance boundary with the current evidence favoring a weak accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>