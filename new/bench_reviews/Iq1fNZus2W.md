Now I have enough calibration data. Let me synthesize my understanding of the paper and the reviewer's critiques.

**Key facts verified from the paper:**

1. **Quantitative ablations for PAA/KSA**: The paper provides ONLY qualitative (visual) ablations for PAA (Figure 9) and KSA (Figure 10). There are NO quantitative metrics (FID, CLIP-I, controllability scores) comparing full attention vs. PAA vs. SWA or KSA at different ε. The main Table 1 compares against other methods, not against ablations of proposed architecture. This is a genuine weakness.

2. **Canny F1 score drop**: The paper does say "minor exception of a narrow margin" but the gap is 0.414 vs. 0.551 — a 24.9% relative drop. This IS substantial and the paper's characterization is misleading.

3. **Condition Cache confounded**: Section 3.2 explicitly states that condition tokens only self-attend within their condition type, and K/V is computed once and cached. This IS a separate design choice from PAA/KSA and is not ablated independently.

4. **Training fairness**: The paper says "To ensure a fair comparison" and describes their LoRA setup. It doesn't explicitly state whether OminiControl2 and UniCombine use the same training regime.

5. **μ and δ values**: The paper states "t ∼ Logit-N(μ, δ), where μ > 0, δ > 1" but never gives specific values. This is a minor issue.

6. **KSA mask reuse**: The paper computes mask at timestep t and reuses at t+1 (Eq. 3-4). The temporal consistency justification is a reference (Zhou et al., 2025) and no quantitative validation is provided for how mask stability changes across denoising steps.

7. **Efficiency claims**: "up to 10× speedup and 5.12× reduction in attention module VRAM" — these are specifically about the attention module, not the full model. The paper is clear about this in the efficiency section.

Now let me write the review.

## Summary

PKA proposes decomposing full multi-condition attention in DiTs into two sparse modules: Position-Aligned Attention (PAA) for spatial conditions (reducing O(N²) to O(N)) and Keyword-Scoped Attention (KSA) for subject conditions (using keyword-activated masks to prune attention), plus a Condition Cache and early-timestep sampling strategy. The paper achieves up to 10× inference speedup on the attention module while showing competitive generation quality on multi-condition tasks.

## Strengths

- **Well-motivated sparsity analysis**: Figures 2 and 3 empirically demonstrate that spatial condition attention is diagonal-dominant and subject condition attention is keyword-activated, providing concrete motivation for decomposing attention by condition type. This observation-driven design is more principled than purely intuition-based sparsity proposals.

- **Strong efficiency gains with quality preservation on most metrics**: Table 1 shows PKA achieving the best FID across all three tasks (52.99, 62.08, 53.01 vs. next-best 61.03, 70.22, 67.40) and best DINOv2 subject consistency. The efficiency improvements (3.90×–10.0× speedup, 2.46×–5.12× VRAM reduction) are substantial and well-demonstrated across varying condition counts.

- **Clean modular design**: The decomposition into PAA, KSA, Condition Cache, and early-timestep sampling is clearly presented (Figure 4), with each component independently motivated and visual ablations provided. The framework is easy to understand and adopt incrementally.

## Weaknesses

### Fatal
None.

### Major

- **No quantitative ablations for PAA and KSA**: The core architectural claims — that PAA and KSA maintain generation quality while improving efficiency — are validated only through visual comparisons (Figures 9 and 10) against full attention and SWA. No FID, CLIP-I, or controllability metrics are reported for these ablations. Without quantitative evidence that these attention modifications preserve quality, the central claim is inadequately supported. This is particularly important because the main Table 1 compares against different methods, not against ablations of the proposed modules, so it's impossible to disentangle which design choices drive the observed quality differences.

- **Canny F1 controllability degradation is mischaracterized**: On Subject-Canny, PKA achieves F1=0.414 vs. UniCombine's F1=0.551 — a 25% relative drop in spatial controllability. The paper describes this as "the minor exception of a narrow margin," which significantly understates the issue. Since PAA's 1:1 position correspondence directly governs spatial condition interaction, this degradation could be a direct consequence of eliminating cross-position spatial attention. The lack of honest discussion of this tradeoff undermines confidence in the claim of "maintaining or improving" controllability.

### Minor

- **Multiple design choices are confounded (Condition Cache, PAA, KSA, early-timestep sampling)**: The Condition Cache mechanism restricts condition tokens to self-attention only within their type and caches K/V across denoising steps. This is a meaningful architectural change (removing cross-condition interactions) that is not independently ablated. The paper's experiments cannot attribute quality or efficiency improvements to individual components.

- **KSA mask temporal stability is assumed but not validated**: Equation 3 computes a binary mask at timestep *t* and reuses it at timestep *t+1* (Eq. 4). While the paper cites temporal consistency, it provides no quantitative analysis of how much masks shift between consecutive steps or how much quality degrades from mask reuse rather than recomputation.

- **Specific values for early-timestep sampling parameters (μ, δ) are not stated**: The shifted logit-normal distribution is described only as "μ > 0, δ > 1" without specific values being reported.

### Trivial
None.

## Nice-to-Haves

- Full-model (not just attention-module) speedup and VRAM numbers, to ground the practical impact of the method.
- A quantitative ablation table comparing full attention vs. PAA-only vs. KSA-only vs. full PKA (with and without Condition Cache and early-timestep sampling) on all metrics.
- Error analysis or failure case visualization demonstrating where PAA's 1:1 restriction causes spatial coherence problems.
- Discussion of failure modes and limitations.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"25% F1 drop means PAA 1:1 restriction is fundamentally flawed"**: The Canny F1 drop is concerning and worth honest discussion, but it cannot be conclusively attributed solely to PAA's design since multiple components are confounded. Kept in a softened form as a Major weakness about mischaracterization, not as a Fatal flaw.

- **"Unfair baseline comparison because baselines may not use same training regime"**: The paper explicitly states training details and says "To ensure a fair comparison." Whether baselines are off-the-shelf or LoRA-fine-tuned is a legitimate question, but not enough evidence exists to claim the comparison is definitively unfair; this is more of a clarification request.

- **"Full-model overhead should be reported instead of attention-module only"**: The paper's headline claims clearly specify "attention module" efficiency. Requesting full-model numbers is a nice-to-have, not a weakness.

- **"Figure 2 analysis is from an existing model, not the proposed one"**: This is a standard motivating analysis — using attention patterns from an existing baseline to identify redundancy is a legitimate empirical approach and does not invalidate the subsequent design.

- **"The x-axis of efficiency figures is unrealistic"**: Each condition having 1024 tokens reflects the standard patch tokenization for typical image resolutions, so this is a reasonable experimental setup.

- **"Nitpicks about formatting/parser artifacts"**: Removed per rules (these are parser issues, not author errors).

## Novel Insights

The most interesting observation from the reviews is the tension between the strong quality metrics (best FID and DINOv2 across tasks) and the 25% Canny-F1 gap on spatial controllability. This suggests PKA's efficient attention decomposition preserves generative quality and subject identity well but may introduce a specific controllability cost for precise spatial conditions — a nuanced tradeoff that the paper's "maintaining or improving" framing flattens. Whether this is due to PAA's position restriction specifically, or is an artifact of the Condition Cache removing cross-condition interaction, cannot be determined without proper ablations.

## Suggestions

- Add a quantitative ablation table that isolates each component's contribution, especially PAA vs. full spatial attention and KSA vs. full subject attention, on all quality and controllability metrics.
- Honestly acknowledge and discuss the Canny F1 gap rather than characterizing it as a "narrow margin."
- Report the specific values of μ and δ used in the early-timestep sampling experiments.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| SANA | N8Oj1XhtYZ.md | 8.5 (Oral) | Much stronger than PKA — SANA has extensive quantitative ablations, full-model efficiency numbers, and comprehensive evaluation. |
| PT-DiT | lTrrnNdkOX.md | 6.4 (Poster) | Comparable efficiency-orientation work with quantitative ablations. PKA is somewhat weaker in validation (lacking quantitative ablations) but has stronger empirical motivation. |
| LinFusion | D2as3jDmRA.md | 6.25 (Reject) | Similar efficiency claims for attention in diffusion models; rejected partly for lacking ablation studies. PKA has similar issues. |
| MDiT | leBbjaUxut.md | 5.0 (Reject) | Claims 7× training speedup but rejected for missing ablation studies to isolate which components drive performance. PKA shares this issue. |
| DeeDiff | 3xHbRLymyZ.md | 4.5 (Reject) | Claims efficiency gains but rejected for unfair comparisons and lack of quantitative ablations. PKA is stronger than this. |
| Sensitivity-Adaptive | JBOMYYX94D.md | 4.75 (Reject) | Overclaimed efficiency (10×, 200×); rejected for missing ablations. |

PKA sits between PT-DiT (6.4, accepted) and the rejected efficiency papers (4.5-5.0). It shares the "missing quantitative ablations" weakness with the rejected group, but has stronger empirical motivation (attention pattern analysis) and more convincing main results (best FID across tasks, strong efficiency). The missing ablations are a significant but not fatal-gap. The paper's contributions are real but the validation could be stronger.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>