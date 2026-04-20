## Summary
This paper proposes channel-wise quantization for visual tokenization, inverting the standard VQ paradigm by quantizing across the channel dimension rather than the spatial dimension. The method produces tokens with a global spatial receptive field that capture both global structure and local details. Using a MaskGIT-style masked-prediction generator on top of this tokenizer, the paper reports competitive image generation results on ImageNet 256×256 and 512×512, and demonstrates cross-domain generalization to MS-COCO text-to-image generation.

## Strengths
- **Simple, well-specified formulation with a clear geometric insight:** Inverting from $(H_1 \times W_1)$ tokens of dimension $C$ (spatial) to $C$ tokens of dimension $H_1 W_1$ (channel-wise) is mathematically straightforward (Eq. 2). The paper cleanly adapts standard VQ losses (Eq. 3) and MaskGIT training to this paradigm without architectural surgery. The intuition that each code vector spans the entire spatial map, giving every token a global receptive field, is sound and supported by the high SSIM scores (0.866 vs. 0.675 for LlamaGen at dim=8 in Table 5).
- **Strong reconstruction quality as a tokenizer:** Table 5 shows the channel-wise tokenizer achieves the best rFID (1.64 at 256 tokens, 0.98 at 512 tokens) and substantially higher SSIM (0.866) compared to VQGAN, MaskGIT, and LlamaGen. This validates that channel-wise tokens capture structural similarity better than localized spatial tokens.
- **Tokenizer generalizes across domains without retraining:** Table 5 shows the ImageNet-trained tokenizer achieves the best rFID (7.95) and SSIM (0.860) on MS-COCO among all compared tokenizers, demonstrating cross-domain transfer.
- **Competitive generation quality with Inception Score gains:** Table 7 shows monotonic improvement with codebook size and model scaling. Ours-H* achieves IS 344.9 on ImageNet 256×256, surpassing MagViT-2's IS of 319.4 (Table 2). Ours-L* achieves best IS (341.5) on ImageNet 512×512 (Table 3).
- **Monotonic scaling behavior with codebook size:** Table 6b shows that both reconstruction (rFID: 2.25→1.33) and codebook unique ratio (72.5%→96.9%) improve consistently as codebook size grows from 1K to 131K, suggesting the approach can benefit from further scaling.

## Weaknesses

### Fatal
None

### Major

- **The "100% codebook usage" headline metric is mathematically uninformative at the reported scale.** The paper repeatedly claims "100% codebook usage" as a central result (Abstract, Section 1, Table 6). With $K=16{,}384$ and $256$ tokens per image evaluated over $50{,}000$ validation images, there are $\approx 12.8 \times 10^6$ token positions to fill. Barring complete model collapse, the pigeonhole principle guarantees every code is used at least once for virtually any functioning VQ model at this scale. A model that concentrates 99% of its probability mass on 100 codes and scatters the remaining $16{,}284$ codes across single edge-case tokens would still report "100% usage." Without reporting **codebook activation entropy**, **per-image unique ratio distribution**, or **code frequency histograms**, the metric cannot support the paper's claim that channel-wise quantization captures image structure better. The "unique ratio" metric in Table 6b is more meaningful, but it is relegated to an ablation table rather than the main evidence.

- **Generation quality advantages over baselines are not statistically validated.** The paper claims to "significantly improve baseline" (Abstract). The best FID reported is 1.87 (Ours-L† with 65K codebook, 64 steps, 634M params) vs. MagViT-2 at 1.78 (307M params, 64 steps) in Table 2. Differences of 0.09–0.14 FID are well within the known noise floor of FID evaluation (typically $\pm 0.2$–$0.5$ depending on seed, batch size, and InceptionV3 implementation details). The paper provides no multi-seed training runs, no standard deviations, and no statistical significance testing. The IS score advantage (e.g., 344.9 vs. 319.4) is more substantial but IS alone is insufficient evidence — both metrics should show consistent, statistically significant improvement to support the "state-of-the-art" framing.

- **Baseline configuration in tokenizer comparisons is outdated and under-tuned.** Table 5 and Table 6a compare the channel-wise tokenizer against LlamaGen/VQGAN. The spatial baseline at dim=256 collapses to 0.29% usage, and the paper's contrast at dim=8 (97% usage) requires severely reduced representational capacity. However, modern spatial tokenizers such as MAGVIT-2, RQ-VAE, and VAR employ entropy regularization, hierarchical quantization, or adaptive commitment losses to maintain high codebook utilization *without* crippling the embedding dimension. By not comparing against a properly configured modern spatial tokenizer of comparable capacity, the paper does not demonstrate that the advantage stems from the channel-wise design itself rather than from avoiding known pitfalls of standard VQ training. The generation results in Table 2 do include MAGVIT-2, but at the *tokenizer* evaluation level the comparison lacks a strong modern spatial baseline.

### Minor

- **Inference efficiency is not quantified.** The paper uses 10-step and 64-step masked iterative decoding (Tables 2–3, 7). Table 7 clearly shows that going from 10 to 64 steps substantially improves FID (e.g., 2.46→2.02 for Ours-L at 16K codebook), but the 6.4x increase in inference compute is not discussed in terms of wall-clock latency or throughput. For methods to be practically useful, a FID-vs-latency trade-off analysis against single-step diffusion or autoregressive samplers is needed.

- **Entropy regularization dismissal without ablation.** Section 3.2 states "We find entropy regularization is bad for our codebook learning" but provides no ablation data or mechanistic explanation. Given that entropy regularization is the standard technique that modern spatial tokenizers use to solve the codebook collapse problem, this claim requires empirical support.

- **Cross-resolution training limitation acknowledged but not addressed.** Section 5 notes that channel-wise tokenizers must be trained separately per resolution (since the code vector dimension $H_1 W_1$ changes with image size). A preliminary investigation into interpolation or adaptive pooling before quantization would strengthen the method's practical viability.

### Trivial
- The paper occasionally uses imprecise language (e.g., "significantly improve" without statistical backing, or "best" without specifying the metric scope), but these are minor presentation issues.
- Figure 1 uses a hand-drawn graph style rather than a formal plot; this is cosmetic.

## Nice-to-Haves
- Report per-seed FID variation or bootstrap confidence intervals to validate that channel-wise quantization's advantage over baselines is consistent.
- Add qualitative reconstructions (side-by-side visualizations of spatial vs. channel-wise tokenizer outputs) to complement the rFID/SSIM numbers in Table 5.
- Investigate whether the channel-wise approach benefits from entropy regularization when tuned specifically for this paradigm, rather than dismissing it outright.

## Removed Points
These points are flagged to be removed; treat them with caution.

- **Harsh Critic: "Unfair comparison with 634M vs. 307M params"** — The paper already includes models at comparable or larger sizes (Table 2 includes LlamaGen-XXL at 1.4B, Open-MAGVIT2-XL at 1.5B, VAR-d30 at 2.0B). The asymmetry where the author's smaller model (634M) is competitive with much larger models (1.4–2.0B) actually strengthens the paper's case, so criticizing this is invalid per the hard rules.
  
- **Harsh Critic: "Methodology unclear on FID variant (FID50K vs FID30K) and CLIP guidance scale"** — Section 4.1 states that evaluation uses "50K generated images compared against the ImageNet training set" for class-conditional generation, which specifies the FID variant. The guidance scale $t$ is defined in Eq. 4 (standard CFG). While the exact value of $t$ used during evaluation isn't specified, the paper does use classifier-free guidance and reports the formulation. This is a minor specification gap, not a methodological flaw.

- **Harsh Critic: "Channel-wise tokenization explored in earlier signal processing/compression literature (1D wavelets or DCT)"** — The paper's contribution is specifically in the context of modern VQ-based visual tokenization for generative modeling. Prior work in signal processing is a different domain. The novelty is in the application to learned visual tokenization for masked-prediction generation, which the reviewers acknowledged.

- **Harsh Critic: "The paper constructs a strawman with vanilla LlamaGen baseline" (expanded version)** — This overlaps with the valid major weakness about missing modern baselines. However, the phrasing about "strawman" is excessive: the paper does compare against MAGVIT-2, VAR, and other modern methods in the generation tables (Table 2). The concern is specifically about the tokenizer-level comparisons, which is captured in the major weakness above.

- **Strength Finder: "100% codebook usage without sacrificing token dimension" as a core strength** — This is downgraded from a strength to a weakness per the Major section above. The metric is not informative at the scale used, so it cannot serve as evidence of the method's quality.

## Novel Insights
The paper's inversion of the VQ quantization axis — from spatial to channel — is a conceptually clean and elegant reframing that had not been systematically explored in modern visual generative modeling. While the core idea is simple (transpose before quantization), the implications are non-trivial: channel-wise tokens intrinsically receive a global receptive field, which helps explain the high structural similarity (SSIM) scores and cross-domain generalization without retraining. However, the paper's central empirical claim — that 100% codebook usage demonstrates superior representation — fails under scrutiny, as the metric is trivially satisfied at the reported token/codebook scale. The paper's real contribution may be less about codebook utilization and more about demonstrating that a global-structure-preserving quantizer trained with standard VQ losses (without entropy regularization) can match modern spatial tokenizers that require more complex machinery. Repositioning the paper around the reconstruction quality and structural coherence benefits, rather than the usage claim, would yield a stronger contribution.

## Suggestions
1. **Replace or supplement the "100% usage" metric with codebook activation entropy or frequency histograms.** Report entropy of the code distribution or a histogram showing how uniformly codes are used. This would provide meaningful evidence of codebook quality rather than mere coverage.

2. **Add a properly tuned modern spatial tokenizer as a baseline in the tokenizer comparison table.** Configure a hierarchical quantizer (e.g., MAGVIT-2 or RQ-VAE-style) with comparable codebook size and parameter budget to establish a fair comparison at the tokenizer level.

3. **Run 3+ training seeds for the main generation results and report mean ± std.** This would validate that FID/IS improvements are consistent and not artifacts of a specific random seed.

4. **Report wall-clock inference latency and throughput** for 10-step and 64-step configurations, and compare against autoregressive and diffusion baselines on a per-image latency basis.

5. **Provide empirical ablation for the entropy regularization claim.** Show what happens when entropy regularization is added to the channel-wise tokenizer — does it help, hurt, or have no effect? A single ablation row in Table 6 would suffice.

6. **Include qualitative reconstruction visualizations** alongside the rFID/SSIM/PSNR numbers to demonstrate the structural quality advantage of channel-wise tokens.

## Score and Decision
Calibration was performed against the following anchors:

- **High-scoring (7–8 avg):** FSQ (8ishA3LxN8.md) scored ~6.5 avg (6,6,8,6, Accept Poster) and BSQ (yGnsH3gQ6U.md) scored 6 avg (6,6,6,6, Accept Poster). Both proposed clean quantization innovations for VQ-based generation with strong experiments. This paper has comparable empirical breadth but weaker statistical validation.
- **Medium-scoring (4–6 avg):** SCQ (V9C0cuEWbR.md) scored ~4.5 avg (6,6,3,3, Withdrawn) — addressed codebook collapse but lacked fair baselines, similar to this paper's tokenizer comparison gap.
- **Low-scoring (1–3 avg):** PQ-VAE (BJ4WgPgFqJ.md) scored ~2.3 avg (1,3,3, Withdrawn) — poorly described method with missing baselines and weak claims. This paper is clearly well above that level in clarity and experimental rigor.

The paper under review sits between the middle and high tiers. It has a genuinely clean idea, strong reconstruction results, and competitive generation numbers. But the core "100% usage" claim is not meaningful as presented, the generation improvements lack statistical validation, and the tokenizer baselines are outdated. Compared to FSQ (which earned acceptance on similar grounds of simplicity + competitive results), this paper is slightly weaker due to the unvalidated usage metric and missing multi-seed results, but not dramatically so.

Compared to SCQ (which was withdrawn due to incomplete comparisons), this paper has stronger empirical results and clearer writing, but shares the issue of not comparing against properly configured modern alternatives at the tokenizer level.

I place this paper in the **borderline-to-weak-accept** range. The fundamental idea is sound and the experimental results are competitive, but the paper overclaims relative to its evidence. The flaws are addressable in rebuttal (adding seeds, entropy metrics, and a stronger baseline would significantly strengthen the submission).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>