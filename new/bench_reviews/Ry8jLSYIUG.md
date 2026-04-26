Now I have enough information to write a thorough meta-review. Let me also check what the paper says specifically about "orders of magnitude" to verify the harsh critic's claim about overclaiming.

The paper says:
- Abstract line 12: "Recent methods operate far below the achievable bounds, often by orders of magnitude"
- Introduction line 23: "indicating capacities orders of magnitude larger than seen in practice"
- Section 2.5: "even under the most aggressive cropping, we should expect around 0.5 bpp or almost 100,000 bits for 256×256px images" — this uses the heuristic bounds
- But then: "For the other augmentations, the conservative capacity is much higher. Therefore, robustness to geometric transformations and compression significantly reduces the capacity but cannot fully explain the low watermarking capacity of current models."
- In discussion: "Our theoretical capacity bounds are an order of magnitude higher than even the best existing models"

Let me check the conservative bounds table more carefully:
- Crop&Rescale 75%: 904 bits (0.005 bpp) for 256×256px — this is roughly 3.5× what Video Seal achieves (256 bits)
- LinJPEG q=10: 26,757 bits (0.136 bpp)
- Rotation 30°: 14,676 bits (0.075 bpp)

So even the conservative bounds show significant headroom for most augmentations. The 904 bits for aggressive cropping is the tightest, but still exceeds Video Seal's 256 bits.

Now let me write the review.

## Summary

This paper presents geometric capacity bounds for image watermarking, showing that theoretical capacities are orders of magnitude larger than what current methods achieve. Under PSNR-only constraints, the bounds are rigorous (counting lattice points in hyperspheres), and the paper systematically rules out alternative explanations (robustness, perceptual quality, data distribution) by showing Video Seal fails at 1024 bits even on a single gray image. The paper demonstrates that simple baselines (linear model, tiling, handcrafted) dramatically outperform learned models in the simplified setting, and trains Chunky Seal (90× larger) to achieve 4× capacity with comparable robustness.

## Strengths

- **Rigorous and intuitive geometric framework for PSNR-only bounds**: The counting-lattice-points-in-hyperspheres approach (Sections 2.3–2.4) is transparent, well-derived, and yields concrete numerical bounds. The progression from absolute capacity through PSNR-constrained to arbitrary cover images is clean, and the worst-case corner analysis showing at most 1 bpp penalty is elegant and tight.

- **Convincing experimental diagnosis of architectural limitations**: The gray-image experiment (Table 1) — showing Video Seal fails at 1024 bits on a single fixed image where theory predicts ~600,000 bits available — is a powerful and elegant diagnostic. The finding that Video Seal at 256×256 achieves nearly the same capacity as at 32×32 directly demonstrates the architecture fails to exploit resolution. The linear model achieving 2048 bits at 44 dB PSNR on the same task falsifies the hypothesis that the bounds are unachievable.

- **Transparent about the limitations of its robustness bounds**: The paper explicitly acknowledges in Section 2.5 that Bounds 10–12 "could be much lower than reality" and are "not valid lower bounds," and provides the conservative Bound 13 alongside. The statement "Our robustness bounds are heuristic rather than formal, leaving ample room for sharper theoretical advances" in the discussion is commendable intellectual honesty.

- **Proposed sanity checks are practical and actionable**: Section 5's falsifiable criteria (linear capacity scaling with image size, linear decrease with PSNR, outperforming linear baselines) provide a concrete evaluation framework for future watermarking methods.

## Weaknesses

### Fatal
None.

### Major

- **The "orders of magnitude" framing overstates the provable gap under the practically relevant setting**: The abstract and Section 1 prominently claim capacity is "orders of magnitude" larger than practice, and Figure 1's log-scale inset combines PSNR-only and heuristic robust bounds without clearly flagging which are proven lower bounds. Under the provably conservative Bound 13, the most aggressive crop setting yields 904 bits for 256×256px — roughly 3.5× Video Seal's 256 bits, which is a meaningful gap but decidedly not "orders of magnitude." For less aggressive augmentations (LinJPEG, rotation), the conservative bound is genuinely large (thousands of bits), so the overall thesis still holds. The paper does acknowledge in Section 2.5 that Bound 13 is conservative and that heuristic bounds may be closer to truth, but the prominent "orders of magnitude" language in the abstract and title is anchored on the heuristic bounds. This is not fatal — the core insight (current architectures dramatically underperform) is well-supported — but the provable version of the claim under robustness constraints is weaker than presented.

- **Chunky Seal's scaling efficiency raises questions about the "substantially higher capacities are within reach" claim**: Scaling the embedder 90× and extractor 23× to achieve 4× capacity improvement is a steep cost. Moreover, LPIPS degrades from 0.0019 to 0.0085 (4.5×), JPEG bit accuracy drops from 99.74% to 98.79%, and multiple robustness metrics see small but consistent regressions. The paper describes LPIPS as "only slightly higher" — while the absolute values remain low, a 4.5× relative increase in perceptual distortion with a 90× model increase is arguably not "slight." Combined with the lack of hyperparameter tuning for Chunky Seal versus Video Seal's "extensive optimization," the scaling experiment more convincingly demonstrates that current architectures scale poorly than that "substantially higher capacities are within reach" through scaling. The paper's discussion section partially moderates this by acknowledging Chunky Seal is not a practical path forward, but the main results presentation could be more balanced.

### Minor

- **The VQ-VAE codebook argument for data distribution effects (Section 2.6) uses a potentially inadequate proxy**: Estimating the number of perceptually distinct images via VQ-VAE codebook sizes (10,240 bits or ~0.05 bpp) is one approach, but the relevant quantity for watermarking collisions is the number of natural images within a PSNR ball of a given cover, which depends on local geometry. The paper acknowledges this is a "conservative" estimate but doesn't discuss how different local neighborhoods might produce more severe collisions than global codebook counts suggest. This is a minor issue because the overall conclusion (data distribution can't explain orders-of-magnitude gaps) is likely still correct given the 2+ bpp headroom at PSNR-only.

- **The gray-image experiments, while elegant, test only the most favorable case for the theory**: The argument that data distribution doesn't matter is partly based on showing failures even on a single gray image. However, showing that a linear model achieves high capacity on a single gray image (where the embedding problem reduces to learning near-orthogonal vectors in high-dimensional space) does not fully transfer to the multi-image setting where the encoder must generalize across diverse covers. The paper partially addresses this with Chunky Seal's real-image experiments, but a linear/handcrafted baseline on real images (not just gray) under PSNR-only constraints would close this gap more convincingly.

### Trivial
None.

## Nice-to-Haves

- A capacity–robustness Pareto frontier comparing Chunky Seal at varying capacities against Video Seal would clarify whether the 4× improvement is a genuine frontier shift.
- An ablation investigating why Video Seal's architecture fails at high capacity (depth, bottleneck width, skip connections) would strengthen the architectural diagnosis.
- Empirical measurement of true capacity under simple linear transformations (e.g., via Monte Carlo at small scales) would help calibrate whether heuristic or conservative bounds are closer to reality.

## Removed Points

- **Heuristic bounds are "not validated lower bounds" (Harsh Critic Point 1, partial)**: While the point about the gap between heuristic and conservative bounds is valid and important (kept as Major weakness above), the original formulation that this "undermines the core claim" overstates the issue. The paper's PSNR-only bounds are rigorous, and even the conservative robust bounds show meaningful headroom for most augmentations. The core insight (architectures underperform) is well-supported; only the specific framing of "orders of magnitude under robustness" is overclaimed.

- **Chunky Seal shows "diminishing returns" undermining the thesis (Harsh Critic Point 2, partial)**: The harsh critic argues this contradicts the paper's thesis. In reality, the paper's thesis has two parts: (1) architectures underperform (well-supported) and (2) higher capacities are achievable (supported by the 4× gain). The 90× cost does suggest diminishing returns from naive scaling, which the paper acknowledges, but 4× improvement is still real improvement. Kept as Major since the scaling cost should be discussed more honestly, but removed the claim that it "contradicts rather than supports the paper's thesis."

- **Missing ablations/experiments (Harsh Critic "Missing Experiments")**: Requests for ablations on Video Seal, capacity–robustness Pareto, multiple diverse covers, etc. are reasonable suggestions for future work but are not required for the paper's core claims. Moved to Nice-to-Haves.

- **Formatting/style issues**: Removed all typography and formatting complaints as parser artifacts.

- **The "data distribution argument relies on inappropriate proxy" (Harsh Critic Point 3)**: The argument has logical merit but the paper's overall conclusion (that data distribution can't explain orders-of-magnitude gaps given the 2+ bpp headroom) is reasonable. Downgraded to Minor.

- **"Chunky Seal's LPIPS regression means quality is not maintained"**: While worth noting (the 4.5× LPIPS increase), the absolute values (0.0085 vs 0.0019) are both near-perceptually-indistinguishable, and the confidence intervals overlap substantially (0.0085±0.0067 vs 0.0019±0.0011). The paper's claim of "comparable" quality is defensible in absolute terms though the relative increase deserves acknowledgment. Integrated into the Major weakness.

## Novel Insights

The paper's most novel insight is the finding that Video Seal's architecture fundamentally fails to exploit resolution: achieving nearly identical capacity at 256×256 and 32×32, while a handcrafted model reaches 456,509 bits on the same image. This transforms the watermarking capacity question from "what is theoretically possible?" into "what is architecturally achievable?" — and the answer is that current learned architectures leave at least 99.8% of theoretical capacity on the table even in the simplest possible setting. The tiling result is particularly illuminating: a 512-bit 32×32 model tiled to 256×256 gives 32,768 bits, yet Video Seal cannot even learn 1024 bits at 256×256 natively, suggesting the bottleneck is specifically in the architecture's ability to distribute information across spatial dimensions.

## Suggestions

- Restrate the "orders of magnitude" claim to clearly distinguish between PSNR-only bounds (rigorous, genuinely orders of magnitude) and robust bounds (where the provable gap narrows). A sentence like "Under PSNR-only constraints, the gap is orders of magnitude; under robustness constraints, even our conservative bounds show at least 3–100× headroom depending on the augmentation" would be more precise.
- Report both heuristic and conservative bounds side-by-side in a single prominent table or figure, making the gap between them an explicit discussion point rather than relegating conservative bounds to a brief mention.
- Consider describing Chunky Seal's LPIPS increase more precisely (e.g., "LPIPS increases from 0.002 to 0.009, remaining in the near-imperceptible range" or similar).

## Calibration Summary

| Anchor Paper | Avg Score | Relevance |
|---|---|---|
| Fantastic Generalization Measures (NkmJotfL42) | 7.0 | Rigorous theory diagnosing fundamental limitations of existing approaches, similar spirit; accepted poster |
| Limits of Deep Learning / SSMs (DhdqML3FdM) | 7.0 | Theory + experiments diagnosing architectural limitations; accepted poster |
| Expressiveness of Rational ReLU Networks (uREg3OHjLL) | 7.4 | Proving capacity limits of architectures; accepted spotlight |
| Lightweight Deep Watermarking (j7b4mm7Ec9) | 7.6 | Directly watermarking-related, practical improvements; rejected despite high scores |
| Capacity Analysis of VSA (6tazBqPem3) | 3.67 | Capacity bounds with questionable practical utility; rejected |
| Domain Generalization Theory (KstDMYkfj4) | 3.8 | Overclaimed theory with vacuous bounds; rejected |

This paper has a strong core (rigorous geometric bounds, clean diagnostic experiments ruling out alternative explanations, and constructive baselines). Its main issue is overframing the robust-setting claim, which is a presentation/claim-tempering issue rather than a fundamental methodological flaw. Compared to accepted theory-papers-diagnosing-limitations (which score 7.0–7.4), this paper has a comparable core contribution but somewhat looser claims in the practical setting. Compared to rejected overclaim papers (scored 3–4), this paper has far stronger experimental support and honest acknowledgment of limitations. The overclaim is real but moderate — the conservative bounds still show meaningful headroom, and the PSNR-only analysis is rigorous.

MY FINAL SCORE: 7.0
MY FINAL DECISION: Accept