Now I have all the information needed. Let me write the final consolidated review.

## Summary

AdcVSR proposes an improved adversarial diffusion compression method for real-world video super-resolution that distills a large 3D DiT teacher (DOVE, 10.55B params) into a compact "2D+1D" student (0.57B params) using a pruned SD2.1 backbone augmented with lightweight 1D temporal convolutions. A dual-head, dual-discriminator adversarial distillation scheme disentangles detail and consistency objectives via five curated data types with head-specific labels. The result achieves 95% parameter reduction and 8× speedup over DOVE with the best temporal consistency (E*warp) across all compared methods.

## Strengths

- **Impressive efficiency gains with strong temporal consistency**: AdcVSR achieves 95% parameter reduction (0.57B vs. 10.55B) and 8× speedup (0.55s vs. 4.42s) over DOVE, while attaining the best E*warp across all 11 methods on both UDM10 (1.67) and VideoLQ (6.74) benchmarks (Table 1, Fig. 4). These are substantial practical improvements.

- **Principled dual-head, dual-discriminator design with curated data types**: The five data types in Eq. 5—student outputs, real videos, shuffled videos, repeated static images, and mismatched image sequences—with carefully assigned (yd, yc) labels independently vary detail richness and temporal consistency, providing orthogonal gradient signals. Table 3 validates this: the dual-head, dual-domain scheme achieves best CLIPIQA (0.6861) and E*warp (2.22), outperforming single-head (E*warp=6.32) and single-domain (CLIPIQA=0.6421) variants.

- **The "2D+1D" architectural insight is well-motivated**: The argument that temporal consistency is inherently simpler than detail synthesis (since the LR video already provides structural and temporal information) justifies replacing heavy 3D attention with lightweight 1D temporal convolutions. Table 2 shows the 2D+1D design narrows the DISTS gap to the 3D model to just 0.0014 while using only 7% of the parameters.

- **Two-stage training curriculum**: Error-minimizing distillation followed by adversarial fine-tuning is a sensible stabilization strategy, and Table 4 confirms both components are necessary (removing adversarial losses degrades LPIPS from 0.3337 to 0.3596; removing the teacher degrades MUSIQ from 61.48 to 50.32).

## Weaknesses

### Fatal
None.

### Major

- **Confounded architecture ablation in Table 2**: The 3D baseline ("A Pruned DOVE") is trained with "the original ADC approach" (single-head, single-domain adversarial distillation), while the 2D+1D model uses the full improved dual-head, dual-domain scheme. This confounds the architectural contribution with the training method contribution—the comparison cannot isolate whether the 2D+1D design itself is responsible for the strong E*warp=1.67, or whether the improved distillation scheme does the heavy lifting. Table 3 partially addresses this by isolating the training method's contribution, but it doesn't answer the critical question: *would a pruned 3D model trained with the same dual-head distillation outperform 2D+1D?* Without this control, contribution (2)—that the 2D+1D design itself suffices to learn from a 3D teacher—is not fully validated.

- **"Competitive video quality" claim understates the quality trade-off**: The paper repeatedly claims AdcVSR maintains "competitive video quality" (abstract, Sec. 4.2, conclusion), but on UDM10, AdcVSR underperforms its teacher DOVE by 0.64 dB PSNR, 0.042 LPIPS, and 0.038 DISTS—these are not negligible gaps (a 16% LPIPS increase and 0.64 dB PSNR drop are meaningful). The paper foregrounds E*warp (where it excels) and backgrounds the quality metrics (where it doesn't). The characterization should be more honest: AdcVSR achieves *better temporal consistency at the cost of some perceptual quality*, not "competitive" quality across the board. The paper does rank in the top 3 on most metrics, so "competitive" has some support, but the trade-off relative to the teacher should be stated explicitly.

### Minor

- **Dual-head "disentanglement" is asserted but not validated analytically**: The paper claims the two heads learn "disentangled" representations (Sec. 3.3, "decoupling the discriminations of details and consistency"), but Table 3 only shows the *combination* works—it does not demonstrate that the heads learn different features. No activation visualization, gradient attribution, or intervention study (e.g., removing one head at inference) is provided. A single-head discriminator with matched total capacity might achieve similar results. This doesn't invalidate the method, but the "disentanglement" claim is stronger than the evidence supports.

- **Ablation tables use different test datasets**: Tables 2, 3, and 4 evaluate on UDM10, YouHQ40, and MVSR4x respectively, preventing cross-comparison of ablations. A single common dataset for all ablation tables would strengthen the analysis.

### Trivial
None.

## Nice-to-Haves

- A controlled ablation where the 3D pruned DOVE model is trained with the dual-head, dual-domain scheme would decisively validate whether the 2D+1D architecture itself contributes or whether the training method drives the improvement. This is the single most impactful addition the authors could make.

- Head-specific attribution maps or activation visualizations would directly validate the disentanglement claim and strengthen the mechanistic understanding of the method.

- Testing on longer video sequences (beyond 25 frames) would reveal whether the 1D temporal convolution with kernel size 3 maintains consistency over longer horizons.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Real videos contribute no gradient signal for the detail head"**: The paper deliberately labels real video details as yd=0 (unlabeled) and uses real images as positive supervision for the detail head, with explicit justification: "we leave real video details unlabeled, and rely on real images as the positive supervision for 'detail' head, encouraging the generator to produce more detail-rich frames." This is a reasoned design choice, not a flaw—real videos may have inconsistent detail quality, so using only real images as positive detail examples is appropriate.

- **"Feature-domain discriminator circularity"**: Using a frozen pretrained backbone (the augmented SD UNet from stage 1) as the feature extractor for the discriminator is standard practice in adversarial training. The discriminator's trainable components (head layers, tail convolutions) provide adversarial signals independent of the frozen features. This does not create true circularity.

- **"Unfair comparison with image SR methods"**: PiSA-SR, AdcSR, and HYPIR are included explicitly for "comprehensive evaluation" and serve as useful baselines that lack temporal modeling. Including them highlights AdcVSR's temporal consistency advantage. The paper does not claim superiority over them on every metric—they actually outperform AdcVSR on some no-reference metrics. This is an appropriate comparison, not an unfair one.

- **"Table 4 'No Teacher' achieving highest PSNR is surprising/unexplained"**: The paper explains this result: removing teacher supervision and relying solely on GT leads to higher PSNR but degraded LPIPS and MUSIQ. This reflects the well-known fidelity-perception trade-off in super-resolution and is not surprising.

- **"1D convolution kernel size 3 not justified"**: Kernel size 3 is the standard default for temporal convolutions and does not require special justification.

- **"Single-GPU latency comparison may not reflect real-world deployment"**: Speculative concern about deployment scenarios; the single-GPU comparison is standard in the field.

- **"Degradation pipeline mismatch with teacher"**: The paper uses the same RealBasicVSR degradation pipeline for training as is standard; whether DOVE used the same pipeline is an implementation detail, not a methodological concern.

## Novel Insights

The five curated data types with head-specific labels (Eq. 5) constitute a particularly clever design: by independently varying detail richness and temporal consistency across data types (shuffled videos = fake for consistency, real images repeated = real for both, mismatched images = real for detail but fake for consistency), the scheme provides orthogonal gradient signals to the two discriminator heads. This transforms a single adversarial signal into a structured multi-attribute supervision, which is both principled and practically effective. The approach could be generalized to other multi-objective adversarial training settings beyond VSR.

## Suggestions

- Add a controlled experiment where a pruned 3D model is trained with the same dual-head, dual-domain distillation scheme. This would cleanly isolate the architectural contribution from the training method contribution and validate (or refute) contribution (2).

- Rewrite the "competitive video quality" claim to explicitly acknowledge the quality-consistency trade-off: "AdcVSR achieves superior temporal consistency at the cost of a modest reduction in spatial perceptual quality relative to the teacher, while remaining competitive with existing methods."

- Evaluate all ablation variants on a single common test dataset to enable direct cross-comparison.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| Würstchen (efficient diffusion architecture) | 8.0 | Much stronger novelty and cleaner ablations; this paper is well below |
| EfficientDM (diffusion model compression, quantization) | 6.5 | Similar domain (diffusion compression), similar novelty level; this paper is comparable but has more ablation concerns |
| RGT (efficient SR transformer) | 7.5 | Clean ablations, strong results; this paper is below |
| MarDini (video generation, insufficient ablation) | 5.5 | Similar concern about insufficient ablation; this paper has stronger practical results |
| Dissecting arbitrary-scale SR (confounded ablations) | 5.25 | Similar confounded ablation concern; this paper has stronger engineering contribution |
| Efficient continuous video flow (incremental, poor ablation) | 4.0 | This paper is clearly above |
| Harry Potter visual representation (overclaimed, confounded) | 2.0 | This paper is well above |

The paper presents a solid engineering contribution with impressive efficiency gains and a clever training scheme. Its main weaknesses—a confounded architecture ablation and an overclaimed "competitive quality" characterization—are meaningful but do not invalidate the practical results. The paper is comparable to EfficientDM in contribution level but has slightly more methodological concerns (confounded ablation). Compared to MarDini (5.5, rejected for insufficient ablation), this paper has substantially stronger practical results and a more novel training scheme, but shares the ablation concern. It is clearly above the low-scoring anchors (which have fundamental flaws or overclaims). I place it in the upper range of borderline—strong practical results but with methodological gaps that limit the depth of the contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>