Now let me search for calibration anchors.Now I have sufficient information to write the final consolidated review.

---

## Summary
CMIC introduces Content-Aware Mamba for Learned Image Compression, proposing two mechanisms to overcome the rigid raster-scan causality of standard Mamba SSMs: (1) Content-Adaptive Token Permutation (CTP), which clusters tokens via a learnable EMA-updated codebook and reorders them so content-similar tokens are scanned contiguously, and (2) Global-Prior Prompting (GPP), which injects cluster-conditioned prompts into Mamba's output projection to encode sample-specific global statistics. The resulting CMIC model achieves strong rate-distortion performance — best on Tecnick and CLIC — with substantially lower memory, FLOPs, and latency than competing Mamba-based LIC models.

---

## Strengths

- **Strong and consistent results on high-resolution benchmarks**: CMIC achieves −21.34% and −17.58% BD-rate on Tecnick (1200×1200) and CLIC (2K) respectively, outperforming all listed baselines on both datasets (Table 1). The advantage growing with resolution is coherent with the method's motivation that content-adaptive long-range modeling matters most when sequences are long.

- **Significant improvement over prior Mamba-based LIC**: CMIC surpasses MambaVC by 6.80–10.09% and MambaIC by 2.17–6.48% BD-rate across all three datasets, while simultaneously reducing peak memory by 78%, FLOPs by 57%, and latency by 39% compared to MambaIC (Table 1, §4.4). This is a compelling efficiency-performance package.

- **Clean, complete ablation study**: Table 2 shows CTP and GPP each contribute independently (+1.8–2.4% and +0.5–1.4% respectively), and they are additive (+2.7–3.6% combined). The baseline is properly defined as vanilla single-scan Mamba. The inference overhead introduced by both components is minimal (4% latency increase, 0.387s → 0.405s, Table 3).

- **Codebook-based clustering is technically sound**: Using EMA-updated shared centroids (rather than per-sample online K-Means) avoids training instability and enables deterministic, efficient inference. The cluster visualization (Fig. 10) confirms that semantically coherent regions (red doors, sky, feathers) are correctly grouped — centroids learn reusable visual patterns across images.

- **Adaptive cluster sparsity emerges without explicit enforcement**: Only 23–26 of 64 centroids activate on average per image (Table 5), with high variance across images, demonstrating that the codebook behaves adaptively to content rather than uniformly.

- **ERF visualizations (Figs. 7–9) provide compelling qualitative evidence**: Per-image ERFs align with semantic structures (shoreline, aircraft, hair/feathers), visually confirming that CMIC's receptive field is content-adaptive while competing methods produce near-isotropic fields.

---

## Weaknesses

### Fatal
None.

### Major
None.

### Minor

- **Mild "consistently outperforms" overclaim**: Table 1 shows MLICv2 achieves −16.16% BD-rate on Kodak while CMIC achieves −15.91% — a 0.25% gap in MLICv2's favor. The paper states "Our CMIC model achieves superior performance... consistently outperforms leading methods across all evaluated datasets" (§4.3), which is factually inaccurate for Kodak. This should be qualified: CMIC achieves best results on Tecnick and CLIC (the higher-resolution datasets), where it leads clearly, while being slightly behind the best single method on Kodak but doing so with significantly fewer parameters, lower latency, and less memory. The resolution-dependent advantage is actually an interesting and honest framing worth making explicit.

- **GPP's causality-relaxation mechanism is imprecisely described**: The paper claims GPP "relaxes Mamba's strict causality" (§1, §3.4). In the deployed model, P_i is a per-token prompt derived from a one-hot cluster assignment (`P = ΓU`), meaning each token's prompt depends only on its own feature vector matched against pre-learned centroids — no other token's features enter the computation. The hidden state h_i remains strictly causal. The paper is partially transparent about this: §4.5 explicitly states the ERF visualization (Fig. 9(c)) uses **soft clustering**, which can introduce real cross-token dependencies through normalized assignment weights, but this is different from the hard-argmax model actually deployed. GPP is more accurately described as adding a semantic-class-conditioned bias to the output projection (analogous to class-token conditioning), which is genuinely useful and empirically validated, but does not constitute causality relaxation in the deployed model. The "mitigates strict causality" framing in the abstract and introduction should be corrected or qualified.

- **Entropy model limitation is asserted but not analyzed**: §4.5 notes "adding CAM to the entropy model yields negligible performance gains while increasing latency," but the authors offer no analysis of *why*. The spatial autoregressive context (SCCTX) in the entropy model assumes local spatial structure in latent codes; if CTP's permutation disrupts this spatial layout, it could degrade the context model's effectiveness. Understanding and reporting this interaction would strengthen the paper.

### Trivial

- **K ablation interpretation**: The authors say "a larger K does not yield much improvement," which is supported by the K=128 vs K=64 gap being only 0.05% (Table 6). However, the K=32→64 jump is 0.94%, suggesting the curve hasn't fully saturated — worth acknowledging in the text, even if K=64 is the right practical choice.

- **ERF clip threshold not justified**: The ERF visualization clips gradients to [0, 0.20] (§4.5). This choice is disclosed but not justified; it is a presentation detail worth one sentence of explanation.

---

## Nice-to-Haves

- **Random permutation baseline**: An ablation comparing content-adaptive token permutation against random shuffling before the Mamba scan would clarify whether the gain is specifically from content-adaptive grouping or partially from any permutation breaking the raster-scan inductive bias. This would strengthen the mechanistic claim of CTP.

- **Per-bitrate-point comparison**: BD-rate averages over a range. Since content-adaptive methods may benefit more at certain operating points (e.g., low bitrate where global redundancy dominates), per-point RD curves showing where CMIC's margin concentrates would be informative.

- **Explicit explanation of the resolution-performance relationship**: The paper's advantage is clearest on high-resolution datasets. Making this explicit as a scope condition (CMIC's content-adaptive scanning provides the most benefit when sequences are long, i.e., high-resolution images) would be honest and useful.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Harsh Critic's structural framing of GPP** — The critic labels the GPP non-causality issue as "Structural" and implies it is a core invalidating flaw. While the mechanism description is imprecise (verified against the paper), GPP still produces measurable and consistent improvements (0.5–1.4% BD-rate). The empirical contribution is real, so this is a Minor issue of imprecise framing, not a structural flaw.

2. **Centroid initialization concern** — The harsh critic flags the spatial-segmentation initialization as potentially problematic. The paper notes EMA updates ensure convergence across training batches (§3.3), and Fig. 10 shows clearly coherent clusters, indicating the initialization concern is effectively resolved in practice. This criticism is removed as addressed.

3. **Directional asymmetry within clusters** — The harsh critic notes that tokens at the start vs. end of a cluster group have different access to hidden state accumulation. This is a generic property of any sequential SSM and is not specific to the claimed contributions of this paper; it is a known limitation of Mamba in general.

4. **ERF clip threshold as a distortion concern** — The critic suggests clipping at [0, 0.20] may make CMIC look globally active. However, the clip value is explicitly reported, and the per-image ERF comparison (Fig. 8) uses the same threshold uniformly across all methods. This is a display choice, not an evidential flaw; moved to trivial.

5. **Choosing FTIC as "SOTA Transformer-based" baseline** — The harsh critic argues FTIC is weaker than MLICv2 and S2CFormer. However, FTIC is used only in the BD-PSNR comparison framing within §4.3, and Table 1 includes all the stronger baselines. Comparing against multiple baseline tiers in different framings is standard; this is not a misrepresentation.

---

## Novel Insights

The most genuinely novel observation from the review synthesis is the discrepancy between the GPP visualization methodology (soft clustering, which has real cross-token gradient flow) and the deployed mechanism (hard argmax, which does not). This suggests that the paper may have empirically discovered that soft-assignment global prompting is more powerful, but has implemented a computationally cheaper hard-assignment approximation. An explicit comparison of soft vs. hard assignment would test whether the 0.5–1.4% GPP gain is partially attributable to genuine non-causal gradient flow (only present in soft) or purely to semantic conditioning of the output projection. This could motivate a lightweight differentiable assignment variant that captures more non-causal benefit at modest additional cost.

---

## Suggestions

1. **Correct the "consistently outperforms" language**: On Kodak, qualify CMIC's result by noting MLICv2 is 0.25% ahead in BD-rate but CMIC achieves this with X% fewer parameters and Y% lower latency — the efficiency-adjusted comparison still favors CMIC.

2. **Reframe GPP accurately**: Describe it as "semantic-class-conditioned output projection" rather than "relaxing strict causality." If the authors want to claim non-causality, they should either (a) implement and test soft/differentiable assignment, or (b) acknowledge the soft-clustering visualization is an idealized analysis of the mechanism.

3. **Add a sentence explaining the entropy model failure**: Why does CAM not help the entropy model? Speculate or analyze whether SCCTX spatial locality assumptions conflict with CTP's permutation of latent spatial structure.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Human Score | Comparison |
|---|---|---|---|
| Frequency-Aware Transformer for LIC (FTIC) | `HKGQDDTuvZ.md` | 6.0 (Accept, poster) | Same domain, similar contribution type; CMIC has stronger results and more complete ablations |
| Lattice Transform Coding | `Tv36j85SqR.md` | 7.2 (Accept, spotlight) | Theoretically deeper; CMIC is more practical/engineering but less principled |
| Idempotence and Perceptual Image Compression | `Cy5v64DqEF.md` | 7.5 (Accept, spotlight) | Novel theoretical framework; stronger originality than CMIC |
| V2M: Visual 2D Mamba | `FowFLhUTgO.md` | 5.5 (Reject) | Mamba for vision, mechanistic questions raised; weaker empirics than CMIC |
| GlobalMamba | `XKQ2qzajbU.md` | 5.0 (Withdrawn) | Mamba scan-order redesign for vision; less practical, weaker results |
| NCA for Lightweight Compression | `gIrVoQEDQv.md` | 3.4 (Reject) | Weak LIC paper, only compared to JPEG — clearly below CMIC quality |

**Assessment**: CMIC is clearly above the GlobalMamba/V2M cluster (5.0–5.5) which attempted similar Mamba scan improvements with weaker results and methodological concerns. It is most directly comparable to HKGQDDTuvZ (FTIC), the poster-accepted LIC-specific Transformer paper that achieved −12.94% on Kodak — CMIC achieves −15.91% with significantly stronger ablations and efficiency analysis. The minor overclaim on Kodak and imprecise GPP framing are real but do not threaten the core contribution. No fatal or major weaknesses exist.

**Originality**: Good — CTP's codebook-based content-adaptive scan for SSMs is novel and well-motivated; GPP as a semantic conditioning mechanism is practical even if the causality framing is imprecise.  
**Importance**: Moderate-to-high — LIC with efficient Mamba is a timely problem; the efficiency gains over MambaIC are practically significant.  
**Claims vs. support**: Mostly well supported; the "consistently outperforms" claim on Kodak is a minor inaccuracy.  
**Experimental soundness**: Good ablation, multiple benchmarks, complexity analysis; minor gaps (random permutation control, entropy model analysis).  
**Writing clarity**: Good overall with minor imprecision in GPP mechanism framing.  
**Community value**: Solid contribution to the Mamba+compression niche with code release.

Positioned at **6.0** — above the 5.0–5.5 band of less-complete Mamba-vision papers and aligned with the 6.0 FTIC anchor (same domain, CMIC is comparably or slightly more innovative with stronger numbers and better efficiency story).

**Score: 6.0**  
**Decision: Accept (Poster)**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>