Now let me search for calibration anchors.Now I have enough calibration data. Let me write the final review.

---

## Summary
SecureGS proposes a steganography framework for 3D Gaussian Splatting (3DGS) that embeds hidden 3D objects, images, or bit messages within a public Scaffold-GS point cloud. The key innovations are: (1) a hybrid decoupled Gaussian encryption representation that stores original-scene attributes explicitly while predicting hidden content attributes via private MLPs, ensuring file format consistency; and (2) a region-aware density optimization (RDO) strategy using DBSCAN and asynchronous gradient accumulation to grow original-scene anchors in the hidden object's spatial region, preventing geometric structure leakage. The method significantly outperforms the prior GS-Hider baseline in rendering fidelity (+1.16 dB / +1.66 dB PSNR), rendering speed (~3× FPS), and storage efficiency.

---

## Strengths

- **Effective geometric structure security via RDO (Fig. 7)**: The RDO strategy visually and compellingly demonstrates that SecureGS conceals the hidden object's geometry in the anchor point cloud. In contrast, GS-Hider's point cloud (Fig. 7a) leaks the microphone silhouette in the "bonsai" scene, while SecureGS (Fig. 7c) shows no discernible trace. The adaptive threshold mechanism (Eq. 6) is technically motivated and non-trivial.

- **Substantial efficiency improvements (Table 1)**: SecureGS achieves 131.71 FPS vs. GS-Hider's 48.28 FPS (~3× speedup) and reduces storage from 468.63 MB to 267.39 MB (43% reduction), with only 7–25% FPS overhead over the Scaffold-GS baseline. These gains flow directly from the anchor-based neural decoding design, which avoids GS-Hider's expensive convolutional coupled-feature pipeline.

- **Rendering fidelity improvement (Table 1, Fig. 5)**: Scene-level PSNR of hidden content improves by +1.66 dB over GS-Hider. Original-scene PSNR marginally surpasses the Scaffold-GS baseline (27.75 vs. 27.62 dB), demonstrating that the decoupled representation prevents mutual interference between original and hidden content.

- **Decoupled visualization capability (Fig. 6)**: SecureGS uniquely supports separate decoding and rendering of original and hidden content, both as 2D views and as separated point clouds. This is a practically meaningful capability for copyright verification that GS-Hider cannot offer.

---

## Weaknesses

### Fatal
- None. The core technical contributions (RDO strategy, hybrid decoupled representation, efficiency improvements) are real and empirically verified.

### Major

- **Lack of formal threat model and quantitative security evaluation**: The paper's primary headline contribution is "security," specifically geometric structure security. Yet the entire security analysis in Section 4.3 is visual inspection of Fig. 7 with the qualitative assertion "almost no traces of the hidden scene can be detected." There is no quantitative security metric (e.g., detection accuracy of a trained point-cloud classifier distinguishing SecureGS models from clean Scaffold-GS), no formal threat model specifying what the adversary can observe or knows, and no adversarial detection experiment. The RDO strategy is well-motivated, but whether it actually resists a determined, informed adversary who can analyze anchor density distributions or probe MLP weight statistics is completely untested. A paper whose primary framing is security cannot support this framing with visual inspection alone; this gap weakens the core claim.

- **Bit-hiding comparison (Table 3) conflates tasks with different difficulty levels**: SecureGS achieves 100% bit accuracy vs. CopyRNeRF's 62.15% and NeRFProtector's 92.69%. However, the paper itself acknowledges the fundamental difference: SecureGS decodes bits "directly from the point cloud" via per-voxel feature cross-validation (no rendering required), while CopyRNeRF and NeRFProtector recover bits from rendered 2D views and are designed for robustness against image-domain distortions. Storing bits in MLP weights is trivially easier than encoding them through a perceptual rendering pipeline. The 100% accuracy figure therefore says little about SecureGS's watermarking capability relative to these baselines; it primarily reflects the easier task. The paper partially acknowledges this ("both these methods use a message decoder for image watermarking, they possess unique advantages in the generalization of message extraction"), but the table is still presented as a superiority benchmark, which is misleading.

### Minor

- **Robustness evaluation limited to random anchor pruning (Table 2)**: Only random pruning is tested as an attack. No steganalytic detection experiments (e.g., training a classifier to distinguish encoded from unencoded models), no re-encoding attack (training fresh Scaffold-GS from rendered outputs), and no informed attacks (zeroing private MLP weights) are evaluated. While exhaustive attack evaluation is expensive, at least one adversarial or steganalytic experiment would substantially strengthen the robustness claims.

- **HDGER ablation lacks format-level validation (Table 4)**: The ablation shows HDGER has nearly zero impact on rendering metrics (0.12 dB PSNR difference). The paper correctly states HDGER's purpose is format security (Sec. 4.5), but provides no format-level validation (e.g., byte-by-byte comparison of the container file against a standard Scaffold-GS file, or confirming that a Scaffold-GS parser opens the container without error or warning). Since the design argument for HDGER is compelling (Eq. 3–5 and the explicit-implicit hybrid design), the omission of format verification is an understandable but genuine gap.

- **Scope limitation understated**: Section 3.2 states only 3D objects (not full 3D scenes) can be hidden, because hiding a full scene would compromise point cloud confidentiality. This is an inherent limitation of the RDO strategy, and is noted only parenthetically rather than being analyzed in the limitations section. Understanding why this limitation exists and what would be needed to extend to scene-level hiding would strengthen the paper.

### Trivial
- None warranting mention beyond the above.

---

## Nice-to-Haves
- A quantitative steganalysis experiment (e.g., a trained classifier on anchor point clouds) would directly validate the geometric structure security claim and transform the security contribution from qualitative to empirically grounded.
- Anchor density analysis as a function of spatial position (for clean Scaffold-GS vs. SecureGS) would directly characterize whether the RDO-induced density change is visually or statistically distinguishable.
- Extending the robustness evaluation to informed attacks (e.g., an adversary who knows the Scaffold-GS carrier format and probes for private MLPs via activation statistics) would add meaningful coverage.
- Discussion of the path toward hiding full 3D scenes rather than objects only, with analysis of what makes this hard, would scope the contribution more clearly.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

1. **3DGS+StegaNeRF baseline suspicious at 20.96 dB (Table 5)** *(Harsh Critic)*: The critic claims this implies a suboptimal baseline implementation. However, 3DGS+StegaNeRF adds a constrained U-shaped decoder on top of the rendered RGB view, which intentionally sacrifices rendered quality to embed the hidden image in the view. The drop from 24–25 dB to ~21 dB is expected given this coupling (the paper explains the method "often suffers loss collapse when directly decoding objects with black/white backgrounds"). This is a method design consequence, not an implementation bug. **Removed as a misreading of the method.**

2. **Density anomaly as an attack surface** *(Harsh Critic)*: The critic suggests that RDO introduces an elevated anchor density in the hidden region detectable by an adversary. However, the mechanism in Eq. 6 works by *growing original-scene anchors* (not hidden anchors) in the hidden region — Fig. 7c shows the covered region is dense with original-scene anchors, so the density increase is camouflaged as normal scene complexity. Whether the density pattern is statistically distinguishable is an empirical question not validated either way; the concern is speculative without measurement. **Downgraded to Nice-to-Have.**

3. **100% bit accuracy as a strength** *(Strength Finder)*: While technically accurate, this strength conflicts with the verified Major weakness that the comparison is fundamentally unfair. The 100% figure is a consequence of the easier decoding task (model weights vs. rendered images), not a genuine superiority over those baselines. **Removed from Strengths as it conflicts with a verified major weakness.**

---

## Novel Insights
The most genuinely novel conceptual insight in SecureGS is the distinction between *file format security* (matching the container file format of the underlying representation) and *geometric structure security* (preventing point cloud visualization from revealing hidden content). Prior work conflated these or addressed only the former. The RDO strategy — using asynchronous gradient accumulation to identify the hidden object's spatial region, then lowering the original-scene anchor splitting threshold locally — is an elegant and non-obvious solution that uses the scene's own density control mechanism as a security primitive. The insight that hiding a 3D object in a larger scene exploits the sparse anchor structure's natural camouflage at object-scale is a useful framing for future 3DGS steganography work.

---

## Suggestions
1. Train a simple PointNet or MLP-based binary classifier on anchor point clouds from clean Scaffold-GS models vs. SecureGS models and report detection accuracy. Near-chance accuracy would strongly validate the geometric security claim; this single experiment would likely resolve the major weakness.
2. Reframe Table 3 as an illustrative capability table with a clear caveat that the comparison methods solve a harder (view-based, distortion-robust) problem, rather than presenting it as a performance benchmark. This would avoid the misleading impression of superiority.
3. Add a format-level validation for HDGER: demonstrate that the container .ply or checkpoint file is parseable by an unmodified Scaffold-GS loader and matches expected attribute dimensions.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to SecureGS |
|---|---|---|---|
| Poison-splat (3DGS security attack) | ExrEw8cVlU | 7.5 (Spotlight) | Stronger: reveals novel attack, formal adversarial evaluation; SecureGS's security claim is weaker by comparison |
| VideoShield (watermarking for video) | uzz3qAYy0D | 6.0 (Poster) | Comparable novelty in filling a gap; VideoShield's zero-shot design is cleaner; SecureGS has more comprehensive evaluation |
| WATER-GS (3DGS watermarking) | H48OMCCiI7 | 4.0 (Withdrawn) | Closest topic match; SecureGS is clearly stronger in novelty (RDO strategy), empirical results, and scope |
| Q00XEQxA45 (image compression+stego) | Q00XEQxA45 | 3.75 (Reject) | Lower-quality paper in steganography domain; SecureGS clearly exceeds this |
| PbheqxnO1e (3DGS compression) | PbheqxnO1e | 7.0 (Poster) | Comparable efficiency focus; SecureGS adds security but has weaker formal validation |

**Assessment**: SecureGS is clearly above WATER-GS (4.0) — it has more novel technical contributions (RDO strategy, decoupled representation), better empirical results, and addresses a more specific problem (geometric structure security) that prior work missed. It falls below Poison-splat (7.5) and PbheqxnO1e (7.0), which have either more formal evaluation or cleaner claimed scope. The closest landing zone is between WATER-GS (4.0) and VideoShield (6.0), with the major weakness of informal security evaluation pulling it below 6.0. The paper sits in the 5.0–5.5 range.

**Originality**: Moderate-high. The hybrid decoupled design and RDO strategy are novel.
**Importance**: Moderate. Addresses a real and underexplored security concern for 3DGS assets.
**Claim support**: Partial. Fidelity and efficiency claims are well supported; security claims are only qualitatively supported.
**Experimental soundness**: Adequate but with notable gaps in robustness and security evaluation.
**Clarity**: Good overall, with some framing issues in Table 3.
**Value to community**: Positive — introduces a well-motivated problem distinction and a technically sound approach.

**Final Score: 5.0** — The paper makes genuine contributions to 3DGS steganography with non-trivial technical novelty, but the primary claim of improved security is not quantitatively validated. This is above rejection (WATER-GS at 4.0) but not at acceptance-without-revision level due to the major weakness in the security evaluation that is central to the paper's framing.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>