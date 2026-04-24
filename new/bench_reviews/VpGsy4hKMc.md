Now let me search for calibration anchors.Now I have enough information to write the final review. Let me compile it.

---

## Summary

FreeSplatter is a feed-forward transformer framework that jointly reconstructs 3D Gaussians and estimates camera parameters from uncalibrated sparse-view images. The model predicts pixel-wise Gaussian maps in a unified reference frame, enabling instant camera pose recovery via PnP without global alignment. Two variants are provided: FreeSplatter-O (object-centric, trained on Objaverse) and FreeSplatter-S (scene-level, trained on BlendedMVS+ScanNet+++CO3Dv2). The core technical contributions are a pixel-alignment loss (𝓛_align) that enforces geometric ray consistency without depth supervision at test time, and a staged training strategy enabling stable convergence without known poses.

---

## Strengths

- **Pixel-alignment loss is the paper's strongest technical contribution (Table 3, Eq. 6):** The ablation shows 3.8 dB PSNR improvement on GSO and 4.5 dB on ScanNet++ when the loss is included — a decisive margin confirming its centrality. The principle (maximizing ray cosine similarity between predicted Gaussian centers and their corresponding camera rays) is clean, principled, and non-trivial.

- **Unified reference-frame design avoids pairwise global alignment (Section 3.2, Eq. 4):** Unlike DUSt3R/MASt3R, which process pairs of images and require iterative global alignment, FreeSplatter predicts all Gaussians in a single canonical frame and recovers poses via a single PnP pass. This is a meaningful architectural simplification with direct practical value.

- **Strong absolute reconstruction results (Table 2):** FreeSplatter-O achieves PSNR of 31.9 on OmniObject3D and 30.4 on GSO without any pose input. FreeSplatter-S surpasses pose-dependent pixelSplat and MVSplat on ScanNet++ reconstruction (PSNR 25.8 vs. 25.0 and 22.6). These numbers are credible and backed by qualitative results.

- **Competitive pose estimation on scene-level data (Table 1):** On ScanNet++, FreeSplatter-S achieves RRA@30° of 0.987 vs MASt3R's 0.993, and RRE of 0.791° vs 0.724°, representing near-parity with a model trained on far more data.

- **Staged training addresses a genuine cold-start problem (Section 3.3):** The insight that purely rendering-supervised training diverges when Gaussian positions are initialized randomly—and the proposed fix (geometric pre-training with 𝓛_pos followed by 𝓛_align fine-tuning)—is well-motivated and confirmed as essential by the authors.

- **Practical downstream integration (Section 4.5, Figure 6):** The pose-free design genuinely simplifies text/image-to-3D workflows by eliminating the need to manually align multi-view diffusion model poses with the reconstruction model.

---

## Weaknesses

### Fatal
None.

### Major

- **GS-LRM is absent from all comparison tables despite being the paper's direct architectural predecessor.** The paper explicitly acknowledges (Section 3.2) that "FreeSplatter employs a transformer architecture similar to GS-LRM." GS-LRM is a posed sparse-view Gaussian LRM trained on Objaverse with a compatible design. Its absence from Table 2 means the >5 dB PSNR advantage over LGM and InstantMesh cannot be cleanly attributed to the pose-free design versus architectural/training-data differences. The paper's most prominent interpretive claim—in Section 4.2, that FreeSplatter-O's results "suggest that camera poses may not be essential for developing high-quality, scalable reconstruction models"—requires a comparison with a properly posed sparse-view Gaussian reconstruction model (GS-LRM being the most natural choice) to be substantiated. Without this, the evidence supports that FreeSplatter outperforms generation-tuned pipelines in a reconstruction setting, but does not isolate the value of pose-free operation. This is not a fatal flaw in the method, but it is a significant evidentiary gap in the paper's core argumentative claim.

- **FreeSplatter-S is evaluated in-distribution on both ScanNet++ and CO3Dv2, its two main scene-level benchmarks.** The model trains on BlendedMVS+ScanNet+++CO3Dv2 and evaluates on held-out splits of ScanNet++ and CO3Dv2. The paper mentions "cross-dataset generalization results" in Appendix A.2.3, but the main paper's claims about generalization capability rest on in-distribution splits. The significance of performance advantages over Splat3R must be interpreted carefully in light of this fact.

### Minor

- **"On par with MASt3R" overstates the CO3Dv2 translation accuracy.** On CO3Dv2, FreeSplatter-S has TE=0.148 vs MASt3R's TE=0.112 — a 32% gap. On ScanNet++ the gap is much narrower (0.110 vs 0.104). While RRA metrics show near-parity, the abstract's unqualified claim of "on par with MASt3R" is misleading for one of the two benchmarks.

- **Depth-map dependency during pre-training is described as a minor limitation but is actually a structural data constraint.** The paper notes (Limitations) that 𝓛_pos "makes it non-trivial to be trained on datasets with no depth labels, e.g., RealEstate10K and MVImgNet." This is not merely an engineering inconvenience — it caps the training data to depth-labelled sets and directly explains both the gap with MASt3R (which trains on far more data) and the restriction to three scene datasets. Framing this as addressable by "future work" understates the architectural coupling.

- **Number of input views for scene-level evaluation is not stated in the main text.** Figure 1 suggests N=2 for scene-level reconstruction, which would make the comparison with 2-view architectures (pixelSplat, MVSplat) architecturally fair, but this is never explicitly confirmed in the experimental section. Stating N explicitly is a minor but necessary clarification.

- **Ablation coverage in the main paper is thin.** Table 3 ablates only 𝓛_align. The staged training (is 𝓛_pos pre-training essential, or merely helpful?), the reference/source embedding asymmetry (all non-reference views share the same learned embedding), and sensitivity to reference view choice are unaddressed. The paper defers several ablations to the appendix; those that are there likely cover some of these, but the main paper's ablation table is sparse for a model with multiple non-trivial design choices.

### Trivial
None worth mentioning.

---

## Nice-to-Haves

- A direct comparison between FreeSplatter-O and GS-LRM on GSO/OmniObject3D would cleanly disentangle architectural vs. pose-free contributions and would substantially strengthen (or refine) the core claim of Section 4.2.
- A sensitivity analysis of reconstruction quality versus reference view quality (e.g., what happens when the first view is occluded or textureless) would characterize real-world robustness.
- Reporting cross-dataset generalization (e.g., Tanks and Temples, ETH3D) for FreeSplatter-S in the main paper rather than only in the appendix would strengthen generalization claims.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **[Harsh Critic, Issue 2 — "2-view mismatch in scene-level comparison"]:** The critic claims pixelSplat and MVSplat are "architecturally 2-view models" and that FreeSplatter-S may use more views, creating an unfair advantage. Figure 1 shows N=2 input views for scene-level, suggesting the comparison is architecturally fair. The underlying concern about stating N explicitly is retained as a Minor weakness above. The framing of "structural information advantage" is removed as unverified.

- **[Harsh Critic — "pixel-alignment loss uses pose supervision"]:** The critic argues that 𝓛_align implicitly uses poses during training, "blurring the 'no camera poses required' framing." The paper is clear throughout that it requires poses *at training time* (like PF-LRM, LEAP, and all other pose-free methods) but not at *inference time*. This is standard and acknowledged; it is not a misframing.

- **[Harsh Critic — COLMAP exclusion is problematic]:** COLMAP is known to fail on sparse-view scenarios and the paper cites prior evidence for this. Excluding it from evaluation is standard practice and not a methodological flaw.

- **[Harsh Critic — FreeSplatter-S included in Table 1 object-centric columns as a distractor]:** The critic calls the inclusion of FreeSplatter-S performance on OmniObject3D/GSO a confusing choice. This is a minor presentation choice that does not misrepresent the model; FreeSplatter-S simply performs worse out-of-domain, which is informative.

- **[Strength Finder — "Versatility" as a strength]:** The paper trains two entirely separate models (FreeSplatter-O and FreeSplatter-S) that share architecture but require different data, masking strategies, and training objectives. Calling this "versatility through a single architecture" is somewhat misleading, as the paper itself lists the need for a unified model as a future limitation.

---

## Novel Insights

The paper's most intellectually interesting contribution, beyond its empirical results, is the insight that the pixel-alignment loss (𝓛_align) can substitute for both ground-truth depth at test time and explicit epipolar-line constraints during feature aggregation. By enforcing ray cosine similarity rather than direct depth supervision, the model learns geometrically consistent Gaussian placement without the need for dense depth labels after pre-training, while simultaneously improving PnP pose estimation quality. This decoupling of geometric regularization from depth-label availability is a generalizable technique worth highlighting: it could benefit other pose-free reconstruction architectures that struggle with Gaussian position initialization.

---

## Suggestions

1. **Add GS-LRM to Table 2.** This is the single most impactful experiment missing from the paper. If FreeSplatter-O still outperforms a posed GS-LRM, the claim that "poses may not be essential" becomes strongly supported; if not, the paper should temper that claim accordingly.
2. **Explicitly state N for scene-level evaluation** in Section 4.1. Even one sentence clarifies the comparison setup.
3. **Soften "on par with MASt3R" in the abstract** to "competitive with MASt3R on ScanNet++ and within range on CO3Dv2" to accurately reflect Table 1.
4. **Promote the cross-dataset generalization results (Appendix A.2.3) to the main paper**, even in compact form, to validate generalization claims that are currently resting on in-distribution evaluation.
5. **Expand the ablation** to include staged training (with/without 𝓛_pos pre-training) and the effect of varying N for scene-level.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison to FreeSplatter |
|---|---|---|---|
| NoPoSplat (Oral, high) | P4o9akekdf.md | 8.0 | Nearly identical problem setting; also pose-free Gaussian from sparse views in unified frame. NoPoSplat uses no depth data at all — a cleaner design — but FreeSplatter handles both objects AND scenes and provides explicit camera estimation. Comparable overall scope. |
| PF-LRM (Spotlight, high) | noe76eRcPC.md | 8.0 | Pose-free LRM with PnP; FreeSplatter extends to Gaussian representation and scene-level — broader contribution. |
| LEAP (Poster, medium-high) | KPmajBxEaF.md | 7.0 | Pose-free 3D modeling; narrower (NeRF, object-only); FreeSplatter is arguably richer in scope and results. |
| SHARE (Reject, medium-low) | EAT5Jpa4ws.md | 5.5 | Pose-free Gaussian splatting without MASt3R comparison; FreeSplatter clearly stronger in baseline coverage and results. |
| DepthSplat (Reject, medium-low) | IcPkW3QNW2.md | 5.0 | Gaussian + depth; no pose-free element; different contribution class. Used as low anchor. |
| studentSplat (Reject, low) | fRXAQfHlmr.md | 4.25 | Single-view scene Gaussian; clearly weaker contribution; used as low anchor. |

**Positioning:** FreeSplatter sits above SHARE (5.5) and LEAP (7.0) in completeness of evaluation and breadth of contribution. Its main weaknesses — missing GS-LRM baseline and in-distribution evaluation — are the same kind of issues that reviewers raised against NoPoSplat (8.0 Oral) and PF-LRM (8.0 Spotlight) but the community still accepted those strongly. The GS-LRM gap is real but does not invalidate the core method. The technical contributions (pixel-alignment loss, staged training, unified-frame Gaussian prediction for both object and scene) are concrete and verified.

Balancing against NoPoSplat (8.0) as the most similar strong paper: FreeSplatter is comparable in scope and somewhat broader (two scenarios), but has the missing GS-LRM baseline and relies on depth data for pre-training (NoPoSplat does not). This justifies a score slightly below 8.0.

**Final Score: 7.0 — Accept (Poster)**

**Originality:** Good — unified Gaussian prediction for pose-free reconstruction across two distinct scenarios is novel, and the pixel-alignment loss is a meaningful methodological contribution.  
**Importance:** High — pose-free sparse-view reconstruction is a key unsolved problem, and this paper makes practical progress.  
**Claim support:** Moderate — strong empirical results, but the headline interpretive claim ("poses may not be essential") is not fully supported without a GS-LRM baseline.  
**Soundness:** Good — no technical errors identified; staged training and ablations are appropriate.  
**Clarity:** Good — methodology is clearly explained; experimental section has a minor gap (N for scene-level).  
**Value to community:** High — the method is practically useful for downstream 3D generation workflows and provides a cleaner baseline for future pose-free reconstruction work.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>