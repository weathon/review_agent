Now I have a thorough understanding of the paper and the calibration landscape. Let me compile the final review.

## Summary

PIRN proposes a prototype-driven reconstruction framework for few-shot multimodal anomaly detection (RGB + 3D surface normals) with three components: Balanced Prototype Assignment (BPA) using balanced optimal transport to prevent codebook collapse, Adaptive Prototype Refinement (APR) using GRU-gated prototype updates at inference, and Multimodal Normality Communication (MNC) that exchanges prototype-level normality cues across modalities via graph alignment and cross-attention. The framework achieves consistent improvements over several baselines on MVTec-3D-AD, Eyecandies, and Real-IAD D3, with a notable 7× FLOPs reduction compared to the prior SOTA (FIND).

## Strengths

- **BPA via balanced optimal transport is well-motivated and principled.** The formulation of patch-to-prototype assignment as a balanced OT problem (Eqs. 1–2) with equal-mass constraints directly addresses codebook collapse. Figure 1 (right) provides visual evidence of more uniform prototype utilization, and the ablation (Table 2) confirms removing BPA drops AUROC_I from 0.922 to 0.883.

- **Significant and genuine computational efficiency advantage.** Table 4 shows PIRN achieves 0.922 AUROC_I with 103.36G FLOPs vs. FIND's 0.921 with 728.46G FLOPs on 10-shot MVTec-3D-AD — an 85% FLOPs reduction and 4.35× faster latency with no accuracy loss. This is a practically meaningful contribution for deployment.

- **Consistent improvements across three benchmarks and multiple shot settings** (Tables 1, 8), with the largest gains at the most data-scarce setting (5-shot), validating the few-shot motivation. Table 3 further confirms that combining modalities yields the biggest relative gains under 5-shot, supporting the MNC design rationale.

- **Thorough ablation studies.** Tables 2, 5, 6, and 7 systematically ablate each module, prototype count, decoder depth, and aggregation method, providing clear evidence for design decisions rather than leaving them as arbitrary choices.

- **MNC exchanges prototype-level rather than patch-level information across modalities.** This is a principled design for few-shot settings where dense cross-modal correspondence is unreliable, and the two-stage alignment-then-injection pipeline is well-structured (Section 3.4).

## Weaknesses

### Fatal
None.

### Major

- **FIND — the few-shot multimodal SOTA from the same first author's group — is omitted from the main comparison table (Table 1).** FIND (Li et al., 2025) is the most direct competitor for the exact problem setting (few-shot multimodal AD), yet it only appears in Table 4 (efficiency), where PIRN achieves 0.922 vs. FIND's 0.921 AUROC_I on 10-shot MVTec-3D-AD — a negligible +0.001 margin. The headline improvement of +3.9 (5-shot) and +3.7 (10-shot) is measured against INP-Former, an adapted 2D method, not against the actual SOTA. Since FIND shares the same first author (Yiting Li) and overlapping co-authors, the authors should have access to FIND's results across all shot settings. The selective inclusion creates a fundamentally misleading impression of the contribution's magnitude. Without FIND in Table 1, the reader cannot assess whether PIRN's advantage holds across all settings or is confined to the one where it happens to be shown. This undermines the paper's central claim of "consistently achieving superior performance."

- **APR's core design assumption — that anomalous patches are assigned diffusely across prototypes — is stated but not empirically verified.** Section 3.3 claims "an out-of-distribution (anomalous) patch tends to be assigned more diffusely across prototypes (i.e., with low affinity to any single prototype), thereby contributing weakly to each prototype context." This is the linchpin of APR: if it fails, prototypes are corrupted by anomalous information during inference, breaking the entire reconstruction framework. While this property is intuitively plausible under balanced OT constraints, the paper provides no direct evidence — e.g., measuring the entropy or concentration of OT assignments for anomalous vs. normal patches. The OT displacement visualization in Fig. 4 provides indirect support (anomalous tokens show larger displacements toward normal prototypes), but this shows reconstruction behavior, not assignment concentration. A simple experiment comparing max assignment probability or entropy of Γ* for synthetically anomalous vs. normal patches would test this critical assumption.

### Minor

- **The ablation study (Table 2) does not specify what replaces each component when removed.** When BPA is removed (row 2), does the model use softmax assignment? When APR is removed (row 3), are prototypes kept static? When MNC is removed (row 4), is there any cross-modal interaction? The "no modules" baseline (row 1, 0.828) is also undefined. Without specifying what each ablated configuration computes, the reader cannot fully interpret the component contributions. This is partially addressable in a rebuttal but reduces the ablation's informativeness as presented.

- **No standard deviations are reported across runs.** In few-shot settings with K=5 or 10, variance can be substantial. Whether PIRN's +3.7 improvement over INP-Former at 10-shot is statistically meaningful is unclear without variance estimates.

- **The conclusion overclaims "significant performance gains in challenging few-shot settings."** Relative to FIND (the actual SOTA), the gain is +0.001 AUROC_I at 10-shot. The honest assessment is that PIRN matches FIND's accuracy at 7× lower compute — which is a genuine and important contribution, but the phrasing should reflect this accurately.

### Trivial
None.

## Nice-to-Haves

- Per-category analysis on MVTec-3D-AD (currently in appendix) could be briefly summarized in the main text, highlighting categories where PIRN underperforms alongside those where it excels. This would strengthen the reader's ability to assess practical applicability.
- Empirical verification of APR's anomaly-diffusion assumption (e.g., a histogram of OT assignment entropy for normal vs. anomalous patches) would significantly strengthen the paper's theoretical grounding.
- Ablating K (prototype count) in few-shot settings rather than only all-shot (Table 5) would be informative, as the optimal codebook size may differ with fewer training samples.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "CFM achieving 0.845 on 10-shot MVTec is not a failure."** This is a presentation framing concern, not a substantive weakness. The paper's characterization that alignment methods "fail" in few-shot is relative — CFM does drop from ~0.954 (all-shot) to 0.845 (10-shot), a significant degradation. This is opinion, not a factual error.

- **Harsh critic: "The paper never demonstrates that memory-based methods actually misclassify unseen normal variations."** This motivation is well-established in the AD literature and is the standard justification for prototype-based approaches. Demanding that each paper re-demonstrate a well-known limitation is scope creep.

- **Harsh critic: "Computational cost of Sinkhorn iterations within each decoder layer at each forward pass is not discussed."** The paper does provide an efficiency comparison in Table 4 showing PIRN is the most efficient method. This concern is already addressed by the data.

- **Harsh critic: "KNN-based graph construction for prototype alignment across modalities assumes prototypes occupy a shared feature space."** The MNC module explicitly uses graph attention to *align* prototypes across modalities (Section 3.4, Stage 1), which is precisely the mechanism for bridging any initial feature-space gap. The alignment step addresses this concern.

- **Harsh critic: "Real-IAD D3 results show PIRN achieving 0.873 vs D3M's 0.890 — PIRN is not the best on image-level detection."** The paper already acknowledges this and explains that D3M uses tri-modal data. This is not a hidden weakness.

- **Strength finder: "Strong performance on Real-IAD D3 with fewer modalities"** — This strength is partially undermined by PIRN not being the best on image-level AUROC_I (0.873 vs D3M's 0.890). However, PIRN does achieve best AUROC_P, so this is still a reasonable supporting strength.

- **Strength finder: "The information bottleneck design is validated by codebook size ablation"** — This is a valid but minor supporting point. Table 5 does show K=100 degrading to 0.901, confirming the bottleneck effect, but this ablation is only in the all-shot setting.

## Novel Insights

The most important insight from reviewing this paper is that its strongest valid claim is not accuracy superiority but efficiency superiority: PIRN matches FIND's accuracy at 7× lower FLOPs. The three modules (BPA, APR, MNC) represent a fundamentally different architectural philosophy than FIND's approach — prototype-driven intra-modal reconstruction with lightweight cross-modal communication versus heavy reconstruction-based methods. This architectural difference is what enables the efficiency gain, and the paper would be substantially stronger if it foregrounded this angle rather than claiming "superior performance." The APR module's design of using balanced OT to implicitly filter anomalous contributions is clever but rests on an unverified assumption; the GRU gating provides a second safety layer, but neither mechanism has been stress-tested against the exact failure mode they're designed to prevent.

## Suggestions

- **Add FIND to all main comparison tables (Table 1) across all shot settings and datasets.** If FIND's published results don't cover some settings, re-run FIND in those settings or clearly note the gap. This is the single most impactful revision possible.
- **Reframe the contribution around efficiency + architectural novelty.** The honest headline is: "PIRN matches the few-shot multimodal SOTA (FIND) at 7× lower computational cost through a principled prototype-driven reconstruction framework." If PIRN genuinely outperforms FIND at 5-shot, that further strengthens the claim.
- **Add a diagnostic experiment for APR's anomaly-diffusion assumption.** Compute and visualize the entropy or max-assignment-concentration of Γ* for normal vs. anomalous patches during inference. This would turn an unverified assumption into a validated design choice.
- **Specify what replaces each component in the ablation.** Clarify whether softmax assignment replaces BPA, static prototypes replace APR, and no cross-modal interaction replaces MNC.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| UIP-AD | /home/wg25r/review_agent/human_reviews_2026/Mam9PS8ENb.md | 4.0 | Same domain (multimodal AD with prototypes), also missing key baselines, no ablation. PIRN is stronger: has thorough ablations, efficiency results, three benchmarks. |
| DPNR | /home/wg25r/review_agent/human_reviews_2026/iO9CRytDvf.md | 2.0 | Prototype-based AD but with factually incorrect claims and superficial experiments. PIRN is much stronger. |
| ESM/Diffusion | /home/wg25r/review_agent/human_reviews_2026/JFtDfVPLt7.md | 3.0 | Overclaimed "SOTA" with per-class hyperparameter overfitting. PIRN's overclaiming is less severe (the results are real, just measured against weaker baselines). |
| MFRM | /home/wg25r/review_agent/human_reviews_2026/WT1a0RLhwd.md | 5.0 | Unverified core design assumptions, solid methodology otherwise. Similar pattern to PIRN. |
| MRAD | /home/wg25r/review_agent/human_reviews_2026/TQkFiW3AEX.md | 6.0 | Good results with memory-based AD, minor baseline issues, accepted poster. PIRN has a bigger baseline issue but stronger efficiency story. |
| THEMIS | /home/wg25r/review_agent/human_reviews_2026/y3UkklvoW9.md | 7.33 | Comprehensive benchmark, well-executed. PIRN is not at this level. |

PIRN sits between MFRM (5.0, similar unverified core assumption) and MRAD (6.0, accepted with minor issues). PIRN's FIND omission is more serious than MRAD's baseline issues (FIND is the direct SOTA from the same group), but PIRN has stronger technical contributions and efficiency gains than MFRM. A score of 5.0 reflects that the paper has genuine contributions undermined by the misleading baseline comparison and unverified APR assumption, but is not fundamentally flawed.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>