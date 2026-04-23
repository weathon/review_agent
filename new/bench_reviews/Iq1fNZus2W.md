Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

The paper proposes Patch-wise and Keyword-Aware Attention (PKA), a framework for efficient multi-condition control in Diffusion Transformers (DiTs). By analyzing attention sparsity patterns in existing multi-condition DiTs (OminiControl), the authors introduce two specialized modules: Position-Aligned Attention (PAA) for spatial-aligned conditions (reducing O(N²) to O(N)) and Keyword-Scoped Attention (KSA) for subject-driven conditions. Combined with a Condition Cache and an early-timestep sampling strategy, the method reports up to 10× inference speedup and 5.12× VRAM reduction relative to the full-attention baseline, while improving generative quality (FID, SSIM) on most benchmarked tasks.

---

## Strengths

- **Principled empirical motivation (Figures 2 and 3):** Attention patterns in spatial-aligned multi-condition DiTs are demonstrably diagonal-concentrated, and subject-driven attention is keyword-activated and sparse. These observations are clearly visualized and directly justify the distinct PAA and KSA designs, making the efficiency rationale more grounded than many generic sparsification proposals.

- **Condition Cache as a structural consequence (Figure 4b):** By restricting condition tokens to self-attention within their own modality (never cross-attending to image tokens), the K/V projections of condition tokens become timestep-independent, enabling cache-and-reuse across the entire denoising trajectory. This is an elegant insight connecting architectural isolation to an inference efficiency mechanism.

- **Consistent quality improvement across most metrics (Table 1):** PKA achieves the best FID across all three tasks (52.99, 62.08, 53.01 vs. UniCombine's 61.03, 70.22, 67.40), best SSIM, and best subject consistency (CLIP-I/DINOv2) on Subject-Canny and Subject-Depth, while being more efficient. Outperforming a full-attention baseline on generative quality metrics is noteworthy for an efficiency method.

- **Early-timestep sampling grounded in perturbation analysis (Figures 5 and 11):** The paper presents a controlled experiment showing SSIM degradation is more severe under early-stage perturbation, motivating the shifted logit-normal sampling (μ > 0, δ > 1). Figure 11 confirms this yields faster convergence and better control fidelity, providing empirical backing for the training-stage design choice.

- **Scalability advantage is proportional to the problem (Figures 7 and 8):** The quadratic baseline cost grows with condition count; PKA's speedup increases correspondingly from ~3.9× at 2 conditions to ~10× at many conditions, confirming that the method effectively counteracts the specific bottleneck it targets.

---

## Weaknesses

### Fatal
None.

### Major

- **Baseline training setup is unspecified, making quality comparisons uninterpretable (Section 4.1).** The paper states that PKA is fine-tuned on a curated Subject200K subset for 20,000 iterations with LoRA. It does not state whether OminiControl2 and UniCombine are used off-the-shelf (trained on their own data under their own protocols) or fine-tuned on the same subset with the same iteration budget. If the baselines are off-the-shelf while PKA is domain-fine-tuned, the quality margins in Table 1 (up to ~20 FID points) may reflect data distribution alignment rather than architectural advantage. The phrase "to ensure a fair comparison, we fine-tune the FLUX.1 model using LoRA" is self-referential—it describes what PKA does, not whether the baselines were treated equivalently. This ambiguity undermines every quantitative quality claim in the paper and needs explicit clarification with evidence (e.g., baseline fine-tuning logs or a controlled ablation).

- **Headline efficiency figures are not backed by quality evaluation at the same condition count (Sections 4.1–4.2).** Quality experiments in Table 1 cover only 2-condition setups (Subject+Canny, Subject+Depth, Canny+Depth), where the actual measured speedup is ~3.9× and VRAM reduction ~2.46×. The abstract and conclusion foreground "up to 10×/5.12×," which are values measured only at high condition counts (Figures 7 and 8) where no quality evaluation exists. The claim "maintaining or improving generative quality" is verified only in the ~3.9× regime; the 10× regime is unvalidated. This is an internal inconsistency in the evidence.

- **Subject-Canny F1 drop is material and misrepresented.** PKA achieves F1=0.414 vs. UniCombine's F1=0.551 on Subject-Canny — a 25% relative reduction in edge controllability. The paper describes this as a "minor exception of a narrow margin" (Section 4.2.3). For a paper that claims "maintaining or improving generative quality" and whose title foregrounds "fine-grained control," a 25% relative drop in the primary spatial controllability metric is not narrow. The paper provides no analysis of why PAA's one-to-one patch alignment causes this regression specifically on Subject-Canny but not Canny-Depth. This weakens the abstract's "maintaining" claim and should be analyzed rather than dismissed.

### Minor

- **Ablation studies report efficiency only, not quality (Sections 4.3.1 and 4.3.2).** The PAA ablation compares PAA vs. full attention vs. SWA variants on latency and VRAM (qualitative figures only). The KSA ablation compares ε settings similarly. Neither ablation reports Table 1's quantitative quality metrics (FID, SSIM, F1, MSE, CLIP-I). This means the paper cannot directly attribute quality outcomes to individual components—it is unknown whether the efficiency gains from PAA or KSA come with quality costs in the 2-condition setting.

- **Condition Cache is not ablated for quality impact (Section 3.2).** The cache mechanism is presented as a consequence of condition self-attention isolation, but no experiment compares cached vs. non-cached inference at the same attention sparsity. The reader cannot determine whether quality is maintained because the cache approximation is accurate (condition representations genuinely do not depend on image state) or despite it.

- **KSA mask reliability at high-noise timesteps is uncharacterized (Section 3.3).** The paper argues that conditioning is most important during early (high-noise) denoising steps, yet the KSA mask is computed from attention between noisy image queries and keyword tokens. At high noise, the image is largely uninformative for spatial localization. The paper provides no characterization of mask accuracy across timestep ranges (e.g., IoU between KSA mask and ground-truth object segmentation), which is the key assumption underpinning the approach.

- **Test set size and FID validity unspecified.** FID estimates are unreliable with small sample counts. The paper reports evaluation on "a subset from Subject200K" without stating how many images are in the test split. With fewer than a few thousand samples, FID variance is large and the reported differences (52.99 vs. 61.03) could be within noise.

### Trivial

- Section 4.2.3's characterization of the Subject-Canny F1 gap as "narrow" should be revised to accurately reflect the magnitude.

---

## Nice-to-Haves

- Report one quantitative quality experiment (FID/SSIM/F1) at 4+ conditions to validate that quality holds in the regime where the headline speedup figures are measured.
- Report PAA/KSA ablation rows directly in Table 1 format so component contributions to quality are directly attributable.
- Analyze why Subject-Canny F1 drops more than Canny-Depth F1—whether this is caused by cross-condition interaction or the PAA alignment itself.
- Report total peak GPU VRAM rather than attention-module VRAM alone, so practitioners can assess practical hardware benefit.
- Provide mask IoU vs. ground-truth segmentation across timestep ranges to validate KSA's correctness assumption.
- Provide failure case visualizations to accompany the positive qualitative results.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic, Section 3.3 — SSIM perturbation analysis confounds coarse layout and conditioning:** The critic argues that SSIM comparing generated images (not vs. ground truth) conflates "coarse layout is determined early" with "conditioning is strongest early." This is a valid theoretical nuance, but the practical conclusion (that early-timestep training helps conditioning) is empirically confirmed by Figure 11's training curves. The distinction does not invalidate the training strategy. Removed as overly speculative.

- **Harsh Critic, Section 3.2 — PAA resolution and aspect ratio compatibility:** The concern that PAA requires compatible spatial resolution/tokenization grid is a reasonable engineering question, but the paper applies to FLUX.1 which has well-defined tokenization; this is an implementation detail rather than a methodological gap. Removed as out of scope.

- **Strength Finder — "LoRA fine-tune on FLUX.1 makes it practical" (Section 4.1):** Generic strength without substantive analytical content. No specific figure/table supports this as architecturally novel. Removed as non-specific.

---

## Novel Insights

The most genuinely novel observation is that distinct condition types in multi-condition DiTs admit *structurally different sparsity patterns* — spatial conditions produce position-aligned diagonal attention, while subject conditions produce keyword-activated sparse attention — and that these patterns can be exploited by specialized modules with provably different complexity classes (O(N) vs. sparse O(N)). The connection between condition-self-attention isolation (the architectural constraint) and KV-cache validity (the implementation consequence) is elegant and potentially generalizable to other condition-injecting architectures. The perturbation analysis showing that early denoising timesteps matter most for conditioning, motivating a shifted training distribution, is an underexplored but practically useful empirical finding.

---

## Suggestions

1. **Explicitly document baseline training setup.** State whether OminiControl2 and UniCombine are used off-the-shelf or fine-tuned on the same data under the same protocol. If they are off-the-shelf, add a fine-tuned baseline row to disentangle data effects from architectural effects.
2. **Retitle or qualify the efficiency claim.** Change "up to 10× while maintaining quality" to "up to 10× speedup (at high condition counts); quality maintained at 2-condition setting with ~3.9× speedup," matching what the evidence actually shows.
3. **Add quantitative quality rows to ablation tables** (PAA, KSA, cache) using the same metrics as Table 1.
4. **Address the Subject-Canny F1 regression honestly.** Either provide an analysis of why PAA loses cross-patch context for Canny edges under subject interaction, or acknowledge this as a known limitation of one-to-one alignment.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Differential Transformer | `/home/wg25r/review_agent/human_reviews/OvoCm1gGhN.md` | 8.0 | Much more foundational contribution to attention mechanisms; extensive ablation; PKA is applied/engineering in scope |
| REPA (DiT training) | `/home/wg25r/review_agent/human_reviews/DJSZGGZYVi.md` | 9.0 | Oral-level; strong theoretical grounding + empirical proof; substantially more rigorous than PKA |
| RL-based conditional diffusion | `/home/wg25r/review_agent/human_reviews/svp1EBA6hA.md` | 6.5 | Controllable generation poster; similar novelty scope, better experimental controls |
| Uncertainty-aware reward modeling | `/home/wg25r/review_agent/human_reviews/eC2ICbECNM.md` | 6.0 | Conditional generation; similar empirical validation quality |
| Star Attention (sparse LLM) | `/home/wg25r/review_agent/human_reviews/KVLnLKjymq.md` | 5.5 | Sparse attention efficiency claim; rejected; somewhat similar issue of efficiency-only ablations |
| FreeLM (overclaim) | `/home/wg25r/review_agent/human_reviews/qgLyKwXVDs.md` | 2.0 | Low anchor; fundamentally overclaimed results with missing baselines; PKA's issues are less severe than this |
| Brain-inspired recognition | `/home/wg25r/review_agent/human_reviews/epFk8e470p.md` | 1.67 | Low anchor; fundamentally flawed experimental design; PKA is substantially better validated |

**Reasoning:** PKA sits between the medium papers (5.5–6.5) that were accepted as posters with real but contained issues, and below the high-scoring papers (8–9) that had rigorous multi-faceted validation. The three major issues — ambiguous baseline training setup, quality not measured at the speed-claimed regime, and a 25% F1 regression mischaracterized as minor — are real methodological gaps that prevent a confident acceptance. However, the core attention sparsity analysis is principled, the efficiency contributions are real, and the method addresses a genuine problem. This places PKA in the 5–5.5 range of the calibration scale, comparable to Star Attention (5.5, rejected for similar issues of efficiency-only ablations and overclaimed claims) but slightly above the low end due to having more substantive quality evidence and a cleaner empirical motivation.

**Final Score: 5.0**

The paper is a reasonable applied contribution to efficient multi-condition DiT control with a well-motivated design, but the experimental rigor is insufficient to support the headline quality and efficiency claims in combination. The ambiguous baseline training setup is the most pressing concern; if resolved favorably, a score of 6 would be warranted.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>