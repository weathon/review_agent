## Summary

Phidias is a reference-augmented diffusion framework for 3D content generation that conditions a pretrained multi-view diffusion model (Zero123++) on retrieved or user-provided 3D reference models, represented as Canonical Coordinate Maps (CCMs). To address the "Misalignment Dilemma" between the input concept image and the reference shape, it introduces three components: meta-ControlNet (adaptive conditioning strength based on image-reference similarity), dynamic reference routing (coarse-to-fine resolution scheduling across denoising timesteps), and self-reference augmentation with curriculum training. The resulting system supports image-to-3D, text-to-3D, 3D-to-3D variation, interactive coarse guidance, and completion in a unified framework.

---

## Strengths

- **CCM-based geometry conditioning specifically designed to suppress texture conflicts.** Rather than conditioning on textured renders, the paper uses Canonical Coordinate Maps, which encode geometry while discarding texture. This is a well-reasoned design choice with direct bearing on the misalignment problem: a reference that shares shape but differs in texture will not introduce spurious color conflicts in the diffusion conditioning signal.

- **Meta-ControlNet demonstrably solves a real failure mode.** The ablation in Fig. 6(a) and Table 3 is convincing: without the meta-controller, the base model trained on retrieved references learns to *ignore* the reference entirely (the disconnected boat example), whereas meta-ControlNet recovers full use of the reference. The gain in PSNR from 14.70 to 16.35 from adding this component alone is substantial and meaningful.

- **Self-reference augmentation delivers the largest single-component gain (PSNR 14.70→16.57, LPIPS 0.227→0.182 in Table 3),** validating the core insight that progressive curriculum training with increasingly difficult augmentations bridges the distribution gap between self-referenced training pairs and noisy retrieval at inference time.

- **Transparent and informative Table 4.** The paper honestly reports performance across retrieval quality levels, including the case where random references actively degrade results below "no reference." This level of transparency is commendable and provides a realistic picture of the method's operating envelope.

- **User-adjustable reference strength λ (Fig. 12)** provides a smooth interpolation knob between concept-faithful and reference-following generation, enabling practical use cases (re-texturing, interpolation) that go beyond a binary reference/no-reference choice.

---

## Weaknesses

### Fatal
None.

### Major

- **Reconstruction stage confound undermines baseline comparisons.** The paper finetuned LGM from 4 views at 256×256 to 6 views at 320×320 for stage 2. Table 4 shows that "Without Reference" (Phidias + enhanced reconstructor, no reference at inference) achieves PSNR 15.90 — already outperforming LGM (14.80), CRM (16.35 is close), and InstantMesh (14.63) on visual quality metrics. Without a controlled experiment — either (a) reporting baselines with the same finetuned reconstructor, or (b) reporting Phidias's own "no reference" row in Table 1 alongside baselines — it is impossible to attribute the reported improvements to the reference mechanism rather than the upgraded reconstruction pipeline. This is the most significant methodological gap in the paper.

- **Dynamic Reference Routing shows negligible quantitative improvement.** Table 3 reports gains of PSNR 14.70→14.76, CD 0.0424→0.0420, F-Score 0.826→0.826 when this component is added in isolation. The qualitative example in Fig. 6(b) is illustrative, but the quantitative evidence is insufficient to support treating this as a standalone key contribution of equal standing to meta-ControlNet and self-reference augmentation. Either a more sensitive evaluation (e.g., stratified by reference-image similarity, or measured on the multi-view images before reconstruction) is needed, or the claim about this component should be substantially qualified.

- **Random reference actively degrades performance below "no reference" (PSNR 14.74 vs 15.90, Table 4), contradicting the robustness claim.** Sec. 4.2 and Fig. 7(b) assert that "Phidias will ignore inappropriate 3D reference … demonstrating robustness to some extreme cases." Table 4 directly contradicts this: a random reference is measurably worse than no reference at all across almost all metrics. The meta-ControlNet does not fully suppress misleading references; it partially mitigates but does not eliminate the harm. The paper's claim of robustness is overstated relative to the evidence.

- **Meta-ControlNet implementation is underspecified.** The meta-controller's mechanism for learning alignment-aware signals is not made explicit: (1) similarity between concept image and reference is not explicitly defined or supervised — it appears to be learned end-to-end through reconstruction loss alone, but this is never stated; (2) whether `y_meta1` is injected additively, via cross-attention, or by other means is not stated; (3) whether `λ` in Eq. (3) is fixed during training or learned is unspecified. These omissions prevent straightforward reproduction of the core architectural contribution.

### Minor

- **No quantitative evaluation for text-to-3D and 3D-to-3D.** Section 5 presents visually compelling examples for four additional applications, but the "unified framework" framing is unsupported without at minimum a small quantitative benchmark for the other modalities. Fig. 8 and Fig. 9 alone cannot substantiate the multi-modal unification claim.

- **Geometry metrics (CD, F-score) show marginal gains over baselines with retrieved references.** CD 0.0402 vs. LGM's 0.0398 and F-Score 0.833 vs. 0.831 are within measurement noise. The authors partially explain this as a metric-GT alignment issue when the reference steers geometry away from GT, which is a valid argument, but a more principled analysis (e.g., restricting evaluation to objects where top-1 retrieval similarity is high) is needed to make this argument quantitatively convincing rather than a post-hoc rationalization.

- **User study protocol underspecified.** The study reports strong preference rates (88–96%) from 30 users, but provides no information on: the number of pairwise comparisons, what was shown (turntable, static renders, single view), whether the reference was shown to raters, the exact evaluation question, or blinding procedure. With only 30 users and no protocol details, the strength of this evidence is unclear.

### Tiny

- Self-reference augmentation details (augmentation types, ranges, semantic augmentation procedure) are entirely deferred to Appendix A.5 despite being a core contribution. At least a brief summary in the main text would be expected.
- The curriculum training schedule (step count, epoch proportions, ratio of self-augmented to retrieved references over time) is not described anywhere accessible in the main paper.
- Fig. 12 demonstrates the λ-controllability on a single object pair; generalization of this effect across categories and reference types is unverified.

---

## Nice-to-Haves

- **Ablation isolating the reconstruction stage upgrade.** Reporting at least "Phidias (no reference, original 4-view 256px reconstructor)" and ideally baselines re-evaluated with the finetuned 6-view reconstructor would cleanly separate the reference mechanism's contribution from the improved reconstruction stage.
- **Correlating meta-ControlNet modulation with measured similarity.** A visualization or scatter plot showing how the effective conditioning strength varies with image-reference CLIP similarity across a test set would provide direct evidence that the meta-controller works as described.
- **Systematic λ-controllability grid** across multiple object categories and reference similarity levels to quantify whether controllability is consistent or category-dependent.
- **Inference time and retrieval latency analysis.** A brief table comparing inference wall-clock time (including retrieval) against baselines would help practitioners assess trade-offs.

---

## Removed Points

*These points were flagged for removal; treat them with caution as they may be factually incorrect, scope creep, or do not meet the removal criteria strictly.*

- **Critic: "GT Reference result is unrealistic and overstates real-world gains."** The paper explicitly labels this result as an upper bound and acknowledges "actual performance should be between Ours (GT Ref.) and Ours (Retrieved Ref.)." The inclusion is transparent and informative, not misleading. Removed.

- **Critic: "No confidence intervals or statistical uncertainty."** Single-run evaluation on a 200-sample benchmark is the norm in 3D generation literature, not a methodological deficiency. Removed.

- **Critic: "First reference-based 3D-aware diffusion model claim is too strong."** The paper specifically distinguishes its feed-forward approach from per-case optimization methods (Wu & Zheng, 2022; Wang et al., 2024b), which is a clear and defensible distinction. The claim is sufficiently scoped. Removed.

- **Critic: "Comparison with baselines is unfair because baselines lack 3D reference."** The comparison intentionally shows what is achievable without references versus with; baselines are not harmed by the comparison asymmetry — they are simply evaluated in their standard setting. This is standard practice. Removed.

- **Critic: "Absence of theoretical justification for meta-ControlNet architecture."** This is an empirical systems paper; demanding theoretical proofs for an architectural design choice is not standard in this subfield. Removed.

- **Spark Finder: "No comparison with reference-based or retrieval-augmented 3D baselines."** The optimization-based methods cited (Wang et al., 2024b; Wu & Zheng, 2022) operate in a fundamentally different setting (per-object optimization, minutes per object vs. feed-forward inference). A direct numerical comparison would not be meaningful. The paper's positioning against these methods as a speed-focused alternative is reasonable. Removed as a strict weakness; kept as a nice-to-have to show qualitative comparisons.

- **Critic: "Ablation should include all pairwise combinations of modules."** The current ablation (each component added individually plus full model) covers the main contributions. Requiring all 2^n combinations is not standard. Removed.

- **Critic: "Broader impact section underdiscussed."** A style and format comment; not evaluated here per instructions. Removed.

---

## Novel Insights

The most genuinely novel observation across the reviews — one not fully surfaced in the paper itself — is the implicit interaction between self-reference augmentation and meta-ControlNet: the curriculum training scheme essentially teaches the meta-controller what "easy" and "hard" references look like by gradually exposing it to a spectrum of similarity levels, creating a bootstrapping dynamic where the meta-controller's adaptivity is a *consequence* of the curriculum rather than purely an architectural property. This explains why training with retrieved references from the start fails (Sec. 3.4) — without the similarity spectrum shaped by curriculum augmentation, the meta-controller has no gradient signal to learn when to suppress or amplify conditioning. The paper treats these as separate contributions, but they are more tightly coupled than presented, and documenting this interaction explicitly would clarify the design rationale and help future work reuse or modify the training procedure.

---

## Suggestions

1. **Add a "Phidias without reference" row to Table 1** (same 6-view finetuned reconstructor, no reference at inference) so readers can directly attribute observed gains to the reference mechanism versus the reconstruction upgrade.

2. **Report an ablation where at least one strong baseline (e.g., LGM) is re-evaluated with the same finetuned 6-view 320px reconstructor** to quantify the standalone contribution of the improved reconstruction stage.

3. **Expand Sec. 3.2** with explicit description of: (a) whether similarity is explicitly supervised or purely end-to-end; (b) how meta-signals are injected (additive/multiplicative/attention); (c) the role of λ during training versus inference.

4. **Qualify the robustness claim in Sec. 4.2** to be consistent with Table 4, e.g., "the model partially mitigates harm from highly misaligned references but does not fully suppress them, as evidenced by random references underperforming the no-reference baseline."

5. **Provide at minimum a small quantitative evaluation (even 50 examples)** for text-to-3D to support the "unified framework" claim, or explicitly scope text-to-3D and 3D-to-3D as qualitative demonstrations and remove the unification framing from the primary contributions.

6. **Expand the dynamic reference routing ablation** with a timestep-stratified analysis or intermediate denoising visualization, given that the current quantitative gains are marginal and the contribution needs stronger evidence.

---

## Evaluation

| Axis | Assessment |
|------|------------|
| **Originality** | Solid. Applying the RAG paradigm to 3D diffusion via CCM conditioning and explicit misalignment handling is a fresh and well-motivated contribution. The meta-ControlNet and curriculum training design are non-trivial extensions beyond vanilla ControlNet. |
| **Importance of research question** | High. Reference-guided 3D generation directly addresses practical bottlenecks in content creation workflows, and the misalignment dilemma is a real obstacle that the paper frames well. |
| **Claims well supported** | Partially. The image-to-3D claims are supported, the meta-ControlNet and self-reference augmentation contributions are convincingly demonstrated. However, the "unified framework" claim lacks quantitative validation for two of three modalities, and the robustness claim is directly contradicted by Table 4. The reconstruction stage confound weakens all comparative gains. |
| **Soundness of experiments** | Moderate. The ablation design is reasonable, and Table 4 is admirably transparent. However, the missing "same reconstructor" control is a notable gap that makes it impossible to fully interpret the main comparison table. Dynamic routing gains are negligibly small. |
| **Clarity of writing** | Good. The narrative is coherent, figures are helpful, and the motivation is well articulated. The main weakness is technical underspecification in the architectural description of meta-ControlNet. |
| **Value to the research community** | Meaningful. The framework is versatile, the applications are practically motivated, and the trained model is likely useful to practitioners. The λ-control knob and the curriculum training strategy are reusable ideas. |
| **Contextualized relative to prior work** | Adequate. The distinction from per-case optimization is clear. The paper would benefit from a more explicit discussion of how much the design borrows from vs. extends ControlNet and Zero123++. |

Overall, this is a paper with a genuinely useful idea, strong qualitative results, and credible core contributions (meta-ControlNet, self-reference augmentation), but with a methodological gap (reconstruction confound) and overstated claims (robustness, unified framework) that the authors should address. As it stands, it reads as a solid but not definitive contribution.