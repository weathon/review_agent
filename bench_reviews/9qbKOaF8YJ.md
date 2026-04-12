## Summary
This paper studies class-incremental semantic segmentation without old-data replay and argues that standard KD over-preserves old representations, causing parameter competition and underuse of previously acquired knowledge. It proposes DKD, a three-part objective combining (i) pruning-based “parameter release” for the old model with an old-distribution matching loss, (ii) a Laplacian/projection-based construction of reusable old-knowledge maps, and (iii) an entropy-based objective intended to maximize shared knowledge between old and new distributions. Empirically, the method is strong on VOC and ADE20K across many incremental settings, especially with a ViT backbone, but some core technical parts are underspecified enough that the mechanism is not yet fully convincing from the paper alone.

## Strengths
- **The paper targets a specific and plausible failure mode of KD-based CISS—over-constraining the student to preserve old distributions in a fixed-capacity model—and builds the method around that diagnosis.** This is more specific than the generic “stability-plasticity tradeoff” framing. The motivation is visible in the method design: `L_Min` weakens the old teacher signal after pruning, while `L_Esti`/`L_Max` try to turn old knowledge into guidance for new learning rather than only a retention constraint.
- **Empirical results are genuinely strong across a broad set of CISS settings.** On VOC (Table 1), the method is competitive or best across 10-1, 2-2, 15-1, 19-1, and 15-5; on ADE20K (Table 2), it is similarly strong across four settings. This is not a single-split win. The paper also includes additional disjoint-setting results and class-wise analyses in the appendix.
- **The ablations are more informative than usual and support that all three losses matter.** Table 12 is especially useful: the full combination improves the old/new balance more than single components alone, and `L_Min` appears particularly important in harder multi-step settings like 10-1.
- **The paper provides nontrivial robustness reporting rather than only single-run headline numbers.** Tables 5 and 13 report repeated-run variability, and the deviations are indeed small in the presented settings.
- **The method appears architecturally somewhat portable rather than being tied only to the paper’s full ViT recipe.** Appendix C.3/C.4 shows DKD added to CoinSeg with both ResNet101 and ViT backbones, with gains on the incremental class and modest overall improvements. That helps support DKD as a transferable training strategy, not only a one-off system.

## Weaknesses

### Major:
- **The central “parameter release” claim is not fully substantiated as a mechanism on the student model.**  
  In Section 3.2(a), pruning is applied to the **old model**: “the release is performed once per step for the old model” (Appendix A.1), and the current model is then trained to match the **pruned old model** through `L_Min`. This clearly weakens the distillation target, but the paper repeatedly phrases this as if it literally frees capacity in the current model. As written, there is no persistent mask, structural sparsification, or direct constraint on student parameters showing that student capacity is actually “released”; instead, the method relaxes the old-knowledge target. That may still be useful empirically, but it is a weaker claim than the paper often makes. A more careful interpretation would be “target relaxation” rather than demonstrated parameter liberation in the learner.
- **`L_Esti` is insufficiently specified and partly mathematically unclear in the main paper.**  
  Equation (4) defines a position map via second-order spatial derivatives of a feature-difference quantity, but the paper does not clearly explain the discrete implementation used in training. The reviewer’s claim that this is entirely “unimplementable” overstates the issue, but the concern about underspecification is valid: for such an operation, readers need to know whether this is a fixed Laplacian kernel, finite differences, autograd-based second derivatives, or something else.  
  More importantly, Equation (5) is hard to parse dimensionally from the main text: `C_t(h,w) = < y_c^*(h,w), f_t(h,w) > / ||f_t(h,w)||_2`, while Eq. (2) defines `y_c^*` as an indicator-style pseudo-label quantity over old classes. The appendix later rewrites this in a way that suggests a vector quantity, but the main presentation does not make that representation explicit. This weakens technical clarity at a core part of the method.
- **The paper’s novelty claims around the entropy term are somewhat overstated relative to what is clearly established.**  
  `L_Max` is presented as maximizing shared knowledge distribution using marginal/conditional entropy. The formulation is reasonable and may be useful here, but from the paper itself it reads more like an information-theoretic regularizer encouraging batch diversity plus low per-sample entropy than a clearly new mechanism unique to CISS. The contribution is better justified as the integration with pruning/knowledge reuse, not as a standalone conceptual advance.
- **Some headline empirical framing is too strong for the actual numbers.**  
  The paper frequently emphasizes “near-upper-bound” or “approaches joint training.” This is credible in some settings, especially average ADE20K summaries, but not uniformly. For example, in VOC 2-2, Table 1 shows 75.0 All for DKD versus 70.3 for joint? The table extraction is noisy, so exact reading is difficult in places, but even from the clean textual summary, gaps to joint training are not negligible in all settings. The broader point is that the evidence supports **strong performance**, but the “near-upper-bound” characterization should be stated more carefully and per setting rather than as a blanket claim.

### Minor
- **Hyperparameter robustness is only partially demonstrated.**  
  The paper does include analysis for `γ` and `τ`, so this is not an omitted issue. Still, the chosen `γ` changes with scenario (“for settings involving more incremental steps ... γ is set to 0.4”), which suggests the method is somewhat schedule-dependent rather than governed by a single robust recipe.
- **Compute overhead is only lightly characterized for a method that introduces spatial second-order structure.**  
  The paper reports epoch-time overhead (e.g., DKD vs MKD/CKD), which is useful, but it does not break down memory overhead or clarify the actual cost of the Laplacian/projection computation. Since the method is positioned as practical and inference-neutral, a clearer training-cost analysis would help.
- **The distinction between the proposed confidence/position maps and prior confidence-based distillation methods could be made sharper.**  
  The paper does explain that its goal is knowledge reuse rather than merely selecting reliable old pixels, but the technical differentiation is not as crisp as it could be, especially given the centrality of these maps.

### Trivial
- **Theoretical analysis is extensive but not always as illuminating as the paper suggests.**  
  The appendix does provide derivations, but much of it verifies local optimization behavior rather than establishing a strong theorem about why DKD should resolve parameter competition in the student. This is not a flaw by itself, but the main-text claim of “theoretical analysis” should not be read as a deep guarantee.

## Nice-to-Haves
- Report an explicit architecture-controlled comparison emphasizing only ViT-based baselines in the main table narrative, even though the appendix already helps address portability.
- Quantify what “parameter release” means operationally in the student: e.g., gradient utilization, activation sparsity, or layer-wise parameter-change statistics after pruning the teacher.
- Visualize the pruned teacher outputs and the learned `P_t` / `C_t` maps to show they correspond to semantically reusable regions rather than merely weaker/noisier supervision.
- Add explicit foreground-vs-background confusion metrics, since background shift is central in CISS and the paper mainly reports mIoU and similarity matrices.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Comparisons are unfair because many baselines use ResNet101 while the method uses ViT, so the results should be discounted.”**  
  Removed in its strong form. The paper does compare against multiple **ViT-based** methods in Table 1/2 (e.g., MIB-ViT, SSUL†-ViT, MicroSeg†-ViT, CoinSeg, MBS†, Nest, Adapter-T, CoMFormer, INC), and the appendix also applies DKD to CoinSeg with ResNet101 and ViT. It is still fair to ask for clearer architecture-controlled emphasis, but the paper is not simply comparing ViT against only weaker ResNet baselines.
- **“The method is unreproducible because code/models are not available.”**  
  Removed. The paper explicitly states that code is included in the supplementary material and details are given in the appendix.
- **“The paper omits basic implementation details like LR schedule and thus is irreproducible.”**  
  Weakened/removed as a major concern. The paper gives optimizer, epochs, learning rates per dataset, hardware, and says more details are in supplementary material. The scheduler specifics could be clearer, but this is not a substantive weakness at ICLR level absent evidence that results hinge on hidden tricks.
- **“The parameter release mechanism has zero effect because zeroed weights immediately regrow under gradient descent.”**  
  Removed as stated, because it misreads which model is pruned. The pruning is applied to the **old model / teacher target**, not to the trainable current model. The real issue is not “zero effect,” but rather that the paper overinterprets teacher pruning as releasing student capacity.
- **“The paper should not claim strong results because it lacks formal significance testing such as paired t-tests.”**  
  Removed as a core criticism. In this empirical area, repeated runs with standard deviations are already a reasonable robustness check; formal significance testing would be a nice-to-have, not a standard requirement.

## Novel Insights
The most important synthesis is that the paper’s empirical strength and its conceptual strength are not perfectly aligned. DKD seems to work well largely because it **relaxes how old knowledge constrains the current learner and turns some of that old knowledge into a selective guidance signal**, which is a useful idea. However, the paper frames this as literal “parameter release” in the student, while the implemented mechanism more clearly acts by **weakening and reshaping the teacher signal**. That distinction matters: it does not invalidate the method, but it changes how one should understand the contribution and what evidence is still needed.

## Suggestions
- **Clarify the mechanism claim.** Rephrase “parameter release” to distinguish teacher-target relaxation from actual student-capacity release unless you can directly measure the latter.
- **Make Eq. (4)–(6) fully explicit in the main paper.** Specify the discrete Laplacian/second-order implementation, tensor shapes, and the representation space of `y_c^*` used in the confidence-map dot product.
- **Add direct evidence for student-side effects.** For example, report per-layer gradient norms, parameter drift, or activation-space occupancy showing that DKD truly reduces competition for new-class learning.
- **Tone down blanket “near-upper-bound” claims.** State this per benchmark/setting where supported.
- **Strengthen the positioning of `L_Max`.** Explain more clearly whether its value is the specific formulation, its interaction with `L_Min`/`L_Esti`, or its role in CISS specifically, rather than implying a broadly new entropy principle.
- **Expand practical-cost reporting.** Include VRAM and throughput overhead for the Laplacian/projection component, not only extra seconds per epoch.