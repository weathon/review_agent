## Summary

This paper demonstrates that the training objective of linear probes on frozen pre-trained backbones has a surprisingly large impact on post-hoc explanation quality. Through extensive experiments across multiple pre-training paradigms (supervised, MoCov2, BYOL, DINO, CLIP), architectures (ResNet-50, ViT-B/16), attribution methods (LRP, IntGrad, B-cos, GradCAM, etc.), and datasets (ImageNet, VOC, COCO), the authors show that training probes with Binary Cross-Entropy (BCE) rather than Cross-Entropy (CE) consistently improves localization metrics such as GridPG and EPG. They also introduce B-cos MLP probes, which further improve both classification accuracy and explanation quality compared to linear probes.

## Strengths

- **Unusually broad and systematic empirical scope.** The paper evaluates 10+ attribution methods, 5 pre-training frameworks, 2 architectures, and 3 datasets with multiple metrics (GridPG, EPG, pixel deletion, compactness, complexity). This breadth gives confidence that the BCE-over-CE pattern is not an artifact of a single setting (Figures 2, 5, Table 1).
- **Clear and practically relevant empirical finding.** The observation that a simple change in probe loss can dramatically alter explanation quality on frozen SSL features is immediately useful to practitioners. The paper translates this into actionable guidance (use BCE, use B-cos MLPs) and releases code.
- **Strong qualitative evidence paired with quantitative trends.** Figures 1, 6, and 7 honestly show failure modes (e.g., CE probes leaking attribution into neighboring grid cells) that mirror the quantitative improvements, lending credibility to the results.

## Weaknesses

### Fatal
None.

### Major

- **Theoretical argument in Section 3.2 omits the effect of regularization.** The paper presents Equations (2)–(3) to argue that CE loss creates an infinite equivalence class of linear probes indistinguishable to the optimizer, making attributions unpredictable. This reasoning holds exactly only in the absence of L2 regularization (weight decay), which is standard in linear-probe training. Weight decay breaks the equivalence by penalizing norms non-uniformly across the class-weight manifold and selecting a unique minimum-norm solution. The paper does not acknowledge this caveat in the main text, discuss whether regularization was used, or show that the pathological behavior persists under standard regularization. This undermines the precise mechanistic claim that shift-invariance is *the* culprit for poor CE attributions.  
  *Why it matters:* The shift-invariance story is presented as the central theoretical justification for using BCE. Without addressing regularization, the causal link between the mathematical argument and the experimental setting is incomplete.

- **Missing controls for hyperparameters and accuracy in the CE–BCE comparison.** The paper does not report whether hyperparameters (learning rate, weight decay, epochs) were tuned separately for CE and BCE, nor does it explicitly match downstream accuracy when comparing explanation quality. Because BCE and CE optimize different objectives, they may converge to solutions of differing quality; Table 1 shows accuracy differences of up to 1.2% (MoCov3 ViT: 76.3% CE vs. 75.1% BCE). Without accuracy-matched comparisons or hyperparameter sweeps, it is harder to isolate the loss function itself as the sole causal driver of the explanation improvements.  
  *Why it matters:* The core claim is that the loss function *causes* better explanations. Confounding by optimization quality weakens this causal interpretation.

### Minor

- **The comparative superlative about pre-training is not formally quantified.** The abstract and conclusion state that probe training matters “much more than the pre-training scheme itself.” While Figure 5 provides compelling visual evidence—the BCE–CE gap dwarfs the variation across pre-training methods for fixed loss—the paper does not report a formal variance decomposition (e.g., ANOVA) or controlled swap experiment to quantify the relative effect sizes. The claim is qualitatively supported but would be stronger with a precise quantification.

- **Inconsistent behavior of conventional MLPs deserves more in-text discussion.** Section 5.2 notes that conventional (non-B-cos) MLPs decrease GridPG on ImageNet while improving EPG on COCO, deferring the full analysis to Appendix E.2. Because this inconsistency undercuts the universality of the “more complex probes help” message, it merits more than a brief mention in the main text.

### Trivial

- None.

## Nice-to-Have

- **Regularization ablation.** Training CE linear probes with and without weight decay and measuring attribution quality would clarify whether the shift-invariance mechanism operates under realistic training conditions or whether an alternative account is needed.
- **End-to-end fine-tuning experiments.** The study is restricted to frozen backbones. Testing whether the BCE advantage persists when the backbone is unfrozen would strengthen practical relevance for downstream users who fine-tune.

## Removed Points
These points are flagged to be removed; treat them with caution.

- **“Quantitatively unsupported” headline claim.** The paper is not “quantitatively unsupported.” Figure 5 and Table 1 provide extensive quantitative evidence that the BCE–CE gap is large and consistent across backbones. What is missing is a *formal decomposition* of variance, not quantitative support altogether.
- **“Straw man” about post-hoc methods.** The paper’s framing that post-hoc attribution methods are often applied without considering probe training is accurate to common practice, not a straw man.
- **GridPG emphasis as misleading.** The paper reports both GridPG and EPG, as well as pixel deletion, compactness, and complexity. The GridPG improvements are larger, but the paper is transparent about this and does not hide the more modest EPG gains.
- **BCE for single-label classification “without discussion.”** The paper cites prior work (Wightman et al., 2021; Böhle et al., 2024) on BCE for image classification, and Equation (4) clearly defines the objective with one-hot targets.
- **Missing explanation of attributions for B-cos MLP on conventional backbones.** While the paper could be more explicit, Section 4.2 states that standard post-hoc methods are applied to the full model (backbone + probe). The attribution computation follows standard backpropagation through the probe and then through the backbone.

## Novel Insights

None beyond the paper’s own contributions.

## Suggestions

- Add a discussion of regularization (weight decay) in Section 3.2, clarifying whether it was used in training and how it interacts with the shift-invariance argument. If weight decay was used, explain why the BCE advantage still emerges.
- Report hyperparameters for CE and BCE probe training, and ideally include an accuracy-matched comparison (e.g., by adjusting training epochs or regularization) to strengthen causal claims.
- Provide a brief ANOVA or range table quantifying the relative variation attributable to pre-training vs. probe objective to substantiate the comparative claim.

## Score and Decision

**Score: 6.0**

**Calibration comparison:**
- **High anchor** — `EJfLvrzh2Q.md` (avg 7.00, Poster): Similar linear-probing setting with solid theory. The reviewed paper has broader experiments but a weaker theoretical foundation; it sits below this anchor.
- **Medium anchor** — `c01YB8pF0s.md` (avg 5.50, Poster): Suffered from overclaiming and limited baselines. The reviewed paper has much more systematic experiments and a clearer empirical finding; it sits above this anchor.
- **Low anchor** — `wsjNCPqziJ.md` (avg 4.50, Reject): Had probing experiments but limited scope and overgeneralized claims. The reviewed paper has far broader validation and more robust evidence; it is well above this anchor.
- **Rejection borderline** — `uOrfve3prk.md` (avg 5.25, Reject): Missing related work, incomplete experiments, and rushed presentation. The reviewed paper is substantially more complete and polished.

The reviewed paper offers a genuine, broadly validated empirical contribution with immediate practical utility. Its weaknesses—an incomplete theoretical argument and missing hyperparameter controls—are real but do not invalidate the core findings. They are addressable in revision and are not severe enough to place the paper in the low-scoring band. Relative to the calibration anchors, it falls between the medium and high bands: stronger empirically than the 5.5 anchor, but lacking the theoretical rigor of the 7.0 anchor. A score of 6.0 and Poster acceptance is appropriate.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>