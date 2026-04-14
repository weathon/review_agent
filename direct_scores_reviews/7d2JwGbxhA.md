## Summary

OCEBO proposes the first pretraining scheme for slot attention–based object-centric models that operates entirely from scratch on real-world data, without relying on frozen, non-object-centric encoders (e.g., DINOv2). The key ideas are: (1) updating the target encoder via exponential moving average (EMA) of the student object-centric model, so the target encoder itself acquires object-centric inductive biases over time, and (2) a cross-view patch filtering mechanism that gates supervision to patches whose cross-view mutual nearest-neighbor condition is satisfied, preventing slot collapse when reconstruction targets are initially uninformative. When pretrained on ~241k images from COCO+, OCEBO achieves competitive unsupervised object discovery performance compared to methods using DINOv2 encoders pretrained on 142M images.

---

## Strengths

- **Principled solution to a well-identified bottleneck.** The paper directly addresses the saturation problem demonstrated by Didolkar et al. (2024) — performance plateauing at ~16k COCO images with a frozen target encoder — with a technically grounded response: make the target encoder a live EMA that absorbs object-centric inductive biases. This is not an incremental tweak but a rethinking of the training paradigm.

- **Cross-view patch filtering is a concrete, verifiable contribution.** The mutual nearest-neighbor consistency criterion as a proxy for target feature quality is intuitive, and Table 1(a) provides stark validation: removing it causes immediate slot collapse (FG-ARI on MOVi-E drops from 54.8 to 27.7). Figure 2 further shows the mechanism is active in a meaningful way throughout training (~10% at epoch 0, plateauing at ~70% by epoch 200).

- **Object-centric inductive biases demonstrated qualitatively.** Figure 3's PCA visualizations provide compelling evidence that the OCEBO target encoder learns instance-level separation (separating individual humans in a crowd, encoding part-whole hierarchies) rather than semantic grouping, validating the paper's core hypothesis about what EMA bootstrapping uniquely provides relative to DINOv2.

- **Quantitative collapse metric.** The paper introduces a cross-view slot consistency measure $d$ that makes slot collapse assessment reproducible and quantitative, complementing qualitative inspection. This is a practical contribution that the community can adopt.

- **Honest reporting of a non-trivial limitation.** The paper explicitly reports that ImageNet (10× larger than COCO) causes a dramatic performance drop and correctly identifies scene composition as the culprit, rather than overclaiming scalability. This self-aware framing strengthens credibility.

---

## Weaknesses

- **Scalability evidence is thin.** The central claim is "removing the upper bound" to enable large-scale pretraining, but only two data points are compared: COCO (118k) and COCO+ (241k), a roughly 2× increase. The paper itself cites the problem as one of orders-of-magnitude, yet provides no evidence of scaling across a meaningful range. The ImageNet experiment actually *worsens* performance, foreclosing the most obvious larger-scale test. The Appendix C scaling plot (mentioned but not visible) would need to show at least 4–5 data points and a consistent trend to support the scalability narrative. As presented, "scalable pretraining" remains an aspiration, not a demonstrated result.

- **mBO gap is real and insufficiently explained.** OCEBO's mBO is substantially weaker than baselines across most datasets (MOVi-C: 27.3 vs. 34.5–44.2; Pascal VOC: 34.4 vs. 37.2–42.0). The paper attributes this to decoder choice (MLP vs. top-k or autoregressive), which is plausible and consistent with the FG-ARI vs. mBO tradeoff pattern visible in Table 2. However, this is not demonstrated: there is no experiment swapping a SPOT-style decoder onto OCEBO's encoder. Without this, the possibility that the OCEBO encoder itself produces coarser features (acknowledged in Section 3.4 as blurry boundaries) contributing to the mBO gap cannot be ruled out.

- **Notation errors in Section 3.2.** Equation (2) assigns the same symbol $\mathbf{p}_{t,1,2}$ to two distinct quantities—the student projection of the reconstructed $q$ and the teacher projection of $z_t$. The reader must infer from context which quantity is intended, and the distinction is central to the loss formulation. Additionally, Equation (5) for the global loss writes $\sum_{i=1}^N$ over what are described as scalar global [CLS] representations $\tilde{z}$; the summation index is inconsistent with the global representation interpretation. While the paper may be correct algorithmically, these notation issues impede verification of correctness.

- **30% perpetually unfiltered patches left unanalyzed.** Figure 2 shows that ~30% of patches never satisfy the cross-view patch filtering condition by epoch 200. The paper observes this but offers no analysis of which patches are excluded (e.g., background, occluded regions, small objects) or whether permanent exclusion is by design or indicative of a limitation. For a method whose correctness depends on the filtering mechanism, this gap is notable.

- **No multi-seed robustness reporting.** Given that the model initializes both encoders randomly and that the method is sensitive to early training dynamics (as suggested by the patch filtering curve starting at 10%), a single random seed provides limited confidence in result stability. This is a standard expectation for novel training procedures.

- **Mask sharpening stage adds complexity and partially reintroduces a frozen encoder.** Section 3.4 introduces a 100-epoch sharpening stage with a frozen target and ℓ₂ reconstruction loss. Table 1(c) shows a substantial performance drop without it (FG-ARI 44.0 → 54.8 on MOVi-E), making it effectively required. During this stage, the method reverts to the frozen-target regime the paper aims to move beyond, introducing a conceptual tension the paper does not fully address. The choice of 100 epochs is not ablated.

- **Compute costs not reported.** Training 400 epochs (300 + 100 sharpening) on a ViT-S with EMA on COCO+ is non-trivial. Reporting GPU-hour costs is essential for practitioners attempting to reproduce or scale this work. This is especially relevant because the paper claims practical large-scale pretraining potential.

---

## Nice-to-Haves

- **Decoder-controlled comparison.** Applying a SPOT-style autoregressive decoder or FT-DINOSAUR's top-k MLP decoder on top of the OCEBO encoder would either confirm or refute the decoder-explains-mBO-gap hypothesis and significantly strengthen the paper's empirical narrative.

- **Target encoder feature evolution visualization.** Displaying PCA maps of the target encoder at epochs 0, 100, and 300 would empirically validate the claim that the encoder progressively acquires object-centric inductive biases over training, rather than leaving this as an inferred property.

- **Fine-grained scaling curve.** Even within COCO's size range (e.g., 10k, 30k, 60k, 118k, 241k images), a smooth scaling curve would be far more persuasive than two data points and would directly address the question of whether the "no upper bound" property holds continuously or only above a threshold.

- **Downstream transfer evaluation.** While the paper's stated scope is pretraining feasibility (not downstream application), a simple linear probe for patch-level tasks or a qualitative detection visualization would make the "pretraining" framing more concrete and help differentiate the encoder quality from segmentation-as-an-end-task.

- **Dataset characteristics analysis for OC suitability.** Section 4.2 correctly identifies that ImageNet's single-centered-object structure is detrimental, but lacks quantitative characterization (e.g., average object count, spatial distribution). This would provide actionable guidance for future dataset construction — a practical contribution beyond this paper.

---

## Removed Points

*These points are flagged for removal. Treat with caution — they are preserved in case context is useful.*

- **"Scratch baseline not provided" (Spark Finder):** This is addressed by ablation (b), where setting λ_oc = 0 reduces OCEBO to DINO pretraining on COCO followed by FT-DINOSAUR fine-tuning. The paper explicitly shows this leads to collapse, constituting the requested comparison.

- **"EMA bootstrapping fails because target encoder lacks OC biases — not subjected to controlled test" (Harsh Critic's ablation complaint):** The ablation in Table 1(b) sets λ_oc = 0, which removes the OC loss entirely, and shows collapse. While this is coarser than a perfect test (EMA without filtering alone), the existing ablation is reasonably informative for the paper's purpose.

- **"Comparison to methods is unfair because OCEBO uses far less data" (implied by comparing in Table 2):** The paper repeatedly and explicitly disclaims direct comparability ("models are not directly comparable"), so this framing issue is already handled.

- **"The global loss $\mathcal{L}_{global}$ sum over N is wrong for scalar global representations":** This is a genuine notation ambiguity but is likely a parser/formatting artifact given that the global loss in DINO operates over [CLS] tokens. The paper's logic is likely correct; this is a presentation/parsing issue already captured in the notation weakness above.

- **"Missing related works":** Not included per review instructions.

---

## Novel Insights

The most genuinely novel conceptual contribution beyond the technical method is the *inversion of the target encoder's role*: rather than importing object-centric structure from a pretrained semantic backbone (the DINOSAUR approach), OCEBO asks the question of whether object-centric structure can be *generated* by bootstrapping. The fact that the EMA-updated encoder in OCEBO learns to separate instances where DINOv2 groups them semantically (as shown in Figure 3) suggests that instance-level scene decomposition is a learnable self-organizing property — one that emerges from slot attention's inductive bias being reflected back into the representation space over time. This observation, if reproducible at scale, has implications beyond object-centric learning for how to construct inductive-bias-aware self-supervised objectives.

---

## Suggestions

1. **Add a decoder-controlled experiment** (OCEBO encoder + SPOT/top-k decoder) to test whether the mBO gap stems from encoder quality or decoder architecture. This is a single additional experiment that would resolve a central ambiguity.
2. **Report GPU hours** for OCEBO relative to DINOSAUR/FT-DINOSAUR to contextualize the practical tradeoff between removing the frozen-encoder dependency and training cost.
3. **Fix notation in Section 3.2**: Assign distinct symbols to the two quantities currently sharing $\mathbf{p}_{t,1,2}$ in Eq. (2), and reconcile the summation index in Eq. (5) with the scalar global representation.
4. **Analyze the 30% perpetually filtered patches** — even a simple frequency-of-occurrence analysis by image region (center vs. border, textured vs. uniform) would substantially clarify the behavior of the filtering mechanism.
5. **Run at minimum 3 seeds** for the ablation configurations in Table 1 and report variance, particularly for the collapse/non-collapse boundary which is the method's central reliability claim.
6. **Expand the scaling evaluation** with intermediate data sizes within COCO to produce a proper scaling curve, which is essential to support the scalability narrative beyond two points.

---

**Novelty:** High — the paper introduces the first demonstrated OC pretraining from scratch on real-world data and the cross-view filtering mechanism is a genuinely novel contribution.

**Technical soundness:** Moderate-to-good — the core approach is well-motivated and ablated. Notation issues in Section 3.2 are real, and the sharpening stage introduces conceptual tension.

**Empirical support:** Moderate — collapse avoidance and component necessity are well-evidenced; however scalability evidence is very limited (2×) and the mBO gap analysis is incomplete.

**Significance:** Good for the object-centric learning community — this unlocks a previously impossible training regime. The broader impact depends heavily on whether future work can demonstrate scaling to genuinely large datasets.

**Clarity:** Good overall, with localized notation problems in the loss formulation equations.

MY FINAL SCORE: <pineapple>6.2</pineapple>