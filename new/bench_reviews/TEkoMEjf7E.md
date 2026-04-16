Now let me check the referenced human reviews to calibrate the score.Now I have enough context to write the final consolidated review. Let me synthesize.## Summary

Phidias introduces a reference-augmented diffusion model for 3D generation that leverages retrieved or user-provided 3D reference models alongside a concept image to guide multi-view diffusion. Three technical components are proposed: (1) meta-ControlNet for adaptive conditioning modulation, (2) dynamic reference routing for timestep-dependent resolution adjustment, and (3) self-reference augmentation with curriculum learning for self-supervised training. The paper frames this as a unified framework supporting text, image, and 3D inputs with a two-stage pipeline (multi-view generation followed by sparse-view reconstruction).

---

## Strengths

- **Novel and principled RAG paradigm for 3D generation.** The analogy to retrieval-augmented generation in NLP/vision is compelling and well-motivated. Using retrieved 3D references to resolve ambiguity in ill-posed single-image-to-3D is a creative and practically grounded approach that could influence future work.

- **Coherent three-component architecture.** Each of the three components addresses a distinct sub-problem (meta-ControlNet → adaptive conditioning strength; dynamic routing → temporal misalignment; self-reference augmentation → training data scarcity). They are mutually reinforcing and clearly motivated.

- **Self-reference augmentation is the strongest technical contribution.** Creating self-supervised training signal via geometric augmentations of target models, combined with progressive curriculum learning, is a practical and clever solution to the problem of scarce aligned target-reference pairs. The curriculum design is non-trivial.

- **Ablation study supports meta-ControlNet and self-reference augmentation.** Table 3 shows that meta-ControlNet yields a substantial standalone gain (PSNR 14.70 → 16.35) and self-reference augmentation also provides meaningful improvement (PSNR 14.70 → 16.57). The full model (PSNR 17.02) substantially exceeds the base.

- **Quantitative improvements over baselines in the retrieved-reference setting.** Table 1 shows Ours (Retrieved Ref.) at PSNR 17.02 vs. CRM's 16.35, with consistent improvements across SSIM, LPIPS, CLIP-P, and CLIP-I. This is the most credible number for the real use case.

- **Versatile application space.** The paper credibly demonstrates the same trained model supports coarse-guided generation, 3D completion, theme-aware variation, and text-to-3D with qualitatively appealing results.

---

## Weaknesses

### Fatal
*None identified. The core claim (reference-augmented feed-forward 3D generation improves quality over no-reference baselines) is supported, even if imperfectly.*

### Major

- **Dynamic Reference Routing has near-zero standalone quantitative effect, yet is presented as a key component.** Table 3: Base PSNR 14.70 vs. "+Dynamic Ref. Routing" PSNR 14.76 — an improvement of 0.06 PSNR, well below what could be considered meaningful. The paper's narrative treats dynamic routing as one of three core contributors, but the ablation does not support this. The qualitative ablation in Fig. 6(b) shows plausible improvement (a missing rope detail), but the absence of quantitative support is a credibility problem for the component's necessity. The paper should either provide a better quantitative signal or lower the emphasis on this component.

- **Random reference degrades performance below "Without Reference," directly contradicting the robustness claim.** Table 4: Random Reference PSNR 14.74 vs. Without Reference PSNR 15.90 — a random reference *actively hurts* more than having no reference at all. Section 4.2 states "Phidias can still generate plausible results even with a random 3D reference," but the quantitative table refutes this as a general claim. The paper provides no analysis of the failure boundary: when does a reference hurt vs. help? How should users or the retrieval system avoid harmful references? This is a genuine and underanalyzed failure mode.

- **Broad claims about generalization and the unified framework are only qualitatively evidenced.** The abstract claims improvements in "generalization ability" and a "unified framework for 3D generation using text, image, and 3D conditions." In practice, generalization is argued through a handful of qualitative examples (e.g., "excavator's dipper" in Fig. 5). The text-to-3D, 3D-to-3D, 3D completion, and interactive generation applications in Sec. 5 have no baselines, no metrics, and no quantitative evaluation. This leaves a significant gap between the paper's claims and the evidence presented. These sections demonstrate the system is flexible enough to run in multiple modes, but not that it is a validated unified framework across those tasks.

### Minor

- **Evaluation restricted to 200 GSO objects — a narrow, clean, well-scanned household object dataset.** GSO is homogeneous and favorable for reconstruction-style metrics. The claimed generalization ability (to out-of-domain, atypical inputs) cannot be assessed from this single dataset. No evaluation on more diverse or challenging data is provided.

- **The user study protocol (Table 2) is underspecified.** Number of examples per comparison pair, randomization procedure, whether raters saw rotating 3D views or static renders, and whether raters were aware of the reference condition are not reported. Preference rates of 88–96% are unusually high and the lack of methodological detail makes them difficult to interpret confidently.

- **Metric-method tension in the retrieved reference setting.** The paper acknowledges this itself: "The results of Ours (Retrieved Ref.) seems marginal... caused by the differences between the retrieved references and GT." Since Phidias explicitly produces reference-guided variation in geometry, outputs faithful to the concept image *and* the reference may not match the GSO GT instance used for reconstruction metrics. The metric design is partially misaligned with the controllable-generation objective, and the paper does not propose an alternative evaluation strategy for this case.

### Trivial

- Sensitivity to the self-reference augmentation design (augmentation types, curriculum schedule specifics) is not analyzed in the main paper. It is unclear whether the specific curriculum matters or whether any reasonable augmentation would achieve similar results.

---

## Nice-to-Haves

- **Visualization of meta-ControlNet's adaptive signal magnitude.** Showing how the conditioning strength varies for aligned vs. misaligned references would reveal whether the meta-controller is meaningfully modulating control or has collapsed to near-constant behavior.

- **Retrieval database scaling analysis.** Reporting performance as a function of database size would clarify scalability and the practical value of the RAG approach. The current 40K Objaverse subset is arguably small.

- **Failure case analysis in the main paper.** Given that Table 4 shows random references degrade performance, representative failure cases are essential for readers to understand the method's operating envelope. The paper mentions failure cases in an appendix but this deserves main-paper treatment.

- **Computational cost analysis.** No inference time, GPU memory, or retrieval cost is reported. This is practically important for adoption.

- **Quantitative evaluation of text-to-3D** using established benchmarks would strengthen the unified framework claim.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **[Harsh Critic] "Unfair comparison because Phidias gets extra information."** The hard rule says to remove "unfair comparison" criticisms where the asymmetry favors the baseline — but here the asymmetry favors the *authors*. However, this concern is more of a framing issue than a structural flaw: the whole premise of the paper is reference-augmented generation, and comparing to baselines without a reference is the natural way to show the benefit of the new input modality. The paper correctly reports "Ours (Retrieved Ref.)" as the primary practical number. A weakened version of this concern (as a framing/overclaiming issue) is absorbed into the Major weakness above.

- **[Harsh Critic] "GT Ref is an invalid evaluation."** Including a GT oracle condition as an upper bound is scientifically reasonable and the authors are transparent about it ("actual performance should be between…"). The concern is partially valid regarding how boldly it is presented in Table 1, but it does not rise to the level of a structural problem. Absorbed as a minor framing note.

- **[Human Finder] "Novelty is incremental over existing ControlNet adaptations."** This is too generic. The combination of three components targeted at the specific misalignment dilemma in reference-augmented 3D generation is non-trivial, and the self-reference augmentation idea is genuinely novel. Hard rule: removed as a vague, one-size-fits-all criticism.

- **[Human Finder] "Two-stage error propagation."** While two-stage pipelines can compound errors, the paper uses the same stage-2 model (fine-tuned LGM) as multiple baselines, and the concern is not specific to Phidias's contribution. Removed as generic.

- **[Multiple reviewers] "Missing related work comparisons."** Per rules, removed — we cannot verify the existence of other works.

- **[Multiple reviewers] "No hyperparameter sensitivity analysis / training log."** Removed per reproducibility rules.

---

## Novel Insights

The most genuinely novel observation across the reviews — supported by the paper's own Table 4 — is that a poorly-matched retrieved reference actively degrades performance *below the no-reference baseline* (PSNR 14.74 vs. 15.90). This creates an implicit retrieval quality threshold: the method requires a sufficiently similar reference to improve over the vanilla model, and below that threshold it becomes harmful. This is practically important for deployment: RAG-style augmentation for 3D generation is not universally beneficial and requires a retrieval system of sufficient quality to stay above the "help vs. hurt" boundary. Understanding and characterizing this boundary is a meaningful open problem that the paper surfaces but does not analyze.

---

## Suggestions

1. **Resolve the Dynamic Reference Routing quantitative gap.** Either demonstrate its benefit with a more targeted experiment (e.g., ablation specifically on high-similarity reference pairs where detail conflicts are most acute) or recalibrate the narrative about its importance.

2. **Add a "random reference vs. no reference" failure analysis.** Characterize what makes a reference harmful — is it a similarity threshold, category mismatch, or orientation? A scatter plot of retrieval similarity score vs. output metric would clarify the operating envelope.

3. **Quantify at least one additional application.** Adding a single quantitative experiment for text-to-3D (e.g., CLIP score on T3Bench prompts) would substantially strengthen the unified framework claim.

4. **Broaden evaluation beyond GSO.** Even a small evaluation on a more diverse dataset (e.g., a challenging subset of Objaverse or in-the-wild internet images) would make the generalization claim more credible.

5. **Specify user study methodology.** Report number of examples per pair, presentation format, and blinding procedure to allow the user study to be interpreted reliably.

---

## Score and Decision

**Calibration anchors:**

- *9n9q0R9Gyw* (RAG Text-to-3D, scores 5,5,5, Withdrawn): Similar RAG premise, weaker evaluation and formulation than Phidias. Phidias is stronger.
- *bjkQTInGes* (Ouroboros3D, scores 5,5,5,5, Withdrawn): Good idea, incremental improvements, limited novelty in components. Phidias has more novel framing and components but similar scope limitations.
- *Z30Mdbv5jO* (ReconX, scores 6,5,5,5,8, Rejected): Good idea, good results, but limited/narrow evaluation. Phidias is qualitatively similar — a good novel idea with real but limited quantitative evidence.
- *FUgrjq2pbB* (MVDream, scores 8,6,6,6, Accepted): Much more comprehensive evaluation, very clear contribution, broader applicability. Phidias is below this level.

**Assessment:**
Phidias has a genuinely novel and well-motivated contribution (RAG for feed-forward 3D generation), solid ablations supporting the core components, and real (if modest) improvements in the realistic retrieved-reference setting. However, the evaluation scope is narrow (image-to-3D on GSO only), the Dynamic Reference Routing contribution is questionable quantitatively, the random-reference robustness claim is directly contradicted by Table 4, and the unified-framework claims rest entirely on qualitative demos. This places Phidias at the borderline — stronger than the 5-scored rejected papers but not reaching the depth of evaluation of accepted papers like MVDream.

**Originality:** Good — the RAG paradigm for feed-forward 3D generation is novel and the self-reference augmentation idea is clever.  
**Importance of research question:** High — quality and controllability in 3D generation are pressing open problems.  
**Claims vs. evidence:** Partially mismatched — the image-to-3D claims are supported, the generalization/unified-framework claims are not quantitatively supported.  
**Soundness of experiments:** Moderate — ablations are solid but the main evaluation has scope and metric alignment issues.  
**Clarity of writing:** Good — the paper is readable and the motivation is clear.  
**Value to research community:** Real — the RAG framing and self-reference augmentation are useful contributions even if the evidence is incomplete.

Positioned just below ReconX (average ~5.8, rejected) given that the scope limitations are comparable but the robustness contradiction in Table 4 is a genuine concern. Positioned above the 5-scored papers because the technical contribution is more coherent.

**Final score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>