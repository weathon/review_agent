Now I have a thorough understanding of the paper and the calibration landscape. Let me synthesize my final review.

## Summary

Phidias proposes a reference-augmented 3D diffusion model that leverages retrieved or user-provided 3D references to improve quality, generalization, and controllability of 3D generation from text, image, or 3D inputs. The method introduces three components: meta-ControlNet for dynamically modulating conditioning strength based on concept-image/reference similarity, dynamic reference routing that adjusts reference resolution across denoising timesteps, and self-reference augmentation with curriculum training for self-supervised learning. The system produces visually compelling results and supports diverse applications including image-to-3D, text-to-3D, 3D-to-3D, interactive generation, and 3D completion.

## Strengths

- **Well-motivated problem framing**: The RAG analogy for 3D generation is timely and well-grounded in real artist workflows (Section 1). The misalignment dilemma is clearly articulated and each component targets a distinct aspect of it—global similarity mismatch (meta-ControlNet), local detail conflict (dynamic routing), and training data scarcity (self-reference augmentation).

- **Principled reference representation**: Converting 3D references into multi-view CCMs rather than meshes/voxels is a sound design that provides compatibility with diffusion model inputs while reducing texture-based conflicts (Section 3.1). This enables the entire pipeline without requiring specialized 3D encoders.

- **Strong quantitative improvements**: Table 1 shows Phidias (Retrieved Ref.) achieving 17.02 PSNR vs. the next-best baseline InstantMesh at 14.63, and GT Ref. achieving 20.37. The user study (Table 2) with 30 participants shows 88–96% preference rates over all baselines, confirming the quantitative gains reflect perceptual quality.

- **Controllability as a genuine capability**: The ability to generate different 3D outputs from the same concept image by varying the reference (Figure 4) and the adjustable λ parameter (Figure 12, Eq. 3) provide explicit user control not offered by prior methods. This is a practical and useful contribution.

- **Unified multi-modal framework**: The same architecture handles image-to-3D, text-to-3D, 3D-to-3D, interactive generation, and 3D completion (Section 5), demonstrating the generality of the reference-augmented formulation.

## Weaknesses

### Fatal
None.

### Major

- **Confounded ablation study does not isolate architectural contributions from training data changes**: The ablation in Table 3 does not cleanly attribute improvements to the proposed components. The base model is trained with self-reference only, while "+ Meta-ControlNet" is trained with both self-reference and retrieved references (Section 4.2: "To evaluate meta-ControlNet, we use both self-reference and retrieved reference for training, as the learning of Meta-Controller requires reference models with varying levels of similarity"). This means the PSNR jump from 14.70 to 16.35 could be partly or largely attributable to the addition of retrieved references during training rather than the Meta-ControlNet architecture itself. Similarly, the "Full Model" row combines all components with the expanded training data. Without an ablation that adds retrieved references to the base model *without* Meta-ControlNet, it is impossible to determine whether Meta-ControlNet's architectural innovation or the expanded training data drives the improvement. This undermines the paper's central claim that each of the three proposed designs contributes meaningfully.

- **Reconstruction backbone differs from LGM baseline, confounding comparisons**: Section 3.5 states "we finetune LGM by expanding the number of input views from 4 to 6 and the resolution of each view from 256×256 to 320×320." The LGM baseline in Table 1 uses the original 4-view, 256×256 configuration. Some portion of the improvement attributed to reference-augmented generation may come from this stronger reconstruction stage. No baseline is provided that runs the finetuned LGM (6 views, 320px) without reference input, making it impossible to isolate the gains from generation vs. reconstruction.

### Minor

- **Random reference degrades performance below no-reference baseline**: Table 4 shows "Random Reference" (PSNR 14.74) performs worse than "Without Reference" (PSNR 15.90). This means providing an uninformative reference actively hurts the model. The paper does not discuss this result or its implications. A reference-augmented system should ideally be harmless when the reference is uninformative. This suggests the model has not fully learned to suppress irrelevant references, and the paper should characterize when references become detrimental.

- **Dynamic Reference Routing shows negligible quantitative improvement**: The ablation (Table 3) shows this component adds virtually nothing on its own (PSNR: 14.70 → 14.76). While Figure 6b shows a qualitative benefit (rope preservation), the quantitative evidence for this component's contribution is extremely weak. The paper should discuss whether the timestep boundaries (t_h, t_m, t_l) are sensitive or well-justified.

- **Evaluation metrics partially misaligned with the stated goal**: The paper acknowledges that "Ours (Retrieved Ref.) seems marginal" and attributes this to "differences between the retrieved references and GT when computing the reconstruction metrics" (Section 4.1). When the method correctly produces output guided by a non-GT reference, reconstruction metrics against GT penalize this. The paper reports both GT Ref. and Retrieved Ref. configurations, and the user study provides complementary evidence, but no metric evaluates fidelity to the concept image while appropriately crediting adherence to reference geometry. This is partially addressed but leaves the quantitative evaluation less informative than ideal.

- **Missing naive reference-augmented baseline**: The paper compares against methods that receive only an image. A simple baseline that also receives the reference (e.g., concatenating reference CCMs as additional input views to an existing multi-view diffusion model, or using standard ControlNet without meta-learning) would establish whether the proposed architectural innovations are necessary beyond simply providing the reference as input.

### Trivial
None.

## Nice-to-Haves

- Report timestep boundaries (t_h, t_m, t_l) for dynamic reference routing and augmentation details for self-reference training to improve reproducibility.
- A quantitative metric for controllability (e.g., output diversity conditioned on different references for the same input image) would strengthen the controllability claim.
- Per-stage error analysis showing multi-view images from Stage 1 alongside final 3D from Stage 2 would reveal where errors originate.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"First reference-based 3D-aware diffusion model" claim is overclaimed**: The related work section itself distinguishes between optimization-based (Wu & Zheng 2022; Wang et al. 2024b) and feed-forward approaches. The claim in the contributions should be qualified as "first feed-forward reference-based 3D-aware diffusion model" but is not factually wrong—just imprecise. Downgraded to trivial and not listed above as it doesn't meaningfully affect the paper's contribution.

- **Missing c_pair details / computational cost of meta-controller**: The paper describes c_pair as "a pair of the concept image and the front-view reference CCM" processed through trainable encoders (Eq. 2–3, Fig. 3a). While the exact concatenation/encoding mechanism could be clearer, the architecture diagram and equations provide sufficient information to understand the design. Computational cost discussion would be nice but is not standard for this venue.

- **Applications demonstrated qualitatively only / cascading errors in text-to-3D / 3D completion failure modes**: This is standard practice in this research area. Demanding quantitative evaluation for application demonstrations and failure analysis of multi-stage pipelines is scope creep beyond what the community typically requires.

- **Missing related works**: Per the rules, I cannot confirm the existence of suggested missing citations and should not flag these.

- **Formatting and presentation nitpicks**: Removed per rules.

## Novel Insights

The most insightful observation across the reviews is the tension between the paper's two evaluation configurations. "Ours (GT Ref.)" demonstrates the method's ceiling when the reference perfectly matches the target, while "Ours (Retrieved Ref.)" shows the practical scenario. The gap between them (PSNR 20.37 vs. 17.02) is substantial and reflects the fundamental difficulty of reference-augmented generation: the method's success is bounded by reference quality, yet evaluation against GT inherently penalizes the method for correctly following non-GT references. This suggests the field needs new evaluation protocols for reference-conditioned generation that disentangle "following the reference" from "matching the ground truth."

## Suggestions

- Add the critical missing ablation: train a standard ControlNet (no meta-controller) with both self-reference and retrieved references using the same training data as the full model. This would isolate whether Meta-ControlNet's architecture or the expanded training data drives the improvement.
- Run the finetuned LGM (6 views, 320px) as a standalone baseline without reference input to isolate gains from the reconstruction stage vs. the reference-augmented generation.
- Discuss the random-reference degradation result and propose or analyze mechanisms to detect and suppress harmful references.

## Score and Decision

**Calibration comparison:**

- **High anchors**: DMV3D (8.0, Spotlight) — single-stage 3D generation with novel denoising via 3D reconstruction; clean experimental validation. DiffSplat (7.0, Poster) — repurposing image diffusion for Gaussian splats; solid but incremental. Phidias has a more novel problem framing than DiffSplat but weaker experimental validation than both.

- **Medium anchors**: 3D-Adapter (5.6, Reject) — similar concerns about ControlNet design choices and confounded ablation, but also showed practical improvements. Phidias has stronger overall results (user study, multiple applications) and a more novel paradigm (reference-augmented 3D), placing it above 3D-Adapter.

- **Low anchors**: SITTO (2.33, Reject) — fundamentally limited experiments and evaluation. Phidias is clearly far above this level with comprehensive experiments, user study, and multiple applications.

Phidias introduces a genuinely novel and well-motivated paradigm for 3D generation, with impressive practical results and a unified framework. However, the confounded ablation (training data changes alongside architecture) and the differing reconstruction backbone are significant scientific rigor concerns that prevent confident attribution of improvements to the proposed components. The paper demonstrates that the overall system works well but doesn't convincingly establish that its specific design choices—rather than simply the availability of a reference—drive the improvements. This places it in the borderline range, above papers with similar concerns (3D-Adapter at 5.6) due to stronger results and novelty, but below clearly strong papers (DiffSplat at 7.0) due to the experimental rigor gap.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>