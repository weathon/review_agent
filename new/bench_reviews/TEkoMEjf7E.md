Let me check the paper content again directly from what was provided. I need to verify the key claims, particularly:

1. Whether Table 4 shows "random reference worse than no reference" 
2. Whether the paper explicitly addresses the information asymmetry concern
3. Whether the ablation actually has a "no reference" baseline

Looking at Table 4 in the paper:
- Without Reference: PSNR 15.90, SSIM 0.836, LPIPS 0.188
- Random Reference: PSNR 14.74, SSIM 0.820, LPIPS 0.226

So yes, random reference IS worse than no reference on most metrics. This is an important point that the Spark reviewer flagged.

Looking at Table 3 ablation:
- Base Model: PSNR 14.70 (this uses standard ControlNet with self-reference)
- + Meta-ControlNet: 16.35
- + Dynamic Ref. Routing: 14.76  
- + Self-Ref. Augmentation: 16.57
- Full Model: 17.02

The ablation shows dynamic routing adds only +0.06 PSNR individually.

Now let me think about the evaluation design concern. The paper compares against methods that don't have 3D reference input. This is a legitimate concern but it's also the whole point of the paper - they're proposing a new paradigm (reference-augmented). The question is whether the comparison is unfair. The paper does provide Table 4 with "Without Reference" which shows 15.90 PSNR - this IS an internal baseline without reference. The baselines get 14.63-16.35 PSNR. So the "without reference" version of their method is roughly comparable to LGM and some others. The improvement to 17.02 with retrieved reference IS meaningful. The "GT Reference" at 20.37 IS an upper bound that's somewhat misleading in a table.

However, I need to consider: the paper's claim is that reference-augmented generation improves over existing methods. Since the whole point is adding a 3D reference, comparing to non-reference methods IS the right comparison for the practical question "can we do better if we have a 3D asset library?" The comparison answers whether having this extra input (which artists actually do have in practice) helps. What would be MORE informative is a reference-aware baseline comparison.

Let me now put together the final review based on my analysis.

Key weaknesses to keep:
1. **Major**: The SOTA comparison is against reference-free baselines - this is the core claim of the paper, so it's fair to compare, but it would be more convincing with a simple reference-aware baseline (e.g., vanilla ControlNet + same reference) to show that the specific architectural innovations matter.
2. **Major**: Modest gains with retrieved references vs. GT references - the practical scenario shows relatively small margins. The GT reference results inflate the narrative.
3. **Major**: Random reference performing WORSE than no reference (Table 4) raises questions about robustness claims.
4. **Major**: Claims about "unified framework" for text/3D conditions are only qualitatively supported.
5. **Minor**: Dynamic reference routing has marginal quantitative impact (+0.06 PSNR) in isolation.
6. **Minor**: Missing inference time comparisons.
7. **Minor**: Evaluation limited to GSO (200 objects, simple shapes).

Weaknesses to remove:
- The harsh critic's claim that the GT reference comparison is "unrealistic" and should be removed from the table - actually, including it as an upper bound IS informative, it just shouldn't be the headline.
- Claims about missing related works (cannot verify).
- Nitpicks about reproducibility of augmentation details (they reference appendix).
- Formatting concerns.

Let me now calibrate the score. Looking at similar papers:
- 3D-Adapter (similar area, multi-view diffusion for 3D): Scores 6,6,5,6,5 = Reject (average 5.6) with concerns about missing comparisons, unclear design
- GeoDream (3D generation, incremental): Scores 3,3,6,3 = Reject
- InfiniteMesh (3D generation, lacking novelty): Scores 3,5,5,5 = Reject
- DMV3D (better 3D generation method): Scores 10,8,6,8 = Accept spotlight
- DiffSplat (3D generation, well-evaluated): Scores 8,8,6,6 = Accept
- Retrieval-augmented text-to-3D: Scores 5,5,5 = Withdrawn/Reject

Phidias has genuine novelty in the reference-augmented 3D generation idea, solid qualitative results, and clear architectural contributions. But the evaluation has significant issues: the practical gains are modest when using realistic retrieved references, random references hurt performance, and the "unified framework" claim lacks quantitative support beyond image-to-3D. The paper is somewhat similar in quality to 3D-Adapter but with a more interesting and novel problem formulation. I'd place it around 5.5-6, which is borderline. The evaluation design issues are significant but not fatal - the idea IS novel and the qualitative results ARE compelling. The ablation DOES show each component helps. But the lack of a reference-aware baseline and the modest retrieved-reference gains push this toward borderline reject.

Actually, let me reconsider. The paper:
- Has a genuinely novel and well-motivated idea (reference-augmented 3D generation)
- Has three solid architectural contributions that are ablated
- Shows clear improvements in the user study
- Has a limitation that the practical gains (retrieved reference) are moderate
- Has no comparison with reference-aware baselines

The evaluation concern is real but not fatal. The core question is: does having a 3D reference help? The answer is yes, and the paper shows this clearly. The meta-question of "does the proposed conditioning scheme beat a simpler one?" is partially addressed by the ablation (base ControlNet vs meta-ControlNet) but could be stronger.

I'll rate this around 5.5.</think>Now I have enough calibration data. Let me synthesize the final review.

## Summary

Phidias proposes a reference-augmented multi-view diffusion model for 3D generation that uses a retrieved or user-provided 3D model to guide generation from a concept image. The method introduces three components: (1) meta-ControlNet, which dynamically modulates conditioning strength based on similarity between the concept image and 3D reference; (2) dynamic reference routing, which adjusts reference resolution across diffusion timesteps; and (3) self-reference augmentations with a progressive curriculum for training. The framework supports image-to-3D, text-to-3D, and 3D-to-3D generation in a unified pipeline.

## Strengths

- **Novel and well-motivated problem formulation**: The idea of using 3D references to disambiguate ill-posed 3D generation mirrors real artistic workflows and is a natural and practically important extension of RAG principles to 3D generation. The "misalignment dilemma" is clearly articulated as a core technical challenge.

- **Architectural contributions targeting a real problem**: The three components (meta-ControlNet, dynamic reference routing, self-reference augmentation) are designed to directly address the misalignment between concept image and 3D reference. Meta-ControlNet is a thoughtful extension of ControlNet that learns to adaptively downweight conflicting references, and the ablation in Table 3 confirms it contributes the largest individual gain (+1.65 PSNR). The progressive curriculum for self-reference augmentation is a sound training strategy that enables robust handling of references at varying similarity levels.

- **Strong qualitative demonstrations and user study**: Figures 4–7 show compelling visual diversity, robustness to misaligned references, and controllability via the λ knob. The user study (Table 2) shows ≥88% preference rates across all baselines, indicating clearly visible quality improvements.

- **Versatile application scope**: The framework supports text-to-3D, theme-aware 3D-to-3D, interactive generation with coarse guidance, and 3D completion, demonstrating broad practical utility beyond the primary image-to-3D task.

## Weaknesses

### Fatal
None.

### Major

- **Evaluation lacks a reference-aware baseline, making it hard to isolate the contribution of proposed mechanisms over simpler conditioning**: All baselines in Table 1 (OpenLRM, LGM, CRM, SV3D, InstantMesh) are reference-free methods. While the comparison establishes that adding a 3D reference helps, it does not demonstrate that the specific architectural innovations (meta-ControlNet, dynamic routing, self-reference augmentation) outperform a simpler reference-conditioning scheme (e.g., feeding CCMs into a vanilla ControlNet on the same Zero123++ backbone). The ablation Table 3 partially addresses this by showing that each component improves over the "Base Model," but the base model already uses self-reference augmentation and ControlNet—there is no comparison against a straightforward baseline that simply concatenates reference CCMs or uses a standard ControlNet without the meta-controller. This leaves the central claim that *the proposed reference handling mechanisms are superior* somewhat under-supported relative to the claim that *reference information helps*, which is a weaker and more obvious statement.

- **Modest gains with retrieved references inflate the narrative around GT-reference results**: Table 1 reports both "Ours (GT Ref.)" and "Ours (Retrieved Ref.)." The GT reference condition (PSNR 20.37) gives a large margin over baselines, but it is an unrealistic oracle setting—the target 3D model is already known. The realistic "Retrieved Ref." condition shows much smaller margins: PSNR 17.02 vs. CRM 16.35 (+0.67), F-Score 0.833 vs. LGM 0.831 (+0.002), CD 0.0402 vs. LGM 0.0398 (actually worse). The paper acknowledges this ("The results of *Ours (Retrieved Ref.)* seems marginal") and attributes it to metric mismatch with retrieved references, but does not provide controlled experiments to support this explanation. The GT-Ref upper bound dominates the narrative but does not reflect practical performance.

- **Random reference hurts performance below no-reference baseline, undermining robustness claims**: Table 4 shows that using a random 3D reference yields PSNR 14.74 vs. 15.90 without any reference—worse on every metric. This means the method can *degrade* generation quality when retrieval fails to return a relevant reference, which is exactly the common deployment scenario. The paper does not discuss this failure mode or propose a mechanism to detect and fall back from bad references. Fig. 7(b) shows some robustness to explicitly conflicting references, but the quantitative evidence in Table 4 tells a more concerning story.

- **The "unified framework" claim for text-to-3D, 3D-to-3D, interactive generation, and completion is supported only qualitatively**: Sections 5 and Figures 8–12 demonstrate these applications with visual examples, but no quantitative evaluation, baselines, or user studies are provided for any task other than image-to-3D. The text-to-3D pipeline in particular depends on an unspecified text-to-image model and retrieval step whose quality substantially affects outputs, but this dependency is uncharacterized. The paper should either scope the claims to image-to-3D (where evidence exists) or provide quantitative support for the other modalities.

### Minor

- **Dynamic reference routing has negligible individual contribution**: In Table 3, adding dynamic reference routing to the base model improves PSNR by only +0.06 (14.70→14.76), with essentially no change in F-Score (0.826→0.826). While it may interact synergistically with other components (the full model achieves 17.02), the isolated contribution is marginal. The qualitative example in Fig. 6(b) (rope detail) is suggestive but not statistically supported.

- **Evaluation limited to GSO with 200 objects**: The GSO dataset consists primarily of simple, rigid, object-centric scans. The generalization claim would be better supported by evaluation on more diverse or challenging datasets (e.g., Objaverse test split with complex topologies or articulated objects).

- **Retrieval procedure under-specified**: The paper states retrieval uses "similarity (Zhou et al., 2024)" on a 40K Objaverse subset, but does not detail what features drive retrieval, how CCMs are generated for candidate references, or whether retrieval is trained jointly. As retrieval quality materially affects outcomes (Table 4), this is a notable gap.

### Trivial
- The Appendix references (A.5, B) for augmentation details and failure cases are mentioned but not included in the main paper, making the training curriculum details inaccessible during review.

## Nice-to-Haves
- Inference time comparison with baselines to establish practical efficiency trade-offs.
- A fallback mechanism that detects unhelpful references (based on Table 4's finding that random references hurt) and automatically switches to reference-free generation.
- Quantitative evaluation (or at least a user study) for text-to-3D and 3D-to-3D tasks to substantiate the "unified framework" claim.
- Analysis of how performance scales with database size beyond the 40K Objaverse subset.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Claim that GT-reference condition should be removed from comparison**: Including GT as an upper bound is informative and common practice; the issue is not its presence but its prominence in the narrative. The reviewer's demand to remove it is unreasonable—it provides useful context.

- **Demand for comparison with specific concurrent works (e.g., RichDreamer, LucidDreamer, One-2-3-45++)**: Per the rules, I cannot verify whether these works exist or are relevant, and listing specific missing references is not appropriate.

- **Formatting and notation nitpicks**: Removed as trivial style complaints.

- **Demand for confidence intervals / standard deviations across runs**: Large-scale 3D generation benchmarks commonly report single-run evaluations; this is not standard in the field.

- **Criticism that baselines should be given the same reference information**: The paper's entire contribution is the reference-augmented paradigm. Giving baselines references without a conditioning mechanism would not be a fair or meaningful comparison. The relevant comparison is against a reference-aware baseline using a simpler conditioning scheme, which the paper partially addresses.

- **Claim that the paper "does not substantiate" its claim to be a reference-based model**: This misunderstands the paper's core contribution, which is precisely to show that 3D references can improve generation—and the ablation and user study support this.

## Novel Insights

The misalignment dilemma is a genuine and underexplored problem in reference-conditioned generation: simply plugging a 3D reference into a ControlNet-style conditioner produces artifacts because the reference diverges from the target, and a model trained to closely follow conditions cannot selectively ignore conflicting information. The meta-ControlNet design, with its learned similarity-based gating, directly addresses this—making it conceptually distinct from vanilla ControlNet. The finding that random references *hurt* performance (Table 4, PSNR 14.74 < 15.90 for no reference) reveals a critical limitation of reference-augmented generation that the community should be aware of: retrieval quality is a bottleneck, and the model lacks an effective fallback mechanism for irrelevant references.

## Suggestions

- Add a simple reference-aware baseline (e.g., standard ControlNet + same CCM inputs on Zero123++ trained with self-reference) to isolate the contribution of meta-ControlNet and dynamic routing beyond simple conditioning.
- Implement a reference quality detection or fallback mechanism: if the retrieved reference is likely irrelevant, automatically reduce λ to zero or skip reference conditioning. This would directly address the random-reference degradation finding.
- Down-weight the GT-reference results in the narrative and focus discussion on the realistic retrieved-reference condition, potentially with an analysis of how the margin over baselines changes across different reference similarity levels.

## Evaluation

- **Originality**: The reference-augmented 3D generation paradigm is novel and well-motivated. The meta-ControlNet design is a meaningful architectural contribution. The paper is the first to systematically address reference misalignment in 3D diffusion models.
- **Importance**: The problem is practically important—artists routinely use references. The finding that reference quality critically impacts generation (including negative impact from irrelevant references) is valuable for the community.
- **Claims support**: The core claim that reference-augmented generation improves over reference-free methods is supported, but the claim that the proposed *mechanisms* are necessary vs. simpler alternatives is under-supported. The "unified framework" claim is only qualitatively supported.
- **Experimental soundness**: Image-to-3D evaluation is solid with quantitative metrics, ablations, and a user study. However, evaluation is limited to GSO, lacks reference-aware baselines, and shows concerning random-reference degradation. The remaining tasks lack quantitative evaluation.
- **Clarity**: The paper is well-written with clear motivation, well-organized sections, and effective figures.
- **Community value**: The RAG-for-3D paradigm, the self-reference augmentation curriculum, and the meta-ControlNet design are all likely to be adopted and extended by other researchers.

## Calibration

Comparing against similar papers: 3D-Adapter received scores of 6,6,5,6,5 (avg 5.6, reject) with concerns about missing baselines and unclear design contributions; DMV3D received 10,8,6,8 (avg 8, accept spotlight) with strong results and clean methodology; DiffSplat received 8,8,6,6 (avg 7, accept); GeoDream received 3,3,6,3 (reject) for incremental design and narrow evaluation; and the retrieval-augmented text-to-3D paper received 5,5,5 (withdrawn/reject). Phidias is above papers like GeoDream and InfiniteMesh (incremental, weak evaluation) but below papers like DMV3D and DiffSplat (strong results, clean methodology). Its novel problem formulation and architectural contributions put it roughly on par with 3D-Adapter, but the evaluation concerns (reference-free baselines, modest retrieved-ref gains, random-reference degradation) and unsupported "unified framework" claim are notable weaknesses that prevent a clear accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>