=== CALIBRATION EXAMPLE 45 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the core idea: 3D style transfer via a “jigsaw” mechanism. It is reasonably descriptive, though “disentangled” is a strong claim that depends on how convincingly the paper demonstrates content/style separation.
- The abstract clearly states the problem, the proposed jigsaw-based strategy, the multi-view diffusion pipeline, and the baking step. It also mentions the main empirical claims: improved style fidelity, view consistency, lower latency, and support for partial references and scene stylization.
- A concern is that the abstract makes fairly broad claims about “decoupling style from content” and “fast, view-consistent stylization” without quantifying the speedup or clearly delimiting when this holds. The paper later shows only a small runtime difference versus one baseline in Table 1, so “substantially lower latency” may be overstated relative to what is actually reported.

### Introduction & Motivation
- The problem is well-motivated: 3D stylization is indeed hard because style, semantics, and multi-view consistency are entangled, and supervised paired 3D stylization data is scarce.
- The paper does identify a gap in prior work: existing methods either rely on heavy per-asset optimization or inject style in ways that can leak semantics. This is a legitimate and relevant motivation for ICLR.
- The contributions are stated clearly, but one important claim is under-justified: that the jigsaw transform provides a principled route to style-content disentanglement. The introduction makes this sound more general than what is actually shown experimentally.
- The intro somewhat under-sells the dependence on a pretrained SDXL-style backbone and on pseudo-paired supervision from rendered Objaverse assets. The actual novelty seems to be more in the training strategy and architecture than in a fundamentally new generative formulation.

### Method / Approach
- The overall pipeline is understandable: construct pseudo style-texture pairs via jigsawed references, train a multi-view diffusion model with geometry injection and reference attention, then bake the outputs to UV texture space. The high-level logic is coherent.
- However, the method is not fully reproducible from the main text. Several key components remain underspecified:
  - The exact implementation of the “reference U-Net” and how its intermediate activations are extracted at timestep \(t=0\).
  - The structure of the “multi-branch attention block,” especially how self-attention, multi-view attention, and reference attention are combined.
  - The training objective(s): the paper says it trains a diffusion model, but does not clearly spell out the loss formulation in the main method section.
  - The baking procedure is described only at a high level; the “seam-aware, confidence-weighted blending” and UV inpainting steps would benefit from more detail.
- A major conceptual question is whether the jigsaw operation truly “suppresses semantics while preserving style” in a way that generalizes. The paper cites Figure 3 and an appendix proof, but the claim is only partially supported:
  - The appendix proof argues that patch shuffling preserves global mean/variance under permutation, but that is not sufficient to preserve higher-order style cues or local texture statistics that matter for artistic style.
  - If style transfer relies on spatially localized motifs, patch shuffling may distort them substantially. The paper acknowledges some failures on text/symbols, but not broader failures on structure-dependent styles.
- There is an internal tension in the approach: the model is trained on jigsawed style references, but at inference the user provides a style image that is also jigsawed. This means the system is not truly “reference-image stylization” in the conventional sense; it is stylization from a transformed version of the reference. The practical effect may be acceptable, but the paper should more explicitly discuss what information is lost and whether this limits user control.
- The use of SDXL as a pretrained diffusion backbone is important. Since the paper relies on feature extraction from a pretrained T2I network, some of the style capacity may come from the pretrained prior rather than the proposed disentanglement mechanism itself. The paper should separate these effects more carefully.
- The “disentanglement” terminology is somewhat stronger than the evidence. The method likely reduces semantic leakage, but it does not establish a formal disentangling representation.

### Experiments & Results
- The experiments do test the central claims to some extent: style fidelity, disentanglement, and multi-view consistency are evaluated, and the paper includes qualitative comparisons, a quantitative table, and ablations on jigsaw settings.
- That said, the evaluation setup is not strong enough for ICLR’s typical bar yet:
  - The test set is only 20 Objaverse objects and 70 reference images total. This is small for a claim of “state-of-the-art” 3D stylization.
  - The paper evaluates mainly on two style datasets/groups: a small WikiArt subset and a self-collected set. It is unclear whether these sufficiently cover the diversity of styles claimed in the abstract and conclusion.
- Baseline comparison raises concerns about fairness and representativeness:
  - The baselines are limited to StyleTex, MV-Adapter, and 3D-style-LRM, which are relevant, but the paper does not compare against other strong recent 3D stylization or multi-view generation variants that may be close competitors.
  - StyleTex is an optimization-based method, so runtime comparison is meaningful, but the paper should be careful not to conflate different operating regimes. The “∼40s” for the authors’ method versus “5.35” for StyleTex in Table 1 is ambiguous; the units are not fully clear in the table as presented.
- The quantitative metrics are only partially convincing:
  - Gram Matrix Similarity and AdaIN Distance measure perceptual style statistics, but they do not directly establish visual quality or content preservation.
  - CLIP score is used as a disentanglement measure, but lower CLIP similarity is not necessarily evidence of better style-content disentanglement; it may also indicate weaker semantic alignment or simply less preservation of object identity. This metric needs much stronger justification.
  - There are no error bars, confidence intervals, or significance tests. With only 20 objects, variance could matter a lot.
- The ablation study on the jigsaw module is useful, but it is incomplete:
  - Figure 5 compares only three settings (w/o train & infer jigsaw, w/o infer jigsaw, and full method). This supports the importance of inference-time jigsaw, but does not isolate whether gains come from training-time pseudo-pair construction, from inference-time suppression of semantics, or from the combination.
  - The appendix mentions ablations on patch size and mask ratio, but the main paper only briefly summarizes them. A more explicit quantitative ablation table would materially strengthen the claims.
- The paper claims generalization to partial stylization, multi-object scenes, and tileable textures, but these appear only as qualitative appendix examples. They are interesting demonstrations, but they do not support strong claims of robust generalization without quantitative or at least more systematic analysis.
- The comparison to 3DGS-based stylization in the appendix is intriguing, but it is not integrated into the main evaluation. Since many ICLR readers will ask whether the method scales to scene-level representations and whether the mesh-baking pipeline is the bottleneck, this deserves more central treatment.
- In short, the results are promising, but the evidence does not yet fully justify a strong SOTA claim across 3D stylization more broadly.

### Writing & Clarity
- The overall narrative is understandable, and the paper has a clear intended structure.
- The main clarity issue is that several central methodological pieces are described at a high level while the exact operational details are deferred or omitted. In particular, the attention design and the diffusion training setup are not explained with enough precision in the main body.
- Figure 3 is conceptually useful for motivating the jigsaw transform, but the argument from classification score and Gram similarity to “semantic suppression with preserved style” remains somewhat heuristic.
- Tables and figures are generally useful in intent, but the paper’s presentation of metrics and runtime in Table 1 is not sufficiently explicit to support clean interpretation without digging into the appendix.
- The appendix contains useful additions, but some of the key claims in the main text depend on those appendix-only details. For an ICLR paper, the main paper should stand more independently.

### Limitations & Broader Impact
- The paper does acknowledge one meaningful limitation: failure on fine-grained text and symbolic patterns due to the SDXL backbone.
- However, the limitations discussion is still incomplete. Important limitations not fully addressed include:
  - Dependence on a pretrained generative backbone and its priors.
  - Reliance on mesh/UV quality and baking quality; the method may be sensitive to poor topology, heavy occlusion, or complex materials.
  - Potential degradation for styles that rely on global composition or semantic structure rather than local texture statistics.
  - The fact that the method is trained on pseudo-pairs from rendered Objaverse assets, which may limit transfer to real-world photos or highly non-photographic inputs.
- The ethical statement is appropriate but generic. One broader impact issue worth noting is that style transfer tools can be used for cultural appropriation or misleading asset generation; the paper mentions this, but not the more specific issue that synthesized textures may falsely imply authorship or provenance of a 3D asset.
- I would also expect a clearer discussion of whether the method may reproduce copyrighted art styles more faithfully than intended, since the paper explicitly demonstrates famous artistic styles in the appendix.

### Overall Assessment
JIGSAW3D presents an interesting and plausible idea: using patch shuffling and masking to turn ordinary textured 3D renders into pseudo style-texture supervision for multi-view diffusion-based stylization. The paper’s main strengths are the clean high-level pipeline, the practical goal of avoiding per-asset optimization, and the promising qualitative results. However, at the ICLR bar, the submission currently leaves too many important questions unresolved: the disentanglement claim is only weakly substantiated, the method details are not fully reproducible, the evaluation is small and lacks statistical rigor, and the evidence for broad SOTA performance is not yet strong enough. I think the paper is promising and potentially publishable, but in its current form it needs stronger methodological specification and more convincing experimental validation before it would clear ICLR’s typical acceptance threshold.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Jigsaw3D, a 3D style transfer framework that aims to disentangle style from semantics by applying patch shuffling and masking to a 2D reference image, then using the resulting “jigsawed” reference to guide a multi-view diffusion model for stylized rendering and texture baking. The main claim is that this yields faster, view-consistent 3D stylization with better style fidelity and less semantic leakage than prior attention-based or optimization-based baselines.

### Strengths
1. **Clear and relevant problem formulation for ICLR-level vision/ML research.**  
   The paper targets an important practical problem: controllable 3D stylization that preserves geometry and multi-view consistency, while reducing the optimization cost typical of score-distillation methods. This is a meaningful direction for ICLR, where efficient controllable generative modeling is a strong theme.

2. **A simple and intuitively motivated disentanglement idea.**  
   The core jigsaw transform—patch shuffling plus masking—has a plausible rationale: destroying global layout while preserving local texture statistics. The paper supports this with an ablation-style analysis (Figure 3, Appendix A.2) showing reduced classification score and preserved Gram-style statistics as shuffling intensity increases.

3. **End-to-end pipeline combines style generation and texture baking.**  
   The method is not just a stylized image generator; it includes multi-view generation, geometry conditioning via normal/position maps, and UV-space baking. This makes the contribution more complete than methods that only generate rendered views.

4. **Good practical coverage of use cases.**  
   The paper reports extensions beyond the main benchmark setting, including partial reference stylization, multi-object scene styling, and tileable texture generation. These demonstrations suggest the approach may generalize to several user-facing workflows.

5. **Reasonable empirical evidence of efficiency.**  
   The reported inference times are much lower than the SDS-based baseline StyleTex and comparable to feed-forward baselines (~40s), while achieving better style metrics on the reported benchmarks. For ICLR, this efficiency angle is valuable if the claims are substantiated.

### Weaknesses
1. **Novelty is moderate and partly incremental relative to existing attention-based stylization work.**  
   The paper’s main novelty is the use of jigsaw shuffling/masking to create style references and pseudo-pairs. However, patch shuffling as a style-preserving/content-destroying operation is already known in 2D stylization and representation learning, and the method builds on a fairly standard multi-view diffusion + adapter + cross-attention design. The contribution feels like a sensible system integration rather than a clearly new learning principle.

2. **Evidence for “style–content disentanglement” is not fully convincing.**  
   The paper argues that jigsawing suppresses semantics while preserving style, but the evidence is limited mainly to a classifier score and Gram similarity plot. A lower classifier score does not necessarily prove disentanglement, and Gram similarity is a narrow style metric. The method may still rely on the diffusion model to infer or hallucinate content from residual cues and training priors.

3. **Experimental evaluation is relatively small for ICLR standards.**  
   The benchmark uses only 20 test objects and 70 reference images total, which is modest for a paper claiming broad generalization. There is no indication of statistical variance, multiple seeds, or significance testing. For an ICLR submission, reviewers typically expect stronger empirical support, more diverse test settings, and clearer robustness analysis.

4. **Baseline coverage seems limited.**  
   The comparisons include MV-Adapter, 3D-style-LRM, and StyleTex, but the broader 3D stylization landscape is larger. In particular, the paper does not convincingly establish superiority over all relevant recent methods or demonstrate how much of the gain comes from the jigsaw idea versus the SDXL backbone and multiview training recipe.

5. **Method description leaves some ambiguity.**  
   Key implementation details are not fully transparent: how exactly the reference U-Net features are extracted and used, how the multi-view attention is structured, how baking confidence weights are computed, and what training data splits/filters are applied. The paper also glosses over how captioning and text-conditioning interact with stylization, which may matter for reproducibility.

6. **Possible metric mismatch with the claimed objective.**  
   The paper uses Gram similarity, AdaIN distance, and CLIP score, but these metrics do not fully capture 3D texture quality, geometric consistency, seam quality, or perceptual faithfulness across views. The paper’s strongest claims about multi-view consistency and baking quality are mostly supported qualitatively.

### Novelty & Significance
**Novelty:** Moderate. The key idea of using patch shuffling and masking to remove semantic structure is interesting and useful, but it is conceptually close to existing 2D style-transfer and feature-shuffling literature. The 3D integration is well-engineered, but the methodological leap beyond combining known components is limited.

**Significance:** Moderate to good if the claims hold. A faster feed-forward 3D stylization method with better style/content separation would be practically useful for graphics and content creation. However, against ICLR’s usual acceptance bar, the paper would need stronger evidence that the disentanglement mechanism is genuinely novel and broadly effective rather than a useful heuristic.

**Clarity:** Fair. The overall pipeline is understandable, and the figures help. But several technical sections are underspecified, and the parser artifacts aside, the exposition still leaves important questions about the exact model design and training regime.

**Reproducibility:** Fair to limited. The paper provides some hyperparameters and states that code is included, which helps. Still, reproducibility would benefit from more precise architectural details, data preprocessing steps, object/reference selection criteria, and more complete ablations.

**Significance relative to ICLR standards:** Borderline. ICLR usually favors methods with either a clearly new learning principle, strong theoretical insight, or exceptionally thorough empirical validation. This work has a useful idea and a solid system contribution, but the novelty and evaluation depth may be below the typical ICLR acceptance bar unless the supplemental material is substantially stronger than the main paper suggests.

### Suggestions for Improvement
1. **Strengthen the disentanglement evidence beyond Gram/CLIP heuristics.**  
   Add experiments showing how much semantic information remains in jigsawed references, e.g., retrieval/classification attacks, mutual-information-inspired probes, or user studies comparing style fidelity vs. content leakage.

2. **Broaden and deepen the evaluation.**  
   Include more test objects, more diverse categories, multiple random seeds, and quantitative measures of multi-view consistency and UV seam quality. Report confidence intervals or standard deviations.

3. **Expand baseline comparisons.**  
   Compare against more recent 3D stylization and multi-view generation methods, and include stronger ablations separating the effects of: jigsawing, reference attention, geometry conditioning, and the baking strategy.

4. **Provide a more precise algorithmic specification.**  
   Clarify the architecture of the style/reference U-Net, the exact attention formulations, the feature extraction point(s), and the UV reprojection/blending procedure. A concise pseudocode algorithm would help substantially.

5. **Demonstrate the method’s limits more systematically.**  
   Since the paper acknowledges failures on text and symbols, test cases with different semantic textures, repetitive patterns, and high-frequency details would help define where the method works and where it breaks.

6. **Justify the design choices experimentally.**  
   The choice of patch size, mask ratio, number of views, and the use of SDXL should be connected to quantitative trade-offs, not only qualitative examples. A small hyperparameter sensitivity study would improve trust.

7. **Clarify the contribution relative to prior patch-shuffling style work.**  
   The paper should explicitly separate what is borrowed from 2D style-transfer literature and what is new for 3D. This would help reviewers assess novelty more fairly and reduce the impression of incremental recombination.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to training-free 2D reference-style methods adapted to the 3D multi-view setting, especially StyleAligned, Visual Style Prompting, and StyleAdapter, because the paper’s main claim is style disentanglement via jigsaw/reference-attention, not just beating a few 3D baselines. Without these, it is not clear whether the proposed reference-attention or the jigsaw trick is better than simpler style-transfer mechanisms.

2. Evaluate against stronger and broader 3D stylization baselines, including recent scene/object stylization systems beyond the three reported methods, and include the 3DGS stylization comparison in the main quantitative table. ICLR reviewers will expect the claim “state of the art” to be tested against the most competitive current methods on the same benchmark protocol, not just a narrow baseline set.

3. Add ablations that isolate each core design choice: jigsaw shuffling vs masking alone, reference-attention vs plain cross-attention, multi-view attention vs single-view generation, and geometry conditioning vs no geometry. Right now the paper does not establish which component actually drives performance, so the contribution is not convincing as a method.

4. Report results on more than 20 test objects and at least one additional dataset or split protocol with diverse topology/materials. The current evaluation is too small for an ICLR-level claim about general 3D stylization, and it risks overfitting to a hand-picked set of showcase objects.

5. Add runtime/memory/scalability evaluation versus per-scene optimization methods under matched hardware and resolution settings. The paper claims fast stylization and better scalability, but the reported “time” numbers are too thin to judge whether the method is practically better in the regimes that matter.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the actual disentanglement claim, not just style metrics and a CLIP score. The paper needs evidence that jigsaw suppresses semantics while preserving style, e.g., mutual information/linear probe results, content-class leakage tests, or a content-preservation metric on generated outputs.

2. Analyze failure modes by reference type and object type. The method is claimed to generalize to diverse artistic styles and objects, but the paper does not show where it breaks: complex brushwork, repeated patterns, text-heavy styles, reflective materials, or highly articulated meshes.

3. Provide variance and significance across multiple random seeds and reference/object pairs. With only averaged metrics over a small test set, it is impossible to know whether the reported gains are stable or driven by a few favorable cases.

4. Examine whether the model is truly view-consistent or merely producing locally plausible textures. The paper needs cross-view consistency measurements beyond style similarity, such as view-to-view feature consistency or UV-space consistency, because 3D stylization quality depends critically on this.

5. Analyze the effect of the jigsaw transform on style statistics more rigorously. The current argument that shuffling preserves style is too informal for a core claim; the paper should show what statistics are preserved or lost and why the chosen patch/mask settings are optimal.

### Visualizations & Case Studies
1. Show UV texture maps, seam regions, and reprojected view overlays for several examples. This would reveal whether the method actually bakes coherent textures or only looks good in rendered viewpoints.

2. Visualize attention maps or correspondence from reference patches to generated regions. This is necessary to verify the claim that the method performs style-aware recombination rather than copying arbitrary patches or leaking semantics.

3. Include failure cases for each cited limitation: text, symbols, fine structures, repetitive patterns, and disoccluded regions. The paper currently mentions failures but does not expose how often they happen or whether they are severe enough to undermine the method’s utility.

4. Show side-by-side results for the same object under multiple wildly different styles, including highly abstract and highly structured references. This would test whether the method preserves object identity while adapting style, or whether it collapses to a narrow set of texture priors.

### Obvious Next Steps
1. Add a fully controlled user study on style fidelity, content preservation, and multi-view consistency. For an ICLR paper making perceptual claims, this is the most direct way to test whether the proposed method is actually preferred by humans.

2. Extend evaluation to multi-reference and style interpolation settings. Since the method is framed around disentangled style statistics, demonstrating controllable mixing would materially strengthen the contribution.

3. Test robustness to incomplete, noisy, or out-of-distribution reference images. The partial stylization appendix hints at this capability, but the paper needs systematic evidence if it wants to claim practical reference robustness.

4. Release a reproducible benchmark protocol for 3D style transfer. The field lacks standardized evaluation, and this paper would be stronger if it defined a clear object/style split, metrics, and baseline suite instead of relying on a small bespoke test set.

5. Compare jigsaw-based disentanglement to simpler alternatives such as random crops, patch dropout, or style-statistics extraction without shuffling. Without these controls, it is unclear whether the proposed jigsaw is essential or just one of several ways to remove semantics.

# Final Consolidated Review
## Summary
JIGSAW3D proposes a 3D stylization pipeline that turns a 2D reference image into a “style-only” jigsawed conditioning signal, uses it to train a multi-view diffusion model with geometry injection and reference attention, and then bakes the generated views into UV textures. The paper’s main promise is practical: faster, view-consistent 3D style transfer with less semantic leakage than prior optimization-heavy or attention-only baselines.

## Strengths
- The central idea is simple and plausible: patch shuffling plus masking is an intuitive way to suppress global semantics while keeping local texture statistics, and the paper provides some supporting analysis showing reduced classification score and preserved Gram-style similarity as shuffling increases.
- The overall pipeline is coherent and practically relevant: pseudo-paired training data from rendered 3D assets, multi-view generation with geometry conditioning, and UV baking into a final texture. This is a complete system rather than a narrowly scoped module.
- The method does demonstrate useful application breadth beyond the main benchmark setting, including partial reference stylization, multi-object scene styling, and tileable texture generation, which suggests the approach is not limited to a single demo setting.

## Weaknesses
- The core “style-content disentanglement” claim is not convincingly established. The paper mostly relies on heuristic evidence such as classification score, Gram similarity, and a CLIP-based disentanglement proxy; none of these actually proves that semantics have been removed while style has been preserved. This is a substantive gap because disentanglement is the paper’s headline contribution.
- The evaluation is too small and too narrow for a strong ICLR claim. Testing on 20 objects and a modest set of reference images, with no variance across seeds or significance analysis, leaves open the possibility that the reported gains are fragile or benchmark-specific. The “state of the art” claim is therefore not well supported.
- Baseline coverage is limited relative to the breadth of the claim. The paper compares against a small set of 3D stylization methods, but not against simpler training-free 2D style-transfer mechanisms adapted to the multi-view setting. That makes it hard to tell whether the proposed jigsaw/reference-attention mechanism is actually better than more direct alternatives.
- The method description leaves important implementation details underspecified, especially around the reference U-Net feature extraction, the exact composition of the multi-branch attention block, the training loss, and the UV baking procedure. This hurts reproducibility and makes it difficult to assess whether the gains come from the proposed idea or from hidden design choices.

## Nice-to-Haves
- A more systematic ablation would help: separate the effects of jigsaw shuffling, masking, reference attention, multi-view attention, geometry conditioning, and baking. Right now the paper shows that the full pipeline works, but not which component is doing the real work.
- More direct evidence of view consistency and texture quality would strengthen the paper, such as UV seam visualizations, cross-view consistency metrics, and attention/patch correspondence visualizations.
- A larger and more diverse evaluation set, ideally with multiple seeds and confidence intervals, would make the empirical claims much more trustworthy.

## Novel Insights
The most interesting insight is that the paper reframes reference-image stylization as a problem of constructing a semantics-suppressed style carrier rather than transferring style from a full natural image. That is a useful conceptual move, because it explains why patch-level shuffling can reduce semantic leakage and why a multi-view generator may benefit from style cues extracted from “destroyed” images instead of clean references. However, the paper stops short of proving that this yields true disentanglement; at present, the jigsaw transform looks like a clever and effective heuristic for reducing content bias, not a principled solution to style/content separation.

## Potentially Missed Related Work
- StyleAligned — relevant as a training-free attention-sharing style transfer method that could be adapted to the 3D multi-view setting.
- Visual Style Prompting — relevant as another training-free attention manipulation baseline for reference-style transfer.
- StyleAdapter — relevant because it uses feature shuffling/positional modifications for stylized generation and is close in spirit to the paper’s disentanglement claim.

## Suggestions
- Add controlled comparisons and ablations that isolate the value of jigsawing from the rest of the system, including direct 2D style-transfer baselines adapted to multi-view generation.
- Report more rigorous evaluation: more objects, more styles, multiple seeds, standard deviations or confidence intervals, and at least one metric that better captures cross-view or UV consistency.
- Clarify the full training and baking pipeline with pseudocode and exact module definitions so the method can be reproduced and its contributions assessed independently.

# Actual Human Scores
Individual reviewer scores: [8.0, 2.0, 4.0, 4.0]
Average score: 4.5
Binary outcome: Reject
