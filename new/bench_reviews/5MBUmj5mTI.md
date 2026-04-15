Now let me read some of the comparison human review files for calibration:Now I have enough to write the final consolidated review. Let me synthesize everything.

---

## Summary

This paper presents the first systematic study of how shape, texture, and color cues influence *learning success* (as opposed to biases of pre-trained models) in semantic segmentation DNNs. The authors propose a pipeline for constructing cue-specific datasets—notably a Voronoi-based surrogate task for texture—and train "cue expert" networks on Cityscapes, PASCAL Context, and CARLA. Late fusion of experts is used to study pixel-level cue influence, and both CNN and transformer backbones are compared.

---

## Claims and Support

**Claim 1: "Generic procedure to derive cue-specific and cue-combination datasets for semantic segmentation"**
Partially supported. The pipeline is presented in Sec. 3 and covers color (architectural restriction), texture (Voronoi surrogate), and shape (HED/EED/CARLA rendering). However, the methods are heterogeneous: color is enforced architecturally while texture and shape rely on dataset transformations of fundamentally different kinds. The term "generic" overstates the uniformity.

**Claim 2: "First cue influence study in semantic segmentation, shifting from bias analysis to learning success"**
Largely supported. The conceptual framing is novel and the related work correctly positions the study against existing bias analyses. The distinction is real and meaningful.

**Claim 3: "Neither texture nor shape clearly dominates; shape+color is surprisingly strong"**
Partially supported. The "shape+color is surprisingly strong" finding is well-supported numerically. However, "neither clearly dominates" is not supported: on Cityscapes, S_SEED-RGB (42.22%) is more than *double* T_RGB (20.10%); on PASCAL Context, S_SEED-RGB (31.32%) vs T_RGB (17.75%) shows the same pattern. Only on the synthetic CARLA dataset does texture approach parity with shape. On real-world datasets, shape+color clearly and substantially outperforms texture+color.

**Claim 4: "Findings hold across CNN and transformer backbones with almost no qualitative difference"**
Partially supported. The main-paper evidence for transformers is limited to Cityscapes (Table 2). The paper states the findings generalize to class level but quantitative transformer results for CARLA and PASCAL Context appear only in the appendix, not the main tables. The qualitative ranking stability on the one shown dataset is real, but the scope of the generalization claim exceeds the main-paper evidence.

**Claim 5: "Quantitatively, small objects and boundary pixels are better predicted by shape experts; texture dominates interiors/large segments"**
Partially supported. Table 4's boundary vs. interior accuracy is the clearest evidence and stands on its own: shape experts consistently outperform texture experts on boundary pixels across all three datasets. The "small objects" claim is limited to two classes on CARLA (Fig. 5). The paper's text says "for the large road segments, the texture expert achieves a high segment-wise recall," which is consistent with Table 4's CARLA interior result. The late fusion attribution is a proxy, not a causal decomposition.

**Claim 6: "First empirical evidence for a consistent and intuitive ordering of cue influences"**
Partially supported. The rankings show broad consistency (Tables 2–3), but notable rank changes exist across datasets (e.g., T_RGB and S_SEED-HS swap positions; T_V shifts substantially on CARLA). More critically, the comparison between shape-expert and texture-expert settings is confounded by the surrogate Voronoi task vs. original-scene evaluation (see Weaknesses).

---

## Strengths

- **Novel problem formulation**: Studying "what can be learned from each cue from scratch" rather than "what biases exist in pre-trained models" is a meaningful and underexplored contribution to the segmentation literature. No prior work has applied this frame systematically to semantic segmentation.
- **Creative texture extraction method**: The Voronoi-based pipeline (Sec. 3) is an inventive solution to the specific problem of generating a valid segmentation task for texture study, where simple patch shuffling (used in classification) destroys semantic integrity.
- **Multi-granularity analysis**: Going beyond dataset-level mIoU to class-level (Fig. 3) and pixel-level (Table 4, Fig. 5–6) analysis yields genuinely informative observations—especially that shape experts consistently outperform texture experts on boundary pixels (Table 4).
- **Three diverse datasets**: Using Cityscapes (urban real-world), PASCAL Context (diverse in/outdoor), and CARLA (synthetic) provides meaningful breadth. The contrast between synthetic CARLA (where texture is highly discriminative) and real-world datasets (where shape+color dominates) is itself an insightful finding.
- **Good experimental hygiene**: Multiple random seeds with reported variance, consistent backbone sizing, and domain-shift-free evaluation included in the appendix reflect careful experimental design.

---

## Weaknesses

### Fatal
*(None: the paper's core contributions—boundary vs. interior analysis, the novel surrogate texture task, the dataset-level cue comparisons—survive the methodological concerns, even if some headline claims are overstated.)*

### Major

- **The central quantitative claim "neither shape nor texture clearly dominates" is not supported by the reported numbers on real-world data.** On both Cityscapes and PASCAL Context, S_SEED-RGB outperforms T_RGB by large, consistent margins (42.22% vs. 20.10% on Cityscapes; 31.32% vs. 17.75% on PASCAL). The paper's own text in Sec. 4.2 simultaneously states "neither texture nor shape clearly dominate" and "shape and texture are equally important cues for successful learning," yet the tables contradict this. The only dataset where texture approaches shape is CARLA (55.89% vs. 44.78%), which is synthetic with unusually discriminative limited textures. The conclusion should instead be: *shape+color substantially dominates texture+color on real-world data; texture is competitive only in synthetic settings with highly discriminative appearance.*

- **The shape vs. texture comparison is confounded by non-parallel task constructions.** The texture expert is trained on a Voronoi surrogate segmentation task with uniform random class-to-cell assignment, differing substantially from the original scene layout. The shape expert is trained on transformed versions of the original images (EED/HED) that preserve the original scene structure. These are not symmetric interventions: the texture expert faces a structurally different training task, with different spatial statistics, class distribution geometry, and boundary characteristics. The poor boundary performance of the texture expert in Table 4 may reflect that Voronoi training never showed organic class boundaries, not that texture is intrinsically poor at boundaries. This confound is not addressed in the main paper beyond a brief mention in Sec. 4.2. The paper's conclusion section frames Table 4 as a general insight about texture, but the result is at least partially an artifact of the Voronoi construction.

- **Architecture generalization claim exceeds main-paper evidence.** The abstract states findings hold across architectures; the main text says "these findings generalize to the class level." However, quantitative transformer results for CARLA and PASCAL Context appear only in the appendix. The main paper's only transformer table is for Cityscapes (Table 2). Until multi-dataset transformer results are in the main paper, the claim of broad generalization is asserted, not demonstrated in the primary submission.

### Minor

- **HED's poor performance conflates domain shift with cue information.** The paper itself demonstrates this directly: applying HED preprocessing at test time yields 55.80% mIoU on Cityscapes vs. 13.38% when trained on HED and tested on RGB (Sec. 4.2). The paper acknowledges this but treats it as an exception rather than recognizing it as a systemic concern affecting all cross-domain cue evaluations. If domain shift can explain a ~42 percentage-point gap for HED, it could explain substantial portions of other cross-domain gaps as well.

- **The "small objects" finding is too narrowly supported.** The claim in the Introduction that "small objects and pixels at object borders are dominantly better predicted by shape experts" is stated as a headline finding but supported only by Figure 5 covering two classes on CARLA. The boundary claim is well-supported (Table 4, three datasets), but the "small objects" claim needs broader evidence.

- **The threshold of 20% IoU for class "support" is set by visual inspection.** This arbitrary threshold (Sec. 4.2) is used to determine which classes an expert "can deal with," affecting the qualitative conclusions about which cues matter per class. No sensitivity analysis is provided.

### Trivial

- **Late fusion attribution language is imprecise.** Describing fusion preference as measuring "which cue contributes most" slightly overstates what a learned fusion over softmax outputs can establish. This is common imprecision in empirical work of this kind, but the boundary-accuracy results in Table 4 (which do not depend on fusion) provide stronger independent support.

---

## Nice-to-Haves

- **Include same-domain (cue-specific) test evaluation for all experts prominently.** The domain-shift-free evaluation in the appendix is important context. Moving at least a summary to the main paper would help readers disentangle cue information content from generalization gap.
- **Quantitative validation that EED removes texture.** The surprise finding that S+C without texture achieves strong results assumes EED eliminates texture; a brief texture-classification test on EED outputs would strengthen this claim.
- **Pre-trained backbone comparison.** Since nearly all deployed segmentation models use ImageNet pre-training (which introduces texture biases), extending even one dataset to pre-trained backbones would substantially increase practical relevance.
- **Statistical significance testing** for rank comparisons, given that ranks are used as the primary claim vehicle.
- **Discuss Voronoi boundary artifact explicitly.** Acknowledging that Voronoi training produces straight-edge boundaries—and that this may inflate measured boundary disadvantage for texture experts—would improve scientific rigor.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Training from scratch does not reflect common practice"** (Spark reviewer): The paper's design rationale is explicit—training from scratch avoids contaminating cue-specific training with pre-trained features. Criticizing the absence of pre-trained experiments is scope creep; this is by design, not oversight. Kept as a nice-to-have.
- **"Small backbone size undermines generalizability"**: Both chosen backbones (~14–16M params) are appropriately matched and the paper explicitly frames this as a controlled study. Requesting ResNet101 or SegFormer-B5 is a generic demand that doesn't harm the core contribution.
- **"Unclear practical significance"** (Human Finder): The paper explicitly scopes practical relevance as future work (Sec. 5) and provides two motivating downstream directions (uncertainty/safety, task complexity quantification). While these aren't fully developed, calling the paper "observational without actionable takeaways" overreaches given the stated scope.
- **"Cannot independently verify availability of CARLA 0.9.14"**: CARLA is a widely-used open-source simulator and cited accordingly. This type of existence/availability concern is excluded by hard rule.

---

## Novel Insights

The most genuinely novel observation—underemphasized in the paper's own framing—is the stark *domain dependence* of cue utility: on synthetic CARLA where textures are limited and highly class-discriminative, texture rivals shape; on real-world datasets with heterogeneous, less-discriminative textures, shape+color dominates by large margins. This finding reframes the field's texture-vs-shape debate: the relative influence of these cues is not a fixed property of DNNs but depends critically on the texture informativeness in the training domain. The paper gestures at this ("Cityscapes is in general relatively poor in texture") but does not foreground it as the organizing principle it could be.

---

## Suggestions

1. Revise the abstract and contribution bullet points to state accurately: *on real-world datasets, shape+color substantially outperforms texture+color; on synthetic data with limited discriminative textures (CARLA), texture becomes competitive.*
2. Explicitly acknowledge in Sec. 3 and Sec. 4.2 that Voronoi training may artificially disadvantage the texture expert at boundaries due to its non-organic boundary geometry, and note what would be needed to disentangle this.
3. Promote the domain-shift-free evaluation (currently in appendix) to a main-paper result, particularly the HED 55.80% result, to contextualize the cross-domain evaluations properly.
4. Move transformer results for CARLA and PASCAL to the main paper (even in abbreviated form) before claiming architecture-generalization in the abstract.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Decision | Scores | Relevance |
|---|---|---|---|
| `Yr4RgiZ7P5.md` (DiST, shape bias) | Reject | 6/6/3/6 | Similar topic; similar scope of architecture testing; rejected partly for limited architecture breadth |
| `NTWtNjlThd.md` (Shape/Texture Disentanglement) | Reject | 6/5/5/5 | Closely analogous: creative but limited cue isolation, biases rather than enforces disentanglement |
| `iVMcYxTiVM.md` (VLM Shape/Texture Bias) | Accept | 6/6/8/8 | Comprehensive analysis, strong empirical coverage across model families, clearer findings |
| `BM9qfolt6p.md` (LucidPPN, color/shape/texture) | Accept | 6/6/8/6 | Novel methodological contribution with validated application |
| `RAB5gmMBPS.md` (Segmentation TTA study) | Reject | 3/5/5/3/3 | Pure empirical study without strong methodological contribution; findings not clearly actionable |

**Positioning:** This paper is above the two rejected shape/texture papers (Yr4RgiZ7P5, NTWtNjlThd) in ambition and scope—three datasets, pixel-level analysis, architecture comparison. It falls below the accepted `iVMcYxTiVM` paper in evidentiary clarity: that paper's findings are consistent and headline claims well-supported, whereas this paper's main claim ("neither texture nor shape dominates on real-world data") is contradicted by its own tables. The paper is better than `RAB5gmMBPS` (which has minimal methodological contribution) but shares with it a tendency toward weaker actionable conclusions than its scope promises.

The paper has genuine novelty—it is the first to systematically study cue influence in segmentation—and includes creative methodology (Voronoi texture), real findings (boundary vs. interior, shape+color surprise), and appropriate experimental breadth. However, the central quantitative claim is overclaimed, and the shape vs. texture comparison is methodologically confounded in a way the paper insufficiently acknowledges. These are significant issues that are addressable in revision but are not trivial. The paper sits at **marginally below acceptance** given current state.

**Final Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>