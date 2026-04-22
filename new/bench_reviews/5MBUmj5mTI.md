Now I have a thorough understanding of the paper. Let me compose the final review.

## Summary

This paper presents the first systematic study of how different image cues (shape, texture, color) and their combinations influence learning success in semantic segmentation DNNs. The authors develop a generic procedure to decompose datasets into cue-specific variants—including a novel Voronoi-based texture extraction method—train "cue experts" on these decompositions, and evaluate their performance on original test images across three datasets (Cityscapes, CARLA, PASCAL Context) using both CNN and transformer backbones, with additional pixel-level late-fusion analysis of location-dependent cue influence.

## Strengths

- **Novel and important research question:** The paper shifts from studying cue *biases* in classification to cue *influence on learning* in segmentation, enabling pixel-level, class-level, and location-dependent analyses that classification studies cannot provide. This is a genuinely underexplored angle (Section 1, abstract).

- **Comprehensive experimental framework with 14 cue combinations:** The systematic shorthand notation (Table 1) covering shape, texture, and color (including V/HS decomposition) enables more granular analysis than prior work. Training from scratch with multiple seeds and reporting standard deviations (Tables 2–3) is methodologically sound.

- **Novel texture extraction method for segmentation:** The Voronoi-mosaic approach (Section 3, Figure 2) solves a genuine technical challenge—prior patch-shuffling methods destroy spatial coherence needed for pixel-level segmentation, making this a real contribution.

- **Actionable location-dependent findings:** Table 4 provides concrete quantitative evidence that shape experts substantially outperform texture experts at segment boundaries (56.49% vs 37.16% on Cityscapes, 19+ pp gaps across all datasets), while Figure 6's late-fusion heatmap visually corroborates this. This fine-grained finding is uniquely enabled by the segmentation setting.

- **Cross-dataset consistency with informative variations:** The core findings hold across three datasets spanning real-world street scenes, synthetic scenes, and diverse indoor/outdoor scenes (Tables 2–3). The variation on CARLA—where texture experts outperform shape experts on interiors due to highly discriminatory synthetic textures (Table 4)—provides a useful controlled comparison that strengthens the interpretation.

## Weaknesses

### Fatal

None.

### Major

- **The evaluation protocol conflates cue informativeness with domain shift.** The central measurement trains experts on cue-decomposed data and evaluates on original images, but cue decompositions induce vastly different domain gaps. EED-based shape extraction preserves much of the spatial layout of the original image (smoothing texture while keeping edges and color gradients intact), while the Voronoi texture extraction completely reorganizes spatial structure—replacing natural object boundaries with Voronoi cell boundaries and assigning classes "uniformly at random" (Section 3). The paper itself provides the strongest evidence that this matters: when HED is applied as test-time preprocessing instead (domain-shift-free), the HED vs. EED ranking reverses on Cityscapes (55.80% vs 48.47%, Section 4.2). This directly demonstrates that domain shift can invert apparent cue rankings. The paper acknowledges this but relegates the domain-shift-free analysis to a brief mention and the appendix, and does not apply it to texture experts. While the paper's qualitative rankings are still informative, the claim in the abstract that "the way DNNs perceive the world can be broken down into distinct sources of evidence" is stronger than what the current evaluation establishes. The paper more precisely shows cross-domain transfer from cue-decomposed training to natural images, which is a valuable but different finding.

- **EED removes texture but does not eliminate it.** The paper treats S_EED-RGB as providing "shape+color without texture" (Table 1, Section 4.2), but explicitly describes EED as a method that "diminishes texture through diffusion" (Section 3)—not eliminates it. Residual texture information likely contributes to S_EED-RGB's strong performance. The CARLA S_mvv baseline (checkerboard texture removal with rendering engine access) partly addresses this but is only reported for S+V. No ablation quantifies how much texture EED retains or how much the residual texture contributes, leaving the core "shape+color without texture" finding potentially inflated by unremoved texture.

- **Color experts are architecturally constrained to 1×1 convolutions, confounding cue informativeness with model capacity.** The paper restricts color experts to "a fully convolutional neural network with two to three (1×1)-convolutions" (Section 4.1) to prevent spatial learning, but this means the "C experts are mostly dominated by T experts as well as S experts" finding (Section 4.2) partly reflects that a network with no receptive field is compared against full ResNet18/SegFormer backbones. Color experts with spatial capacity could learn spatial color distributions (e.g., sky is blue at the top), which might make color far more informative than 1×1-convs allow. While the architectural constraint is well-motivated (preventing shape/texture leakage into color), the resulting comparison does not cleanly isolate the contribution of color as a cue.

### Minor

- **Voronoi cells introduce confounding spatial patterns while destroying natural spatial priors.** The random class assignment eliminates spatial priors that exist in real data (e.g., road at bottom, sky at top), while Voronoi cell boundaries create artificial edge structure. The texture expert thus faces a double penalty at test time: it learned from data with absent spatial priors and present artificial structure, neither of which exists in natural images. This is a specific manifestation of the domain-shift concern but worth noting separately for the texture decomposition design.

- **Segment-wise recall analysis (Figure 5) is shown for only two classes on CARLA.** The claim that "small objects and pixels at object borders are dominantly better predicted by shape experts" would be strengthened by systematic evaluation across more classes, especially small-object classes like pole, traffic sign in Cityscapes.

- **Statistical significance of ranking differences is unclear.** In Table 2, S_EED-HS (19.48 ± 3.19) and T_RGB (20.10 ± 0.98) on Cityscapes CNN have overlapping ranges. Whether the claimed ordering is robust to random variation is not tested, especially for adjacent entries in the ranking.

### Trivial

None.

## Nice-to-Haves

- A domain-shift-free or domain-shift-mitigated evaluation for texture experts (e.g., evaluating T_RGB on Voronoi-mosaic test images with Voronoi-assigned labels) would substantially strengthen the core claim about relative cue importance.
- A quantification of EED's residual texture (e.g., via local autocorrelation or Gabor energy metrics) would clarify how much the S_EED results depend on unremoved texture.
- A color expert ablation with spatial capacity (e.g., training on spatially-shuffled-within-segment data while preserving spatial color distributions) would disentangle color cue poverty from architectural restriction.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that the paper's conclusions are "unreliable" and that this is a "structural issue that cannot be resolved without redesigning the evaluation protocol."** The domain-shift concern is real and major, but the paper does acknowledge it (Section 4.2, HED pre-processing results) and provides complementary evidence. The findings about location-dependent influence (Table 4), class-level effects (Figure 3), and CARLA's divergence (texture > shape on interiors) are still informative even with the domain-shift confound. The concern warrants major-flagged criticism but not a fatal "unreliable" verdict.

- **Harsh Critic's claim that the paper does not report domain-shift-free results for texture experts.** While true that a Voronoi-to-Voronoi evaluation is absent, the CARLA dataset partially serves this role—texture is more discriminatory in CARLA (highly distinctive synthetic textures) and indeed texture experts outperform shape experts on interiors (Table 4). This provides indirect evidence that texture's informativeness depends on the domain, which the paper discusses.

- **Harsh Critic's demand for a "no information" baseline on Voronoi diagrams (uniform fill, no texture) to test whether Voronoi spatial structure is learnable.** This is an interesting suggestion but goes beyond what the current results require—the "no info" row in Table 2 (randomly initialized DNNs achieving 0.25% mIoU) already provides a floor, and whether Voronoi cells alone enable learning is an additional experiment rather than a flaw in the current design.

- **Strength Finder's claim that the paper shows "strong empirical evidence that shape+color without texture achieves surprisingly high performance" with S_EED-RGB "far exceeding" T_RGB on Cityscapes (42.22% vs 20.10%).** While the numbers are correct, given the domain-shift confound (Major weakness #1), describing this as "strong evidence" overclaims. The gap is informative but its magnitude is partially driven by asymmetric domain shift, not purely by cue informativeness. Downgraded to a qualified strength.

- **Strength Finder's "publicly available code and data generation procedure" claim.** The paper states "Our code including the data generation procedure is publicly available at TBA" (Section 1)—the URL is placeholder ("TBA"), so this cannot be verified as a strength at this time.

## Novel Insights

The paper's CARLA results provide a natural experiment that illuminates the domain-shift issue: texture experts outperform shape experts on CARLA interiors (89.83% vs 82.63% accuracy, Table 4) specifically because CARLA's limited set of highly distinctive synthetic textures makes texture a stronger cue than in real-world datasets. This cross-dataset divergence is not just a limitation—it is informative. It suggests that the relative importance of shape vs. texture is genuinely domain-dependent rather than universally shape-favored, which partially mitigates the domain-shift concern: if the rankings were purely artifacts of asymmetric transfer, we would not expect a clean reversal that aligns with the known properties (low texture diversity) of real-world data.

## Suggestions

- Reframe the abstract and conclusion to clearly distinguish "what DNNs learn from each cue" from "how well cue-decomposed training transfers to original images." The current abstract claim that "the way DNNs perceive the world can be broken down into distinct sources of evidence" is stronger than the evidence supports; a more precise framing (e.g., "cue-specific training transfers to original images with domain-dependent effectiveness") would be more defensible and still impactful.
- Report domain-shift-free results (cue extraction as test-time preprocessing) for all cue experts where feasible, and give this analysis equal prominence with the cross-domain evaluation in Tables 2–3.
- Add a brief quantitative analysis of EED's residual texture (even a simple metric like comparing local variance in EED vs. original images) to bound how much the S_EED results might rely on unremoved texture.

## Score and Decision

**Calibration anchors:**

- **High band (>7):** `rmg0qMKYRQ` (avg 8.0, shape/texture bias in generative classifiers) — this paper similarly studies cue biases systematically but with a cleaner evaluation methodology (direct likelihood comparison, no domain-shift confound). The paper under review is less methodologically clean but addresses a harder problem (segmentation vs. classification) with richer analysis granularity.
- `PBjCTeDL6o` (avg 8.0, UNI framework for addressing texture/colour bias in interpretations) — addresses a related problem with strong theoretical grounding. The current paper has less theoretical foundation but broader empirical scope.
- **Medium band (4–6):** `aM7US5jKCd` (avg 5.25, adversarial robustness in segmentation with methodological confound) — similarly identifies a methodological confound in existing evaluation protocols. This was rejected, partly due to presentation issues and incremental contribution. The current paper has a similar methodological confound (domain shift) but acknowledges it and provides partial mitigation.
- `bb2Cm6Xn6d` (avg 5.5, LLVM perception properties) — systematic empirical study with some evaluation protocol concerns, rejected.
- `SYBdkHcXXK` (avg 6.0, frequency aliasing in segmentation) — novel diagnostic finding + mitigation, accepted poster. The current paper is similar in providing a novel diagnostic framework but lacks the mitigation component.
- **Low band (<3):** `WRxCuhTMB2` (avg 1.67) — fundamentally flawed methodology, no comparison. The current paper is far stronger than this.
- `V73W8MXnNW` (avg 3.0) — visual cue decomposition with limited novelty/poor baselines, withdrawn.

The paper under review has a real methodological confound (domain shift) but also makes genuine contributions (novel segmentation-specific cue analysis, Voronoi texture extraction, location-dependent findings). It is comparable to the medium-band papers that identify evaluation issues in existing paradigms but still provide useful empirical insights. The domain-shift concern is significant but partially addressed by the authors' own CARLA results and HED pre-processing analysis. The paper is stronger than `aM7US5jKCd` (which was rejected at 5.25) because it provides richer multi-granularity analysis and genuinely novel findings (boundary vs. interior effects), but weaker than `SYBdkHcXXK` (accepted poster at 6.0) which offers both diagnostic findings and mitigation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>