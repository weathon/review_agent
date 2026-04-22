## Summary

CLIP-Map proposes a mapping-based compression framework for CLIP models that replaces conventional select-based pruning with learnable Kronecker-factored transformation matrices (F^in, F^out) for width compression and learnable linear combinations (L_depth) for depth compression, followed by knowledge distillation retraining. The method is motivated by model-growth techniques (LiGO, LeTs) reversed for compression, and introduces Diagonal Inheritance Initialization to stabilize the optimization of mapping parameters.

## Strengths

- **Conceptually clean adaptation of mapping-based growth to compression**: The identification of challenges in reversing the growth direction (partial mapping inapplicability, parameter overhead, multimodal adaptation) and the design of Full-Mapping with Kronecker Factorization as a solution is well-motivated and technically sound (Eqs. 3–4 reduce parameter complexity from O(D₁²D₂²) to O(D₁D₂)).

- **Strong empirical gains at extreme compression ratios**: At 1.0% compression, CLIP-Map_tiny achieves 15.8 TR@1 on MSCOCO versus TinyCLIP's best 12.5 (progressive 3×25ep), a 3.3-point improvement (Table 1). At 10% compression, CLIP-Map_small achieves 38.4 vs 36.2 TR@1 vs progressive TinyCLIP. These are consistent and meaningful improvements.

- **Diagonal Inheritance Initialization is well-justified and critical**: Table 5 shows a dramatic gap—diagonal init achieves 28.9% IN-1K versus 4.9% (Xavier) and 4.4% (Kaiming). The variance analysis in Eqs. 5–8 formally explains why standard initializations cause multiplicative variance scaling, justifying the diagonal design.

- **Mapping optimization provides meaningful gains beyond diagonal initialization on retrieval metrics**: Table 4 shows that 5 mapping epochs + 20 retraining epochs improves MSCOCO TR@1 from 33.8 (Manual Drop + distillation) to 38.3 (+4.5 points) and IR@1 from 20.2 to 23.1 (+2.9 points), within the same total epoch budget.

- **Good ablation studies**: Table 4 (mapping/retraining duration), Table 5 (initialization methods), and the training loss ablation (A.8) provide useful insight into design choices. The finding that excessive mapping epochs (7+18) degrade performance is an informative negative result.

## Weaknesses

### Fatal
None.

### Major

- **The paper overclaims the "mapping vs. selection" paradigm distinction**: The paper frames mapping-based compression as fundamentally superior to select-based pruning because it "preserves as much information from the original weights as possible" (Abstract, Contribution 1). However, diagonal initialization (F^in = F^out = I_{D_2×D_1} with diagonal=1) produces W' equal to the top-left D_2×D_2 submatrix of W — which is functionally identical to the "hard parameter removal" the paper criticizes. The method's viability thus hinges on the very approach it argues against. The paper does not explicitly confront this tension. While the subsequent mapping optimization does improve upon the diagonal initialization, the gains are modest on ImageNet (+1.0%, Table 4) though more meaningful on retrieval (+4.5% TR@1). The paper should honestly acknowledge that the practical contribution is an improved initialization + optimization procedure rather than a fundamentally different paradigm from weight selection.

- **Large unexplained regressions on specific downstream datasets at the 39M scale**: Table 2 shows that at the ViT-39M/16 scale, CLIP-Map suffers catastrophic drops on VOC2007 (76.0→22.2, −53.8 points) and Oxford Pets (80.8→48.5, −32.3 points) compared to progressive TinyCLIP, while simultaneously achieving large improvements on Stanford Cars (51.7→69.2, +17.5) and FGVCAircraft (15.7→50.8, +35.1). The paper's description that results at the base scale are "competitive" and "comparable" (Section 4.2) glosses over these extreme swings. These regressions suggest the mapping may distort certain feature subspaces in ways that are dataset-specific, which could indicate a fundamental limitation of the approach at moderate compression ratios. The paper should investigate and discuss this inconsistency rather than presenting only aggregate narratives.

### Minor

- **Efficiency claims based on seen samples rather than compute**: Table 3 highlights that CLIP-Map_base uses 0.30B seen samples versus TinyCLIP-39M/16's 0.75B. However, the mapping stage requires forwarding through the entire frozen teacher model alongside the mapped student for every training step, which may be more expensive per sample than standard distillation-only training. A comparison in terms of total FLOPs or wall-clock time (the paper mentions A.6 for this but the main text claims remain unsupported without it) would strengthen the efficiency argument.

- **Meta-CLIP variant underperforms progressive TinyCLIP at 10% compression**: Table 1 shows CLIP-Map_base (Meta-CLIP) at 10% compression achieves 34.3 TR@1, below progressive TinyCLIP's 36.2. This is an exception to the claimed consistency of improvements and is not discussed. While the Meta-CLIP variant still outperforms non-progressive TinyCLIP (33.8), the paper's generalization claims should be qualified.

- **No analysis of how far learned mappings deviate from diagonal**: The paper mentions (Section 4.3) that "the distribution of the mapping matrix gradually evolves from an initial diagonal pattern toward a more uniform structure" but does not quantify this. Knowing how much F^in and F^out deviate from identity after training would help assess whether the learned mapping captures meaningful cross-dimensional relationships or remains near-diagonal (which would mean the method is essentially doing weight selection with a small perturbation).

- **Kronecker factorization expressivity is unanalyzed**: The factorization constrains the mapping from O(D₁²D₂²) to O(D₁D₂) free parameters — a reduction of roughly D₁·D₂/2 in degrees of freedom. No experiment tests whether this constraint hurts performance relative to a less aggressive factorization or a block-structured approximation.

## Nice-to-Haves

- Compute-matched comparison with "Manual Drop + distillation" at the same total FLOPs (not just same epoch budget), to determine whether the mapping stage's gains are cost-effective.
- Analysis of why VOC2007 and Oxford Pets regress so severely at the 39M scale, which could reveal whether the mapping introduces systematic biases in certain feature subspaces.
- Comparison with random small-model initialization + distillation to isolate the contribution of pretrained weight inheritance versus the distillation stage.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The entire mapping optimization stage adds 1% accuracy" (Harsh Critic #1)**: The critic cherry-picked ImageNet top-1 from Table 4. The same table shows +4.5% on MSCOCO TR@1 and +2.9% on IR@1 from the mapping stage. While the ImageNet gain is indeed modest, the retrieval gains are meaningful and the critic's summary is misleading by omission.

- **"Gains are concentrated at extreme compression where absolute performance is poor" (Harsh Critic #2)**: This is partially valid but overstates the issue. At 10% compression (8+3M parameters), CLIP-Map achieves 38.4 TR@1, which is a viable operating point for many applications. The 1% compression results are indeed in a regime of poor absolute performance, but the method explicitly targets extreme compression as its strength. This is more of a scope consideration than a flaw.

- **"SparseGPT and other modern baselines are missing" (Harsh Critic, Missing Experiments #2)**: SparseGPT is a one-shot pruning method for LLMs, not a CLIP compression method with distillation. The paper's comparison set (TinyCLIP, CLIP-KD, MobileCLIP, MoPE-CLIP) covers the CLIP compression landscape. Requesting baselines from a different domain is scope creep.

- **"Abstract claim that mapping 'preserves as much information as possible' is information-theoretically false" (Harsh Critic, Section-by-Section)**: This is an imprecise marketing claim, not a technical assertion the paper relies on. The Kronecker-structured mapping is clearly a constrained subspace. This is a rhetorical overclaim, not a methodological error.

- **"Missing λ ablation in main paper" (Harsh Critic, Section 3.2.4)**: The paper states the ablation is in A.8. This is standard appendix placement for a secondary hyperparameter. Not a significant omission.

- **Strength Finder claim: "Training efficiency—fewer seen samples for comparable or better performance"**: While Table 3 supports this claim on sample count, the per-sample compute cost difference between the mapping stage and standard distillation is not analyzed. This strength is weakened by the corresponding minor weakness about compute-based efficiency claims.

## Novel Insights

The most revealing finding in this paper is one it does not adequately discuss: the method's success is built on a diagonal initialization that is semantically equivalent to the "select-based" weight inheritance the paper argues against. This creates a paradox—the mapping paradigm only works because it starts from a selection, and the learned deviations from that selection, while beneficial, are relatively modest (particularly on ImageNet). This suggests that the real contribution may be less about "mapping versus selection" and more about the value of an optimization stage that refines a selection-based initialization toward a better local optimum before distillation. The extreme dataset-specific swings at the 39M scale (Table 2) further suggest that the mapping's effect is highly non-uniform across the feature space, which is an important open question for future mapping-based compression work.

## Suggestions

- Explicitly acknowledge in Section 3.2.3 that diagonal initialization is equivalent to weight selection, and reframe the contribution as "starting from weight selection and optimizing beyond it" rather than "replacing selection with mapping." This would make the paper's claims more honest and actually strengthen the narrative by showing that even a small departure from selection yields consistent gains.
- Add a brief discussion of the VOC2007/Oxford Pets regressions in Table 2, even if the explanation is preliminary (e.g., hypothesis about mapping distorting class-discriminative features for fine-grained categories). The current silence on these results undermines trust in the method's generality.
- Report total training FLOPs or wall-clock time in Table 3 alongside seen samples, to make the efficiency claims verifiable.

## Score and Decision

**Calibration anchors:**

- **High-scoring anchors**: Data-free CLIP distillation (avg 8.0, Oral — novel DFKD method for CLIP with cleaner claims); LLM-Streamline (avg 7.5, Spotlight — similar structure of layer pruning + lightweight replacement with strong gains); Intra-Fusion (avg 7.2, Spotlight — genuine paradigm shift from selection to fusion in pruning); DAREx (avg 7.5, Spotlight — comprehensive analysis of delta-parameter pruning with practical improvements). CLIP-Map falls below these because its core mapping contribution is smaller than claimed, and the unaddressed regressions at 39M are a significant gap.

- **Medium-scoring anchors**: Parameter sharing compression (avg 5.5, Poster — strong experiments on small datasets with limited baselines); FiPS tensor decomposition (avg 4.75, Reject — limited baselines and insufficient analysis); ResidualViT (avg 5.75, Reject — token reduction for CLIP with mixed results). CLIP-Map has a more novel core idea and better experiments than FiPS, and more comprehensive evaluation than the parameter sharing paper. It is stronger than these.

- **Low-scoring anchors**: Efficient Low-Rank Diffusion (avg 2.5 — incremental combination of existing ideas, overclaimed); Delta parameter editing (avg 2.33 — shallow exploration); VQ compression (avg 4.67 — overclaimed FLOPs reductions). CLIP-Map is clearly better than these, with genuine novelty and solid empirical results.

CLIP-Map sits between the medium and high anchors. It has genuine novelty and strong empirical results at extreme compression, but the overclaiming of the mapping paradigm's distinction from selection and the unexplained large regressions at the 39M scale are significant weaknesses that prevent it from scoring with the 7+ papers. At the same time, it is clearly above the 4-5 range papers that have fundamental methodological issues or trivial contributions. A score of 5.5 reflects its position as a borderline paper with real contributions that are somewhat undermined by overclaimed framing.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>