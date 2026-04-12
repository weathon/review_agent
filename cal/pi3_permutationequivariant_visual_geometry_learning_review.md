=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary
This paper proposes **π³**, a feed-forward multi-view geometry model that removes the fixed-reference-view design used by prior methods and instead enforces **permutation equivariance** over the input image set. The model predicts per-view local point maps and relative camera geometry, and empirically shows strong performance across pose estimation, point-map reconstruction, video depth, and monocular depth, with especially convincing gains in robustness to input ordering.

## Strengths
- **The paper identifies and directly targets a concrete, under-examined failure mode of recent feed-forward geometry models: dependence on an arbitrary reference frame.** This is not just a philosophical reformulation; the paper backs it with a targeted robustness study (Sec. 4.4, Table 6) showing that reference-dependent baselines vary substantially when the input order changes, whereas π³ exhibits near-zero variance.
- **The architectural change is specific and conceptually clean.** In Sec. 3.1, the authors explicitly formulate permutation equivariance and implement it by removing order-dependent components such as frame-index positional embeddings and special reference/camera tokens, while retaining a practical alternating view-wise/global attention backbone.
- **Empirical results are strong across multiple geometry tasks rather than only one narrow benchmark.** Examples include Sintel pose ATE improving from **0.167 (VGGT) to 0.074** (Table 1), Sintel video-depth Abs Rel improving from **0.299 to 0.233** (Table 4), and strong point-map reconstruction results on ETH3D and DTU (Table 3).
- **The method appears to buy robustness without sacrificing efficiency.** On KITTI video depth, π³ runs at **57.4 FPS**, faster than VGGT’s **43.2 FPS** while also using fewer parameters (**959M vs 1.26B**, Table 4).
- **The paper is unusually transparent about an important optimization difficulty.** Appendix A.4 openly states that training the core formulation from scratch is unstable and discusses the “cold start” issue of dense relative supervision; that transparency makes the paper easier to assess and improves credibility.
- **The appendix includes a useful controlled comparison against VGGT under matched training conditions.** Table 8 is not the main result, but it is informative: when trained from scratch under the same setup, the plain π³ variant underperforms VGGT, while a π³ variant with a global proxy head can outperform it on some datasets. This helps clarify where the method’s strengths and difficulties lie.

## Weaknesses
###: Fatal
- None.

### Major:
- **The main empirical gains are not cleanly attributable to the proposed architecture because the final model is not trained from scratch and reuses strong VGGT priors.** Appendix A.2 states: *“Our final model is not trained from scratch. Instead, we initialize the weights for the encoder and the alternating attention module from the pre-trained VGGT model, and we keep the encoder frozen during training.”* This is a real methodological caveat. Since the paper’s core claim is that removing the fixed-reference inductive bias improves robustness and accuracy, the causal attribution is blurred: some gains could come from inheriting strong pretrained geometric representations from VGGT rather than from permutation equivariance alone.
- **The paper itself shows that the proposed reference-free formulation is difficult to optimize from scratch, which weakens the strength of the “new paradigm” claim.** Appendix A.4 explicitly says that training π³ from scratch with only the core objectives leads to *“suboptimal convergence”* and attributes this to a cold-start problem from the \(N \times N\) relative constraints. The authors stabilize training by introducing an auxiliary **global proxy** head that uses a reference view during training. While this does not invalidate the final inference-time architecture, it does materially weaken the claim that the method fully eliminates reference-view bias in the learning process.
- **The ablation study is too coarse to isolate which design choices are responsible for the gains.** Table 7 compares Model 1, Model 2, and the full model, but Appendix A.6 shows that these variants differ in more than one way at once: camera token usage, pose formulation, loss computation, and point-map normalization all change together. As a result, the ablation supports that the full package helps, but does not cleanly establish the individual contribution of permutation equivariance, scale-invariant point maps, affine/similarity-invariant pose supervision, or the decoder design.
- **The robustness evidence would be stronger with a clearer evaluation protocol and harder stress tests.** Table 6 is compelling, but Sec. 4.4 only evaluates permutations formed by rotating which frame is first, not arbitrary shuffles. In addition, because point-cloud metrics are computed after alignment to ground truth, the paper should explicitly specify whether alignment is solved independently for each permuted output or held fixed across permutations. This does not negate the result, but the current protocol leaves some ambiguity about how much of the near-zero variance comes from the model versus the evaluation alignment.

### Minor
- **The paper overstates “simplicity” in places.** The abstract and introduction describe the approach as “simple and bias-free,” but the actual training recipe is fairly involved: two-stage training, large-scale multi-dataset aggregation, VGGT-based initialization, frozen encoder, separate confidence-head training, and—when training from scratch—a proxy task for stabilization (Appendix A.2, A.4). The inference architecture is clean; the full training pipeline is less so.
- **The terminology around camera pose invariance is imprecise.** Section 3.3 repeatedly uses “affine-invariant camera pose,” but the text itself describes ambiguity under a **similarity transformation**: rigid motion plus one global scale. That is narrower than general affine invariance.
- **Dynamic-scene claims are not as thoroughly substantiated as static-scene claims.** The paper states that π³ handles static and dynamic content and cites an internal dynamic-scene dataset in training, but the main quantitative experiments do not separately analyze dynamic-scene performance or failure modes in detail.
- **The point-map and pose evaluations rely on post-alignment, which is standard but somewhat weakens claims of raw geometric fidelity.** This is not a flaw by itself, but for a paper emphasizing robustness and geometry quality, some additional relative or alignment-free diagnostics would strengthen the case.

### Trivial
- **The confidence head is trained separately rather than jointly with the rest of the model** (Appendix A.2). This is a reasonable engineering choice, but the paper does not analyze whether this affects confidence calibration or usefulness in difficult regions.

## Nice-to-Haves
- A stronger attribution study: train π³ and a reference-based counterpart under as-matched-as-possible conditions, including variants both with and without VGGT initialization.
- Expand the robustness evaluation to include **fully random permutations**, not only cyclic changes of the first frame, and report results for longer sequences.
- Add a clearer protocol note for Table 6 stating exactly how alignment is handled across permutations.
- Provide more component-wise ablations, especially isolating: (i) removal of camera/reference token, (ii) similarity-invariant pose loss, (iii) scale-invariant point-map alignment, and (iv) initialization choice.
- Include a focused dynamic-scene evaluation or failure-case section in the main paper, since dynamic-scene capability is highlighted in the framing.
- Report compute scaling with sequence length \(N\) (latency/memory/FLOPs), since permutation-equivariant multi-view attention is central to the approach.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The model is not truly end-to-end because the confidence head is trained separately.”**  
  Removed as a main weakness. Appendix A.2 clearly explains this as a practical training choice after the main stages, and this alone does not undermine the core claims.

- **“The method is not really reference-free because evaluation still uses scale/alignment.”**  
  Removed. The paper’s claim is about eliminating a **fixed reference view in the model architecture/prediction parameterization**, not about avoiding standard geometric alignment used for evaluation under scale/similarity ambiguity.

- **“Comparison to monocular models is unfair because π³ uses multiple views.”**  
  Removed as a core criticism. The paper is explicit that these are contextual comparisons, and even states in Sec. 4.3 that π³ is *“not explicitly optimized for single-frame depth estimation.”* This is not misleading enough to count as a substantive flaw.

- **“The ROE scale solver is too computationally expensive, so efficiency claims are invalid.”**  
  Removed. The criticism is speculative: the paper does not present this alignment as an inference-time bottleneck, and the reported FPS results concern inference performance on benchmarks. There is not enough evidence in the paper to elevate this into a real weakness.

- **“Zero-shot claims are invalid because of training data overlap on ScanNet/TUM.”**  
  Removed in its stronger form. The paper explicitly states in Appendix A.5 that zero-shot pose estimation is evaluated on **Sintel and TUM-dynamics**, and that ScanNet/ScanNet++ are seen during training. This is disclosed rather than hidden. At most, the paper could present the distinction more prominently.

- **Strengths such as “the paper is well-written” or “the experiments are extensive.”**  
  Removed as too generic without paper-specific substance.

## Novel Insights
The most important synthesis across the evidence is that this paper’s contribution is stronger on **representation and robustness at inference** than on **training methodology**. The permutation-equivariant design appears genuinely useful: the robustness results and cross-task performance indicate that removing the designated reference frame is not merely aesthetic, but addresses a real instability in recent feed-forward geometry systems. At the same time, the paper also reveals an interesting tension: a model that is cleaner and less biased at inference can still depend on biased or asymmetric training scaffolding to become learnable. That makes the work meaningful and technically interesting, but also means the strongest claim is not “reference bias has been eliminated end-to-end,” rather “reference dependence in the deployed architecture can be removed, with measurable robustness benefits, though current optimization still leans on strong priors or curricula.”

## Suggestions
- Add a central experiment explicitly decomposing the source of gains: compare π³ vs. a reference-based variant under identical initialization, frozen/unfrozen settings, and compute budgets.
- Promote Appendix A.4 into the main paper and be more precise in the claims: emphasize that the **final architecture** is reference-free and permutation-equivariant, while **optimization from scratch remains challenging**.
- Strengthen Table 6 with arbitrary shuffles and a protocol statement for alignment handling.
- Revise terminology from “affine-invariant” to “similarity-invariant” unless a broader affine ambiguity is actually intended and justified.
- Add a small but focused dynamic-scene benchmark or failure-case analysis in the main text to support the paper’s broad applicability claims.
- Expand ablations so each claimed ingredient is varied one at a time, especially the removal of the camera/reference token and the use of scale-/similarity-invariant supervision.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
