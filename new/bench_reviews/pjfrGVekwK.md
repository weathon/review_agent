Now I have enough calibration. Let me write the final review.

## Summary
This paper proposes Variational Bayes Gaussian Splatting (VBGS), a framework that reformulates Gaussian Splatting optimization as variational inference over GMM parameters with conjugate priors, enabling closed-form sequential updates for continual learning without replay buffers. The method demonstrates comparable reconstruction quality to gradient-based 3DGS on static datasets while showing superior resistance to catastrophic forgetting in streaming settings.

## Strengths
- **Closed-form sequential update derivation**: The paper derives mathematically sound variational update equations (Eq. 25-26) using Normal Inverse Wishart and Dirichlet conjugate priors, enabling parameter updates from streaming data without storing past observations. This differs fundamentally from standard 3DGS which requires backpropagation through a renderer and typically needs replay buffers for continual learning.

- **Empirical demonstration of forgetting resistance**: Figure 3 shows VBGS maintains reconstruction PSNR over 60-200 sequential steps while gradient-based baselines degrade sharply after initial peaks. This is the most compelling evidence supporting the core claim—the method successfully integrates new data without overwriting prior knowledge in the evaluated settings.

- **Component reassignment heuristic**: Section 3.4 introduces a practical mechanism for reassigning unused mixture components based on negative ELBO sampling, addressing coverage issues in random initialization scenarios. Figure 5b shows this improves PSNR on Habitat room datasets compared to standard VBGS.

## Weaknesses

### Fatal
None

### Major
- **RGBD supervision dependency changes the problem setting**: The paper claims to optimize "3D Gaussian Splats" but actually optimizes a GMM likelihood on 3D point clouds derived from RGBD frames (Section 4.2: "VBGS is trained on the 3D point cloud... In contrast, the gradient-based approach is optimized using multi-view image reconstruction"). This means VBGS receives ground-truth depth supervision while the 3DGS baseline must infer geometry from RGB images alone. Table 1 comparisons are therefore asymmetric—depth supervision solves a significant portion of the 3D reconstruction problem that 3DGS must learn from scratch. The authors acknowledge this in Section 5 ("reliance on RGBD data"), but the positioning as a "3DGS alternative" is misleading when the input modalities differ fundamentally. This limits applicability to scenarios where dense depth is available (robotics/SLAM) rather than the broader NVS use cases 3DGS targets.

- **Accumulating statistics cause vanishing plasticity over time**: The update rule (Eq. 26) accumulates concentration parameters indefinitely: ν_{t,k} = ν_{t-1,k} + Σ γ_{k,n}. As t → ∞, the effective learning rate (Δ/ν_t) vanishes, making the model increasingly rigid—a phenomenon known as plasticity loss in continual learning. The claim in Section 3.3 that "Components without recent data assignments revert to their prior values" is mathematically incorrect; the prior influence (η_0) is diluted as ν_t grows unbounded. While the 60-200 step experiments don't expose this severely, it fundamentally undermines the claim of suitability for long-term autonomous navigation where agents must adapt to new environments indefinitely. This is a known limitation in Bayesian continual learning that requires explicit forgetting mechanisms (decay factors, sliding windows) not present here.

### Minor
- **Computational efficiency comparison uses mismatched budgets**: Section 4.1 compares VBGS (1 update per patch) against Gradient (100 training steps per patch), then reports wall-clock time (0.03s vs 0.05s). While the paper's intent is to show VBGS converges faster, this setup conflates algorithmic efficiency with optimization budget. A fairer comparison would either match step counts (1 vs 1) or measure time-to-convergence at fixed accuracy. The reported speedup may partially reflect the baseline being forced to over-optimize each batch rather than inherent algorithmic superiority.

- **Limited evaluation of long-term continual learning**: Experiments run for 60-200 sequential steps, which is insufficient to assess the plasticity loss concern or memory scaling in realistic deployment scenarios (e.g., hours of robot operation). The paper would benefit from extended runs (1000+ steps) to demonstrate whether performance degrades as ν_t accumulates, and analysis of memory/compute scaling with time.

### Trivial
- **Title slightly overclaims**: "Variational Bayes Gaussian Splatting" suggests optimization of the splatting rendering pipeline, but the method optimizes a GMM on point coordinates with rendering applied only post-hoc for visualization (Section 3.1: "For 3D rendering, we use the renderer from Kerbl et al."). "Variational Bayes Point Cloud Clustering for Splatting" would be more precise, though the current title is acceptable given the final visualization uses splatting.

## Nice-to-Haves
- Introduce a forgetting factor or decay mechanism on sufficient statistics (η_t = ρ η_{t-1} + Δ with ρ < 1) to enable true online adaptation in non-stationary environments and prevent plasticity loss.

- Add an ablation quantifying how much performance gain comes from depth supervision vs. the variational update rule by training the Gradient baseline on RGBD point clouds as well.

- Include failure cases showing scenarios where the accumulation rule fails (e.g., dynamic scenes with moving objects) to illustrate limitations of the current forgetting mechanism.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 1 (Structural: Fundamental Mismatch)**: While the RGBD vs RGB asymmetry is valid, the criticism that this "invalidates the core claim" is too strong. The authors explicitly scope the method for robotics/SLAM applications where depth is available (Section 5), and are transparent about the limitation. The comparison is asymmetric but not fraudulent—it demonstrates VBGS works well in its intended setting. Moved to Major weakness with appropriate framing.

- **Harsh Critic Point 2 (Vanishing Plasticity)**: The mathematical observation about ν_t accumulation is correct, but calling it a "structural flaw" that "contradicts the claim of robust online adaptation" overstates the issue for the paper's evaluated scope. The experiments don't claim indefinite adaptation, and plasticity loss is a known open problem in Bayesian CL. Moved to Major weakness with calibrated severity.

- **Harsh Critic Point 3 (Unfair Computational Comparison)**: The 1-step vs 100-step comparison is indeed mismatched, but the paper's claim is about convergence speed (VBGS reaches target performance in 1 step), not per-step efficiency. This is a presentation clarity issue rather than fundamental unfairness. Moved to Minor weakness.

- **Strength Finder Point on "Computational efficiency per update step"**: This conflicts with the verified weakness about mismatched budgets. The 0.03s vs 0.05s comparison is not apples-to-apples since VBGS does 1 step while Gradient does 100. Removed as it overstates the efficiency claim.

- **Generic strengths about "Problem Motivation"**: Statements like "Addressing catastrophic forgetting in scene representation is a high-value problem" are too generic and apply to any CL paper. Removed per instructions.

## Novel Insights
The paper's core insight—that conjugate prior structure in GMMs enables closed-form sequential updates that naturally resist forgetting—is not novel in the Bayesian literature (this is standard variational continual learning), but its application to 3D scene representation via Gaussian splats is a meaningful contribution. The observation that VI-based updates maintain performance in streaming settings where gradient methods fail is empirically demonstrated but theoretically expected from Bayesian principles. The genuinely novel aspect is the component reassignment heuristic for handling unknown data statistics in streaming 3D reconstruction, which addresses a practical gap in applying GMMs to real-world scene modeling.

## Suggestions
1. **Clarify positioning**: Explicitly frame VBGS as a method for RGBD-based scene reconstruction (robotics/SLAM) rather than a general 3DGS alternative. Add a sentence in the Abstract noting the RGBD requirement upfront.

2. **Address plasticity loss**: Add a Discussion paragraph acknowledging the ν_t accumulation issue and proposing future work on forgetting mechanisms (e.g., exponential decay, sliding windows, or component pruning). This would demonstrate awareness of the limitation.

3. **Fairer efficiency comparison**: Add a supplementary experiment comparing VBGS (1 step) vs Gradient (1 step) to isolate algorithmic differences, or report time-to-convergence at fixed PSNR thresholds.

4. **Extended continual learning evaluation**: Run a 1000+ step experiment on a subset of data to show whether PSNR plateaus or degrades as ν_t grows, providing empirical evidence on the plasticity concern.

## Score and Decision

**Calibration anchors compared:**

| Paper | Avg Score | Comparison to VBGS |
|-------|-----------|-------------------|
| Universal Beta Splatting (51JEkjP0gF) | 6.00 | Strong empirical results with clear generalization; VBGS has comparable math but weaker positioning |
| StreamSplat (SaiDRQU7Ez) | 6.67 | Online 3DGS from video streams; VBGS has better CL results but RGBD limitation |
| PAC-Bayes Continual Learning (hWw269fPov) | 5.33 | Theoretical bounds for CL; VBGS has stronger empirics but similar acknowledged limitations |
| Variational Model Merging (SKtC3JTCyr) | 5.00 | VI with Gaussian posteriors; rejected despite solid theory due to approximation concerns—similar to VBGS's RGBD issue |
| Parametric SDF (mhHlsuGNRW) | 5.50 | Rejected due to unfair multi-view vs monocular comparison; directly analogous to VBGS's RGBD vs RGB asymmetry |
| VPrompt (KVXKX5ue0D) | 3.00 | VI for continual learning; withdrawn due to incremental novelty—VBGS has stronger empirical contribution |
| GS4 Semantic SLAM (QVU0l5wMJu) | 4.50 | RGBD-based GS-SLAM; withdrawn with concerns about unfair comparisons—similar positioning issues |

**Reasoning**: VBGS sits between the solid-but-flawed papers (Parametric SDF at 5.5, Variational Model Merging at 5.0) and the stronger accepts (Universal Beta Splatting at 6.0, StreamSplat at 6.67). The mathematical derivation is sound and the continual learning results are compelling, but the RGBD dependency and unaddressed plasticity loss prevent a clear accept. Compared to Parametric SDF (rejected at 5.5 for unfair multi-view vs monocular comparison), VBGS is more transparent about its RGBD limitation but faces a similar asymmetry issue. Compared to VPrompt (3.0, withdrawn), VBGS has stronger empirical validation and clearer contribution. The paper's strengths align more with the 5.5-6.0 range papers that were accepted despite acknowledged limitations (e.g., GTO at 5.5, ReSplat at 5.5).

**Final score**: 5.5 — borderline accept/poster. The paper makes a solid methodological contribution with good empirical validation, but the RGBD positioning and plasticity concerns prevent a higher score. With revisions addressing the positioning clarity and acknowledging limitations more explicitly, this could be a strong poster.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>