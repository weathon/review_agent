=== CALIBRATION EXAMPLE 40 ===

# Final Consolidated Review
## Summary

π[3] proposes a fully permutation-equivariant feed-forward architecture for visual geometry reconstruction that eliminates the reliance on a fixed reference view, a pervasive inductive bias in prior methods. By predicting affine-invariant camera poses and scale-invariant local pointmaps—without positional embeddings or reference tokens—the model guarantees consistent outputs regardless of input ordering, achieving state-of-the-art performance across camera pose estimation, depth estimation, and pointmap reconstruction benchmarks.

## Strengths

- **Genuinely novel architectural paradigm.** The removal of reference-view tokens and frame-index positional embeddings to enforce permutation equivariance is a clean, principled departure from the dominant DUSt3R→VGGT lineage. The formal equivariance definition (Eq. 1–3) and the architectural realization (Figure 3) are well-matched, and this is the first work to systematically identify and address reference-view bias in feed-forward 3D reconstruction.

- **Convincing empirical validation of the core claim.** Table 6 demonstrates near-zero standard deviation across input permutations on DTU and ETH3D (e.g., 0.003 vs. VGGT's 0.033 on DTU accuracy), providing strong evidence that the architecture achieves genuine permutation equivariance in practice, not just in principle.

- **Consistent SOTA improvements across diverse tasks.** The method outperforms VGGT on Sintel camera pose ATE (0.074 vs. 0.167), Sintel video depth Abs Rel (0.233 vs. 0.299), and ETH3D point map accuracy (0.194 vs. 0.280), while simultaneously being smaller (959M vs. 1.26B params) and faster (57.4 vs. 43.2 FPS). This breadth of improvement from a single design change is compelling.

- **Efficient model design.** Using 36 alternating attention layers versus VGGT's 48 (Appendix A.1) while achieving superior results demonstrates that the reference-free formulation yields a more efficient use of model capacity, rather than simply adding compute.

## Weaknesses

- **From-scratch training instability undermines the "simple and bias-free" framing.** Appendix A.4 reveals that training π[3] from scratch with only its core objectives leads to "suboptimal convergence" due to a "cold start problem" with N×N relative constraints. A reference-based proxy task (global pointmap head with cross-attention to a reference view) is needed to stabilize optimization, and the main model initializes from VGGT pretrained weights. The introduction's characterization of the approach as "simple and bias-free" (Section 1) overstates the case—the optimization landscape is genuinely harder without a reference anchor, and the training pipeline still depends on a reference-biased teacher. This should be discussed prominently, not buried in an appendix.

- **Dynamic scene claims are not quantitatively substantiated.** The abstract, introduction (Figure 1), and method description prominently claim effectiveness on dynamic scenes, yet all quantitative benchmarks (DTU, ETH3D, 7-Scenes, NRGBD, Sintel, etc.) evaluate static scenes. The internal dynamic dataset mentioned in Section 3.4 is never evaluated. This gap between claimed capability and demonstrated evidence is significant for a paper that positions dynamic-scene handling as a key advantage.

- **Inference-time scale recovery is not clearly explained.** During training, the optimal scale factor s* (Eq. 4) is computed using ground-truth depth to align predictions. During inference, ground truth is unavailable, yet the paper does not explain how scale is handled for practical deployment. Evaluation relies on Umeyama+ICP alignment to ground truth (Section 4.2), which is infeasible in real applications like robotics navigation where metric scale matters. The paper should clarify what a user receives at inference and how to obtain metric scale without GT.

- **Ablation confounds architecture and supervision changes.** Models 1 and 2 in Table 7 introduce camera tokens (breaking equivariance architecturally) and simultaneously switch to reference-frame-based loss formulations (Appendix A.6). This makes it impossible to isolate whether the performance gains come from the permutation-equivariant architecture or from the relative supervision scheme. A cleaner ablation—keeping the architecture identical and varying only the loss—would substantially strengthen the causal attribution of improvements.

- **O(N²) camera loss complexity is an unaddressed scalability concern.** Equation 8 sums over all ordered view pairs (i≠j), yielding quadratic cost in sequence length. For the N≤24 used in training (Appendix A.2), this is manageable, but the introduction claims applicability to "video sequences" and "unordered image sets" without discussing whether pair subsampling is needed for longer sequences or how performance scales with N.

## Nice-to-Haves

- Test truly random full-sequence permutations (not just cycling each frame to the first position) to further validate equivariance, though the architectural guarantee and cyclic results make this likely redundant.

- Show concrete visual failure cases where baseline methods produce degraded reconstructions due to poor reference view selection; Figure 2 shows metric variation but no visual evidence of catastrophic failures that would make the problem more visceral.

- Explain the mechanistic connection between permutation equivariance and the low-dimensional pose manifold (Figure 4/6); the observation is interesting but the causal link is asserted rather than analyzed.

- Provide a dedicated discussion of practical scale recovery strategies at inference time (e.g., known object size, IMU integration, single metric depth sensor).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **VGGT initialization breaks equivariance (Harsh Critic):** The concern that initializing from a non-equivariant VGGT model introduces implicit order-dependent biases is theoretically interesting, but Table 6 empirically demonstrates near-zero permutation variance, indicating the architecture enforces equivariance regardless of initialization source. The weights don't encode positional information—only the architecture does.

- **Speed advantage unexplained (Spark Finder):** The 57.4 FPS vs. 43.2 FPS difference is explained in Appendix A.1: π[3] uses 36 alternating attention layers versus VGGT's 48 layers, directly accounting for the speed difference.

- **Coordinate convention ambiguity in Eq. 7 (Harsh Critic):** Requesting explicit Cam-to-World vs. World-to-Cam convention is a clarity nitpick; the equation is self-consistent as written.

- **Differentiability of s* solver (Harsh Critic):** While a legitimate technical question, the ROE solver from MoGe (Wang et al., 2025c) has a closed-form solution for scale alignment that is differentiable. Even if s* were detached, gradients still flow through the predicted points x̂ in Eq. 5. This doesn't constitute a substantive weakness.

- **Missing related works (various):** Per hard rules, omitted.

- **Formatting issues in equations (Harsh Critic):** Acknowledged as parser artifacts, not paper issues.

## Novel Insights

The most striking insight emerging from the reviews is the tension between the paper's conceptual elegance (reference-free, permutation-equivariant geometry) and its practical training reality (reference-biased initialization, reference-based proxy task for stability). This suggests that the reference-free formulation represents a *representation-level* advance—the right output parameterization—but that the *optimization landscape* for learning such representations from scratch remains fundamentally harder than reference-anchored alternatives. Future work might focus on better initialization strategies or curricula specifically designed for relative constraint optimization, rather than relying on knowledge distillation from reference-based teachers. The low-dimensional pose manifold observation (Figure 4/6) hints that permutation-equivariant architectures may implicitly regularize toward structured, physically plausible camera trajectories—a property that could be leveraged more explicitly.

## Suggestions

- Elevate the from-scratch training difficulty and proxy-task dependency from Appendix A.4 to the main paper (at minimum in Section 3.4 or the Limitations section), and soften the "simple and bias-free" language in the introduction to reflect that the inference-time design is bias-free but the training pipeline is not.

- Add a brief subsection or paragraph explaining the inference pipeline for a practitioner: what does the model output, and how does one recover metric scale or perform multi-view fusion without ground-truth alignment?

- Include at least one quantitative evaluation on a public dynamic-scene benchmark (e.g., a dynamic subset of Sintel or another established dataset) to substantiate the dynamic-scene claims that currently appear only in qualitative figures.

- Redesign the ablation (Table 7) to isolate equivariance architecture from relative supervision, ideally by training the same architecture with reference-based vs. relative losses while keeping all other components constant.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
