## Summary

Jigsaw++ introduces a generative framework for producing category-agnostic complete shape priors from partially assembled 3D objects, using a rectified flow model with a novel 2D-3D coordinate-to-color mapping and a "retargeting" fine-tuning stage. The method demonstrates improved shape reconstruction quality over existing assembly baselines on both fracture and part assembly datasets, and shows robustness to missing pieces. However, the practical integration of these priors into actual reassembly pipelines remains incomplete, and the evaluation primarily measures shape completion quality rather than assembly improvement.

## Strengths

- **Novel shape prior generation for reassembly**: The paper addresses a genuine gap in 3D reassembly—the lack of complete shape priors to guide fragmented inputs—and proposes a concrete generative solution. Table 1 demonstrates consistent improvements in reconstruction quality (CD, precision, recall) across multiple baselines and datasets.
- **Clear problem formulation and honest scope**: Section 3.1 explicitly defines the input (partially assembled point cloud), output (complete shape prior), and purpose (guidance layer for reassembly, not a replacement). The paper honestly acknowledges limitations in Section 6.3, identifying failure modes for scale extremes, unseen categories, and complex topologies with visual examples.
- **Robustness to incomplete inputs**: Table 2 left shows the method maintains near-identical performance with 20% of pieces randomly removed (CD 2.0 vs 1.8 ×10⁻³, recall 59.4% vs 59.4%), demonstrating practical utility for real-world scenarios where fragments are missing.
- **Rectified flow enables efficient fine-tuning**: The ablation in Figure 5 validates that reducing reverse sampling steps to k=1/10 improves results over full sampling, supporting the claim that straight trajectories enable efficient retargeting without overly mimicking biased inputs.

## Weaknesses

### Fatal

None.

### Major

- **Evaluation protocol measures shape completion rather than assembly improvement** — The paper claims Jigsaw++ improves "existing assembly methods" and demonstrates this in Table 1 by comparing geometric fidelity metrics (CD, Precision, Recall) between "baseline assembly output" and "complete shape prior." But these metrics compare fundamentally different outputs: a fractured/partial reassembly versus a complete generated shape. A complete shape will naturally score lower CD and higher precision/recall against ground truth than a fractured assembly, regardless of whether the prior meaningfully guides reassembly. The paper acknowledges in Section 6.2 that it "encountered challenges in finding an algorithm that effectively utilizes the complete shape prior" and resorts to an oracle-like matching scheme using ground truth positions for the assembly experiment in Table 2 right. This means the central claim—that the method improves assembly—is not directly validated by the experiments shown. This is the paper's most significant gap.

- **Unclear whether pretrained 2D features are genuinely leveraged** — The core innovation of using DINOv2/LEAP is motivated by leveraging "massive 2D datasets" (Section 3.2, Section 4). However, the coordinate-to-color mapping $c_i = \lfloor 255 o_i \rfloor$ encodes spatial positions as RGB values, which is structurally disconnected from the semantic and textural content DINOv2 was pretrained on. The paper provides no feature activation analysis or ablation comparing against a version without the 2D encoder, leaving it unclear whether the pretrained features contribute meaningfully or whether the network simply learns the mapping from scratch during retargeting. The claim of first application to 3D generation is also imprecise—coordinate-based color encodings have been used in neural fields and volumetric rendering.

### Minor

- **Missing ablation isolating the contribution of retargeting** — The paper does not evaluate the base generative model's zero-shot prior quality without fine-tuning, nor does it compare against a standard 3D shape completion baseline that skips the 2D mapping entirely. Without these, the improvements in Table 1 could stem from the underlying generative model rather than the specific retargeting strategy. The paper mentions the approach in Section 5 but provides no training hyperparameters (steps, learning rate, regularization), making it difficult to assess the significance of the fine-tuning contribution.

- **Limited demonstration of practical prior integration** — The Table 2 right experiment uses closest-point matching with ground truth positions, which is not feasible in a real unsupervised setting. While the paper correctly identifies this as an area for future work, the absence of even a baseline demonstration of a deployable pipeline (e.g., using the prior as a soft constraint or guiding loss in an actual assembly algorithm) weakens the practical impact claim.

### Trivial

- The caption for Figure 4 describes the latent of "low likelihood" as $x_0 \rightarrow x_0$, which appears to be a typo for the perturbed latent.
- Minor notation inconsistency: the paper uses both $X_t$ and $Z_t$ for the ODE trajectory in Section 4 (Eqs. 1-3 vs. the surrounding text).

## Nice-to-Haves

- A feature visualization showing how DINOv2/LEAP features respond to coordinate-mapped inputs would strengthen the claim that 2D priors are being leveraged.
- An ablation comparing the coordinate-to-color mapping against alternative encodings (e.g., standard point cloud encoders) would isolate the contribution of the 2D-3D bridge.
- A demonstration of non-oracle prior integration (e.g., using the generated prior as a geometric constraint during pose optimization) would make the assembly improvement claim more concrete.

## Removed Points

- The harsh critic's claim about "metric conflation" is substantially correct (see Major weakness above) but the framing that this "invalidates the central claim" overstates the case. The shape prior quality itself is a valid contribution even if the assembly integration is incomplete. The claim is weakened but not invalidated.

- The harsh critic's claim about "severe domain mismatch" in the 2D-3D mapping is valid as a concern about whether pretrained features are used, but the claim that this "forces the network to relearn from scratch" is speculative without ablation evidence. Weakened to Major concern about unclear feature utility rather than fundamental architectural flaw.

- The harsh critic's request for training hyperparameters is a reproducibility concern but falls under the category of missing implementation details that could be addressed in rebuttal. Moved to Minor rather than treated as fatal.

- The harsh critic's claim that the abstract's "category-agnostic" claim is "directly contradicted" by limitations is overstated—the limitations section honestly bounds the scope rather than contradicting the claim. The paper does work across categories on the tested datasets; failure on extreme cases is normal.

- The strength finder's claim that "Jigsaw++ is designed as a plug-in improvement over any existing assembly algorithm" is overstated—the paper demonstrates this only in a limited way with oracle matching. The strength is kept but rephrased to emphasize shape reconstruction improvement rather than assembly orthogonality.

## Novel Insights

None beyond the paper's own contributions. The core idea of generating complete shape priors to guide reassembly is straightforward and well-executed within its limitations, but the review synthesis does not reveal additional novel insights not already present in the paper.

## Suggestions

- Reframe the evaluation to more directly measure the method's contribution: add a comparison showing how a standard 3D completion model performs on the same task, and include at least one non-oracle demonstration of using the prior to guide assembly.
- Include an ablation removing the retargeting stage to show the base model's capabilities, and an ablation comparing with and without the 2D encoder to validate the pretrained feature contribution.

## Score and Decision

The paper presents a clear and interesting contribution—generating complete shape priors for 3D reassembly—with solid empirical results on shape reconstruction quality. However, the central claim about improving assembly methods is not fully validated by the experiments shown, and the use of pretrained 2D features is not convincingly demonstrated. 

Compared to calibration anchors:
- **High-scoring papers** (e.g., UniRestore3D, scores 8/8/6/5, accept) had clear validation across multiple tasks with strong experimental grounding; this paper's assembly claims are less directly validated.
- **Borderline papers** (e.g., Diff-Shape scores 1/5/3/5 reject; what does SD know scores 5/5/3/5 reject) had significant evaluation concerns similar to this paper's assembly validation gap. Papers with scores 5-6 typically had clear core contributions but incomplete downstream validation.
- **Strong accepted papers** like Atlas Gaussians Diffusion (6/8/8/8) had thorough ablation and well-validated claims.

This paper's strengths (novel shape prior generation, clear problem formulation, honest limitations) are balanced against real but not fatal weaknesses (incomplete assembly validation, unclear 2D feature utility). It fits the borderline range where the core contribution is meaningful but the claims exceed the experimental validation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>