=== CALIBRATION EXAMPLE 33 ===

# Final Consolidated Review
## Summary
This paper studies image compression specifically for cloud-hosted MLLM inference, proposing a lightweight transform-neck that maps compressed image latents directly into an intermediate layer of the MLLM visual encoder, plus a surrogate loss that avoids backpropagation through the full MLLM. The method is evaluated in three deployment scenarios—fixed human-perception codec, joint human/machine optimization, and machine-only optimization—and shows clear gains over feeding reconstructed compressed images to MLLMs, while also reducing inference-side computation by avoiding full image reconstruction.

## Strengths
- **The paper identifies and cleanly formulates a genuinely new systems problem:** compression tailored to downstream MLLMs rather than conventional perceptual reconstruction or classical task-specific vision models. The distinction is explicit in Figure 1 and Sections 1–2, and the motivation is convincing: existing coding-for-machines pipelines typically assume training through the downstream task model, which is much less practical when the receiver is a billion-parameter MLLM.
- **The core method is technically coherent and practically well matched to the problem.** The transform-neck bridges codec latents to an intermediate visual-encoder layer (Section 3.3), and the surrogate loss in Section 3.4 is specifically designed to avoid backpropagation through the LLM while still shaping features for multimodal use.
- **The paper does more than propose a single operating point; it offers a useful deployment spectrum.** The three scenarios ((d1) fixed perceptual codec, (d2) joint human/machine optimization, (d3) machine-only optimization) are a concrete and practically relevant design contribution, not just a presentation device.
- **Empirical gains over the most direct baseline are substantial and consistent across tasks.** In Figure 3, using reconstructed compressed images causes large drops, while the proposed method recovers much of the lost downstream performance across captioning, VQA, REC, and few-shot classification. The d2/d3 settings further improve over d1, supporting the claim that codec retraining can make latents more MLLM-friendly.
- **The complexity argument is one of the paper’s strongest pieces of evidence.** Table 3 shows a large reduction relative to the image-domain post-processing alternative because the method operates in latent space and skips decoding plus early visual processing. This directly supports the practical appeal of the approach rather than being a secondary convenience.
- **The ablations are meaningful and tied to the paper’s actual design choices.** Figure 6 and Figure 7 provide evidence that the CE and distillation terms play complementary roles, and that the selected insertion depth in the visual encoder is not arbitrary.
- **The paper does provide some evidence beyond a single encoder/codec/task setup.** It evaluates multiple downstream tasks and MLLMs, includes an additional codec in ablation (TIC), and shows non-CLIP-ViT examples in Section 4.6. This does not fully prove broad universality, but it does show the method is not a one-off tuned result.

## Weaknesses

### Fatal
- None.

### Major:
- **The paper overstates the breadth of its generality claims relative to what is actually demonstrated.** The abstract and introduction repeatedly claim broad applicability to “various MLLMs, neural image codecs, and multiple application scenarios,” and even suggest readiness “regardless of their architectures.” The evidence is meaningfully positive but narrower: core training is centered on ImageNet and primarily on CLIP ViT-L/14 features (Section 4.1: “we use the CLIP visual encoder as \(C\) for training and conduct our primary experiments on MLLMs that incorporate it”), with one extra codec ablation and two non-CLIP examples in Section 4.6. That supports promising transfer, but not as strong a universality claim as the paper currently makes.
- **The surrogate training objective is not truly task-agnostic in the strong sense implied by the framing.** Section 3.4 defines the CE term using a labeled classification dataset and class text labels, and Section 4.1 states the transform-necks are trained on ImageNet. This is a reasonable proxy objective, but it is still a specific semantic supervision setup rather than a broadly task-free adaptation method. Since the paper evaluates captioning, VQA, REC, and few-shot classification, it should be more explicit that the method relies on object/category supervision as a transfer signal rather than direct supervision from those downstream tasks.
- **Comparative evaluation is too narrow for some of the broader motivational claims.** Experimentally, the main baselines are reconstructed-image inference and reconstructed-image post-processing. Those are important and fair baselines, but the paper’s rhetoric in Sections 1–2 goes further, positioning the method against existing coding-for-machines paradigms more broadly and arguing they are impractical or inapplicable for MLLMs. The paper does not actually compare against a serious adapted feature-bridging or proxy-based alternative, so the evidence supports “better than straightforward reconstruction and image post-processing” more strongly than “the right practical route versus competing coding-for-machines strategies.”
- **The “universality” claim for reuse across MLLMs sharing a visual encoder is plausible but not directly isolated and validated.** The paper states that “we do not need to train separate systems for the different MLLMs or tasks” (Section 4.1) and claims the transform-neck is “readily applicable to multiple MLLMs that share the same visual encoder, without the need for retraining.” However, the evaluation does not cleanly center this as a controlled experiment—for example, training once and then systematically reusing the same transform-neck across multiple MLLMs that share the encoder, compared against per-MLLM retraining. The existing results suggest transfer, but do not fully nail down this specific claim.

### Minor
- **The rate–accuracy evidence in the main paper is somewhat sparse.** Figure 3 shows only a small number of rate points, while the headline claim includes “up to 60–80% bit-rate reductions under the same recognition accuracy.” The claim may well be supported in the appendix, but in the main paper the tradeoff curves are not dense enough to make that headline result feel fully established on their own.
- **The progressive training schedule appears empirical and somewhat ad hoc.** Equation (4) uses fixed epoch cutoffs \(E_1=20, E_2=40\) and a 1:100 weighting ratio between CE and distillation in the mixed phase. The ablation shows the combined strategy helps, but there is little sensitivity analysis on whether this schedule is robust or brittle.
- **For non-CLIP visual encoders, the role of the CE term is under-explained.** Section 3.4 says the CLIP text encoder is used “independently of the visual encoder integrated into the MLLM under consideration,” and Section 4.6 reports positive transfer to other visual encoders. This empirical result is useful, but the paper does not sufficiently explain why CLIP-text supervision should remain well aligned when the visual encoder family changes.
- **One of the showcased tasks is evaluated in a custom setup rather than an original published protocol.** Section 4.1 explains that the few-shot classification evaluation was designed by the authors because original code was unavailable. This is acceptable, but it does make that specific result less standardized than the others.
- **The complexity analysis is strong but selective.** Table 3 compares against the post-processing baseline and omits shared components, which is fair for incremental comparison, but the paper could better disentangle how much benefit comes specifically from avoiding image reconstruction versus avoiding the extra post-processing network.

### Trivial
- None.

## Nice-to-Haves
- Add denser rate–accuracy curves or a BD-rate style summary in the main paper.
- Include a direct controlled reuse experiment for one transform-neck across several MLLMs sharing the same visual encoder.
- Provide a small sensitivity study for the progressive schedule and loss weights.
- Add a stronger adapted coding-for-machines baseline, even if simplified, to support the broader comparative framing.
- Include some failure-case analysis showing when latent adaptation does not close the gap to uncompressed inputs.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper should compare to specific additional modern MLLMs or missing related works.** Removed because this depends on external expectations and missing-work speculation rather than verified shortcomings from the paper itself.
- **Criticism about lack of confidence intervals / variance reporting.** Moved out because this is not clearly a field-standard requirement for this style of large-scale benchmarking, and the absence does not undermine the paper’s core claims.
- **Reviewer comments about inaccessible or nonexistent code/models/benchmarks, or reproducibility concerns rooted in availability.** Removed by rule.
- **Several “transferable weaknesses” appended to Review 1 about region features, SAM, clustering, cross-attention aggregation, etc.** Removed because they clearly refer to a different paper and do not match the submitted manuscript.
- **The claim that universality across shared encoders is “trivially true.”** Softened/removed in that form: the paper’s reuse claim is not trivial in practice, but it is insufficiently isolated experimentally. The issue is overclaiming and incomplete validation, not vacuity.
- **Demand for an upper-bound experiment that backpropagates through the full MLLM.** This would be informative, but the paper’s stated contribution is precisely to avoid full-MLLM training for practicality reasons; requiring this as a core flaw would overstep the paper’s scope.

## Novel Insights
The most interesting synthesis across the reviews is that the paper is strongest when read as a **practical latent-interface method for MLLM-compatible compression**, not as a definitive general framework for all MLLMs. Its real contribution is the systems-level decomposition: treat the visual encoder as the training-time anchor, learn a latent-to-intermediate bridge there, and thereby decouple compression adaptation from the expensive language side of the model. The experiments do support this decomposition. What they do **not** yet fully establish is that the same recipe is broadly architecture-agnostic or task-agnostic in the strong sense suggested by the framing, because the supervision and primary training regime remain closely tied to CLIP/ImageNet semantics.

## Suggestions
- Reframe the contribution more precisely: emphasize that the method is **demonstrated primarily in a CLIP/ImageNet-trained regime with promising transfer beyond it**, rather than claiming broad architecture-agnostic universality.
- Add a direct experiment for the shared-visual-encoder reuse claim: train one transform-neck once and evaluate it across multiple MLLMs using that encoder, optionally comparing against per-MLLM retraining.
- Strengthen the empirical framing against alternatives by adding at least one adapted coding-for-machines baseline beyond reconstructed-image post-processing.
- Clarify the limitations of the CE surrogate: it uses labeled classification data and likely emphasizes object-centric semantics; stating this explicitly would make the paper more credible.
- Expand the main-paper rate–accuracy evidence with more operating points or a compact aggregate metric.
- Add a brief discussion of why CLIP-text supervision still helps when transferring to non-CLIP visual encoders, since Section 4.6 suggests this works but Section 3.4 does not explain it well.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
