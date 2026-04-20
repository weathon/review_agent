## Summary

This paper presents an empirical finding that vanilla Transformers can operate directly on individual pixels (1×1 "patches") without the locality inductive bias traditionally assumed necessary for vision, achieving competitive or slightly better results across supervised classification, self-supervised pre-training, and diffusion-based generation. The most valuable contribution is the "two trends" analysis (Figure 2), which shows that at fixed input size, reducing patch size monotonically improves accuracy — extending to the locality-free extreme — while at fixed sequence length, the locality-free case fails due to insufficient input information. The paper is explicitly framed as a finding rather than a practical method, which shapes how it should be evaluated.

## Strengths

- **Provocative, foundational research question with clear experimental validation.** The paper challenges a widely accepted assumption in vision architecture design — that locality is fundamental — and supports its claim with consistent results across three distinct paradigms (supervised, self-supervised, generation) and four datasets (Table 3, 4, 5). That the pattern holds across discriminative and generative tasks makes it difficult to dismiss as a narrow artifact.

- **The two-trends analysis (Figure 2) is pedagogically valuable.** By disentangling the confounding relationship between sequence length, input size, and patch size, the paper explains why the locality-free case was previously overlooked: at fixed sequence length it performs worst (63.4% at p=1), while at fixed input size it performs best (81.8% at p=1). This is a genuinely useful contribution for practitioners who design ViT scaling experiments.

- **Transparent framing and intellectual honesty.** The paper explicitly states that its contribution is a finding, not a practical method (Section 1, Section 7), and honestly acknowledges that patchification is a useful efficiency-quality tradeoff. This prevents the common sin of overstating applicability.

- **Controlled comparison at the same model size.** Tables 3–5 compare /1 and /2 variants with identical depth, hidden dimension, and head count (Table 2), isolating the tokenization difference. The scaling behavior on ImageNet (Table 3b) is encouraging: the /1 vs /2 gap grows from +0.4% at ViT-S/B to +1.3% at ViT-L, suggesting the benefit compounds with model capacity.

## Weaknesses

### Major

- **The /1 vs /2 comparison confounds locality removal with increased per-sample compute.** While the paper matches model parameters between variants, the attention computation scales quadratically with sequence length. On CIFAR-100 (32×32), the /1 variant processes 1024 tokens vs 256 for /2 — a 4× increase in attention FLOPs per forward pass. On ImageNet at 64×64, it is a 16× increase. The modest accuracy gains (+0.4–1.3% on ImageNet) could reasonably be attributed to the higher attention budget and finer token granularity rather than the removal of locality per se. Figure 2b partially addresses this by showing the monotonic improvement curve, but the end-to-end comparison between /1 and /2 in Tables 3–5 still mixes granularity benefits with the locality-free claim. This weakens the paper's headline assertion that locality is unnecessary rather than merely suboptimal compared to fine tokenization.

- **Marginal improvements are presented without statistical rigor.** The reported gains are small (ImageNet +0.4% for ViT-S and ViT-B, NYU-v2 RMSE −0.08, DiT FID −0.11) and reported without multiple random seeds or variance estimates. The pattern's consistency across multiple tasks is reassuring, but the ImageNet +0.4% deltas for ViT-S/B are within typical training-noise range for such benchmarks. A reader cannot assess whether these are real effects or variance artifacts.

### Minor

- **The locality-removal vs. higher-resolution-abstraction distinction is not cleanly separated.** The pixel-as-token approach simultaneously removes locality bias and provides the finest possible token granularity. A model using overlapping patches or local-window attention at the same token resolution as /1 would test whether locality is truly unnecessary or just an approximation to fine-grained tokenization. Without such a control, the paper demonstrates that fine tokenization works without locality priors — which is valuable, but not quite the same as proving locality is unnecessary. The permutation experiment (Section 5) shows patchification matters more than position embeddings, but it is a destructive experiment (shuffling pixels) rather than a constructive comparison to locality-aware fine-grained tokenization.

### Trivial

- None beyond cosmetic presentation issues.

## Nice-to-Haves

- Attention map visualization across layers for /1 vs /2 (as requested by one reviewer) would be an attractive addition to show how spatial structure is learned organically without locality priors, but its absence does not weaken the core finding.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"The evaluation protocol conflates locality removal with uncontrolled sequence length and compute scaling, invalidating the headline comparisons" (Critical Issue 1, Harsh Critic):** While the compute disparity is real, the paper explicitly acknowledges and discusses this (Figure 2, Section 1, Section 7). The /1 vs /2 comparison is one data point within the broader two-trends analysis, not a standalone claim. The criticism is valid but overstates its severity — it does not "invalidate" comparisons, it contextualizes them.

- **"The evidence only demonstrates that higher-resolution tokenization helps, not that locality itself is an unnecessary inductive bias" (Critical Issue 2, Harsh Critic):** The paper's conclusion is that locality is *not fundamental* (not that finer tokens are always better). The fact that a purely pixel-based model works at all — and even trends better — is the evidence. If locality were truly necessary, the locality-free model would fail. The paper already addresses the entanglement of locality and resolution through the two-trends framework.

- **"The central claim remains empirically unconvincing" (end of Critical Issue 2, Harsh Critic):** Overstated given the consistency of results across four datasets, three paradigms, and multiple model sizes. The claim is a finding, not a method proposal.

- **"The pixel permutation experiment provides no insight into why the learnable locality-free variant succeeds" (Critical Issue 3, Harsh Critic):** The permutation experiment's purpose is to compare the *destructive* removal of locality (permutation) vs the *constructive* removal (pixel-as-token). The paper uses it to argue that permutation also removes location equivariance (weight sharing), explaining its destructive effect. The harsh reviewer misunderstands the experiment's role.

- **"The vocabulary size argument in Section 3 is conceptually misplaced for standard ViTs which project continuous RGB values linearly" (Section-by-Section, Harsh Critic):** The paper's vocabulary size claim (256³ vs 256^(3·p·p)) is indeed about discrete tokenization schemes, not linear projections. However, this is a minor rhetorical aside in Section 3, not a core argument. The paper correctly uses linear projections in all experiments.

- **"The ImageNet experiments never explicitly state the input crop size for the /1 variant" (Section 4.1, Harsh Critic):** The paper states input is operated at "a much lower resolution" and Figure 2b uses 64×64. This is described, albeit implicitly in the main text with details in the appendix. Not a substantive gap.

- **"The DiT experiment operates on VQGAN latents, not raw pixels, diluting the framing" (Section 4.3, Harsh Critic):** The paper is transparent about this (Section 4.3, first paragraph). It extends the finding to a different representation, not a pure pixel experiment, which is a strength, not a weakness.

- **All formatting/typo/whitespace criticisms:** Parser artifacts, not author problems.

## Novel Insights

The paper's most distinctive contribution is not merely showing that pixel-level ViTs work — this has been attempted before (e.g., iGPT, Perceiver) — but rather the systematic decomposition of why prior attempts failed: the confounding relationship between patch size, input size, and sequence length, formalized as L = HW/p². The two-trends analysis (Figure 2a vs 2b) is a clean pedagogical tool that will likely be cited in future ViT-scaling literature. The finding that the accuracy-patch-size curve remains monotonic even at the locality-free extreme p=1, combined with the observation that removing position embeddings only costs 1.5% while patchification destruction costs up to 25.2%, quantifies the relative importance of these two locality mechanisms for the first time.

## Suggestions

- If space permits in a revision, add results for a locality-aware fine-grained baseline (e.g., overlapping patches with the same token resolution as /1) to fully disentangle the "locality unnecessary" vs "finer tokens better" interpretations.
- Report results over 3 random seeds for the marginal ImageNet gains (+0.4%) to rule out training noise.
- Include the attention map visualizations mentioned in the appendix for the /1 vs /2 comparison — these would strengthen the qualitative evidence without requiring new experiments.

## Score and Decision

**Calibration:** I compared this paper against several anchors:
- **Vy6sjPt2Vr.md** (Superpixel Transformer, scores 6,8,5,3, rejected): A paper proposing a novel tokenization with a similar "what's really driving the improvement" ambiguity; received mixed scores with two reviewers below acceptance.
- **3OyaXFQuDl.md** (LLM compute-optimal sampling, scores 6,6,8,8, accepted poster): A strong empirical finding paper with confounded comparisons that scored well due to consistent multi-experiment evidence and transparent framing.
- **RtDok9eS3s.md** (Simplifying Transformer Blocks, scores 8,8,6, accepted poster): An ablation paper with strong results and clean experimental design.
- Several analysis papers challenging assumptions scored 5-6 due to conceptual gaps (0sJ8TqOLGS, tZk3LnvVtK).

This paper sits between the Vy6sjPt2Vr and 3OyaXFQuDl anchors. Like Vy6sjPt2Vr, it has a compelling core finding but the causal mechanism (locality removal vs finer tokenization) is not fully disentangled. Like 3OyaXFQuDl, it presents consistent multi-paradigm evidence and is transparent about being a finding rather than a method. Unlike lower-scoring analysis papers (5-6), it does not have a fundamental conceptual gap — the findings are real, even if the causal interpretation is partially ambiguous. The two-trends analysis adds pedagogical value beyond the raw numbers, pushing it above borderline anchors. However, the marginal size of improvements in key cases (ImageNet +0.4%) and the absence of statistical reporting prevent it from reaching 7-8 territory.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>