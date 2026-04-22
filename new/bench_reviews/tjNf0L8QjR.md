Now I have a solid understanding. Let me synthesize my final review.

## Summary

This paper investigates whether the inductive bias of locality—neighboring pixels being more related than distant ones—is necessary for vision Transformers. By treating individual pixels (1×1 patches) as tokens instead of grouped patches (e.g., 2×2 or 16×16), the authors remove locality from patchification and use learned (not spatially-structured) position embeddings. Across three tasks—supervised classification, self-supervised MAE pre-training, and DiT-based image generation—the pixel-level models match or outperform patch-based baselines of the same parameter count, leading to the conclusion that locality is not a necessary inductive bias for vision.

## Strengths

- **Well-defined, important research question with ambitious scope**: The paper asks a precise question about the role of locality as an inductive bias and tests it across three distinct paradigms—supervised learning (Table 3 across four datasets including CIFAR-100, ImageNet, Oxford-102-Flowers, NYU-v2), self-supervised MAE (Table 4), and DiT-based image generation (Table 5)—spanning discriminative and generative tasks with different architectures. On CIFAR-100, ViT-S/1 achieves 86.4 vs. 83.7 Acc@1 for ViT-S/2; on ImageNet at 64×64, ViT-S/1 achieves 74.1 vs. 72.9.

- **Insightful "two trends" analysis (Figure 2)**: The paper identifies that the interplay between patch size, sequence length, and input size produces opposite trends depending on experimental protocol—fixed sequence length (Fig. 2a) makes pixel-level worst, while fixed input size (Fig. 2b) makes it best. This is a useful characterization of why prior work following standard protocols would miss the pixel-as-token benefit.

- **Systematic dissection of locality sources in ViT (Section 5)**: The pixel permutation experiment with distance thresholds (Figure 5) decouples the two sources of locality—position embeddings and patchification—and demonstrates that patchification carries far more locality prior (removing PE costs only 1.6% Acc@1 vs. 25.2% from full pixel permutation). The graded permutation design with Hamming distance thresholds δ provides nuanced evidence about locality structure.

- **Honest framing about limitations**: The paper explicitly states it introduces no new method, acknowledges the quadratic computational cost, and notes that "patchification is still a simple and effective idea that trades quality for efficiency, and locality is still highly *useful*" (Section 3). Table 1 provides a clear taxonomy of inductive biases across ConvNet, ViT, and the proposed approach.

## Weaknesses

### Fatal

None. While the central claim is overstated (see Major), the empirical findings are real and valuable—the paper does document a genuine phenomenon.

### Major

- **The central claim that "locality is not a necessary inductive bias" is confounded by simultaneous increases in sequence length and compute**: Every comparison between pixel Transformers (patch size 1) and patch Transformers (patch size 2) simultaneously removes locality AND quadruples the sequence length, yielding ~16× more self-attention FLOPs. The paper's own "two trends" analysis partially acknowledges this: Figure 2a (fixed sequence length) shows the locality-free model is *worst*; Figure 2b (fixed input size) shows it is best—but Figure 2b does not control for compute. The paper concludes "resolution is the enabler for ViT, not locality" (Section 6), but a more parsimonious reading of the evidence is that the benefit of finer tokenization (more tokens, more compute) outweighs the cost of losing locality—a weaker and less surprising statement. No experiment isolates locality from sequence length/compute (e.g., comparing a pixel Transformer with global attention vs. a pixel Transformer with local windowed attention at identical FLOPs). This does not invalidate the empirical findings, but the strong causal claim about locality being unnecessary is not justified by the experimental design.

- **The pixel permutation experiment (Section 5) directly contradicts the central claim, and the attempted reconciliation is unverified**: Permuting pixels within patches drops accuracy by 25.2% (Figure 5, T=25K), demonstrating that locality in patchification is *critically important* for ViT. The paper hypothesizes this is because permutation also breaks "location equivariance" (weight sharing), but this hypothesis is untested. A simple control—permuting pixels while correspondingly permuting position embeddings and/or the linear projection weights—would disentangle lost locality from lost weight-sharing. Without this control, the paper simultaneously claims locality is unnecessary (Section 4) and that locality destruction is devastating (Section 5), creating an internal tension that is hypothesized away rather than resolved.

### Minor

- **No compute-controlled or training-budget-controlled comparison**: Across all experiments, pixel models have identical parameter counts but vastly higher FLOPs than patch baselines. A fairer comparison would train patch models with proportionally larger hidden dimensions, more training steps, or compare pixel models equipped with sparse/local attention at matched FLOPs. The paper acknowledges the quadratic cost but does not attempt any compute-equivalent comparison, which makes the performance gaps in Tables 3–5 difficult to interpret in terms of the locality claim specifically.

- **The DiT generation experiments operate on VQGAN latent tokens, not raw pixels**: VQGAN encoders are trained with convolutional (locality-biased) architectures, so the "locality-free" claim is diluted—the model's input already encodes strong locality priors from the tokenizer. The paper should acknowledge this caveat more prominently when claiming the finding generalizes to image generation.

- **No standard deviations or multiple runs reported**: Key results like the Oxford-102-Flowers comparison (46.3 vs. 45.8 Acc@1) and DiT FID comparison (4.05 vs. 4.16) have small gaps that could fall within run-to-run variance. Reporting variance would strengthen the reliability of these comparisons.

- **Permutation equivariance claim is misleading (Section 3)**: The paper states the pixel Transformer is "permutation equivariant at the pixel level," but with learned position embeddings, it is *not* permutation equivariant—position embeddings break this equivariance. The claim should be clarified to state that the *architecture* (self-attention + MLP) is permutation equivariant, but the full model with position embeddings is not.

### Trivial

- The claim that this work "puts a conclusive remark" (Section 6) on the iGPT vs. ViT debate is overly strong given the confounds in the experimental design.

## Nice-to-Haves

- A compute-controlled ablation (e.g., pixel Transformer with windowed/local attention vs. global attention, at matched FLOPs) would directly isolate whether locality *in the architecture* matters, independently of sequence length.
- Visualizing learned position embeddings for pixel models to check whether they rediscover 2D spatial structure would inform whether the model is effectively re-learning locality from data.
- Testing with sub-quadratic attention mechanisms (e.g., linear attention) would make compute-controlled comparisons feasible at higher resolutions.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that "no experiment in the paper disentangles [locality from sequence length], so the paper cannot substantiate its core claim"**: This overstates the issue. The Figure 2a analysis (fixed sequence length, varying input size) *does* partially isolate the factors—showing that when sequence length is held constant, removing locality (smaller input) hurts. The paper's two-trend analysis is a real (if incomplete) attempt at disentanglement. The issue is that the *causal claim* is too strong for the evidence, not that the paper provides zero relevant evidence. Downgraded from Fatal to Major.

- **Harsh critic's claim that the paper is "misleading" in the abstract for not mentioning compute cost**: The abstract does state "it's computationally less practical to directly operate on individual pixels." The information is present, just not quantified. This is a presentation nitpick, not a substantive error.

- **Strength Finder's claim about permutation equivariance as a "hypothesis about location equivariance preservation" being an insightful strength**: While the hypothesis is interesting, it is unverified, so calling it a "strength" is premature. Moved to Minor weakness (untested hypothesis).

- **Harsh critic's claim that the iGPT discussion is "speculation without evidence"**: The paper's interpretation that iGPT fell short due to resolution rather than locality is explicitly supported by its own Figure 2a analysis showing resolution matters. This is a reasonable (if not conclusive) inference from the paper's own data.

- **Harsh critic's demand for missing related works**: Per instructions, criticism about missing related works is removed.

## Novel Insights

The most interesting insight emerging from the review is the tension between Sections 4 and 5 of the paper: the pixel-as-token experiments argue locality is unnecessary, while the permutation experiments show locality destruction is devastating. The paper's own hypothesis—that the difference is explained by location equivariance (weight sharing) being preserved in one case but not the other—is both the key to reconciling these findings and the most important avenue for future work. If true, the proper conclusion would not be "locality is unnecessary" but rather "locality is unnecessary *provided weight sharing across spatial locations is maintained*," which significantly narrows and qualifies the finding.

## Suggestions

- Reframe the central claim from "locality is not a necessary inductive bias" to "locality is not a necessary inductive bias when sufficient compute and resolution are available and weight sharing is maintained"—this is a more precise and defensible statement that still challenges conventional wisdom.
- Run the permutation + position-embedding control experiment to resolve the internal tension between Sections 4 and 5.
- Add compute-controlled comparisons (e.g., ViT-S/2 trained for 4× more epochs, or with wider hidden dim to match FLOPs) to separate the effects of locality removal from compute scaling.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Vision Transformers Need Registers (oral) | /human_reviews/2dnO3LLiJ1.md | 8.0 | Much stronger: clean, well-controlled finding with simple solution; no confound concerns. Our paper has a valuable but confounded finding. |
| On the Role of Discrete Tokenization (spotlight) | /human_reviews/WNLAkjUm19.md | 7.0 | Stronger: theoretical grounding + empirical findings on tokenization. Our paper is more empirical and has the overclaim issue. |
| A Spitting Image: Superpixel Transformers | /human_reviews/Vy6sjPt2Vr.md | 5.50 | Similar: alternative tokenization study with partial evidence. Our paper has a clearer and more surprising message but also a confounded claim. |
| On the Long Range Abilities of Transformers | /human_reviews/lnffMykYSj.md | 4.50 | Similar: studies inductive biases in attention, makes overclaimed causal conclusions from empirical evidence. Our paper is comparable but with more complete experiments. |
| Joint Training Does Not Transfer (withdrawn) | /human_reviews/qdJ1jJzyVP.md | 2.6 | Much weaker: fundamentally confounded claims, rebuttal paper. Our paper has real empirical findings even if the causal claim is overstrong. |
| Texture Patterns in TxTN | /human_reviews/MrOefpTvev.md | 2.33 | Much weaker: poorly described method, unclear experiments. Our paper is far more rigorous. |

The paper sits between the medium-scoring tokenization/inductive-bias papers (4.5–5.5) and the well-controlled high-scoring papers (7.0–8.0). The empirical finding is genuinely surprising and valuable, and the two-trend analysis + permutation experiments provide real insight. However, the central causal claim is confounded (Major weakness), and the permutation experiment creates an internal contradiction that is hypothesized away rather than resolved. The paper is stronger than the typical rejected inductive-bias studies (4–5 range) because the finding is more surprising and the experimental scope is broader, but weaker than the accepted papers (7+) because the core claim outpaces the evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>