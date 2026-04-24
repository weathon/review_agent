Now let me do calibration searches to score the paper properly.Now I have enough information to write the final review.

---

## Summary

This paper presents an empirical finding rather than a new method: vanilla Transformers can operate by treating each individual pixel as a token (1×1 "patch") and still achieve competitive or superior performance compared to standard patch-based ViTs. The paper tests this across supervised classification, self-supervised learning (MAE), and image generation (DiT), and separately ablates the two locality-encoding mechanisms in standard ViT (position embeddings vs. patchification). The finding challenges the assumption that locality is a necessary inductive bias for vision architectures.

---

## Strengths

- **Figure 2's two-trend analysis is a genuine conceptual contribution** (Sec. 4.1). By explicitly separating the "fixed sequence length" trend (Fig. 2a, where locality-free is worst) from the "fixed input size" trend (Fig. 2b, where locality-free is best), the paper provides an intellectually honest explanation of why pixel-level ViTs were not discovered earlier, and resolves an apparent contradiction in the literature. This is original and insightful.

- **Section 5's patchification ablation is clean and novel.** The pixel permutation design — corrupting locality in patchification while keeping weight-sharing intact — cleanly isolates patchification from position embeddings. Key result: removing PE entirely costs only −1.6% accuracy, while permuting nearly all pixel pairs (T=25K) costs −25.2%. This is a causal, interpretable finding with clear gradient (T, δ parameters).

- **Breadth of coverage strengthens the core finding.** The result holds across supervised classification on CIFAR-100 and ImageNet (Table 3a–b), fine-grained classification on Oxford-102-Flowers (Table 3c), depth estimation on NYU-v2 (Table 3d), MAE self-supervised pre-training (Table 4), and DiT-based generation (Table 5). Cross-task consistency makes the finding harder to dismiss as a single-domain artifact.

- **Depth estimation extends the finding to spatial tasks** (Table 3d). ViT-S/1 achieves RMSE 0.72 vs. ViT-S/2's 0.80 on NYU-v2, a task that explicitly requires spatial structure — making the case for locality-free architectures stronger than purely on classification.

- **Paper is honest about its scope and limitations.** The abstract explicitly says "this work does not introduce a new method," acknowledges the computational impracticality of pixel-level training, and frames the contribution as a "finding" to inform future architecture design. This intellectual honesty is appropriate and rare.

---

## Weaknesses

### Fatal
None.

### Major

- **The main experimental comparisons conflate locality removal with sequence-length increase.** In Tables 3a–d and Table 4, ViT/1 is compared against ViT/2 at the *same input size*, which automatically gives ViT/1 4× more tokens (e.g., 1,024 vs. 256 tokens on 32×32 CIFAR-100; 4,096 vs. 1,024 on 64×64 ImageNet). The paper's own Figure 2b — the only controlled experiment fixing input size across patch sizes — shows the true effect of locality removal at matched input size is a modest +0.8% (81.0% → 81.8%, ViT-S on 64×64 ImageNet). The paper explicitly cites Beyer et al. (2022) and Hu et al. (2022) showing that longer sequences benefit ViT, so the paper is aware of this confound. However, the headline numbers (+1.5% to +2.7% on CIFAR-100, Table 3a) come from comparisons that simultaneously change both locality *and* token count. The paper frames these as evidence that "locality is not necessary," but they are also consistent with "more tokens help." The discussion of Figure 2b (the clean controlled experiment) is present but somewhat buried. This confound meaningfully weakens the strength of the headline claim, and Figure 2b's +0.8% gap should be placed more centrally in the paper's argument.

- **ImageNet experiments are conducted at non-standard 64×64 resolution**, far below the community benchmark of 224×224 (where SOTA ViTs achieve 80%+). The paper's ImageNet numbers peak at 76.9% (ViT-L/1, Table 3b), which the paper itself acknowledges is below the state of the art. Low resolution may disproportionately benefit ViT/1 because 64×64 images carry limited local structure, making patchification less useful. The paper does not quantify this resolution effect, and the core claim cannot be verified at the standard evaluation setting where practitioners would care about it.

### Minor

- **The DiT case study is in VQGAN latent space, not raw pixel space.** The paper says in Section 4.3: "the input space is also changed from raw pixels to latent tokens" — it operates on 32×32×4 VQGAN feature maps. The latent codes are 8× downsampled from the original 256×256 image, and the VQGAN encoder that produced them is itself locality-biased. DiT-L/1 treats each 1×1 latent vector as a token (vs. DiT-L/2's 2×2 grouping). While the paper is transparent about this, calling this "pixels as tokens" in the abstract and framing it as a generalization result for "image generation" is slightly misleading — the finding is strictly about latent token granularity, not raw pixel generation. The strength of this generalization should be presented more carefully.

- **The strong DiT-L/2 baseline (8.90 FID no-guidance vs. reference DiT-XL/2 at 10.67 FID) is unexplained.** This large gap makes interpreting the DiT/1 vs. DiT/2 comparison harder: the training recipe improvement responsible for the 8.90 baseline may interact differently with /1 and /2, confounding the DiT comparison.

- **No statistical significance is reported for any result.** For some of the smallest gaps — e.g., ViT-T/1 + MAE (86.0%) vs. ViT-T/2 + MAE (85.7%) in Table 4, or DiT-L/1 FID (4.05) vs. DiT-L/2 (4.16) in Table 5 — it is unclear whether differences exceed noise. Single-run evaluation is standard in the field, but reporting at least the variance across 2–3 seeds for the central claim would strengthen the paper.

### Trivial

- Figure 2b shows ViT-S while Figure 2a uses ViT-B — acknowledging this model size inconsistency in the caption would improve clarity.

---

## Nice-to-Haves

- A **sequence-length-matched comparison** (e.g., ViT-S/1 on 32×32 vs. ViT-S/2 on 64×64, both producing ~1K tokens) would directly test whether the advantage of /1 is from locality removal or token count increase. This would make the central claim unambiguous.
- A small-scale **raw pixel generative model** (e.g., 64×64 DiT on actual pixels rather than latents) would more directly support the "pixels as tokens for generation" framing.
- Visualizations of **attention maps at matched token counts** for /1 vs. /2 would provide direct evidence that ViT/1 uses genuinely non-local attention patterns, rather than just benefiting from higher resolution.
- **Standard 224×224 ImageNet experiment** (even with FlashAttention for efficiency, or on a subset of training steps) would validate whether the finding holds at the community benchmark resolution.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Missing related works**: The harsh critic alluded to prior work on pixel-level Transformers. Per the hard rules, we do not claim missing related works.
- **Reproducibility concerns / hyperparameter disclosure**: Details deferred to the Appendix (which was stripped by the parser) are not a real gap.
- **Strength: "Fair and controlled experimental methodology"** (Strength Finder): Removed because it conflicts with the verified major weakness about sequence-length confound — the comparisons are not fully controlled.
- **Generic strength about importance of the research question**: Removed as non-specific.

---

## Novel Insights

The most genuinely novel insight in this paper is the *two-trend decomposition* (Figure 2): prior work always varied patch size under fixed sequence length (as is standard in NLP), which made locality-free architectures look worst. By instead fixing *input size* — a natural regime for vision — the trend reverses: decreasing patch size to the locality-free limit is monotonically beneficial. This resolves a decade-long assumption and clearly explains why pixel-level ViTs were never discovered by the community's standard evaluation protocol. Additionally, the pixel-permutation ablation's key finding — that patchification contributes far more locality bias than position embeddings (−25.2% vs. −1.6%) — is a precise, causally interpretable result that is useful independent of the main claim.

---

## Suggestions

1. **Restructure the argument around Figure 2b** rather than Table 3. If the key controlled experiment (fixed input size) gives +0.8%, present this upfront as the conservative evidence and frame Table 3's larger gains as showing practical benefit when longer sequences are allowed. This would eliminate the overstatement risk.
2. **Add the sequence-length-matched ablation** (ViT-S/1 on 32×32 vs. ViT-S/2 on 64×64) as a key table — it directly answers the main confound and would substantially strengthen the paper.
3. **De-scope the DiT case study** or rename it: "latent token granularity in DiT" is a more accurate and defensible description than "pixels as tokens for generation."
4. **Report multi-seed variance** on the central CIFAR-100 and DiT comparisons.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to paper under review |
|---|---|---|---|
| LVSM (minimal 3D bias, Oral) | `QQBPWtvtcn.md` | 7.67 | Same "remove inductive bias" framing but proposes a new method achieving SotA; stronger in scope and practical impact |
| Vision Transformers Need Registers (Oral) | `2dnO3LLiJ1.md` | 8.00 | Similar "finding+analysis" paper from ViT community (also FAIR), cleaner experiments, practical fix proposed |
| ChannelViT | `CK5Hfb5hBG.md` | 6.50 | ViT tokenization modification with multi-task empirical validation; similar empirical scope, proposes actual method |
| Label-Focused Inductive Bias | `cH3oufN8Pl.md` | 6.67 | Inductive bias in vision classification, accepted poster; comparable scope |
| Structured Initialization for ViT (Reject) | `z9UBpl4pv5.md` | 4.33 | Also about ViT inductive biases, much weaker method and analysis; clearly below this paper |
| Long-Range Abilities of Transformers (Reject) | `lnffMykYSj.md` | 4.50 | Inductive bias for transformers, weaker empirical support; below this paper |

**Reasoning:** The paper under review is a genuine finding paper from FAIR/Meta with multi-task coverage, a clean ablation section, and notable intellectual honesty. Its main weakness — the sequence-length confound in headline experiments — is real and partially addressed within the paper itself (Figure 2b), which is both a limitation and a sign of good faith. It is clearly above the 4.33–4.50 range of the rejected papers. Compared to ChannelViT (6.5), it has comparable empirical breadth and similar task structure, but lacks a practical method and has a more prominent methodological confound. Compared to ViT Registers (8.0) and LVSM (7.67), it is less rigorous and provides a weaker practical contribution. I place it between ChannelViT and the Oral acceptances, at **6.5**. The finding is real, the paper is honest, and the ablation section is a genuine contribution — but the confound in the main experiments and non-standard evaluation settings prevent a higher score.

**Score: 6.5 — Accept (Poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>