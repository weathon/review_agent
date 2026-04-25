Now let me search for calibration papers.Now I have enough information to write the final review. Let me compile everything.

---

## Summary

The paper proposes ARM (Autoregressive Reward Model), a low-rank parametrization of the reward model used in reward-augmented decoding (RAD) for controlled language generation. The central insight is that RAD's learned reward matrix is empirically low-rank (≈ 10²), far below both the vocabulary size (|V| = 50,257) and the model dimension (d = 768), so a structurally constrained model can match RAD's quality at the cost of only one forward pass instead of k. The authors formalize this via a matrix-completion framing of the RAD training objective, propose a bilinear parametrization (Eq. 7) with a natural regularization (Eq. 11), and demonstrate competitive detoxification and sentiment control results alongside a 10× discriminator speedup over RAD.

---

## Strengths

- **Novel matrix-completion reformulation (§3.1.1, Eq. 5):** Recasting RAD's training as incomplete matrix completion is the paper's most technically original contribution. It provides a clean, principled lens for analyzing why RAD is over-parameterized and why low-rank reward models can suffice. This framing goes meaningfully beyond prior work.

- **Empirical rank analysis with direct evidence (Figure 1):** The rank of R̂_RAD is measured by computing N full rows via |V| forward calls each, then applying SVD. The result — that rank stays at ~10² regardless of N, far below d = 768 — directly motivates the ARM architecture and is a non-trivial empirical finding.

- **ARM architecture cleanly derived from analysis (Eq. 6–8):** The bilinear parametrization h(x)·W·e(v) follows directly from the low-rank observation, the rank bound rank(R̂_ARM) ≤ d is transparently stated, and the connection to the softmax bottleneck is appropriately acknowledged.

- **Regularization design validated by ablation (Eq. 11, Figure 5a–b):** Pushing Δr̂ toward 0 for random tokens provides a natural "abstaining" mechanism. Figure 5 confirms it both lowers ARM's rank and improves fluency — an ablation result that is substantively informative.

- **Efficient comparison (Table 1, Figure 6):** The per-token timing measurement on an RTX A6000 GPU clearly demonstrates ARM's constant ~0.001 s/token vs. RAD's linear scaling to ~0.010 s/token at top-k = 80. The efficiency claim is backed by real wall-clock numbers, not just theoretical FLOP counts.

- **Distillation finding is well-motivated (§5.4):** The explanation for why distilled ARM outperforms ARM-resp-only (teacher pre-compresses ambiguous multi-response labels into a single deterministic target) is a concrete and practically useful insight.

---

## Weaknesses

### Fatal
None.

### Major

- **ARM-distilled outperforming its teacher RAD in Figure 4 is not adequately explained.** If ARM minimizes squared distance to RAD's predictions (Eq. 10), a lower-capacity model cannot systematically exceed the teacher unless there is a favorable evaluation asymmetry (e.g., the DistilBERT classifier used for Positive Rate happens to align more with ARM's inductive bias) or if β hyperparameters are not matched across comparisons. The paper's conjecture in §5.4 — "distillation provides a single deterministic target" — explains why ARM-distilled might *match* ARM-resp-only, but does not explain why it *exceeds RAD*. This inconsistency is notable because it directly touches whether the low-rank inductive bias provides a genuine regularization benefit or whether there is a confound in the experimental setup. The authors should verify whether ARM-distilled's rank (Figure 5a shows rank ≈ 10–20) is lower than RAD's, and whether this correlates with the performance gap, or provide an alternative account.

### Minor

- **The primary evidence for "ARM matches RAD" relies predominantly on the distillation setting, which carries some circularity.** The ARM-distilled variant is trained to mimic RAD's outputs (Eq. 10). Demonstrating that a student matches its teacher is a weaker claim than demonstrating that the low-rank inductive bias is competitive in its own right. In the more informative comparison (ARM-resp-only vs. RAD), the paper consistently reports "slightly worse fluency" (Figure 3) and "slightly lags behind" (Figure 4). The paper is transparent about both settings, but the headline framing — "ARM performs on par with RAD" — is most strongly supported by the distillation result. The paper should be more precise in what it is claiming in each setting.

- **Efficiency gains are reported for the discriminator component in isolation, not for the full decoding pipeline.** Figure 6 and §5.6 measure only the reward model's per-token time. When guiding LLaMA-2-7B with TinyLLaMA, the base model forward pass dominates compute, making the practical end-to-end speedup considerably less than 10×. The paper would be substantially stronger with a full pipeline timing comparison, or an explicit statement that Figure 6 reports discriminator-only time.

- **Limited task diversity: both evaluated tasks are effectively binary attribute control.** Detoxification and sentiment control are both binary-ish classification problems. Whether the low-rank structure of the reward matrix holds for more nuanced attributes (e.g., formality, factual accuracy, multi-attribute control) is not tested. The paper's Limitations section (§6) hints at this ("certain toxicity patterns require high rank") but does not address the generalizability of the core claim beyond the two tasks.

### Trivial

None worth noting.

---

## Nice-to-Haves

- An explicit rank ablation over the dimensionality of W in Eq. (7) (e.g., rank-r constraint with r ≪ d) would tighten the connection between the theoretical motivation and the architecture, and test whether the full d × d matrix is actually necessary.
- A third controlled generation task with a more complex attribute would bolster the claim that the low-rank structure is general.
- Error bars or confidence intervals across evaluation runs would strengthen all comparisons, especially the close ARM-distilled vs. RAD result.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Rank threshold underspecification (Harsh Critic, §1 detail):** The main text (line 105) says "standard singular value cutoff" with details deferred to Appendix C.4. The appendix exists in the original submission; this is a parser-stripping artifact, not an author omission. Removed per the appendix rule.

- **Missing ARM-vs-DExperts comparison (Harsh Critic):** DExperts IS shown as a baseline in both Figures 3 and 4. The comparison is present; this criticism is factually incorrect.

- **PPO/Quark comparison inflates ARM's significance (Harsh Critic):** The paper explicitly states these methods require "training using feedback from the evaluation pipeline" (§5.4), acknowledging the training protocol difference. The criticism is a strawman of the paper's framing.

- **Ablation should report numerical values rather than scatter plots (Harsh Critic):** A minor presentation preference, not a methodological flaw. Scatter plots convey the trade-off curves clearly. Removed as a trivial style nitpick.

- **Strength: Low-rank training data theoretical argument (Strength Finder):** The main body defers the proof to Appendix B.1–B.2, and the rank-1 construction for unique-prefix datasets is for a simplified case. The generalization to real datasets with repeated prefixes is claimed empirically. This is a real strength but overstated as a standalone point; it is folded into the matrix-completion framing above.

---

## Novel Insights

The paper's most genuinely novel observation is not ARM itself but the empirical demonstration that a high-capacity model (RAD) trained via supervised learning on a sparse signal (reward matrix with ~1 observed entry per row) spontaneously learns a much lower-rank representation than its architecture permits. This connects reward modeling to the matrix completion literature in a principled way and suggests that rank may be a useful inductive probe for understanding when discriminator capacity is being "wasted" in the guided-decoding setting. The regularization strategy — pushing token-marginal predictions toward zero for random vocabulary items to encourage abstaining — is a simple but well-motivated design choice that has broader applicability in reward model training.

---

## Suggestions

1. Add a brief passage in §5.4 that analyzes *why* ARM-distilled exceeds RAD on sentiment: check if ARM-distilled's rank is lower and whether this produces better-calibrated scores for the DistilBERT classifier used for evaluation.
2. Report full end-to-end pipeline timing (base LM + discriminator) alongside Figure 6 to give a realistic picture of practical speedup.
3. State the singular value threshold precisely in the main text (not just "standard cutoff") — even one sentence pointing to the value used gives readers confidence in Figure 1.
4. Test on one additional task with a more graded or multi-dimensional attribute to stress-test the low-rank assumption.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Decision | Relation to this paper |
|---|---|---|---|
| `jY5oml9fe9.md` (SASA, self-detoxification) | 6.0 | Accept (Poster) | Similar scope — efficient controlled decoding for toxicity; comparable empirical grounding but less analytical |
| `shgx0eqdw6.md` (ARGS, reward-guided search) | 7.0 | Accept (Poster) | Decoding-time alignment via reward; stronger novelty in framing (eliminates RL training entirely) |
| `UAA2nWUtVl.md` (CARDS, cascade reward sampling) | 5.75 | **Reject** | Reward-guided efficient decoding; rejected due to overclaims and weak theoretical grounding — stronger analytical weakness than the paper under review |
| `7ohlQUbTpp.md` (Collab, mixture of agents decoding) | 6.25 | Accept (Poster) | Controlled decoding via mixture; narrower contribution, different approach |
| `l2zFn6TIQi.md` (AcT, activation transport) | 7.5 | Accept (Spotlight) | Controlling LM/diffusion via optimal transport; broader scope and stronger theory |
| `Yz7ts36V7A.md` (Backoff decoding) | 3.67 | Reject | Inference efficiency LM; lacks novelty and rigorous evaluation — clearly weaker than this paper |
| `n7iwmPacDt.md` (Polybasic speculative decoding) | 3.0 | Reject | Speculative decoding theory; very weak theoretical and empirical execution — clearly weaker |

**Positioning:** This paper sits between CARDS (5.75, rejected) and SASA (6.0, accepted) / Collab (6.25, accepted) in quality. It is notably stronger than CARDS analytically — the matrix-completion framing and rank analysis are substantive, the claims are honest, and the paper does not overclaim. It is comparable to SASA in scope (two tasks, one main architectural insight) but with a cleaner theoretical lens. It falls short of ARGS (7.0) and AcT (7.5) in novelty and breadth. The major weakness (ARM-distilled outperforming teacher without explanation) is real but does not invalidate the core contribution. The minor weakness about efficiency being discriminator-only is important for practitioners but not paper-breaking.

**Final score: 5.5** — this is a solid, honest, technically grounded contribution at the level of an accept, but limited in task scope, with one unresolved empirical anomaly (ARM-distilled > RAD teacher) that nudges it below a confident 6.

**Originality:** Moderate-good. The matrix-completion framing is novel; the bilinear architecture and distillation are straightforward.
**Importance of research question:** Good. Efficiency vs. quality in decoding-time control is practically relevant.
**Claims well-supported:** Mostly yes, with the caveat on ARM-distilled outperforming teacher.
**Soundness of experiments:** Good. Two tasks, multiple baselines, honest reporting of both ARM variants.
**Clarity:** Good. Paper is clearly written and well-organized.
**Value to research community:** Moderate-high. Provides a useful lens (matrix completion) and a practical efficient alternative to RAD.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>