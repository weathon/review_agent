## Summary
This paper introduces ARM, a low-rank autoregressive reward model that reformulates RAD training as a reward matrix completion problem and proposes a linear baseline-plus-marginal parametrization. ARM achieves RAD-level controlled generation quality (detoxification and sentiment control) with O(1) inference cost per token instead of O(k), demonstrated through clear efficiency measurements and trade-off curves across multiple base models.

## Strengths

- **Clean mathematical reformulation of RAD as matrix completion.** Section 3.1.1 successfully maps the RAD training objective to an incomplete reward matrix approximation task, providing a novel analytical lens that clarifies why rank analysis is relevant to decoding-time guidance and differentiates the value-function and Q-function paradigms.

- **O(1) inference cost validated with clear efficiency measurements.** Table 1 and Figure 6 unambiguously demonstrate ARM requires only 1 model call per decoding step versus k calls for RAD, with latency curves showing ARM's time-per-token remains constant (~0.001s) as k scales to 80 while RAD's increases linearly to ~0.010s. This is a well-measured, practically significant result.

- **Both training regimes are presented and compared honestly.** The paper reports results for ARM distilled from RAD (§5.4) and ARM trained from scratch on original dataset responses. Figure 3 and Figure 4 show that even the "resp. only" variant outperforms GeDi and DExperts baselines and remains competitive with RAD, and the paper provides a reasonable explanation for why distillation yields slightly better results (the teacher already compresses conflicting continuations into a single deterministic target, §5.4 p.245).

- **Principled regularization that explicitly controls rank.** The regularization loss (Eq. 11) that pushes marginal rewards for unobserved tokens toward the prefix baseline is well-motivated within the low-rank framework. Figure 5 shows this reduces effective rank from ~40-60 to ~10-20 while improving the toxicity/fluency trade-off, demonstrating rank management as a practical tool.

## Weaknesses

### Fatal
None.

### Major

- **Distillation dependency is underacknowledged relative to the framing.** The paper's abstract and introduction position ARM as "a simpler but more efficient low-rank parametrization" that "performs on par with the more flexible RAD parametrization," implying a direct structural replacement. However, the best-performing ARM variant requires distillation from a trained RAD teacher. While the paper does present ARM trained from scratch (§5.1, §5.4) and these results are competitive, the honest takeaway is that ARM functions more as a distillation target than as a standalone alternative that avoids the RAD training cost entirely. The gap between the distilled and scratch-trained variants (visible in Figures 3 and 4, especially at extreme β values) is real but the paper's framing does not sufficiently foreground this distinction. This matters because a practitioner considering ARM as a replacement for RAD must still train a RAD model to achieve the best results, partially negating the claimed efficiency benefits.

- **The low-rank assumption is validated only on training prefixes.** The rank estimation in §3.1.2 and Figure 1 is computed exclusively on a random subset of training prefixes (N ≤ 4000 from the training dataset D_f). The primary motivation for ARM's rank cap (rank ≤ d) relies on the claim that RAD's learned reward matrix is empirically low-rank. However, guided decoding operates on held-out or out-of-distribution prompts at inference time, and the paper provides no evidence that the low-rank property holds for these novel contexts. If the true reward surface for unseen prefixes requires higher rank, ARM's capacity ceiling would be reached. The limitations section (§6) acknowledges that "further qualitative research is needed to investigate whether certain toxicity patterns require high rank to represent them," but defers what should be a core validation.

### Minor

- **Theoretical justification in §3.1.3 for why training data supports a low-rank solution is somewhat hand-wavy.** The argument that "incompleteness makes it easier to learn a low-rank approximation" pivots on an edge case where every prefix appears exactly once in the dataset, guaranteeing a rank-1 compatible solution. As the paper itself acknowledges (p.111), "empirically calculating the minimal rank of the data is challenging due to the very large number of prefixes." The leap from "a low-rank solution exists for the incomplete matrix" to "RAD will converge to a low-rank solution" is not formally established. This does not undermine the core claim since the empirical rank measurement (Figure 1) does show RAD produces low-rank outputs, but the theoretical analysis overreaches.

- **The fluency improvement from regularization may conflate rank reduction with reward magnitude smoothing.** Figure 5's ablation shows that removing regularization increases both rank and perplexity, but the causal mechanism is not isolated. Pushing predicted rewards toward the prefix baseline dampens extreme reward scores, which could improve fluency simply by reducing the perturbation to the base LM distribution — independent of whether this is due to the rank constraint or the magnitude clipping. The paper does not include an ablation that separates these effects.

- **Score scale matching between ARM and RAD is not explicitly addressed.** The trade-off plots in Figures 3 and 4 vary β to trace curves, but since ARM and RAD have fundamentally different output parametrizations, identical β values do not guarantee comparable control strength. Without calibration of reward score scales (e.g., normalizing by score variance or clipping), the curves may be slightly misaligned, which could affect the direct comparison.

### Trivial

- Notation in Eq. 7 defines r_b as r_b(v|x) = ⟨h(x), w⟩, which does not depend on v despite the notation suggesting it does. This is intentional (the baseline score is the same for all tokens) but is confusing notation that could be clarified.

## Nice-to-Haves

- Include qualitative examples of prompts where ARM (distilled or scratch) diverges significantly from RAD, to reveal whether the low-rank constraint smooths over nuance or causes systematic bias. This would help practitioners understand when ARM's approximation breaks down.

- Evaluate whether the regularization improves fluency via rank reduction or reward magnitude clipping through an additional ablation (e.g., enforce low rank without mean-variance regularization, or apply magnitude clipping without the rank-constraining regularization).

- Report the rank of R_RAD and R_ARM on held-out prompts matching the inference-time distribution to validate that the low-rank assumption generalizes to novel contexts.

## Removed Points

- **"Structural: The core claim is invalidated by distillation dependency; ARM functions as compression, not standalone replacement."** The paper does present ARM trained from scratch as a separate experimental regime (§5.1, §5.4), and the scratch variant performs competitively (Figures 3 and 4 show ARM resp. only close to RAD and above baselines). The critique conflates "best variant requires distillation" with "only variant that works requires distillation." The framing concern above (Major #1) captures a fair version of this without overstating it.

- **"Evidential: Low-rank property not supported for OOD/held-out prompts."** This is a valid concern captured in Major #2, not a fundamental invalidation. The paper does empirically show ARM matches RAD quality on held-out prompts during generation evaluation (Figures 3, 4), which implicitly validates the low-rank assumption works in practice for those contexts. The missing piece is explicit rank measurement on test prefixes, which is a notable omission but not a contradiction of the core result.

- **"Theoretical justification in §3.1.3 is mathematically vacuous."** Overstated. The section provides supporting intuition for why a low-rank approximation is sufficient, not a rigorous proof. The empirical finding (Figure 1) is the actual basis for ARM's design — the theory is supplementary. This is addressed in Minor #3 as a hand-wavy theoretical argument, not a fatal flaw.

- **"Missing β normalization / score calibration makes curves potentially misaligned."** A fair methodological point captured in Minor #5. Not fundamental.

- **"Frozen embedding constraint is a limitation of the parametrization."** Correct that ARM's expressivity is tied to the quality of frozen base model embeddings, but this is a design choice (to promote generalization to unseen tokens, §5.1) rather than a flaw. It's a standard practice in this area and not specific to ARM. Removed.

- **Eq. 7 notation criticism (r_b depends on notation but not on v).** Minor notation issue captured in Trivial #1. The critic called it "minor but confusing" — appropriate as trivial.

- **Ablation conflating rank reduction with smoothing:** kept as Minor #4 (reasonable version).

## Novel Insights
The reformulation of RAD as a matrix completion problem genuinely connects the controlled decoding literature to the well-studied literature on matrix factorization and the softmax bottleneck in language modeling. The key insight — that a reward model need not represent the full space of prefix-token reward pairs but only a low-rank subspace of them — is both mathematically clean and practically useful. The explicit regularization toward a prefix baseline (Eq. 11) as a rank-control mechanism is an underexplored idea that deserves wider attention beyond the controlled generation context.

## Suggestions

1. Reframe the abstract and introduction to more clearly distinguish between ARM as a distillation target versus ARM trained from scratch, and state upfront that the distillation variant requires an initial RAD training step. The current framing overpromises standalone replacement.

2. Add a rank analysis on held-out test prefixes (matching the inference-time prompt distribution) to Table 3 or an appendix figure. This would directly address the gap between training-time rank measurement and the deployment regime.

3. In the ablation (Section 5.5), isolate the rank-reduction effect from the reward-magnitude-smoothing effect by adding an experiment that enforces low rank without the regularization loss, or applies clipping to the unregularized model.

4. Clarify Eq. 7 notation to avoid implying the baseline depends on v, e.g., write r_b(x) instead of r_b(v|x) and explain that the baseline is token-independent by design (consistent with the dueling network analogy).

5. In the limitations section, explicitly acknowledge the distillation dependency and the frozen-embedding constraint as practical constraints beyond the noted API variability.

## Score and Decision

For calibration, I compared against:
- **SASA** (`jY5oml9fe9.md`, avg 6.00, Accept poster): Lightweight controlled decoding algorithm with strong experiments on toxicity benchmarks; some methodology limitations (generalization from small dataset, PPL trade-offs) but accepted. ARM has comparable experimental rigor on similar tasks.
- **CARDS** (`UAA2nWUtVl.md`, avg 5.75, Reject): Solid efficiency experiments with reward-guided decoding but rejected for overclaiming ("rigorous" without proofs) and limited scope (single dataset). ARM has broader evaluation and is more honest about limitations.
- **VMV8gefvq8.md** (avg 6.00, Accept poster): Manifold-constrained reparameterization for model compression with strong empirical results but motivation not well motivated. Pattern similar to ARM — clean idea, solid experiments, motivation slightly weaker.
- **MCNC-related papers** (avg 6.00-7.00): Papers with strong compression efficiency results consistently score 6+. ARM's efficiency demonstration is comparably clean.
- **O6DKGUwv0m.md** (avg 5.00, Reject): Distillation paper with unclear motivation and insufficient evidence. ARM's distillation approach is much better motivated and supported by clear empirical comparisons.

ARM is stronger than the Reject anchors (CARDS, GOOD, FIRE) whose weaknesses were more severe (unsubstantiated theoretical claims, missing baselines, unclear contributions). ARM's weaknesses — primarily the distillation-dependency framing and the training-only rank validation — are real but not fatal, and the paper is transparent about presenting both training regimes. The efficiency gains are well-measured and practically significant. Relative to the Accept anchors (SASA at 6.00, MCNC at 6.00), ARM's experimental protocol is similarly rigorous but its theoretical claims are slightly less substantiated. I position ARM slightly below the 6.0 cluster of the Accept anchors due to the framing issue around distillation dependency, but above the 5.0 Reject anchors.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>