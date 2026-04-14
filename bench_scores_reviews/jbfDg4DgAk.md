## Summary

SpARK proposes a **Sparse Watermark** paradigm for LLMs: instead of biasing every generated token, only tokens immediately following words with specific Part-of-Speech (POS) tags are watermarked, and detection tests only those same positions. The z-test is adapted to count only the sparse watermarked positions, maintaining statistical power while constraining far fewer tokens. On two models (Llama2-7b, Phi-3) across four long-form QA and summarization datasets, SpARK substantially preserves text quality (ROUGE-L, perplexity, semantic similarity) relative to dense watermarking baselines while achieving ≥99% TPR.

---

## Strengths

- **Substantial and consistent quality improvement backed by numbers.** On Phi-3 long-form QA, SpARK-Verb degrades ROUGE-L by only ~5% vs. 14–66% for all baselines at matched TPR ≥ 99%. The trend holds on Llama2-7b and across summarization tasks (Tables 1–2). The improvement is large enough that it cannot plausibly be explained by measurement noise.

- **Conceptually clean design choice: test only where you watermark.** Prior work computes z-scores over all T tokens even when most are unlabeled; the insight that restricting T to watermarked positions preserves statistical power under sparsity is simple but underutilized. Replacing T with the anchor-selected count in Eq. 1 is the correct adaptation and it works.

- **POS anchoring as a synchronization mechanism is a concrete and practical novelty.** Using POS tags as deterministic, syntax-grounded anchors that can be independently recovered at detection time solves the "how does the detector know where to look?" problem without side channels. This is specific to this paper and not a restatement of prior work.

- **Cross-model evaluation.** Running identical experiments on both Llama2-7b and Phi-3 (a smaller, stronger model with different generation behavior) meaningfully strengthens the generalization case for the core claim.

- **Hard restriction with small γ is a purposeful design, not an accident.** The paper explicitly motivates γ=0.05 with hard restriction (no δ) because sparse positions must fully exploit their statistical budget. The contrast with SelfHash's γ=0.25 and additive δ is deliberate.

---

## Weaknesses

### Fatal
None.

### Major

- **Confounded comparison between SpARK (γ=0.05, hard restriction) and baselines (γ=0.25, soft logit biasing) — this is the central reproducibility and validity concern.** SpARK uses a green list covering only 5% of the vocabulary with a hard (infinite-δ) restriction, while all baselines use γ=0.25 with finite additive δ. These are two simultaneous differences: (a) fraction of watermarked *positions* and (b) per-token watermark strength at each selected position. The quality advantage could stem from (b) alone — a hard restriction to 5% of the vocabulary at rare positions may accidentally steer the model toward high-probability tokens under the oracle LM, explaining Figure 3's counterintuitive result (SpARK perplexity < no-watermark). Without an ablation isolating sparsity from per-token strength — e.g., running baselines at γ=0.05 with hard restriction, or running SpARK with γ=0.25 — the attribution of quality gains to *sparsity per se* is not established. As presented, the main claim could be rephrased as "a hard restriction to 5% of the vocabulary with very few constrained tokens preserves quality," which is a weaker and different contribution.

- **Robustness gap at moderate-to-high attack rates contradicts the "circumventing the trade-off" framing.** At 50% substitution on Llama2-7b, SpARK-Verb achieves 72.4% TPR vs. SelfHash at 92.3% and Unigram at 91.4% — a ~20 pp gap (Table 3). On Phi-3, SpARK-Verb reaches 72.5% vs. Hard at 98.6% and Unigram at 99.3%. SpARK's quality advantage comes with a meaningful robustness disadvantage under heavy attack; this is a shift in the trade-off, not a circumvention of it. The conclusion's claim "without degrading quality" and the abstract's framing of "mitigating this trade-off" are overstated relative to Table 3.

### Minor

- **The counterintuitive perplexity result (Figure 3: SpARK < no watermark) is unexplained.** The figure description states that the "No Watermark" (labeled "None") method shows the *highest* perplexity while SpARK-Determiner shows the *lowest*. A hard restriction to 5% of the vocabulary at selected positions could systematically force the model to pick tokens that happen to be high-probability under the oracle Llama2-13b (an alignment between the restricted green list and the oracle's high-probability mass). This would make the result a statistical artifact of the γ=0.05 hard restriction rather than evidence of semantic preservation. The paper reports this as evidence of quality but provides no mechanistic explanation, leaving a notable gap.

- **Short-text regime is entirely unaddressed.** The paper explicitly limits evaluation to long-answer datasets because POS anchors must be sufficiently dense. For short texts (e.g., chat turns, code snippets, brief answers), the number of anchor positions may fall well below the threshold needed for reliable z-test significance. This is not merely a nice-to-have: practical deployment of LLM watermarks covers short outputs extensively. The absence of any short-text analysis leaves a major gap in the practical validity of the method.

- **POS tagger not specified, and the online word-boundary detection mechanism is underspecified.** The paper says "we verify when the model has generated a full word by determining if the next token with the highest probability is the start of a new word," but during actual sampling, a different token (not the argmax) may be chosen. This creates a potential mismatch between generation-time anchor decisions and detection-time anchor recovery. The paper does not quantify how often this mismatch occurs, nor which POS tagger is used (spaCy, NLTK, Stanford, etc.). Near-perfect TPR in Tables 1–2 suggests the mismatch is rare in practice, but the mechanism should be stated precisely for reproducibility.

- **No empirical comparison with zero-distortion methods (Christ et al., 2023; Kuditipudi et al., 2023)**, both of which are discussed in related work as the directly relevant competitors on quality. The paper claims SpARK "outperforms previous LLM watermarking methods in quality" but omits these methods from the experimental table. The paper does note these methods struggle with low-temperature settings, which partially explains the omission, but no empirical comparison is provided even for the high-temperature setting where they should be competitive.

### Tiny

- **Algorithm 1, line 6 contains a pseudocode inconsistency.** `Sample(G)` should sample from the modified probability distribution P_M (which has been restricted to G by `ApplyGreenList`), not directly from G as a set. As written, the distinction between the green *list* and the modified *distribution* is blurred, making the algorithm harder to implement from the pseudocode alone.

- **Eq. 1 deviates from the standard one-proportion z-test denominator.** The standard binomial denominator is √(γ(1−γ)T); the paper writes γ√((1−γ)T), which differs by a factor of √γ. This formula appears to be inherited from Kirchenbauer et al. (2023) and likely reflects an implementation convention rather than a mathematical error, but it should be explicitly noted or justified.

---

## Nice-to-Haves

- **Ablation table: sparsity × γ × hard/soft restriction.** Running baselines at γ=0.05 with hard restriction (or SpARK at γ=0.25 with soft restriction) would cleanly demonstrate whether quality gains are attributable to sparsity or per-token constraint strength. This is the single highest-value addition.

- **Quality–detectability Pareto frontier plots for all methods.** Comparing methods at a single operating point (TPR ≥ 0.99) does not reveal whether SpARK dominates, is dominated, or merely lies on a different frontier. A full sweep over γ for SpARK alongside δ sweeps for baselines would make the comparison much more informative.

- **TPR vs. document length plot.** Given that sparsity means fewer anchor tokens per document, showing that TPR remains high even as text length decreases would directly address the short-text concern and strengthen the detectability claim.

- **Adaptive attacks targeting POS anchor positions.** The threat model assumes adversary awareness of the watermark scheme, so an adversary could specifically substitute tokens near anchor POS words. A targeted attack of this kind would test whether the POS-anchored structure is a security asset or liability.

- **Inference latency measurement.** Online POS tagging at every word boundary introduces overhead relative to hash-based methods. Reporting tokens/second with and without SpARK would clarify the deployment cost.

- **Evaluation on at least one additional language or style domain** to bound the scope of the POS-anchoring approach, since POS frequencies and tagger reliability vary substantially across languages and genres.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **[REMOVED – missing related works]** The harsh critic raises the absence of connections to entropy-thresholded watermarking, edit-robust watermarking, and selective watermarking literature. Per review policy, missing related works are excluded because we cannot verify the existence of external references.

- **[REMOVED – scope creep / non-standard demand for an empirical systems paper]** The harsh critic calls for a formal or semi-theoretical analysis of detection power under sparse, POS-conditioned selection. Demanding theoretical proofs from an empirical systems paper is not a standard expectation in this field setting.

- **[REMOVED – non-standard demand]** Requests for human evaluation and confidence intervals. Human evaluation is not a standard requirement for an algorithmic watermarking contribution; single-run evaluation is the norm for large-scale LLM benchmarks of this type.

- **[REMOVED – applies to any paper]** Generic strengths: "the paper is well-written," "the topic is important," "the algorithms are clearly presented." These are not distinguishing features.

- **[REMOVED – overstated]** The harsh critic argues the claim of a "novel type of LLM watermark" is oversold. While the contribution is incremental rather than paradigm-shifting, POS-conditioned sparse testing is a genuine and specific methodological contribution that is not a trivial restatement of prior work.

- **[REMOVED – factually incorrect framing]** The harsh critic's characterization of the null distribution assumption as undefended ignores that the detection algorithm counts only POS-selected positions as T; under that restricted count, if POS selection is independent of vocabulary partitioning (which it is by design, since hash depends on context), the null expectation γT for selected positions is correctly inherited. The TPR/TNR results in Tables 1–2 empirically validate the calibration.

---

## Novel Insights

The combination of concerns about the γ=0.05 confound and the perplexity-below-no-watermark result points to a genuinely underexplored phenomenon: **hard vocabulary restriction to a very small green list may systematically bias LLM outputs toward tokens that are high-probability under oracle models**, because the small green list must by chance include high-frequency vocabulary items, and hard restriction forces their selection even when marginally better alternatives exist. If true, this would mean SpARK's quality advantage is partly an artifact of token-level corpus statistics rather than sparsity architecture — but it would *also* be an interesting observation about how different watermarking mechanisms interact with language model probability distributions. Disentangling this mechanism would be a valuable contribution in its own right.

---

## Suggestions

1. **Run the critical ablation**: Re-run all baselines at γ=0.05 with hard restriction (no δ, green-list only), and run SpARK at γ=0.25 with hard restriction. Report the resulting quality and detectability. This single experiment would either (a) confirm that sparsity is driving the quality gains or (b) reveal that per-token constraint strength is the main factor, requiring a reframing of the contribution.

2. **Add a short-text benchmark** (e.g., TriviaQA short-answer, GSM8K). Report TPR as a function of answer length alongside anchor token count per document. If SpARK fails below some threshold, state it explicitly in the paper as a scoping constraint.

3. **Explain the perplexity result mechanistically.** Analyze whether the SpARK-Determiner green list, evaluated over the corpus, has higher average token frequency in a reference corpus than the baseline green lists. If so, acknowledge this explicitly and separate it from the quality preservation argument.

4. **Specify the POS tagger** (name, version, whether run incrementally on partial text or on completed words only) and formalize the word-boundary detection rule in Algorithm 2. State explicitly whether generation-time POS decisions (online, argmax-based proxy) can diverge from detection-time POS decisions (offline, on completed sequence) and at what rate.

5. **Qualify the robustness discussion.** In Section 4.3 and the conclusion, explicitly state that SpARK sacrifices robustness at high attack rates (50% substitution: ~20 pp below SelfHash on Llama2) in exchange for quality, and frame this as a trade-off shift rather than a trade-off circumvention.