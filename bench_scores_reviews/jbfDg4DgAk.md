Now I have enough information to write the consolidated review. Let me cross-check the key claims.---

## Summary

SpARK proposes *Sparse Watermarking* for LLMs: rather than watermarking every generated token, it restricts generation to a green list only at tokens that immediately follow words carrying selected Part-of-Speech (POS) tags (Verb, Noun, Determiner), then runs the z-score detection test only over those anchored positions. This decouples the quality penalty from the watermark's detection power. Evaluated on Llama2-7b and Phi-3 across four datasets (two long-form QA, two summarization), SpARK reduces ROUGE-L degradation to 10–22% compared to 22–65% for dense baselines, while maintaining ≥99% TPR. Robustness under light attacks is competitive, but degrades under heavy paraphrase and high-substitution scenarios.

---

## Strengths

- **Quantitatively large and consistent quality gains.** On Llama2-7b (Table 1), SpARK-Determiner reduces ROUGE-L by 11% vs. 22–47% for the baselines at matched detectability (≥99% TPR). On Phi-3 (Table 2), the gap widens further (6% vs. 14–66%). These are not marginal differences; they are large and replicate across two very different model families and two task types. Semantic similarity gains (0.836 vs. 0.652–0.765 on Llama2-7b) point in the same direction.

- **Elegant and well-motivated core design.** The key insight—that sparse watermarking is only useful if detection is also conditioned on the same sparse positions—directly explains why naïvely reducing watermark density in prior methods does not solve the trade-off. The paper states this clearly in Section 3.3: "Attempting to watermark sparsely without knowing the location of the watermarked elements would be akin to using the previous watermark methods with low strength." This is genuinely non-obvious and distinguishes SpARK conceptually from simply lowering δ.

- **Anchor-after-POS design avoids POS-consistency circularity.** The choice to watermark the token *after* a POS-tagged word, rather than the word itself, is a careful engineering decision that prevents the watermarked token from corrupting the POS tag of the anchor. The paper explicitly notes (Section 3.3): "watermarking those words directly would not guarantee it to have the same POS tag after being watermarked." This design choice ensures consistent encoding/decoding without look-ahead.

- **Perplexity evidence provides corroborating signal.** Figure 3 shows SpARK-Determiner achieves the lowest perplexity and lowest variance under a Llama2-13b oracle across both task types. Using a stronger external model as an oracle (following Jovanović et al., 2024) is a methodologically reasonable choice, providing a metric independent of the task reference texts.

---

## Weaknesses

### Fatal
None. The core claim (sparse watermarking with selective detection significantly preserves quality at matched detectability) is supported by clear empirical evidence.

---

### Major

- **Absence of random-sparse baseline makes the core attribution unverifiable.** The most critical missing experiment is a comparison against a watermark that randomly selects the same fraction of positions (matching SpARK's effective density) and uses the same γ=0.05 hard restriction, but without POS anchoring. Without this, it is impossible to determine whether the quality gains stem from POS-anchoring specifically—which is the claimed design contribution—or from *any* sparse selection scheme. If random sparse selection performs equally well, the POS contribution reduces to mere consistency in locating watermarked positions at detection time, and the paper's framing around POS as a principled anchor needs to be substantially revised.

- **γ mismatch undermines the fairness of quality comparisons.** SpARK uses γ=0.05 while all baselines use γ=0.25. A smaller green list (5% of vocabulary) under hard restriction is extremely constraining at each watermarked position but is applied rarely. Baselines use a larger green list (25%) applied everywhere. The quality advantage could at least partly reflect the lower density of forced deviations rather than any benefit of POS-anchoring per se. A fair analysis would either: (a) include a Pareto curve varying both sparsity and γ for SpARK and δ for baselines at matched expected perturbation budget, or (b) provide an intuitive argument for why γ=0.05 is the right operating point given the sparsity. The paper does note this choice is to "increase the strength of each watermark token," but does not analyze whether the quality improvement survives at matched γ or matched expected KL divergence.

- **Robustness is materially weaker under heavy attacks, yet the conclusion overstates this.** At 50% substitution on Llama2-7b, SpARK-Verb drops to 72.4% TPR vs. 92.3% for SelfHash and 91.4% for Unigram. Under DIPPER 40L-400 on Llama2-7b, SpARK-Verb falls to 43.5% vs. 69.5% for SelfHash. The conclusion states "competitive robustness against both substitution and paraphrasing attacks," which is accurate for SpARK-Determiner under DIPPER but misleading for the Verb variant under high-substitution. The paper should either honestly acknowledge the robustness–sparsity trade-off as a first-order limitation, or show how design choices (e.g., POS tag selection) affect the robustness profile.

- **POS tagger unspecified and word-boundary heuristic underexplained, harming reproducibility.** Algorithm 2 uses `POS(W, T)` without naming the tagger implementation. Universal POS tags (Petrov et al., 2012) define the tagset but not the tagger. More importantly, Section 3.3 states that word completion is detected by checking "if the next token with the highest probability is the start of a new word." This heuristic is problematic: during stochastic sampling, the actually generated token is often not the highest-probability token. If the sampled token does *not* start a new word while the argmax does, the encoding and detection processes may disagree about when a new word has started—producing systematic mismatches between the encoding-time anchor identification and the detection-time re-identification. The paper claims this "could consistently inform us when a full word has been generated" but provides no empirical validation of the mismatch rate.

---

### Minor

- **Effective T per document never reported.** The number of tokens actually watermarked per document (effective T) is fundamental to understanding the method, yet it is never reported for any POS tag variant or dataset. This quantity determines the z-test's reliability (especially for shorter texts), allows fair density comparisons with dense methods, and characterizes the practical operating regime. Table 7 (appendix) reports document frequency of POS tags but not the resulting token count. At minimum, the expected T and its distribution should appear in the main results.

- **Insertion/deletion resilience claimed but not tested.** Section 3.3 explicitly states POS anchoring "makes the watermark more resilient to insertions/deletions of tokens in the generated text." The robustness section (4.3) tests synonym substitution and DIPPER paraphrasing, but not insertions or deletions. The claim is plausible but remains unsubstantiated in the experiments.

- **No adaptive attacks targeting POS anchors.** Because SpARK's anchor structure is publicly described, an informed adversary could minimally rewrite anchor words (e.g., converting verbs to noun phrases) to shift POS tags without substantially altering meaning. This would reposition or eliminate anchor positions, degrading TPR without requiring heavy edits. The paper tests standard attacks from prior work, which is appropriate for a baseline robustness evaluation, but the absence of even a discussion of this obvious adaptive attack weakens the robustness story.

- **Semantic similarity metric encoder unspecified.** Tables 1 and 2 report "Semantic similarity" (Sem.) but nowhere identify the embedding model or similarity function used. This is needed to reproduce the numbers and to interpret the metric's alignment with true semantic fidelity.

- **Statistical validity of the z-test under POS conditioning not formally addressed.** The null hypothesis (no watermark applied) implies that any token's probability of falling in the green list is γ by construction of the pseudo-random hash. POS-conditioning selects *which* positions to test, but does not in principle change the null probability at those positions—so the test should be approximately valid. However, the paper does not state this argument, leaving readers uncertain. A one-paragraph argument (not a full proof) clarifying why the null remains valid under selective position testing would close this gap. The empirical TNR of 98–100% is reassuring but is not a substitute for the argument.

---

### Tiny

- The conclusion writes "encodes watermark information into the generated text, **without degrading its quality**." Tables 1–2 clearly show ROUGE-L and semantic similarity are still reduced relative to no watermark. The accurate statement is that degradation is substantially smaller than that of prior methods.

- Figure 3 showing "No Watermark" with the highest perplexity under Llama2-13b is somewhat counterintuitive and is not explained in the text. It is plausible (the 7b model may produce less "13b-preferred" text than the green-list-constrained output), but the paper should note explicitly why this result is expected rather than leaving the reader confused.

---

## Nice-to-Haves

- **TPR vs. effective T curve.** A plot of detectability as a function of the number of anchored positions actually scored would reveal the minimum text length for reliable detection—practically important for short-form tasks.

- **Human evaluation or stronger quality metric for a subset of outputs.** ROUGE-L measures reference overlap, not fluency or coherence. For at least one dataset, a GPT-4-based evaluation or grammar error count on a sample of outputs would provide complementary evidence for the central quality claim.

- **Pareto frontier plots for SpARK and baselines.** Showing quality vs. TPR as γ varies for SpARK alongside δ-varying curves for dense methods would make the trade-off geometry visible and allow readers to compare operating points honestly.

- **Discussion of multilingual and short-form generalization.** SpARK depends on English POS taggers; its applicability to other languages or informal registers (e.g., code, tweets) is an open question worth flagging.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Title/abstract language is "too strong"** (Harsh Critic): The abstract's "outperforms previous methods in quality" is supported by the evidence in Tables 1–2. This is not an exaggeration.

- **Perplexity of "No Watermark" being highest is "unexplained"** (Harsh Critic): The paper explicitly states (Section 4.2) that Llama2-13b is used as the oracle model following prior work. This explains why the 7b non-watermarked output is not necessarily the lowest-perplexity text under a 13b evaluator. The explanation exists and is reasonable; the criticism that it is "not well integrated" or "counterintuitive without explanation" misreads the paper.

- **Claim that SelfHash example in Table 4 uses "aggressive hyperparameters"** (Harsh Critic): δ=10 for SelfHash was selected by the paper's own hyperparameter search to achieve TPR >0.99, which is the common evaluation condition across all methods. The resulting severe quality degradation is the *consequence* of needing high TPR with a dense watermark—not a cherry-picked parameter. The comparison is valid.

- **Requesting theoretical proofs / confidence intervals for large-scale single-run benchmarks** (Harsh Critic): Single-run evaluation with a standardized benchmark (WaterBench-style) is the norm in this sub-field. Requesting confidence intervals is not a standard expectation for this type of empirical comparison paper.

- **"Unfair comparison because SpARK's heavy local restriction at selected positions"** (Harsh Critic): This is not unfair to SpARK—it uses hard restriction at fewer positions. The density reduction is the *mechanism* of the contribution. Calls to "match the number of affected positions" would effectively prohibit evaluating the method at all.

- **Absence of limitations section** (Harsh Critic): The paper is evaluated on scientific content, not on the presence of a formally labeled limitations paragraph. The weaknesses described above (robustness, POS dependency, reproducibility) constitute the real substantive gaps.

---

## Novel Insights

The genuinely novel conceptual contribution here is the *selective detection* pairing: sparse watermarking is not useful unless the detector *also* conditions on the sparse positions, because including non-watermarked tokens in the z-test dilutes the signal. This is a clean insight that prior entropy-thresholding or selective-embedding approaches (e.g., Lee et al., 2023 for code) did not frame as a general design principle. The corollary—that hard restriction with very small γ is sustainable precisely because the restriction occurs rarely, preventing the vocabulary collapse that would occur if applied at every position—is also interesting and is implicitly demonstrated rather than explicitly argued. Making this argument explicit would sharpen the paper's theoretical contribution considerably. Beyond these, the reviews do not surface insights beyond the paper's own contributions.

---

## Suggestions

1. **Run the random-sparse baseline.** Select the same fraction of positions as SpARK-Verb/Noun/Determiner (estimated from Table 7's document frequencies), apply γ=0.05 hard restriction at those random positions, and run the same evaluation. This single experiment would either confirm POS anchoring as essential or reveal that the gains are primarily from sparsity itself—either outcome is informative and publishable.

2. **Report effective T statistics.** For each POS variant and each dataset, report the mean and standard deviation of the number of watermarked positions per document. This belongs in Table 1/2 or a dedicated appendix table. Include the minimum T required for reliable detection at a 1% false positive rate given γ=0.05.

3. **Acknowledge and reframe the robustness trade-off.** Rather than claiming "competitive robustness," explicitly frame the robustness–quality Pareto as the main trade-off: SpARK occupies a region of high quality and moderate robustness. Show this geometrically if possible. Robustness can be tuned upward by increasing anchor density or choosing more attack-resilient POS tags—say this explicitly.

4. **Specify the POS tagger and validate the word-boundary heuristic.** Name the tagger (e.g., spaCy's `en_core_web_sm`, NLTK, Stanza). Empirically measure the mismatch rate between encoding-time and detection-time anchor identification on a held-out corpus to quantify the practical reliability of the word-boundary heuristic.

5. **Specify the semantic similarity encoder.** Identify the sentence embedding model used and report the metric (e.g., cosine similarity of sentence-BERT embeddings) to allow reproducibility.

6. **Add an intuitive argument for z-test validity.** In Section 3.4, add one paragraph explaining that under the null (no watermark), the probability of any selected position's token being in the green list is γ by pseudo-random construction of the hash function, so POS-conditioning on which positions to test does not change the null distribution—and is therefore valid.

---

**Overall assessment:** SpARK is a solid paper with a clear, well-motivated contribution and genuinely large empirical quality improvements relative to baselines. The central ideas are sound and the execution is reasonable. However, the missing random-sparse ablation is a significant hole—it is the single experiment needed to confirm that POS anchoring is a meaningful design choice rather than a proxy for any-sparse scheme. The γ mismatch and the overstated robustness claims also need to be addressed. The paper is interesting and likely above the acceptance threshold on novelty and empirical significance, but requires these ablations and clarifications to be convincing at ICLR's standard. **Novelty: moderate-to-solid**; **Technical soundness: moderate** (good ideas, gaps in validation); **Empirical support: good for the quality claim, weak for robustness and POS-specific attribution**; **Significance: moderate-to-high for the practical deployment problem**; **Clarity: good at the concept level, incomplete at the implementation and statistical level**.