Now I have enough context to write the final consolidated review.

---

## Summary

SpARK introduces *Sparse Watermarking*, a category of LLM watermarking that applies watermarks only to tokens following words with specific Part-of-Speech (POS) tags (Verb, Noun, Determiner), and adjusts the z-test detection to check only those anchored positions. By leaving the majority of tokens undisturbed and focusing statistical detection on the sparse watermarked subset, SpARK substantially reduces degradation to generated text quality while maintaining TPR ≥ 99%. Experiments on Llama-2-7b and Phi-3 across four datasets (long-form QA and summarization) demonstrate consistent quality improvements over four dense-watermarking baselines at matched detectability, with meaningful but acknowledged robustness tradeoffs under heavy attacks.

---

## Strengths

- **Substantially improved quality-detectability tradeoff with compelling margins.** On Llama-2-7b, SpARK-Determiner reduces ROUGE-L degradation to only 11.07% (Long-form QA) and 10.99% (Summarization) versus at least 22% for the best-performing baseline (Hard watermark) and up to 50% for Unigram. On Phi-3, SpARK-Verb degrades quality by only 5.17% vs. 13.75% for the next-best baseline. These are large, consistent gaps across two models and two tasks, not marginal improvements.

- **Semantic similarity preservation is distinctive and well-documented.** SpARK-Verb achieves 0.836 (Llama-2) and 0.850 (Phi-3) semantic similarity between watermarked and non-watermarked text, compared to 0.765/0.567 for Hard and 0.652/0.425 for Unigram. The difference is not a statistical artifact; the qualitative example in Table 4 concretely illustrates why — dense watermarking with SelfHash forces incoherent outputs (e.g., "Not enough crowd," "Marketton"), while SpARK-Determiner produces fluent, faithful text.

- **The core insight is principled and novel.** The observation that the statistical test gains nothing from including non-watermarked tokens in T, and that decoupling watermark placement from detection scope can improve quality without harming detectability, is a clean and non-obvious contribution. Unlike entropy-based skipping (Lee et al., 2023), which was domain-specific to code and required dynamic thresholding, SpARK's POS anchor provides a deterministic, language-structure-based positional scheme applicable to general text generation.

- **Perplexity results corroborate quality claims with an independent metric.** Figure 3 shows SpARK-Determiner achieves consistently lower median perplexity and smaller variance than all baselines, using Llama-2-13b as the oracle. The reduced variance is a notable secondary finding, suggesting SpARK produces more predictable generation quality.

---

## Weaknesses

- **Robustness drops significantly under heavy attacks, with no mechanistic explanation.** At 50% substitution on Llama-2-7b, SpARK-Verb falls to 72.4% TPR and SpARK-Determiner to 67.6%, compared to SelfHash at 92.3% and Unigram at 91.4% (Table 3). The paper acknowledges this but offers no analysis of *why*. The likely mechanism — that 50% random substitution has a meaningful probability of replacing POS anchor words themselves, which destroys anchor-next-token relationships — should be verified and discussed. This is not a fatal flaw (the paper is explicit about the tradeoff), but the absence of mechanistic understanding limits actionable guidance on when SpARK is and isn't appropriate.

- **Targeted attacks against POS anchors are not evaluated or even acknowledged.** Section 3.2 states that the adversary is "aware of the presence of watermarks." If the watermarking scheme is publicly known (which is typically assumed for cryptographic security), an adversary can trivially identify which tokens are watermarked by POS-tagging the generated text and targeting only the tokens immediately following anchor-POS words. This targeted substitution would be far more efficient at defeating the watermark than the random substitution evaluated in Table 3. For a method with publicly derivable anchor positions, this is a non-trivial security gap that the paper must at least discuss and bound.

- **POS tagger consistency between generation and detection is never analyzed.** The method requires that the POS tagger produces the same label for a given word both during incremental autoregressive generation (where the context is a partial sentence) and during post-hoc detection (where the full text is available). POS taggers can produce different tags for the same word in different sentential contexts or when context is truncated. Any mismatch causes the detector to miscalibrate T (the denominator of the z-score), degrading TPR. Neither a theoretical bound nor an empirical ablation (e.g., how often do encoder-time and decoder-time POS labels disagree?) is provided.

- **Hard restriction with γ = 0.05 creates a forced low-quality sampling regime that is not analyzed.** SpARK uses hard restriction rather than a logit bias δ: at watermarked positions, the model must sample from only 5% of word-starting tokens. If no high-probability token falls in the green list at a given position, the model is forced to emit a low-probability token, risking quality degradation at precisely the moments that matter most (rare or syntactically constrained contexts). The paper argues that sparse watermarking preserves quality, but does not verify that the green-list tokens at watermarked positions are systematically in high-probability regions. A plot of the probability mass of the sampled green-list token (relative to the argmax token) would close this gap.

- **The fraction of watermarked tokens is not reported in the main paper.** The core reason SpARK outperforms dense baselines on quality is that it watermarks a much smaller fraction of tokens. This fraction is the key variable that explains the quality-detectability tradeoff, yet the paper defers it to Table 7 in the appendix. Without this number in the main text, readers cannot assess whether SpARK's advantage reflects a genuine algorithmic insight or simply a different operating regime (fewer watermarked tokens → less distortion → lower robustness). This should be a central result, not an appendix entry.

- **Algorithm 1, Line 6 contains a pseudocode inconsistency.** Line 6 reads `Sample(G)`, which implies unconditional sampling from the green list G at every step. But the described method only restricts sampling to G when the POS condition is met; at other positions, sampling should proceed from the original distribution. The `POSWatermark` subroutine (Algorithm 2) returns the original P_M unchanged when the condition is not met, so the correct instruction would be `Sample(P_M(t))`. This inconsistency between the pseudocode and prose description is confusing and should be corrected.

- **Figure 3 x-axis labels are garbled and internally inconsistent.** The x-axis labels include "Selfish," "Llama2," and "None," which do not correspond to any method name used elsewhere in the paper (the methods are Hard, LeftHash, SelfHash, Unigram, SpARK-Verb, SpARK-Noun, SpARK-Determiner, and No Watermark). This makes Figure 3 difficult to interpret reliably without the original figure rendering.

- **TNR for SpARK-Determiner is measurably lower than some baselines.** Table 1 shows SpARK-Determiner achieving TNR of 98.0–98.8%, versus 100% for Hard watermark and Unigram. While 1–2 percentage points may seem small, in deployment this means a non-trivial false-positive rate on human-written text. The paper does not discuss this or whether it can be mitigated by threshold tuning.

---

## Nice-to-Haves

- Human evaluation of text quality (e.g., fluency ratings) would strengthen the quality claims beyond ROUGE-L and semantic similarity, which are imperfect proxies for human preference.
- An ablation over different POS tag combinations (e.g., Adjectives, Adverbs, Prepositions) beyond the selected three, including tags with less than 100% document frequency, would clarify how sensitive performance is to the choice of anchor tags.
- Cross-model detection experiments (watermark with Llama-2, detect assuming Phi-3's vocabulary) would test whether the scheme is robust to model version or architecture differences in the detector.
- Measuring and reporting inference latency overhead from real-time POS tagging would be useful for practitioners, though this does not affect the paper's core claims.
- Evaluating the False Positive Rate on human-written text (in addition to unwatermarked LLM text) would better characterize the TNR figures and their real-world implications.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **[REMOVED] Lee et al. (2023) as a required baseline.** Lee et al.'s entropy thresholding was specifically designed for *code generation* to preserve correctness — it is not a general-purpose text watermarking method. Criticizing SpARK for not including it as a baseline in a general-text setting is scope creep.

- **[REMOVED] Distortion-free methods (Christ et al., Kuditipudi et al.) as required baselines.** The paper explicitly notes that sampling-based schemes "struggled to produce a detectable watermark for low-temperature settings" (citing Piet et al., 2023). Since the paper's operating regime (TPR > 0.99) is incompatible with these methods in practice, excluding them as baselines is reasonable and explained.

- **[REMOVED] Demand for statistical confidence intervals across all tables.** Single-run evaluation is the norm in the LLM watermarking literature (Kirchenbauer et al., Zhao et al., and related works all report single-run figures). Demanding confidence intervals as a gating condition is above the standard of the field.

- **[REMOVED] Criticism of using Llama-2-13b to evaluate Llama-2-7b perplexity as fundamentally flawed.** The paper explicitly cites Jovanović et al. (2024) as precedent for this methodology. The criticism that shared architecture inflates the quality signal may have marginal merit but is not a meaningful objection given established community practice.

- **[REMOVED] "Contribution granularity" complaint about introducing a category with one instantiation.** Defining a broader class (Sparse Watermark) and demonstrating it through one instantiation (SpARK) is entirely standard in ML systems papers (cf. how "attention mechanisms," "residual connections," etc. were introduced). This is not a weakness.

- **[REMOVED] Demand for theoretical proofs for z-test validity under POS conditioning (as a gating condition).** The z-test validity concern is real (see Weaknesses), but demanding a full statistical proof before acceptance would be above the standard for an empirical systems paper. The requirement is a clear explanation and empirical verification, not a formal theorem.

- **[REMOVED] Requests for larger datasets and more models.** The paper covers two models and four datasets across two task types. This is adequate for an ICLR submission in this subfield; the current model zoo and dataset coverage are not obviously insufficient.

---

## Novel Insights

The most genuinely novel insight — validated by the empirical evidence — is that **watermark detection strength derives entirely from the tested token subset, not from the full generated text length.** This separation of the *generation regime* from the *detection scope* is non-obvious: prior work implicitly treats every token as both a watermark carrier and a detection signal. SpARK demonstrates that if anchor positions are known to the detector, restricting the z-test denominator T to only those positions preserves detection power while allowing the rest of the vocabulary to be sampled freely. The corollary — that the quality-detectability tradeoff is not fundamental but is instead an artifact of conflating generation and detection scope — has implications beyond SpARK itself, potentially motivating future work on other deterministic positional anchors (e.g., syntactic roles, semantic heads) that balance robustness and quality differently than POS tags.

---

## Suggestions

1. **Add a targeted-attack experiment.** Implement a substitution attack that specifically replaces tokens immediately following anchor-POS words, and report TPR under this attack. If TPR degrades substantially below random substitution, acknowledge this as a genuine security limitation and discuss whether a secret POS-tag selection mechanism could mitigate it.

2. **Report the watermarked-token fraction prominently in the main body.** Move the "% of tokens watermarked" figures from Table 7 (appendix) into Table 1/2 or a dedicated main-text figure. This number is essential context for interpreting the quality-detectability tradeoff.

3. **Fix Algorithm 1, Line 6.** Change `Sample(G)` to `Sample(P_M(t))` to correctly reflect that non-watermarked positions sample from the unmodified distribution.

4. **Provide a POS consistency ablation.** Report the POS agreement rate between incremental (generation-time) and full-context (detection-time) tagging on a held-out text sample. Even 1-2% disagreement per anchor position could meaningfully reduce T and hurt TPR for short texts.

5. **Analyze or empirically bound the hard-restriction pathology.** For each POS-anchored watermarking event, report the average probability rank of the sampled green-list token. If the median rank is low (e.g., top-5), the concern is mitigated empirically; if it frequently falls outside the top-100, the hard restriction is potentially a quality risk for rare syntactic contexts.

6. **Fix Figure 3 x-axis labels** to match method names used in Tables 1–2.

---

**Axis Evaluations:**
- **Novelty:** Moderate-to-high. The POS-anchor mechanism is a clear and specific idea, and the decoupling of generation scope from detection scope is a genuine conceptual advance, not merely an engineering variation.
- **Technical soundness:** Moderate. The core statistical framework is sound, but there are real unresolved questions around POS consistency, hard-restriction pathology, and the absence of targeted-attack analysis. The pseudocode error is a presentation gap.
- **Empirical support:** Moderate-to-strong for quality claims; notably weaker for robustness claims, which are the method's primary limitation.
- **Significance:** Moderate-to-high. The quality improvements over baselines at matched detectability are large and consistent, and the underlying idea generalizes beyond the specific POS instantiation.
- **Clarity:** Adequate overall, with specific lapses in the pseudocode and Figure 3 that require correction.