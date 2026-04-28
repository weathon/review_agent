Now I have sufficient context from the paper and calibration anchors. Let me synthesize the final review.

Key observations from reading the paper:
1. The paper proposes SpARK, a sparse watermarking method using POS tags to select which tokens to watermark
2. It claims to maintain high detectability (TPR > 99%) while preserving text quality better than dense methods
3. The method watermarks tokens following words with specific POS tags (Verb, Noun, Determiner)
4. Detection relies on re-identifying watermarked positions by running a POS tagger on received text

Key issues to verify from the harsh critic:
1. **Synchronization fragility**: The critic claims the method breaks under paraphrasing/substitution attacks because POS structure changes. Looking at Table 3, SpARK does show drops (e.g., Llama2 50% Substitution: SpARK-Verb 72.4% vs SelfHash 92.3%). This is a real weakness documented in the paper itself.

2. **Statistical validity of z-test**: The critic claims POS tagger errors violate the binomial test assumptions. The paper does use Equation 1 (z-score) assuming fixed T and constant γ. POS tagger errors would indeed affect this, but the paper doesn't analyze this. This is a valid concern.

3. **Capacity normalization**: The critic argues that comparing sparse (5-15% tokens) vs dense (100% tokens) methods without normalizing for watermark capacity is unfair. The paper does acknowledge it watermarks fewer tokens but claims this is the point of sparsity. However, the critic's point about efficiency (bits per token) is valid - the paper doesn't measure embedding efficiency.

4. **False Positive Rate on human text**: The paper reports TNR on unwatermarked model text, not human text. This is a gap.

Let me now write the final review with proper calibration.

## Summary
This paper proposes SpARK, a sparse watermarking method for LLMs that selectively embeds watermarks in tokens following words with specific Part-of-Speech (POS) tags. The method aims to mitigate the quality-detectability trade-off by watermarking only 5-15% of tokens while maintaining statistical detectability through focused testing on watermarked positions. Experiments on Llama2 and Phi-3 across QA and summarization tasks show SpARK achieves >99% TPR with substantially lower ROUGE-L degradation and higher semantic similarity compared to dense watermarking baselines.

## Strengths
- **Effective quality preservation with maintained detectability**: Table 1 demonstrates SpARK-Determiner achieves 100% TPR on Llama2 while reducing ROUGE-L degradation to 11.07% compared to 22.37% for the Hard watermark baseline, with higher semantic similarity (0.814 vs 0.765). This empirically validates the core claim that sparse watermarking can preserve text quality.
- **Linguistically-grounded token selection mechanism**: The method introduces a novel approach using POS tags as synchronization anchors (Section 3.3, Algorithms 2-3), enabling the detector to identify watermarked positions without explicit markers. This differs fundamentally from prior dense watermarking approaches that modify every token.
- **Comprehensive empirical evaluation**: The paper evaluates multiple models (Llama2-7b, Phi-3), tasks (Long-form QA, Summarization), and attack types (substitution at 10/30/50%, DIPPER paraphrasing), providing broad coverage of performance characteristics (Tables 1-3, Figure 3).

## Weaknesses

### Fatal
None

### Major
- **Synchronization fragility under syntax-modifying attacks**: The detection mechanism (Algorithm 3) requires the receiver to re-identify watermarked positions by running a POS tagger on received text. This creates a hard dependency on syntactic structure remaining intact. Table 3 shows severe TPR drops under attacks that modify syntax: for Llama2 at 50% substitution, SpARK-Verb achieves 72.4% vs SelfHash's 92.3%—a 20 percentage point gap. The paper acknowledges this ("robustness lessens" at 30%+ substitution, line 312) but frames SpARK as "competitive" despite the substantial security gap. A watermark relying on surface syntax for synchronization is inherently vulnerable to the paraphrasing and substitution attacks it claims to resist. This limitation is not addressed by tuning and reflects a structural constraint of the approach.

- **No capacity-normalized efficiency analysis**: The paper compares sparse methods (watermarking ~5-15% of tokens depending on POS frequency) against dense methods (100% of tokens) and shows better quality metrics. However, it does not normalize for watermark capacity (bits embedded per token) or measure embedding efficiency. The quality advantage could be entirely explained by embedding less total information rather than embedding more efficiently. To claim SpARK improves the quality-detectability trade-off rather than simply operating at a lighter load, the authors should demonstrate higher detectability per unit of quality loss or per bit embedded. Without this, it is unclear whether SpARK is more susceptible to truncation or dilution attacks that remove small portions of text.

### Minor
- **POS tagger error propagation unanalyzed**: Equation 1 assumes a binomial test with fixed trials T and constant probability γ. However, the trials are selected by a POS tagger applied to output text, which has ~97% accuracy with non-random errors correlated with syntactic ambiguity. The paper provides no analysis of how POS tagging errors propagate into FPR/TPR metrics or whether systematic biases on certain syntactic structures could inflate false positives on human text. This affects the statistical validity of the detection test.

- **False positive rate not evaluated on human text**: The paper reports TNR on unwatermarked *model* text, not human-written text. Given the POS-tagging dependency, which may behave differently on human vs. model syntax patterns, FPR on genuine human writing is a critical missing metric for real-world deployment.

- **Inconsistent robustness across models**: Table 3 shows SpARK-Determiner performs well on Phi-3 for paraphrasing (87.1% TPR at DIPPER 40L, outperforming baselines) but lags on Llama2 (74.3% vs SelfHash 75.0%, nearly tied). For substitution attacks, the gap widens (76.6% vs 99.3% for Unigram on Phi-3 at 50%). The paper does not analyze why robustness varies substantially across models or which model characteristics drive this variance.

### Trivial
- **Implementation ambiguity in word boundary detection**: Algorithm 2 Line 5 requires verifying "when the model has generated a full word," but LLMs generate subword tokens. Detecting word boundaries autoregressively requires decoder-specific logic (whitespace detection, tokenizer-dependent checks) that introduces latency and complexity not discussed in the paper.

## Nice-to-Haves
- **Capacity-efficiency plots**: Adding a plot of quality vs. watermark entropy (bits embedded) would clarify whether SpARK is more efficient or simply lighter.
- **Synchronization failure case studies**: Showing examples where paraphrasing caused the detector to examine wrong tokens would help readers understand the robustness boundary.
- **Generation latency measurements**: Reporting tokens/second with SpARK vs. baselines would quantify the computational overhead of running a POS tagger during autoregressive generation.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 1 (Synchronization Fragility)**: This was KEPT as a Major weakness because it is factually correct and supported by Table 3 data in the paper. The 20 percentage point gap at 50% substitution is real and substantive.

- **Harsh Critic Point 2 (Statistical Validity)**: This was KEPT as a Minor weakness. The concern about POS tagger error violating binomial assumptions is valid, though the paper's empirical results show high TPR/TNR in practice, suggesting the impact may be limited.

- **Harsh Critic Point 3 (Capacity Normalization)**: This was KEPT as a Major weakness. The lack of efficiency normalization is a genuine gap in the experimental analysis.

- **Harsh Critic claim about "missing FPR on human text"**: This was KEPT as a Minor weakness—it is a real gap, though not fatal since TNR on model text is reported.

- **Strength Finder claim about "novel linguistically-grounded mechanism"**: KEPT as a strength—this is accurate and specific.

- **Strength Finder claim about "competitive robustness"**: MOVED to Removed Points. While Table 3 shows SpARK is competitive on Phi-3 for paraphrasing, the Llama2 substitution results (72.4% vs 92.3%) contradict the "competitive" framing. The weakness overrides this strength.

- **Generic strength about "clear motivation"**: KEPT but weakened—the motivation is clear, but the execution has structural limitations.

- **Nitpick about "implementation ambiguity in Algorithm 1 Line 5"**: Moved to Trivial—this is a minor presentation issue, not a fundamental flaw.

## Novel Insights
The paper identifies an underexplored dimension in watermarking: using linguistic structure (POS tags) as a synchronization channel rather than watermarking all tokens. While this approach successfully demonstrates that sparse watermarking can achieve high detectability with better quality preservation, it reveals a fundamental tension: any synchronization mechanism based on surface linguistic features inherits the fragility of those features under adversarial editing. This suggests future sparse watermarking methods should either (1) use more robust anchors (e.g., semantic or discourse-level features less susceptible to paraphrasing), or (2) incorporate error-correcting mechanisms that tolerate synchronization failures. The POS-based approach serves as a useful proof-of-concept for sparse watermarking but also illustrates the limits of syntax-dependent synchronization in adversarial settings.

## Suggestions
1. **Add capacity-normalized analysis**: Plot detectability (TPR at fixed FPR) against watermark entropy or bits embedded per token to demonstrate whether SpARK is more efficient or simply lighter than dense methods.

2. **Evaluate FPR on human-written text**: Test the detector on human-authored samples from the same domains (ELI5, FinanceQA, MultiNews, QMSum) to ensure the POS-based selection does not introduce systematic false positives.

3. **Analyze POS tagger error impact**: Quantify the disagreement rate between the generator's implicit POS logic and the detector's POS tagger, and measure how this affects the z-score distribution under the null hypothesis.

4. **Include adaptive attack evaluation**: Test an attack that specifically targets anchor words (e.g., replacing verbs with noun synonyms to change POS tags), which would directly probe the synchronization fragility identified in the robustness analysis.

5. **Report generation latency**: Measure tokens/second during autoregressive generation with SpARK to quantify the computational overhead of POS tagging at each step.

## Calibration and Scoring

I retrieved the following anchor papers for calibration:

**High-scoring anchors (avg ≥ 6):**
- `/home/wg25r/review_agent/human_reviews_2026/EhDgP69DJG.md` (PMark, avg 7.00, Accept): This paper provides theoretical guarantees for distortion-free semantic watermarking with multi-channel constraints. Unlike SpARK, PMark offers formal robustness analysis and proves distortion-free properties. SpARK lacks theoretical grounding and has documented robustness gaps.
- `/home/wg25r/review_agent/human_reviews_2026/HA8vzzT6Ax.md` (Watermark vs speculative sampling, avg 6.67): Focuses on statistical detectability with solid analysis.
- `/home/wg25r/review_agent/human_reviews_2026/t38nZqqi3Z.md` (LLM fingerprinting, avg 6.50): Evaluates stealthy watermarking across adversarial settings with comprehensive robustness analysis.

**Medium-scoring anchors (avg ~5):**
- `/home/wg25r/review_agent/human_reviews_2026/kxEM2vc7ne.md` (LingoLoop Attack, avg 5.00): Uses POS-based mechanisms but for attack purposes; reviewers noted insufficient defense evaluation and white-box assumptions.
- `/home/wg25r/review_agent/human_reviews_2026/neE8pqIqyR.md` (PRO watermarking, avg 5.00): Empirical open-source watermarking with robustness weaknesses against distillation attacks.
- `/home/wg25r/review_agent/human_reviews_2026/Vvks41GeL9.md` (DynamicBias, avg 5.50): Addresses quality-detectability trade-off empirically; rejected due to incremental novelty over prior dynamic bias methods.
- `/home/wg25r/review_agent/human_reviews_2026/HCp1xl0sol.md` (SimKey, avg 5.00): Semantic-aware watermarking with small/inconsistent robustness improvements; rejected.

**Low-scoring anchors (avg ≤ 4):**
- `/home/wg25r/review_agent/human_reviews_2026/U9LUhiOaLV.md` (PromptHash, avg 3.00): Instruction-side watermarking; rejected for lacking adaptive adversary evaluation and incomplete experimental validation.
- `/home/wg25r/review_agent/human_reviews_2026/SzrQBJDYHn.md` (Combinatorial watermarking, avg 4.50): Edit detection method; rejected for severe text quality degradation (delta=5.8) and missing baselines.
- `/home/wg25r/review_agent/human_reviews_2026/yr06ivlnaG.md` (DERMARK, avg 4.50): Multi-bit watermarking; rejected for limited baselines and missing bit match rate metrics.

**Positioning:** SpARK is stronger than the low-scoring anchors (3.0-4.5 range) because it demonstrates clear quality improvements with comprehensive experiments across multiple models and tasks, and the core empirical claims are supported by data. However, it is weaker than the high-scoring anchors (6.5-7.0 range) because it lacks theoretical analysis, has documented robustness gaps (20% TPR drop under substitution), and does not normalize for watermark capacity.

SpARK is most comparable to the medium-scoring anchors (5.0-5.5 range):
- Like SimKey (avg 5.00), SpARK shows quality improvements but has inconsistent robustness gains and lacks efficiency analysis.
- Like DynamicBias (avg 5.50), SpARK is empirically thorough but incrementally novel and missing theoretical grounding.
- Unlike PRO (avg 5.00), which was rejected for robustness against model modifications, SpARK's robustness issues are against text edits—a more standard evaluation—but the 20-point gap is still concerning.

The synchronization fragility (Major weakness) is similar to weaknesses in PromptHash (avg 3.00), which was rejected for only evaluating "oblivious paraphrasing" without adaptive adversaries. However, SpARK does evaluate substitution and paraphrasing attacks (albeit not adaptively), and shows competitive results on Phi-3, placing it above PromptHash.

The capacity normalization gap (Major weakness) is similar to DERMARK (avg 4.50), which was rejected for limited empirical evaluation. SpARK's evaluation is more comprehensive, placing it above DERMARK.

Given these comparisons, SpARK sits at the upper end of the medium range. It has real structural limitations (synchronization fragility, no efficiency analysis) that prevent it from reaching the 6+ range, but its empirical demonstration is solid enough to exceed the 4.5 range. I position it at **5.5**, slightly above SimKey (5.0) and PRO (5.0) due to better experimental coverage, but below DynamicBias (5.5) and PMark (7.0) due to lack of theoretical analysis and documented robustness gaps.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>