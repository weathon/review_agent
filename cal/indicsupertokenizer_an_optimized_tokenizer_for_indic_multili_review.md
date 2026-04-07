=== CALIBRATION EXAMPLE 27 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the focus (Indic tokenizer) and the key feature ("optimized"). The abstract makes strong, quantifiable claims regarding fertility score improvements (39.5% over LLaMA4, 18% over Sutra) and inference throughput gains (44%). These claims are supported by the results in the paper. The abstract also succinctly summarizes the scope of the evaluation and ablations. No major issues.

### Introduction & Motivation
The introduction effectively motivates the problem of multilingual tokenization, specifically for Indic languages, highlighting challenges of script diversity, morphology, and fairness (skewed fertility). The five research questions are well-posed and guide the paper's contributions. The contributions are clearly listed. One minor point: the claim of being "the first to carry out a comprehensive benchmarking" is somewhat overstated, as there are prior comparative studies (e.g., Karthika et al., 2025a,b), though the extent of evaluation here (intrinsic metrics, downstream tasks, inference latency, ablations) is indeed thorough.

### Method / Approach (Section 3)
The method builds directly on SuperBPE's two-stage curriculum (subword then superword learning). The novel adaptations are: 1) using LLaMA-4 regex for pre-tokenization in Stage 1 (instead of GPT-2), 2) introducing sentence-level boundary constraints in Stage 2 to prevent cross-sentence merges, and 3) applying these to a carefully curated multilingual corpus. The design choices are justified empirically (e.g., Table 1 shows LLaMA-4 regex improves fertility). However, the sentence-boundary constraint is mentioned only briefly and not ablated; its necessity and impact are unclear. The method is reproducible given the description and promised release of code.

**Major Concern: Missing Baseline.** The core two-stage algorithm is from SuperBPE (Liu et al., 2025a). The paper compares against BoundlessBPE (one-stage) but does **not** include SuperBPE as a baseline. This is a critical omission. To claim state-of-the-art, a direct comparison with SuperBPE on Indic languages is necessary to demonstrate that the adaptations (LLaMA-4 regex, sentence constraints) yield measurable improvements over the original SuperBPE.

### Experiments & Results (Sections 4 & 5, Appendix)
The experimental evaluation is extensive, covering intrinsic metrics, downstream task performance, inference latency, and detailed ablations.

**Intrinsic Evaluation (Section 4.1):** Tables 3, 4, 6, 7 convincingly show IST achieves superior fertility, NSL, Rényi efficiency, and bytes-per-token across most languages. The comparison includes a wide range of strong baselines. The evaluation framework and dataset will be a valuable resource.

**Downstream Task Performance (Section 4.2):** Results in Table 8 show that models trained with IST achieve comparable English performance and a slight improvement on average for Indic benchmarks (0.388 → 0.394). However, the differences are small, and the paper does not report statistical significance or standard deviations across multiple runs. For ICLR, this lack of statistical rigor weakens the claim of "maintaining comparable performance." The results suggest the tokenizer does not harm performance, but the improvement is marginal.

**Inference Latency (Section 4.3):** The 44% throughput improvement (Table 5) is compelling and directly tied to the reduced sequence lengths (Table 21). The experimental setup (Triton on 8 H100s) is reasonable. This is a strong practical result.

**Vocabulary Allocation & Glitch Tokens (Sections 4.4, 4.5):** The analysis of corpus-driven vs. explicit merging is informative. The glitch token analysis (Figure 5) is interesting and suggests IST uses vocabulary more efficiently, but its connection to downstream model quality is not established.

**Tokenizer Replacement (Section 4.6):** The ReTok-style experiment shows that IST can be integrated into a pre-trained model via continual pretraining with minimal performance drop. This is a useful finding, though the evaluation is limited to a few benchmarks.

**Ablation Studies (Section 5):** These are comprehensive and address the posed research questions. Key findings: two-stage outperforms one-stage BoundlessBPE (Table 12), fertility plateaus after 10G training data (Table 13), transition point of 90% works well (Table 14), and 200K vocabulary is sufficient (Table 15). A minor issue: the choice of 90% transition point is not theoretically justified; it is determined empirically but without a clear criterion (e.g., balancing subword vs. superword coverage).

**Extended Results (Appendix):** Provide full tables for completeness.

### Writing & Clarity
The paper is generally well-written and structured. Some minor clarifications are needed:
- Figure 1 is referenced but not provided in the text (likely a parsing artifact).
- Table 6 (Rényi entropy/efficiency) is mentioned in the text but its location in the flow is slightly awkward.
- The discussion on loss vs. task performance (Appendix C.1) is speculative and should be tempered; it's a hypothesis, not a proven result.
- The description of the glitch token analysis (Section 4.5) could be clearer. It states both tokenizers "share the first 90% of the vocabulary," but later says they share the first 180K IDs (out of 200K). This should be consistent.

### Limitations & Broader Impact
The paper lacks an explicit limitations section. Important limitations to acknowledge include:
- The tokenizer is optimized for Indic languages; its performance on other language families is untested.
- The two-stage approach requires setting a transition point, which may be corpus-dependent.
- The downstream task improvements are small; the primary benefit is efficiency, not accuracy.
- The inference latency gains are demonstrated on a 1B model; scaling to larger models may have different dynamics.
The ethics statement is appropriate, and the reproducibility promise is strong.

### Overall Assessment
This paper presents a carefully engineered tokenizer for Indic multilingual LLMs, achieving state-of-the-art fertility scores and significant inference throughput gains. The work is thorough, with extensive benchmarking and ablations. The main weakness is a lack of algorithmic novelty, as the core two-stage approach is from SuperBPE, and a direct comparison with SuperBPE is missing. The downstream task improvements are marginal and not statistically validated. For ICLR, which emphasizes novel algorithmic contributions, this may be a borderline submission. However, the practical impact (efficiency gains) and the comprehensive empirical study are valuable. If the authors can strengthen the novelty claim (e.g., by showing their adaptations yield clear improvements over vanilla SuperBPE) and add statistical rigor to the downstream evaluation, the paper would be much stronger.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces IndicSuperTokenizer (IST), a tokenizer designed for Indic multilingual LLMs that combines a two-stage curriculum (subword followed by superword/multi-word learning) with language-specific pre-tokenization. The core contribution is achieving state-of-the-art token efficiency (fertility score) across 22 Indian languages, English, and code, which translates to significant inference throughput gains (44% over LLaMA-4) while maintaining competitive downstream task performance.

### Strengths
1.  **Extensive and Rigorous Evaluation:** The paper provides a thorough intrinsic evaluation across 9 baseline tokenizers, 24 languages (22 Indic + English + code), and multiple metrics (fertility, normalized sequence length, bytes per token, Rényi efficiency). The extrinsic evaluation via pretraining and downstream benchmarks (English and Indic) adds practical weight to the claims.
2.  **Comprehensive Ablation Studies:** The paper systematically investigates key design choices through ablations on training data size, vocabulary size, the transition point between subword and superword learning, vocabulary allocation strategies, and normalization, providing a robust empirical foundation for the proposed method.
3.  **Clear Practical Impact:** The demonstrated 44% improvement in inference throughput over a strong baseline (LLaMA-4) is a significant and tangible benefit for model deployment, directly linking tokenizer design to real-world efficiency.
4.  **Strong Baselines and Reproducible Setup:** The paper benchmarks against a wide array of relevant tokenizers (including Sutra, Sarvam, LLaMA-4, Gemma-3) and details training configurations (iso-compute pretraining, CPT experiments), enabling fair comparison and facilitating reproducibility.

### Weaknesses
1.  **Limited Analysis of Linguistic Alignment:** While the tokenizer is motivated by better linguistic alignment (e.g., avoiding morpheme fragmentation), the paper provides only qualitative examples (Figure 1) and fertility scores as evidence. A deeper analysis (e.g., of token boundaries against morphological or syntactic units) is lacking to substantiate claims of improved linguistic faithfulness.
2.  **Incomplete Release of Artifacts:** The evaluation framework and dataset are stated as "will be released," which is a promise rather than a current guarantee of reproducibility. For a conference like ICLR, the availability of these resources at submission time strengthens the contribution.
3.  **Superficial Treatment of Multi-Word Token Trade-offs:** The discussion on why multi-word tokens lead to higher training loss but comparable performance (Appendix C.1) is largely speculative. A more rigorous analysis (e.g., probing the representation space or analyzing gradient norms) could better explain this phenomenon.
4.  **Narrower Downstream Gains:** The improvements on downstream task performance, while positive for Indic benchmarks, are marginal. On English benchmarks, IST performs comparably to the LLaMA-4 tokenizer, which may limit the perceived novelty for a general audience, though the focus is rightly on multilingual efficiency.
5.  **Missed Connection to Morphology Literature:** The paper briefly mentions but dismisses morphology-aware segmentation (Appendix C.2) due to latency, but a more detailed comparison or discussion on how IST's implicit learning relates to explicit morphological analysis (e.g., MorphTok) could provide deeper insights.

### Novelty & Significance
**Novelty:** The primary novelty lies in the careful adaptation and extensive evaluation of a two-stage subword/superword curriculum (inspired by SuperBPE) specifically for the challenging and underexplored domain of Indic languages. The systematic ablations and comparisons (e.g., against BoundlessBPE's one-stage approach) provide new insights into effective multilingual tokenizer design.
**Significance:** The work is highly significant for the multilingual NLP community, particularly for low-resource and morphologically rich languages. It demonstrates that tokenizer optimization is a critical lever for improving both efficiency (inference throughput) and fairness (balanced fertility across languages). The provided "recipe" and strong empirical results offer a valuable blueprint for future multilingual model development.

### Suggestions for Improvement
1.  **Conduct a Linguistic Analysis of Token Boundaries:** Perform a quantitative evaluation of how well IST's tokens align with gold-standard morphological or syntactic segments (using available Indic language tools) to move beyond fertility scores and substantiate claims of linguistic alignment.
2.  **Release Evaluation Framework with Submission:** To fully meet ICLR's reproducibility standards, make the evaluation framework and dataset publicly available at the time of submission (e.g., via anonymous GitHub or supplementary material).
3.  **Deepen the Analysis of Multi-Word Token Effects:** Expand Appendix C.1 beyond hypothesis. For instance, analyze the distribution of superword tokens across languages, their frequency, and their impact on the perplexity of specific syntactic constructs to better explain the loss/performance discrepancy.
4.  **Strengthen the Downstream Evaluation:** Include a wider variety of generative or seq2seq tasks (beyond multiple-choice QA) for Indic languages to assess if efficiency gains come at any cost to generation quality or fluency.
5.  **Clarify the Contribution Relative to SuperBPE:** While the adaptation to Indic languages is clear, the paper should more sharply articulate what specific design choices (e.g., pre-tokenization regex, sentence-boundary constraints, vocabulary allocation) led to improvements over directly applying SuperBPE, and why these are crucial for the Indic context.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to other multi-word tokenizers (e.g., WordPiece, Unigram) and recent methods like BPE-dropout.** Without this, it's unclear if the gains are specific to the two-stage design or are common to any multi-word approach.
2. **Evaluate on cross-lingual transfer tasks (e.g., machine translation, cross-lingual NLI).** The current benchmarks are monolingual; cross-lingual tasks are essential to claim improved multilingual alignment.
3. **Ablation on superword tokens by training a model with only Stage-1 (subwords) and comparing downstream performance.** Table 25 suggests Stage-1 is competitive, but a controlled experiment is needed to isolate the contribution of superwords.
4. **Scale experiments to larger models (e.g., 7B parameters).** Throughput gains and performance claims on a 1B model may not generalize to scales typical of modern LLMs.
5. **Compare to tokenizers trained with a uniform data distribution.** The paper advocates corpus-driven allocation; a uniform distribution would test if the current approach truly mitigates bias against low-resource languages.

### Deeper Analysis Needed (top 3-5 only)
1. **Systematically analyze the linguistic validity of learned superword tokens.** The claim of "linguistically aligned tokens" is supported only by cherry-picked examples; a frequency-based analysis across languages is needed.
2. **Break down fertility scores by word length and morphological complexity.** Aggregate scores may mask whether improvements come from handling complex words or common short words.
3. **Analyze out-of-vocabulary (OOV) behavior on rare or unseen words.** A tokenizer with many multi-word tokens may fragment rare words more; this must be checked.
4. **Study cross-lingual embedding alignment (e.g., via translation pairs).** Without this, claims of "culturally inclusive" and improved multilingual representation are unsubstantiated.
5. **Provide statistical significance testing for latency/throughput results.** Table 5 reports means and standard deviations, but no tests show if differences are reliable across multiple runs and conditions.

### Visualizations & Case Studies
1. **Side-by-side tokenization visualizations for diverse sentences across all languages.** Figure 1 is limited; comparative examples are needed to show where IST succeeds or fails versus baselines.
2. **Case study on a low-resource language (e.g., Santali) with high fertility.** Analyze why fertility remains elevated (3.72) and whether tokenization patterns are linguistically sound.
3. **Scatter plot of vocabulary allocation (per script) versus fertility improvement.** This would visually demonstrate if allocation correlates with fairness gains.
4. **t-SNE visualization of "glitch token" embeddings for IST versus BPE.** This would illustrate the claim that IST utilizes vocabulary more efficiently.
5. **Examples of code tokenization.** The paper includes code in evaluation but does not show how multi-word tokens affect code representation, which could be harmful.

### Obvious Next Steps
1. **Publicly release the tokenizer and evaluation framework.** Reproducibility is critical for ICLR; the paper only promises future release.
2. **Train and evaluate a larger model (e.g., 7B) with IST.** Scaling up is necessary to demonstrate practical utility.
3. **Integrate efficient morphological pre-tokenization for all languages.** The paper mentions latency issues, but optimizing this could further improve linguistic alignment.
4. **Extend the tokenizer to non-Indic languages to test generalizability.** The method may be language-specific; testing on other families would strengthen the contribution.
5. **Evaluate few-shot/zero-shot cross-lingual performance.** This is a key use case for multilingual LLMs and should be assessed.

# Final Consolidated Review
## Summary
This paper introduces IndicSuperTokenizer (IST), an optimized tokenizer for Indic multilingual LLMs that combines a two-stage curriculum (subword then superword learning) with language-specific pre-tokenization. It achieves state-of-the-art fertility scores across 22 Indian languages, English, and code, translating to a 44% inference throughput improvement over LLaMA-4 while maintaining competitive downstream task performance. The work is supported by extensive ablations on design choices.

## Strengths
- **Comprehensive and rigorous empirical evaluation.** The paper benchmarks against 9 strong baselines across 24 languages using multiple intrinsic metrics (fertility, normalized sequence length, bytes per token, Rényi efficiency) and provides extrinsic evaluation via pretrained models on English and Indic benchmarks. The ablations systematically investigate training data size, vocabulary size, transition point, and allocation strategies, forming a robust empirical foundation.
- **Clear and significant practical impact.** The tokenizer delivers a 44% improvement in inference throughput (Table 5), a tangible deployment benefit directly linked to reduced sequence lengths. This efficiency gain is demonstrated in a controlled, iso-compute pretraining setup and in a continual pretraining experiment, showing the tokenizer's utility in both from-scratch and adaptation scenarios.
- **Valuable methodological recipe and insights.** The paper provides a clear, empirically-grounded recipe for multilingual tokenizer design, including the superiority of corpus-driven vocabulary allocation over explicit merging (Section 4.4), the effectiveness of a two-stage over a one-stage approach for Indic languages (Table 12), and the plateauing of gains with data scaling (Table 13). The analysis of "glitch tokens" also offers an interesting, data-driven perspective on vocabulary utilization.

## Weaknesses
- **Marginal and statistically unverified downstream performance gains.** While the tokenizer achieves large efficiency improvements, its impact on downstream task accuracy is small (e.g., Indic benchmark average improves from 0.388 to 0.394 in Table 8). The paper does not establish the statistical significance of these differences, which weakens the claim of "maintaining comparable performance." The improvements, while positive, are not a primary driver of the contribution.
- **Limited substantiation of improved linguistic alignment.** The core motivation includes producing more "linguistically aligned tokens," but evidence is primarily indirect (improved fertility scores) and qualitative (Figure 1). A quantitative analysis of how well token boundaries align with morphological or syntactic units across languages is missing, leaving an important claim partially unverified.
- **Artifacts are promised but not yet released.** The evaluation framework and dataset are stated as "will be released." For full reproducibility and impact, availability at the time of decision would strengthen the contribution, though a commitment is standard and acceptable.

## Nice-to-Haves
- A deeper analysis of the trade-offs introduced by superword tokens (e.g., their frequency distribution, impact on the representation of rare words, or a more rigorous investigation of the loss/performance discrepancy noted in Appendix C.1).
- Evaluation on a broader set of generative or sequence-to-sequence tasks for Indic languages to assess any potential impact on generation quality beyond multiple-choice QA.
- Testing the generalizability of the approach on a few non-Indic, morphologically rich languages to suggest broader applicability.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **"Missing comparison with SuperBPE baseline."** The paper's contribution is the adaptation and optimization of the two-stage curriculum (from SuperBPE) for Indic languages, with novel modifications (LLaMA-4 regex, sentence-boundary constraints, corpus-driven allocation for 22 languages). A direct comparison with an off-the-shelf SuperBPE is not required to establish the value of this targeted, engineered solution.
- **"Lack of statistical significance for latency results."** The paper reports standard deviations for Time-To-First-Token (Table 5: ± 0.36 ms), and the throughput difference is large (117.99 vs. 169.42 tokens/s). The results are clearly robust.
- **"Need for cross-lingual transfer task evaluation."** The paper's scope is designing an efficient tokenizer for Indic LLMs, evaluated via standard monolingual benchmarks and inference efficiency. Demanding cross-lingual task evaluation is scope creep.
- **"Requirement to scale experiments to larger (e.g., 7B) models."** The paper demonstrates principles on a 1B model, which is a standard and valid experimental scale for tokenizer ablation. The efficiency gains are rooted in sequence length reduction, a effect that should generalize.
- **"Need to compare against WordPiece, Unigram, or BPE-dropout."** The chosen baselines (LLaMA-4, Sutra, Gemma-3, etc.) are state-of-the-art, widely-used tokenizers for multilingual LLMs. Adding historical or less common algorithms is not necessary for establishing SOTA.

## Novel Insights
The paper provides a systematic, data-driven blueprint for building equitable multilingual tokenizers, moving beyond isolated algorithmic proposals. Key novel insights include: 1) For Indic languages, a two-stage curriculum (subword then superword) outperforms a single-stage, boundary-relaxed approach (BoundlessBPE), preserving linguistic integrity while capturing multi-word expressions. 2) Corpus-driven vocabulary allocation—simply training on a concatenated multilingual corpus—is more effective than the modular approach of training and merging script-specific tokenizers, which introduces distributional interference. 3) Tokenizer optimization can significantly reduce the number of "glitch" (under-trained) tokens in a model's embedding space, suggesting multi-word tokens promote more efficient vocabulary utilization than purely subword-based expansion.

## Suggestions
- Include a quantitative analysis of token boundary alignment, perhaps using available Indic morphological analyzers on a sample, to substantiate claims of improved linguistic faithfulness beyond compression metrics.
- To bolster the downstream evaluation, report statistical significance tests (e.g., confidence intervals via bootstrapping) for the benchmark results, especially for the marginal gains on Indic tasks.
- If possible, release the evaluation framework and dataset as supplementary material or via an anonymous repository to immediately enable reproducibility and benchmarking.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
