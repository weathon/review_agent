=== CALIBRATION EXAMPLE 29 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: an optimized tokenizer for Indic multilingual LLMs. The abstract clearly states the problem (inefficient tokenization for morphologically rich Indic languages), the proposed solution (IndicSuperTokenizer combining subword and multi-word tokenization), and the key results (SOTA fertility scores, 44% inference throughput improvement). All claims are supported in the main paper. One minor note: the abstract mentions "44% improvement in inference throughput over LLaMA4," but Table 5 reports Output Throughput (OTPT) in tokens/sec, not a direct percentage improvement. The 44% figure likely comes from (169.42 - 117.99)/117.99 ≈ 43.6%, which should be explicitly stated or clarified.

### Introduction & Motivation
The introduction effectively motivates the problem. It highlights the specific challenges of Indic languages (diverse scripts, rich morphology) and the unfairness/inefficiency caused by high fertility scores from existing tokenizers (e.g., LLaMA-4's score of 10.5 for Oriya). The five research questions are well-posed and guide the paper's investigations. The contributions are clearly listed and match the content of the paper.

### Related Work
This section adequately covers relevant tokenization algorithms (BPE variants), multilingual tokenizers, and pre-tokenization strategies. It correctly positions the work relative to SuperBPE (two-stage curriculum) and BoundlessBPE (one-stage, relaxed constraints). A more critical discussion of how *IndicSuperTokenizer* differs from SuperBPE beyond the pre-tokenization regex change would be beneficial. The claim of being "the first to carry out a comprehensive benchmarking... in both pretraining from scratch as well as continual pretraining settings" (Section 1) is strong but plausible given the focused Indic context.

### Method / Approach (Section 3)
The two-stage curriculum (subword then superword learning) is clearly described, following Liu et al. (2025a). The use of LLaMA-4 regex for Stage 1 pre-tokenization is a specific design choice justified by results in Table 1. The sentence-level boundary constraint to prevent cross-sentence merges is a sensible addition.

**Concerns & Questions:**
1.  **Reproducibility Gap:** The exact LLaMA-4 regex pattern is not provided. Stating "we replace GPT-2 rules with LLaMA-4 regex" is insufficient for replication. This must be included in the appendix or code.
2.  **Transition Point Justification:** The choice of the transition point `t` (90% of vocab) is later ablated (Table 14), but the initial rationale for this design is missing from the method description. Why start with 90%? Is it based on pilot studies or prior work?
3.  **Vocabulary Allocation:** Section 3.3 and Figure 2 mention a vocabulary "distributed across language scripts," but the method for achieving this distribution during the *corpus-driven alignment* is unclear. Is it achieved solely by the data mix (Table 10), or are there explicit controls during BPE merges?
4.  **Superword Semantic Faithfulness:** The method assumes that cross-word merges in Stage 2 capture semantically meaningful "superwords" (collocations, idioms). While the fertility score improvement supports efficiency, an analysis of the types of superwords learned (e.g., are they truly multi-word expressions like "wake up" or arbitrary frequent character sequences?) would strengthen the linguistic alignment claim.

### Experiments & Results (Section 4 & 5)
The evaluation is extensive, covering intrinsic metrics, downstream task performance, inference latency, and detailed ablations. The evaluation framework and dataset (22 Indic languages, English, code) are appropriate and will be a valuable contribution if released.

**Concerns & Questions:**
1.  **Baseline Completeness & Fairness:** Table 22 reveals that training details for many baselines (DeepSeek-R1, GPT-OSS, LLaMA-3.2, etc.) are "Not publicly disclosed." This is a common challenge. The authors should explicitly discuss the difficulty of perfectly controlled comparisons and focus on the demonstrated relative efficiency gains. The inclusion of Sutra and Sarvam, which are Indic-focused, is good.
2.  **Statistical Significance:** Results for inference latency (Table 5) include standard deviation for TTFT but not for OTPT. For downstream tasks (Tables 8, 11, 25), no measures of variance or statistical significance are reported. Given the small performance differences (e.g., English average of 0.279 for both LLaMA-4 and IST), it's crucial to indicate whether these differences are meaningful. Multiple runs with confidence intervals or significance tests are needed for ICLR.
3.  **Downstream Performance Interpretation:** The extrinsic results show that IST is *comparable* to the LLaMA-4 tokenizer on English/Indic benchmarks, not uniformly better. This is honestly reported but requires more nuanced discussion. The authors correctly argue that the key benefit is efficiency (inference throughput) without sacrificing quality. This trade-off should be emphasized in the main results discussion.
4.  **Inference Latency Setup:** Section 4.3 and Appendix C.4 describe the latency test. It is good that input sequences are matched by content. However, the sentence "Latency was measured using standard metrics..." is vague. A citation or brief definition of TTFT and OTPT in the main text would improve clarity. The 44% throughput gain is impressive, but it would be useful to correlate this directly with the reduction in average sequence length (from 784 to 379 tokens in Table 21).
5.  **Ablation Clarity:**
    *   Table 12 (Two-Stage vs. One-Stage): The one-stage variant "IST-BR" uses BoundlessBPE's regex. The results show very similar fertility for most languages, with IST winning slightly. The authors claim one-stage can "overfit to arbitrary character sequences," but no evidence is provided to support this claim in the comparison. What is the qualitative difference in the learned tokens?
    *   Table 13 (Data Size): The plateau at 10G is interesting. Was this 10G used only for Stage 1? The caption says "only in Stage 1," but the section header should make this clearer.
    *   "Glitch Token" Analysis (4.5): This is a fascinating analysis. However, the description is dense. A clearer explanation of the reference vector construction and what Figure 5 demonstrates (IST has fewer under-trained tokens in the tail) would help. The conclusion that multi-word tokens promote efficient vocabulary utilization is plausible but should be framed as a hypothesis supported by this observation.

### Writing & Clarity
The paper is generally well-written and logically structured. Some sections are information-dense, requiring careful reading (e.g., the glitch token analysis). A few specific clarifications are needed:
*   Figure 1: The examples are helpful, but the caption is cut off ("see for e.g. Bengali, Tamil"). A complete caption describing what the figure shows for each language would be better.
*   Section 4.4: The terms *explicit merging* and *corpus-driven alignment* are defined, but the flow could be smoother. Consider a brief summarizing sentence at the start of the paragraph.
*   Appendix C.1: The discussion on loss vs. task performance is insightful and should be integrated into the main paper's discussion of downstream results, as it helps explain why comparable accuracy is a win given the efficiency gains.

### Limitations & Broader Impact
The ethics statement appropriately addresses data sources and bias mitigation. The reproducibility statement promises the release of code, framework, and datasets, which is excellent.

**Key Limitations Missing:**
1.  **Generalizability Beyond Indic:** The tokenizer is optimized for a specific set of languages with particular scripts and morphological features. Its effectiveness for other multilingual contexts (e.g., agglutinative languages like Turkish or Finnish, or non-alphabetic scripts) is not discussed. A brief discussion of the scope of applicability is needed.
2.  **Potential Negative Side Effects of Superwords:** While superwords improve fertility, they could potentially harm compositional generalization. If "in the morning" is a single token, does the model struggle to understand "in the afternoon" by analogy? Acknowledging this potential trade-off would strengthen the paper.
3.  **Computational Cost of Tokenizer Training:** The two-stage curriculum and ablations over data size (up to 50G) imply non-trivial training costs. A brief comment on the computational budget for tokenizer training itself would provide a fuller picture.

### Overall Assessment
This is a strong, practical paper that makes a clear contribution. It provides a well-engineered tokenizer (IndicSuperTokenizer) with comprehensive empirical evidence showing significant improvements in efficiency (fertility, inference throughput) for Indic languages while maintaining downstream task performance comparable to established baselines. The systematic ablations are a major strength. For acceptance at ICLR, the authors must address the core issues of **reproducibility (providing the exact regex)**, **statistical rigor (significance testing for downstream results)**, and **clarity in the ablation interpretations**. Adding a discussion on limitations, particularly generalizability and potential downsides of superwords, would also elevate the paper. If these concerns are adequately addressed in a revision, the paper represents a solid contribution suitable for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces IndicSuperTokenizer (IST), a tokenizer designed for multilingual LLMs focusing on 22 Indian languages, English, and code. Its core contribution is a two-stage curriculum (subword then superword learning) combined with script-aware pre-tokenization and a corpus-driven vocabulary allocation strategy. The authors demonstrate state-of-the-art performance on intrinsic metrics like fertility score and show improved inference throughput, while maintaining competitive downstream task performance in both pretraining-from-scratch and continual pretraining settings.

### Strengths
1. **Comprehensive and Rigorous Evaluation**: The paper evaluates IST across 24 languages using multiple intrinsic metrics (fertility, NSL, Rényi efficiency, bytes-per-token) and compares against 9 strong baselines, including recent models like LLaMA-4 and Sutra. The extrinsic evaluation on downstream tasks (English and Indic benchmarks) and detailed inference latency analysis provide a well-rounded assessment.
2. **Extensive Ablation Studies**: The authors conduct systematic ablations on critical design choices, including tokenizer training data size, vocabulary size, transition point between stages, normalization effects, and vocabulary allocation strategies (explicit vs. corpus-driven). This provides strong empirical grounding for their design decisions.
3. **Practical Impact and Reproducibility**: The reported 44% improvement in inference throughput over LLaMA-4 is a significant practical gain. The authors commit to releasing their evaluation framework, dataset, and tokenizer, which enhances reproducibility and utility for the community.
4. **Addressing a Clear Gap**: The work effectively highlights and addresses the pronounced tokenization inefficiencies (e.g., fertility scores as high as 10.5) for Indic languages in existing multilingual tokenizers, framing it as an issue of fairness and computational efficiency.

### Weaknesses
1. **Incomplete State-of-the-Art Claims**: While IST achieves the best average fertility, Table 3 shows it does not outperform the best baseline (Sutra) in every single language (e.g., for Santali (sat), IST scores 3.72 vs. Sutra's 2.03). The paper would benefit from a discussion of these specific outliers and the factors behind them.
2. **Under-Explained Phenomena**: The hypothesis that IST's slightly higher training loss (mentioned in Appendix C.1) does not hurt downstream performance due to a "less sharply peaked" prediction space is interesting but not substantiated with analysis (e.g., by examining per-token probability distributions or calibration). This remains speculative.
3. **Limited Exploration of Morphological Integration**: The paper discusses but does not integrate morphology-aware pre-tokenization due to latency concerns (Appendix C.2). While the latency trade-off is valid, this feels like a missed opportunity to explore a more linguistically grounded method, especially for morphologically rich languages. A hybrid or efficient approximation could have been considered.
4. **Superficial Continual Pretraining Analysis**: The continual pretraining experiment (Section 4.6) shows that replacing a pre-trained model's tokenizer with IST works reasonably well. However, the results in Table 11 show a slight performance drop on several benchmarks (e.g., Indic XNLI, Indic Paraphrase). This trade-off between efficiency gains and potential quality erosion is not deeply analyzed.

### Novelty & Significance
The paper's novelty lies in the systematic adaptation and rigorous evaluation of a two-stage subword/superword tokenization curriculum (inspired by SuperBPE) for the specific and challenging context of Indic languages. The incorporation of script-specific pre-tokenization (LLaMA-4 regex) and detailed investigation of vocabulary allocation strategies for multilingual fairness are valuable contributions. The significance is high for the multilingual NLP community, as it provides a proven recipe for building more efficient and equitable tokenizers for linguistically diverse regions, with demonstrated real-world benefits in inference speed. The work meets ICLR's emphasis on technically sound, impactful research with clear empirical results.

### Suggestions for Improvement
1. **Analyze Performance Variations**: Provide a focused analysis explaining why IST underperforms relative to Sutra in specific languages like Santali and Sindhi. Is it related to training data quantity, script complexity, or the superword learning stage?
2. **Deepen the Loss-Performance Discussion**: To strengthen the argument in Appendix C.1, provide empirical evidence. For example, analyze the entropy or variance of the next-token probability distributions for IST vs. a standard BPE tokenizer on a sample corpus.
3. **Expand Future Work on Morphology**: Propose concrete directions for efficiently integrating morphological analysis, such as using fast, approximate segmenters or learning morphological rules during tokenizer training, to bridge the gap between linguistic insight and computational efficiency.
4. **Refine the Continual Pretraining Narrative**: Discuss the observed slight performance drops in Table 11 more explicitly. A sensitivity analysis (e.g., varying the amount of continual pretraining data) could help understand the conditions under which tokenizer replacement is most effective.
5. **Justify Hyperparameter Choices**: While ablations are provided, add a brief rationale for the final chosen vocabulary size (200K) and transition point (90%), explaining why these points represent the best balance on the trade-off curves observed.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Train and evaluate models at larger scales (e.g., 7B, 13B).** The paper only uses 1B models, but tokenizer efficiency claims for LLMs are not credible without validation on common LLM scales where embedding and softmax costs matter more.
2. **Include a direct, controlled comparison with SuperBPE and BoundlessBPE as primary baselines.** The paper relegates BoundlessBPE to an ablation (IST-BR) and does not compare with SuperBPE in the main tables. To claim SOTA fertility, these recent, directly relevant methods must be compared head-to-head with identical training data and vocabulary size.
3. **Evaluate on a broader set of non-Indic, morphologically rich languages (e.g., Turkish, Finnish, Arabic).** The paper claims a general recipe for multilingual tokenizers but only tests on Indic languages and English. Without validation on other language families, the generality of the method is unsupported.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the per-language downstream task performance, not just averages.** The aggregated Indic benchmark average hides potential regressions in specific languages. To claim equitable improvements, show per-language results (especially for low-resource ones) and correlate them with fertility gains.
2. **Quantify the semantic quality and utility of learned superwords.** The paper claims superwords capture meaningful multi-word expressions but provides no analysis of their types, frequencies, or impact on language modeling (e.g., via surprisal). Without this, the benefit over arbitrary character sequences is unclear.
3. **Provide a deeper analysis of the trade-off between fertility and model performance.** The paper notes competitive average scores but does not investigate why lower fertility sometimes correlates with slightly worse performance (e.g., on MMLU, DROP). This is critical to understand the limits of compression.

### Visualizations & Case Studies
1. **Show side-by-side tokenization examples for complex, morphologically rich sentences across all baselines.** The single example in Figure 1 is insufficient. To demonstrate linguistic alignment, visualize token boundaries for sentences that expose fragmentation issues in baseline tokenizers and show how IST improves.
2. **Visualize the distribution of superword lengths and their coverage across languages.** A histogram showing the length (in words) of learned superwords per language would reveal whether the method captures idiomatic phrases or just frequent short function word sequences.

### Obvious Next Steps
1. **Release the tokenizer and evaluation framework code with the paper.** The reproducibility statement promises release, but for a paper claiming a new SOTA tokenizer and benchmark, the artifacts are essential for validation and adoption. The lack of immediate availability severely undermines impact.
2. **Conduct an ablation on the data mixing ratio for vocabulary allocation.** The paper uses a corpus-driven allocation but does not systematically ablate the effect of different language sampling ratios on fertility and performance. This is a core design choice for multilingual tokenizers that remains unguided.

# Final Consolidated Review
## Summary
This paper introduces IndicSuperTokenizer, a tokenizer optimized for 22 Indian languages, English, and code. It combines a two-stage subword-superword learning curriculum with script-aware pre-tokenization, achieving state-of-the-art average fertility scores and a 44% improvement in inference throughput over LLaMA-4 while maintaining comparable downstream task performance.

## Strengths
- Achieves superior compression efficiency across 22 Indic languages, with consistent improvements in fertility score, normalized sequence length, and bytes-per-token over 9 strong baselines, including recent models like LLaMA-4 and Sutra (Tables 3, 4, 7).
- Empirically grounds design choices through extensive ablations on training data size, vocabulary size, transition point, and vocabulary allocation strategies, providing a reproducible recipe for multilingual tokenizer training (Section 5, Tables 12-16).
- Shows practical impact with significant inference throughput gains (44% over LLaMA-4) and maintains competitive accuracy on English and Indic benchmarks, validating utility in both pretraining-from-scratch and continual pretraining settings (Tables 5, 8, 11).

## Weaknesses
- **Reproducibility gap:** The exact LLaMA-4 regex pattern used for pre-tokenization is not provided, hindering replication of the tokenizer training.
- **Lack of semantic analysis for superwords:** While superword learning improves fertility, no evidence is presented to show that the learned multi-word tokens are semantically meaningful rather than arbitrary frequent sequences, undermining the claim of linguistic alignment.
- **Incomplete dominance in fertility scores:** IST does not outperform the best baseline (Sutra) in every language, notably for Santali (sat) where IST scores 3.72 vs. Sutra's 2.03 (Table 3), without explanation for this outlier.
- **Limited statistical validation for downstream results:** Downstream task performances are reported as single scores without variance measures or significance tests, making it difficult to assess whether the small differences (e.g., English average of 0.279 vs. 0.279) are meaningful.

## Nice-to-Haves
- Analysis of generalizability to non-Indic, morphologically rich languages.
- Evaluation with larger-scale models (e.g., 7B parameters) to confirm efficiency gains at common LLM sizes.
- Deeper investigation into the trade-off between fertility and model performance, such as per-language downstream task breakdowns.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Include the LLaMA-4 regex pattern in the appendix or released code to ensure reproducibility.
- Conduct a qualitative analysis of learned superwords (e.g., by sampling and categorizing them) to validate their semantic utility.
- Provide a brief discussion of the Santali outlier case, exploring potential reasons based on data or script characteristics.
- Add confidence intervals or multiple-run statistics for downstream task evaluations to strengthen the robustness of the performance claims.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
