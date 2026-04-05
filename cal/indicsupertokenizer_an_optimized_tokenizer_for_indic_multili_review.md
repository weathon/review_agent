=== CALIBRATION EXAMPLE 22 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy:** The title accurately reflects the paper's focus. The abstract clearly states the problem (inefficient tokenization for Indic languages), the method (two-stage subword–superword learning with optimized pre-tokenization), and key results (improved fertility, throughput, and competitive downstream performance).
- **Supported Claims:** The claim of a "44% improvement in inference throughput" is supported by Table 5 (Output Throughput: 169.42 vs 117.99). The "39.5% improvement in average fertility score over LLaMA-4" appears supported by the tables, though the abstract should specify this is an average across the evaluated languages, as individual language gains vary.
- **Concerns:** The abstract implies a uniform gain in performance, but downstream metrics in Section 4.2 show the IST and LLaMA-4 tokenizers achieve nearly identical average English scores (0.279 vs 0.279). The abstract mentions "maintaining comparable performance," which is fair, but the phrasing "leading to more linguistically aligned tokens" implies a quality gain that is not strongly borne out by the downstream numbers, only the efficiency metrics.

### Introduction & Motivation
- **Motivation:** The problem is well-motivated. The fertility disparity between English and Indic languages in existing models is a genuine bottleneck for cost and fairness in the Global South.
- **Gap:** The gap is clearly identified: standard subword methods struggle with Indic morphology and script diversity, and superword techniques are underexplored in this context.
- **Contributions & Claims:** Contributions are clearly listed. However, the claim *"To the best of our knowledge, we are the first to carry out a comprehensive benchmarking... in both pretraining from scratch as well as continual pretraining settings"* is likely overstated. Works like Dagan et al. (2024), ReTok (Gu et al., 2024), and others have studied tokenizer replacement, pretraining, and CPT extensively, even if not specifically for the full Indic spectrum. The novelty here is the **Indic-specific systematic study and the superword adaptation**, not the benchmarking paradigm itself. The authors should calibrate this claim to emphasize the Indic/superword novelty rather than the benchmarking novelty.
- **Research Questions:** The five questions are useful, but the introduction promises answers that are only partially delivered. For example, Question (i) asks how to improve low-resource without degrading high-resource. While Table 3 shows low fertility for English, there is no explicit discussion of the *trade-off curve* or whether improving Indic fertility inherently biases the vocabulary away from English, other than the aggregate tables.

### Method / Approach
- **Clarity & Reproducibility:** The two-stage curriculum is well-described and generally reproducible. The use of NFKC normalization and sentence-aware boundary constraints are strong practical choices.
- **Assumptions & Justification:** The method assumes whitespace tokenization is sufficient for Stage 1 pre-tokenization across all Indic languages. While most Indic languages use whitespace, some edge cases or historical texts may not. NFKC handles character composition, which is excellent.
- **Logical Gaps & Implementation Details:**
  - **Sentence-Level Boundary Constraints:** Section 3.2 states, *"merges are free within sentences but are disallowed across sentence delimiters."* It is unclear how this is implemented algorithmically. Does the training script filter merge candidates that cross sentence IDs? Does it inject a special end-of-sentence marker? This detail is critical for reproduction, especially when handling noisy web text where sentence boundaries might be inconsistent.
  - **Transition Point:** The choice of a 90% transition point is shown to be effective via ablation, but the intuition is thin. Why does reserving only the top 10% of the vocabulary for superwords work best? The paper mentions that early transitions weaken morphological coverage; expanding on how the 90% threshold preserves the morphology-frequency balance would strengthen the reasoning.
  - **Code Data Ambiguity:** Section 3.3 and Table 2 mention "code" data. It is unclear if this refers to programming code or code-mixed text (e.g., Hinglish). Programming code has very different tokenization properties than natural language code-switching. Clarification is needed.

### Experiments & Results
- **Testing Claims:** The experiments effectively test efficiency claims (fertility, throughput) and provide adequate downstream testing for a tokenizer paper.
- **Baselines:** The baselines are strong and contemporary (LLaMA-4, Sutra, Gemma-3, GPT-OSS). The inclusion of BoundlessBPE (IST-BR) in ablations is excellent for isolating the value of the two-stage approach.
- **Missing Ablations/Statistical Rigor:**
  - **Evaluation Data Leakage:** Section 3.6 states the evaluation set is *"curated from the same sources as the training corpus."* This is a significant methodological concern. Intrinsic metrics like fertility are relatively robust to this, but evaluating a tokenizer on data from its own training distribution risks inflating performance due to corpus-specific frequency statistics and potential near-duplicate leakage. **The authors must evaluate on a held-out test set from completely disjoint sources (e.g., Wikipedia held out from training, or distinct Common Crawl snapshots) to validate generalization.**
  - **Downstream Variance:** Table 8 reports single-point estimates for 1B models. 1B models trained for ~50B tokens can exhibit significant variance based on seed and data ordering. To claim "competitive performance" with confidence, the authors should report standard deviations over multiple seeds or runs, or at least acknowledge this variance.
  - **Latency Analysis Nuance:** Table 5 shows a massive gain in Output Throughput (44%) but a negligible gain in TTFT (19.17 vs 18.98 ms). This is actually a correct and expected result: TTFT is compute-bound during prefill (sensitive to batch size and model size), while OTPT is memory-bound and sensitive to sequence length during autoregressive decoding. The paper should explicitly attribute the gain to the decode phase, showing a deeper understanding of the LLM efficiency stack.
- **Results Support:** The intrinsic results strongly support the SOTA fertility claims. The downstream results support the "comparable performance" claim but do not show that the new tokenizer *improves* model quality, only that it maintains it at lower cost.

### Writing & Clarity
- **Clarity:** The writing is generally clear and well-structured. The progression from intrinsic metrics to downstream to ablations is logical.
- **Table/Figure Issues:** Due to parser artifacts, some tables are fragmented, but the content is recoverable. Figure 1 is referenced for examples of avoiding fragmentation, which is helpful context.
- **Potential Confusion:** In Section C.1, the paper hypothesizes why loss might be higher for tokenizer models despite good downstream performance. While this is a valuable discussion, it reads more like a post-hoc justification than a proven mechanism. The link between "semantically overlapping candidates" and "inflated cross-entropy" is intuitive but lacks empirical evidence (e.g., analyzing prediction distributions). This section should be framed carefully as a hypothesis rather than a confirmed finding.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors acknowledge the latency overhead of morphology-aware tokenization (Section C.2) and dismiss it based on engineering complexity. This is a valid scope decision.
- **Missed Limitations:**
  - **Domain Shift:** Superword tokenizers can be brittle to domain shifts. If "in the morning" is a single token, the model might struggle with low-frequency collocations or domain-specific jargon not seen during tokenizer training. The paper should discuss whether the superword vocabulary is frozen or if it generalizes well to unseen domains (e.g., biomedical or legal text where "in the morning" might be irrelevant).
  - **Evaluation Leakage:** As noted, using training-adjacent data for evaluation is a limitation that is not discussed in the Limitations section.
  - **Scale Dependence:** The downstream evaluation is limited to 1B models. It is unclear if the fertility benefits translate linearly to larger models (e.g., 8B or 70B) where context efficiency and long-range dependencies might benefit differently from superwords.
- **Broader Impact:** The ethics statement focuses on responsible development and avoiding PII, which is standard. The paper effectively argues that efficient tokenizers reduce compute costs, which has a positive societal impact for democratizing Indic LLMs. This is a strong point.

### Overall Assessment
This paper presents a solid, engineering-focused contribution to the field of multilingual tokenization, specifically targeting the under-served Indic language spectrum. The systematic combination of NFKC normalization, script-aware regex, and a two-stage subword–superword curriculum yields compelling intrinsic efficiency gains (fertility and throughput) over state-of-the-art baselines. The extensive ablations on vocab size, transition points, and data size are highly valuable for practitioners.

However, the paper has notable weaknesses that must be addressed. The most critical is the **evaluation data provenance**: curating the evaluation set from the same sources as the training corpus introduces a risk of distributional bias or leakage, potentially inflating the reported efficiency gains. The authors must validate their findings on a strictly held-out, disjoint test set. Additionally, the downstream evaluation relies on 1B models without variance reporting; while the "comparable performance" claim is likely true, the lack of statistical rigor weakens the evidence. The claim of being the "first comprehensive benchmarking" is overstated given prior work on tokenizer analysis and CPT.

Despite these issues, the core contribution—a robust, efficient tokenizer that significantly improves inference throughput for Indic models without degrading quality—is valuable and aligns well with ICLR's interest in scalable and equitable ML infrastructure. If the authors can address the evaluation data concerns and temper the benchmarking novelty claims, this paper represents a strong addition to the literature.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces IndicSuperTokenizer (IST), a tokenization framework optimized for multilingual Large Language Models with a focus on 22 Indic languages, English, and code. By combining Unicode-aware, script-specific pre-tokenization with a two-stage subword-to-superword learning curriculum, IST achieves state-of-the-art intrinsic efficiency metrics, notably reducing fertility scores by ~39.5% compared to LLaMA-4 and improving inference throughput by 44% for a 1B-parameter model. Through extensive ablations and downstream evaluations, the work provides a systematic, empirically validated recipe for building multilingual tokenizers that balance computational efficiency with linguistic fidelity.

### Strengths
1. **Comprehensive Multi-Dimensional Evaluation:** The paper rigorously evaluates IST across intrinsic metrics (fertility, NSL, Rényi entropy, bytes-per-token) and extrinsic metrics (12 downstream benchmarks, latency/throughput), demonstrating consistent gains over 9 strong baselines. Evidence: Tables 3, 4, 6, and 7 show IST outperforms competitors in 20-23 of 24 language/code settings.
2. **Systematic & Data-Driven Ablations:** The authors conduct thorough ablations on critical design choices, including one-stage vs. two-stage learning, transition points, vocabulary scaling, dataset size, and vocabulary allocation strategies. Evidence: Tables 12-15 clearly justify the 90% transition point and 200K vocabulary cap, demonstrating diminishing returns beyond 10GB of training data.
3. **Tangible Efficiency & Fairness Gains:** The work directly addresses a recognized pain point in multilingual NLP (computational inequity due to high fertility in morphologically rich scripts), translating intrinsic improvements into real-world deployment benefits. Evidence: Table 5 reports a 44% increase in output throughput (169.42 vs 117.99 tokens/sec) without degrading task accuracy.
4. **Strong Commitment to Reproducibility & Open Science:** The paper provides detailed implementation protocols, training data distributions, hyperparameter tables, and metric definitions, alongside a commitment to release the evaluation framework and dataset. Evidence: Appendices B & D, and the Ethics/Reproducibility Statement explicitly outline experimental setups and artifact release plans.

### Weaknesses
1. **Marginal & Statistically Unverified Downstream Gains:** While intrinsic metrics improve substantially, downstream performance across English and Indic benchmarks shows only marginal parity, with occasional slight regressions (e.g., Indic XNLI drops 0.347→0.346, MILU drops 0.261→0.258 in Table 8 & 11). The paper lacks confidence intervals or significance testing to confirm these variations are not detrimental.
2. **Opaque Baseline Training Regimes Limit Fairness Claims:** Fertility scores are highly dependent on training data quality and distribution. The baselines (LLaMA-4, Gemma-3, GPT-OSS) have undisclosed training corpora and tokenizer parameters (noted in Table 22), making it difficult to isolate algorithmic improvements from data regime advantages.
3. **Reliance on Undocumented References:** The frequent benchmarking against a "LLaMA-4" tokenizer lacks a public, citable technical specification or official tokenizer release. This ambiguity hinders exact replication and raises questions about the validity of the primary baseline comparison.
4. **Limited Real-World Linguistic Stress Testing:** The evaluation focuses on monolingual text per language. Indic NLP in practice heavily involves code-mixing, transliteration (e.g., Roman-script Indic words), and dialectal variation. The tokenizer's robustness to these prevalent real-world conditions is unexplored.

### Novelty & Significance
**Novelty:** The algorithmic novelty is incremental rather than foundational. The two-stage curriculum and regex-based pre-tokenization are direct adaptations of SuperBPE and BoundlessBPE. However, the novelty lies in the careful constraint design (sentence-boundary guarding against cross-sentence merges), the systematic empirical recipe for morphologically rich scripts, and the explicit rejection of explicit script-merging in favor of corpus-driven alignment. 
**Clarity:** The paper is well-organized, with a logical flow from problem motivation to methodology, results, and ablations. Tables are clearly referenced, and metric definitions in the appendix are thorough. Minor issues include occasional typos ("reffered") and dense table formatting, but these do not impede comprehension.
**Reproducibility:** Strong. The inclusion of a detailed evaluation framework promise, explicit hyperparameter listings (Appendix Table 19), data mix breakdowns (Table 20), and clear baseline documentation efforts (Table 22) aligns well with community standards. Releasing the code/data will significantly elevate this score.
**Significance:** High practical significance. The demonstrated reduction in token counts directly translates to lower training FLOPs, reduced inference latency, and improved cost-efficiency for multilingual deployments. For ICLR, the work meets the bar for solid empirical rigor and addresses a critical gap in equitable multilingual AI, though the lack of fundamental algorithmic breakthrough and marginal task-level improvements position it as a valuable applied contribution rather than a paradigm-shifting theoretical advance.

### Suggestions for Improvement
1. **Add Statistical Rigor to Downstream Comparisons:** Include confidence intervals (e.g., via bootstrapping) or statistical significance tests for the benchmark results in Tables 8 and 11 to substantiate the "maintaining comparable performance" claim, particularly where minor score drops are observed.
2. **Clarify Baseline Data Parity:** Explicitly acknowledge the training data mismatch or, ideally, fine-tune/retrain a key open-source baseline (e.g., Sutra or a LLaMA-3 variant) on the exact same 10GB/50B-token corpus used for IST. This would isolate the tokenizer algorithm's contribution from dataset quality effects.
3. **Evaluate Code-Mixing & Transliteration Robustness:** Introduce a stress-test subset containing realistic code-mixed (e.g., Hinglish) and transliterated text. Report fertility and decoding behavior for these scenarios, as they represent a major fraction of real-world Indic NLP usage and are notoriously challenging for script-segmented tokenizers.
4. **Replace or Clearly Cite the LLaMA-4 Baseline:** If "LLaMA-4" refers to an internal/pre-release or unofficial tokenizer, swap it for an officially documented baseline (e.g., LLaMA-3.1 or Mistral-Nemo) or provide a direct link/citation to its exact specification. This is crucial for reviewer and community verification.
5. **Contextualize Throughput Gains for Larger Models:** Table 5 demonstrates gains on a 1B model. Add a brief discussion or small-scale experiment analyzing whether the 44% throughput improvement scales linearly to 7B/13B+ models, where inference often bottlenecks on memory bandwidth (KV cache) rather than tokenization compute.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Scale throughput and latency evaluation to larger models (≥3B), longer generation horizons (≥512 tokens), and diverse decoding strategies; without this, the claimed 44% inference efficiency gain is not convincing beyond a narrow, unrepresentative 1B/256-token setup.
2. Evaluate downstream performance on morphology-aware and realistic code-mixed benchmarks (translation, code-switching, multi-turn instruction following) to prove that super-tokenization preserves fine-grained reasoning, otherwise the compression likely masks degraded linguistic capability for mixed-script inputs.
3. Report training dynamics (loss, gradient norms, validation perplexity) normalized by raw input bytes rather than token counts; without this byte-aligned comparison, it is impossible to verify whether the 40% fertility reduction reflects true algorithmic efficiency or simply reduces the effective training data seen by the model.

### Deeper Analysis Needed (top 3-5 only)
1. Decouple intrinsic fertility metrics from extrinsic downstream performance on a per-script basis, because aggregated averages hide whether the claimed "equitable" distribution actually harms the lowest-resource scripts like Sindhi or Santali.
2. Quantify how the 90% stage-2 transition point impacts robustness on out-of-distribution or noisy text, as heavy reliance on frequency-driven collocations likely causes brittle generation when input diverges from the curated training mix.
3. Correlate super-token frequency with error types in the generated text (e.g., hallucination, repetition, or truncated sentences), proving that merging high-frequency phrases does not degrade discourse-level coherence or instruction adherence.

### Visualizations & Case Studies
1. Provide explicit failure cases where stage-2 merging produces semantically incoherent tokens or incorrectly fuses distinct syntactic units, demonstrating awareness of boundary violations rather than only showing idealized compressions.
2. Plot per-script vocabulary allocation against both fertility score and downstream accuracy to visually validate the corpus-driven alignment claim and expose whether low-resource scripts remain systematically under-optimized.
3. Show layer-wise attention heatmaps comparing IST-trained and baseline models on complex Indic prompts, directly revealing whether super-tokens concentrate attention on meaningful linguistic boundaries or cause destructive averaging across merged phrases.

### Obvious Next Steps
1. Evaluate tokenizer compatibility with supervised fine-tuning (SFT) and preference optimization (RLHF), as multi-word tokens fundamentally alter probability distributions for reward modeling and safety alignment, making this critical for any real-world deployment claim.
2. Benchmark long-context capabilities (e.g., needle-in-a-haystack, document-level multi-hop QA) to determine if sequence compression effectively expands the usable context window or obscures fine-grained retrieval signals.
3. Release the exact training corpus composition, regex rule sets, and transition-scheduling scripts alongside the tokenizer checkpoint, since the method’s performance relies on tightly coupled preprocessing that cannot be independently verified or reproduced from the appendix alone.

# Final Consolidated Review
## Summary
This paper introduces IndicSuperTokenizer, a multilingual tokenizer optimized for 22 Indic languages, English, and source code. By combining script-aware pre-tokenization with a two-stage subword-to-superword learning curriculum, it achieves state-of-the-art fertility scores and a 44% increase in inference throughput over recent baselines. The work is supported by extensive ablations on vocabulary allocation, transition points, and dataset scaling, alongside extrinsic validation on 1B-parameter models trained both from scratch and via continual pretraining.

## Strengths
- **Systematic, well-controlled ablation suite:** The paper isolates the impact of critical tokenizer design choices, providing actionable empirical guidance (e.g., corpus-driven vocabulary allocation strictly outperforms script-specific merging; 90% transition point optimally balances morphological coverage and multi-word compression; diminishing returns beyond 10GB of training data).
- **Clear translation of intrinsic metrics to deployment efficiency:** The consistent SOTA performance across fertility, Rényi efficiency, and bytes-per-token directly maps to a measurable 44% throughput gain during autoregressive decoding on identical compute budgets, validating the practical value of the design.
- **Rigorous cross-script baseline comparison:** The evaluation spans 24 language/code settings against 9 contemporary multilingual tokenizers, with transparent metric definitions and a clear methodological framework that surpasses prior fragmented studies on low-resource script tokenization.

## Weaknesses
- **Evaluation data provenance risks distributional leakage:** Section 3.6 explicitly states the evaluation corpus is curated from the same sources as the training data. For intrinsic metrics like fertility and NSL, testing on corpora statistically aligned with training data inflates reported gains and fails to demonstrate robustness to out-of-distribution, noisy real-world text. A truly disjoint, held-out test set is required to validate generalization.
- **Downstream validation lacks statistical rigor and scale:** Claims of maintaining "comparable performance" rest on single-run point estimates for 1B-parameter models without confidence intervals or variance reporting across random seeds. Given the known sensitivity of small models to data ordering and initialization, and the lack of evaluation beyond the 1B scale, the work cannot substantiate claims about tokenizer efficiency for modern multi-billion parameter architectures where KV-cache dynamics and long-range dependencies dominate inference.
- **Incremental methodological novelty framed as foundational:** The core approach adapts existing two-stage BPE and regex-based pre-tokenization paradigms rather than introducing a new algorithmic mechanism. While the engineering constraints (sentence-boundary guarding, NFKC normalization) are well-executed, the paper's novelty claim of being the "first to carry out a comprehensive benchmarking... in both pretraining and continual pretraining settings" overstates the contribution, as prior work has extensively studied tokenizer impact on training regimes and downstream transfer. The contribution is empirical and systematic, not algorithmic.

## Nice-to-Haves
- Evaluate robustness to realistic code-mixed (e.g., Hinglish) and transliterated inputs, which dominate practical Indic NLP usage but fall outside the current pure-script evaluation.
- Extend latency and throughput measurements to ≥7B models and longer generation horizons to verify whether the 44% throughput gain scales linearly when memory bandwidth, rather than tokenization compute, becomes the bottleneck.
- Report per-script variance in downstream task performance to explicitly verify whether the "corpus-driven alignment" strategy inadvertently under-optimizes for ultra-low-resource scripts (e.g., Sindhi, Santali) despite overall average gains.
- Clarify the exact tokenization regex rules and transition scheduling scripts upon artifact release, as performance is tightly coupled to these preprocessing details.

## Novel Insights
The most consequential insight of this work is empirical rather than algorithmic: a unified, corpus-driven vocabulary allocation strategy fundamentally outperforms modular, script-specific tokenizer merging, and reserving precisely the top 10% of the vocabulary for cross-word superwords optimally balances morphological fidelity with sequence compression. The paper convincingly demonstrates that this two-stage allocation reduces "glitch token" overfitting in the vocabulary tail, suggesting that superwords act as regularizers that stabilize embedding utilization. This provides a practical recipe for multilingual tokenizer design: prioritize unified distributional learning over linguistic modularity, and use the long-tail vocabulary exclusively for frequent collocations rather than rare subwords.

## Potentially Missed Related Work
- **MorphoToken** (Brahma et al., 2025) — directly explores morphologically grounded tokenization for Indian languages and complements the authors' latency-based rejection of explicit morphology analysis. Discussing this work would strengthen the positioning of IST's efficiency trade-offs.
- **Tokenization Fairness & Language Bias** (Petrov et al., 2023; Ahia et al., 2023) — foundational work on how vocabulary allocation inherently biases multilingual models; citing these would ground the paper's equity claims in broader literature on representational fairness.

## Suggestions
1. **Replace or augment the evaluation corpus** with a strictly disjoint test split (e.g., held-out Wikipedia articles or distinct Common Crawl snapshots) and report intrinsic metric variance across these partitions to prove generalization beyond the training distribution.
2. **Report downstream metrics with statistical rigor:** Provide standard deviations or confidence intervals across ≥3 training seeds for the 1B models, or explicitly acknowledge the limitations of single-run comparisons in the main text rather than relegating them to appendices.
3. **Reframe novelty claims** to accurately reflect the paper's strengths: position the work as a definitive empirical benchmark and optimization recipe for Indic multilingual tokenizers, rather than claiming methodological firsts that overlap with existing tokenizer CPT and transfer literature.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
