=== CALIBRATION EXAMPLE 22 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is about a tokenizer for Indic multilingual LLMs. However, “optimized tokenizer” is somewhat vague given the paper’s actual contribution is a specific two-stage BPE variant plus pre-tokenization and vocabulary allocation.
- The abstract clearly states the problem, method, and headline results. It also makes the central empirical claim: improved fertility and throughput with comparable downstream performance.
- A key concern is that the abstract emphasizes “new state-of-the-art in fertility score” and “44% improvement in inference throughput,” but the paper’s throughput comparison in Table 5 is based on a single 1B model pair and a small benchmark set. This is a strong claim that is not yet convincingly generalized.
- The abstract’s “comparable performance on English and Indic benchmarks” is directionally supported, but the reported benchmark averages in Tables 8 and 11 are very close, and some tasks drop nontrivially. The claim should be phrased more cautiously.

### Introduction & Motivation
- The motivation is strong and relevant for ICLR: tokenizer choice affects efficiency, fairness, and multilingual modeling, and Indic languages are a meaningful setting with real script and morphology diversity.
- The gap in prior work is reasonably identified: most tokenizers are English-centric, and multilingual tokenization for Indic languages remains underexplored.
- The research questions are clearly stated and useful.
- That said, the introduction sometimes overreaches by implying broad fairness and cultural inclusivity gains from fertility improvements alone. The paper mostly demonstrates segmentation efficiency, not directly fairness, cultural representation, or downstream equity.
- The claim that the work provides a “systematic recipe” is plausible, but the evidence is mostly a set of ablations on one tokenizer family and one corpus. It is more accurate to call it a well-studied design rather than a universal recipe.

### Method / Approach
- The core method is understandable: Stage 1 learns subwords under whitespace constraints; Stage 2 relaxes constraints to learn multiword “superwords,” with sentence-boundary restrictions. This is a clear and potentially useful idea.
- However, some methodological details are still underspecified for reproducibility and theoretical clarity:
  - In Section 3.1, the exact criterion for the transition point \(t\) is only described as a vocabulary threshold; the paper does not fully specify whether merges are resumed from the same counts, whether statistics are reset, or how the Stage 1 vocabulary is preserved.
  - The sentence-boundary constraint is important, but the implementation details are not fully transparent. How are sentence delimiters detected across languages and noisy web data?
  - In Section 3.3, the “shared vocabulary of 200K tokens, distributed across language scripts” is central, but the allocation rule is not fully formalized in the main text. Table 10 gives one example of script-level proportions, but the full allocation mechanism seems underexplained.
- The language-specific pre-tokenization story is promising, but the paper conflates multiple interventions: regex changes, Unicode normalization, morphology-aware segmentation, and two-stage learning. This makes it hard to isolate which component is responsible for the gains.
- The discussion in Section 4.6 about replacing a pretrained tokenizer with IST is interesting, but the embedding initialization formula is only partially legible in the parsed text. More importantly, the method relies on freezing all but embeddings and LM head during continual pretraining, which limits the strength of the conclusion about tokenizer replacement generally.
- Edge cases and failure modes are under-discussed:
  - Multiword tokens may help frequent collocations but could harm compositional generalization or rare phrase handling.
  - Sentence-boundary merges might interact badly with code, markup, or noisy OCR/web text.
  - The method’s behavior on code-switching beyond the English/Indic mix is not analyzed.
- There are no formal theoretical claims requiring proof, so completeness is mostly about algorithmic clarity. On that front, the paper would benefit from more precise pseudocode and implementation details.

### Experiments & Results
- The experiments do address the main claims: intrinsic tokenization quality, throughput, downstream model performance, ablations, and tokenizer replacement.
- The baseline set is strong and relevant overall, especially LLaMA-4, Sutra, Gemma-3, GPT-OSS, and other general-purpose tokenizers. This is appropriate for an ICLR audience.
- Still, there are several important concerns:
  - The intrinsic metrics are all tokenizer-centric. Fertility, NSL, bytes-per-token, and Renyi efficiency are informative, but they do not directly establish better language modeling quality. The paper partly addresses this with downstream evaluation, which is good.
  - The downstream experiments are limited to a single model scale (1B) and a narrow training setup. That is enough for a controlled study, but not enough to support broad claims about tokenizer superiority across model sizes or training regimes.
  - The throughput result in Table 5 is striking, but it is only reported for one pair of models and one serving setup. The paper does not show confidence intervals over the whole benchmark set, nor does it report variance across hardware/load conditions beyond TTFT error bars.
  - Table 8 and Table 11 show very small average differences on downstream tasks. In Table 8, English average is identical (0.279 vs 0.279), and Indic average improves only modestly (0.388 to 0.394). In Table 11, the averages are also nearly the same. These results support “no major degradation,” but not a strong downstream superiority claim.
  - The paper says IST performs best in 20/24 languages for fertility and 23/24 for NSL, but the gains in some languages are tiny. The analysis would be stronger with significance testing or at least variability estimates.
- A major missing ablation is disentangling the contributions of:
  1. LLaMA-4 regex vs GPT-2 regex,
  2. two-stage curriculum vs one-stage learning,
  3. script-proportional vocabulary allocation,
  4. NFKC normalization,
  5. any morphology-aware preprocessing.
  The current ablations touch several of these, but not in a clean factorial way. This makes it hard to identify the true causal drivers.
- Dataset and evaluation choices are mostly appropriate, but one issue is that the tokenizer evaluation corpus is drawn from the same sources as training data. That is not necessarily wrong for intrinsic tokenizer evaluation, but it makes the evaluation less informative about robustness to distribution shift. For downstream tasks, the use of standard benchmarks is good.
- ICLR would likely expect more evidence that improved fertility translates into system-level gains beyond one model and one serving environment.

### Writing & Clarity
- The paper is generally understandable, and the high-level idea is clear.
- The main clarity issue is not grammar but methodological entanglement: several design choices are introduced together, making it hard to tell what exactly IST is and which parts are essential.
- Some sections could be clearer about what is measured and what is claimed:
  - Section 4.1 mixes fertility, NSL, bytes-per-token, and Renyi metrics, but the causal interpretation of these metrics is not always consistent.
  - Section 4.5 on “glitch tokens” is interesting but somewhat hard to interpret without a more direct connection to downstream behavior.
  - Section 4.6 needs a clearer explanation of tokenizer replacement and what parts of the pretrained model are updated.
- Tables are generally informative, though the paper would benefit from more compact summary tables with mean ± std across languages, rather than many large per-language tables without uncertainty estimates.

### Limitations & Broader Impact
- The paper acknowledges one important limitation: morphology-aware preprocessing can hurt throughput due to language ID and analyzer overhead, which is a good and honest point.
- However, the broader limitations are not fully discussed:
  - The tokenizer is optimized for a specific set of Indic scripts and languages; its portability to other low-resource multilingual settings is unknown.
  - The gains are shown primarily on fertility and throughput, with only modest downstream changes. The paper should acknowledge that tokenizer improvements may not reliably translate into large benchmark gains.
  - The method may be less effective for highly noisy text, code-switching, or domains with lots of markup/code.
- Broader societal impact is mostly positive in intent, but there is a potential downside: optimizing tokenizers for a particular set of languages can shift capacity away from other languages or dialects if used in a shared multilingual model. This tension is not discussed.
- The ethics statement is fine as far as it goes, but I would expect a more concrete discussion of who benefits, who might be left out, and how the language/script allocation choices could reproduce bias if deployed widely.

### Overall Assessment
This paper is a solid tokenizer-focused systems paper with a genuinely relevant multilingual setting and a thoughtful set of ablations. The strongest contribution is the empirical demonstration that a two-stage subword-to-superword tokenizer with script-aware pre-tokenization can substantially improve fertility and inference throughput for Indic languages without obvious downstream harm. That said, for ICLR’s bar, the main weakness is that the causal story is not yet clean enough: several design choices are bundled together, the downstream gains are modest, and the throughput result is shown in only one controlled setting. I think the contribution is promising and likely useful, but the paper would be stronger with cleaner ablations, clearer algorithmic specification, and more cautious claims about generality and downstream impact.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes IndicSuperTokenizer (IST), a multilingual tokenizer tailored to 22 Indic languages plus English and code. The core idea is a two-stage BPE curriculum: first learn subword units within whitespace boundaries, then allow cross-word merges to capture frequent multiword expressions, combined with script-aware regex pre-tokenization and normalization. The paper reports improved intrinsic efficiency metrics—especially fertility and normalized sequence length—plus modest or competitive downstream results and measurable inference throughput gains compared to several modern multilingual tokenizers.

### Strengths
1. **Addresses a genuinely important problem for ICLR standards: tokenizer efficiency for multilingual LLMs.**  
   The paper targets an underexplored but practically meaningful issue: tokenizer quality across morphologically rich, low-resource Indic languages. This is aligned with ICLR’s interest in infrastructure-level methods that affect training/inference cost and fairness.

2. **Broad evaluation across many languages and multiple intrinsic metrics.**  
   The paper evaluates on 22 Indic languages, English, and code, and reports fertility, normalized sequence length, R\'enyi entropy/efficiency, and bytes-per-token. This is stronger than many tokenizer papers that only optimize a single metric.

3. **Evidence of significant intrinsic gains over strong baselines.**  
   IST shows lower fertility than the compared tokenizers in most languages, with especially large improvements over general-purpose tokenizers on several Indic languages. For example, Table 3 shows substantial reductions relative to LLaMA-4 for languages such as Oriya, Santali, and others, and the abstract claims a 39.5% average fertility improvement over LLaMA-4 and 18% over Sutra.

4. **The paper goes beyond intrinsic metrics and tests downstream effects.**  
   A notable positive is that the tokenizer is evaluated in pretraining-from-scratch and continual-pretraining settings, and the authors also measure inference throughput and latency. This makes the work more relevant to ICLR than a purely descriptive tokenizer study.

5. **Ablation studies help justify design choices.**  
   The paper includes ablations on data size, transition point, vocabulary size, normalization, and one-stage vs. two-stage training. These experiments provide some support that the proposed design is not arbitrary.

6. **Practicality and release intent are emphasized.**  
   The authors state they will release an evaluation framework and dataset, which would improve reproducibility and usefulness to the community if completed.

### Weaknesses
1. **The claimed gains on downstream performance are modest and sometimes inconsistent.**  
   The strongest evidence is intrinsic efficiency, but ICLR generally expects either a clearly new algorithmic insight or compelling evidence that efficiency gains matter in practice. On downstream benchmarks, improvements are often tiny or mixed: Table 8 shows essentially equal average English performance (0.279 vs. 0.279) and only a small Indic average gain (0.388 vs. 0.394). Some individual tasks even regress slightly.

2. **The comparison to baselines is not always fully convincing as a controlled scientific study.**  
   The paper compares against tokenizers from many model families with different training data, vocab sizes, and unknown preprocessing details. While the authors acknowledge limited public documentation, this still weakens causal claims. In particular, some “better” results may stem from training distribution or vocabulary scale rather than the specific IST design.

3. **Some core claims would benefit from stronger statistical rigor.**  
   The paper reports averages and many tables, but it does not clearly present significance tests, confidence intervals, or variance across training runs for the main tokenizer comparisons. For a paper making broad claims of “state of the art,” ICLR reviewers would likely want stronger evidence that improvements are robust and not sensitive to corpus choice or random seed.

4. **The real novelty relative to recent tokenizer work is somewhat incremental.**  
   IST combines ideas already seen in recent work: two-stage subword/superword learning (similar in spirit to SuperBPE), regex pre-tokenization, normalization, and multilingual data mixing. The contribution seems more like a careful adaptation and empirical validation for Indic languages than a fundamentally new tokenizer framework.

5. **The throughput/latency evidence is encouraging but narrow.**  
   The latency study uses 1B models on 8 H100 GPUs and a 200-example set. This is useful, but the setup may not generalize to other model sizes, serving stacks, or batch regimes. Also, throughput gains are strongly influenced by sequence length reduction, so the results are somewhat expected rather than deeply diagnostic.

6. **The paper under-explains some methodological choices.**  
   The exact script-level vocabulary allocation, corpus curation, filtering, and the handling of code/English mix are not fully transparent in the main text. The role of morphology-aware segmentation is also not fully developed, since the authors explore it but do not adopt it, leaving some design space underexplained.

7. **Potential fairness/cultural claims are stronger than the evidence.**  
   The paper uses language about “equitable” and “culturally inclusive” tokenization, but the evaluation is primarily fertility/efficiency-centric. There is limited direct evidence that the tokenizer improves fairness in any broader sense beyond token count parity.

### Novelty & Significance
**Novelty:** Moderate. The main novelty lies in adapting a two-stage subword-to-superword curriculum, coupled with Indic-specific pre-tokenization, to a broad multilingual Indic setting and validating it carefully. However, the approach appears to be an incremental synthesis of prior ideas rather than a fundamentally new tokenizer paradigm.

**Significance:** Moderate to good. Tokenization is an important lever for multilingual LLM efficiency, and the paper shows meaningful gains for Indic languages, which is practically valuable. That said, ICLR’s acceptance bar typically favors contributions with either strong conceptual novelty or clearly demonstrated downstream impact; here the intrinsic gains are stronger than the task-level gains, so the overall significance is promising but not fully निर्णायक.

**Clarity:** Fairly good overall. The paper is structured clearly and the motivation is easy to follow. However, several implementation details and evaluation assumptions are only partially specified, and the heavy reliance on many intrinsic metrics can obscure the main takeaway.

**Reproducibility:** Mixed. The authors provide many experimental details, ablations, and say they will release artifacts, which is positive. Still, reproducibility is limited by incomplete disclosure of baseline tokenizer training data/protocols, possible implementation differences across frameworks, and the absence of full statistical reporting.

### Suggestions for Improvement
1. **Strengthen the empirical claims with statistical analysis.**  
   Report confidence intervals, seed variance, and significance tests for main intrinsic and downstream results. This would make the SOTA claims much more credible.

2. **Provide a more controlled comparison against strong tokenizer baselines.**  
   Where possible, retrain baseline tokenizers on the same corpus and with matched vocab sizes and preprocessing. This would isolate the effect of the tokenizer design rather than confounded data differences.

3. **Clarify the exact contribution relative to SuperBPE and BoundlessBPE.**  
   A sharper conceptual comparison would help reviewers understand whether IST is a new algorithm, a new multilingual recipe, or an application of existing techniques to Indic languages.

4. **Expand downstream evaluation or make the negative/neutral results more central.**  
   If the goal is to show practical importance, test larger models, more tasks, or longer-context settings where token efficiency might produce clearer end-to-end wins. Alternatively, frame the paper primarily as an intrinsic tokenizer study and temper downstream claims.

5. **Include more detail on corpus construction and balancing.**  
   Explain filtering, deduplication, script normalization, and how English/code/Indic proportions were chosen. This is especially important for multilingual tokenizer training.

6. **Report tokenization quality beyond fertility.**  
   Add qualitative analysis of segmentation behavior, OOV handling, and examples where IST helps or hurts. Some error analysis on languages with weaker gains would also be valuable.

7. **Temper fairness language unless directly supported.**  
   If the main evidence is efficiency, say so explicitly. To substantiate fairness/inclusivity claims, add analyses on performance parity, sequence-length parity, or downstream effect parity across languages.

8. **Make the evaluation framework and tokenizer release concrete.**  
   Since the paper promises release, providing links, licenses, and a minimal reproducibility checklist would substantially improve the paper’s impact and trustworthiness.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to strong multilingual tokenizer baselines trained under the same data/vocab budget, especially SentencePiece Unigram, byte-level BPE, and a fair reimplementation of BoundlessBPE/SuperBPE on the same corpus. Without matched-budget comparisons, the claim that IST is state-of-the-art for Indic tokenization is not convincing under ICLR standards.

2. Evaluate on a truly held-out benchmark corpus, not text curated from the same sources as tokenizer training. Intrinsic gains on in-domain data can overstate generalization; ICLR reviewers will expect fertility and compression results on unseen corpora, domain-shift text, and noise/code-mixed text.

3. Add scaling experiments for downstream training that isolate tokenizer effects across model sizes, not just one 1B model. The core claim is that the tokenizer improves efficiency and preserves performance, and that needs verification at multiple scales because tokenization effects often change with model capacity.

4. Include a stronger downstream comparison on tasks where Indic morphology and long-context compression should matter, such as translation, document QA, summarization, or retrieval over long Indic documents. Current benchmarks are mostly short-form classification-style tasks, which are weak evidence for the claimed practical benefit of superwords.

5. Report compute-normalized pretraining quality, not only iso-FLOPs and final benchmark scores. If IST reduces sequence length but slightly worsens or matches loss, the paper needs a fuller efficiency picture: tokens/sec during training, steps-to-quality, and validation loss/perplexity at fixed wall-clock.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze whether the fertility gains come from meaningful linguistic units or just more aggressive compression. Without token purity or linguistic alignment analysis, the paper’s “morphologically aligned” and “semantically faithful” claims are not established.

2. Break results down by script, morphology type, and language-resource level with significance testing. The paper claims fairness and robustness across 22 languages, but it needs variance, confidence intervals, and stratified analysis to show gains are not driven by a few high-resource scripts.

3. Quantify the trade-off between shorter sequences and token ambiguity. Superwords can help compression but hurt compositionality; the paper needs evidence that IST does not degrade rare-word handling, OOV robustness, or compositional generalization.

4. Analyze the effect of the sentence-boundary constraint. The paper introduces this as important, but does not show how much it matters for stability, EOS behavior, or downstream performance relative to unconstrained cross-sentence merging.

5. Provide a proper causal explanation for why Stage-2 helps beyond Stage-1. The current discussion asserts that two-stage training is better, but it needs analysis of merge statistics, token frequency distributions, and what kinds of tokens are actually learned in the second stage.

### Visualizations & Case Studies
1. Show side-by-side tokenization examples for frequent words, morphologically complex words, named entities, code-mixed text, and idioms across multiple Indic scripts. This would reveal whether IST learns useful units or just arbitrary long tokens.

2. Visualize token frequency histograms and tail utilization for IST versus baselines. The paper claims fewer “glitch tokens” and better vocabulary efficiency, but a clear distribution plot is needed to see whether the tail is truly healthier.

3. Add attention-length or throughput breakdowns versus input length. Since the efficiency claim depends on shorter sequences, the paper should show when the latency gains persist and when they vanish, rather than one averaged benchmark.

4. Provide failure cases where IST makes segmentation worse than baseline tokenizers. A few concrete examples would test the “linguistically grounded” claim and show where multiword merging harms interpretation or model robustness.

### Obvious Next Steps
1. Train or adapt a larger multilingual model with IST and compare it against the same model using the original tokenizer at equal wall-clock and equal compute. That is the clearest test of whether the tokenizer is actually useful for ICLR-level claims about scalable LLMs.

2. Evaluate on out-of-domain and mixed-script Indic data, especially noisy social media, OCR, and code-mixed corpora. If IST is meant to be practical for real multilingual deployment, those are essential next checks.

3. Release and benchmark a standardized tokenizer suite with reproducible corpus selection, training code, and exact preprocessing rules. The paper currently depends on many design choices that could shift results; an ICLR-quality contribution needs a cleaner reproducible protocol.

4. Compare against byte-level and character-aware alternatives on robustness and efficiency. Without that, it is hard to know whether the gains come from a genuinely better tokenizer or just a favorable subword design for the chosen evaluation set.

# Final Consolidated Review
## Summary
This paper proposes IndicSuperTokenizer (IST), a multilingual tokenizer for 22 Indic languages plus English and code. The main idea is a two-stage BPE curriculum: first learn subword units within word boundaries, then allow cross-word merges to capture frequent multiword expressions, combined with script-aware pre-tokenization and normalization. The paper reports substantially better intrinsic tokenization efficiency than several strong multilingual baselines, along with modest downstream preservation of model quality and some inference throughput gains.

## Strengths
- **Addresses an important and underexplored problem for multilingual LLMs.** Tokenization efficiency for morphologically rich, script-diverse Indic languages is genuinely consequential for training cost, inference latency, and fairness across languages. The paper is well-motivated and targets a setting where English-centric tokenizers clearly underperform.
- **Broad intrinsic evaluation and useful ablations.** The paper evaluates fertility, normalized sequence length, Renyi entropy/efficiency, and bytes-per-token across 22 Indic languages, English, and code, and includes ablations on training data size, transition point, vocabulary size, normalization, and one-stage vs. two-stage training. This is stronger than a typical single-metric tokenizer paper.
- **The core efficiency results are convincing at the intrinsic level.** IST consistently improves fertility and sequence compactness over the compared tokenizers, often by large margins on Indic languages, and the paper also shows higher bytes-per-token and better Renyi efficiency. The sequence-length reduction is large enough to plausibly matter for inference.
- **The paper goes beyond intrinsic metrics and tests system impact.** It includes downstream pretraining experiments and an explicit latency/throughput study, which is important because tokenizer papers often stop at fertility alone. The continual-pretraining experiment also shows the tokenizer can be swapped into an existing model without catastrophic degradation.

## Weaknesses
- **The downstream gains are small and do not match the strength of the intrinsic claims.** Table 8 shows essentially identical English average performance and only a tiny Indic average gain, while Table 11 is similarly flat. This supports “no obvious harm,” but not a strong claim that the tokenizer materially improves model quality.
- **The causal story is still muddled because several interventions are bundled together.** IST combines script-aware regex, NFKC normalization, vocabulary allocation across scripts, two-stage subword/superword learning, and optional morphology-aware preprocessing. The ablations help, but they do not cleanly disentangle which component drives the gains, so the paper is more of a carefully engineered recipe than a sharply identified method.
- **The evidence for general efficiency gains is narrow.** The throughput result is based on one 1B model pair and one serving setup, and the downstream results are also limited to a single model scale. That is enough for a controlled study, but not enough to support broad claims about tokenizer superiority across model sizes or deployment regimes.
- **The comparison against baselines is not fully controlled.** Many baselines come from different model families with different training corpora and undisclosed preprocessing. That is unavoidable to some extent, but it weakens the strength of the “state-of-the-art” claim, especially for a paper centered on intrinsic tokenizer comparisons.

## Nice-to-Haves
- A matched-budget reimplementation of the strongest tokenizer baselines on the same corpus would make the comparisons much more convincing.
- More qualitative segmentation examples, especially for code-mixed text, rare words, and failure cases, would help verify that the gains come from linguistically meaningful units rather than aggressive compression.
- A clearer release plan for the evaluation framework and datasets would improve reproducibility and practical impact.

## Novel Insights
The most interesting insight is that for Indic multilingual settings, the biggest win may come not from simply increasing vocabulary size, but from structuring tokenizer training so that meaningful subword units are established first and multiword units are learned only later. The paper’s ablations suggest that script-aware pre-tokenization and a late-stage relaxation of word-boundary constraints work together to improve vocabulary utilization, and the “glitch token” analysis hints that reserving the tail of the vocabulary for frequent multiword expressions may reduce under-trained junk at the end of the vocab. That said, the paper’s own downstream results also show that better token efficiency does not automatically translate into meaningful benchmark gains.

## Potentially Missed Related Work
- **SentencePiece Unigram / byte-level BPE variants** — relevant as standard multilingual baselines and useful for a more controlled comparison.
- **BoundlessBPE** — directly relevant because it also relaxes pre-tokenization constraints and is the closest conceptual comparator.
- **SuperBPE** — relevant since IST’s two-stage subword-to-superword curriculum is closely related to this line of work.
- **ReTok** — relevant for the tokenizer replacement / continual-pretraining angle.
- **Morphtok** — relevant because the paper discusses morphology-aware segmentation for Indic languages, even though it does not adopt it in the final system.

## Suggestions
- Reframe the contribution more cautiously: strong intrinsic tokenizer efficiency for Indic languages, with limited but promising downstream validation.
- Add a matched-data, matched-vocab comparison against the strongest tokenizer baselines, ideally retrained on the same corpus.
- Report confidence intervals or multi-seed variance for the main intrinsic and downstream results.
- Include a small set of held-out, noisy, and code-mixed evaluations to test robustness beyond the training-source distribution.
- Provide a sharper algorithmic description of the transition from Stage 1 to Stage 2, including exact handling of vocabulary preservation and sentence-boundary constraints.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
