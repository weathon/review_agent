# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
As large language models (LLMs) are pretrained on massive web corpora, careful selection of data becomes essential to ensure effective and efficient learning. While perplexity (PPL)-based filtering has demonstrated strong performance, it suffers from drawbacks: substantial time costs and inherent unreliability of the model when handling noisy or out-of-distribution samples. In this work, we propose a simple yet powerful alternative: a prior-based data filtering method that estimates token priors using corpus-level term frequency statistics, inspired by linguistic insights on word roles and lexical density. Our approach filters documents based on the mean and standard deviation of token priors, serving as a fast proxy to PPL while requiring no model inference. Despite its simplicity, the prior-based filter achieves the highest average performance across 20 downstream benchmarks, while reducing time cost by over 1000× compared to PPL-based filtering. We further demonstrate its applicability to symbolic languages such as code and math, and its dynamic adaptability to multilingual corpora without supervision.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a prior-based data filtering approach that replaces perplexity-based filtering with a simple frequency-based proxy. By leveraging corpus-level token priors (term frequencies) and their document-level mean and variance, the method identifies noisy or ill-formed data efficiently. It achieves comparable or better downstream performance than PPL-based methods while being 1000x faster.

### Strengths
- Efficient and elegant alternative to PPL filtering.
- Strong empirical results across 20 benchmarks and two model sizes.
- Grounded in interpretable linguistic insights (lexical density, word frequency).
- Scalable and robust, it extends to code and math data.

### Weaknesses
- Results mostly on English and related datasets; generalization to other scripts unclear: The experiments are focused almost entirely on English or similar languages, so it’s hard to tell how well the method would hold up for other scripts or morphologically rich languages. Since the approach depends on token frequencies, it might behave very differently for languages with complex segmentation or compounding (like Japanese or Turkish). Even a small multilingual check would help back up the broader claims.
- Some claims (e.g., automatic "learnability" detection) lack quantitative rigor: The paper argues that the method can automatically detect “learnable” data, but this idea isn’t really supported by clear evidence. There aren’t concrete metrics or controlled experiments showing that the filtered data is actually easier for a model to learn from. As it stands, the claim feels more intuitive than proven.
- May remove low-frequency, high-value data: Because the method penalizes documents with rare token patterns, it could unintentionally throw away useful or unique data like text from small domains, rare languages, or creative writing. This might reduce the diversity of the pretraining corpus, especially in the long tail where valuable but infrequent patterns live. A short analysis of this trade-off would make the results more convincing.
- Some sections read more as post-hoc rationalization (linguistic grounding) than rigorous analysis.
- Limited novelty from a machine learning standpoint

### Questions
-How sensitive are results to tokenizer choice or token granularity?
- Could the method over-filter rare but meaningful content?
- How does this interact with deduplication or semantic filtering pipelines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a prior-based data filtering method as a simple, fast, and powerful alternative to perplexity (PPL) for cleaning massive web corpora used in LLM pretraining. Inspired by linguistic insights on lexical density, the method estimates token priors using corpus-level term frequency and filters documents based on the mean and standard deviation of these priors, requiring no model inference. Despite its simplicity, the prior-based filter achieves state-of-the-art average performance, outperforming PPL-based filtering across 20 downstream benchmarks. The proposed approach is also over 1000 times faster than PPL filtering and demonstrates robust applicability to multilingual corpora and symbolic languages like code and math.

### Strengths
* This paper proposes a simple yet effective data filtering method that achieves efficient information filtering using only statistical features.
* Experiments show that the method is not only fast but also outperforms PPL-based approaches in terms of performance.
* The authors demonstrate the generality of the method, showing that it can be applied to text, code, and math data.

### Weaknesses
* The experiments are conducted only on a 3B-scale dataset, which is too small to be convincing. I believe the authors should use a larger dataset (e.g., over 100B) and test on more complex tasks to better demonstrate the superiority of the proposed data filtering method.
* In terms of writing, Chapter 2 includes excessive discussion of basic NLP concepts such as word frequency. I suggest that the authors reduce such background explanations or move them to the appendix.
* The authors only test on unigrams and do not extend the analysis to n-grams. I believe they should further discuss the filtering effectiveness for n-grams.
* The paper lacks experiments on the choice of filtering thresholds and filtering ratios. I suggest that the authors conduct additional experiments on these two hyperparameters to determine the optimal settings.

### Questions
* Have the authors considered combining different methods to achieve better filtering performance?
* Is there any scaling experiment on data filtering? The authors should test with larger data proportions and model sizes to fully demonstrate the effectiveness of the filtering approach.
* Why did the authors choose word frequency instead of n-grams or more complex statistical metrics? Previous work in statistical NLP has shown that n-grams are often more effective than using word frequency alone.
* Is there a detailed experiment on the filtering ratio to determine the optimal filtering threshold?
* Would different tokenizers significantly affect the filtering results? Moreover, can the filtering method remain effective across different tokenizers?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a text data filtering method based on word frequency prior to replace mainstream perplexity-based filtering strategies. Drawing upon statistical patterns of word frequency and lexical density in linguistics, the method identifies and filters anomalous documents deviating from the overall distribution by calculating the mean and standard deviation of the logarithm of word frequency within documents. Experimental results demonstrate that this method achieves performance superior to or equivalent to perplexity-based filtering across 20 downstream tasks, while reducing filtering time to approximately 0.1% of the original method, exhibiting remarkable computational efficiency. Furthermore, the method shows good applicability in both symbolic languages and multilingual corpora, without requiring pre-trained models or manually annotated data.

### Strengths
1. This method employs word frequency statistics for filtering, eliminating the need for training or invoking reference models, thereby reducing implementation complexity and resource dependency.

2. Across 20 downstream tasks encompassing world knowledge and common-sense reasoning, the model demonstrated superior average performance compared to baseline models.

3. This method is linguistically grounded in word frequency and lexical density, possessing an interpretable statistical foundation.

### Weaknesses
1. The paper should thoroughly discuss the advantages of PPL in capturing semantic coherence and contextual dependencies. Emphasizing only its computational cost and noise sensitivity issues may be misleading.

2. The author should consider incorporating additional lightweight or baseline metrics (such as character repetition rate and language model scores) for comparison to enhance the persuasiveness of the experimental results.

3. The authors should strengthen the discussion on the scaling relationship between computational efficiency and corpus size. Since experiments were conducted only at a 6B-word scale, they should demonstrate feasibility at larger scales.

4. Although the hypothesis of a priori approximate PPL is proposed, it lacks rigorous theoretical derivation or boundary condition analysis.

### Questions
1. The method uses the median as the central estimate for “normal documents.” For corpora with highly skewed distributions or mixed multimodal distributions (e.g., a blend of professional forums and news), is the median still a robust estimator? Have other robust central measures been considered?

2. The experiment demonstrated the effectiveness of processing mixed Chinese-English corpora. However, for variants within the same language (such as academic English versus colloquial English), word frequency distributions may differ. Can the method distinguish stylistic variations within the same language from genuine noise?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a prior-based filter that estimates a token's context free probability from corpus term frequencies (priors). For each document, it computes the mean and standard of token priors and drops outliers far from corpus medians, motivated by (i) frequency $\approx$ function/content role and (ii) well-formed text having a stable lexical density. This method approximates PPL filtering while being $~1000\cdot$ cheaper (no model inference) and often outperforms PPL-based and DSIR baseline across  nearly 20 downstream tasks. The code is reproducible.

### Strengths
1. Practical and speedy. This method requires only frequency counts and reported wall-clock is orders of magnitude lower than PPL-based pipelines.
2. Clear, language agnostic intuition: This paper links token frequency to function/content roles; uses document-level $\mu, \sigma$ to capture lexical density.
3. It covers 2 model sizes with 20 tasks. Extensive experiments.

### Weaknesses
1. Frequency-driven criteria can over-filter rare but valuable content (named entities, dialects, minority languages). This paper discusses some multi-lingual behavior in Section 3.4.2 but they use Chinese and English as an example. These two languages are very common in web. Broader language settings might be needed.
2. Baseline scope might be limited. Comparisons focus on PPL and DSIR; other lightweight filters (classifier-based, memorization/dup measures, n-gram LM perplexity, quality/explicit toxicity heuristics) are not covered.
3. Scaling to large models: The authors report results for 137M-1.5B GPT-2 like models. It's unclear whether the advantage persists at 7B-70B scales or with instruction-tuning datasets.

### Questions
Can the authors add more baselines to compare and see whether other lightweights methods have shorter time and also good performance? For instance, baselines listed in related work?

### Soundness
2

### Presentation
2

### Contribution
2
