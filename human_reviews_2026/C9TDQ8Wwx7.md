# Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Long-context language models unlock advanced capabilities in reasoning, code generation, and document summarization by leveraging dependencies across extended spans of text. However, a significant portion of readily available long-text data lacks meaningful long-distance dependencies; most spans can be predicted using only local context. Training on such data is inefficient, making careful data selection crucial. Therefore, we introduce LongFilter, a framework for curating training data tailored to long-context pretraining. LongFilter measures the information gain provided by extended context by contrasting model predictions under long-context versus short-context settings, thereby identifying samples where long-range dependencies are essential. Experiments with LLaMA-3-8B, extending its context length from 8K to 64K, show that LongFilter efficiently selects high-quality data and yields substantial improvements on benchmarks such as HELMET, LongBench, and RULER.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the prevailing data engineering practice for long-context LLMs, which primarily relies on sequence length as a proxy for quality. The authors argue that many long sequences do not genuinely require long-range understanding, and training on them is inefficient. They introduce LongFilter, a data curation framework designed to select training data based on its "long-range information" content.

The core of LongFilter is a scoring function derived from information theory. It quantifies the information gain provided by an extended context by measuring the KL divergence between a model's next-token predictions under a long context versus a short context. A practical and efficient score is developed based on this principle, which prioritizes sequences where access to a longer context significantly reduces the model's prediction loss. The authors conduct experiments by continually pre-training LLaMA-3-8B to extend its context from 8K to 64K. The results show that training on data selected by LongFilter leads to substantial performance improvements on a suite of long-context benchmarks compared to baseline data selection strategies.

### Strengths
1. The central thesis—that data for long-context training should be selected based on its actual need for long-range information—is both intuitive and powerful.
2. LongFilter is elegantly derived from the information-theoretic concept of Conditional Mutual Information, lending it strong theoretical credibility.
3. The experimental results are compelling, showing consistent and significant improvements across multiple models and benchmarks. The ability to achieve the performance of 4B tokens of training with only 1.5B tokens of filtered data is a strong testament to the method's efficiency.
4. The proposed filtering method, while computationally intensive to run, is conceptually simple and does not require modifying the model or training process. It can be applied as a preprocessing step to any long-text dataset.
5. The token-level case study provides insightful qualitative analysis, visually demonstrating that the filter correctly identifies semantically coherent prose as high-value and repetitive or unstructured content as low-value.

### Weaknesses
1. The primary drawback of LongFilter is its high computational cost. The paper notes that scoring a large corpus required 32 H100 GPUs for a day. This is a significant barrier for researchers with limited resources. A discussion on more efficient approximations of the score would make the work more accessible.
2. The quality of the selected data is contingent on the capabilities of the model used for scoring. The paper uses a very strong model (LLaMA-3.1-8B). It is unclear how performance would be affected if a smaller or less capable model were used for scoring.
3. The choice of l_short (4K) and l_long (64K) seems somewhat arbitrary. The effectiveness of the filter might be sensitive to these hyperparameters, but this sensitivity is not explored.

### Questions
The final scoring function (Eq. 7) weights the loss difference (L_short - L_long) by exp(-L_long). This means that tokens on which the model is very uncertain even with long context (i.e., high L_long) will receive a very low score, regardless of the information gain. Did you consider alternative formulations, such as using the raw loss difference L_short - L_long? What is the justification for this specific weighting scheme?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work focuses a key challenge in training long-context language models (LLMs): much of the available long-text data does not actually require extended context to be modeled effectively.
To tackle this, the authors propose LongFilter, a framework that quantifies the information gain from using longer contexts by comparing model predictions under short and long context settings.
This allows them to curate pretraining datasets where genuine long-range dependencies are present.
They demonstrate that applying LongFilter during pretraining, specifically when extending LLaMA-3-8B’s context length from 8K to 64K, leads to substantial improvements on several benchmarks (HELMET, LongBench, RULER).
The paper also provides analyses showing which types of text segments most benefit from long-context modeling.

### Strengths
1. This work addresses an important but often overlooked issue in scaling up LLMs’ context windows, i.e., not all “long” data is useful for learning true long-range dependencies.
2. The authors propose a clear and interpretable metric (information gain) for identifying valuable training samples.
3. The paper provides an additional analysis on what kinds of text benefit most from extended contexts, informing future research and practical data selection.

### Weaknesses
1. All the results and analysis are made on one model (i.e., LLaMA-3-8B). It could be better to show the effectiveness on more backbone models.
2. Beyond the long-context understanding tasks, to further demonstrate the improvements on processing long-range information , it could be better to show the performance on long generation tasks, such as long reasoning and long writing tasks.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Much readily available long-text data does not genuinely require extended context, as most spans can be predicted using only short-range context. Not all long sequences provide meaningful long-context information, and some long sequences can dilute the training signal. To solve this, the authors propose LongFilter, a framework for curating training data tailored to long-context pretraining. The core idea of LongFilter is to quantify the "information gain" provided by an extended context. The method uses a scoring function derived from the Kullback-Leibler (KL) divergence between the next-token prediction distribution given the long context versus the short context. The score is formulated as the reduction in prediction loss when using the long context compared to the short context, weighted by the model’s confidence on the token given the full context.A high score signifies that the extended context is crucial for making accurate predictions.
The authors conducted experiments by extending the Llama-3.1-8B model from an 8K context length to 64K. They filtered data from the SlimPajama dataset, selecting the top 20% of high-scoring data for training. Models trained on LongFilter-selected data showed "substantial improvements" on benchmarks like HELMET, LongBench, and RULER, achieving average gains of over 2 points. Meanwhile, LongFilter significantly improves training efficiency. Experiments showed that training with 1.5B filtered tokens achieved performance comparable to training on 3-4B unfiltered tokens.

### Strengths
The paper show the fact that sequence length alone is an insufficient proxy for data quality. Much long-text data does not genuinely require long-range dependencies, which can dilute the training signal. 

The proposed method, LongFilter, quantifies the informational value of extended context, which is a significant contribution for LLM pretraining.

The experiments are well-founded. The experiments use Llama-3.1-8B, extend it to a significant context length (8K to 64K) , and train on a large-scale, standard dataset SlimPajama. The method is also compared against relevant baselines like ProLong and LongWanjuan. The results show substantial improvements across a suite of demanding long-context benchmarks, including HELMET, LongBench, and RULER, and demonstrate that training on 1.5B filtered tokens can achieve performance comparable to training on 3-4B unfiltered tokens.

### Weaknesses
The paper highlights the training efficiency gains, but the data curation step itself appears to have a high computational cost. The authors report that scoring each corpus required 32 NVIDIA H100 GPUs for a single day.

The paper uses the Llama-3.1-8B model as the scoring model to conduct experiments and achieve favorable results. However, the paper does not test other models for generating the scores, which suggests a lack of generalizability regarding the choice of the scoring model in the experiments.

### Questions
The Llama-3.1-8B model was used for scoring. Have you experimented with using other models for this step? How does the choice of the scoring model impact the quality of the selected data and the performance of the final trained model?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LongFilter, a data curation framework that selects long-context pretraining data based on the information gain provided by extended context, measured via KL divergence between next-token prediction distributions under short (e.g., 4K) and long (e.g., 64K) contexts. The core insight—that sequence length alone is insufficient to ensure meaningful long-range dependencies—is well-motivated and validated through experiments on LLaMA-3-8B extended from 8K to 64K context. Results on HELMET, LongBench, and RULER show consistent gains (e.g., +2+ average points), with particularly strong improvements in recall (Needle-in-a-Haystack) tasks.

### Strengths
* Novel and principled metric: The use of conditional mutual information (via KL divergence) to quantify long-range dependency is theoretically grounded and practically effective.

* Strong empirical validation: Experiments across multiple benchmarks and data domains (ArXiv, Books, CommonCrawl) demonstrate robustness. LongFilter outperforms length-based baselines (ProLong) and a recent alternative (LongWanjuan).

* High practical impact: Achieves comparable performance with half the training tokens, significantly improving data efficiency.

* Transparent methodology: The scoring function (Eq. 6–7) is interpretable as a confidence-weighted loss reduction, and token-level visualizations (Figure 5) provide intuitive validation (e.g., low scores for repetitive TikZ code).

### Weaknesses
* Computational cost: Scoring requires forward passes with both short and long contexts using a large model (Llama-3.1-8B), which may be prohibitive for very large corpora despite optimizations.

* Limited model scope: Evaluation is restricted to LLaMA-3-8B; generalizability to other architectures (e.g., Mamba, RWKV) or smaller models is unverified.

* Task coverage: Benchmarks focus on retrieval and structured reasoning; performance on narrative coherence or open-ended generation is not assessed.

* Baseline fairness: ProLong and LongFilter share high-quality data (since ProLong uses unfiltered data that includes LongFilter’s top samples), potentially underestimating LongFilter’s true advantage.

### Questions
* How sensitive is LongFilter to the choice of base model for scoring? Would a smaller or differently trained model yield comparable rankings?

* Could the scoring be approximated more cheaply (e.g., via attention patterns or n-gram statistics) without sacrificing effectiveness?

* Does LongFilter improve reasoning depth (e.g., multi-hop QA) or primarily retrieval fidelity? The current benchmarks emphasize the latter.

* How does performance scale with the selection ratio (e.g., top 10% vs. 20%)? Is there a point of diminishing returns?

### Soundness
3

### Presentation
2

### Contribution
2
