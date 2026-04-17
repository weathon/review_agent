# HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Comparing human and model performance offers a valuable perspective for understanding the strengths and limitations of embedding models, highlighting where they succeed and where they fail to capture meaning and nuance. However, such comparisons are rarely made, as human performance on embedding tasks is difficult to measure. To fill this gap, we introduce HUME: Human Evaluation Framework for Text Embeddings. While frameworks like MTEB provide broad model evaluation, they lack reliable estimates of human performance, limiting the interpretability of model scores. We measure human performance across 16 MTEB datasets spanning reranking, classification, clustering, and semantic textual similarity across linguistically diverse high- and low-resource languages. Humans achieve an average performance of 77.6% compared to 80.1% for the best embedding model, though with substantial variation: models reach high performance on some datasets while struggling on notably low-resource languages. Our human annotations also reveal multiple dataset issues. We additionally benchmark nine LLMs as annotators on reranking, classification, and STS tasks, finding that they fall short of human performance (76.1% vs. 81.2%) despite offering scalability advantages. We provide human performance baselines, insights into task difficulty patterns, and an extensible evaluation framework that enables a more meaningful interpretation of results and informs the development of both models and benchmarks. Our code, dataset, and leaderboard are publicly available at https://github.com/embeddings-benchmark/mteb.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HUME, a framework for measuring human performance on text embedding tasks from MTEB. The authors evaluate human annotators across 16 datasets spanning four task categories: reranking, classification, clustering, and semantic textual similarity (STS).

### Strengths
The selection of 16 diverse datasets across multiple languages, domains, and task types is well-motivated and thorough. The inclusion of both high-resource and low-resource languages adds an important cross-lingual perspective.

### Weaknesses
While the paper addresses an interesting question, I have concerns:
1. **Necessity and Scope of the Study**
	- **Limited sample sizes**: With only 20-50 instances per task, can we really draw reliable conclusions about human performance on these benchmarks? 
	- **Missing task categories**: The framework only covers 4 of MTEB's task types (reranking, classification, clustering, STS). What about retrieval, which is a very important task nowadays?
2. **Methodological Concerns**
	- **Statistical significance**: The paper presents many performance comparisons but doesn't include significance testing. With small sample sizes, how confident can we be that observed differences are meaningful?

### Questions
**Unclear Motivation for Human-Model Comparison**

I'm struggling to understand the fundamental premise of this paper. Why should we compare embedding models to human performance at all?

This isn't like evaluating generative models, where we compare LLMs to humans because we want human-level intelligence and knowledge. Embedding models serve a completely different purpose. We never use humans to generate embeddings or perform large-scale semantic search in practice. The goal is optimizing downstream task performance, not replicating human judgment. So what does human performance actually tell us about embedding quality?

The paper repeatedly interprets superior model performance as suspicious, suggesting models are "exploiting patterns rather than achieving true semantic understanding." But why can't models just be legitimately better at these tasks? Machine learning is designed to find patterns beyond human perception. When a model outperforms humans at emotion classification, maybe it's actually better at detecting emotional patterns consistently, rather than exploiting flawed data.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces HUME, a framework for measuring human performance on text embedding tasks and comparing it directly to state-of-the-art embedding models. It evaluates human annotators across 16 MTEB datasets spanning classification, clustering, reranking, and semantic textual similarity in multiple languages. Results show that human performance is competitive but not dominant, often ranking around the upper-middle of model performance, with notable advantages in certain multilingual tasks. The authors analyze task difficulty, dataset quality, and human agreement, highlighting that low human performance often reflects dataset ambiguity rather than human limitations. The paper concludes with implications for benchmark design, cultural and linguistic gaps in current models, and recommendations for more reliable evaluation practices.

### Strengths
1. The paper tackles a timely and important problem by grounding text embedding evaluation in human performance. The motivation is well articulated, and the direction is practical and relevant for improving how benchmark results are interpreted.

2. The experimental design is generally solid, with clear task selection and consistent evaluation protocols. While some aspects could be expanded (see weaknesses), it establishes a strong foundation for systematic human–model comparison across languages and task types.

3. The discussion and implications are well developed, offering clear insights into dataset quality, evaluation reliability, and multilingual gaps. The proposed future suggestions are concrete and meaningful, adding to the significance and forward-looking value of the work.

### Weaknesses
1. The study’s reliance on only two annotators for most tasks (and, in some multilingual settings, just one) severely limits the robustness and representativeness of the claimed “human” performance. With such a small pool, the results risk reflecting individual annotator biases rather than general human ability, making the observed human–model gap less reliable as an empirical reference point. In addition, the number of annotated examples per dataset—ranging only from 30 to 50 items—is far too small relative to the scale of the original benchmarks. This raises concerns about statistical power and generalizability: subtle effects may be missed, while task-specific variability could be overstated or underrepresented in the final conclusions.

2. Although the paper identifies several dataset quality issues, it does not sufficiently analyze their underlying causes from the perspective of model training. In particular, there is no supporting evidence or exploration of factors such as training data distributions, domain coverage, or cultural mismatches that might explain the observed human–model performance differences. This lack of analytical depth weakens the explanatory power of the findings, especially in multilingual settings where humans outperform models in some tasks but not others.

3. The paper directly compares human annotators and embedding models but lacks a key experiment: whether large language models can follow the same annotation protocol. Evaluating LLMs under the same instructions would not only provide a useful reference baseline but also offer practical value by reducing the cost of future human evaluations. This experiment could significantly strengthen the paper’s claims regarding the human–model gap.

4. While the paper covers classification, clustering, STS, and reranking tasks, it notably omits retrieval tasks, which are arguably the most central and widely used application of text embeddings in real-world systems. Excluding retrieval significantly limits the practical relevance and completeness of the study’s conclusions.

5. The paper provides insufficient clarity on whether the human labeling instructions were fully aligned with the original dataset annotation schemes. Any divergence in labeling guidelines could introduce inconsistencies and confound the comparison between human annotators and model performance, raising questions about the validity of the measured human–model gap.

### Questions
1. (W1) Could the authors provide justification for using such a small number of annotators and annotated samples? Do they have any evidence or pilot studies suggesting that this limited pool and sample size are sufficient to produce stable and representative estimates of “human” performance?

2. (W2) Can the authors provide a more detailed analysis of the underlying factors contributing to the observed human–model performance differences, such as training data coverage, domain bias, or cultural and linguistic variability, especially in multilingual tasks?

3. (W3) Would it be feasible to evaluate LLMs by instructing them to follow the same annotation protocol as human annotators? If so, how might such an experiment serve as a complementary or cost-efficient reference point in future work?

4. (W4) Why were retrieval tasks excluded from the study, and how might including them—given their centrality to text embedding applications—affect the overall findings and implications?

5. (W5) How closely were the human labeling instructions matched to the original dataset annotation schemes, and could any inconsistencies between the two affect the validity of the human–model comparisons?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HUME, a framework for measuring human-level performance on text embedding benchmarks. They compared human performance with 13 embedding models on 4 major categories of tasks (classification, clustering, semantic textual similarity, and reranking) in MTEB and analyzed the consistency of human annotators as the entry point, putting forward suggestions for the development of future evaluation benchmarks.
The main contributions include: (1) Proposing a framework for evaluating human embedding capabilities, which is currently relatively lacking; (2) The assumption that tasks with low human consistency may reflect the ambiguity of data or definitions rather than the true "superhuman" model capabilities offers a new guidance for the future development of benchmarks.

### Strengths
Supper cool idea!
1. The paper is original in reframing benchmark evaluation: a human evaluation framework that quantifies human-level performance across text-embedding tasks and relates it to model results.
2. The study is of good methodological quality, covering sixteen datasets and four task categories with transparent protocols and clear reporting.
3. The paper’s significance lies in revealing that many “superhuman” model claims arise in tasks with low human agreement, highlighting the need for benchmark reform and more interpretable evaluation standards.

### Weaknesses
The small and homogeneous annotator pool (all male, limited in number, with single annotators for some languages) limits generalizability and the evaluation sets are small and lack confidence interval reporting, reducing statistical robustness.

There is a lack of a more direct and powerful analysis of the performance attribution of the superhuman model (directly attributing low human consistency to data/task design and quality issues is not rigorous enough). 

The retrieval is the most widely used embeddinng application, which is not included in this work. I'm very curious and excited about how well this framework could guide the retriver evaluation.

Additionally, the proposed “agreement-weighted evaluation” idea is conceptually interesting but underdeveloped.

### Questions
How do you ensure that the low agreement among your annotators truly reflects data ambiguity, rather than annotation fatigue or lack of clear instructions? It is possible to supplement the analysis of model bad cases and whether the model as a whole leans towards certain labels.

Could you share a concrete formula for your proposed “agreement-weighted evaluation”? It is suggested that several feasible agreement-based weighted evaluation methods be proposed.

### Soundness
3

### Presentation
2

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
The paper introduces HUMA, a human performance evaluation on the MTEB benchmark. Results show that humans achieve an average score of 77.6%, compared to 80.1% for the best embedding model. Further analysis reveals that text embedding models underperform humans on low-resource languages, and that some datasets exhibit low quality or label ambiguity.

### Strengths
(1) The paper presents a comprehensive human performance evaluation of MTEB, covering reranking, classification, clustering, and STS datasets.

(2) Based on the human performance results and inter-annotator agreement analysis, the authors identify several problematic datasets containing labeling ambiguities (e.g., the emotion classification dataset), which may lead to unrealistic model evaluations..

### Weaknesses
(1) The motivation of conducting a comprehensive human performance evaluation of MTEB is not very convincing to me. The two possible reasons I can get are (1) estimating an uppper bound of dataset performance, similar to how human performance on GLUE was previously used as a target to achieve. But this is not the case here since the paper shows that current models already surpass human performance; and (2) as the paper points out, identifying problematic datasets where humans perform poorly or where inter-annotator agreement is low. However, this second motivation seems somewhat weak to me.

(2) The retrieval task, which is widely used in real-world applications, is not included in the human performance evaluation.

(3) Most tasks includes only two annotations, and for each task there are only 20-50 instances annotated. This raise concerns on the reliablity of the evaluation.

(4) Comparing the performance of native low-resource language speakers with text embedding models primarily trained on English data may not be a fair comparison.

### Questions
(1) In L99-102, the paper states that "Human evaluation is well established in NLP, especially for generative tasks like machine translation, summarization, and dialogue.  In contrast, embedding-based tasks have relied almost exclusively on automated metrics, with little attention to human baselines.". This motivation is not convincing to me. For these generation tasks, automatic evaluation metrics are often unreliable, which requires human evaluation. However, for embedding-based tasks that have well-defined ground truth labels, human evaluation may not be that necessary.

### Soundness
2

### Presentation
2

### Contribution
2
