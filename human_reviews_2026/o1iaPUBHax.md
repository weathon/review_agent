# Consider Size not Language: Effects of External Evidence in Multilingual Medical Question Answering

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
This paper investigates, for the first time, Multilingual Medical Question Answering across high-resource (English, Spanish, French, Italian) and low-resource (Basque, Kazakh) languages. We evaluate three types of external evidence, such as local repositories, dynamically web-retrieved content, and LLM-generated explanations with models of varying size. Our results show that larger models consistently perform the task better in English for both the baseline evaluations and when adding external knowledge. Interestingly, retrieving the evidence in English often surpasses language-specific retrieval, even for non-English queries. These findings challenge the assumption that language-related external knowledge uniformly improves performance and reveal that effective strategies depend on both the source of language resources and on model scale.  Furthermore, specialized static repositories such as PubMed are limited: while they provide authoritative expert knowledge, they lack adequate multilingual coverage and do not fully address the reasoning demands of the task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the first study on MMQA across both several well-studied languages (English, Spanish, French, Italian) and low-resource ones (Basque, Kazakh). The authors evaluate models of different sizes using three types of external evidence—local repositories, dynamically retrieved web content, and LLM-generated explanations. Authors show, across results, that larger models perform best in English, both with and without added external knowledge. Notably, retrieving evidence in English often outperforms language-specific retrieval, even for non-English queries. These results challenge the idea that external knowledge in the query’s language always improves performance. The study also finds that while expert repositories like PubMed provide reliable knowledge, they lack multilingual depth and do not fully support the complex reasoning required by MMQA tasks.

### Strengths
1) Code and resources are publicly available.

2) The proposed study allows for the study of different levels of language coverage, and also offers an original view comparing the use of resources (web vs. local), compared to models without external resources. Local use is the most interesting from my point of view, with in particular the idea of ​​using private data (e.g. patient records) which must remain local.

3) This systematic comparison allows the choice of models and strategy to be adapted according to material and resource constraints.

### Weaknesses
1) Part of the study is quite classic, especially everything concerning the study of the size of the models.

2) The benchmark ultimately appears to be a translation of existing data, which is a priori unverified: translation errors can be numerous, particularly for languages ​​with few resources, which adds a significant bias to the results.

3) Few models have ultimately been studied, in particular models adapted to the medical field (eg MedGemma), or even open-source models (eg Apollo or OLMo). Similarly, no reasoning model has been benchmarked.

### Questions
1) For MedExpQA, why not use a translated version of the data?

2) For the Search Query Generation, what is the quality of translations, especially for low-resource languages?

3) What is the impact of the translation?

4) Why did you finally stop at English benchmarks, when there are now other resources in other languages ​​in MCQA? The latter being real data and not translated.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates multilingual medical question answering by evaluating how different types of external evidence (local repositories, web search, and LLM-generated content) affect performance across six languages with varying resource availability. The authors test multiple LLMs of different sizes and report that larger models generally perform better, English-based retrieval often outperforms target-language retrieval even for non-English queries, and external knowledge integration shows diminishing or negative returns for larger models. While the empirical scope is broad and the multilingual focus is valuable, the paper suffers from methodological concerns, limited explanatory depth, and conclusions that largely confirm expected patterns without providing novel insights into the underlying mechanisms.

### Strengths
1. This paper proposes to address an essential real-world problem, medical QA in low-resource languages.
2. The authors conduct a broad and systematic comparison across languages, retrieval methods, and model scales.
3. Some findings (e.g., English retrieval often helps non-English queries; diminishing returns of retrieval for large models) are practically informative.
4. Generally clear structure and presentation, with use of multiple benchmark datasets.

### Weaknesses
1. Query generation uses Llama-3.3-70B-Instruct while evaluation uses Llama-3.1-70B; these are nearly identical models, risking bias toward Llama-family strengths.

2. Basque and Kazakh data are machine-translated without expert verification, undermining the validity of “multilingual” claims.

3. Different methods provide unequal amounts of evidence (5 vs 10 vs 32 docs), so results may reflect information quantity, not retrieval quality.

4. The title’s “size not language” framing ignores that language effects remain strong, especially for low-resource languages.

5. No confidence intervals or significance tests despite small test size (125 items).

6. Paper reports patterns (e.g., retrieval hurts large models) but gives no causal insight into why.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
**Problem and Motivation:**
The paper addresses a significant gap in medical LLM research, which is overwhelmingly concentrated on English. This English-centric focus creates two primary problems: (1) It is difficult to generalize findings or apply models to other languages, and (2) nearly all expert medical knowledge bases (e.g., PubMed, StatPearls) are in English, making it unclear how to provide accurate, evidence-grounded answers for medical queries in other languages.
While scaling LLMs has shown they can memorize a substantial amount of domain knowledge (parametric knowledge), it is unknown how this internal knowledge interacts with external evidence, especially in a multilingual context. This work investigates whether LLMs' parametric knowledge is sufficient for multilingual medical QA or if external knowledge is required, and how the language of that external knowledge impacts performance.

**Method and Research Questions:**
This paper presents a systematic investigation of Multilingual Medical Question Answering (MMQA) across both high-resource (English, Spanish, French, Italian) and low-resource (Basque, Kazakh) languages.
The core methodology is to evaluate LLMs of varying sizes (from 8B to 72B parameters) under different knowledge conditions. The authors compare model performance using only internal (parametric) knowledge against performance when augmented with three distinct types of external evidence.
The study is guided by three main research questions:
- RQ1: Is there a single method (i.e., knowledge source) that consistently performs best across all languages?
- RQ2: How do retrieval quality and generation accuracy compare when using English as the evidence source versus using evidence in the target language?
- RQ3: Do LLMs encode sufficient internal (parametric) knowledge for optimal performance, or is external knowledge necessary?

**Knowledge Sources Investigated:**
The authors evaluate three distinct sources of knowledge to augment the LLMs:
- Local Knowledge Repositories: This uses a static, pre-existing English corpus from the MedExpQA benchmark. This corpus includes content from PubMed, Wikipedia, StatPearls, and medical textbooks. The top 32 relevant documents are retrieved for each question.
- Dynamic Web Search: This source uses dynamically retrieved web content. The authors use two different search APIs (Cohere and Google Search via Serper) to find relevant snippets. Retrieval is performed both in English and in the specific target language (e.g., Basque) for comparison.
- Parametric (LLM-Generated) Knowledge: This strategy tests the model's own internal knowledge. An LLM (Llama-3.3-70B-Instruct) is prompted to generate explanations and answers to the query without any external documents. This generated text is then used as the "evidence" for the final answer generation step.

**Experimental Setup:**
- *Dataset:* The study uses the test set (125 questions) from the CasiMedicos dataset, a multilingual, parallel collection of medical exam questions.
- *Languages:* The evaluation covers 6 languages; 4 high-resource (English, Spanish, French, Italian) and 2 low-resource (Basque, Kazakh). The low-resource versions were created for this study via translation with Claude-4-Sonnet.
- *Models:* 9 models from 3 families are tested: Qwen: 8B, 14B, 32B (Qwen 3), 72B (Qwen 2.5); LLaMA: 8B, 70B (Llama 3.1); Gemma: 12B, 27B (Gemma 3), and MedGemma (27B)

**Results & Key Findings:**
- Finding 1 (Best Overall Strategy): The most effective and consistent strategy across all languages (including high and low-resource) was using LLMs augmented with web-retrieved documents in English.
- Finding 2 (English Superiority): Retrieving evidence in English consistently surpassed the performance of language-specific retrieval.
- Finding 3 (Model Scale is Key): Larger models (e.g., Llama 70B, Qwen 72B) consistently outperformed smaller models in all settings.
- Finding 4 (RAG is Not Always Better): The paper challenges the idea that RAG is a uniform improvement. For large models (e.g., >70B) answering in high-resource languages, adding external knowledge (especially from static repositories or LLM-generated explanations) often degraded performance compared to the baseline (no retrieval). This suggests their strong internal (parametric) knowledge conflicts with or is not improved by the provided evidence.
- Finding 5 (Static Repositories are Insufficient): Locally stored, "authoritative" repositories like MedExpQA (PubMed, etc.) provided the least benefit. The results indicate these sources have incomplete domain coverage and inadequate multilingual support compared to dynamic web retrieval.

### Strengths
- The paper addresses an important problem: the intersection of multilinguality, model scaling, and external evidence integration for medical QA.
- The experimental design spans several languages and systematically compares static (MedExpQA), dynamic (web search), and parametric (LLM-generated) knowledge sources across a range of open-source LLMs of varying sizes.
- The work is data-driven, employing 6 languages, both high-resource and low-resource, moving beyond the more commonly explored English/French/Spanish scope.

### Weaknesses
- *Too small test set.* The paper's conclusions are drawn from a test set of only 125 questions. While the reviewer understands the difficulty of finding multilingual parallel medical QA benchmarks, this sample size is insufficient for making broad, generalizable claims, making the results highly susceptible to variance.
- *Use of unverified machine-translated data.* The low-resource languages (Basque, Kazakh) are not natively sourced but are "silver" translations from Claude-4-Sonnet. This introduces a major confounding variable: it is impossible to distinguish between a model's poor performance due to the language being low-resource and its poor performance due to processing potentially flawed, "unnatural" synthetic translation artifacts. The lack of any human verification or quality analysis of these translations, despite the small test set size, undermines the low-resource language findings.
- *Incomplete related work.* The investigation of "LLM-generated explanations" as a knowledge source is presented without a thorough comparison to existing, formal "generate-then-read" approaches in the medical domain (e.g., MedGENIE@ACL24). This omission weakens the framing of this part of the methodology.
- *Limited and asymmetrical exploratioin of knowledge sources.* The parameters for each knowledge source are narrowly explored. For instance, "Local Knowledge" is limited to the top 32 documents and, most importantly, is sourced only from an English corpus (MedExpQA). This creates an asymmetrical comparison, as it's impossible to know if the failure of "local" knowledge is due to it being static or due to the language mismatch. A fair comparison would have required testing local repositories in their respective target languages.
- *Retriever.* The retrieval similarity metric(s) for static document retrieval (Section 3.1). Is it cosine similarity, dense retrieval, BM25, or hybrid? How are top-k documents scored?
- *Decoding strategies.* The decoding parameters (e.g., temperature, top-p, greedy vs. sampling) for the LLMs are not specified, which is a critical variable in generation tasks.
- *Query generation.* The process for generating queries for the "WebSearch" and "Parametric" sources is not fully detailed.
- *Insufficient detail on statistical significance.* While standard deviations are reported (Table 2), there is limited description of test conditions, repeated runs, or variance across random seeds. The claim that certain results are “statistically significant” (e.g., observed boosts in specific configurations) cannot be robustly validated without description of experimental rigor.
- *Scope and generalizability.* The experiments, while broad linguistically, are focused on multiple-choice QA over clinical cases. It remains unclear how findings generalize to other medical tasks (e.g., summarization, free-form question answering, or patient-facing dialogue)—limiting the broader applicability for some portions of the medical AI community.
- *Absence of error analysis or qualitative case studies.* The paper’s narrative emphasizes quantitative gains or losses (see Table 2), but provides no systematic error dissection. There is a missed opportunity to illustrate the nature of “failure cases”—e.g., what types of questions in Basque or Kazakh are hurt by cross-lingual retrieval, or how content retrieved in a non-target language affects answer quality in practice. Likewise, no real qualitative or visual breakdown is included, undermining claims about “mismatched” or “noisy” evidence.
- *Presentation quality.* The paper's presentation quality is below the expected standard, with low-resolution figures, acronyms introduced multiple times, missing spaces before or after punctuation, wrong citation formats (\citet instad of \citep), etc.

### Questions
- Are results across Table 2 and Table 3 statistically significant under repeated runs or across random seeds? Could the authors clarify the number of repeated trials and provide confidence intervals or statistical testing for the main outcomes?
- While the authors state the code and data will be released, they fail to specify a license, which is a requirement for open-sourcing. What is the planned license for the public release of the code and resources?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the use of RAG for medical question answering across multiple languages. The paper uses an existing benchmark and automatically translates it into two different, low-resource, languages. The paper experiments with different methods to augment the questions with external information, such as CoT, web search, pubmed articles etc. The results do not appear very conclusive and expectedly, model size seems to be the best performance predictor, CoT and RAG across similar-sized models and even languages affect the performance marginally.

### Strengths
Studies on low-resource languages are generally important.

### Weaknesses
The contribution of the paper does not feel substantial - in the end, there is only one experiment, in which QA performance of 9 LLMs is evaluated with various prompting techniques (bar the results reported in Table 3, but they seem tangential to the paper's core investigation)

The results are inconclusive, and I also don't trust the differences given the small size of the test set - 125 questions. Reported results should be complemented by statistical significance analyses to confirm that the findings are indeed significant. Furthermore, I don't think that automatically translating the resources in other languages shows much more other than the effects of ["translationese"](https://www.cambridge.org/core/journals/natural-language-processing/article/emerging-trends-translationese/D39ADC5B44B06358A153F8926F88DD93), especially since the translations seems to not have been validated by L1/L2 speakers of said languages. The paper claims that "retrieving information in English for non-English questions" is better than retrieving information in the original language. Given that the "original language" questions are translated from English using an LLM, i am not at all surprised by that finding.

The writing is bad. Many claims are asserted without evidence, e.g. in lines 92-93: "common assumption in knowledge-augmented medical Question Answering: that adding external evidence uniformly improves performance". Who assumes that? Similarly, most of pages 7-9 is text, that makes no references to existing literature or other figures/tables, thus making it very hard to distinguish from being generated by an LLM.

### Questions
Please provide statistical significance tests to your results and substantiate your claims either by further analyses or by reference to existing literature.

### Soundness
2

### Presentation
2

### Contribution
2
