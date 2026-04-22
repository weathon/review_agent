# Poly-FEVER: A Multilingual Fact Verification Benchmark for Hallucination Detection in Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 8, 4

## Abstract
We present Poly-FEVER, a large-scale multilingual benchmark for fact verification and hallucination detection in large language models (LLMs). Poly-FEVER extends FEVER, Climate-FEVER, and SciFact to 77,973 labeled claims across 11 languages—English, Chinese, Hindi, Arabic, Bengali, Japanese, Korean, Tamil, Thai, Georgian, and Amharic—curated to preserve logical equivalence across scripts and morphologies and to focus on verifiable Supported/Refuted cases. We augment each claim with topic metadata derived from a 22-topic LDA model to enable topic-aware evaluation. Claims are translated using Google Cloud Translation and validated with GEMBA scores averaging~90 across languages, supporting high semantic fidelity. Using Poly-FEVER, we benchmark ChatGPT-3.5, LLaMA‑2 (7B/13B/70B), and LLaMA‑3.1‑8B under language-wise, general, and classification prompt families, and study self-detection via rephrasing. We further probe resource imbalance by correlating accuracy with automated web presence (Google hit counts) and test retrieval-augmented generation via DPR over Wikipedia. Results show pronounced cross-lingual disparities: high-resource languages (e.g., English, Chinese) achieve the strongest accuracy, while lower-resource languages (e.g., Amharic, Tamil) lag; accuracy correlates with web presence. Topic structuring consistently benefits lower-resource settings, and RAG provides selective gains (notably in Arabic and Amharic), but can conflict with strong internal priors in high-resource languages. Poly-FEVER establishes a rigorous, publicly available foundation for responsible, language-adaptive evaluation of hallucination mitigation in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents Poly-FEVER, a large-scale multilingual benchmark designed for fact verification and hallucination detection in large language models. The benchmark builds on three existing English datasets by extending them by translating their verifiable claims into ten additional languages. In total, Poly-FEVER covers 11 languages that span a wide range of resource levels, writing systems, and linguistic families, providing a rich and diverse testbed for evaluating factual consistency across languages.

### Strengths
The paper’s main strength lies in the Poly-FEVER benchmark itself. Its construction is methodical—built by extending well-established datasets, ensuring high-quality translations, and incorporating GEMBA validation. The design, featuring parallel claims across 11 linguistically and typologically diverse languages, addresses a clear and well-motivated gap in the existing fact-verification literature.

### Weaknesses
Maybe I misunderstood, but there seems to be an inconsistency between the caption of Table 3 and the explanation in Section 5.5. The table caption describes the experiment as measuring the “percentage improvement … after translating nuanced topics into English,” whereas Section 5.5 (around line 441) refers to accuracy gains from topic structuring. These appear to be two different procedures, so clarification on which setup the results actually correspond to would be helpful.

### Questions
See the problem I raised in the weakness. I hope the authors could answer the question I have there.

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
4

### Summary
This paper presents Poly-FEVER, a new dataset for multilingual hallucination detection. The dataset is constructed by turning subsets of three fact verification datasets, FEVER, Climate-FEVER, and SciFact, into document-free general claim evaluation tasks and translating the claims into 11 languages using Google Translate, with translations validated using automatic methods (GEMBA). The paper evaluates GPT3.5 and LLama-2/3 families of models and presents results broken down by language and topic model-derived topics. They show that models generally perform much worse on non-English languages. They also explore correlation of performance with web search count.

### Strengths
S1. The motivation is important and interesting for multilingual researchers looking at how hallucination changes across languages. The dataset could be of use for future researchers.

S2. The exploration of how topics relate to claim verification is scientifically interesting and novel to the best of my knowledge.

### Weaknesses
W1. Some of the results in the main result (5.3) stand out for being substantially below random chance (50%), suggesting that some of the LLMs may be biased towards one of the two labels. E.g. 19.47% and 26.34%by LLaMA-2 70B. That these performances are so low on a binary classification task suggests something went wrong with the evaluation. 

W2. I am not convinced it is valid to cast document-conditioned verification tasks into a documentless hallucination detection task, especially SciFact. Scientific claims, especially those in that dataset, are ambiguous by nature and require grounding context to be evaluatable. Removing the context makes the task nearly impossible for many statements.

W3. From Figure 4 it is claimed that "models possibly favoring languages that dominate web content, affecting their accuracy in languages with less online presence." However, while English (the original source language) is very high and the lowest-presence language, Tamil, is very low, the rest of the results do not back up this claim; Japanese has the second highest overall accuracy despite havinga lower search count than Tamil, the lowest-performing language. Further analysis of this figure, including regression analysis to verify the trend, would be beneficial, though I would suggest removing the result because it does not support the hypothesis.

W4. The analysis of topic-wise performance in L426-434 is speculative and not backed up by evidence or citations. Claims like "absence of universal standards in evaluating greatness in sports or the arts further complicates claim verification" and "Topics such as Politics, Sports, Film/Television, and Warfare History prove challenging across languages due to their subjective nature, where personal biases and interpretations can obscure the distinction between fact and opinion" should be worded carefully to not be interpreted as scientific findings of the study. 

W5. Similarly, L.443 "topic structuring provides additional contextual grounding, which helps compensate for weaker internal representations in lower-resource languages." and L 445 "[For Tamil] indicating that explicit topic awareness does not always align with the model’s existing representations or may interfere with its learned knowledge." -- I do not think that the presented results support either of these claims, the former of which is not consistent across all topics and languages. There is no evidence of whether low performance on the benchmark correlates with weaker internal representations, since there is no investigation of the internal representations or what it would mean to "align" with them.  

W6. Table 3 is hard to interpret; the caption seems to suggest it is about translating back from the source language into English (the claim's original language) and then measuring factuality, but the text (L441 to 449) talks about LDA topic structuring 

W7. I am not convinced that some of the ablation studies provide scientific value. 
	* Specifically, Section 5.2, the study of temperature, does not make sense to me for a binary classification task. Raising temperature effectively just adds random noise to the prediction by adding to the probability of the less-likely output. In nearly every row of Table 7, you can see that raising temperature just brings wherever the performance was at temperature 0.1 closer to chance 50%). Rows that were below 50% increase and those that were above decrease.

W8. Manual auditing of the translations that produced the dataset seems necessary but limited. Paragraph L.228-232 describes that the authors are "fluent in multiple languages" and reviewed selected claims from multiple translators, but no details about how many claims, how many annotators per claim, etc are provided. It is also hard to understand a GEMBA score of ~90 in the context of building a dataset; how do we know that an average of 10% worse than perfect translations is sufficient to ensure dataset quality? How long was the tail of this score distribution? Why not throw out the translations that scored below some (high) threshold? 

W9. Figure 2 is dense, hard to understand, and doing too much. It seems to illustrate all the case studies at once but ultimately does not help to understand the paper (The caption also does not explain what the figure is meant to be; is it an overview of the case studies? )

### Questions
Q1. What LLM(s) were used for tables 3 and 4? Was it evaluated over the entire 77.9K * 11 claim dataset? What explains the massive improvement on Arabic and Amharic just by adding topic labels? This is a counterintuitive finding that merits understanding better. 

Q2. Some pieces of language are hard to understand, e.g. "All results are illustrative and reproducible but not tuned." -- what does "illustrative" mean? 

Q3. Paragraph 316-320 is confusing and doesn't seem relevant to the question of web search hit counts. Why do we need to "simulate varied internet user environments?"  

Q4. How does performance break down across the 3 subsets of the dataset? FEVER/Climate-FEVER/SciFact

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce PolyFEVER, a multilingual benchmark for fact verification that spans 11 languages. The authors evaluate a set of foundation models and compare their results on the benchmark against language web presence and other variables. The multilingual data is generated by translating English text. The paper is an important step towards AI fairness across high- and low-resource languages.

### Strengths
- The argument for developing a multilingual fact verification benchmark is very compelling, and it is highlighted by the results showing that lower-resource languages correlate with worse overall performance on the task. We need more papers in the field tackling this sort of problem, and this paper takes a good stab at enabling this line of work.

- Figure 1 is a nice addition that contextualizes the problem well. This language balance is reasonable and helpful, as is the balance among different topics.

- The presentation of results is comprehensive and informative. The RAG experiments over Wikipedia in particular are interesting.

- The paper is well-written.

### Weaknesses
- It would be ideal to include claims originally written in the target languages instead of translating English claims. Translated claims introduces bias, and ignores nuances that would otherwise be modeled if drawing from content that was originally written for audiences who speak these languages.

- The language models used for evaluation are a bit old, especially GPT-3.5. This isn't a critical issue, but the paper may be seen as more timely and receive better reception if some of the experiments are updated to include contemporary models released in 2025.

- Fact verification is intrinsically difficult as there are many natural language claims that are not strictly true or false. This paper partially addresses this by adopting the FEVER approach of including a "not enough info" label, but this fails to capture the nuance of partially subjective or contextual claims. A discussion of this concept could enhance the paper.

### Questions
- It would be nice to include more of a discussion w.r.t. future work on methods for this benchmark.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Poly-FEVER which is a multilingual benchmark extending FEVER, Climate-FEVER, and SciFact to 77,973 labeled claims across 11 languages for evaluating hallucination detection in LLMs. The authors translate English claims using Google Translate, validate translations with GEMBA scores, apply LDA for topic modeling (22 topics), and benchmark several LLMs (ChatGPT-3.5, LLaMA-2, LLaMA-3.1-8B) under various prompt designs. They investigate correlations between accuracy and web presence (Google hit counts) and test retrieval-augmented generation (RAG) using DPR over Wikipedia. Results show performance is difference in high-resource languages, with topic structuring benefiting from lower-resource settings and RAG providing mixed results.

### Strengths
1. The paper introduced 77,973 claims across 11 typologically diverse languages is substantial includes high-to low-resource settings and multiple scripts (logographic, alphabetic, abjad). The scale and coverage is fair.
2. Including LDA topic distributions as metadata enables topic-stratified analysis, which is underexplored in prior multilingual NLP benchmarks.

### Weaknesses
1. experimental design is a little confused: THe work compared accuracy across languages conflates translation quality, cultural bias in source data (Wikipedia), LLM pretraining distributions, and linguistic properties. Only attributes gaps to resource imbalance but does not control for claim difficulty, domain familiarity, or translation errors.
2.Section 5.4 correlates web hits with accuracy but does not establish causation. Alternative explanations (e.g., translation errors more frequent in low-resource languages) are not ruled out.
3. It Seems Retrieving only English Wikipedia for all languages (Appendix B.2) is a critical flaw. Multilingual Wikipedia or language-specific corpora would be more appropriate.

### Questions
Why retrieve English Wikipedia for non-English claims? Did you experiment with multilingual retrieval or language-specific corpora?
Have you controlled for translation quality when correlating web hits with accuracy? Could low-resource languages simply have worse translations that introduce label noise?

### Soundness
2

### Presentation
3

### Contribution
2
