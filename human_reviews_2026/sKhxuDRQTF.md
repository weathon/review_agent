# Evaluating the Retrieval Robustness of Large Language Models

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Retrieval-augmented generation (RAG) generally enhances large language models' (LLMs) ability to solve knowledge-intensive tasks. But RAG could also lead to performance degradation due to imperfect retrieval and the model's limited ability to leverage retrieved content.
In this work, we evaluate the robustness of LLMs in practical RAG setups (henceforth retrieval robustness). We focus on three research questions: (1) whether RAG is always better than non-RAG; (2) whether more retrieved documents always lead to better performance; (3) and whether document orders impact results.
To facilitate this study, we establish a benchmark of 1500 open-domain questions, each with retrieved documents from Wikipedia. We introduce three robustness metrics, each corresponds to one research question. 
Our comprehensive experiments, involving 11 LLMs and 3 prompting strategies, reveal that all of these LLMs exhibit surprisingly high retrieval robustness; nonetheless, different degrees of imperfect robustness hinders them from fully utilizing the benefits of RAG.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces an evaluation framework for the "retrieval robustness" of large language models (LLMs) used in Retrieval-Augmented Generation (RAG) systems. It investigates if RAG always improves performance and how results are affected by the number and order of retrieved documents. To do this, it contributes a new benchmark of 1500 open-domain questions and three novel robustness metrics: no-degradation rate, retrieval size robustness, and retrieval order robustness. Comprehensive experiments across 11 LLMs reveal that while models are surprisingly robust (often >80% on these metrics), their imperfect robustness creates sample-level trade-offs, forfeiting RAG's potential gains. The study concludes that improving this robustness is a key challenge that cannot be solved by model scaling alone.

### Strengths
This paper has the following strengths:

- It formalizes "retrieval robustness" through a novel set of sample-level metrics. By proposing concrete measures for no-degradation, retrieval size, and document order robustness, it provides the community with a rigorous and standardized methodology to empirically quantify how LLMs handle the inevitable imperfections of real-world RAG pipelines.

- It creates a practical benchmark specifically designed for evaluating RAG robustness. This benchmark is a valuable resource, as it mirrors common RAG setups (using diverse open-domain QA and strong, widely-used retrievers), enabling standardized testing, replication, and fair comparison of different models and retrieval strategies.

- It conducts a comprehensive empirical study covering 11 modern LLMs and 3 prompting strategies. This broad analysis provides an insightful snapshot of the current landscape, establishing that while LLMs are surprisingly robust, their "imperfect robustness" remains a critical bottleneck that limits RAG's potential and presents a key challenge for future research.

### Weaknesses
This paper has the following weaknesses:

- The paper fails to discuss or compare against "Astute RAG" [1], a highly relevant study that also explores imperfect retrieval robustness and leverages the model's intrinsic knowledge. This omission makes it difficult to assess the paper's novel contributions.

- The rationale for curating a new, small-scale benchmark (1500 samples) from existing datasets is unclear, as is the advantage over using the original, larger datasets. The limited sample size raises concerns about the generalizability and statistical robustness of the empirical findings.

- The reliability of the results is questionable, as they depend on Llama3.3-70B-Instruct for correctness judgments. Using a model that is not state-of-the-art for evaluation may introduce significant noise and inaccuracies into the core metrics.

[1] Wang, Fei, et al. "Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models." arXiv preprint arXiv:2410.07176 (2024).

### Questions
- Regarding related work, could the authors please discuss "Astute RAG"? That paper also seems to address imperfect retrieval and the use of a model's internal knowledge, so a direct comparison would be necessary to clarify this paper's specific novel contributions.

- Could the authors expand on the rationale for creating a new 1500-sample benchmark? It's unclear why this was preferable to using the original, larger datasets, and there are concerns that this limited scale may not be sufficient to support statistically robust or generalizable conclusions.

- The core metrics depend on Llama3.3-70B-Instruct for correctness judgments. Given that this model is not state-of-the-art and may introduce evaluation errors, what steps were taken to validate its reliability as a judge? Have the authors considered using a more advanced LLM or human evaluation to confirm the accuracy of the computed metrics?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to evaluate the retrieval robustness of LLMs in "practical RAG setups". The authors construct a benchmark based on the Wikipedia corpus and real retrievers (BM25, BGE) and propose three new robustness metrics (NDR, RSR, ROR). The paper presents a comprehensive experimental analysis of 11 mainstream LLMs across different retrieval sizes, orders, and prompting strategies.
Overall, this work proposes a set of valuable evaluation metrics and conducts a thorough and insightful empirical study around a specific clean corpus setting.

### Strengths
1. The most significant strength of this paper is its exhaustive experimental design. The authors systematically investigate the robustness of 11 mainstream LLMs under a natural retrieval setting using the Wikipedia corpus. The study covers a wide range of variables and the experiments are thorough. Compared to previous benchmark work that focused on "artificially" constructing various types of adversarial noise, this paper's setup (strong LLM + natural top-K retrieval) helps us, in the current stage of LLM development, to re-evaluate the robustness of SOTA LLMs in retrieval-based QA, especially in "knowledge conflict" scenarios (i.e., where their own parametric knowledge conflicts with imperfect retrieved knowledge).

3. The authors devise three reasonable and meaningful robustness metrics (NDR, RSR, ROR) that closely correspond to their research questions. The experimental analysis based on these metrics reveals several interesting conclusions, for example:
    - For some models, the impact of the NDR metric is quite significant. Furthermore, some model families (like Llama) exhibit a trend where larger models show worse NDR robustness compared to smaller ones. I think NDR can be a robustness metric for evaluating base model.
    - The experiment finds that for most models the reversed document order yields better results. This aligns with findings on recency bias in prompt engineering and is a practical discovery.
    - The experiment in Section 5.5 on different prompting strategies is also interesting, although I have a different interpretation of the results than the authors (see "Questions").

### Weaknesses
1. I think there is a lack of rigor in contribution positioning and over-claiming in this paper:
    - The authors' core argument is that "previous benchmark work used a large amount of artificially synthesized/constructed noise data, and is therefore not realistic". However, this argument ignores that the "real internet" retrieval environment is itself flooded with a large amount of noise, errors, and misinformation. From this perspective, previous work (e.g., inserting counterfactual noise) can be seen as a method of simulating this "real internet noise" in a controlled laboratory environment.
Conversely, the "pure Wikipedia" retrieval corpus constructed by this paper—a highly fact-checked and relatively "clean" corpus—is, in fact, a less realistic scenario for simulating open internet retrieval. A model retrieving from the real internet cannot possibly consider only clean Wikipedia text while completely ignoring other noisy websites.
    - I think in work pre-dating 2023 (i.e., before the rise of strong LLMs), evaluation under a "natural retrieval top-k" setting would not be novel. Does this paper clearly articulate its true position within the developmental timeline of RAG evaluation? A more accurate contribution of this paper might be positioned as: "Re-visiting the 'natural retrieval' setting in the era of strong LLMs, using new metrics (like NDR) to evaluate the resulting knowledge conflict problem". 

2. The paper's evaluation focuses primarily on "non-thinking" models. However, current SOTA LLMs' robustness have been widely boosted with their thinking abilities. I think including more thinking-based SOTA LLMs in the experiments and analysis would benefit the paper and improve the validness.

### Questions
1. In Section 5.5, the authors conclude that "using OwnKnow might limit the maximum performance models can possibly achieve". I have a different opinion: observing the results, for smaller models, OwnKnow generally shows a decrease in performance, but for larger and better models, OwnKnow often leads to a performance improvement. I find this point to be intuitive and reasonable. As the model's capability enhances, it can rely more on its own knowledge, or in other words, when the model has stronger discernment ability, referencing its own knowledge should always be beneficial. This is analogous to how humans with a certain degree of cognition and discernment understand the world / answer questions based on reference materials.

2. If the author can better declare their contribution and well address my concerns in Weakness point 1 with rigorous justification, I will adjust my assessment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies how reliably large language models perform in retrieval-augmented generation (RAG) under realistic settings. The authors build a benchmark of 1,500 open-domain QA samples with Wikipedia retrievals and define three metrics: No Degradation Rate, Retrieval Size Robustness, and Retrieval Order Robustness. The authors aim to measure whether RAG consistently helps, scales with more documents, and remains stable across document orders. Testing 11 LLMs with different prompting strategies, they find that most models are robust in practice but still show inconsistencies that limit RAG’s full benefits. The work offers a practical benchmark, clear robustness metrics, and comprehensive insights for improving dependable RAG systems

### Strengths
1) The paper defines three clear quantitative metrics—No Degradation Rate, Retrieval Size Robustness, and Retrieval Order Robustness—that formalize intuitive aspects of retrieval robustness in RAG systems.

2) It provides a practical benchmark of 1,500 open-domain QA samples using Wikipedia retrievals and two standard retrievers (BM25 and BGE), offering a reproducible setup grounded in real-world retrieval conditions.

3) The experimental coverage is broad, evaluating 11 LLMs and three prompting strategies, giving a comprehensive empirical picture of current RAG robustness across models.

### Weaknesses
1) The novelty is limited. The core idea is primarily a large-scale empirical study rather than a methodological or theoretical innovation. The proposed metrics formalize well-known intuitions but do not introduce new techniques or insights into why robustness varies.

2) The benchmark focuses only on open-domain QA with Wikipedia, which restricts generalization to specialized domains or other RAG use cases such as reasoning, dialogue, or summarization.

3) The evaluation relies heavily on one judging model (Llama-3.3-70B-Instruct), yet potential biases or inconsistencies across evaluators are not analyzed. No statistical significance tests are reported

4) The analysis of retrieval size and order is shallow. It tests only three orderings (original, reversed, shuffled) and does not explore learned or adaptive re-ranking, nor deeper interactions between retrieval noise, context length, and answer quality.

5) The reproducibility and release timeline are unclear since code and scripts are promised only after acceptance, preventing full verification during review.

### Questions
- How consistent are robustness scores when using different evaluator models? Have you checked for systematic bias toward any LLM family?

- Why were only three simple document orderings tested? Would a learned or heuristic re-ranking strategy affect the results?

- Can you provide finer-grained analyses of which samples lose performance under larger retrieval sizes or different orders?

- What are the trade-offs of the OwnKnow and S2A prompting strategies across models and datasets?

- How sensitive are the proposed metrics to evaluation noise or variation in the scoring model? Can you report confidence intervals or variance across repeated runs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a sample-level metric to measure retrieval robustness and conducts insightful analysis. The paper is easy to follow. However, the practical utility is limited by a relatively narrow experimental setting focused on knowledge-intensive QA with Wikipedia knowledge base. Another question is can the proposed metrics be enhanced to better capture the observed performance trade-offs across individual samples?

### Strengths
1. The paper proposes a sample-level metric to measure retrieval robustness, which is highly relevant for the practical deployment of RAG systems. The research questions are well-chosen and address core concerns of RAG practitioners. Comprehensive experiments are conducted with 11 modern LLMs with 3 prompting strategies. The work gives insight of how modern LLMs react to external information with various quality.

2. Detailed experiment design! I really appreciate the analysis shown in Sec.5.3 "models keep trading off performance across individual samples...imperfect robustness on retrieval size.". (Also Sec.5.4)

3. Paper writing is easy to follow.

### Weaknesses
1. The experiment scopes remain limited on 1) knowledge-intensive open-domain QA tasks. 2) limited knowledge base for retrieval. The two limits limit the practical benefit of the research. Robustness on more domains and more various settings are encouraged. E.g., how the three metrics will be on Google Search? How the LLMs will behave if the queries are a blend of knowledge-intensive ones and non-knowledge-intensive ones. The good robustness of models on knowledge-intensive queries with Wikipedia as knowledge base is natural.

### Questions
1. As stated in Sec.5.3 and 5.4, cases where "...hurting performance on some examples while gaining performance on others." are observed. Is there a way to enhance the three metrics NDR, RSR and ROR, to include this phenomenon?

### Soundness
3

### Presentation
4

### Contribution
3
