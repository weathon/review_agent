# Jackal: A Real-World Execution-Based Benchmark Evaluating Large Language Models on Text-to-JQL Tasks

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Enterprise teams rely on the Jira Query Language (JQL) to retrieve and filter issues from Jira.
Yet, to our knowledge, there is no open, real-world, execution-based benchmark for mapping
natural language queries to JQL. We introduce Jackal, a novel, large-scale text-to-JQL
benchmark comprising 100,000 natural language (NL) requests paired with validated JQL
queries and execution-based results on a live Jira instance with over 200,000 issues. To reflect
real-world usage, each JQL query is associated with four types of user requests: (i) Long NL,
(ii) Short NL, (iii) Semantically Similar, and (iv) Semantically Exact. We release Jackal, a
corpus of 100,000 text-to-JQL pairs, together with an execution-based scoring toolkit, and
a static snapshot of the evaluated Jira instance for reproducibility. We report text-to-JQL
results on 23 Large Language Models (LLMs) spanning parameter sizes, open and closed
source models, across execution accuracy, exact match, and canonical exact match. In this
paper, we report results on Jackal-5K, a 5,000-pair subset of Jackal. On Jackal-5K, the best
overall model (Gemini 2.5 Pro) achieves only 60.3% execution accuracy averaged equally
across four user request types. Performance varies significantly across user request types:
(i) Long NL (86.0%), (ii) Short NL (35.7%), (iii) Semantically Similar (22.7%), and (iv)
Semantically Exact (99.3%). By benchmarking LLMs on their ability to produce correct and
executable JQL queries, Jackal exposes the limitations of current state-of-the-art LLMs and
sets a new, execution-based challenge for future research in Jira enterprise data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents Jackal, the first large-scale, execution-based benchmark for evaluating large language models on the text-to-JQL task. It constructs 100,000 natural language–to–Jira Query Language (JQL) pairs validated on a live Jira instance with over 200,000 issues. Each JQL query is associated with four linguistic variants (Long NL, Short NL, Semantically Similar, and Semantically Exact) to reflect realistic query diversity. The benchmark evaluates 23 open and closed LLMs using execution accuracy as the primary metric, revealing that even the best models (e.g., Gemini 2.5 Pro) achieve only around 60 % overall accuracy, with strong sensitivity to phrasing.

### Strengths
S1. Execution-based realism. Jackal grounds evaluation in actual Jira executions, providing a reproducible and practical measure of correctness beyond superficial string matching.

S2. Scale and diversity. With 100 K pairs and four natural-language variants per query, it captures a wide spectrum of real-world phrasing styles rarely seen in prior text-to-DSL datasets.

S3. Comprehensive benchmarking. Evaluating 23 LLMs—spanning proprietary and open-source models—offers a valuable cross-model comparison and baseline for future text-to-DSL research.

### Weaknesses
W1. Examples of JQL appear too late in the paper.
The first concrete example only appears in Figure 2 and Section 3.2, while richer multi-clause JQLs are deferred to Appendix A (Table 2, p. 13). Most readers unfamiliar with JQL syntax cannot easily understand the benchmark’s linguistic or structural challenges until late in the paper. This weakens accessibility and motivation. A concise JQL primer with several example queries and their natural-language counterparts should be placed early (e.g., in the Introduction or Section 2) to help readers grasp what a JQL looks like and why translating to it is challenging.

W2. Limited discussion of what makes text-to-JQL hard and how prompting was handled.
Although the authors note that JQL involves custom fields, permissions, and linked-issue traversals, they do not deeply explain why mapping text to such constructs is particularly hard for LLMs. Moreover, the paper lacks clarity about the prompting setup during evaluation—what context (schema, field values, example queries) was given to each model. Without specifying whether models saw project-specific schemas or value enumerations, it is difficult to interpret why models fail on Short or Semantically Similar variants. Providing concrete prompt examples and an ablation on context richness would strengthen the technical depth

W3. Unclear differentiation from text-to-SQL..
Section 2.5 briefly mentions that JQL differs from SQL due to non-relational data, permissions, and linked-issue graphs, but the paper never empirically contrasts the two tasks. The benchmark could have highlighted distinct linguistic or structural phenomena (e.g., relative dates, visibility filters) through comparative analysis or shared-metric baselines. Without such evidence, the contribution risks appearing as a narrower variant of existing text-to-SQL work. For example, although text-to-SQL and text-to-visualization are very different tasks, techniques for them based on LLMs are very similar.

W4. Missing comparison with simple RAG or retrieval-augmented baselines.
Many of the difficult Short or Semantically Similar examples could plausibly be solved with retrieval of schema descriptions or example JQL templates, yet no RAG baseline is tested. A simple retrieval-plus-generation pipeline could provide a meaningful reference point, revealing whether LLM failures stem from lack of schema memory or reasoning ability. Including such baselines would clarify whether the challenge lies in recall or compositionality.

W5. Lack of broader generalization analysis and verifiability.
All evaluations are conducted on a single Jira instance snapshot, which raises two interrelated concerns: generalization and reproducibility. Because each Jira instance has its own customized schema, permission structure, and field vocabulary, results derived from one snapshot may not hold for other configurations. Without experiments on multiple instances or unseen project schemas, the paper cannot convincingly claim model generalizability.

Moreover, although the authors promise to release a static snapshot, the dataset and benchmark code are not yet publicly available. This makes it impossible for reviewers to independently verify the reported execution accuracies or replicate the evaluation pipeline. For a benchmark paper, reproducibility and openness are central to credibility. Unless the data, schema snapshot, and execution toolkit are released at submission time, the community has no way to trust that the reported results are accurate or that the benchmark functions as described.

### Questions
See the above weaknesses W1-W5.

### Soundness
3

### Presentation
2

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
Jackal consists of 100,000 real-world natural language (NL) and JQL query pairs validated on a live Jira instance containing over 200,000 issues. Each JQL query is accompanied by four natural language variants—Long NL, Short NL, Semantically Similar, and Semantically Exact—to reflect diverse enterprise user intents. The benchmark includes an execution-based scoring toolkit and a reproducibility package.

### Strengths
By establishing a reproducible and realistic evaluation framework for enterprise query translation, Jackal bridges an important gap between academic benchmarks and production-grade tasks.

### Weaknesses
Although the dataset is described as real-world, the queries are generated programmatically and paraphrased by LLMs rather than collected from actual users. This may limit linguistic authenticity and introduce model biases into the data.

Given that JQL can differ across organizations due to custom fields and permissions, it remains unclear how models trained on Jackal would generalize to unseen instances. A cross-instance validation experiment, even at smaller scale, would make the benchmark more robust.

### Questions
The dataset relies heavily on LLM-generated paraphrases. How do the authors ensure that the test data are not inadvertently contaminated by model-generated biases that might favor certain evaluated LLMs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents Jackal, a benchmark for natural-language-to-Jira Query Language (JQL) translation. The dataset includes 100K synthetic NL–JQL pairs generated with an LLM and validated on a live Jira instance. The authors evaluate 23 LLMs on a 5K subset (Jackal-5K) using execution accuracy, exact match, and canonical exact match. The key finding is that even the best proprietary models achieve only ~60% execution accuracy, exposing the difficulty of text-to-JQL translation. While the dataset is large and practical, the paper’s contribution is primarily engineering rather than research.

### Strengths
- S1. Constructing an execution-based benchmark for JQL is a useful community resource.

- S2. The dataset appears systematically filtered and validated, with reproducibility in mind.

- S3. The empirical evaluation covers a broad range of models and gives an approximate performance landscape.

### Weaknesses
- W1 (No novelty). The benchmark design directly mirrors Text-to-SQL datasets (Spider, BIRD, Spider 2.0). There is no conceptual or methodological innovation—only a domain change.

- W2 (Synthetic data). All natural-language queries are generated by an LLM and only lightly spot-checked. This makes the dataset less reliable for linguistic or semantic evaluation.

- W3 (Weak analysis). The results section is mostly leaderboard numbers; there is no meaningful error taxonomy, model insight, or analysis of what linguistic phenomena cause failure.

- W4 (Circular evaluation risk). Because an LLM generated the NL queries and another LLM is tested on them, there may be style leakage or artificial easiness/hardness not discussed.

- W5 (Lack of research contribution). The paper reads like a technical report. There’s no new algorithm, learning strategy, or evaluation framework that would justify inclusion in a top ML venue.

### Questions
- Q1. How diverse are the LLM-generated natural-language variants? Any quantitative measure of linguistic diversity or entropy across prompt types?

- Q2. Are the results reproducible if the Jira instance evolves (e.g., issue deletions or field changes)?

- Q3. Could the same methodology be applied to any DSL trivially? If yes, what scientific insight does JQL bring?

- Q4. Did you analyze whether models simply memorize structural patterns of JQL rather than truly reasoning about semantics?

### Soundness
3

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
3

### Summary
The paper introduces "Jackal," a novel, large-scale, execution-based benchmark for evaluating Large Language Models (LLMs) on the task of translating natural language to Jira Query Language (JQL). It effectively identifies a significant gap in existing resources and presents a comprehensive solution, complete with a dataset, an evaluation toolkit, and a reproducibility package. The empirical evaluation of 23 LLMs is thorough and reveals critical limitations in current state-of-the-art models.

### Strengths
1. Fills a Critical Research Gap: The paper's primary contribution is addressing the notable absence of a high-quality, public benchmark for Text-to-JQL. It correctly identifies the limitations of existing small-scale, non-execution-based datasets. By providing 100,000 validated NL-JQL pairs, execution results on a live Jira instance with over 200,000 issues, and a full reproducibility package, Jackal enables rigorous, open research in an area with significant real-world enterprise relevance.

2. Well-Designed and Realistic Benchmark Framework: The benchmark construction is thoughtful and reflects real-world usage: 1) The inclusion of four distinct user query variants (Long NL, Short NL, Semantically Similar, Semantically Exact) effectively captures the spectrum of how users naturally formulate requests, moving beyond literal translations. 2) The emphasis on execution accuracy as the primary metric is methodologically sound and aligns with best practices in semantic parsing, correctly avoiding the pitfalls of string-based matching (EM, CEM), which the results show to be nearly useless. 3) Grounding the evaluation in a real, large-scale Jira instance ensures the queries are valid and the results are practically meaningful.

3. Comprehensive and Revealing Empirical Evaluation: The scale of the model evaluation is impressive, covering 23 proprietary and open-weight LLMs across various sizes. The results are compelling and clearly demonstrate that even the best models (e.g., Gemini 2.5 Pro at 60.3% accuracy) struggle significantly, especially with under-specified (Short NL) and paraphrased (Semantically Similar) queries. This analysis provides a valuable snapshot of the current capabilities and limitations of LLMs for this specific enterprise task.

### Weaknesses
1. Limited Technical Novelty: The paper's core contribution is the benchmark itself rather than a novel methodological advancement. It applies established practices from Text-to-SQL (execution-based evaluation, multiple query types) to a new domain (JQL). The error analysis, while useful, remains somewhat high-level and does not delve deeply into the root causes of failure modes or propose specific, novel technical solutions beyond suggesting known directions like constrained decoding.

2. Limitations in Evaluation Scope and Analysis: 1) The decision to report results only on the 5,000-pair "Jackal-5K" subset, while understandable for initial publication, underutilizes the full 100,000-pair dataset. A scaling analysis or results on the full set would have been more powerful. 2) The study focuses exclusively on LLMs. A comparison with older, rule-based or specialized semantic parsing approaches for JQL would have provided better context for understanding the absolute and relative progress represented by LLMs. 3) There is no discussion of the computational cost or latency involved in the execution-based evaluation, which is a practical concern for real-world deployment and future research using this benchmark.

### Questions
1. will the execution of jql spend a lot of time?
2. in text-2-sql, there are many stages, for example, schema linking, could your benchmark provide more studies on this like previous text2sql methods? e.g.[1,2]

[1] Benchmarking the text-to-sql capability of large language models: A comprehensive evaluation.  
[2] Pet-sql: A prompt-enhanced two-round refinement of text-to-sql with cross-consistency

### Soundness
3

### Presentation
3

### Contribution
3
