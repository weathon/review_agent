# DeepScholarBench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis

- Decision: Reject
- Scores: 4, 4, 4, 8, 6, 4

## Abstract
The ability to research and synthesize knowledge is central to human expertise and progress. A new class of AI systems—designed for generative research synthesis—aims to automate this process by retrieving information from the live web and producing long-form, cited reports. Yet, evaluating such systems remains an open challenge: existing question-answering benchmarks focus on short, factual answers, while expert-curated datasets risk staleness and data contamination. Neither captures the complexity and evolving nature of real research synthesis tasks. We introduce DeepScholar-bench, a live benchmark and automated evaluation framework for generative research synthesis. DeepScholar-bench draws queries and human-written exemplars from recent, high-quality ArXiv papers and evaluates a real synthesis task: generating a related work section by retrieving, synthesizing, and citing prior work. Our automated framework holistically measures performance across three key dimensions—knowledge synthesis, retrieval quality, and verifiability. To further future work, we also contribute DeepScholar-ref, a simple, open-source reference pipeline, which is implemented on the LOTUS framework and provides a strong baseline. Using DeepScholar-bench, we systematically evaluate prior open-source systems, search agents with strong models, OpenAI’s DeepResearch, and DeepScholar-ref. We find DeepScholar-bench is far from saturated: no system surpasses a geometric mean of 31% across all metrics. These results highlight both the difficulty and importance of DeepScholar-bench as a foundation for advancing AI systems capable of generative research synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new benchmark and evaluation framework for assessing related work writing in research papers. The benchmark includes 63 recently published papers spanning 18 distinct domains. The evaluation framework measures performance on seven fine-grained metrics covering knowledge synthesis, retrieval quality, and verifiability. Experimental results reveal the challenging nature of this task, as no system achieves a geometric mean exceeding 31% across all metrics.

### Strengths
1. This is the first paper to evaluate the related-work writing capabilities of the Deep Research system.

2. The proposed evaluation framework provides comprehensive coverage of key aspects.

3. The paper is clearly written and easy to follow.

### Weaknesses
1. In lines 174–175, the notation description is somewhat confusing. I suggest placing each symbol immediately after its corresponding concept (e.g., S should directly follow “a set of relevant sources” rather than being separated by a comma). Additionally, consider separating different notations and their descriptions with semicolons to avoid reader confusion.

2. For quotation marks, please use ``` `` ``` and ``` '' ``` to represent the left and right quotation marks, respectively.

3. If an error analysis were included here—identifying the causes and proportions of the error cases—it would provide valuable insights for future work.

### Questions
No

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
3

### Summary
This paper introduces DeepScholar-bench, a live benchmark for Generative Research Synthesis systems. The authors argue that existing benchmarks are ill-suited, as they are often static or focus on short-form answers. DeepScholar-bench addresses this by using recent ArXiv papers as a data source. The task is to generate a related work section given a paper's abstract. The framework evaluates systems on three dimensions: Knowledge Synthesis, Retrieval Quality, and Verifiability. The authors also provide an open-source baseline, DeepScholar-ref. An evaluation of systems shows the benchmark is far from saturated, with no system surpassing a 31% geometric mean score.

### Strengths
1. The paper tackles the highly significant problem of evaluating complex GRS systems. The core idea of using recent ArXiv papers as a live and continuously updating data source is novel and effectively mitigates data staleness and contamination.
2. The three-pillar evaluation framework is comprehensive and well-justified. The fine-grained metrics allow for a nuanced analysis of system capabilities
3. The thorough empirical study establishes the benchmark's difficulty, and the inclusion of the DeepScholar-ref baseline is a valuable contribution for future research

### Weaknesses
1. The use of human-written related work sections as exemplars is a key weakness. Human authors are constrained by page limits and their own limited knowledge. This means the exemplar may be incomplete, which unfairly penalizes an AI system that performs better retrieval or synthesis than the human author. This specifically impacts the validity of the Retrieval Quality and Verifiability metrics.
2. While the live aspect is valuable for GRS, the live aspect have explored in other domains [1,2,3], which should be acknowledged.
3. Using a paper's full abstract as the input query  does not seem to reflect a realistic user scenario for a GRS task. A user would more likely start with a research question or topic, not a completed abstract. This design choice may impact the generalizability of the results.

[1] HOH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation
[2] PAT-Questions: A Self-Updating Benchmark for Present-Anchored Temporal Question-Answering
[3] MAC: A Live Benchmark for Multimodal Large Language Models in Scientific Understanding

### Questions
1. Given that human exemplars are imperfect, how does your evaluation framework account for a system that finds relevant, important papers missed by the human author? Would this not unfairly penalize the system on metrics like Reference Coverage?
2. Why was the full abstract chosen as the input query? How do you think system performance would differ if the prompt was a more realistic, shorter input, such as the paper's title or a one-sentence research question?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents DeepScholar-Bench, a live and automated benchmark designed to evaluate large-language-model systems that perform generative research synthesis—that is, retrieving, analyzing, and writing related-work sections with proper citations. The benchmark constructs monthly datasets from recent arXiv papers across 18 computer-science domains (63 papers in the June-2025 release), pairing each paper’s abstract with its human-written related-work section as an exemplar. The authors introduce a unified evaluation framework measuring three aspects: knowledge synthesis (organization and coverage), retrieval quality (relevance, coverage, and document importance), and verifiability (citation precision and claim coverage).
To provide a reproducible reference system, they also release DeepScholar-Ref, a modular open-source pipeline built atop the LOTUS semantic-operator framework that performs retrieval, filtering, ranking, and aggregation using LLMs. Experiments compare open-source, commercial, and search-agent systems, showing that current models reach only about 30% of human-level performance—highlighting the difficulty and future potential of reliable automated research synthesis.

### Strengths
The paper’s originality lies in formally defining generative research synthesis as a measurable task and introducing the first live benchmark for it. Rather than focusing on static QA or summarization datasets, the authors frame related-work generation as an end-to-end process involving retrieval, synthesis, and citation verification—an elegant combination of multiple research threads that were previously studied in isolation. The inclusion of verifiability metrics (Citation Precision and Claim Coverage) is particularly novel, addressing a key gap in evaluating factual grounding within long-form academic writing.
In terms of quality, the benchmark is thoughtfully engineered: papers are carefully filtered for publication status and clean structure, citations are resolved through both arXiv and OpenAlex, and evaluation criteria are explicitly defined and reproducible. The accompanying DeepScholar-Ref pipeline demonstrates solid engineering discipline, showing how an LLM can be operationalized into a modular retrieval-and-synthesis workflow.
The paper also excels in clarity. The data-collection process, metric design, and limitations are clearly presented with transparent assumptions and detailed ablations (e.g., retrieval-API comparison, oracle analysis). Figures and tables effectively communicate both methodology and findings, and the authors are candid about underestimations in human verifiability metrics.
Finally, the significance is high: DeepScholar-Bench provides a foundation for evaluating an emerging class of “research-assistant” systems. Its live, renewable design enables longitudinal comparison of academic-generation models and encourages more rigorous work on reliable citation and retrieval. The combination of benchmark, metrics, and open reference pipeline makes this paper a strong step toward standardizing the evaluation of LLM-based scientific writing systems.

### Weaknesses
## Field- and age-bias in Document Importance (DI)
The DI metric relies on median raw citation counts, which vary substantially by research field and publication year. Even with a median, the metric can still privilege older or citation-dense areas (e.g., NLP) and disadvantage emerging domains. Newly published papers inherently carry fewer citations regardless of relevance. The authors acknowledge this limitation; as a result, DI may misrepresent impact across domains and over time.

## Limited dataset scale and representativeness
The benchmark currently includes 63 papers across 18 CS subfields. While diverse—and each related-work section averages ~23 citations under strict filtering criteria—the small sample raises two concerns: (i) whether the dataset size ensures statistical robustness for metric/model comparisons, and (ii) whether specialized or cross-disciplinary research is underrepresented. 
Aggregating multiple monthly snapshots is non-trivial due to contamination risk (if models have seen prior releases). Independent of contamination, domain coverage remains limited at present.

## Subjectivity in citation importance labeling
Reference Coverage assumes the exemplar’s “important” citations constitute the gold set, yet “importance” is implicitly shaped by the original author’s citation style and emphasis, which can introduce bias. Without additional annotation or inter-rater validation, can this metric truly reflect an objective notion of citation importance?

## Noise and underestimation in verifiability metrics
Citation Precision and Claim Coverage depend on LLM-based entailment, which can be noisy. For human exemplars, the absence of sentence-level evidence spans systematically underestimates verifiability (as the authors note). Given the benchmark’s automated setup, the current verifiability numbers should be interpreted with caution, especially when comparing humans to systems.

### Questions
## DI calibration (field & age effects)
Is it possible to report DI stratified by publication year and cs.* domain (or provide a simple normalization ablation) to show whether DI rankings change materially?

## Reliability & stability of results
What is the run-to-run variability for both generation and LLM-as-judge decisions, and which metrics are most vs. least stable given the 63-query size (e.g., confidence intervals or similar estimates)?

## Reliability of important citation labels
Since RC hinges on “important” vs. “non-important” citations in exemplars, what methods support the reliability of this labeling (e.g., rubric/prompt specification, inter-rater checks, or ensemble agreement)?

## Where is the main bottleneck? (oracle ablation)
Given the large gains under oracle retrieval, can you break down per-stage contributions (retrieval vs. filtering vs. ranking) to identify the dominant bottleneck?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
DeepScholarBench introduces a new live benchmark and automated evaluation framework for generative research synthesis systems. The benchmark constructs realistic tasks by scraping recent, peer-reviewed ArXiv papers and using their related work sections as high-quality exemplars. Each task provides a paper’s title/abstract as input and requires an AI system to retrieve relevant literature, synthesize a long-form related work section, and cite sources. The authors contribute an automated data pipeline that continually updates the dataset (e.g. monthly) to avoid staleness. They also develop a holistic evaluation measuring knowledge synthesis, retrieval quality, and verifiability via seven fine-grained metrics, which show strong agreement with human judgments. Additionally, the paper provides DeepScholar-ref, an open-source reference pipeline built on the LOTUS framework as a baseline solution. Extensive experiments evaluate 14 baseline systems – including open-source methods (e.g. STORM, OpenScholar), search-agent pipelines with LLMs (GPT-4.1, Claude, Gemini, etc.), and OpenAI’s DeepResearch – on the benchmark. The results show significant room for improvement: no system exceeds ~31% (geometric mean) across the metrics, highlighting the difficulty of true research synthesis and the value of DeepScholarBench for driving progress.

### Strengths
The paper targets an important problem: how to robustly evaluate AI systems on complex research synthesis tasksbeyond simple Q&A. Its originality lies in creating a live, continuously evolving benchmark – a forward-looking design that avoids the pitfalls of static datasets becoming outdated. The methodology is high-quality: the authors build a careful data pipeline to source realistic tasks from recent accepted papers, enforcing quality controls (conference acceptance, proper references) to ensure the evaluation set is credible. The evaluation is holistic and well-motivated: instead of a single score, it breaks performance down into understandable dimensions (writing coherence, coverage of essential content, retrieval relevance, source importance, citation correctness), providing a nuanced view of system capabilities. Automating these metrics with LLM-based judges and entailment checks is innovative and enables scalable, repeatable evaluation – notably, the authors verify these automated metrics against human judgments, lending credibility to their approach. The clarity of the exposition is another strength: the paper is organized with the reader in mind, offering conceptual figures (e.g. illustrating the pipeline and evaluation framework) and concrete examples. Lastly, the significance is clear – DeepScholarBench fills a crucial gap by providing a benchmark that current state-of-the-art models struggle on (none exceeding ~39% on key metrics), thereby establishing a challenging platform that will likely spur significant research and improvements in generative research assistants.

### Weaknesses
1. **LLM-based evaluation bias**: The reliance on LLM judges (e.g., for Organization & Coherency) may favor formulaic writing over genuine quality. Adding human evaluation or refining prompts could help validate that metrics reflect true scholarly standards.
2. **Reference coverage limitations**: The metric penalizes systems for citing valid alternative papers not in the human reference list, rather than evaluating whether different sources cover the same content. Consider assessing content coverage instead of exact reference matching.
3. **Narrow scope**: The dataset covers only 63 CS queries from one quarter. Expanding to other domains (biomedical, social sciences) and creating a fixed benchmark subset alongside monthly updates would improve generalizability and enable longitudinal comparisons.
4. **Weak citation detection**: The baseline (DeepScholar-ref) scores near 0% on Document Importance, missing high-impact papers. Systems should incorporate citation metrics or seminal paper detection to identify influential references.

### Questions
1. Reference Coverage Metric: How are “important” references determined in the human exemplar? Is this labeling done manually or via an automated heuristic/LLM? Clarifying this process (and whether alternative relevant sources are somehow acknowledged) would help us interpret the reference coverage scores.
2. Generality: The dataset currently draws from computer science papers. Do the authors plan to extend DeepScholarBench to other research domains (e.g. medicine, physics), or is the focus primarily on CS/AI literature? Expanding domain coverage could increase the benchmark’s impact.
3. Live Benchmark Dynamics: With monthly updates, how will the authors ensure fair comparison over time? For example, if a new model is evaluated on a future set of queries, will there be a way to compare its performance to the scores reported in this paper (on the June 2025 set)? It would be useful to know if a static test subset or leaderboard versioning is planned to handle the evolving data.
4. Baseline Pipeline Improvements: DeepScholar-ref performs well, but notably struggles with retrieving highly-cited documents (Document Importance ≈0.01). What are the authors’ thoughts on augmenting the pipeline with an “importance awareness” (e.g. factoring in citation counts or influence of sources during retrieval)? This could potentially boost that metric – have such improvements been considered for future versions of the reference pipeline?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DeepScholar-bench, a new benchmark designed to evaluate AI systems on the task of generative research synthesis. The core idea is to create a "live" benchmark that stays current by automatically sourcing new queries from recent, high-quality ArXiv papers. The specific task is to generate a "Related Work" section for a given paper's title and abstract. The authors propose a comprehensive, automated evaluation framework that assesses performance across three key dimensions: Knowledge Synthesis, Retrieval Quality, and Verifiability. They also contribute DeepScholar-ref, an open-source reference pipeline. Through a systematic evaluation of 14 existing systems, including open-source models, commercial search agents, and OAI's DeepResearch, the paper demonstrates that the task is far from solved, with no system surpassing a geometric mean of 31% across all metrics.

### Strengths
1. The paper addresses a critical and timely problem. s AI systems for long-form, knowledge-intensive tasks become more common and rigorous.

2. The evaluation framework is comprehensive with multiple fine-grained metrics (Table 1). The authors perform manual validation for their LLM-based metrics, showing high agreement (70-82%) with human judgments (Table 7, lines 963-968).

### Weaknesses
- Seems contradiction in the  scope of retrieval: The paper motivates the task by emphasizing retrieval from the "live web" (lines 015, 050). However, the experimental setup explicitly restricts all systems to retrieving only from the ArXiv API (line 361). The benchmark may not be fully evaluating the systems' ability to navigate the complexities of the true "live web."

- The presentation in Table 2 of human-written exemplars receiving very low scores for Citation Precision (.278) and Claim Coverage (.205) in a main results table next to AI systems that score as high as .944 (Cite-P) and .937 (Claim Cov) can be misleading, though a footnote (lines 345-348) was given that that this is a severe underestimate.

### Questions
1. Given the acknowledged issues with automatically measuring the verifiability of human writing, have you considered performing a small-scale manual evaluation on a subset of the human-written exemplars? I'd say that even with a small sample would provide a much clearer "gold standard" for the verifiability metrics.

2. The cost and latency analysis in Table 5 is excellent and highly valuable. However, costs are not reported for the proprietary Search Agent baselines (e.g., those using GPT-4.1, Claude, etc.). Could you add these figures to provide a complete and fair comparison?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DeepScholar-Bench,a benchmark and automated evaluation framework for generative research synthesis.Evaluates systems along three axes — synthesis, retrieval, verifiability — using seven automated metrics. The authors also release DeepScholar-Ref, an open-source reference pipeline built on LOTUS.

### Strengths
- The benchmark design, data pipeline, and multi-axis metric structure are coherent.
- Clear task decomposition: synthesis / retrieval / verifiability.
- Open-source infrastructure and a reproducible reference pipeline DeepScholar-Ref.
- Timely motivation — evaluation gap in long-form research reasoning
- Clean writing, organized figures and tables
- Logical experiment and ablations (oracle, retriever variants) with consistent metric and reporting

### Weaknesses
- No quantitative human–metric correlation (claimed “manual validation” only)
- Heavy reliance on LLM-as-judge introduces potential bias, no transparency or calibration details, it’s really unclear whether the metrics reflect the real quality. No human/user evaluation to confirm synthesis usefulness
- Dataset scale is small (63 ArXiv CS papers) and domain coverage is narrow, which limits the robustness and generalizability of the benchmark
- The novelty is incremental vs. prior deep-research benchmarks
- Tables are kinds of dense, with limited analysis, no significance reproted

### Questions
- How stable are results across LLM-judge versions or calibration settings? What's the correlations between automated metrics and human ratings
- How are “important references” defined for Reference Coverage?
- How to prevent that the models have been trained on the same ArXiv data used for benchmark construction?

### Soundness
2

### Presentation
3

### Contribution
2
