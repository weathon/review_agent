# KRAMABENCH: A Benchmark for AI Systems on Data-to-Insight Pipelines over Data Lakes

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Discovering insights from a real-world data lake potentially containing unclean, semi-structured, and unstructured data requires a variety of data processing tasks, ranging from
extraction and cleaning to integration, analysis, and modeling. This process often also demands domain knowledge and project-specific insight. While AI models have shown remarkable results in reasoning and code generation, their abilities to design and execute complex pipelines that solve these data-lake-to-insight challenges remain unclear. We introduce KramaBench which consists of 104 manually curated and solved challenges spanning 1700 files, 24 data sources, and 6 domains. KramaBench focuses on testing the end-to-end capabilities of AI systems to solve challenges which require automated orchestration of different data tasks. KramaBench also features a comprehensive evaluation framework assessing the pipeline design and individual data task implementation abilities of AI systems. We evaluate 8 LLMs using our single-agent reference framework DS-Guru, alongside both open- and closed-source single- and multi-agent systems, and find that while current agentic systems may handle isolated data-science tasks and generate plausible draft pipelines, they struggle with producing working end-to-end pipelines. On KramaBench, the best system reaches only 55% end-to-end accuracy in the full data-lake setting. Even with perfect retrieval, the accuracy tops out at 62%. Leading LLMs can identify up to 42% of important data tasks but can only fully implement 20% of individual data tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces KRAMABENCH, a large-scale benchmark designed to evaluate AI systems on end-to-end data-intensive reasoning—specifically, the ability to design and execute full data-science pipelines across heterogeneous, noisy data lakes.
The benchmark comprises 104 curated tasks drawn from 1,700 files and six domains (archaeology, astronomy, biomedical, environmental, legal, and wildfire).
Each task is derived from real-world studies and paired with reference solutions, sub-task annotations, and evaluation scripts.

The authors propose three complementary evaluation settings:

End-to-end automation – overall task correctness;

Pipeline design – coverage of key functional components;

Sub-task implementation – correctness on atomic operations.

A minimal baseline framework, DS-Guru, enables single-agent LLMs (e.g., GPT-4o, Claude-3.5, DeepSeek-R1, Llama3.3) to attempt these challenges.
The benchmark also evaluates agentic systems such as Hugging Face Smolagents, OpenAI Deep Research, and Gemini Agentic Mode.

### Strengths
Originality: First benchmark targeting end-to-end data-to-insight reasoning across heterogeneous data lakes.

Quality: Comprehensive curation and evaluation pipeline; transparent methodology with open-source assets.

Clarity: Well-written with detailed ablations, case studies, and ethical considerations.

Significance: Reveals key weaknesses of state-of-the-art LLM agents (e.g., data-dependent reasoning), guiding future research in automated data science.

Breadth: Covers six domains and multiple data modalities (structured, semi-structured, unstructured).

Practical utility: Enables reproducible assessment of both academic and commercial agentic frameworks.

### Weaknesses
Limited evaluation diversity: While the benchmark spans multiple domains, all experiments are English-only and rely on a single execution environment; multilingual or multimodal tasks would strengthen generality.

Absence of multi-agent baselines: The paper focuses on single-agent and light agentic systems; evaluating collaborative agents or planner–executor architectures would provide additional insight.

Computation cost reporting: Token-level cost is discussed but system runtime and hardware variance could be presented more rigorously.

Limited human comparison: Including a human-in-the-loop baseline would contextualize model performance relative to expert data scientists.

Statistical significance: Some percentage improvements (e.g., +1.3 %) could benefit from confidence intervals or variance analysis.

### Questions
1-How does KRAMABENCH handle tasks with stochastic or non-deterministic outputs (e.g., probabilistic models)?

2- Could future iterations support incremental learning or multi-agent collaboration settings?

3- Did the authors observe any correlations between model size and success on specific domains (e.g., biomedical vs. legal)?

4- Are the obscured-input experiments sufficient to rule out partial memorization—could LLMs have seen similar structure from public repositories?

5- Would integrating structured metadata (schemas, ontology hints) into DS-Guru improve pipeline design?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
1. This paper introduces KRAMABENCH, a benchmark designed to evaluate AI systems’ ability to solve end-to-end data science pipelines on real-world data lakes (heterogeneous, unclean, multi-source data). It includes 104 manually curated tasks across 6 domains (e.g., archaeology, biomedical research), 1700 files, and 24 data sources, paired with expert reference solutions and a 3-tier evaluation framework (end-to-end automation, pipeline design, sub-task implementation). The authors also propose DS-Guru, a lightweight single-agent framework for baseline testing, and evaluate 8 LLMs + 4 agentic systems (e.g., smolagents DR, OpenAI Deep Research).

### Strengths
1. Fills a critical gap in existing benchmarks (e.g., DS-1000, ARCADE) by focusing on end-to-end data lake processing rather than isolated tasks (code generation, text-to-SQL). Unlike prior work, it emphasizes real-world complexity (noisy data, multi-file integration, domain-specific knowledge) and requires systems to orchestrate all pipeline stages (discovery, cleaning, analysis).
2. Rigorous task curation via a 4-step validation process (curation → cross-contributor verification → key functionality identification → sub-task curation) ensures high-quality, reproducible reference solutions.
3. Extensive experiments across 8 LLMs, 4 agentic systems, and 3 input modes (Full, Oracle, Trimmed) provide statistically meaningful results. Ablation studies (e.g., retrieval mechanism, sample size, iterations) further validate conclusions about performance drivers.

### Weaknesses
1. The paper focuses on single-agent systems (DS-Guru, smolagents DR) but barely explores multi-agent architectures, which are increasingly proposed for complex data tasks (e.g., dividing pipeline stages across specialized agents). This is a missed opportunity, as multi-agent systems may mitigate single-agent limitations (e.g., context window constraints, heterogeneous skill requirements).
2. The results show large performance gaps across domains (e.g., smolagents DR achieves 60% accuracy in Environment but 16.67% in Astronomy), but the paper provides minimal analysis of why (e.g., is Astronomy’s larger file size, more complex data formats, or domain unfamiliarity the cause?). Without this, readers cannot generalize findings to other domains.

### Questions
Beyond Weaknesses. The following is the suggestions for the authors.
1. Incorporate Human-in-the-Loop Evaluation: Since real-world data science often involves human-AI collaboration, add a fourth evaluation tier (e.g., human oversight of pipeline revisions) to measure how systems assist human experts. This would increase the benchmark’s relevance for industry use cases.
2. Scalability and Cost Analysis: Extend experiments to larger data lakes (e.g., 10k+ files) to test scalability of retrieval mechanisms (OPS vs. agentic retrieval). Add cost-per-accuracy metrics (e.g., dollars/task, energy consumption) to guide practical adoption.

### Soundness
3

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
4

### Summary
This paper introduce KramaBench, a benchmark to evaluate LLM-based systems on data science tasks. KramaBench includes 104 mannually curated tasks from 1700 real-world files across 24 sources in 6 domains (archaeology, astronomy, biomedical research, environmental science, legal discovery, and wildfire prevention). The authors then provide a comprehensive evaluation framwork with several settings (1) end-to-end automation, (2) pipeline design, and (3) individual task implementation. With experiments on 8 LLMs using DS-Guru reference framework and several agentic systems, the authors find that current systems struggle with end-to-end pipeline generation, even with perfect retrieval.

### Strengths
- **Comprehensive Evaluation**: The multi-level evaluation framework (end-to-end, pipeline design, sub-task implementation) provides valuable insights into where systems fail.
- **Rigorous Curation Process**: The 4-step validation process involving multiple contributors, cross-validation, and manual verification of reference solutions demonstrates strong quality control. Grounding tasks in published studies ensures real-world relevance and avoids artificial task design.
- **Diverse Tasks**: The benchmark spans multiple domains with varying data formats, file counts, and difficulty levels, providing good coverage of real-world scenarios.

### Weaknesses
- **Unclear Motivation and Weak Problem Positioning**: The paper lacks a compelling motivation section explaining why this benchmark is needed now and what specific real-world problems it addresses that existing benchmarks cannot. The introduction jumps directly into the solution without establishing the problem's urgency or providing concrete use cases where current benchmarks fail.
- **Poor Figure Quality**: Figure 1 is a low-resolution raster image with blurry, barely readable text, which is unacceptable for a top-tier venue. The figure should be vector graphics (SVG/PDF) with crisp, readable text. This significantly impacts the paper's professionalism and readability.
- **Severely Insufficient Visual Explanations**: The paper has only 2 figures across 9 pages of main content, making it difficult to understand key components. Critical missing visualizations include (1) DS-Guru architecture diagram showing the three variants and data flow, (2) benchmark curation pipeline flowchart for the 4-step process, (3) task structure schema showing relationships between end-to-end tasks, key functionalities, and sub-tasks, (4) evaluation framework diagram.

### Questions
- What is the conceptual or methodological novelty beyond creating a benchmark with more realistic data science tasks?
- Can you provide validation of the LLM judge against human evaluators, including agreement rates and calibration analysis?
- Can you provide detailed computational cost analysis (tokens, time, dollars) for running each system on the full benchmark?

### Soundness
2

### Presentation
1

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
### Summary
This paper proposes KRAMABENCH, a comprehensive benchmark for evaluating AI systems’ end-to-end capabilities on data-intensive tasks. Targeting the gap of existing benchmarks that focus on isolated steps rather than full data science pipelines, KRAMABENCH comprises 104 manually curated tasks across 6 domains (archaeology, astronomy, biomedical research, etc.), covering 1700 files from 24 sources. The benchmark designs three evaluation settings (end-to-end automation, pipeline design, sub-task implementation) and evaluates 8 LLMs alongside multiple agentic systems (e.g., smolagents DR, OpenAI DR). Key findings include: current systems struggle with end-to-end pipeline generation (top accuracy 50%), agentic control flows significantly outperform structured loops, and fine-grained data-dependent reasoning and holistic data lake understanding are major bottlenecks. The paper also provides detailed ablation studies, failure analyses, and full reproducible artifacts, offering valuable insights for future research on automated data science systems.

### Strengths
1. **Gap-Filling Innovation**: Addresses a critical limitation of existing benchmarks by focusing on end-to-end data science pipelines (from data discovery to insight generation) rather than isolated tasks (e.g., code generation, text-to-SQL), aligning with real-world data science workflows.
2. **Comprehensive Design**: Covers diverse domains, heterogeneous data types (structured/semi-structured/unstructured), and multi-faceted evaluation dimensions, ensuring the benchmark’s generality and robustness.
3. **Rigorous Experimentation**: Evaluates a wide range of LLMs and agentic systems, conducts in-depth ablation studies (retrieval mechanisms, data leakage, iteration counts) and failure analyses, providing conclusive evidence for key claims.
4. **High Reproducibility**: Publicly releases all code, data, workloads, and evaluation scripts, adhering to top conference standards for reproducibility and facilitating follow-up research.





### Weaknesses
1. **Limited Generalizability of Data Leakage Evaluation**: The data leakage assessment only covers 20% of tasks (focused on legal and wildfire domains) with synthetic identifiers/numerics. This narrow scope may not fully reflect how systems rely on memorized data across different domains or task types.

2. **Insufficient Analysis of Domain-Specific Performance Gaps**: The paper reports significant performance variations across domains (e.g., wildfire: ~50% accuracy vs. astronomy: ~16% for smolagents DR), but lacks in-depth analysis of why certain domains are more challenging (e.g., data complexity, domain knowledge requirements).

3. **Lack of Multi-Agent Evaluation**: Current evaluations focus on single-agent systems, but multi-agent collaboration is a promising direction for complex data-intensive tasks—omitting this dimension limits the benchmark’s coverage of state-of-the-art approaches.


### Questions
1. Given that the data leakage evaluation only covers 20% of tasks and is concentrated in the legal and wildfire domains, how do you ensure that the results obtained using synthetic data can generalize to other domains and task types to accurately reflect the characteristics of systems' dependence on memorized data?(Related to Weakness 1)

2. Regarding the significant performance gap of smolagents DR between the wildfire domain (~50% accuracy) and the astronomy domain (~16% accuracy), do you plan to supplement in-depth analysis from dimensions such as data complexity and domain knowledge requirements to explain the root causes of difficulty in different domains?(Related to Weakness 2)

3. Multi-agent collaboration has become a promising cutting-edge direction for solving complex data-intensive tasks. Why did you not incorporate this dimension in the evaluation, and does this benchmark have plans to supplement multi-agent system testing in the future to cover the latest technical approaches?(Related to Weakness 3)

### Strengths
See Summary

### Weaknesses
See Summary

### Questions
See Summary

### Soundness
3

### Presentation
3

### Contribution
2
