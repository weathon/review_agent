# OpenDataBench: Real-World Benchmark for Table Insight Generation and Question Answering Over Open Data

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
The promise of Large Language Models (LLMs) for data analysis is hindered by benchmarks that inadequately reflect real-world complexities, including multiple large tables and external knowledge. Moreover, they mainly focus on fact retrieval via Question Answering (QA) and overlook the critical task of exploratory insight generation. To address these gaps, we introduce OpenDataBench, a benchmark built from governmental open data capturing these practical challenges. It features two types of tasks: multifaceted Table QA tasks that require answering complex decomposable questions with either text or graphs, and Table Insight tasks that challenge models to generate expert-level findings from exploratory data analysis.
We evaluate state-of-the-art LLMs and our proposed agentic solution on OpenDataBench. Our experimental results indicate that even top-performing models struggle with both tasks. This highlights a significant gap between current model capabilities and the demands of realistic data analysis. OpenDataBench serves as a rigorous benchmark for advancing research on LLM-driven data analysis systems capable of addressing both reactive question answering and proactive insight discovery. Code and sample data are available at https://anonymous.4open.science/r/opendatabench-8AFA/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on the challenge of evaluating LLMs on real-world data analysis, where existing table reasoning benchmarks fail to capture the complexity of large, multi-table datasets and the open-ended nature of insight generation.
To address this, the authors construct OpenDataBench, a benchmark built from governmental open data featuring tasks for multifaceted table QA and Table Insight Generation, and introduce two corresponding agentic solutions, including an Answer Agent and an Insight Agent. Experiments show that even state-of-the-art models like GPT-4o and Gemini 2.5 perform poorly on these tasks, revealing a significant gap between current LLM capabilities and realistic data-analysis demands.

### Strengths
1. The paper targets a real and important challenge that current LLM benchmarks for table reasoning do not reflect the complexity and messiness of real-world data analysis.

2. The idea of building a realistic benchmark from government open data is solid and practical, offering a credible way to capture large-scale, multi-table, and heterogeneous data scenarios often encountered in real applications.

### Weaknesses
1. The novelty is limited. The benchmark mainly extends existing table QA and insight-generation setups to larger, real-world datasets without introducing fundamentally new tasks or evaluation methods.

2. The proposed agents (Answer Agent and Insight Agent) are largely incremental combinations of existing techniques like code generation, self-correction, and reflection, offering engineering value but little conceptual innovation.

3. The evaluation results mainly confirm that current LLMs struggle with large and messy tables, rather than revealing new insights into how to overcome these challenges.

### Questions
see weakness

### Soundness
3

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
2

### Summary
This paper introduces OpenDataBench, a benchmark designed to evaluate table understanding capabilities of language models using real-world open data from governmental sources. The benchmark addresses two limitations of existing table reasoning benchmarks: lack of real-world complexity (large-scale, multi-table datasets with external knowledge) and narrow task scope (focusing only on question answering while neglecting insight generation). OpenDataBench features two main tasks: Table QA (answering complex decomposable questions with text or visual outputs) and Table Insight (generating expert-level findings from exploratory analysis). The authors also propose two agentic solutions: an Answer Agent with fail-safe modules for Table QA and an Insight Agent using graph-based exploration for Table Insight.

### Strengths
1 The proposed dataset represents a good advancement over existing benchmarks that use small, clean tables.

2 The benchmark formalizes both reactive (Table QA) and proactive (Table Insight) data analysis tasks. 

3 The paper employs a four-stage pipeline (question generation, scoring, answer generation, human verification) with multiple LLM judges and human experts.

### Weaknesses
1 The use of LLM-based evaluation for insight generation, while practical, introduces subjectivity. The quantitative metrics may not fully capture insight quality, but provides limited validation of the evaluation methodology's reliability.

2  The multi-agent approaches involve complex pipelines with multiple LLM calls, but the paper lacks analysis of computational requirements, inference latency, or cost considerations, which are important factors for real-world deployment.

### Questions
please refer to the weakness

### Soundness
2

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
4

### Summary
The authors propose a new benchmark, OpenDataBench, which consists on two tasks: multi-faceted table question-answering and table insights. The former assesses factual reasoning over composable questions (with multiple subquestions), while the latter, challenges models to provide expert-level insights , which can require in-depth analysis. The authors leverage a filtering technique, human annotators  and LLM as a judge in the construction of the data for both tasks. The tasks are based on publicly available complex, heterogeneous datasets that contain large tables. The authors propose a table serialization technique to bypass the need to pass the full table to the LLM. The authors also provide two agents for each task which outperform existing models and agents. The authors also include error analysis and an ablation wrt the proposed table serialization.

### Strengths
- The authors propose a complex enough benchmark that can be leveraged to assess agentic workflows. This is an active area of research and it is important to have access to realistic benchmarks for evaluation.

### Weaknesses
- The paper is missing precise implementation details for the new proposed agents

### Questions
- Add one liner to explain how different types of variables are handled in the main paper so there’s no need to go to the appendix to understand the full details.
- It would be good to add more clearly the percentage of discarded data points and questions as well as describe how many times you require to prompt an LLM to provide a better notion of how expensive the dataset creation is.
Even though there is a section that discusses error analysis, one particular failure mode that is not discussed in detail is what happens if there are questions that use multiple tables and the agent fails to fetch all the relevant tables? 
- Typically, SQL is the tool of choice to extract information from one or more tables. Is it the case that the python code can leverage libraries to perform SQL queries on the tables?
- It would be interesting to add to the ablation study if instead of naively passing the first 10 rows, what if we pass the table schema and a number of rows? Is that equivalent to the proposed feature serialization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposed a new benchmark targeting two tasks -- table question answering over large tables, and table insight generation.
The source tables are collected from public government datasets, and question-answer pairs are generated with LLM followed by human verification, keeping only QA pairs with low execution-result agreement for LLM generated code. Table Insights are extracted from publicly written report based on the tables. Baselines together with error analysis are also provided.
Main contributions of this paper include
1. Created a benchmark for QA over large tables
2. While not the first to create a insights generation benchmark, the insights are extracted from organic sources (human written reports) rather than planted into tables

I think the motivation for this work is clear and if the idea is executed properly this can be an important contribution, but some sections (dataset construction/eval metrics/baseline) lacks sufficient detail to judge soundness of the work, so for now I'm recommending reject.

### Strengths
1. This benchmark addresses current limitations/shortcomings in existing table-based QA
2. The insights are collected organically from human written reports, which seems more representative of human interests compared to existing benchmark
3. The answers are not limited to textual format but also covers data visualization/chart generation

### Weaknesses
1. Missing some details in dataset construction/eval metrics/baseline design & analysis

    a. Unclear to me how insights are extracted from the human written reports. Also through prompting?

    b. For table QA, is your naive baseline (feeding first 10 rows to LLM in single turn) generating code to produce answer or just prompted to generate output? 

    c. baseline analysis -- what's the proportion of answer agent needing to go back to revise the generated code? how many iterations do you allow answer agent to run before stopping

    d. Insight agent -- the pipeline seems similar to the data construction stage for your table QA benchmark generation (except human verification)? how do you determine what questions to keep and what's the stopping criteria? how do you rank the generated questions? 

    e. Insight generation metrics -- the proposed G-eval based score compares generated insights against 'ground-truth' insights, but original G-eval score is comparing summary against source article. Also, G-eval reports four separate scores (Coherence, Consistency, Fluency, Relevance). Why was only one combined score reported? How are those different aspects combined to get a single score? I think the description of the proposed g-eval inspired metrics does not have sufficient detail.

    f. Analysis: as I understand, the answer are generated using LLM generated code, and the answer agent is also prompted to generate code, even though the dataset construction employs multiple LLM providers to mitigate bias, is there any chance that the ones that are answered correctly also happens to be generated with the same model? i.e. in the successfully answered portions, say by Gemini, any chance they just happened to be questions whose answers were already generated by Gemini?


 

2.  Missing some reference
    a. How does this dataset compare to DataBench (Grijalba et al 2024) -- this also seems to be targeting QA over large tables

Jorge Osés Grijalba, L. Alfonso Ureña-López, Eugenio Martínez Cámara, and Jose Camacho-Collados. 2024. Question Answering over Tabular Data with DataBench: A Large-Scale Empirical Evaluation of LLMs. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13471–13488, Torino, Italia. ELRA and ICCL.

### Questions
1. Seems like question types are limited to those that are answerable with python code, as the dataset construction process with python execution for answer generation seems to guarantee that. What's the justification behind limiting it to these types of questions? How are you prompting the LLM to generate questions?

2. I appreciate the authors acknowledged the potential subjectivity of insights and their attempt to address it. I'm curious if the authors have conducted analysis on annotator agreement for the generated insights

3. "After executing the code from all four models, we measured the answer consensus to filter out questions that yielded unanimous agreement across all four LLMs. Such instances were deemed to indicate a low level of analytical complexity, making them unsuitable for a benchmark to challenge state-of-the-art models." => 

    a. how do you measure answer consensus? 

    b. what is the justification for the claim that 'unanimous agreement' indicates 'low level of analytical complexity' -- seems like this is just to intentionally construct an adversarial set that is hard for current systems, not so much representative of true distribution/human interests? 

    c. if the executed code has no agreement, how do you decide which answer to keep? 

4. Human verification stage -- "During this stage, annotators also filtered out questions for qualitative reasons,
such as not being insightful, being too ambiguous to permit a definitive answer, or requiring
external knowledge that was unavailable." => what is considered 'insightful'?

### Soundness
1

### Presentation
3

### Contribution
2
