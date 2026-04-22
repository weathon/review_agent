# DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8

## Abstract
Deep Research Agents (DRAs) are emerging as one of the most practical classes of LLM-based agents. Given an open-ended research task, they find, analyze, and synthesize large numbers of online sources to produce a comprehensive report at the level of a research analyst. This can compress hours of manual desk research into minutes. However, a comprehensive benchmark for systematically evaluating the capabilities of these agents remains absent. To bridge this gap, we introduce DeepResearch Bench, a benchmark consisting of 100 PhD-level research tasks, each meticulously crafted by domain experts across 22 distinct fields. To evaluate DRAs comprehensively, we propose two complementary and fully automated methodologies. The first is a reference-based method with adaptive criteria to assess the quality of generated research reports. The second evaluates a DRA’s information‑retrieval and collection capabilities by assessing its effective citation count and overall citation accuracy. By conducting extensive human consistency experiments, we demonstrate that our evaluation methods are highly aligned with expert judges and faithfully reflect human judgments of quality differences among DRA-generated content. We are open-sourcing DeepResearch Bench and key components of these frameworks to accelerate the development of practical LLM-based agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents DeepResearch Bench, a benchmark for evaluating Deep Research Agents (DRAs) that perform open-ended web-based research. It comprises 100 expert-curated tasks across 22 domains and introduces two automated evaluation frameworks: RACE for report quality and FACT for retrieval accuracy. Experiments and human studies show strong alignment between these frameworks and expert judgments. Overall, the work provides a solid foundation for systematic and scalable evaluation of LLM-based research agents.

### Strengths
1.	The paper introduces a well-motivated and comprehensive benchmark for Deep Research Agents, grounded in real-world user queries and expert-curated tasks across 22 domains, resulting in a high-quality and realistic evaluation suite.
2.	The proposed RACE and FACT frameworks jointly assess report quality and retrieval reliability, effectively addressing a key gap in existing evaluation methods — making this a timely and impactful contribution.
3.	The experiments are extensive and insightful, covering both commercial and open-source DRAs, and offering clear comparative results and valuable guidance for future research and development.

### Weaknesses
There is no major weakness in this paper. The only minor concern lies in the size of the benchmark — with 100 tasks, its scale may still limit robustness for fine-grained statistical analysis or long-tail domain evaluation. However, given the high quality and expert-level design of the tasks, this limitation is acceptable and does not detract from the overall contribution.

### Questions
n/a

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
In order to evaluate the performance of Deep Research Agents (DRAs) that gather and summarize information for research tasks, this paper presents a benchmark named DeepResearch Bench, with 100 PhD-level research tasks. This work also proposes two automatic methods for evaluation with Judge LLM, including a reference-based approach to measure report quality using adaptive criteria and reference reports, and a citation-based approach assessing the accuracy and effectiveness of retrieved sources. This paper further shows its consistency with human evaluation to highlight its potential.

### Strengths
1. This paper contributes the first benchmark to evaluate Deep Research Agents. The design is also grounded in the real-world human research tasks, which are collected based on user interaction with Chatbots.

2. The work conducted comprehensive experiments on popular DRAs, including one open-sourced one, and also compared the DRAs with search-enabled LLMs.

3. The work also analyzed the consistency of the proposed evaluation with human evaluation.

### Weaknesses
1. The major concern comes from the validation of the consistency with human evaluation. It's unclear why the final set of tasks is designed to have 50 tasks for Chinese and English, respectively, and which split the experiments are conducted on. It's noted in line 364 that the human evaluation was conducted on the Chinese tasks, which might limit the claim of consistency with human evaluation if the main results are obtained on English tasks.

2. There are 100 tasks in total, with 50 Chinese and 50 English tasks, across 22 domains. The number of tasks per domain and language might be limited.

3. The authors conducted a comparison of human evaluation consistency using various Judge LLMs to select the candidate for Judge LLM. However, the performance of the proposed metrics is highly dependent on the Judge LLM, and the RACE evaluation also relies on the reference reports. While Gemini models are used as the Judge LLM, and Gemini-generated reports are used as the reference reports, this might introduce model bias and favor Gemini-based DRAs, which is also shown in the experiment results.

4. The introduction of the evaluation frameworks in Section 3 is fragmented and eliminates many details, such as the computation of the FACT framework. More complete descriptions of the methods should be included in the main text to increase clarity, potentially moving space from Figures 1, 3, and 4.

### Questions
1. How is the variance across the domain or across tasks within a domain?

2. As the final set includes English and Chinese tasks, if both are evaluated for the main results, is there any insight into the DRA performance across languages?

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
4

### Summary
This paper introduces DeepResearch Bench, a benchmark designed to evaluate deep research systems. The benchmark consists of 100 expert-authored PhD-level tasks across 22 domains. For evaluation, the authors propose two methods i.e.,
(1) RACE which uses LLM-as-a-Judge across four dimensions (comprehensiveness, depth, instruction-following, and readability),
(2) FACT, which assesses citation accuracy and the number of verifiable evidence sources.
They ran experiments comparing both deep research agents and llm + web tool calls. they also conducted human eval to study alignment between llm as a judge and expert human annotations.

### Strengths
- The problem is timely and there is not many standard benchmarks nor evaluation strategies  designed for deep research systems.
- The problems encompass a large set of topics.
- they evaluate from several different perspectives i.e., both quality of the report as well as citation quality.
- Conducted human eval to validate llm based evaluation
- covered deep research systems as well as llm + search tools

### Weaknesses
- Some formatting problems in prompt template Appendix M
- The baseline set omits several relevant open-source and hybrid systems such as Deep Researcher, OpenScholar, or opensrouced LLM + tool-use/web-agent frameworks. T
- It is not explained how the 100 benchmark tasks were chosen from 44 K filtered queries. Did annotators manually reviewed all queries to come up with 100 queries?
-  The benchmark lacks any quantitative or qualitative analysis of task difficulty. 
- In Section 3.1.1, it is unclear whether the RACE criteria are fixed per task or dynamically generated per query. 
- The FACT framework measures accuracy and count but lacks a notion of coverage i.e., how many claims in the report remain unsupported. Please see https://arxiv.org/abs/2411.17375 for claim coverage eval.
- Section 4.1.1 implies that Gemini-generated reports serve as the reference. This design choice risks biasing evaluation in favor of Gemini-based agents.

### Questions
- How were the final 100 benchmark tasks selected from the 40k+ samples?
- Are the task difficulty levels annotated or estimated in any way? How is the distribution of difficulty across the 40k samples and 100 selected questions?
- Are RACE’s adaptive criteria regenerated for each query instance or fixed once per domain?
- Was the Judge LLM calibrated  before correlation tests?
- Why were Gemini-generated reports used as reference as mentioned in 4.1.1 ? could this bias results toward Gemini agents?

### Soundness
3

### Presentation
3

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
The paper introduces **DeepResearch Bench**, a new benchmark consisting of 100 carefully curated, PhD‑level research tasks spanning 22 domains. It proposes two automated evaluation frameworks:  

* **RACE** – a reference‑based, dynamically weighted “LLM‑as‑judge” protocol that generates task‑specific criteria and dimension weights (Comprehensiveness, Insight, Instruction‑following, Readability) and scores reports relative to a high‑quality reference.  
* **FACT** – an automated pipeline that extracts statement‑URL pairs from agent reports, deduplicates them, and judges factual support using a separate judge LLM, yielding Citation Accuracy and Average Effective Citations per task.  

The authors validate RACE and FACT by (i) extensive human consistency experiments (pairwise agreement, Pearson/Spearman correlations), (ii) robustness analyses (reference selection, length inflation, judge model choice), and (iii) comparisons across a wide range of commercial Deep Research Agents (DRAs) and LLMs with search tools. All data, prompts, and code are released. The paper makes a solid engineering contribution by releasing a well‑designed benchmark and novel, human‑aligned evaluation frameworks for Deep Research Agents. The methodology is sound, the experiments are extensive, and the analyses address many potential pitfalls. However, the reliance on proprietary judges, limited benchmark scale, and lack of statistical significance reporting temper the impact. Overall, the work is a valuable resource for the community and merits inclusion in the conference, albeit not as a flagship oral contribution.

### Strengths
1. **Clear Gap Identification** – The paper convincingly argues that existing benchmarks either target isolated capabilities (browsing, code) or single‑domain research, leaving a need for a comprehensive DRA evaluation suite.  
2. **Benchmark Construction Pipeline** – Leveraging a large in‑house query log (≈96 k queries) and a taxonomy‑driven filtering/classification pipeline yields a realistic distribution of tasks. Expert‑authored tasks and a bilingual (50 EN/50 ZH) set improve ecological validity.  
3. **Innovative Evaluation Design** – RACE’s dynamic weighting and reference‑based scoring address known pitfalls of static rubrics and absolute LLM scoring (e.g., length bias, generic high scores). FACT provides a concrete, interpretable measure of citation grounding, which is highly relevant for research‑assistant agents.  
4. **Human‑Alignment Validation** – The human consistency study is thorough: 70+ annotators, pairwise agreement (PAR = 71 % for RACE vs. 59 % for a vanilla prompt), and high Pearson correlations (OPC ≈ 0.99). The filtered‑task analysis further strengthens confidence.  
5. **Robustness Analyses** – Sensitivity to reference choice, report length inflation, and judge LLM (closed‑source vs. open‑source) is explored, showing that rankings are stable.  
6. **Reproducibility Commitment** – All prompts, hyper‑parameters, and scripts are released; the paper details cost per evaluation and provides open‑source judge alternatives.  
7. **Ethical Considerations** – The authors appropriately describe anonymization of query logs, fair compensation of annotators, and discuss potential misuse of generated reports.

### Weaknesses
1. **Scale and Diversity Limitations** – 100 tasks, while high quality, may still be insufficient to capture the full variance of real‑world research queries, especially for emerging sub‑domains (e.g., AI‑ethics, climate modeling). This limits statistical power for fine‑grained comparative analysis.  
2. **Reliance on Proprietary Judge LLMs** – The primary RACE and FACT evaluations depend on Gemini‑2.5‑Pro/Flash APIs. Although the authors provide open‑source alternatives (Qwen3) and show similar rankings, the absolute scores and some correlation metrics dip noticeably, raising concerns about reproducibility for researchers without access to the same commercial models.  
3. **Citation Extraction Fragility** – FACT assumes that agents emit citations in a parsable format; the paper notes that several DRAs (Claude Research, Kimi) could not be evaluated because of UI‑specific citation styles. This suggests that FACT may penalize agents for UI design rather than factual grounding.  
4. **Baseline Coverage** – The experimental section focuses on commercial DRAs and a few open‑source agents (LangChain ODR). There is no systematic evaluation of baseline retrieval‑only pipelines (e.g., traditional IR + summarizer) that could contextualize the performance gap.  
5. **Statistical Reporting** – While human‑consistency numbers are presented, confidence intervals or statistical significance tests for the differences between agents are absent. It is unclear whether the observed gaps (e.g., Gemini‑2.5‑Pro vs. OpenAI DRA) are robust.  
6. **Potential Conflict of Interest** – Many evaluated agents are from the same ecosystem (Google, OpenAI, Perplexity) that also provide the judge LLMs. The paper does not discuss mitigation of possible bias introduced by using a judge model that may be tuned toward its own family’s output style.  
7. **Limited Discussion of Failure Cases** – The analysis of low citation accuracy for high‑citation agents (e.g., Gemini‑2.5‑Pro) is brief; deeper error analysis (hallucinated citations, URL unreachable, mis‑matched statements) would improve understanding of trade‑offs.

### Questions
1. **Task Sampling Variability** – How sensitive are RACE scores to the specific set of 100 tasks? Have you performed a leave‑one‑out or bootstrapped analysis to estimate variance across different task subsets?  
2. **Open‑Source Judge Calibration** – When using Qwen3 as the judge, did you fine‑tune prompts or temperature settings to match the proprietary judge’s behavior? Could the performance gap be reduced with prompt engineering?  
3. **Citation Format Normalization** – Could you elaborate on the preprocessing pipeline that normalizes citations across agents? Is there a risk that aggressive cleaning removes legitimate citation cues, thereby affecting FACT scores?  
4. **Statistical Significance** – Did you compute confidence intervals for the pairwise agent comparisons (e.g., via bootstrap of human scores)? If not, could you add these to substantiate claims of superiority?  
5. **Ethical Review of Web Scraping** – FACT retrieves full web pages via Jina Reader. Were any of the source pages subject to robots.txt restrictions or copyright considerations, and how were these handled?

### Soundness
3

### Presentation
4

### Contribution
3
