# EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Machine learning models for clinical prediction rely on structured data extracted from Electronic Medical Records (EMRs), yet this process remains dominated by hardcoded, database-specific pipelines for cohort definition, feature selection, and code mapping. These manual efforts limit scalability, reproducibility, and cross-institutional generalization. To address this, we introduce EMR-AGENT (Automated Generalized Extraction and Navigation Tool), an agent-based framework that replaces manual rule writing with dynamic, language model-driven interaction to extract and standardize structured clinical data. Our framework automates cohort selection, feature extraction, and code mapping through interactive querying of databases. Our modular agents iteratively observe query results and reason over schema and documentation, using SQL not just for data retrieval but also as a tool for database observation and decision making. This eliminates the need for hand-crafted, schema-specific logic. To enable rigorous evaluation, we develop a benchmarking codebase for three EMR databases (MIMIC-III, eICU, SICdb), including both seen and unseen schema settings. Our results demonstrate strong performance and generalization across these databases, highlighting the feasibility of automating a process previously thought to require expert-driven design. The code will be released publicly. For a demonstration, please visit our anonymous demo page: https://anonymoususer-max600.github.io/EMR_AGENT/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an LLM-based agentic framework for extracting analysis cohorts from EHR-based datasets, specifically those from an ICU setting (MIMIC, eICU, SICdb). As there are no established evaluation sets for this task yet, the authors define several use cases to evaluate their agent.

### Strengths
- This paper develops a thorough agentic framework that matches expert workflows, including external database documentation and iterative querying of the data to identify necessary data items. 
- The authors propose a standardised evaluation protocol for assessment, which is applied across three commonly used datasets, one of which is a very recent dataset published after their underlying LLM. 
- The authors perform detailed ablation studies of each component of their framework.

### Weaknesses
- The paper is generally well written but I was missing some details on the main experiments (see Questions below)
- The evaluation sets for Cohort Selection are limited to 7 relatively simple scenarios. The ground truth comparisons to Harutyunyan and Sheikhalishahi are similarly limited to simple selections from those benchmarks. As a result, these scenarios are not representative of the actual breadth and depth of real-world cohort selections, which often involves complex combinations of observations (e.g., sepsis or ventilation). 
- The evaluation sets for Code Mapping are limited to common labs, which are again among the simplest use cases for data harmonisation between ICU datasets. More complicated concepts such as medications or ventilation parameters are important for many research projects but not covered by the evaluation. 
- It is hard to assess how impressive the performance of the agent is. The accuracy of ~30% for code mapping does not seem impressive but it is unclear whether this is driven by a bad agent or by any inherent uncertainty in the mapping task. A comparison to human mappers would help put those results in context.

### Questions
- I am unclear how accuracy was calculated for the Feature Selection task. For the example of Evaluation Set 1, is this just the proportion of cells in stay ID, gender, age, and length of stay that contained the same value as in the ground truth extract? Is this just for the subset of correctly identified patients? What if additional features/columns are returned, are those penalised or just ignored? What about permutations of the order of features?
- I am unclear how Code Mapping was evaluated. Was this just a multi-class prediction that assigned each possible feature (e.g., each entry in d_labitems and d_items in MIMIC) to either one of the 56 features and a bucket "other"? How can it be that "Ours w/o Candidate Matching "got a result of 0 for both F1 and bAcc?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces EMR-AGENT, an agentic framework to replace manual rule writing with dynamic language-model driven interaction to extract and standardize structured clinical data. The authors construct a standardized evaluation protocol and codebase for EMR preprocessing. The approach is evaluated through component-level ablations, comparisons with other LLM-based approaches, and evaluation on a held-out EMR database.

### Strengths
The paper is clear and well-written, and claims are properly backed by evidence and/or results.

The work seems significant in its application area, and provides strong solutions for described problems and limitations of existing work.

Experiments are sound and transparent, components of the approach are properly ablated, results are averaged over repeated trials, and the appendix contains sufficient details on specific prompts, benchmark construction process, and relevant reproduction details for the baselines.

### Weaknesses
While the work seems significant in its application area, it would likely be better suited for an application-specific journal, conference, or relevant workshop. 

**Main Reasoning**

- While the described approach may be novel for EMR preprocessing specifically, similar approaches featuring described components such as iterative refinement, error handling, dynamic contexts featuring manuals or metadata, etc. are common and in some cases more comprehensive and generalizable in other applications of agentic LLMs such as search, literature review, or software engineering.
- The described prompts and steps are specific to EMR preprocessing, which is a relatively narrow application of a specific field.
- There are no other contributions such as additional training or fine-tuning approaches that would make the work a better fit.

While these reasons on their own are not necessarily grounds for rejection, I feel that in aggregate the paper is not a strong fit for ICLR.

### Questions
Given that my concerns are not with the execution or quality of the work, but mainly its relevance to ICLR specifically, I am open for discussion with the authors on my concerns outlined in the weaknesses section above.

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
This paper presents EMR-AGENT, a large language model (LLM)-driven framework that automates preprocessing of Electronic Medical Records (EMRs)—specifically cohort selection, feature extraction, and code mapping—tasks that traditionally rely on labor-intensive, database-specific SQL pipelines. The proposed system replaces manual rule-writing with an agent-based architecture that dynamically interacts with EMR databases through SQL queries, observes query outputs, and reasons over schema and documentation to iteratively refine its extraction strategy. EMR-AGENT is composed of two agents: the Cohort and Feature Selection Agent (CFSA) and the Code Mapping Agent (CMA). Both agents perform schema linking and guideline generation before engaging in SQL-based exploration, error correction, and iterative refinement.

To evaluate the framework, the authors build a new benchmark suite called PreCISE-EMR, covering three major ICU datasets—MIMIC-III, eICU, and SICdb—with tasks measuring both seen (MIMIC-III, eICU) and unseen (SICdb) schemas. Results show that EMR-AGENT significantly outperforms existing Text-to-SQL baselines such as PLUQ, EHR-SeqSQL, and DIN-SQL across all databases. CFSA achieves up to 0.94 F1 on MIMIC-III and 0.81 on unseen SICdb, while CMA attains 0.65 F1 on eICU. Ablations confirm that interactive SQL observation and schema-guided reasoning are critical for performance, and external documents (manuals, memos) greatly enhance robustness. The system generalizes across multiple LLM backbones, with Claude-3.5-Sonnet and Claude-3.7-Sonnet performing best, and demonstrates the feasibility of replacing rigid, expert-driven EMR pipelines with flexible, language model-based automation

### Strengths
- EMR-AGENT reframes cohort and feature extraction as an interactive reasoning problem rather than static rule application, marking a clear conceptual advance over existing harmonization frameworks such as YAIB, ACES, and Clairvoyance, which rely on handcrafted mappings.
- The paper carefully decomposes the agent workflow into interpretable modules (Schema Linking, SQL-based Observation, Error Feedback, and Candidate Matching). The visual diagrams (Fig. 2a–b, p. 4) make clear how these stages correspond to classical data engineering steps, lending transparency and auditability to the LLM-driven process.
- The authors do not merely report end-task performance but construct a full benchmarking environment (PreCISE-EMR)—with PostgreSQL instances, human-curated ground-truth mappings, and evaluation code—establishing a foundation for future reproducible research.
- The model maintains high accuracy across MIMIC-III, eICU, and SICdb, despite the latter being unseen during pretraining. This is particularly impressive given that schema heterogeneity is one of the hardest obstacles in real-world EMR integration.
- The benchmark includes human clinical oversight (two physicians, two nurses, one informatics expert), strengthening the validity of the evaluation results. 
- By adapting leading Text-to-SQL models (PLUQ, EHR-SeqSQL, DIN-SQL, REACT) to the same schema environment, the authors show that naive translation systems fail dramatically (F1 ≤ 0.1 on some datasets), reinforcing the claim that dynamic, context-aware exploration is essential.
- The evaluation in Table 4 confirms that performance scales with reasoning capacity: Claude-3.5-Sonnet > Claude-3.5-haiku > open-source Llama3.1 and Qwen2.5, highlighting the need for reasoning-rich models while offering an efficient option for constrained settings.

### Weaknesses
- While EMR-AGENT automates query generation and schema reasoning, it stops short of true end-to-end preprocessing. Manual approval is still implicitly required to verify final cohorts and mappings, and the paper does not quantify human-in-the-loop correction effort.
- Even though CMA outperforms baselines, absolute F1 values (≈ 0.5) remain modest, suggesting that vocabulary harmonization remains an open challenge, especially for free-text or ambiguous concept labels.
- Cite the work of MEDS which has done extensive work to create a data schema for EMR/EHR (https://dl.acm.org/doi/abs/10.1145/3711896.3737608?casa_token=MkA4rBEmTXMAAAAA:_fER13VZVnNzAtiZ6GE2dgZIvmIGmhsObsrNj1u9zx1oamEj4-YU-emnUZZVyOX3OVAzvMPefRkv, https://openreview.net/forum?id=IsHy2ebjIG)
- While the framework claims to replace manual rule writing, the paper provides no measurement of time saved or human equivalence benchmarks (e.g., accuracy vs. expert engineer).
- Iterative querying over large databases could be computationally expensive, especially under tight token limits. The paper does not estimate query cost or latency across agents.

### Questions
- How does EMR-AGENT ensure semantic fidelity between user intent and generated SQL queries in ambiguous natural language inputs?
- Can you quantify the average number of SQL iterations or corrections per task, and how this correlates with performance?
- What mechanisms prevent hallucinated schema entities or unsafe queries (e.g., full-table scans on large hospital databases)?
- How would EMR-AGENT handle an EMR system with missingness or limited column metadata? does performance degrade gracefully or fail catastrophically?

### Soundness
3

### Presentation
3

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
This paper introduces EMR-AGENT, agent-based framework for automating preprocessing of EMRs. The framework replaces manually crafted, schema-specific rules with LM-driven agents that iteratively query, observe, and reason over database schemas and documentation. Authors also introduce a new benchmark suite built on three EMR databases to assess performance under both seen and unseen schema settings. Ablation studies confirm the importance of schema-guided linking, SQL-based observation, and error feedback.

### Strengths
- Introduces the first agent-based, LLM-driven framework for EMR preprocessing
- Pretty comprehensive experimental design, including multi-database evaluation, detailed ablation studies, and model generalization tests.
- Empirical evidence from MIMIC-III, eICU, and SICdb
- Benchmark contribution is valuable.

### Weaknesses
- Limited interpretability analysis into how agents make reasoning decisions or handle ambiguous schema mappings
- No comparison of computational costs against other baselines unless i am missing something since agent-based methods are very expensive when recursive

### Questions
- How does EMR-AGENT handle ambiguous or missing schema documentation in entirely novel hospital EMRs beyond the three tested datasets?
- Could the framework be adapted for continuous schema evolution (e.g., schema drift in live hospital systems)?
- Have the authors evaluated whether EMR-AGENT’s extracted data lead to comparable or superior downstream model performance (e.g., mortality prediction) compared to manual preprocessing pipelines?

### Soundness
2

### Presentation
3

### Contribution
2
