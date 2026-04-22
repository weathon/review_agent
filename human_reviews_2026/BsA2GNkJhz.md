# CellAgent: LLM-Driven Multi-Agent Framework  for Natural Language-Based Single-Cell Analysis

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 2

## Abstract
Single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (ST) data analysis are pivotal for advancing biological research, enabling precise characterization of cellular heterogeneity. However, existing analysis approaches require extensive manual programming and complex tool integration, posing significant challenges for researchers. To address this, we introduce CellAgent, an autonomous, LLM-driven approach that performs end-to-end scRNA-seq and spatial transcriptomics data analysis through natural language interactions. CellAgent employs a multi-agent hierarchical decision-making framework, simulating a “deep-thinking” workflow to ensure that analytical steps are logically coherent and aligned with the overarching research goal. To further enhance its capabilities, we develop sc-Omni, a high-performance, expert-curated toolkit that consolidates essential tools for scRNA-seq and spatial transcriptomics analysis. Additionally, we introduce a self-reflective optimization mechanism, enabling automated, iterative refinement of results through specialized evaluation methods, effectively replacing traditional manual assessments. Benchmarking against human experts demonstrates that CellAgent achieves significant improvement in efficiency across multiple downstream applications while maintaining excellent performance comparable to existing approaches and preserving natural language interactions. By translating high-level scientific questions into optimized computational workflows, CellAgent represents a step toward a new, more accessible paradigm in bioinformatics, allowing researchers to perform complex data analyses autonomously. In lowering technical barriers, CellAgent serves to advance the democratization of the scientific discovery process in genomics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The manuscript proposes CellAgent, a large language model (LLM)–driven multi-agent framework designed to automate end-to-end single-cell RNA-seq (scRNA-seq) and spatial transcriptomics (ST) analyses through natural language interaction. The system includes three cooperating agents (Planner, Executor, and Evaluator) simulating a “deep-thinking” workflow.

### Strengths
The paper’s main strengths are its timely idea, an LLM-driven multi-agent framework for single-cell and spatial analyses, and its practical integration of many standard tools to automate end-to-end workflows via natural-language queries. The experiments are comprehensive, and the performance is better than the compared methods.

### Weaknesses
However, the novelty beyond tool orchestration is limited, and benchmarks lack rigor (unclear baselines, no variance/error bars). The system’s reliance on proprietary LLMs also weakens reproducibility, and the biological validation largely reaffirms known findings rather than yielding new insights.

### Questions
Here are several comments:
1. While the proposed multi-agent structure (Planner–Executor–Evaluator) is conceptually neat, similar frameworks have already appeared in recent LLM-agent works such as AutoBA, AutoGen, and BioGPT-Agent. The manuscript does not sufficiently clarify what is fundamentally novel beyond combining these paradigms with existing scRNA-seq pipelines. The “self-reflective optimization” idea is interesting, but remains conceptually vague. It’s unclear whether this is an adaptive fine-tuning loop, reinforcement feedback, or a rule-based scoring system.
2. The paper reports high success rates (96%) and better-than-human efficiency but provides limited methodological transparency. No details on how human expert baselines were defined or how evaluation bias was mitigated. Figures like Table 1 aggregate many metrics without error bars, variance, or reproducibility analysis.
3. Although the system outputs biologically consistent results (e.g., trajectory inference of hematopoietic stem cells), it remains unclear how much biological reasoning actually comes from the model versus pre-existing toolkits (like Celltypist, Slingshot, Harmony, Tangram). CellAgent mainly orchestrates existing methods, rather than discovering new biology. Thus, the biological insights appear demonstrative rather than novel.
4. The paper benchmarks CellAgent against individual tools such as scGPT, Tangram, SpaGE, etc., but not against integrated automated frameworks (e.g., ezSingleCell, scPipeline, or AutoBA). Moreover, it is not clear whether CellAgent used identical datasets and preprocessing steps. This could artificially favor its results due to built-in optimizations or curated defaults.
5. The Related Work section is mostly descriptive and does not critically position CellAgent among existing AI agent frameworks in bioinformatics. The authors should explicitly discuss why prior LLM-based pipelines are insufficient and how CellAgent extends them.
6. I have tried CellAgent online. Currently, it only allows the uploaded files that are less than 1GB. But lots of scRNA-seq data or single-cell-resolution ST data have larger sizes unless the analysis is based on the toy dataset. The authors should resolve this problem.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents CellAgent, an LLM-based agent for the analysis of single-cell data. The system follows a three-step analytical pipeline comprising planning, execution (including tool selection and code generation), and evaluation. The framework is implemented as a natural-language chat interface through which users can upload single-cell or spatial data and perform interactive exploration. The paper also includes a benchmark comparing the agent against other computational utilities and a baseline LLM in terms of biological task accuracy and ease of use.

### Strengths
Chat interface implemented at supported by the authors might be a helpful resource for people starting with single cell analysis.

### Weaknesses
This work is not the first attempt to automate single-cell analysis through agentic systems. Existing coding copilots such as Claude Code, OpenAI Codex, and Microsoft Copilot already support major single-cell analysis utilities, allowing users to analyze datasets locally and at lower cost. Moreover, domain-specific agents like Biomni offer broader functionality than CellAgent, while more specialized tools such as scAgent and Cell2Scale can perform key downstream analyses including cell-type annotation and exploration of cellular identity. The main distinction between Biomni and general-purpose copilots versus CellAgent lies in their flexibility—they are not constrained by a hardcoded set of tools, making them more adaptable to specific datasets and more easily extensible to new packages. 

CellAgent also does not implement any guardrails to control the LLM hallucinations or any procedures to inspect the quality of generated code that could potentially compromise the quality of the findings.

### Questions
Below I suggest modifications that can help improve the paper.

1. Illustrate the novelty and efficiency of the proposed approach as compared to existing alternatives. How would **Cursor** on the one hand and **Biomni** perform the same task — are they worse than **CellAgent**?

2. Illustrate the robustness of **CellAgent** to LLM hallucinations. Also, how would **CellAgent** behave when prompted with a complex task that includes multiple steps?

3. The paper develops two code bases, **scOMNI** and **CellAgent**. While **scOMNI** is public, **CellAgent** is not. It is hard to evaluate the validity of the approach when the code for it is hidden. In **scOMNI**, I notice that many paths are hard-coded; this would be a major limitation to running the method locally.

4. There are several unclear aspects in the paper. For example, how do you run other methods for the benchmark? How do you compute the *AS* score for spatial domain identification in embryo, breast cancer, and kidney datasets, which do not have ground-truth data?

5. Cell-type identification is performed on a relatively easy dataset with coarse cell-type labels — extend it to more complex datasets with fine-grained labels and compare with other agentic approaches for cell typing, including a simple baseline where you give the top 50 marker genes to GPT-5 and ask it to reason over the data.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CellAgent, a large language model–driven multi-agent framework for fully automated single-cell and spatial transcriptomics analysis through natural language interaction. The system integrates three agents, namely Planner, Executor, and Evaluator, operating under a self-reflective optimization loop that plans tasks, generates code, and evaluates results. A comprehensive benchmarking across over 60 datasets is also presented.

### Strengths
1) The study presents a technically sophisticated and well-structured multi-agent framework t
2) The inclusion of the sc-Omni toolkit provides a strong and practical foundation, consolidating many established analysis methods into a unified framework.
3) The comprehensive benchmarking on multiple downstream tasks supports the robustness and practical relevance of the method.

### Weaknesses
1) The biological interpretation of the outputs is limited. The evaluation focuses mainly on computational metrics without assessing whether the inferred results align with known biological mechanisms.
2) The evaluation of “human expert performance” lacks sufficient detail on the experimental design, number of participants, and reproducibility of human assessments.
3) The author do not discuss the advantages / disadvantages of their framework against some other relevant systesm, most notably SpatialAgent:
https://www.biorxiv.org/content/10.1101/2025.04.03.646459v1.abstract

### Questions
I would ask the authors to address the weaknesses highlighted above:
1) Please include biologically grounded evaluations, such as testing whether CellAgent recapitulates known spatial tissue structures in benchmark datasets. Walking the reader through one representative dataset out of the 60 would suffice
2) Provide more details on the human expert evaluation, including participant expertise, evaluation protocol, and inter-rater variability.
3) Please ensure that you exhaustively discuss the state-of-the-art.

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
5

### Summary
This paper introduces CellAgent, a multi-agent framework that uses LLMs organized into three biological Planner, Executor, and Evaluator to automate end-to-end scRNA-seq analysis, including preprocessing, batch correction, cell-type annotation, and trajectory inference. CellAgent combines a curated toolkit sc-Omni of domain methods, runs generated code in a notebook sandbox, and applies a self-reflective optimization loop.

### Strengths
1. CellAgent automates single-cell and spatial analyses via natural language, which is a meaningful system that has clear practical value.

2. The paper includes rich biological visualizations, enhances interpretability, and aligns with standard practices in single-cell and spatial omics.

### Weaknesses
1. CellAgent proposes an empirical engineering system by combining existing tools, but fails to elevate the framework into a principal scientific formulation. It does not clearly articulate why or under what principles the multi-agent coordination works.

2. There is only one type of  LLM (GPT-4o) that both interprets results and decides which algorithm is best, which raises the risk of self-circularity. Masking algorithm IDs alone do not guarantee fairness. The absence of human-expert-LLM-related validation or multi-LLMs consensus undermines the robustness and objectivity of the reported performance claims.

3. It’s unclear whether the framework ensures scientific validity, or simply produces metric scores and visual plots. Since it lacks expert-reviewed case studies or even a single example where the agent detects and corrects a biologically unreasonable output. In single-cell analysis, the former is not enough.

4. The paper lacks ablation studies to justify the necessity of each component, or based on different LLM APIs apart from GPT-4o. Only plainly stating that it simulates the workflow of deep-thinking is not reasonable.

### Questions
Could you release exact dataset lists (accession IDs), all preprocessing scripts or methods with random seeds?

Could you specify the configuration or computational cost of the LLM components, including key parameters (e.g., temperature, tokens, API version, number of calls per task) and the estimation of the total inference cost and runtime for the experiments?

What is the success rate of CellAgent? What will happen if it chooses the wrong tool? Have you analyzed the types of tasks or datasets where CellAgent fails or produces suboptimal pipelines?

### Soundness
2

### Presentation
3

### Contribution
2
