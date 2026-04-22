# DS-STAR: Data Science Agent via Iterative Planning and Verification

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Data science, which transforms raw data into actionable insights, is critical for data-driven decision-making. However, these tasks are often complex, involving steps for exploring multiple data sources and synthesizing findings to deliver insightful answers. While large language models (LLMs) show significant promise in automating this process, they often struggle with heterogeneous data formats and generate sub-optimal analysis plans, as verifying plan sufficiency is inherently difficult without ground-truth labels for such open-ended tasks. To overcome these limitations, we introduce DS-STAR, a novel data science agent. Specifically, DS-STAR makes three key contributions: (1) a data file analysis module that automatically explores and extracts context from diverse data formats, including unstructured types; (2) a verification step where an LLM-based judge evaluates the sufficiency of the analysis plan at each stage; and (3) a sequential planning mechanism that starts with a simple, executable plan and iteratively refines it based on the DS-STAR's feedback until its sufficiency is verified. This iterative refinement allows DS-STAR to reliably navigate complex analyses involving diverse data sources. Our experiments show that DS-STAR achieves state-of-the-art performance across three challenging benchmarks: DABStep, KramaBench, and DA-Code. Moreover, DS-STAR particularly outperforms baselines on hard tasks that require processing multiple data files with heterogeneous formats.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to develop data science agents with LLMs. This paper proposes DS-STAR, consisting of (1) data analysis agent to provide description of the given data; (2) a planning-verification-refinement-execution pipeline to solve the problem; (3) debugging agent to debug code; (4) retriever to retrieve relevant data files. This paper conducts extensive experiments on several data science benchmarks: DABStep, KarmaBench, and DA-Code.

### Strengths
- The writing is of good-quality and easy to follow.
- The investigated research problem – data science agents – are of significance.
- Extensive ablation studies are provided to verify the effectiveness of the proposed agent framework.

### Weaknesses
- Although the research problem is interesting, the technical solution turns out to be plain and naïve. The proposed DA-STAR is lack of technical depths. All the components are well-known recipes for LLM agent community. There are a large number of existing papers that already discuss the planning-execution pipeline. Also, the proposed debugging agent and retriever are also widely investigated for coding agents. As such, I believe the novelty of this paper is quite limited, which is clearly under the bar of top-tier conferences like ICLR.

- The performance improvement is marginal as shonw in Table 2 and Table 3. DS-STAR fails to consistently outperform all the baselines in all the settings, which further decrease the contribution of this paper.

- No empirical results on DA-STAR with open-sourced LLMs. The reliance on commercial LLMs further decrease the contribution of this paper.

### Questions
I think this paper does not make substantial contribution to the community. I believe the rebuttal phase cannot resolve the fundamental limitation of this paper.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
We introduce a data science agent (called DS-STAR), which aims to tackle complex data science tasks automatically. DS-STAR makes three key contributions: exploring and extracting context from diverse data formats, evaluating the sufficiency of the analysis plans, and iteratively refining plans. Experiments on three challenging benchmarks show the effectiveness of DS-STAR.

### Strengths
1. Developing LLM-driven agents for automating end-to-end data science pipelines is interesting and can augment human analysts.

2. The proposed DS-STAR is reasonable and can be applied to various data science tasks. 

3. DS-STAR achieves substantially better performance over several representative baselines across all three challenging benchmarks.

### Weaknesses
1. The reported experimental results are significantly better than the baseline methods. However, after looking at the results on the DABStep leaderboard, AgenticData achieved much higher accuracy on the easy level than DS-STAR. In addition, CambioML DS Agent achieved significantly higher accuracy at both the easy and hard levels than DS-STAR. It would be better to provide some in-depth explanation and analysis.

2. The DS-STAR framework relies on multiple LLM API calls throughout its pipeline. A comparative analysis of its computational cost and latency against the baseline methods should be included to provide a complete assessment of its efficiency.

### Questions
A comparative analysis of its computational cost and latency against the baseline methods should be included to provide a complete assessment of its efficiency.

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
3

### Summary
This paper proposes DS-STAR, a new agent framework designed to address data science tasks, with a particular focus on handling heterogeneous data formats (e.g., CSV, JSON, unstructured text) that current methods often struggle with. The DS-STAR agent is structured as a workflow composed of three main modules: (1) a data file analysis module to explore and extract context from diverse data files, (2) a verification module (using an LLM as a judge) to evaluate the sufficiency of the analysis plan, and (3) an iterative planning and refinement module that sequentially builds and optimizes the solution. The authors evaluate their method on three data science benchmarks (DABStep, KramaBench, DA-Code), demonstrating improved performance over existing baselines.

### Strengths
1. The paper addresses a practical problem. Real-world data science tasks frequently involve a variety of heterogeneous data sources. 
2. The experimental evaluation is relatively thorough. The authors have validated their approach on three benchmarks and included ablation studies to demonstrate the contribution of the different components.

### Weaknesses
1. The primary weakness of this paper is its limited novelty. The proposed method essentially defines a specific workflow for a data science agent, but the core components are very similar to those used in many existing agent frameworks. Beyond the data file analysis module, the other components—such as using an LLM for verification or employing an iterative planning/refinement loop—are common, almost standard, techniques in the current agent literature. 
2. The experimental evaluation, while comprehensive in accuracy, appears to be incomplete in a critical practical dimension: efficiency. The paper does not present an analysis of the computational time or monetary cost (e.g., API calls) of the proposed method compared to the baselines. Given the multi-module, iterative nature of DS-STAR, it is highly likely that it incurs significantly higher latency and cost.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DS-STAR, a novel data science agent framework designed to automate complex data science tasks through a robust process of iterative planning and verification. The framework is designed to overcome limitations of existing LLM agents, namely their struggle with heterogeneous data formats and the tendency to generate sub-optimal analysis plans due to the difficulty of verifying plan sufficiency in open-ended tasks.

### Strengths
- Robust Handling of Heterogeneous Data: The proposed Data File Analysis Module is crucial for real-world applicability. By automatically exploring and extracting structural and content summaries from various heterogeneous formats (CSV, JSON, unstructured text, etc.), DS-STAR significantly broadens the scope of data science agents beyond the structured data limitation of previous methods.


- Effective Iterative Planning Strategy: The use of sequential planning and iterative refinement, starting with a simple executable step and incrementally building the solution based on verification feedback, effectively mimics the interactive analysis process of a domain expert working in a Jupyter Notebook. This approach is highly beneficial for tackling complex, multi-step analysis tasks involving dependent results.

- Superior Performance on Hard Tasks: DS-STAR achieves SOTA performance across multiple challenging data science benchmarks. Its significant outperformance on "Hard Tasks" that require processing multiple, heterogeneous data files strongly validates the efficacy of its key design components, especially the data analysis and verification modules.

### Weaknesses
1. Reliability and Bias of the Verification Module: The performance of DS-STAR is highly dependent on the quality of the LLM Judge's judgment of "plan sufficiency." The paper lacks an in-depth analysis of the Judge's reliability, generalizability, and potential inherent biases across different task difficulties or underlying LLM architectures. Errors in the Judge's assessment could lead the agent down flawed iterative paths or cause it to prematurely accept sub-optimal solutions.

2. Efficiency and Cost Concerns: The iterative planning and multi-step verification process (Planner $\rightarrow$ Coder $\rightarrow$ Judge $\rightarrow$ Refinement) is inherently a multi-turn loop. Compared to single-shot planning or baselines that rely only on code execution success, solving a single task likely requires multiple LLM API calls. The paper must quantify the average inference time and API call count for DS-STAR and discuss the trade-offs in latency and computational cost.

3. Boundary Limitations of Data Summarization: While the analysis module is designed to handle heterogeneous data, the ability of a Python script and LLM to generate an accurate and complete text summary for highly complex, large-scale, or deeply nested unstructured data remains questionable. If the generated summary is incomplete or flawed, subsequent planning will be fundamentally biased from the outset.

### Questions
Sensitivity Analysis of the LLM Judge: The quality of the LLM Judge determines the framework's performance ceiling. Please conduct a sensitivity analysis on the Judge agent, including: 
       1. varying the underlying LLM used for the Judge (e.g., substituting GPT-4 with a powerful open-source model); 
       2. varying the specific prompt template used to instruct the Judge. This analysis is critical for assessing the framework's robustness and practical deployability.

### Soundness
2

### Presentation
2

### Contribution
2
