# LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
The rapid advancement of Large Language Models (LLMs) has opened new opportunities in data science, yet their practical deployment is often constrained by the challenge of discovering relevant data within large data lakes. Existing methods struggle with this: single-agent systems are quickly overwhelmed by large, heterogeneous files in the data lakes, while multi-agent systems designed based on a master–slave paradigm depend on a rigid central controller that requires precise knowledge of each sub-agent's capabilities. To address these limitations, we propose a novel multi-agent communication paradigm inspired by the blackboard architecture for traditional AI models. In this framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents--either responsible for a partition of the data lake or general information retrieval--volunteer to respond based on their capabilities. This design improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents expertise. We evaluate our method on three benchmarks that require explicit data discovery: KramaBench and modified versions of DS-Bench and DA-Code to incorporate data discovery. Experimental results demonstrate that the blackboard architecture substantially outperforms baselines, including RAG and the master–slave multi-agent paradigm, achieving between 13% to 57% relative improvement in end-to-end task success and up to a 9\% relative gain in F1 score for data discovery over the best-performing baselines across both proprietary and open-source LLMs. Our findings establish the blackboard paradigm as a scalable and generalizable communication framework for multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a multi-agent communication framework for data discovery in data science based on a blackboard architecture. Instead of a master–slave controller assigning subtasks, a main agent posts requests (what information/data is needed) to a shared blackboard. Subordinate agents—each specializing in a partition of the data lake or general retrieval—self-select to respond if they can help. This shifts coordination from centralized routing to decentralized volunteering, improving scalability and flexibility when data lakes contain thousands of heterogeneous files. The system is evaluated on KramaBench and modified DS-Bench/DA-Code with explicit discovery phases, showing 13–57% relative gains in end-to-end task success and up to 9% relative F1 improvement in file discovery over strong baselines (RAG, master–slave multi-agent), across both proprietary and open-source LLMs.

### Strengths
Scalable, decentralized coordination: The blackboard lets agents volunteer based on capability, avoiding brittle central task routing and scaling better to large, heterogeneous data lakes.
Practical for real data discovery: Partitioning the lake and binding agents to clusters reduces context overload and noise sensitivity versus single-agent or naïve RAG setups.
Consistent empirical gains and generality: Outperforms RAG and master–slave baselines across multiple benchmarks and LLM types, indicating robustness and portability of the paradigm.

### Weaknesses
- Incremental novelty: The “blackboard” paradigm is a classic AI architecture; the paper mostly repackages it for LLM agents without a clear, principled advance in control or learning. The comparison to master–slave is framed conceptually, but the paper lacks a formal analysis of when/why blackboard wins (e.g., under capability uncertainty, overlapping expertise, or partial observability).
- No isolation of the communication factor: what if master–slave uses the same file/search agents but with learned routing or voting?
- Data-lake partitioning is done by filename-only clustering with Gemini. This is fragile for enterprise lakes (cryptic names, mixed schemas) and may bias results. No ablation compares filename-only vs content-aware clustering (embeddings, schema inference), nor sensitivity to cluster granularity.
- The paper repeatedly refers to the benchmark as “DS-Bench,” but the correct name is DSBench

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a multi-agent system to find relevant information from large data lakes. Instead of using a master-slave model, this paper proposes a multi-agent communication paradigm inspired by the blackboard architecture of traditional AI. Experiments on three data discovery benchmarks show that the proposed method significantly outperforms existing techniques (RAG and master-slave).

### Strengths
1.  Developing an LLM-driven system for automating end-to-end data science pipelines is interesting and can augment human analysts.

2. This paper is well-organized and easy to read.

### Weaknesses
1. The technical contribution of the work is marginal. The whole pipeline is somewhat not novel and relies on LLM API calls. 

2. The proposed framework relies on multiple LLM API calls throughout its pipeline. A comparative analysis of its computational cost and latency against the baseline methods should be included to provide a complete assessment of its efficiency. In particular, we usually use BM25 and BGE embedding to retrieve relevant files from a large pool in practice. The proposed method may face scalability challenges when retrieving relevant files from a large candidate pool, leading to potentially prohibitive latency in real-world applications.

3. More advanced baselines should be compared in terms of both automatic data science and file discovery.

### Questions
1. The proposed framework relies on multiple LLM API calls throughout its pipeline. A comparative analysis of its computational cost and latency against the baseline methods should be included to provide a complete assessment of its efficiency. In particular, we usually use BM25 and BGE embedding to retrieve relevant files from a large pool in practice. The proposed method may face scalability challenges when retrieving relevant files from a large candidate pool, leading to potentially prohibitive latency in real-world applications.

2. More advanced baselines should be compared in terms of both automatic data science and file discovery.

### Soundness
2

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
The authors propose a multi-agent communication paradigm inspired by the classical blackboard architecture in traditional AI systems. In the proposed framework, a central agent posts queries or requests to a shared blackboard, and a set of autonomous subordinate agents, each responsible for a specific partition of the data lake or a particular retrieval strategy, volunteer to respond based on their capabilities.

### Strengths
The paper offers a fresh perspective on coordination and task allocation in multi-agent systems, drawing inspiration from classical AI architectures but applying it to modern data retrieval challenges.

### Weaknesses
1. The motivation for adopting a blackboard-style coordination mechanism is somewhat underdeveloped. Given the reasoning ability of LLM-based agents, a master agent could, in theory, dynamically decide which sub-agent to query without needing an explicit shared blackboard. This concern is particularly relevant because the current instantiation only includes two helper agents (file and search agents), making the benefits of the blackboard design less evident.
2. It would make this paper stronger if the authors could show more fine-grained analysis on benchmarks. For example, the performance of different architectures based on the number and types of data files.

### Questions
Please check the weaknesses section for more details.

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
This paper proposes a novel multi-agent system for information discovery in data science. Specifically, it proposes a blackboard multi-agent system in which the main agent does not assign tasks to subordinate agents but posts requests to the blackboard. And the subordinate agents autonomously decide whether to respond based on their expertise. The paper evaluates its proposed agent system on three benchmarks including KramaBench, DSBench, and DA-Code and shows its advantage over existing agent systems related to RAG and Master-Slave systems.

### Strengths
1. The blackboard multi-agent system is an interesting idea. Similar systems are practically used in large-scale Internet services such as Uber and other on-demand shared riding platforms. It is excellent to apply similar ideas to LLM-based multi-agent systems.

2. The experiments on three benchmarks including KramaBench, DSBench, and DA-Code show advantages over popular agent systems related to RAG and Master-Slave systems.

### Weaknesses
1. It would be interesting to further analysis how the “Answering” agents behave in the multi-agent system. For instance, how to handle cases that none of the existing subordinate systems choose to respond? Or, on the opposite, how to handle cases that multiple subordinate systems are competing to respond? More analysis or details would further enhance the depth of the paper.

2. The paper seems to lack the integration or comparison with multi-turn agent system like the ReAct systems. The agent performance typically improves significantly by planning, acting, and reflecting for multiple turns. It is unclear how the blackboard agent system can be integrated into the multi-turn ReAct framework.

3. It is unclear why or how the work modifies DSBench and DA-Code for evaluation. It is recommended to release the modification details for the community to reproduce and align the results. Also, for DSBench, it is unclear whether the data analysis task or data modeling task are considered in the evaluation.

### Questions
Typo & Formatting: The Jing et al., 2025 paper uses DSBench instead of DS-Bench. It is recommended to align with the original paper.

### Soundness
3

### Presentation
4

### Contribution
4
