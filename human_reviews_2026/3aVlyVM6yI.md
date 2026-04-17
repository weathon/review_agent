# IAgent: A Web Search Framework for Noise Isolation and Extended Information Access

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 4

## Abstract
Web search agent frameworks built on Large Language Models (LLMs) can leverage multi-source, real-time external information, demonstrating strong potential. However, for deep research tasks that require the integration of massive information, existing agent frameworks not only generate substantial noise during search and context management, which affects the output of the final answer, but also suffer from the inherent limitations of relying solely on web browsing for information acquisition. To address the noise generated during the search process, we design a new web search filtering module that can isolate irrelevant webpages at an early stage. To mitigate the contextual noise that impacts the final answer, we introduce an isolation-based stepwise verification module, which reduces LLM contextual bias and ensures more neutral and reliable outputs. Moreover, to expand information acquisition and address the challenge of insufficient reasoning, we introduce a new agent that enables scalable information acquisition through APIs and programmatic automation, while also supporting efficient deep reasoning. Experimental results show that on the GAIA and WebWalkerQA benchmarks, IAgent achieves state-of-the-art performance among open-source deep research agents. Furthermore, IAgent reduces token cost by 33\% compared to our baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce IAgent, a novel multi-agent framework designed to address two critical challenges in existing systems: noise interference and monolithic information acquisition. IAgent has three main contributions: 1) A context-isolated filter module to mitigate search noise and LLM inertia bias at an early stage. 2) An isolation-based stepwise validation module to counteract contextual noise and sycophantic tendencies. 3) The introduction of a CoderAgent that extends information gathering capabilities beyond simple web browsing by leveraging APIs and programmatic automation. Experiments on the GAIA and WebWalkerQA datasets show that IAgent achieves better performance and reduces token costs by 38% compared to the baseline.

### Strengths
1. The design of IAgent appears to offer effective and insightful solutions to the drawbacks of current web search agents (noise interference and monolithic information acquisition).
2. The paper has Excellent Clarity and Presentation, and the figures illustrating the problems and methods are also clearly displayed.
3. According to the experimental results in the paper, IAgent outperforms the baselines on the two deep research benchmarks and consumes fewer tokens. Furthermore, the experiments are also quite comprehensive, including an effectiveness analysis for each module within IAgent.

### Weaknesses
1. When testing the cost of answering, the authors only experimented on the first 20 queries. This proportion is a bit small for a dataset with a total of 166 queries, and they were not a random sample of 20 queries. Whether the 38% token cost reduction can actually be achieved needs further verification.
2. In the experimental section, the backbone LLM used in IAgent and the various baselines are different, which may mean that the final experimental results are influenced by more than just the agent architecture.
3. The results in Table 1 have quite a few blanks. While this is certainly due to different agents testing on different benchmarks and the high cost of experimentation, if the results for WebWalkerQA could be completed for "Ours," it would help improve the robustness of the paper. Alternatively, use other datasets that smolagents has been tested on?

### Questions
1. Have the authors considered using a stronger open-source LLM, such as Qwen3-32B, as the backbone model to conduct unified testing across the different agent frameworks? This would help more rigorously test the performance differences between the various Agent architectures.

### Soundness
3

### Presentation
3

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
The authors introduce IAgent, an agent scaffold for web search tasks with specific filtering and context management steps for robustness and token efficiency. The authors argue that LMs exhibit “inertia bias” (i.e. they go along with their prior choices in context) and “contextual bias”, where increasing amounts of information and noise yield poor design choices by the agent. Their agent scaffold uses a “manager” and “worker/search” agent setup with a few extra design decisions: 1) to mitigate “inertia bias” due to noise, they include a filtering step of retrieved webpages for the “search” agent, 2) a module for reviewing the “manager” agent’s reasoning chain prior to producing a final answer, and 3) adding another worker agent with access to a code environment. They evaluate their approach on GAIA and WebWalker against other open-source (+ o3) approaches and provide a breakdown for how the agent solves these tasks.

### Strengths
1. IAgent improves over the general smolagent scaffold in both performance and cost on both GAIA and WebWalkerQA.
2. The general strategy of robust filtering of webpages and handling context management through separate LM calls is clever. 
3. The analysis in Figure 6 and 7 is useful for understanding the reasoning trajectory of the agent with respect to each of its components.

### Weaknesses
1. Table 1 is confusing. There are several (10+) submissions on the public GAIA leaderboard [1, 2] with higher scores across all 3 levels, with many of these scaffolds being open source. It should be made more clear what the distinction between these results and the reported GAIA numbers are.
2. There are few ablations on the design of this scaffold, which includes several design choices on top of existing web agents. The primary analysis is Figure 6 and Figure 7, which show the agent usage / token breakdown of the main experiments, but it is unclear how to interpret the usefulness of each component separately.
3. Many of the baselines in Table 1 do not seem fair – the choice of model is not consistent across each agent scaffold, making it difficult to compare. The authors also argue that “carefully selecting model combinations is crucial”, but the baselines were not tuned fairly with this in mind.
4. There is little qualitative or quantitative analysis on where IAgent improves over existing, more task-specific scaffolds. smolagents is designed to be a general agentic scaffold, so the fact that IAgent can improve upon it is not very surprising.
5. In Table 5, we see that Config 2 uses Claude-4 and other newer frontier models. Many of the Table 1 experiments (including smolagent) do not use these newer models, which makes it unclear whether improvements are due to the scaffold or due to better base LMs.

### Questions
1. “Our experimental results demonstrate that IAgent achieves a significant performance enhancement over the smolagents baseline on the GAIA benchmark (Mialon et al., 2023) under similar model configurations.” Similar model configurations seems to just mean they use the same models, but IAgent appears to be a specific instantiation of the more general smolagents framework, so how much of its improvements over smolagent are attributed to more task-specific tuning for these types of tasks?
2. Why were smolagents results on WebWalkerQA not included? The paper states “As smolagents provide results solely on GAIA, we restrict evaluation of this configuration to the same benchmark”, but since the analysis centers on IAgent vs. smolagent, why not run smolagents on this benchmark?
3. Can you clarify how the results in Table 1 compare to the online reported results and why they were omitted / not included? Many officially reported scaffolds achieve 90%+ on GAIA Level 1 for example.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces context-isolated filtering module and isolation-based stepwise validation module for reducing noises occurred during search-intensive tasks. Additionally, the paper utilizes CodeAgent as an additional module for solving mathematical problems or repetitive subtasks. By integrating the proposed module on top of SmolAgent, IAgent improved peformances on the GAIA benchmark, and reduced the token cost.

### Strengths
* The paper points out concrete weaknesses in current web search agents (e.g., inertia bias) and frames “noise isolation & verification” as a main direction.
* The proposed modules are quite simple, well motivated, and empirically effective.  
* The study provides clear quantitative insights into how each module contributes (e.g., validation correcting 6.7% of errors, filter reducing 27% redundant searches).

### Weaknesses
[W1] While the proposed modules are intuitive and simple, the proposed modules lack research novelty but rather a straightforward application aimed at deep research tasks.

[W2] Different backbone across methods
* In table 1, backbone LLMs for baselines are different from that of IAgent. As the main contribution of this work is orthogonal to choice of the backbone LLM, the authors should demonstrate superiority of IAgent over the baselines (e.g., SmolAgent, OWL) under identical backbone LLMs.
 
* In line 367: “The first configuration (Config. 1), intended to provide a direct baseline comparison with smolagents, powers all agents with GPT-4.1 (OpenAI, 2025), whose performance is similar to that of GPT-4o (OpenAI, 2024).”, why not compare IAgent and SmolAgent under GPT-4o? I think “performance of GPT-4o and GPT-4.1 is similar” is not a valid reason for comparing IAgent + GPT-4.1 with SmolAgent + GPT-4o. 

If the authors show that IAgent is remarkably superior to SmolAgent under the same LLM backbone, I will raise my recommendation.

[W3] Why did the authors not provide results corresponding to SmolAgent in WebWalker? As SmolAgent is a direct baseline, I think the authors need to show the superiority of IAgent over SmolAgent in WebWalker benchmark.


[W4] Different subset sampling across analysis.

Analysis regarding token usage and ablations on “filter module in Search Agent” are done on subset of GAIA benchmark (20 tasks out of 165 tasks), but the subsets are not unified (prior one is first 20 problems, and second one is randomly sampled 20 problems). Authors should fix the subset, and conduct additional analysis on the fixed subset.

### Questions
[Q1] What is the performance of IAgent without CoderAgent?

[Q2] The tables are mainly presents final score on each benchmark. Is it possible to quantitively analyze whether the context-isolation module indeed reduces inertia bias?

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
4

### Summary
The paper introduces IAgent, an agentic web search framework that explicitly addresses the limitations of Deep Research methods corresponding to noisy search results and incorrect reasoning steps. Specifically, a filtering module is introduced to remove irrelevant webpages  from the search results. Next, a reasoning verification module is added to validate the correctness of the reasoning process in a stepwise fashion. The paper also adds a code-execution approach when multiple parallel iterations of search calls are necessary. The method is benchmarked on popular evaluation sets, such as GAIA and WebwalkerQA.

### Strengths
1. The methodology is described very well, with clear figures and the paper is easy to read. 
2. The experimental results are impressive, with considerably better performance than the Smolagents baseline

### Weaknesses
1. The baseline comparison is not apples-to-apples since a variety of models are used, making it hard to understand how reliable the improvements are. The smolagents baselines uses GPT-4o while IAgent approach uses considerably more recent/performant models. The authors should consider adding a GPT-4.1 variant of Smolagents. 
2. While there is an overall improvement over Smalagents, there is a significant degradation in performance on level-3, representing the more difficult subset. This brings into question whether the paper’s method is mainly effective for the more easier deep research questions. I would have expected the opposite, since the proposed CoderAgent and reasoning-verification should be more effective for the harder tasks. The authors should consider benchmarking on BrowseComp, which has been considered much more harder benchmark for deep research.
3. The approach in the paper looks highly hand-stitched and only relying on closed-source models, making it unclear how to pursue further improvements with training. The authors should consider showing performance with an open-source model like GPT-OSS-20B or Qwen3-32B. Would be great if they can further improve performance of these models with an approach like SFT over closed-source model trajectories. 
4. The related work section needs more revision, specifically to incorporate discussion on how the proposed method differs from existing multi-agent methods. The related work currently just reads like a summary of prior work in each sub-category.

### Questions
1. While token-count cost is compared, how does the comparison of inference time look like for IAgent vs Smolagents when using the same LLM?

### Soundness
3

### Presentation
3

### Contribution
3
