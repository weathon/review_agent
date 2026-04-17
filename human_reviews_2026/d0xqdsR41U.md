# WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Powered by a large language model (LLM), a web browsing agent operates web browsers in a human-like manner and offers a highly transparent path toward automating a wide range of everyday tasks. As web agents become increasingly capable and demonstrate proficiency in general browsing tasks, a critical question emerges: $\textit{Can they go beyond general browsing to robustly handle tasks that are tedious and complex, or chores that humans often avoid doing themselves?}$ In this paper, we introduce \textbf{WebChoreArena}, a new fully reproducible benchmark comprising 532 carefully curated tasks over 300+ hours, designed to address more labor-intensive and tedious tasks. WebChoreArena systematically integrates three key challenges: (i) $\textbf{Massive Memory}$ tasks requiring accurate retrieval of large amounts of information in the observations, (ii) $\textbf{Calculation}$ tasks demanding precise mathematical reasoning, and (iii) $\textbf{Long-Term Memory}$ tasks necessitating long-term memory across multiple webpages. Built on top of the fully reproducible and widely adopted four WebArena environments, WebChoreArena ensures strict reproducibility and enables fair, direct comparisons with the established WebArena benchmark, offering key insights into agent progress. Our experimental results demonstrate that as LLMs evolve, significant performance improvements are observed on WebChoreArena. These findings suggest that WebChoreArena is well-suited to measure the advancement of state-of-the-art LLMs with greater clarity. Nevertheless, the results also indicate that even with GPT-5, there remains substantial room for improvement compared to WebArena, highlighting the increased challenges posed by WebChoreArena.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **WebChoreArena**, a new benchmark designed to systematically evaluate LLM-powered web agents on tedious, memory-intensive, and reasoning-demanding web tasks. The benchmark consists of 532 manually constructed tasks across four simulated WebArena environments — *Shopping*, *Shopping Admin*, *Reddit*, and *GitLab* — totaling over 300 hours of human annotation. Tasks are categorized into Massive Memory, Calculation, Long-Term Memory, and Others, enabling comprehensive assessment of agents’ cognitive and operational capabilities. Fully compatible with WebArena for reproducibility and comparability, the authors evaluate AgentOccam and BrowserGym frameworks using several state-of-the-art LLMs, including GPT-4o, Claude Sonnet 3.7/4, Gemini 2.5 Pro, and GPT-5.

### Strengths
The paper presents a well-motivated and timely benchmark that addresses an important gap in evaluating LLM-based web agents on complex, tedious, and memory-intensive tasks. It demonstrates strong methodological rigor, with 532 carefully curated tasks covering diverse domains and requiring multi-step reasoning, long-term memory, and precise calculation. The benchmark’s full compatibility with WebArena ensures reproducibility and enables fair cross-agent comparison. The authors conduct comprehensive experiments using multiple state-of-the-art frameworks (AgentOccam and BrowserGym) and several leading LLMs (GPT-4o, GPT-5, Claude, Gemini), providing valuable insights into model weaknesses and task-specific performance trends. In addition, the clarity of writing, systematic task taxonomy, and inclusion of human baselines make the paper highly accessible and informative. Overall, WebChoreArena is a significant and practical contribution that is likely to become an essential resource for future research on web agents and long-horizon reasoning.

### Weaknesses
First, the study lacks algorithmic innovation, focusing solely on benchmark construction rather than proposing new agent mechanisms. Second, all experiments are confined to simulated environments, which limits the ecological validity and generalizability to real-world dynamic websites. Third, the model coverage is insufficient — only a few large proprietary models (e.g., GPT and Claude series) are tested, with no inclusion of diverse model sizes or more open-source models (such as LLaMA, Qwen, or Mistral families). Expanding the evaluation to a broader spectrum of model scales and architectures would make the benchmark more representative and strengthen its conclusions. Finally, the error analysis remains descriptive without suggesting concrete improvements, and the benchmark’s high manual construction cost may hinder scalability.

### Questions
1. Model Diversity:
The paper evaluates several leading models (GPT-4o, GPT-5, Claude, Gemini), but the coverage remains limited. Could the authors expand the experiments to include a wider variety of both open-source and proprietary models, such as LLaMA, Qwen, Mistral, or DeepSeek, to better understand performance trends across different architectures and scales?

2. Model Scaling Analysis:
Have the authors examined how model size or reasoning depth affects performance on different task types (e.g., Massive Memory vs. Calculation)? Including a scaling curve analysis could provide valuable insights into whether task success correlates with model capacity or training methodology.

3. Closed-Source Model Comparison:
Beyond GPT and Claude, are there plans to evaluate other strong closed-source models, such as Gemini Ultra, Claude Opus, or proprietary enterprise agents? This would help position WebChoreArena as a more comprehensive benchmark for the broader LLM ecosystem.

4. Open-Source Baseline Integration:
Would the authors consider including smaller open-source baselines (e.g., 7B–14B models) to establish a clearer performance hierarchy and make the benchmark more accessible for the academic community?

5. Generalization Beyond Simulation:
Since the current setup relies on simulated WebArena environments, do the authors plan to extend the benchmark to real-world dynamic websites and test whether the same models maintain consistent performance under non-deterministic conditions?

### Soundness
3

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
4

### Summary
Given the complexity limitations of current agent benchmarks, this work proposes a new WebChoreArena benchmark, that features more challenging tasks stressing massive memory, calculation, and long-term memory handling. Experimental results with various agents reveal directions for future agent development.

### Strengths
1. **Benchmarking Contribution to the Agent Community.**
> Regarding recent agent progress. As existing mainstream benchmarks are beginning to be solved, this work introduces more complex tasks, particularly targeting massive memory, calculation, and long-term memory handling scenarios, to facilitate the development of more capable agents.

2. **High-quality example curation process.**
> The tasks are manually collected by agent researchers. Measures of task properties are also reported.

3. **Decent benchmarking effort.**
> This paper benchmarks multiple major open-source agent frameworks, with a series of LM backbones. Further, multiple analyses were conducted to offer more insights.

### Weaknesses
1. **Lack of Fine-Grained Evaluation.**
> As the tasks become more complex (i.e., involve more checkpoints that agents need to achieve), a single end-task evaluation tells less information. It would be more informative if more fine-grained intermediate evaluations, especially targeting each atomic requirement in the task instructions, were included in the benchmark.

2. **Unclear in complexity?**
> Although tasks are claimed to be more challenging than the WebArena benchmark, the best agent (with GPT-5) already achieves ~50% success rate, indicating that the benchmark may be largely solved by the frequently-updated models soon (?). Further on this point, it is very likely that the two agent frameworks experimented in this work do not bring out the best performance of GPT-5 (as opposed to ChatGPT or other popular agent frameworks), therefore it is possible that some existing agents already can solve 60-70% of this benchmark.

### Questions
1. How realistic are the tasks in WebChoreArena, and what processes have been applied to ensure the realism of tasks? As opposed to overly driving up the task complexity, potentially to a point where humans wouldn’t even need to do such tasks. 
On the other hand, is the purpose of this benchmark to: (i) reflect how agents perform in practice, (ii) act as stress tests to agents, or others?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce WebChoreArena, a benchmark of 532 human curated web agent tasks designed to be more tedious and complex than WebArena tasks. These tasks feature three challenge types: massive memory, calculation, long term memory. The paper evaluated leading models including GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Pro with two agent frameworks of AgentOccam and BrowserGym.

### Strengths
The paper shows clear problem focus and task taxonomy: benchmark that targets tedious chores underrepresented in prior work.

The paper builds on top of existing, proven sandbox environments of WebArena, saving efforts for adaptation and potential environment pitfalls. 

Annotators followed explicit guidelines to curate tasks, with cross checking to minimize labeling, evaluation errors.

### Weaknesses
It would be helpful to see more analysis on agent’s use of calculation related tools in completing requirements, on top of the fact that models don’t always choose to use calculators. 

Has the author examined if models are given sufficient tools or agentic design that enables long term memory. An example could be a notebook page, or a function call that allows models to write/read/search from a notebook. It would be great to see some analysis on how well models utilize these functions to complete tasks that require longer term memory. 

It would be great to provide more details on how different modality is used for the agent’s action.

### Questions
Listed above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new benchmark called WebChoreArena, built on top of the WebArena simulation environments, WebChoreArena ensures strict reproducibility and enables fair, direct comparisons with the established WebArena benchmark, offering key insights into agent progress. This new benchmark comprises of 532 carefully curated tasks designed to extend the scope of WebArena beyond general browsing to more labor-intensive and tedious tasks. WebChoreArena systematically integrates three key challenges: 1) massive memory, 2) calculation, and 3) long-term memory. The experimental results demonstrate that as LLMs evolve, represented by GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro, significant improvements in performance are observed on WebChoreArena. These findings suggest that WebChoreArena is well-suited to measure the advancement of state-of-the-art LLMs with greater clarity. Nevertheless, the results also indicate that even with Gemini 2.5 Pro, there remains substantial room for improvement compared to WebArena, highlighting the increased challenges posed by WebChoreArena.

### Strengths
Benchmark is well established on an existing infrastructure by WebArena, and following suit the best practices also makes this benchmark very reproducible and realistic, including good outcome-based reward to prevent reward hacking.

Code repository and containers are well organized online and reproducible, documentation is good

The paper is clear and reasonable about why this benchmark is needed, on top of the existing WebArena benchmark. It's focus on chore tasks, with longer horizon but more repetitiveness is indeed valid concern and justifies the new benchmark to differentiate from existing. The large memory required is quite interesting indeed and can test the model's capability.

### Weaknesses
As a benchmark paper, it would always be nice to present a bigger leaderboard and allow people to submit results. It mainly uses AgentOccam and BrowserGym as the agent framework, with several strong base models, but it would be really adding strength with more. 

For Memory-intensive long horizon tasks, it would be nice to experiment with agent orchestration that has special focus on compression/condensation of observations, instead of more generic ones. One example could be OpenHands with has condenser feature.
More details about the data distribution and what is done to improve the data annotation matches what real world scenario presents is crucial. Given the environments are simulated, it's even more important to have realistic queries/tasks that closes the sim2real gap.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
