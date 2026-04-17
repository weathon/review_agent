# RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 2

## Abstract
Large language model alignment via reinforcement learning depends critically on reward function quality. However, generic reward models often underperform on heterogeneous task distributions due to distribution shifts, while training task‑specific reward models is costly and prone to annotation difficulty, catastrophic forgetting, and loss of generalization.
We present RLVR (Reinforcement Learning from Agent Rewards), a unified, agent‑driven framework that dynamically assigns tailored reward functions to individual training queries. RLVR combines two automated LLM‑based stages. First, the tool generation stage where web-agents and code-agents generate rule‑, metric‑, and model‑based reward functions and wrap them as a callable tool. Then, there is a reward tool calling stage where a central decision LLM assign the reward function tools to individual queries.
Across diverse tasks including translation, summarization, question answering, and mathematics, RLVR delivers $5$–$10$% average improvement over a widely‑used generic reward model (Skywork‑Reward‑V2) and matches GPT‑4.1‑as‑judge performance, while generalizing well to untrained benchmarks such as BenchMAX, AIME-2024 and Arena-Hard-v2.
Ablation studies show performance drops of $40$%, $77$%, and $198$% when removing the web‑agent, code‑agent, and selection backbone, with the backbone achieving $86.50$% selection accuracy near the theoretical ceiling of top reward models. The retrieval module locates optimal tools reliably, with an average first‑page rank of $5.64$.
By systematically leveraging and extending existing reward sources, RLVR offers a scalable path to high‑quality RL alignment over heterogeneous task domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes RLAR, an agent-driven reward design framework for multi-task RL post-training of LLMs. A code-agent synthesizes rule/metric reward functions, and a web-agent retrieves and wraps task-specific reward models; a decision LLM then selects an appropriate reward tool per query. Experiments on mixed tasks (translation, summarization, QA, math, multi-turn, conditional generation) report 5–10% average gains over a single generic reward model and competitive results versus LLM-as-judge, with better efficiency. The work argues dynamic, task-aligned reward construction improves robustness and cost-effectiveness for heterogeneous training.

### Strengths
1. Idea: Clear, practical framing of “tool-ized” reward design that orchestrates rule/metric tools with retrieved reward models, addressing known issues of distribution shift and monolithic reward models in heterogeneous training.

2. Efficiency: Reports substantially lower time and token cost than purely generative RMs used as judges, which is valuable for scaling RL post-training.

3. Breadth: Evaluates across several task categories and includes some out-of-domain generalization checks (e.g., BENCHMAX, MT-Bench).

### Weaknesses
1. Benchmark currency and coverage: MT-Bench is aging; more recent reasoning/agentic evals like alpaeval2 or Arena-Hard2 are omitted, and broader widely-used public leaderboards for instruction-following/reasoning are missing, making claims on generalization and reasoning less persuasive.

2. Baseline sufficiency on verifiable tasks: For math and other verifiable domains, recent RLVR-style baselines and strong verifiable-reward pipelines are not included; without these, it is hard to isolate the advantage of the agentic reward selection vs. known robust verifiable rewards.

3. Model scale vs. compute budget: Experiments are on very small base models, yet the compute setting (e.g., 8×H100) could handle at least 7B with standard memory optimizations (LoRA/QLoRA, paged optimizers). This gap weakens external validity and the strength of the conclusions.

4. Limited ablations: The paper lacks deeper ablations on the agent components and selection policy (e.g., turning off web-agent, varying code-agent rule coverage, or analyzing decision LLM selection accuracy vs. oracle). This limits clarity on where gains truly come from.

5. Result magnitude vs. recent literature: The improvements—while consistent—are modest and compare unfavorably to recent reports that small LLM-judgers can rival or surpass GPT‑4-level reward/judge behavior in some settings; the paper does not position results against that line of work, and the "ground true" judger ability of GPT4.1 is not conving me.

### Questions
1. Agent usage and ablations: Which agent contributes most to gains in each task family (translation, math, multi-turn)? Please provide ablations isolating code-agent only, web-agent only, and both, plus a “no decision LLM” variant (e.g., rule-based selection) to quantify each component’s contribution.

2. Verifiable tasks: For math and other verifiable domains, can you add stronger RLVR-style baselines and established verifiable pipelines, and report pass@k, exact numeric match, and failure modes?

I would be happy to discuss with the author in the RB stage.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RLVR (Reinforcement Learning from Agent Rewards), an agent‑driven framework that improves LLM alignment by dynamically assigning query‑specific reward functions. RLVR uses two automated stages: 1. Tool Generation – Web and code agents create rule‑, metric‑, and model‑based reward tools; 2. Reward Assignment – A decision LLM selects the best reward tool for each query.
Across tasks like translation, summarization, QA, and math, RLVR achieves 5–10% performance gains over a leading generic reward model, matches GPT‑4.1‑as‑judge quality, and generalizes to unseen benchmarks. Code agents successfully produce executable tools in 94.9% of cases, and web agents integrate 47.6% of retrieved resources, enabling scalable and effective reward construction.

### Strengths
1. Analysis shows RLAR consistently generates and deploys high‑quality, task‑aligned rewards from diverse sources, with code agents creating executable tools in 94.9% of cases and web agents integrating 47.6% of retrieved repositories.

2. During training, RLAR adaptively selects from a portfolio of evaluators, with LLM‑based reward models used in 96.4% of cases and task‑specific rule‑based checks applied when appropriate. This matching of evaluators to domain characteristics produces smoother, more informative advantage estimates and stronger policy updates compared to using a single generic reward.

3. Results demonstrate RLAR’s ability to fuse diverse reward sources into a coherent and effective signal for LLM reinforcement learning.

### Weaknesses
1. The paper could benefit from a more detailed literature review and explicit comparisons with other dynamic reward selection or adaptive RL alignment methods to position RLVR/RLAR within existing research.
2. The criteria for reward tools are unclear to me. The paper should explain the evaluation metrics or qualitative measures used to judge tool quality, possibly including human or benchmark‑based assessments, error analysis, and robustness checks.

### Questions
See the weaknesses.

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
3

### Summary
This paper introduces RLAR (Reinforcement Learning from Agent Rewards), an agent-driven framework designed to address the limitations of generic reward models in multi-task LLM alignment. The core idea is to dynamically assign tailored reward functions to individual training queries, improving performance on heterogeneous task distributions.

### Strengths
- The paper is well-organized and clearly articulates the problem of reward model generalization.
- It proposes a novel and promising conceptual solution to a significant bottleneck in RL alignment: the high cost and catastrophic forgetting associated with training numerous task-specific reward models.
- The concept of an "agentic" system that dynamically selects reward functions is nontrivial and makes novel contribution.
- The engineering part may also contributes to the community.

### Weaknesses
+ The experiments are limited to small-scale models. While this serves as a good proof-of-concept, the paper would be much stronger if the findings were validated on larger models.
+ Also, to demonstrate true scalability and practical applicability, current tasks and datasets need to be expanded.
+ Some analysis about the effiency would be appreciated.

### Questions
Since the authors are using H100 and A100, 0.6B and 1B models are a bit too small. Why not choose a larger model like 7B or 3B?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a generic tool composed of code and web agents, which jointly produce a task-based, callable reward signal for LLM training. The web agent crawls reward repositories (e.g., from Huggingface, Modelscope, or GitHub) based on semantic similarity between the task and a repository's README file. Concurrently, a code agent produces a rule-based or metric-based reward signal. An LLM then combines or selects the tools generated from these two sources to produce the final reward signal for a given query. The proposed software is shown to select the correct reward tools for a limited number of tested tasks, including summarization, translation, and RLHF, among others.

### Strengths
- A tool that automates the provisioning of reward signals for any specified task could significantly accelerate research by abstracting away the challenging reward-design component of LLM training. Furthermore, such a tool could serve as a standard benchmark, allowing various approaches to be meaningfully compared using a common reward signal.
- The idea of using both web and code agents to generate the reward signal is compelling, as it offers the potential to cover a broad range of tasks by having one agent generate new signals (code) while the other re-purposes existing ones (web).

### Weaknesses
- The paper's overall quality and presentation are subpar. It suffers from numerous typos and incomplete sentences. Furthermore, key sections are underdeveloped (e.g., code agents), while critical details about core mechanics are omitted (e.g., the web agent's ranking, filtering, and semantic similarity computation). These omissions create significant gaps in understanding. Finally, the limitations section exceeds the page limit.
- The empirical evaluation is narrow and insufficient. Given the ambitious goal of a "universal reward signal," a far more rigorous and thorough evaluation is expected, covering a much wider variety of tasks and problem types than the limited set presented.
- A critical omission is the lack of validation studies comparing the quality of the generated reward models against heuristic, human-designed rules. The approach delegates major components of its process to LLMs, yet these black-box components are not thoroughly investigated or verified, either through human studies or isolated component analysis.

### Questions
- The paper fails to address the web agent's potential failure modes. It should clarify what fallback mechanism is in place if a specified task has no corresponding reward repository on the crawled websites. This is a crucial omission for a tool presented as 'generic'. The same applies to the code agents.
- The paper omits any discussion of the significant legal and ethical considerations of this approach. The authors must explain what mechanisms, if any, are in place to ensure the agents respect repository licensing and terms of use. Without this, the tool's practical and responsible deployment is questionable.

### Soundness
2

### Presentation
1

### Contribution
2
