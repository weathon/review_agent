# InnovatorBench: Evaluating Agents’ Ability to Conduct Innovative AI Research

- Decision: Accept (Poster)
- Scores: 2, 8, 6

## Abstract
AI agents could accelerate scientific discovery by automating hypothesis formation, experiment design, coding, execution, and analysis, yet existing benchmarks probe narrow skills in simplified settings. To address this gap, we introduce InnovatorBench, a benchmark-platform pair for realistic, end-to-end assessment of agents performing Large Language Model (LLM) research. It comprises 20 tasks spanning Data Construction, Filtering, Augmentation, Loss Design, Reward Design, and Scaffold Construction, which require runnable artifacts and assessment of correctness, performance, output quality, and uncertainty. To support agent operation, we develop ResearchGym, a research environment offering rich action spaces, distributed and long-horizon execution, asynchronous monitoring, and snapshot saving. We also implement a lightweight agent that couples explicit reasoning with executable planning using frontier models such as Claude-4, GPT-5, GLM-4.5, and Kimi-K2. Our experiments demonstrate that while frontier models show promise in code-driven research tasks, they struggle with fragile algorithm-related tasks and long-horizon decision making, such as impatience, poor resource management, and overreliance on template-based reasoning. Furthermore, agents require over 11 hours to achieve their best performance on InnovatorBench, underscoring the benchmark's difficulty and showing the potential of InnovatorBench to be the next generation of code-based research benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to benchmark the capabilities of LLM agents in AI research. First, this paper proposes InnovatorBench, which evaluates the LLM agents in data construction, filtering, augmentation, loss design, reward design , and scaffold construction. Second, this paper proposes ResearchGym, a sandbox for LLM agents to operate in the computers, providing several types of action interfaces. Empirical analyses on InnovatorBench are provided to demonstrate the performance of current LLMs and LLM agents.

### Strengths
- Comparison on several LLMs are provided.
- The writing is of good quality.

### Weaknesses
- What does the task in InnovatorBench look like? This paper only provides an example of Task 14, with other types of tasks unrevealed.
- The number of the tasks in InnovatorBench is too limited, with only 20 tasks in total for 6 types of different tasks. The empirical findings from such a tiny benchmark can be unreliable. This also makes the reusability of the proposed InnovatorBench limited.
- The experiments on InnovatorBench require computing machines with 8x80 GB GPUs, which are too costly for research use.
- Lack of empirical results of open-sourced light-weight LLMs, such as Qwen3-32B.
- Typos: some references are cited in wrong formats.

### Questions
I think I would not change my opinion during discussion phase; so there are no questions.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors present a new benchmark that tests research ability on more complex tasks, which is backed by a new evaluation infrastructure harness that allows for more complex tasks. The work includes evaluations and failure analysis of leading frontier models with a simple agent.

### Strengths
* The tasks provided are difficult and realistic, which complements or advances existing benchmarks.
* Research gym genuinely seems very useful to move to a more realistic action space (for e.g. with snapshotting and multi-node control). This allows evaluations to test tasks that are substantially more complex. 
* The failure analysis was informative and interesting (notably, I wonder if non-Claude agents performed worse as they were only trained to use specific tools like bash or python).

### Weaknesses
* While the results are presented reasonably well, it could be clearer. For example, the fact that the tasks were validated by annotators reproducing the reference score in the reasoning gym version was only present in the Appendix but it was a key question about how valid/reviewed are the tasks.
* It might have been useful to test other scaffolds to ensure that the agents were not substantially worse than they could have otherwise been (e.g. openhands or codex).
- One concern is that all performance on this benchmark will be driven by memorization, since all the tasks are based on existing papers that are in training data. It would be more convincing if the authors presented evidence that this was not a concern (i.e. by looking at successes).

### Questions
* are agents allowed to use the snapshot functionality via tool calls?
* will you release the tasks publicly?

nits:
- missing parenthesis line 235.
- missing word line 265

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a benchmark of 20 LLM engineering tasks that span the lifecycle of LLM development. Also, a task execution environment is introduced as the framework for running these tasks. In addition, the performance of four models is assessed on these tasks using the ReAct agent, and it is shown that Claude 4 Sonnet achieves the best overall performance and in many of the subtasks, largely from its superior tool use. GPT-5 achieves the best performance in implementing agent scaffoldings, this seems largely due to designing agent scaffoldings that are more resilient to models' failure to use tools.

### Strengths
Originality:
- This paper follows the approach of many others in scraping high-quality conference papers for tasks to re-implement.
- The implementation of several agent tools, such as asynchronous command execution, is a simple yet powerful improvement that overcomes one of the most common failure modes I've seen in agent scaffoldings.

Quality:
- Perhaps my biggest source of uncertainty in this paper is the decision to implement an entirely new task execution framework from scratch. This is not an easy problem, and I would feel much more reassured if it had built upon existing battle-tested frameworks.
- There are no confidence intervals on any of the metrics presented, which makes me think that perhaps these are all based on single rollouts, which would really undermine my confidence in the results.
- This paper claims to be unique in that it tests models' creativity, but it follows a similar approach to many other papers in scraping existing conference papers for tasks to implement. It also provides starter code based on those repos. So I'm not sure what is fundamentally unique about this task that allows it to test creativity in ways that others don't.
- Developing 20 new research engineering tasks is a commendable accomplishment. I do have some concern about overlap with tasks in existing benchmarks, given that it also scrapes from conference papers.
- The comparison table also lists multi-GPU, multi-node, and save-and-restore as important differentiators of InnovatorBench, and yet no ablations are performed to show that these features substantially make a difference to model performance. So that's the understanding of the data.
- Similarly, I would appreciate an ablation that shows how important it is to give the agent asynchronous command execution ability, and whether removing that substantially decreased model performance. Because if that doesn't make that big of a difference, it really undermines the motivation for this new, separate framework (and, in any case, it probably could have easily been added to, e.g., https://inspect.aisi.org.uk/)
- Don't get me wrong, it's entirely possible that this was very high-quality evals engineering. It just seems to me to be an instance of undifferentiated heavy lifting.

Clarity:
- The writing is clear and concise, if a bit grandiose.
- In the comparison of different AI benchmarks, it’s shown that InnovatorBench has a “max eval times” of four. I don’t know what “max eval times” means, and it’s not mentioned anywhere else in the paper.

Significance: 
-  My guess is that this represents at best a small improvement in the difficulty of research engineering tasks, as demonstrated by the test-time scaling experiment. Though the decision to implement an entirely new framework undermines the community's ability to build on it.

### Weaknesses
See strengths section, listed together

### Questions
1. Using a simple React agent loop, how did you prevent the agent from submitting early, which is a common failure mode with LLMs?
1. What does it mean to have eval times of 4?
1. You list the time horizon as 2 to 36 hours. What do you mean by that exactly? Is that the amount of time it takes a human to complete the tasks? And do you have any human baseline data?
1. You list multi-node as a key feature of the framework, but it looks like only 1 task makes use of more than one 8× H100 machine. Do you stand by this as an important feature? Or could you remove that task without significant decrease in quality?
1. How many agent runs were performed for each task and model? Do you have any summary statistics? Are the differences between the models statistically significant?

### Soundness
3

### Presentation
2

### Contribution
3
