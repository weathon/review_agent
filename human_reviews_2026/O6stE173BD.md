# BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Large language models (LLMs) have shown strong reasoning and coding capabilities, yet they struggle to generalize to real-world software engineering (SWE) problems that are long-horizon and out-of-distribution. Existing systems often rely on a single agent to handle the entire workflow—interpreting issues, navigating large codebases, and implementing fixes—within one reasoning chain. Such monolithic designs force the model to retain irrelevant context, leading to spurious correlations and poor generalization. Motivated by how human engineers decompose complex problems, we propose structuring SWE agents as orchestrators coordinating specialized sub-agents for sub-tasks such as localization, editing, and validation. The challenge lies in discovering effective hierarchies automatically: as the number of sub-agents grows, the search space becomes combinatorial, and it is difficult to attribute credit to individual sub-agents within a team. We address these challenges by formulating hierarchy discovery as a multi-armed bandit (MAB) problem, where each arm represents a candidate sub-agent and the reward measures its helpfulness when collaborating with others. This framework, termed Bandit Optimization for Agent Design (BOAD), enables efficient exploration of sub-agent designs under limited evaluation budgets. On SWE-bench-Verified, BOAD outperforms single-agent and manually designed multi-agent systems. On SWE-bench-Live, featuring more recent and out-of-distribution issues, our 36B system ranks second on the leaderboard at the time of evaluation, surpassing larger models such as GPT-4 and Claude. These results demonstrate that automatically discovered hierarchical multi-agent systems significantly improve generalization on challenging long-horizon SWE tasks. Code is available at https://github.com/iamxjy/BOAD-SWE-Agent.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the problem of automated development of SWE-Agents. They consider a multi-agent setup structured such that an orchestrator agent calls multiple sub-agents. The main concern of the paper is optimizing the start-up prompts for the orchestrator and subagents. They treat this as a bandit problem, maintaining and growing a set of sub-agents over time. Each sub-agent is scored using a with an LLM credit assignment approach. The paper's evaluations show strong results on SWE-Bench Verified and Live.

### Strengths
The major strength of the work is the novel formulation of the problem as an MAB problem and the demonstration different aspects of the  formulation leads to improvement of performance.

1. The framing of optimising multi-agent systems as multi-armed bandits is novel in the context of SWE-Agents. The work successfully formulates a UCB strategy for this context as well as a way to grow the number of subagents. Ablations demonstrate how optimising sub-agents explicitly outperforms static approach.

2. The use of LLM based credit assignment technique in the multi-armed bandit setup is novel and very interesting here. It shows a way to learn from specific mistakes the system might be making in a modular way. Ablations also support the importance of this metric.

3. Evaluation show improvements on standard benchmarks, especially on SWE-Bench Live.

### Weaknesses
1. The major limitation of this work is that it is limited to only improving the starting prompt of the system. From the difference in performance of scaffolds present on https://www.swebench.com/, it is reasonable to believe that design decisions can impact the behavior of agent, as much or maybe more than prompts.

2. The analysis of the behaviour of the system is limited to short qualitative analysis. This is currently not substantiated with example or quantitative metrics. There is also no exposition of what sorts of agents are "discovered".

### Questions
1. Related to W1, can this idea be extended to agent design as well as prompt content? 

2. Related to W2:
a) Is there quantitative evidence for multi-agent systems being better in the sorts of ways described in 5.4
b) What sorts of agents are "discovered" and is there a way to characterise the trajectory of the system as is discovers these?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper searches for prompts of sub-agents for software engineering. The resulting agent consists of a set of sub-agents and an orchestrator (all parameterized by prompts). During inference, the agent will ask the orchestrator to call sub-agents sequentially to solve a problem. To search for the set of sub-agents, it maintains a pool of candidate sub-agents, picks and evaluates a subset of candidates each time, and updates the sub-agents' information according to the evaluation results afterward. The paper uses UCB to balance exploration and exploitation during search. It can also add a new sub-agent each round to improve diversity. LLM-judge is used to assign credits to each sub-agent as well. The searched agent is evaluated on two popular benchmarks and demonstrated strong performance.

### Strengths
The method is intuitive and interesting, although it largely relies on the abilities of LLMs for judging and proposing sub-agents. We do need various ways to balance exploration and exploitation. The searched agent demonstrates strong performance on popular benchmarks as well. 

The paper is generally well-written and easy to understand.

### Weaknesses
* Missing naive baseline, such as evolution search. One can treat all prompts of sub-agents and/or orchestrators as parameters and use LLMs + evolution search to optimize them. There are various prior works that balance exploration and exploitation for naive LLM tree search as well. The authors discussed this baseline in the method section, claiming it is prohibitively expensive with no experimental results. 
* I'm not sure if the comparison with baselines is fair, missing experimental details. It would be great to include the costs of calling Claude-4 (which model?) for each method. I am not sure if the method is stochastic or deterministic either. If not deterministic (hard to imagine LLM to be deterministic even with temperature=0), how noisy is the evaluation? 
* The meta-prompts (e.g., for proposing and refining sub-agents) look long and detailed. Can the method discover novel sub-agents that are not expected, given the meta-prompts? 
* It would be great to include the searched prompts for various methods as well.

### Questions
* Are there results of the naive evolution search?
* Can you provide more experimental details such as the variance of evaluation, the cost for each method, and meta-prompts for each method? 
* Can you share the final best-performance prompts of the searched sub-agents?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes BOAD, a method for discovering and selecting subagents to use as tools for an orchestrator agent in software engineering tasks. It considers subagents in a set as arms of a multi-armed bandit and assign credits with both the test-based final outcome reward and LLM-as-a-judge process reward. After running this bandit optimization on 12 issues from SWE-Bench Verified, the discovered multi-agent system can perform well on both SWE-Bench Verified, and SWE-Bench Live, which doe not have repository-level overlap with the problems in SWE-Bench Verified.

### Strengths
1. Interesting formulation of the multi-agent system discovery problem as a multi-armed bandit. This creates balance between exploration and exploitation while selecting and evolving subagents.

2. Great performance on SWE-Bench Live demonstrates the effectiveness of the method.

3. Comprehensive ablation showing the effectiveness of having the subagents, customizing the orchestrator, and using hindsight helpfulness for credit assignment.

### Weaknesses
1. Lack of details about the actual optimization process. How many agents are there in the final set of subagents? How many of the top agents are from the expanded set or the initial set of subagents? What are the final top agents selected? Qualitatively why are they better than other subagents? Readers need these details to get a better idea of the final discovered system.

2. Single run evaluation is insufficient. The non-determinism of LLM agents results on lots of randomness in every agent run, which impacts the optimization process. Is BOAD always effective as reported or will its performance fluctuate across multiple runs?

### Questions
1. Any plans to compare against manual optimization conducted by human engineers in the loop?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Previous work in agentic frameworks for code generation have shown a worrying trend -- performance drops significantly from in-distribution benchmarks (SWE-Bench Verified) to out-of-distribution benchmarks (SWE-Bench-Live). The authors posit that one of the root causes for this is the long-horizon and complex nature of SWE-tasks, which makes it hard for single-agent frameworks to delegate and solve tasks. Furthermore, current multi-agent systems rely on expensive evolutionary optimization algorithms, which is impractical in the current domain. 

The authors hypothesize that a multi-armed bandits (MAB) presents an efficient framework to build and optimize such multi-agent systems to solve such tasks. Specifically, they propose BOAD, which treats hierarchical multi-agent systems as a sequential decision-making process with the reward signal for evaluating each sub-agent (each arm in the MAB) coming from an LLM-as-a-judge.

BOAD achieves impressive performance on SWE-Bench Verified and SWE-Bench-Live.

### Strengths
- The paper is very well written and I thank the authors for explicitly stating the MAB formulation in the context of the code generation task.
- In terms of novelty, I was pretty surprised but a multi-arm bandit optimizer hasn't been tried before for software engineering tasks.

### Weaknesses
**Methodology:**
* **Training using 12 problems**: My understanding is that the SOTA accuracy number was reached after including 12 problems from SWE-Bench-Verified in the design set for BOAD. This raises two concerns:
	* Why 12 problems specifically? Does increasing or decreasing the set of design problems drastically effect final performance?
	* None of the other baselines are automatically tuning the multi-agent system. I understand that an evolutionary agent here might be less efficient than BOAD, but including a result for how BOAD fairs against an evolutionary agent would undoubtedly present a useful point of comparison to understand the tradeoff.

* **Efficiency Experiment:** One of the motivating points of using BOAD (from the introduction) is that it is hypothesized to be more efficient than evolutionary multi-agent frameworks at the same task. However, the Token Analysis experiment is not enough to justify the efficiency for two reasons:
	* There isn't a direct comparison against an evolutionary multi-agent framework.
	* Efficiency can be achieved by either reducing token usage OR by reducing the underlying model size:
		* Specifically, In terms of floating point operations, querying a smaller model more times is more efficient than querying a larger model less times.
		* For example (be advised: this is a rough analysis without access to underlying data):
			* Assuming equivalence in all other aspects, a `30B` model can generate `20%` more tokens than a similar `36B` model yet have the same total cost.
			* Under this logic, the OpenHands single-agent that uses `Qwen3-coder-30B` and achieves a 51.6% resolution rate for SWE-bench Verified in is actually *more efficient*  than SWE-Agent+BOAD using the `OSS-36B` model which achieves a 53.2% resolution rate.
		* My recommendation:
			* Getting access to TFLOP data is extremely hard. Instead, most works use normalized cost, which is defined as the fractional cost of querying a smaller model per token compared to the cost of querying the largest model. e.g. if cost for 36B model is `1` per token, then cost for a 30B model will be `0.8334`.
			* Then, such methods will show an efficiency-performance tradeoff curve.
			* The key insight is to demonstrate that the efficient-agent algorithm is at the pareto froont of the tradeoff curve for different configurations.
			* Look at Figure 3 of this paper [https://arxiv.org/abs/2504.07247](https://arxiv.org/abs/2504.07247) for an example of how this is computed and analyzed. This might be a good paper for the related works as well.


**Overall:** I'm slightly leaning towards rejecting the paper. The results are pretty impressive and I generally haven't seen a similar mutli-agent system optimized with bandits before, but there are some issues with the experimental setup that need to be better understood before proceeding. I'm happy to discuss these points with the authors during the discussion period.

### Questions
Look at weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
