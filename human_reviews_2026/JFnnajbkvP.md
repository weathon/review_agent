# EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
A fundamental limitation of current AI agents is their inability to learn complex skills on the fly at test time, often behaving like “clever but clueless interns” in novel environments. This severely limits their practical utility. To systematically measure and drive progress on this challenge, we first introduce the Jericho Test-Time Learning (J-TTL) benchmark. J-TTL is a new evaluation setup where an agent must play the same game for several consecutive episodes, attempting to improve its performance from one episode to the next. On J-TTL, we find that existing adaptation methods like reflection, memory, or reinforcement learning struggle. To address the challenges posed by our benchmark, we present EvoTest, an evolutionary test-time learning framework that improves an agent without any fine-tuning or gradients—by evolving the entire agentic system after every episode. EvoTest has two roles: the Actor Agent, which plays the game, and the Evolver Agent, which analyzes the episode transcript to propose a revised configuration for the next run. This configuration rewrites the prompt, updates memory by logging effective state–action choices, tunes hyperparameters, and learns the tool-use routines. On our J-TTL benchmark, EvoTest consistently increases performance, outperforming not only reflection and memory-only baselines but also more complex online fine-tuning methods. Notably, our method is the only one capable of winning two games (Detective and Library), while all baselines fail to win any.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work introduces a benchmark and a framework focusing on test-time learning. The new benchmark is created from an existing set of text-based games, where the capacity of improvement during learning is considered. The proposed framework, EvoTest, consists of an actor and an evolver that changes the actor's learning configuration by altering the policy prompt, memory, hyperparameters, and available tools.

### Strengths
A. The work investigates an important problem of decision-making, which is online learning for fast adaptation.

B. The content of the submission is clearly presented, and the reading flow is good.

### Weaknesses
I. The work contains some hidden assumptions and unjustified claims:

   a. The proposed framework relies on the assumption that the LLM backbone contains sufficient information for solving the task of interest. While this assumption is realistic, it is never clearly stated.

   b. The comparison with fine-tuning methods is incomplete as the performance of EvoTest with qwen/qwen3-32b for the actor and the evolver is never reported. This is the only way to make a fair comparison.

   c. The work is missing a limitation section.

   d. There seems to be a hidden assumption on credit assignment because in Line 258, it is stated: "It records state-action pairs ($o_t, a_t$) that immediately preceded a score increase in a 'success' table." This means the framework assumes the credit should be given to the immediate state-action pair, which is not true in the general case. Discussing this aspect would be useful. 

   e. Several claims are made about the possible behavior of the evolver: "[the evolver] can learn to increase exploration
temperature in early episodes and simultaneously add a new strategic heuristic to its prompt based
on its findings, a multi-faceted adaptation that other methods are incapable of." Sharing examples of this behavior would strengthen this claim. For example, plotting the exploration temperature during learning would be interesting.

   f. In Section 5.1, no explanation is provided to justify the choice of the LLM backbone.


II. More precision is required to assess the work properly:

   a. A more precise description of the Evolver action space would help clarify the contribution. Especially, the explanation of the deployment-time memory and tool-use routines, which remain high-level without disclosing the details.

   b. Section 4.3 about the selection of configuration using UCB is unclear. Indeed, it is not clear if the agent interacts with the envorinment in a seperate training phase, in which case the additional interactions with the environment should be taken into account, or if a different configuration can be attempted at each step during learning, this would raise a problem of credit assignment because it would not be clear which configuration led to a fruitful outcome. 


III. The presentation can be improved:

   a. The appendix is rich, but most of its content is not referenced in the main text.

   b. A bigger font size can be used for all figures. Additionally, a shared legend can be used for Figure 2 to avoid repetition.

   c. Line 329, "claude" at the end of the line can be removed.

### Questions
N/A

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
The paper introduces EvoTest, an evolutionary framework designed to solve the problem of AI agents being unable to learn on the fly once deployed. EvoTest uses a gradient-free, two-agent system: an Actor agent attempts a task, and an Evolver agent analyzes its transcript. Based on this analysis, the Evolver evolves the entire agentic system for the next episode by:
* rewriting its prompt, 
* updating its memory, 
* tuning hyperparameters, 
* refining its tool-use routines. 

On the newly introduced J-TTL benchmark, EvoTest demonstrated superior performance, significantly outperforming reflection-based and memory-only baselines and proving capable of solving tasks that all other methods failed.

### Strengths
This paper addresses a novel problem. With the emergence of LLM as agents, new problems appear, that include allowing such agents to quickly adapt to tasks unseen during training. 

Solving a problem is a core task that many industrial actors are currently trying to tackle.

The authors insist on their gradient free approach, which indeed does not require retraining/finetuning of the weights.

The benchmark is well motivated and established.

The result appear strong, and the ablation study provides intuition to the reader on the relevance of the difference components, even if second order results would allow to distinguish the true performance gaps of the compared methods.

### Weaknesses
My main concern is the relevance of the paper for the readers of ICLR. While clearly framed as a scientific paper (with e.g. the outlined list of contributions, of research questions), the paper still feels like an engineering paper in the sense that it primarily aims at pushing up a metric. A clear description of the scientific problem in the introduction could help change this impression, given to the reader. I am not sure how learning representations are here (even implicitly) discussed.

My second concern is the lack of introductory examples (can be done in one example) of:
1. The addressed research problem, show how current agent fail at addressing adapting to one of the existing games.
2. the task given to the agent. It would greatly help understanding if the (current) Figure 1 could incorporate an example of the tasks provided in the Benchmark. Even if these have been introduced in an existing paper, it would make this work more comprehensible on its own.

This can be done by referencing content placed in the appendix, if space is limited, but is important to help the reader understand the problem, and assert that it is relevant.

No limitations discussion. I am quite sure that a discussion could be made on e.g. the drawback of not letting the agent learn (in the primal sens of modifying its neural weights).

Lack of second order reported metrics (mean +/- std)

### Questions
The perfect writing quality of this paper leads me to not having any question, everything is clearly defined.

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
The paper presents Jericho Test-Time Learning (J-TTL), a few-shot benchmark based on Jericho, and EvoTest, an LLM-based test-time learning framework for J-TTL.

EvoTest comprises of two LLM agents, an actor and an evolver. An actor agent interacts with the Jericho environment given its configuration. An evolver agent proposes new configurations based on the interaction history. An actor agent's configuration includes prompts, interaction memories, LLM inference hyperparameters, and game-specific tools.

The evaluation results on J-TTL demonstrate that EvoTest improves over the non-learning baseline and other self-improvement baselines.

### Strengths
- The paper reads well with a clear logic flow. The concept of few-shot learning and test-time learning is not new, but the proposed LLM-based method in a text-based environment is novel to my knowledge.
- The experiments are thorough with a good variety of baselines and key ablations. Empirically, EvoTest seems to be meaningfully better than previous methods.
- Code and prompts with examples and interaction logs are provided for reproducibility.

### Weaknesses
My main concern is the potentially limited scope. The evaluations are limited to a subset of Jericho games. This raises questions of 1) whether the set of games is cherry-picked, and 2) whether the method is over-engineered to work specifically on Jericho games. Despite that EvoTest is empirically superior to previous baselines, it is compared with methods with a broader scope.

Branching off of this, from the ablation (Table 3), it seems that prompt optimization and the evolution selection strategy (UCB) are the two major factors for the improvement. I would like to see how much the specific prompt-evolving prompt from the evolver contributes to this. This should be verifiable by transplanting the prompt-evolving part in the evolver master prompt to other prompt optimization methods, such as EvoPrompt.

### Questions
1. In Section 4.3, do you discard all the options that were not selected? Wouldn't that make $\chi^{(e)}$ the only candidate with $n(\chi)>1$? Or will the evolver agent generate repeated child configurations?
2. How is the initial configuration determined? Do you use a generic one for all games or do you use specific ones for different games? Does the initial configuration affect the final performance?

Minor issues / Typos

- Eq 5, fix the argmax and define $n$ and $N$.
- In multiple places, the quotation marks aren't correctly used (``` `` ``` and `''`)

### Soundness
3

### Presentation
3

### Contribution
3
