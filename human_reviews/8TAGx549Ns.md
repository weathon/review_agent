# REX: Rapid Exploration and eXploitation for AI agents

- Avg Score: 4.00
- Decision: Reject
- Scores: 5, 3, 5, 3

## Abstract
AI agents leveraging the capabilities of Large Language Models (LLMs) and Reinforcement Learning (RL) techniques have garnered growing attention due to their commendable performance in autonomously executing real-world tasks. Effective exploration of the action space is paramount for the successful accomplishment of diverse tasks by these AI agents. In this paper, we propose an enhanced approach for $\textbf{R}$apid $\textbf{E}$xploration and e$\textbf{X}$ploitation of action space for LLM-based AI agents, called $\textbf{REX}$. Existing LLM-driven agents have inherent limitations, such as a heavy reliance on precise descriptions for decision-making, and the lack of a systematic approach to leverage try-and-fail procedures akin to traditional RL. REX introduces an additional layer of rewards and integrates concepts similar to Upper Confidence Bound (UCB) scores, leading to more robust and efficient AI agent performance. This approach has the advantage of enabling the utilization of offline behaviors from logs and allowing seamless integration with existing foundation models while it does not require any model fine-tuning. Through comparative analysis with existing methods such as Chain-of-Thought(CoT) and Reflexion, REX-based methods demonstrate comparable performance and, in certain cases, even surpass the results achieved by these existing techniques. Notably, REX-based methods exhibit remarkable reductions in execution time while systematically exploring the action space of AI agents, enhancing their practical applicability across a diverse set of scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper motivates their method by identifying current ICL-agents lack: 1) systematic rewards incorporation, 2) Exploration-Exploitation trade-off, 3) Accuracy and latency. They introduced a method called Rapid Exploration and eXploitation for AI agents (REX) to address the aforementioned issues.

In particular, REX uses MCTS as the base search algorithm to

### Strengths
- The paper explores a very relevant topic that is less looked at, which is the exploration problem in reasoning tasks.
- The paper is easy to read, explains their method well
- The motivation of exploration and exploitation is clear and is a topic that needs attention in LLM-based reasoning/agents

### Weaknesses
- The evaluation is somewhat lacking, only evaluated on blocksworld and GSM8K, and ablations were missing (only statistics of # of actions were given), no intuition on how to tune the hyperparameters B,K,C
- Baselines is missing, see question 1
- some implementation details are missing, see question 2
- needs more intuition/analytics for how REX works, which one is better in what situation, when to use it, see question 3 & 4

### Questions
How come RAP scores are missing for every evaluated environment? It is mentioned RAP is not compatible with OAI APIs, maybe supply more results with open-source models?

how was each action generated? If im not mistaken they are each reasoning steps. And if temperature is 0, how come there are variations to the steps generated when prompted twice? is it caused by using previous history of steps in the UCB prompt (State-Action-Feedback in figure 4)?

Do you have some intuition on when is REX-UCL better than REX-UCB and vice-versa? (Paper only included REX-UCL is better at exploration than REX-R)

How come when more steps (6) is required, method is worse than Reflexion? Shouldn't it be the opposite, since REX employs search trees. Do you have any intuition?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses the problem of improving LLM-based reasoning techniques through algorithms that wrap the core LLM inferences. To this end, the paper investigates ways to use Monte Carlo Tree Search with LLM reasoning, by tracking the success of different reasoning traces and using this to guide subsequent reasoning attempts. Two approaches are proposed:
1) Planning full trajectories and using upper confidence bound (UCB) scores to guide the selection of actions from the sequence
	- Uses in-context learning as the mechanism to update model action reward estimates
	- Feedback is provided to the LLM by labeling prompt actions as high or low reward
2) Planning at the level of individual actions and using UCB to reweight predicted token logits for different actions

Three algorithms are proposed (a variant of the first approach uses only raw rewards, rather than UCB scores) and evaluated on a block arrangement task and grade school mathematics problems. The proposed algorithms are compared with two baselines from the literature and found to have higher accuracy in some tasks. The new algorithm is shown to have favorable time complexity compared to three baselines.

### Strengths
# originality
Combining search techniques with LLMs is an emerging area. MCTS is only recently being studied in this context, making the work original in that respect.

# quality
The proposed algorithms are evaluated on two different domains with varying size and reasoning requirements. Both accuracy and time complexity are compared.

# clarity
The figures and algorithm listings help clarify the core algorithms.

# significance
Enabling LLMs to incorporate feedback into reasoning is a crucial capability for amortizing expensive inference and coping with partially unknown application domains. This is an important problem and new techniques to address this problem - like using MCTS to structure search and track action quality and uncertainty - are important. The work aims to leverage MCTS while reducing the number of LLM invocations needed compared to prior work (RAP).

### Weaknesses
# originality
The MCTS formulation is similar to RAP in some ways, but differs in many key details, particularly around how the world modeling is done, where RAP explicitly models the world but REX uses the implicit model of the LLM. Augmenting LLMs with search techniques is not itself novel, but this is not a substantial lack of originality per se either.

# quality
Evaluation results only weakly support the REX models. The performance improvements are primarily for (some) small Blocksworld domains, with comparable performance on GSM8K. This makes it unclear to what degree the algorithm is an improvement over existing methods.

Benchmarking is only done on relatively simple example tasks, making it difficult to assess how well these results translate to real world use cases.

Evaluation does not match techniques on the number of tokens they generate or evaluation calls they make. This makes fair comparisons impossible, as the different methods are afforded different computational budgets for the same objective.

# clarity
The text can be hard to follow. Partially this is due to framing the technique as reinforcement learning, despite the lack of a policy being learned (though presumably the MCTS count tables could be preserved and treated as a learned policy of sorts). The core algorithm description was difficult to follow and the main figures did not support understanding this algorithm.

# significance
The results do not show substantial improvements over prior efforts, limiting the significance attributable to the technical novelties.

### Questions
- Figure 2
	- Consider adjusting the visuals here to make the differences clearer. Perhaps by showing where LLM evaluations occur along the search tree to make it clearer that REX-UCL is evaluating the LLM for each step of the depth of the tree. 
	- Figure 3 was helpful, but I could not understand what Figure 2 was meant to illustrate by highlighting some lines in the rollout as red.
- Table 1
	- Is it possible to provide uncertainty estimates on the scores provided? (For example, using multiple seeds for different runs of the algorithms)
	- What does "size" mean in this table?
	- What is the intuition for why the REX methods do worse in the largest Blocksworld scenario? They show strong advantages in the 2-step scenario (80% vs 41.7%), but their advantage shrinks in the 4-step scenario (44.6% vs 42.0%) and becomes a deficit in the 6-step scenario (25.4% vs 30.0%).
	- Is there other evidence for the strength of REX on GSM8k-test? The results look very close to CoT for REX-UCB and to Reflexion for REX-UCL. Could the results be subdivided by number of steps required? 
		- Note that Reflexion requires `3n` time complexity, while REX-UCL requires `dn`, so on a 3-step problem the two are the same (if I understood correctly), while on a 4 or more step problem REX-UCL will be more computationally demanding unless search depth is limited.
	- What search depth(s) (`d`) were used for the experiments?
- Section 5.4
	- Could the results in table 1 be replicated with a model that exposes logits and compared?
		- These additional results would strengthen the claims about the core technique.
		- This would also enable a comparison to RAP.
	- How were `C`, `B`, and `K` selected for the experiments?
- General
	- Consider computing the results with (approximately) matching compute budgets for the methods being compared. Ideally this would be a matching number of generated tokens. A more coarse comparison would be matching the number of inference calls made. Regardless, some baseline equivalence should be established for the compute budget to allow fair comparison of the accuracy.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents REX – a method to achieve more efficient exploration of the action space, leading to a more robust performance in LLM based AI agents.  Through empirical results in blocksworld and GSM8K (math word problems), the approach is shown to achieve superior performances relative to existing approaches.

### Strengths
Overall the paper is well motivated and well written. Proposed mechanisms such as integration of rewards into the prompt are novel.

### Weaknesses
The approach seems to be limited to problems with discrete actions, and assumes the scores can be mapped to HIGH/LOW which may not always be possible.

### Questions
1.	The paper presents two algorithms : REX-UCB and REX-R, which differ only in the way the rewards are classified. I wonder whether these could have been unified into one coherent approach. If not, it is important to explicitly mention when one should use which algorithm.

2.	Is the approach limited to discrete actions? Could it be adapted for continuous action space problems?

3.	Is it assumed that a user can map the UCB scores to HIGH/LOW? This may not always be possible. 

4.	Can the method deal with non-binary rewards?

5.	How would the method fare in tasks that involve a long sequence of steps till the final solution is found? Would the LLM’s prediction accuracy (of intermediate steps -with reference to line 7 in Alg 1) be a limitating factor in such tasks?

6.	Can the tasks described be solved with reinforcement learning alone? If so, this would be a natural baseline to add to the experiments.

7.	An intuitive example to accompany 1. And 2. In Section 4.1 would have provided a much clearer picture of the approach.

8.	The introduction could have included more emphasis about how the contributions (1), (2) and (3) are achieved.

9.	In section 4.1 - `form’ should be ‘from’. This mistake is also repeated in later parts of the paper.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose an algorithm called Rapid Exploration and Exploitation (REX) to improve the performance of AI agents while balancing the exploration-exploitation trade-off. REX introduces an additional layer of rewards and integrates concepts similar to Upper Confidence Bound (UCB) scores, leading to more robust and efficient AI agent performance. The authors evaluate the performance of the proposed method in several benchmark tasks.

### Strengths
- Important and timely topic
- Nice connections between RL/bandit literature and LLM
- I agree with the authors that MCTS is a promising approach to combine with LLM; hence, the authors try to make progress in a good direction.

### Weaknesses
- Regarding experiments, I have several concerns. First, I consider that the authors should compare of the performance of their LLM-based approaches with SOTA methods that are not based on LLMs. Second, the authors state that "RAP is not compatible with OpenAI APIs; therefore, we have exclusively focused on evaluating the time complexity of RAP, without delving into its accuracy", but I do not think that this is a good reason why the authors do not have to empirically evaluate RAP. If the authors emphasize the better computational complexity of their proposed method, they may want to add rigorous theoretical analysis.

- The ideas presented in this paper are quite similar to Tree-of-Thoughts (Yao et al., 2023). Though this is a recent work, but uploaded on arXiv on May 17, 2023; hence, this cannot be regarded as a concurrent work unless I misunderstand the ICLR rule. I would recommend the authors to at least the similarities and differences and compare the authors' proposed method with Yao et al. (2023) in same benchmarks.
    - Yao, Shunyu, et al. "Tree of thoughts: Deliberate problem solving with large language models." arXiv preprint arXiv:2305.10601 (2023).

- Maths are sometimes weird to me. For example:
    - Algorithm 1, line 3: $\forall (s,a) \in Q$. $Q$ is not a set of state-action pairs. This part should be $\forall (s,a) \in S \times A$.
    - Algorithm 1, line 14: $\forall (s,a) \in H$. $H$ is a time step.
- The overall alogrithmic flow is not clear to me. I guess it is mainly because "AGENT.FUNCTION()" (FUNCTION = {SOLVE, VALIDATEACTION, CALCULATEUCB}) is not explained. In addition to the math problems mentioned earlier, the algorithmic flow in this paper is hard to follow.

### Questions
[Q1] Could you tell me the similarities and differences between the authors' approach and Tree-of-Thoughts (Yao et al., 2023)? Is it possible to compare the performance of the two methods in the same benchmark? Because the authors use BlocksWorld and GSM8K and Yao et al. (2023) use Game of 24, Creative Writing, and Mini Crosswords, I guess some of them are used for the comparison. 

[Q2] Although I can guess what are Agent.Solve and CALCULATEUCB to some extent, there is almost no information on what is Agent.ValidateAction. Could you elaborate more?

[Q3] I suspect that the high performance of REX in BlocksWorld is due to the data leak. Did the authors use the ordinary BlocksWorld environments with default settings?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
