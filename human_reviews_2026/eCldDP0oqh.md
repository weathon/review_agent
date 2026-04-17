# Estimating the Empowerment of Language Model Agents

- Decision: Reject
- Scores: 8, 4, 6

## Abstract
As language model (LM) agents become more capable and gain broader access to real-world tools, there is a growing need for scalable evaluation frameworks of agentic capability. However, conventional benchmark-centric evaluations are costly to design and require human designers to come up with valid tasks that translate into insights about general model capabilities. In this work, we propose information-theoretic evaluation based on empowerment, the mutual information between an agent's actions and future states, as an open-ended method for evaluating LM agents. We introduce EELMA (Estimating Empowerment of Language Model Agents), an algorithm for approximating effective empowerment from multi-turn text interactions. We validate EELMA on both language games and scaled-up realistic web-browsing scenarios. We find that empowerment strongly correlates with average task performance, characterize the impact of environmental complexity and agentic factors such as chain-of-thought, model scale, and memory length on estimated empowerment, and that high empowerment states and actions are often pivotal moments for general capabilities. Together, these results demonstrate empowerment as an appealing general-purpose metric for evaluating and monitoring LM agents in complex, open-ended settings. Code available: https://anonymous.4open.science/r/EELMA-E227

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an information-theoretic framework, called EELMA (Empowerment Estimation for Language Model Agents), to quantify the empowerment of LLM-based agents. Empowerment is defined as the mutual information between an agent’s actions and the resulting future states, representing how much the agent can influence its environment. EELMA applies this concept to language-based settings, where actions and states are represented by text embeddings, and mutual information is estimated via contrastive learning (InfoNCE). The method is evaluated in several environments, including Gridworld, Tower of Hanoi and WebArena. Results show that empowerment correlates strongly with task reward and captures patterns of agentic behavior such as controllability and power-seeking.

### Strengths
1. Timely and relevant topic.

The paper addresses a central question for the emerging field of LLM-based agents: how to measure agentic capability in a goal-free, interpretable, and scalable way.

2. Conceptual novelty.

Adapting empowerment from control theory to language environments is creative. It introduces an information-theoretic lens to quantify an agent’s potential influence without relying on pre-defined tasks or rewards.

3. Goal-agnostic evaluation framework.

The work moves beyond conventional benchmarks and offers an interesting direction for measuring emergent autonomy in complex environments.

4. Potential broader impact.

The strong correlation between empowerment and actual performance across diverse settings suggests that empowerment captures meaningful structure in agent behavior. Empowerment could become a useful signal for monitoring or constraining power-seeking tendencies, which connects to safety and interpretability concerns in agentic AI.

### Weaknesses
1. Conceptual ambiguity in defining empowerment for language models.

In its original formulation, empowerment measures how an agent’s actions causally influence future physical states. In language models, however, the “future state” is a symbolic context, not a physical one. This makes the notion of empowerment somewhat ambiguous: the estimated mutual information may reflect linguistic diversity or discourse influence rather than true causal control.

2. Simplified experimental environments.

The toy setups (Gridworld, Tower of Hanoi) are helpful for clarity but limited in linguistic and social complexity. While the WebArena tests are more realistic, the evaluation still relies on scripted interactions. The claim of scalability would be stronger with more open-ended or multi-agent scenarios (Maybe Minecraft?).

### Questions
* Do you envision empowerment being used not only as an evaluation metric but also as a design for improving LLM agents? For example, could empowerment be incorporated into agent training or architecture design?

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
The paper introduces a method for evaluating the capabilities of LM-based agents based on empowerment. They propose EELMA, a way to estimate empowerment from multi-turn text interactions using existing approaches to contrastive (InfoNCE) mutual-information estimation combined with text embedders. Results show that across controlled language-based environments like Gridworld, Tower of Hanoi and web-browsing tasks (WebArena), EELMA’s empowerment estimates strongly correlate with average task performance, decrease with environmental complexity, and increase with model scale, memory, and chain-of-thought reasoning. Other empirical results suggest that EELMA can flag pivotal actions such as authentication events, pointing towards use cases for open-ended evaluation or safety monitoring of LM agents.

### Strengths
- As far as I can tell, this paper is the first to investigate measuring empowerment in text-based domains.

- The paper’s experiments convincingly demonstrate that the EELMA estimate is close to the direct empowerment in the Gridworld and Tower of Hanoi environments.

- The experiments in the Gridworld and Tower of Hanoi are quite rigorous and explore scaling capabilities across several axes (model size, CoT, memory, etc.). The results show convincing trends across several model types.

- The paper is easy to understand and read and the graphics cleanly present experiment results.

### Weaknesses
- The algorithmic contribution is minor, combining existing mutual information estimation approaches for empowerment with text embedders so that they can be used with LM agents in text domains.  

- Existing papers have demonstrated the same result of the main claim (that effective empowerment can approximate agentic performance) in environments very similar to Gridworld and Tower of Hanoi, simply without representing the environment and actions in text. The experiments in this domain are really only showing me that adding the text-based embedding on top of infoNCE doesn’t alter the results of existing works.

- The experiments in the more complicated domain of WebArena are more limited (i.e., only 3 data points across each type of environment) and demonstrate mixed results. I think putting a larger emphasis on novel domains (e.g. WebArena) instead of environments in which empowerment has been thoroughly explored (gridworlds or similar small, discrete games) would greatly strengthen the paper.

- Training the embedders and successor representation requires additional rollouts (potentially synthetically generated as mentioned in the paper) which limits the generality of the method. Are there any ideas for relaxing the need to collect additional rollout in the specific environment so that the measure of empowerment can generalize to unseen environments?

### Questions
- Why was a p value of 0.001 chosen in the right side of Figure 8? This makes the error bars overlap and doesn’t give a meaningful sense of the statistical significance. I would recommend choosing a p-value that makes the error bars not overlap.

- Your estimated empowerment for invalid passwords in section 5 is negative. Shouldn’t empowerment always be non-negative? What is going on here?

- How do you expect your approach and results to scale to more complicated and longer horizon tasks? What is the current main bottleneck to scaling up?

- Why did you choose to use discounted rewards in WebArena instead of success rate (as far as I know, the original WebArena work uses success rate)? How do the results change with success rate instead of discounted reward?

Misc:
- Appendix K has some formatting errors
- Line 1261 and 1402 has an undefined ref
- Section C.1 has several punctuation errors and typos.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes EELMA, a method to estimate the empowerment of language model agents. Empowerment is defined as the mutual information between an agent’s actions and its future states. The authors argue that it provides a task-agnostic, open-ended way to measure agentic capabilities. They implement EELMA using embedding-based representation learning and InfoNCE-style mutual information estimation. The method is evaluated in both structured environments (GridWorld and Tower of Hanoi) and a more realistic setting (WebArena). Experiments show that empowerment correlates well with goal-conditioned performance, and can highlight influential decision points in trajectories.

### Strengths
1. Introduces a task-agnostic, theoretically grounded metric for evaluating LLM agents.

2. Proposes a scalable estimation algorithm for empowerment in language-based environments.

3. Shows empirical evidence of strong correlation between empowerment and average task reward.

### Weaknesses
1. The semantic meaning of empowerment remains abstract in complex language settings.

2. The proposed method still relies on known reward functions for validation, which weakens its claim of being fully task-agnostic.

3. The computational cost of training EELMA is not discussed in practical terms.

### Questions
1. How sensitive are the empowerment estimates to the choice of embedding model?

2. Would fine-tuning the embeddings improve performance, and at what cost?

3. How can EELMA be used in practice, for example, during online deployment?

4. In settings where high empowerment is not aligned with human goals, how should we interpret or act on those results?

### Soundness
3

### Presentation
3

### Contribution
3
