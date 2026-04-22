# Speculative Actions: A Lossless Framework for Faster AI Agents

- Avg Score: 7.50
- Decision: Accept (Oral)
- Scores: 6, 6, 10, 8

## Abstract
AI agents are increasingly deployed in complex, interactive environments, yet their runtime remains a major bottleneck for training, evaluation, and real-world use. Typical agent behavior unfolds sequentially, where each action requires an API call that can incur substantial latency. For example, a game of chess between two state-of-the-art agents can take hours. We introduce speculative actions, a lossless acceleration framework for general agentic systems. Inspired by speculative execution in microprocessors and speculative decoding in LLM inference, our method uses faster models to predict likely future actions and executes them in parallel, committing only when predictions match. We evaluate speculative actions across gaming, e-commerce, and web search environments, and additionally study a lossy extension in an operating systems setting. Across domains, we achieve up to 55% next-action prediction accuracy, translating into substantial latency reductions. Finally, we present a cost–latency analysis that formalizes the tradeoff between speculative breadth and time savings. This analysis enables principled tuning and selective branch launching, to ensure multi-branch speculation delivers practical speedups without prohibitive cost growth.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a way to increase the speed of agent-environment interactions by leveraging speculative execution.

They call this framework "Speculative Actions". It is a lossless framework that predicts the K most likely next actions using fast models, enabling multiple steps to be executed in parallel (before the real next action of the slow model is obtained).

The authors evaluate it on multiple settings, showing that it yields noticeable speedups.

### Strengths
This paper addresses the very practical need of maximizing efficiency and throughput in agentic settings, where API calls can be costly and time-consuming.

The authors show that the proposed method works well on a diversity of practical settings.

The paper is mostly easy to follow and understand. The logical order of presentation makes sense.

The paper includes bar charts that visually illustate the advantage of the proposed method. I appreciate the error bars in Figure 2.

### Weaknesses
There are a few places where terms, acronyms, or notations are used before first being defined and explained. More on this below.

Page 1:

Define what the acronym "MCP" means ("Model Context Protocol") before it's first used. Explain what it means, either here or on page 2 (where it says "MCP servers for agentic systems...").

Page 2:

"while waiting results" -> "while waiting for results"

Page 3:

"with simple implementation" -> "with a simple implementation"

"in computer architecture" -> "in the field of computer architecture"

"wrong and" -> "wrong, and"

"an Markov" -> "a Markov"

"(MDP) (st, at), where st denotes" -> "(MDP). We let st denote..."

Page 4:

"a set of API responses {\hat{a}_t}" -> "a set of k API responses {\hat{a}_t^(i)}_{i=1}^k". This explains what k refers to before it's used later on.

You have not defined what "Exp" means in Exp(α) and Exp(β). Define the notation before it's first used, by explicitly stating that Exp(λ) means an exponential distribution with rate λ.

Page 5:

"speculation need to be" -> "speculation must be"

"via fork" -> "via forking"

"Consider a game at turn t, " -> "Consider a game at turn t: "

"reasoning eliciting" -> "reasoning-eliciting"

"prompt" should not be in italics inside math formulas Use \text{prompt} to avoid this.

"applying predicted" -> "applying the predicted"

Page 6:

"match" should not be in italics inside math formulas. Use \text{match} to avoid this.

"If there exist no match" -> "If no match exists"

"next turn where Q is in turn to" -> "next turn, where it is Q's turn to"

"play while time is" -> "play, while time is"

"and computational complexity" -> "and the computational complexity"

Page 7:

"agent need to" -> "the agent needs to"

"is the API calls... that are needed" -> "is the API call... that is needed"

Page 8:

"predicts API call" -> "predicts the API call"

Space missing after "yielding predicted states".

"ht + 1" the "t + 1" is not properly subscripted.

"k ∈ {1,3}" should be typeset as a formula.

"k = 3" should be typeset as a formula.

Page 9:

"Our evaluation shows that the Speculator:" -> "Our evaluation shows that the Speculator-Actor system:" ?

### Questions
Page 3:

By "regenerating failures", did you really mean "recovering from failures"?

Figure 2: What do the error bars show? Standard deviation? Standard error? A 95% confidence interval for the mean (computed with which method)?

What's the best way to set k in practice? Can this be done in an online manner?

Alternatively, could one automate the selection of k situationally (i.e., on a step-by-step basis)? For example, you could have a fast model that's been trained to predict the *thinking times* of the slow models and/or transition function, and that could allow you to budget the number of speculations appropriately.

Are error bars missing from Figures 3 and 4? Were these run for multiple trials?

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
The paper proposes Speculative Actions, a framework for accelerating AI agent–environment interactions by predicting future actions with a smaller, faster “Speculator” model, while the main “Actor” validates them asynchronously. The idea is to make agent execution more parallel and efficient, analogous to speculative decoding in LLMs or speculative execution in CPUs. The paper demonstrates this concept across multiple domains (chess, e-commerce, HotpotQA, and OS tuning) and claims consistent speedups.

Overall, the motivation is reasonable as reducing agent latency makes sense, especially in complex API-driven workflows. However, in many realistic agent scenarios, I don't think the latency bottleneck is as severe as claimed, and the proposed speculative mechanism may introduce new costs or practical issues that are not fully addressed.

### Strengths
- The paper identifies a clear, relevant problem: latency in sequential agent–environment interactions.

- The proposed speculative framework is simple and implementable, with clear lossless and lossy variants.

- Multi-domain experiments demonstrate feasibility and some measurable speedups.

- The topic (efficient agent execution) is timely and of practical interest.

### Weaknesses
The paper provides strong empirical results but lacks deeper theoretical justification for why the speculative framework remains “lossless” under all conditions. Moreover, the evaluation mainly focuses on latency gains without a detailed analysis of trade-offs in resource consumption or potential instability in large-scale multi-agent settings. I may raise my evaluation on this paper if the authors could provide better justification for this concern.

### Questions
- The paper claims “up to 30% end-to-end speedup.” What is the variance across environments, and how were these averages computed?
- Could speculative execution introduce hidden costs that offset real-world gains?
- In multi-step speculation, how do you control exponential growth in parallel branches?
- How reproducible are the speedups given that API latency for large models may not be consistent?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper proposes the concept of speculative actions, which uses speculative models in sequential environments to achieve significant speedups (up to 30%). Concretely, the paper considers the setting where an API determines the next action based on the current state, and this API is instantiated by either an expensive LLM (e.g., high-reasoning mode) or a human response. The paper proposes an algorithm that uses a much faster model, in this case a smaller LLM, to predict the likely output and to precompute the next state based on this action. If the action matches the prediction of the expensive model (or human), the algorithm can directly proceed to the next state, thereby processing two steps at a time. Otherwise, the environment generates the next state based on the true action, progressing without overhead compared to the sequential baseline. The paper considers four different environments with different constraints, and shows that the proposed approach achieves speedups of up to 30%.

### Strengths
The paper considers an important topic, namely, decreasing the latency of LLM agents in sequential environments, and introduces a novel algorithm to achieve significant speedups. The proposed approach is naturally inspired by speculative decoding from other domains, but the paper makes a significant contribution by showing that it is also applicable to this setting. The paper is well written, with clear illustrations and a convincing motivation. The claims are backed up by substantial experimental evidence across a wide range of real-world environments.

### Weaknesses
I think the cost-vs-latency tradeoff is an important aspect of this approach and should be discussed in the main paper, e.g., using the additional page of the camera-ready version (if applicable). In a practical application, it would be great for the user to have a tunable knob between costs and latency.

I found two typos: sequantial (L293) and gameply (L316).

### Questions
How can a user find an appropriate speculative model? Did any smaller LLMs not achieve a satisfactory performance?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a new general framework called "Speculative Actions" in which agents predict future states of the world that will come about as a consequence of e.g. the environment, other actors, computation, API calls, and performs API calls based on that prediction.

### Strengths
The idea is solid and the experiments convincingly show a speedup.

### Weaknesses
No weaknesses

### Questions
No questions

### Soundness
3

### Presentation
3

### Contribution
3
