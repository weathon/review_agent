# Test-Time Adaptation for LLM Agents via Environment Interaction

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions.
This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time.
To address these issues, we propose two distinct strategies for adapting LLM agents by leveraging environment-specific information from interaction that is available during deployment.
First, an online syntactic alignment (SA) method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format.
Second, a deployment-time dynamics grounding (DG) method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with an in-context world model.
We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation.
Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost.
We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents.
For example, on the WebArena multi-site split, this method increases the agent's success rate from 2\% to 23\%.
We release our code.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes 2 test-time adaptation methods for LLM-based agents, which allows the agent explore the environment before the task and update its parameter or context based on the exploration. The experiment shows that this method can improve the performance of the web agent compared with WAM (using world model to predict the next observation and filter the action)

### Strengths
1. This idea is natural but effective. The most prior knowledge in the environment agent have, the better the agent can perform.

2. Lifelong learning is also a important topic. This paper try to give a solution to the problem by training. (Although its solution has some weaknesses)

### Weaknesses
1. The writing is really poor. Those 2 figures can't really present the idea of 2 methods.

2. The classification of Syntactic Mismatch and Semantic Mismatch is meaningless. They are both the result of the lack of prior knowledge in the environment. Although the author try to seperate them by using 2 different methods, it looks like more weird, since they are both solve the lack of prior knowledge, no matter what prior knowledge it is. Such mapping is useless, it just like to increase the complexity of the paper.

3. Almost half of the results in Table 2 is empty, which may cause unfair comparison. For example, is AWM really worse than your method in other task? (gpt-4o-mini & Qwen). Is it because the AWM's result comes from a old gpt-4o-mini model? You need to reapplied this method in another benchmarks to prove your claim.

4. If the environment can't be explored, this method will not work, although I agree most of the environment can be explored.

As a result, although I quite like this idea, the soundness and the writing are not good enough, so I'll give a borderline score to this paper.

### Questions
1. Explain each variation clearly, like I1:n−1 in equation 3.

2. Fully polish your writing, make it more clear and easy to understand.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the core challenge of Large Language Model (LLM) agents' poor generalization when facing novel, unseen environments. The authors innovatively decompose this challenge into two primary failure modes: Syntactic Mismatch and Semantic Mismatch.

To address these issues, the paper proposes two annotation-free Test-Time Adaptation (TTA) strategies:

1. A parametric (PA) method that learns a lightweight adaptation vector, δ, to bias the model's output distribution, enabling rapid alignment with the environment's syntactic structure.
2. A non-parametric (NPA) method that, prior to task execution, actively learns the environment's causal dynamics through an exploration phase and provides them as context to the agent, thereby constructing a non-parametric world model.

### Strengths
1. Introduces Test-Time Adaptation (TTA) into the domain of LLM agents, proposing targeted solutions for syntactic and semantic mismatches.
2. The proposed methods are plug-and-play and exhibit strong adaptability.
3. Provides detailed analysis and comprehensive ablation studies.

### Weaknesses
1. The combination of the parametric (PA) and non-parametric (NPA) methods does not demonstrate synergistic effects; in fact, their integration fails to yield better performance.
2. The generalizability of the parametric adaptation method is not sufficiently demonstrated, as its experiments were confined to the Qwen-2.5 family of models.
3. In more diverse and complex environments, the parametric adaptation method might require more training steps, leading to increased computational latency.

### Questions
1. Have the authors considered validating the effectiveness of both methods on a broader range of models, such as Llama-3 or GLM-4.5?
2. For the non-parametric adaptation method, the comprehensiveness of the exploration (e.g., exploring the full functionality of a website, not just a subset) is critical. How do the authors ensure that the exploration is sufficiently comprehensive?

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
2

### Summary
The paper proposes two method for test-time adaptation of LLM-based agents: a "parametric" and a "non-parametric" adaptation. The parametric adaptation proposes, for each episode, to learn a single adaptation vector which is added to the final hidden representation before the final projection layer, i.e., the vector acts as a logit bias. The non-parametric adaptation proposes a one-time deployment pipeline per environment: (1) derive exploration goals from the environment description via LLM calls, (2) run an LLM agent to explore and log (observation, action, new-observation) transitions, (3) have an LLM summarize these logs, (4) have a reasoning model filter out duplicate rules; at test time, the filtered rules are appended to the agent's context.

### Strengths
### Clarity

The writing is clear and free of typos. The figures are well designed and effectively illustrate the proposed adaptation strategies. The paper is also well structured: it opens by partitioning LLM-agent failures into two categories and then develops a corresponding adaptation strategy for each, yielding an easy-to-follow narrative.

### Originality

Based on my understanding of the literature, the proposed ideas seem reasonably novel in the context of LLM-based agents.

### Quality

This paper (i) characterizes failure cases of LLM-based agents, (ii) formalizes deployment constraints that bound viable solutions, (iii) derives methods that adhere to those constraints, and (iv) evaluates them against established agent baselines.


### Significance

The paper tackles out-of-distribution (OOD) performance in LLM-based agents which is an essential issue for robust deployment.

### Weaknesses
While Section 2 offers useful background on test-time adaptation and LLM-based agents, the paper stops short of situating its specific design choices within the existing literature. For instance, the proposed “parametric adaptation” appears to extend the methodology of “Steering LLM Reasoning Through Bias-Only Adaptation” [1] to the context of LLM-based agents.

[1] Sinii, V., Gorbatovski, A., Cherepanov, A., Shaposhnikov, B., Balagansky, N., & Gavrilov, D. (2025, May 24). Steering LLM reasoning through bias-only adaptation (arXiv:2505.18706 [cs.LG], v1). arXiv. https://arxiv.org/abs/2505.18706

### Questions
I strongly suggest to contextualize the design choices for both the “parametric” and “non-parametric” adaptation methods in relation to prior work, highlighting key precedents and  differences.

### Soundness
3

### Presentation
3

### Contribution
3
