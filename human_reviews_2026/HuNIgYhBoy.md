# Benchmarking World-Model Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Model-learning agents should gather information to learn world models that support many downstream tasks and inferences, such as predicting unobserved states, estimating near- and far-term consequences of actions, planning action sequences, and detecting changes in dynamics. Current methods for learning and evaluating world models diverge from this goal: training and evaluation are anchored to next-frame prediction, and success is scored by reward maximization in the same environment.

We propose **WorldTest**, a protocol to evaluate model-learning agents that separates reward-free interaction from a scored test phase in a different but related environment. WorldTest is open-ended---models should support many different tasks unknown ahead of time---and agnostic to model representation, allowing comparison across approaches.

We instantiated WorldTest with **AutumnBench**, a suite of 43 interactive grid-world environments and 129 tasks across three families: masked-frame prediction, planning, and predicting changes to the causal dynamics. We compared 517 human participants and three frontier models on AutumnBench. We found that humans outperform the models, and scaling compute improves performance only in some environments but not others.

WorldTest provides a novel template---reward-free exploration, derived tests, and behavior-based scoring---to evaluate what agents learn about environment dynamics, and AutumnBench exposes significant headroom in world-model learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces WorldTest, a novel evaluation protocol that bifurcates agent assessment into a reward-free "interaction phase" for model building and a "test phase" to evaluate generalization on derived, goal-oriented tasks. The authors implement this in AutumnBench, a benchmark with 43 grid-world environments and 129 tasks. Large-scale experiments with 517 humans and three AI models reveal that humans significantly outperform AI, highlighting fundamental deficits in the models' exploration strategies and hypothesis-testing abilities. The work provides a valuable framework and empirical toolkit for a more rigorous evaluation of world models.

### Strengths
- The WorldTest protocol's separation of learning and testing is a key contribution, shifting evaluation from task-specific performance to a deeper understanding of world dynamics. Its representation-agnostic and behavior-based nature ensures broad applicability.
- AutumnBench serves as a rich and extensible implementation, featuring 43 diverse environments and 129 tasks across prediction, planning, and counterfactual reasoning. The underlying DSL facilitates future community expansion.
- The large-scale human study provides a robust baseline, and the analysis moves beyond simple metrics to reveal AI's deeper shortcomings in exploration and metacognition, evidenced by analyses of "reset" usage and action perplexity.
- The paper is well-written, with a clear and logical structure that makes the contributions easy to understand.

### Weaknesses
- The empirical evidence is confined to 2D grid-worlds, which may limit the generalizability of the conclusions to more complex physical domains, despite the paper's broader claims.
- The use of a GUI for humans versus a text interface for AI is a potential confound. This difference may partly explain behavioral discrepancies, such as the varied use of the "reset" action, and warrants further discussion.
- Attributing model failures to metacognitive deficits is insightful but would be strengthened by more direct evidence. Ablation studies could help disentangle these deficits from artifacts of prompt design or limitations in pre-training data.

### Questions
- Recent work has explored RL fine-tuning LLMs for decision-making. Have you considered evaluating RL fine-tuned models on your benchmark to see if this addresses some of the observed limitations? This would complement the current zero-shot results.
- Could you elaborate on the specific challenges you anticipate when instantiating the WorldTest framework in a more complex domain, such as a 3D physics simulation?
- For the AI models, is there a correlation between the quality of their exploration in the interaction phase (e.g., action perplexity dynamics) and their final performance in the test phase? Such an analysis could offer clues for improving exploration strategies.

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
2

### Summary
The paper proposes testing world models beyond the environment of interaction. Traditional model-based RL is only interested in the learned policy given the learned model.
This inherently makes sense to me, given that it is hard to define casually which elements of the environment should be directly learned to guide the learning of a good policy ‘in imagination’.
This benchmark allows for environment interaction during the training phase to learn a world model, but no reward or task is available. It creates a very simple grid-like world inspired by the ARC challenge and Minigrid environments. 
 After the test phase, the agent is tasked with solving a task that requires a good world model — for example, a task directly related to reconstructing observations at the end of an episode, rendered as a classification objective (this is chosen to simplify the output space for LLMs).Action planning and change detection from the nominal dynamics.
I don’t fully comprehend how this benchmark is useful for development and provides new interpretability aspects compared to existing benchmarks - it seems to be specifically applicable to LLMs.
The proposed Change Detection and Masked Frame Prediction tasks align with the typical reconstruction loss used in evaluating world models, and the planning corresponds to task success.

### Strengths
- large benchmark 
- well presented

### Weaknesses
- not 100% clear how these benchmarks can guide the design of new algorithms
- interrelation to existing model-based evaluations e.g., with continuous action space is missing
- missing novel insight based on the benchmark or mechanistic interpretability

### Questions
Q1: How can this benchmark help in developing new algorithms?

Q2: How is it, in principle, different from evaluating task success and reconstruction error in typical world-modeling studies?

Q3: Often, we are not interested in learning the full dynamics but rather the task-relevant dynamics when considering world-model learning. Any comments on why we enforce learning the full dynamics?

Q4: Can this benchmark support any non-LLM-based methods?

Q5: How well are open-source models performing?

Q6: All LLMs tested seem to perform similarly — why might that be, and do you have any thoughts on this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work focuses on evaluating the ability of models - LLMs in particular - to develop a world model of an environment. The contribution is two fold, first introducing a general purpose framework for evaluating a world model (World Test) then a suite of gridworld tasks that implement the world test framework (Autumn Bench). Results are then presented across human participants and reasoning models, showing that human out-perform LLMs, and that LLM performance can't always be improved by having more compute time.

### Strengths
The paper is well scoped to a conference paper and relatively well written. The authors clearly contextualise their approach within the existent literature on evaluating world models. Additionally the framework for world model evaluation seems well defined and described.

### Weaknesses
There are two weaknesses worth highlighting, first the overall presentation and polish of the paper, second the methodology for evaluating LLMs.

The paper needs to be proofread, there are a number of typos throughout (in particular the same sentence appears to be repeated verbatim at line 341 and 343). Additionally all empirical results are reported in plots none of which have titles or axis labels. This is particularly confusing in figure 6 where plots share a y axis despite showing different quantities. I would also note the discussion section introduces a collection of concepts - like metacognition at line 460 - which have not previously been introduced or unpacked.

In comparing LLM to Human Performance it seems to rely at least in part on the quality of the prompt describing the task. By comparison the human user interface is well designed - have the authors tried a variety of prompting strategies to allow them to confidently say their findings represent the best these models can do?

### Questions
To what degree is the poor performance of the LLMs a result of the task formulation in the prompt?

Given that these are grid world environments, can you provide a clearer notion of what this method can and cannot evaluate compared with say a 3d simulated environment?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the current view of world model evaluation by proposing a new inference time protocol called “WorldTest”. WorldTest contains two phases that a pretrained reasoning model or a human subject undergoes: (1) Goal & reward free interaction phase and (2) Test phase. During the interaction phase, the agent explores the “rules” of the environment with no explicit reward function or goal state. During the test phase, the agent must solve a specific objective on an environment derived from the base environment. The same process is followed to evaluate world model capabilities in humans. The paper reports that humans vastly outperform reasoning agents due to their superior metacognitive abilities like experiment design, uncertainty quantification and flexible belief updates and suggests improving these abilities in current reasoning models.

### Strengths
1. The goal of this paper to tackle evaluation of world model capabilities is important and well timed. There has been speculation among researchers regarding world model capabilities in reasoning models. The final results expose this gap for LLM reasoning agents.
2. The paper presents a simple and intuitive idea to use a reward free exploration phase and a goal oriented test phase for world model evaluation.
3. Experiments and results are well presented and easy to follow. If the method is truly novel and first of its kind, it has the potential to become a standard world model benchmark for reasoning and agent based LLMs in the near future.
4. Use of “reset” action in humans as a hypothesis testing tool is an insightful conjecture - something that did not cross my mind immediately when reading the paper.

### Weaknesses
Also adding Questions here along with Weaknesses:
1. While I really liked the presented idea, I am not sure how novel the contributions are when compared to DiscoveryWorld [1] and URLB [2]. I find the paper lacking in justifying the the utility of their implemented WorldTest method against numerous similar testbeds available. For example, URLB presents very similar ideas with the only difference being that during test phase, the agent is tested on the same environment. A simple variation on the environments used in URLB like changing the robot/walker parameters (size, gripper, etc) would yield a WorldTest like scenario. 
2. Do the authors claim “AutumnBench” as a part of the contribution? If so, is there a specific reason for creating and using “AutumnBench” instead of other existing tasks like BabyAI [3] or ARC-AGI2 [4]. I can imagine a minigrid BabyAI environment could be easily tuned in such a way that there is no inherent goal during the exploration phase and the already existing difficulty levels could provide variations to the existing task for the “Test” phase (masked prediction, planning, etc).
3. I found the categorizatoin of world model evaluations on Page 1 and 2 rather vague and confusing (non-interactive, representation-based, gym-like, unsupervised RL). It does not seem to serve any purpose other than introducing a few related works. Perhaps move them to a related works section and contrast the paper’s approach against the past works more clearly? This would benefit future readers in assessing why they would want to use WorldTest instead of other benchmarks. 
4. As a follow up to the related works, The authors consider ARC-AGI [5] as a non-interactive environment. However, it seems like (a) an internal rule based system/proto-world model based on shown examples (to answer - how did we arrive from initial to final state?) is required to solve the test problem and (b) several interactions must be made to select grid size and place colored tiles on the grid during test. Could AutumnBench be a more difficult version of ARC-AGI (similar to ARC-2, ARC-3)? If not, please clarify where exactly the added complexity and novelty lies for AutumnBench.
5.  Why were traditional world models from RL literature like Dreamer-v3, Transformer WMs (IRIS), etc not compared or discussed in this paper? To adapt this benchmark to traditional WMs, it seems like a simple conversion of current observation grids to integer vectors instead of string vectors and enumerated actions to another output vector for a policy/planning method. Rewards could be removed during the WM learning/exploration phase. How would such a model be evaluated and how would they perform?
6. How are the environments changed during the test phase for Masked Frame Prediction and Planning tasks? I understand that the Change Detection task inherently requires some perturbation to the dynamics. For the other tasks, are there specific rule based systems that consistently alter the environments? Would the authors provide examples for clarity, if any?
7. While the authors promptly present a few abstract conjectures on why reasoning models failed to match the scores of humans, it would have been great to add a discussion on more concrete steps to improve current reasoning models. How would one go about adding the metacognitive abilities like “better uncertainty quantification” and “felxible belief updating”? I would appreciate a few sentences on this topic during the discussion phase.
8. The authors mention several times - "Due to cost constraints, we evaluated each model’s performance based on a single trajectory completion per problem". To help improve the reproducibility of this work, would the authors be open to disclosing and/or discussing the cost of evaluating the three models used in the paper?
9. **Not part of my evaluation**: I am curious what the authors think about evaluating their testbed on an RL based finetuned reasoning model. Several works have shown that RL based finetuning on a more specific set of behaviors yield superior performance [6] [7]. Similarly, what would they comment on program synthesis methods like [8] or human cognition inspired models like [9]. I would like to hear the authors’ take on how each method would score against WorldTest and harder benchmarks.

### Questions
Asked in the weakness section. Please refer above.

I lean towards a weak accept since I think such a dataset and evaluation protocol is essential to uncover the flaws in today's reasoning agents that are widely being used for coding and other tasks. These tasks require vaguely similar meta-cognitive components that today's reasoning agents severly lack. However, I'd recommend the authors to further develop the work towards (1) comparison with more types of world models seen in RL and (2) concrete discussion on improving world model capabilities of current reasoning models.

References

*[1] Peter Jansen, et. al. Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents. Advances in Neural Information Processing Systems, 2024.*

*[2] Michael Laskin, et. al. Urlb: Unsupervised reinforcement learning benchmark. Advances in
Neural Information Processing Systems, 2021*

*[3] Chevalier-Boisvert, M., D’Eramo, C., Willems, L., Beaudoin, P., Pascanu, R., & Courville, A. (2019). BabyAI: A platform to study the sample efficiency of grounded language learning. arXiv preprint arXiv:1810.08272.*

*[4] Chollet, F., Knoop, M., Kamradt, G., Landers, B., & Pinkard, H. (2025). ARC-AGI-2: A new challenge for frontier AI reasoning systems. arXiv preprint arXiv:2505.11831*

*[5] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019*

*[6] Zhai, S, et. al. (2024). Fine-tuning large vision-language models as decision-making agents via reinforcement learning. arXiv:2405.10292*

*[7] Chen, J., et.al. (2025). G1: Bootstrapping perception and reasoning abilities of vision-language models via reinforcement learning. arXiv preprint arXiv:2505.13426*

*[8] https://substack.com/home/post/p-172998849*

*[9] Wang, G., et.al. (2025). Hierarchical Reasoning Model [Preprint]. arXiv. [https://arxiv.org/abs/2506.21734]*

### Soundness
3

### Presentation
3

### Contribution
2
