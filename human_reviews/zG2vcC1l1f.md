# KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies in Curiosity-driven Exploration

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
In continuous control tasks, Soft Actor-Critic (SAC) has achieved notable success by balancing exploration and exploitation. However, SAC struggles in sparse reward environments, where infrequent rewards hinder efficient exploration. While novelty-based exploration methods help address this issue by encouraging the agent to explore novel states, they introduce challenges, such as the difficulty of setting an optimal reward scale and managing the interaction between novelty-based exploration and SAC’s stochastic policy. These complexities often lead to inefficient exploration or premature convergence and make balancing exploration-exploitation challenging. In this paper, we propose KEA (Keeping Exploration Alive) to tackle the inefficiencies in balancing the exploration-exploitation trade-off when combining SAC with novelty-based methods. KEA introduces an additional co-behavior agent that works alongside SAC and a switching mechanism to facilitate proactive coordination between exploration strategies from the co-behavior agent and the SAC agent with novelty-based exploration. This coordination allows the agent to maintain stochasticity in high-novelty regions, preventing premature convergence and enhancing exploration efficiency. We first analyze the difficulty of balancing exploration-exploitation when combining SAC with novelty-based methods in a 2D grid environment. We then evaluate KEA on sparse reward control tasks from the DeepMind Control Suite and compare against two state-of-the-art novelty-based exploration baselines --- Random Network Distillation (RND) and NovelD. KEA improves episodic rewards by up to 119\% over RND and 28\% over NovelD, significantly improving learning efficiency and robustness in sparse reward environments.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Reinforcement Learning tasks with sparse rewards are always hard to solve. Curiosity-based approaches are derived to facilitate the exploration. However, Soft Actor-Critic (SAC) does not perform well even with curiosity-based exploration. To address this issue, this paper introduces a new exploration strategy KEA (Keeping Exploration Alive), which combines entropy-based exploration and curiosity-based exploration strategies by introducing a simple switching mechanism between two policies with different variance. This mechanism decides which policy to deploy based on the intrinsic rewards.

For experiments, the authors concluded a simple 2D discrete navigation tasks and analysed the learned policy’s entropy and novelty. Results show that their method can improve performance or learning speed for NovelD and RND. They also tried the methods on a set of sparse continuous control tasks. KEA outperforms baseline algorithms. An ablation study of different UTD (Update-to-Data), KEA is more consistent than other baseline methods.

### Strengths
1. This paper addresses an important research questions. Applying Off-policy to tasks with sparse rewards is always tricky and existing curiosity-based methods are not working out of the box.
2. The paper is well motivated via a good visualization. The interplay between policy entropy and state novelty is interesting to look into.
3. The proposed algorithm is simple to implement with a minimal extra computing budget. The switching mechanism is clearly written.
4. The ablation study of various UTD ratios is also interesting. 
5. Relevant literature is good summarised.

### Weaknesses
1. The scope of the paper is rather limited as the authors only tackle Soft Actor-Critic (SAC) in their paper. This paper can be extended to other off-policy algorithms as they all share the issues of premature convergence of the exploration and the influence from different UTD ratios.
2. The experiments are also not very convincing. First, the proposed 2D navigation task is too simple and the performance improvement is also not significant. Second, NovelD and RND both are used as baseline algorithms and NovelD has achieved better performance compared to RND. Yet, the latter experiments are all about RND. Thirdly, in the continuous tasks, the three tasks are not trained for enough time. The performance imporvement is also limited.
3. The presentation/clarity of the paper should also be improved. For example, I  didn't fully get Fig. 1's layout in the beginning. Some sentences are also written in a vague way without further explaination (See questions section). The training of the co-behavior policy is not mathematically defined.

### Questions
1. In Fig. 1, I didn't understand why "high intrinsic rewards can cause premature exploitation of novel regions". Could you further elaborate this sentence?
2. The statement of "Soft Actor-Critic (SAC) is known to be sensitive to the scale of rewards" should also be property justified either with a small experiment or some relevant literatures.
3. How is the co-behavior policy actually trained?
4. Does high stochasity always help with collecting diverse data from the environment? Methods such as novelD may encourage agent to try new actions to reach new states. This would consequently create more diverse data. But after introducing the high-variance co-behaviour policy, the agent may fall back to random policy and thus, collect repeated transitions.
5. Could you justify why RND is used for futher analysis not NovelD?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed a switching mechanism to cooridnate exploration behaviors from two agents, an SAC agent with curiosity-based exploration strategy, and a co-behavior agent with only stochastic policy to draw actions. The proposed algorithm used the novelty intrinsic reward to choose which agent to output action by comaparing with a threshold. Therefore, the proposed algorithm can actively corredinate two different exploration strategies.

### Strengths
The approach presented in this paper is straightforward and easy to implement. It effectively combines two exploration strategies (curiosity-driven exploration and stochastic policy sampling). Experimental results demonstrate improved performance compared to approaches that rely on a single exploration strategy.

### Weaknesses
1. The underlying assumption for the switching criterion is not intuitive. In Equation (4) and the following explanation (from around Line 239 to Line 245), it is stated that for low intrinsic reward regions (i.e., low novelty), KEA switches to curiosity-based exploration to encourage the agent to explore novel areas. Conversely, for high novelty regions, KEA switches to a more stochastic sampling policy. However, it would make sense if it was the other way around, as high intrinsic rewards usually indicate that the area is novel and worth exploring further, making it more logical to apply a strong exploration strategy. Conversely, low intrinsic reward regions (low novelty) might be better suited for stochastic sampling. I recommend that this assumption be supported with more quantitative analysis or additional comparative experiments.

2. The paper conflates curiosity-based exploration and novelty-based intrinsic reward exploration. They are generally considered two distinct approaches for encouraging exploration. While this doesn't undermine the core idea of the paper, clearer distinctions between the two should be made, and the terminology used should be more precise. Including more related representative works would also help.

3. The experimental section can be improved. First, I suggest that the authors include additional baselines that focus specifically on exploration, beyond just the backbone algorithms. Since KEA’s core idea is to combine two exploration strategies, only comparing it against SAC, RND, and NovelD seems more like an ablation study (with only one agent in KEA involved). Representative SOTA works like [1][2][3] would offer better context for evaluating KEA’s performance. Additionally, for KEA itself, it would be valuable to explore other factors, such as how different values for the switching criterion (hyperparameter $\delta$) affect the model's switching behavior, how many times were $\pi^{SAC}$ and $\pi^B$ activated in training, etc.

4. The writing and expression can be improved, for example: (1) Lines 199-203 and Lines 128-130 are repeated verbatim, and (2) paragraphs like Lines 308-315 and Lines 404-411 could be better presented in table format for clearer understanding.

[1] Devidze, Rati, Parameswaran Kamalaruban, and Adish Singla. "Exploration-guided reward shaping for reinforcement learning under sparse rewards." Advances in Neural Information Processing Systems (2022).

[2] Trott, Alexander, et al. "Keeping your distance: Solving sparse reward tasks using self-balancing shaped rewards." Advances in Neural Information Processing Systems (2019).

[3] Hong, Zhang-Wei, et al. "Diversity-driven exploration strategy for deep reinforcement learning." Advances in neural information processing systems (2018).

### Questions
1. In Figure 2 (right-upper part), when the intrinsic reward is higher than a threshold, KEA switches to the co-behavior agent $\pi^B$. While this region is novel (i.e., high intrinsic reward), meaning the agent has not been well trained here. Why does the non-selected agent $\pi^{SAC}$ show an uneven distribution? Generally, for unfamiliar samples (those that have been rarely trained), the policy tends to output a relatively uniform distribution.

2. In Section 3.2, how is UTD set for both $\pi^{SAC}$ and $\pi^B$ respectively? In the UTD experiment, how is the batch size set? Since batch size has an impact on UTD performance, more details would be helpful.

3. The reference paper [4] presents a framework similar to KEA, where an additional agent is trained, and a switching control mechanism is designed. I encourage the authors to discuss this relevant work and make comparisons.

[4] Mguni, David, et al. "Learning to shape rewards using a game of two partners." Proceedings of the AAAI Conference on Artificial Intelligence. 2023.

I would like to increase the score if these concerns are addressed.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces KEA (Keeping Exploration Alive), a framework designed to enhance exploration efficiency in reinforcement learning, specifically for sparse reward environments. The core contribution of the paper is a novel method that combines the Soft Actor-Critic (SAC) approach with a co-behavior agent, designed to improve exploration by proactively coordinating multiple strategies. This coordination is facilitated by a dynamic switching mechanism, which alternates control between the SAC agent, which includes a curiosity-driven exploration method, and the co-behavior agent based on the novelty of the state being explored. This mechanism enables KEA to maintain high stochasticity in high-novelty regions and avoid premature convergence, thereby improving both learning efficiency and robustness.

### Strengths
- The method is easy to understand and contributes to alleviating an important issue in the field of reinforcement learning. 

- Proposes a unique combination of SAC and a co-behavior agent with a dynamic switching mechanism.

- Shows substantial performance gains in benchmarks like 2D navigation tasks and robot control.

- Paper well written and easy to follow

### Weaknesses
- The paper should discuss how the switching mechanism threshold is tuned and its effect on the performance. This appears to be a key mechanism but somehow ablation studies on this component are limited. 

- The evaluation lacks comparison with papers that address a similar issue, such as [Never give up: Learning directed exploration strategies, Badia et al., 2020]. I would suggest adding stronger baselines to validate the proposed method.

- The paper’s proposed method, KEA, is designed specifically with SAC in mind, leveraging SAC’s ability to handle exploration-exploitation trade-offs in off-policy settings. However, the authors do not discuss how KEA might generalize to other off-policy algorithms. 

- Finally, one of my main concerns comes from the switching mechanism, which utilizes a fixed threshold to decide when to switch between the SAC agent with curiosity-driven exploration and the co-behavior agent. While this approach appears effective in simulations, using a fixed threshold might lack the flexibility needed for different environments or tasks, especially those with varying reward sparsity or novelty patterns. This inflexibility could limit KEA’s adaptability, potentially leading to less effective exploration in more complex or dynamic scenarios

### Questions
-How sensitive is the model to the switching threshold parameter? 

- Was this threshold hand-tuned or optimized in some manner?

- What is the computational impact of using two agents in terms of memory and processing requirements?

- Does the model perform well in highly dynamic environments with frequently changing goals?

- Can the method applied to other RL algorithms, beyond SAC?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes Keeping Exploration Alive (KEA), an algorithm that switches between exploration policies depending on the value of the intrinsic reward in order to reduce sub-optimal performance due to explore-exploit tradeoffs.

### Strengths
This paper is generally well written, and tackles a valid problem in RL, how to gracefully handle the explore-exploit tradeoff. The core results are clear, and the resulting algorithm seems to outperform the baseline by a significant amount on the evaluation task suite. The approach is, to my knowledge, original, though other methods have tackled this problem before in various ways.

### Weaknesses
The core issue I have with this paper is that there's very little exploration of the proposed algorithm beyond claims of superior performance. The core concept is simple (not a bad thing), but there's no real consideration given to what makes it good or useful. Seeing as the core contribution of this work is very simple, it should be straightforward to do deeper analysis and benchmarking, but this paper doesn't do much of that, and largely just concludes "it works."

As a result, I can't help but feel this paper is just sort of insubstantial and doesn't contribute that much to the field. It proposes an idea that is novel, but straightforward as an extension of past work, and doesn't do anything with the idea beyond show that it outperforms the baseline. If accepted as is, I'm not sure how much this will actually move the topic of exploration forward.

As such, I'm inclined to recommend rejection, but would like to see an expanded version of this work submitted to a future conference, because I do like the core idea of dynamically switching between explore and exploit policies for off-policy RL.

### Questions
Some additional questions and suggestions:

-In figure 3, the SAC line seems to be flat? I am surprised this task cannot be learned at all by SAC, and that the best method only reaches a ~60% success rate. Perhaps the degree of sparsity is not clear to me from the task description. Perhaps that sparsity could be elaborated on somehow?

-I suggest adjusting page spacing so that the critical algorithms 3 and 4 are on the same page, since they are connected and key to this paper's contribution

-Figure 4 feels like supplemental material to me, since it's basically just validating hyperparameter choices.

-The caption on figure 6 should say what the shaded regions represent.

-Also in figure 6, it looks like the algorithms haven't clearly converged on most of the three tasks. 500k environment steps is not a very large number, perhaps it would be worthwhile to run these experiments for the commonly-used 1 million steps to make sure there's separation at convergence?

-Why is it valuable to vary update to data ratios so much? Is there a reason this is a key hyperparameter for this algorithm? If so that should be explained in the paper. It's not clear how this relates to explore-exploit tradeoffs

-For that matter, why is sigma (which appears to be a key hyperparameter) not varied? It seems like performance should be sensitive to that parameter since it controls algorithm switching (the core contribution)

-Further, when and how much does KEA actually switch between the policies? This is the core phenomenon of this paper, so surely there should be some confirmation it's actually switching regularly and in a structured way, no?

### Soundness
2

### Presentation
3

### Contribution
2
