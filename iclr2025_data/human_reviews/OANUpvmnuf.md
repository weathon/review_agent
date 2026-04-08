## Human Reviewer 1

### Summary
This paper study utilizing LLMs to improve exploration for multi-agent RL setting. The key motivation is that LLMs, pre-trained on general knowledge, can extract task-relevant guidance for improving exploration. The proposed method,  LEMAE, identifies key states by prompting LLMs, and encourages the agents to move forward key states using a memory tree. Experiment results on SMAC and MPE show that LEMAE outperforms various RL methods in terms of improvement speed and final performance.

### Strengths
Developing an effective way to leverage the linguistic knowledge in LLMs for RL is a promising approach. This paper makes a step forward by proposing the idea of 'discriminating key state' and tree search. I believe this could be a valuable supplement to RL community with the integration of large foundation models. 

In addition, I list some other pros as follow:

1. Overall the paper is well-organized and well-established. It is easy to follow the motivation and method design. 
2. The proposed method seems a general method that can be utilized for wide RL problems, e.g., single agent setting. It would be interesting to see the results on single agent RL tasks.
3. The experiment is comprehensive, including wide ranges of baselines and evaluation tasks.

### Weaknesses
It seems that LEMAE is quite limited to specific tasks, whose state is simple to describe. A key point of LEMAE is to address the symbolic state representation issue for LLMs. However, for tasks with complex state representation, such as robotics control, it is hard to explain a specific state value. LEMAE is hard to be applied in such domain, which is an important sub-filed of RL tasks. 

Besides, LEMAE is proposed for multi-agent RL problem, but it lacks a consideration for solve MARL training, e.g., cooperation between partners. Taking this into consideration would make the method more reliable and efficient.

There are also some more suggestions on the writing:
1. I feel that the introduction of the method in Section 1 is a simple re-writing of the abstract. 
2. I do not find the importance of putting "CHOICES ARE MORE IMPORTANT THAN EFFORTS" in the title. I think it is not the main highlight of this paper. Maybe it is better to consider from the perspective of LLMs.

### Questions
Refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes to improve exploration in multi-agent environments by generating key states with the aid of LLMs and then use hindsight experience replay to generate auxiliary rewards. Specifically, task descriptions as well as task-relevant knowledge is converted to text and then fed to an LLM that is tasked to generate a function (e.g., written in Python) that takes in a state and returns whether this is a goal state. During a typical online reinforcement learning procedure, each experienced state is examined by the key state checking function, and if it is masked as a key state, hindsight experience replay will be used to add auxiliary rewards to prior states in the same trajectory/episode. The proposed algorithm was tested on the Multiple-Particle Environment and the StarCraft Multi-Agent Challenge.

### Strengths
Hindsight experience replay is a very effective auxiliary reward generation methods that have shown successes in many RL environments. However, its main limitation is the difficulty to correctly “identify” goal states. This paper proposes to use LLM to generate such “key states”.

### Weaknesses
- Connection with other LLM-based reward shaping algorithms. While the paper discusses many LLM-based reward generation algorithms in their related works, better and more comprehensive comparison with these methods are needed in the experiment sections to better demonstrate the effectiveness of the proposed algorithm (e.g., Xu et al., 2023 and Liu et al., 2023 cited in the paper). Specifically, as mentioned in the paper, specific prompts for description the task/state space/action space are used for each environment independently. Whether the same set of prompts are used in relevant baselines? If the prompt is good enough for the LLM to generate key state description functions, it seems natural that with similar prompts the LLM can also directly generate good auxiliary rewards and/or reward functions, which are done by many existing LLM-based reward generation algorithms. It would be very helpful to clearly describe the settings and make sure that the relevant baselines are also exposed to the same prompts. 

- There is only one LLM-based reward generation baseline adopted. Are there particular reasons for only including this LLM-based baseline algorithm? Since the proposed method leverages knowledge from LLMs, it would be more natural to include more LLM-based baselines.

- Generalizability of the LLM-based key-state detector to other environments. The adopted environments have discrete and small state/action spaces, which makes it easier to create key-state detectors. Also, the environments are relatively simple to describe with natural language. I wonder how this method performs in environments with more complex state space/action space/transition dynamics, for example in robotics tasks?

- Minor: relation with Go-Explore [1]. This key states memory tree seems to be related to go-explore. Could the authors discuss their similarities and differences?


[1] Ecoffet, Adrien, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, and Jeff Clune. "Go-explore: a new approach for hard-exploration problems." arXiv preprint arXiv:1901.10995 (2019).

### Questions
- How does the proposed method compare with other LLM-based reward shaping algorithms?
- Could the authors clarify whether the same prompts were used across all LLM-based methods?
- How well can the proposed algorithm scale to environments with complex state space/action space/transition dynamics?
- It seems that the proposed method can be applied to both single-agent and multi-agent environments. Is there a particular reason to use it primarily in multi-agent environments?
- How does the key states memory tree compare to the memory structure used in Go-Explore?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes to channel informative task-relevant guidance from a knowledgeable LLM for efficient Multi-Agent Exploration. It uses LLM to localize key states and use hindsight reward for exploration (also use memory tree to orginaze the key states). The results show that the proposed methods achieve better results than most "traditional" deep MARL methods.

### Strengths
1. important topic and novel methods: Efficient multi-agent exploration is important for MAS, the proposed methods leverage LLM for this goal, which is novel as the reviewer knows.  

2. memory tree to orginaze the key states for guided exploration (instead of greedy redundant exploration) is interesting and novel.  

3. The paper is clear and easy to understand.

4. The results show that the proposed methods achieve better results than most baseline methods.

### Weaknesses
1. Proposition 4.1 is one of hte most important motivation for task-relevant information based exploration. But the setting is the one-dimensional asymmetric random walk problem. What is means for a more realistic scenarios (e.g., MPE, SMAC)?  Could the authors discuss how the insights from Proposition 4.1 might generalize or relate to more complex environments (higher-dimensional state spaces with multiple agents) like MPE and SMAC?

2. The key states localization is the most important for the proposed methods. But current description (i..e, Section 4.2)  is not clear about the method detaills. (besides discrimination by LLM rather than generation; m discriminator functions). Could the authors discuss the process of generating the m discriminator functions (or the detailed prompts), how the number m is determined, or how the discriminator functions are applied to states in practice. And how to make sure the LLM-based discriminators can work well as the designed intention?

3. The baselines compared seems to be general MARL methods. It is better to compare MA-exploration methods and methods that also combine LLM with MARL (e.g., [1,2,3]).  Could the authors explain why these specific baselines were not included, and could you either add comparisons to these methods or discuss how their approach differs conceptually from these related works. This would help contextualize your contribution within the most relevant recent literature.

[1] Learning When to Explore in Multi-Agent Reinforcement Learning. TCYB 2023.  (which learns to when to explore)
[2] Controlling Large Language Model-based Agents for Large-Scale Decision-Making: An Actor-Critic Approach. (which uses two LLMs for exploration-exploitation tradeoff)

### Questions
see weakness.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper presents an approach for using LLMs to identify key states for multi-agent reinforcement learning agents to more efficiently explore in procedurally generated environments. Given a trajectory, the LLM generates key states, which are then organized in a key states memory tree, and used in a hindsight experience replay-like approach.

### Strengths
- I believe the paper provides sufficient detail in the appendix to reproduce it.
- The direction of the paper, addressing hard exploration in multi-agent systems, is an important one.

### Weaknesses
## Major
A) The main results of the paper center around The StarCraft Multi-Agent Challenge (SMACv1) by Samvelyan et al. (2019). However, it is well established by now that SMACv1 is saturated and too simple. The authors should use the 2022 released SMACv2 [1] for their experiments. From the SMACv2 paper: “after years of sustained improvement on SMAC, algorithms now achieve near-perfect performance. In this work, we conduct new analysis demonstrating that SMAC lacks the stochasticity and partial observability to require complex *closed-loop* policies. In particular, we show that an *open-loop* policy conditioned only on the timestep can achieve non-trivial win rates for many SMAC scenarios.”

B) Using language observations or LLMs to guide exploration of RL agents for hard exploration problems has been extensively studied in the single-agent setup. The authors have not covered important very related methods (e.g. [2-4]) in this space. In particular I would expect Motif [3] and OMNI [4] to be tried out as a baseline in the multi-agent RL (MARL) setting. In fact, I would go as far and say that the proposed method is not MARL-specific at all, so the authors really need to show that it provides benefits over existing exploration methods in the single agent setting.

C) Related to B, it is even unclear to me whether explicit key states are necessary for exploration in the MARL environments used here. I would like to see a comparison to MARL with general intrinsic reward mechanisms such as E3B [5] to answer the question whether key states significantly improve exploration over generic intrinsic reward methods.

D) The authors claim that “KSMT helps avoid redundant exploration throughout the state space, particularly beneficial in more complicated real-world scenarios” despite that no experiments on such real-world problems are provided. Either tone it down or provide additional empirical evidence in more-real world environments. I would be very interested in the latter, in particular seeing experiments on environments that do not have a symbolic observation space.

[1] Ellis, B., Moalla, S., Samvelyan, M., Sun, M., Mahajan, A., Foerster, J. N., & Whiteson, S. (2022). SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning. arXiv. https://doi.org/10.48550/arXiv.2212.07489

[2] Mu, J., Zhong, V., Raileanu, R., Jiang, M., Goodman, N., Rocktäschel, T., & Grefenstette, E. (2022). Improving Intrinsic Exploration with Language Abstractions. arXiv. https://doi.org/10.48550/arXiv.2202.08938

[3] Klissarov, M., D’Oro, P., Sodhani, S., Raileanu, R., Bacon, P.-L., Vincent, P., … Henaff, M. (2023). Motif: Intrinsic Motivation from Artificial Intelligence Feedback. arXiv. https://doi.org/10.48550/arXiv.2310.00166

[4] Zhang, J., Lehman, J., Stanley, K., & Clune, J. (2023). OMNI: Open-endedness via Models of human Notions of Interestingness. ICLR 2024.

[5] Henaff, M., Raileanu, R., Jiang, M., & Rocktäschel, T. (2023). Exploration via Elliptical Episodic Bonuses. arXiv. https://doi.org/10.48550/arXiv.2210.05805


## Minor
- LLMs instead of LLM (e.g. in the title); also "Efforts" -> "Effort" in the title
- Be more precise with what you mean by “redundant exploration”
- Use more formal language (e.g. not “brand-new” in line 089)
- L160: “to clarity” -> “to clarify”
- Figure 2 is somewhat unclear to me, in particular the LLM-mapping from Prompt to Response. I particularly find it hard to map what’s formalized in 4.2 (e.g. L209 to L212) to what’s shown in Figure 2. Also, what feedback flows directly back from response to prompt?

### Questions
- L190: What do you mean by “the initial policy is asymmetric”?
- L247: What’s the rationale for using Manhattan Distance instead of, for example, cosine similarity?

What the authors would have to demonstrate to see an improved rating from me:
- Redo experiments on SMACv2 (A)
- Add baselines Motif and OMNI, requiring them to be evaluated on MARL (B)
- Add E3B baseline, again integrating that into the MARL setup, or, alternatively (since the proposed method is not MARL-specific) run the proposed method on single-agent procedurally generated environments. (C) 
- Demonstrate that the proposed method also works for non-symbolic observations (D)

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
5