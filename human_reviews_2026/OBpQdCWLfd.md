# ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) - an automata-based formalism for reward specification - are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications;  (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a framework using foundation models (FMs) to automate Reward Machine (RM) generation from natural language specifications for reinforcement learning (RL). The core idea is to leverage FMs to produce not only the RM's automata structure but also the executable labeling functions to address sparse-reward, long-horizon embodied control tasks. The framework also proposes conditioning the RL policy on language embeddings of RM states to enable compositional generalization. The proposed method demonstrates its effectiveness by solving several challenging benchmark tasks (MiniGrid, Craftium, MetaWorld) intractable for standard RL baselines.

### Strengths
1. Effective FMs and RM pipeline with strong empirical results on sparse, long-horizon RL benchmarks (MiniGrid, Craftium, MetaWorld) where baselines (ICM, PPO, SAC) fail.
2. By generating an explicit RM automaton, the framework offers a degree of interpretability and debuggability compared to opaque, end-to-end FM reward signals.
3. Robust automated labeling code gen through LLM generator-critic loop for plan grounding.
4. The use of language embeddings for the policy conditioning enables zero-shot generalization on a compositional task.

### Weaknesses
1. The grounding mechanism (using labeling functions) relies on access to the simulator's high-level, symbolic state representation (env.carrying, obj.type, obj.is_open). This might limit the framework's direct applicability to real-world scenarios or learning from raw pixels/sensory data, a common limitation but still noteworthy.

2. While LLMs-driven RM generation and labeling functions shows novelty in automation, the strong performance on non-Markovian tasks feels incremental, as RMs' efficacy in such settings is well-established (e.g., [1, 2, 3]). Expected results limit surprise, though the proposed framework remains valuable for practitioners.

[1] Rodrigo Toro Icarte, Toryn Q Klassen, Richard Valenzano, and Sheila A McIlraith. Reward machines:
Exploiting reward function structure in reinforcement learning. Journal of Artificial Intelligence
Research, 73:173–208, 2022.

[2] Rodrigo Toro Icarte, Toryn Klassen, Richard Valenzano, and Sheila McIlraith. Using reward machines
for high-level task specification and decomposition in reinforcement learning. In International
Conference on Machine Learning, pp. 2107–2116. PMLR, 2018.

[3] Andrew Li, Zizhao Chen, Toryn Klassen, Pashootan Vaezipoor, Rodrigo Toro Icarte, and Sheila
McIlraith. Reward machines for deep rl in noisy and uncertain environments. Advances in Neural
Information Processing Systems, 37:110341–110368, 2024.

### Questions
1. How sensitive is the framework to errors or inefficiencies in the FM-generated labeling functions, especially for more complex environments? What is the typical failure rate, and how much manual correction was needed during experiments?

2. What are the concrete next steps or necessary innovations required to extend ARM-FM to environments where only raw observations (e.g., pixels) are available, requiring the labeling function predicates themselves to be learned?

3. While RMs offer structure, manually designing them becomes intractable for complex tasks. Does the FM-based generation approach scale effectively? Were there task complexities or types where the FM struggled significantly to produce a useful RM structure, even with the critic loop?

4. Regarding the zero-shot generalization results (e.x., Figure 10), success relies on the new task's sub-goal embeddings being 'close to an embedding seen during training.' How can we quantify this 'closeness' in the embedding space? For example, does a specific cosine similarity threshold between a new sub-goal embedding and the closest training sub-goal embedding correlate with successful transfer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a novel methodology for tackling complex, long-horizon tasks in reinforcement learning (RL). At the core of the approach is the use of a foundational model (FM) to produce a high-level task decomposition, which is formalized as a Language-Aligned Reward Machine (LARM). In this framework, each state of the reward machine is annotated with a natural language description of the corresponding subgoal, enabling the agent to condition its policy on these instructions. This alignment facilitates goal-directed behavior and supports zero-shot generalization to previously unseen tasks.

As the agent transitions through the reward machine states, it receives structured intermediate rewards that guide learning and enhance sample efficiency. Notably, the FM also produces executable code to define the labeling function, which determines when a subtask has been completed. In doing so, the FM provides all essential components—task decomposition, subgoal descriptions, intermediate rewards, and labeling logic—allowing the agent to leverage LARM in novel environments with minimal human supervision.

Experimental results across a diverse set of benchmarks demonstrate the effectiveness of the proposed approach, outperforming strong baselines such as RL with intrinsic motivation and LLM agents.

### Strengths
I thoroughly enjoyed reading this paper. It presents a compelling case for the use of foundational models (FMs) and reward machines (RMs) as effective tools for communicating task structure to reinforcement learning (RL) agents. Particularly noteworthy is the FM’s ability to autonomously generate all necessary components—from the RM structure to the labeling function—based solely on a natural language task description and an image of the environment. The idea of associating natural language instructions with each RM state is especially interesting, as it enables zero-shot generalization and, depending on the quality of the subgoal descriptions, can even lead to near-optimal task execution despite the decomposition into potentially myopic subtasks.

The paper also offers a broad and thorough empirical evaluation, spanning environments from 2D grid worlds to complex 3D settings in robotics and Minecraft. Across these diverse domains, the proposed method consistently demonstrates strong performance. The authors further validate their approach by testing generalization capabilities and conducting a sensitivity analysis, highlighting the importance of both intermediate rewards and state embeddings to the agent’s success.

Finally, I believe this work opens up several promising avenues for future research, some of which I have outlined in the weaknesses section of my review.

### Weaknesses
While I found the paper insightful and well-executed overall, there are some areas where further clarification or experimentation would strengthen its contributions. 

First, the role of human verification in Figure 3 is underexplored. The only mention of this “human-in-the-loop” process appears in the final paragraph of the conclusion. From what I understand, the human reviewer inspects the LARM generated by the FM and may reject it. However, it remains unclear what happens after a rejection: Does the human refine the prompt, or simply initiate additional rounds of self-improvement? Moreover, how frequently were LARMs rejected or refined during the experiments? What were the typical reasons for rejection? For instance, were there cases where bugs in the labeling function or poorly designed intermediate rewards prevented task completion? Clarifying whether such issues arose—and how they were handled—would help contextualize the robustness of the method. Alternatively, if no human verification was performed during experiments, that should be explicitly stated.

Second, in Section 3.1, the paper attributes the superior performance of DQN+RM primarily to the intermediate rewards provided by the reward machine. While this is a plausible explanation, I believe the state embeddings may also play a significant role. Since the environments are partially observable, the RM state effectively acts as a form of memory, allowing the agent to retain contextual information across time steps. In contrast, the baseline agents are memoryless, which may explain their underperformance—not necessarily due to the absence of intermediate rewards, but because they lack the capacity to maintain relevant state information. To better isolate the contribution of intermediate rewards, it would have been helpful to include the “No RM rewards” ablation from Section 3.4 in Figures 6 and 7. Additionally, comparing against an LSTM-based agent would help determine whether memory alone accounts for the observed performance gap. The same considerations apply to the experiments discussed in Section 3.2.

Third, while the zero-shot generalization result is promising, it currently feels somewhat anecdotal. The paper presents a single instance in which the agent successfully generalizes to a novel task. A more systematic evaluation—spanning a broader range of tasks—would strengthen the case for the generalization capabilities of LARMs.

Finally, a discussion on optimality would be valuable. The FM can describe RM states using either myopic or non-myopic natural language instructions. For example, a myopic instruction like “Get the key” may lead the agent to complete subtasks sequentially without considering long-term consequences, potentially resulting in suboptimal behavior (e.g., choosing a key that doesn’t lead to the door). In contrast, a non-myopic instruction such as “Get the key and then go to the door” could guide the agent toward more globally optimal behavior. It would be interesting to explore whether prompting the FM to produce non-myopic subgoal descriptions could improve long-term performance.

### Questions
1. Could you elaborate on the specific role and responsibilities of the human verifier during the experimental evaluation?
2. How frequently was the LARM generated by the FM deemed unsatisfactory by the human verifier?
3. What were the most common issues or failure modes observed in the LARM generation process by the FM?
4. When an LARM was rejected by the human verifier, what steps were taken next? Was the prompt revised or refined, and if so, how was this process carried out?
5. Does the labeling function rely solely on information observable by the agent, or does it access privileged information from the environment's internal state? If it uses privileged information, this could pose challenges for real-world deployment—particularly in sim-to-real scenarios—since such information may not be available outside the simulation.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ARM-FM, which uses a large FM to automatically generate RMs from natural language task descriptions. The generated LARM includes state-level natural language descriptions and labeling functions. Policies are trained on the product MDP of the environment and RM, conditioned on the embedding of the current RM state instruction. Experiments on MiniGrid, Craftium (3D), and other benchmarks demonstrate improved zero-shot generalization.

### Strengths
+ I believe the main strength of the paper is the use of language embeddings of RM state descriptions for conditioning the policy. 
+ Using FMs to automate RM construction from natural language specifications is useful since manual RM design requires expertise.
+ The experiments and visualizations across multiple environments are commendable.

### Weaknesses
- My main concern is the limited novelty. RL over RMs and product MDPs is well established. The main addition here is automating RM generation with an FM and embedding the RM states, which is conceptually incremental in my view for ICLR.
- Another concern I have is that relation to prior work is not sufficiently clear. The distinction needs to be well articulated. This includes a large body of work on FM-guided RL frameworks such as ReAct, SayCan, Voyager, Eureka etc and the list goes on. 
- Most tasks considered seem to involve small domain variation rather than truly different domains. It remains unclear whether the learned embeddings generalize under larger domain or modality gaps. So is the paper over claiming generalization? 
- There are missing comparisons in evaluation with recent vision language or FM-guided RL frameworks 
- The human verification weakens the claim of full automation.
- Reproducibility may be an issue.

### Questions
1) Precisely what is novel beyond FM-generated RM structure/code and conditioning on language embeddings of RM states? What core technical insight would not be obvious from prior RM + product-MDP and FM-guided RL work?
2) Can you provide a comparison table clarifying how your pipeline differs from closest work that already uses LMs to synthesize automata/RMs/LTL specs, and from FM-guided RL?
3) Is there a theoretical claim you can formalize?
4) What fraction of tasks required human verification/edits? 
5) How are RM-state language embeddings obtained?
6) How large are the domain gaps actually used (measured by a concrete metric)? Can you report performance as a function of controlled gap magnitude and type?
7) Can you compare against stronger FM-guided baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the problem of synthesizing reward machines from natural language. The aim then is to use the synthesized reward functions for single and possibly multi-task RL. This works in three parts. First a reward machine language specification is created. This describes the underlying finite state transducer structure of the reward machine, e.g. the states, transitions, alphabet. The second part describes the grounding functions for each input symbol. Finally each state is tagged with a natural language description describing the abstract state of the agent in it's policy. The abstract description is additionally encoded as a text embedding. A key contribution of the work is to propose using this text embedding as the representation of the automata state to the RL-agent. The insight is that this provides two features:

1. If the state is sufficiently descriptive, if provides strong clues as to what the agent should do at that state, e.g., go collect the key for a door.
2. If the state descriptions semantically (or exactly) align, then the policy can re-use experience from other tasks.

Empirical results show improvements against sparse feedback baselines and signs of multi-task generalization.

### Strengths
1. Formal representations of tasks such as reward machines offer a powerful mechanism to unambiguously/explicitly provide additional structure and memory to an RL agent. Being able to easily generate such representations is a worth while research program.
1. The exact method used in this paper is to my knowledge novel. While the translation of natural language to an automaton has been explored in other works, this work adds an interesting contribution of a natural language embedding to each state.

### Weaknesses
1. The approach for constructing the automaton seems rather ad-hoc and relies somewhat blindly on the LLM's ability to provide a faithful translation. This is in contrast to techniques which seek to reduce the learning of the automaton structure to classic learning algorithms based on membership oracles [1].
2. The language embeddings are the only source of automata structure embedding. it's unclear how reliably the automata's structure is actually relayed. While likely enough for short horizon tasks, the concern is that for complicated automata with many loops, explaining what the state is in natural language is non-trivial. On the one hand, compared to domain agnostic embeddings, e.g., RAD embeddings [2], these provide vital domain specific clues for what the policy should do. I would be curious what would happen when combining such approaches.
3. Perhaps the largest issue is it's unclear how to test for alignment in the work aside from simply optimizing the reward and checking the outcome. The grounding function in particular seems non-trivial to verify without a large amount of human in the loop work.

[1] L*LM: Learning Automata from Demonstrations, Examples, and Natural Language
[2] Compositional automata embeddings for goal-conditioned reinforcement learning

### Questions
Have you explored compositional tasks which aren't sequences, but have conditional logic? E.g., If x occurs, then you need to do y. My concern is the state embedding sharing becomes less useful since the local and global automata structure can non-trivially change when adding a task.

### Soundness
3

### Presentation
3

### Contribution
2
