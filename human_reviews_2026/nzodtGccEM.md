# Don't Just Fine-tune the Agent, Tune the Environment

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large Language Model (LLM) agents show great promise for complex multi-turn tool-use tasks, but their development is often hampered by the extreme scarcity of high-quality training data. Supervised fine-tuning (SFT) on synthetic data leads to overfitting, whereas standard reinforcement learning (RL) struggles with a critical cold-start problem and training instability. To address these challenges, we introduce $\textbf{Environment Tuning}$, a novel training paradigm that enables agents to learn complex behaviors directly from problem instances without relying on pre-collected expert trajectories. $\textbf{Environment Tuning}$ orchestrates this learning process through a structured curriculum, actionable environment augmentation that provides corrective feedback, and fine-grained progress rewards to ensure stable and efficient exploration. Using only 400 problem instances from Berkeley Function-Calling Leaderboard (BFCL) benchmark, our method not only achieves competitive in-distribution performance against strong baselines but also demonstrates superior out-of-distribution generalization, overcoming the performance collapse common to SFT-based approaches. Our work presents a paradigm shift from supervised fine-tuning on static trajectories to dynamic, environment-based exploration, paving the way for training more robust and data-efficient agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new method named environment tuning, which tries to design different difficulty-level environments (by actionable environment augmentation techniques) to train the model in different stages. By using such curriculum learning, the model can learn to handle ambiguity, recognize functional gaps, and perform information retrieval from noisy contexts.
The experiments show that the model can improve their performance on the benchmark tasks within every stage. It also improves the importance of actionable environment augmentation.

### Strengths
1. The writing is clear and easy to understand.

2. Considering improving the environment instead of improving the algorithm to make the model better is an interesting idea. I just believe the data is more important than the algorithm in agent RL, so considering the environment is an important topic.

3. I believe this project is a good realization of the idea of curriculum learning, the four stage is clear and makesence

4. The experiments are reasonable and can prove the effectiveness of actionable environment augmentation and .

### Weaknesses
1. The reward design may have some flaws. It just calculates the average reward of each step and uses GRPO to train the model, so the bad action at the end of the trajectory will not be punished (since the final reward is high). This may cause a credit assignment problem.

2. It seems that this work is trying to distill preliminary knowledge in the environment to the model for stable RL training. However, such preliminary knowledge may come from the annotator of the benchmark, in other words, the author should prove their OOD performance does not come from telling the bias of BFCL's author. So, the ablation study should contain more benchmark tasks.

3. The analysis can only prove that such a method can improve the OOD performance, but it doesn't figure out why it can improve the OOD performance. (refine ability? Better instruction following? other?) Comparing stage 1-4's model on OOD task is more important. (Also see Questions 5)

### Questions
1. In Table 2 the 23.87 should be orange.

2. The transit point (Remark 3.1) is not clear; it is not easy to understand how to find it (even in Appendix D).

3. The topic in BFCL V3 is not clear in this passage, for example, "Is there a web search task in BFCL V3?" If yes, the web search should not be OOD.

4. Why is the progress reward lower than the final score in Figure 4.

5. Does the OOD improvement really come from the model's refinement ability? Maybe an OOD case is important. Also, you can train the model on the OOD task and show the scaling line to figure out whether it can scale in OOD tasks

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
This paper introduces "Environment Tuning," a novel training paradigm designed to equip Large Language Model (LLM) agents for complex, multi-turn tool-use tasks, particularly under conditions of extreme data scarcity. Its core innovation involves a shift from static, trajectory-based imitation learning towards a dynamic, environment-based exploration strategy. This allows agents to learn directly from problem instances without dependence on pre-collected expert demonstrations. The method demonstrates strong performance on the BFCL-V3 benchmark using only 400 problem instances, significantly enhancing the capabilities of base models and exhibiting robust out-of-distribution generalization—a known weakness of traditional supervised fine-tuning (SFT) approaches.

### Strengths
The paper presents Environment Tuning, an innovative framework that reorients LLM agent training from static trajectory imitation to a dynamic, environment-driven learning process. By effectively integrating a structured curriculum, actionable environment augmentation, and fine-grained progress rewards, the method directly tackles the critical challenges of data scarcity and training instability in multi-turn tool-use scenarios. The experimental results on BFCL and ACEBench are compelling, demonstrating substantial improvements over both supervised fine-tuning and standard reinforcement learning baselines. The manuscript is well-written, with a strong motivational foundation. The conceptual shift of "tuning the environment" itself constitutes a genuine and noteworthy contribution to the field.

### Weaknesses
The primary weaknesses relate to empirical rigor and the scope of validation. The "actionable environment augmentation" mechanism, while central to the method, is evaluated qualitatively; a more systematic, quantitative analysis of its specific impact is needed. The claims regarding generalization, though promising, would be more persuasive if supported by results from a wider array of diverse environments. The analysis would benefit from reporting statistical significance and performance variance. Furthermore, the method's current reliance on controllable simulation environments may pose challenges for real-world deployment. Finally, the theoretical justification for the specific four-stage curriculum design could be more thoroughly elaborated.

### Questions
Regarding Environment Augmentation: The "Actionable Environment Augmentation" is a key component of your method's novelty. Could you please elaborate on its implementation?
1. How are the corrective hints generated—are they based on heuristic rules, programmatic templates, or perhaps an LLM-based generator?
2. What measures are implemented to ensure these hints provide guidance without leaking the correct solution or overly constraining the agent's exploration?
Regarding the Training Curriculum: The proposed four-stage curriculum is empirically sound, but its design appears somewhat heuristic.
1. Could you elaborate on the rationale for selecting these particular four stages? How sensitive are the final results to the ordering of stages or the specific criteria used for transitioning between them?
2. Were any alternative or adaptive curriculum scheduling strategies explored during your development? If so, please summarize those findings.

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
4

### Summary
This paper introduces Environment Tuning, a RL framework designed to improve LLM agents’ performance in multi-turn tool-use tasks. Instead of only fine-tuning the model, the authors propose systematically modifying the environment to provide structured, corrective, and reward-rich feedback. The method combines a structured curriculum, actionable environment augmentation, and fine-grained progress rewards to guide learning efficiently. Experiments on BFCL and ACEBench demonstrate notable gains in both in-distribution and out-of-distribution performance, highlighting the approach’s effectiveness and generalization potential.

### Strengths
1. The core idea of enhancing agent training by incorporating environment-level variations is creative and interesting.
2. The method is well-designed and clearly presented, with intuitive reasoning and an easy-to-follow structure.
3. Experimental results across multiple datasets convincingly validate the approach’s effectiveness and robustness.

### Weaknesses
1. Although the concept of Environment Tuning is intriguing, the term itself may be somewhat misleading. In the proposed framework, environment changes are predefined and discrete, rather than dynamically adapted based on the agent's learning progress. Thus, the word “tuning” might suggest a level of adaptivity that is not actually present. Terms like environment augmentation or dynamic environment design could better capture the nature of the approach.
2. The experimental section lacks direct comparison with other recent multi-turn tool-use approaches that also use RL or hybrid training. While related works are discussed in Section 2, their omission from the experimental benchmark makes it difficult to fully assess the method's relative advantage over competing paradigms in similar problem settings.
3. The paper’s progress reward relies on ground-truth evaluation signals provided by the benchmark datasets. This raises the question of how the method would generalize to scenarios without explicit ground-truth feedback. Moreover, since these signals are dataset-provided, it would be valuable to clarify whether existing methods truly lack access to or use of such supervision—otherwise, the claimed novelty of the progress reward may be overstated.

### Questions
Please check my comments in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work aims to tackle the problem of limited expert training data, simultaneously overcoming overfitting in SFT and cold start and instability in RL by designing a structured curriculum with training environment. The environment provides corrective feedback and dense reward signals, achieving not only better in distribution performance but also better generalization with less data.

### Strengths
1. Proposed a clear and progressive roadmap that tackles each challenge of sparse reward, long horizon, and instability. 

2. Novelty in tuning the environment feedback instead of tuning the model prompt. 

3. Extensive ablations were performed to show the importance of each stage and enhancement.

### Weaknesses
1. In real-world applications, environments are oftentimes not accessible or modifiable. Tuning environmental feedback will be expensive or even impossible. 

2. Revealing internal tool constraints (Section 3.3) requires prior knowledge from humans, which is expensive and involves heavy reasoning burdens for human experts to provide meaningful explanations, especially in a multiturn setup where more complicated and chained reasons for error would appear. 

3. Discovering inter-tool dependencies via exploration (Section 3.3): The paper proposes to embed inter-tool dependencies in detailed environment feedback. This is confusing to be considered as exploration since the model is not actively searching for broader or more informative states in the environment.

### Questions
1. OOD Generalization performance is only compared to SFT. It would be interesting to demonstrate the proposed method’s generalization ability compared to vanilla RL methods without dense reward and detailed feedback. 


2. In Figure 5b, why did the model's performance not drop when transitioning to the next stage, where the model was only trained in easier stages but was exposed to new and more challenging tasks? 


3. How easy is the proposed method to apply to a task in an unseen field? Do practitioners need to provide detailed environment error feedback and redesign the dense reward component or other modules? Discussing how much manual engineering is needed for a new task will be helpful for evaluating the method’s generalization to unseen tasks.

### Soundness
3

### Presentation
3

### Contribution
2
