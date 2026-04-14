# Policy Optimization under Imperfect Human Interactions with Agent-Gated Shared Autonomy

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We introduce AGSA, an Agent-Gated Shared Autonomy framework that learns from high-level human feedback to tackle the challenges of reward-free training, safe exploration, and imperfect low-level human control. Recent human-in-the loop learning methods enable human participants to intervene a learning agent’s control and provide online demonstrations. Nonetheless, these methods rely heavily on perfect human interactions, including accurate human-monitored intervention decisions and near-optimal human demonstrations. AGSA employs a dedicated gating agent to determine when to switch control, thereby reducing the need of constant human monitoring. To obtain a precise and foreseeable gating agent, AGSA trains a long-term gating value function from human evaluative feedback on the gating agent’s intervention requests and preference feedback on pairs of human intervention trajectories. Instead of relying on potentially suboptimal human demonstrations, the learning agent is trained using control-switching signals from the gating agent. We provide theoretical insights on performance bounds that respectively describe the ability of the two agents. Experiments are conducted with both simulated and real human participants at different skill levels in challenging continuous control environments. Comparative results highlight that AGSA achieves significant improvements over previous human-in-the-loop learning methods in terms of training safety, policy performance, and user-friendliness.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper presents AGSA, a shared autonomy framework for learning from human feedback and gating when the robot passes control the human expert. The agent regards states that require intervention as undesirable, assigning negative rewards to state-action pairs that precede human intervention. AGSA trains a gating agent, and a learning agent. The gating agent requires human teachers to provide a binary signal on whether the current state is worth intervention, control for T steps when intervening, and provide a preference signal for whether the current segment is better than the previous segment. The learning agent is trained to optimize avoiding states that precede control switching to the human agent. The simulated results show that AGSA achieves high performance compared to baselines across multiple task and simulated performance levels. The ablation results indicate that the components are necessary for overall performance.

### Strengths
The manuscript is clear and the approach is well-described. I appreciate the motivating example, as it offers intuition about the limitations of ensemble-based uncertainty quantification and failure detection. While the experiments could use more detail about the quantity and type of data each method had access to, the results seem strong and indicate good performance of AGSA.

### Weaknesses
The paper presents a motivating example to demonstrate the limitations of ensemble-based uncertainty quantification and failure detection. However, it lacks an evaluation of how effectively the AGSA gating agent addresses these specific failure cases. The framework assumes a high level of expertise and understanding from the human operator, including familiarity with the robot's policy and decision-making criteria for interventions. Subsequently, these challenges are not addressed by the user study, which does not examine statistical testing nor provide sufficient details.

### Questions
In Section 3.1 Motivating example, what is the performance of the AGSA gating agent when handling the failure cases of uncertainty estimation and failure detection? It would helpful to show how AGSA addresses the issues in the motivating example.

It would be helpful to discuss the potential challenges of the assumptions AGSA places on the human. The human participant needs to have a very good understanding of the task, the robot’s policy, and how to change the robot’s policy in order to provide the 3 steps AGSA requires. For instance, is it possible the user could intervene but not know if the state necessarily required intervention? 

How much time do humans have time to examine the intervention quality? How long did this take? How is it that they don’t have to “make real-time decisions that can be influenced by tiredness, carelessness, or network latency” when they still need to teleoperate the agent for T steps during intervention? If this occurs offline, then it makes sense how the human’s decisions would be free from latency. It would be great to comment on practical considerations and suitable tasks for a framework like AGSA.

Can state-action pairs that precede control switching by more than 1 timestep warrant avoiding? Currently the learning agent learns to avoid the single previous state, did you also consider larger windows?

For the real-human participant experiments, what was the total number of participants, and what was their breakdown? Was the study IRB approved? No statistical testing is done on the human subjects.

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
The paper introduces AGSA (Agent-Gated Shared Autonomy), a novel framework designed to optimize reinforcement learning (RL) policy training in environments with imperfect human interaction. AGSA addresses key issues in human-in-the-loop learning, including reward-free training, safe exploration, and the challenges associated with suboptimal human control. Unlike existing methods, which often rely on perfect human intervention, AGSA uses a gating agent that determines when to involve human intervention, thus reducing the need for constant human oversight. This gating agent learns from human evaluative feedback and preferences regarding intervention timings and trajectories, aiming to minimize reliance on direct human monitoring or demonstrations. Theoretical insights provide performance bounds for the gating and learning agents, while experiments in continuous control environments with both simulated and real human participants highlight AGSA's advantages in safety, performance, and user-friendliness over previous approaches​.

### Strengths
- Innovative Framework for Human-in-the-Loop RL: AGSA introduces a unique approach to shared autonomy in reinforcement learning (RL) by using an agent-gated model that minimizes the need for perfect human intervention, enhancing training efficiency in reward-free settings.
- Rigorous Theoretical Proof: The paper offers theoretical guarantees, providing performance bounds for both the gating and learning agents, supported by clear proofs and analyses. Extensive experiments across diverse continuous control environments (e.g., robotic locomotion, autonomous driving) demonstrate AGSA’s advantages in training safety, efficiency, and robustness over other approaches.
- Potential Real-World Impact: AGSA’s design addresses the practical challenges of imperfect human feedback in RL, enabling more reliable and safe training in applications like autonomous driving and robotics. By ensuring safe exploration and learning efficiency even with suboptimal human intervention, AGSA broadens the scope of feasible real-world RL applications, especially in unpredictable, dynamic environments.
- Clear and Well-Written Presentation: The paper is well-organized, with detailed explanations of AGSA’s components, including gating agent training and feedback processing, making complex ideas accessible to readers.

### Weaknesses
- Reliance on Predefined Feedback Structures: AGSA depends on specific types of human feedback, namely evaluative and preference feedback, which may not always be feasible or scalable in real-world applications. A discussion on how to adapt AGSA for more passive or implicit feedback—where direct input from humans is minimal—could make the framework more versatile and user-friendly in broader applications.
- Generalization to Higher-Dimensional Action Spaces: The current evaluations are performed on tasks with relatively low-dimensional action spaces, such as robotic locomotion and simplified driving tasks. While these are challenging environments, it remains unclear how AGSA would perform in domains with more complex action requirements (e.g., humanoid robotics or high-degree-of-freedom manipulators). Testing or discussing potential challenges and solutions in these more complex settings would strengthen the generalizability of the approach.
- Evaluation of Human Burden: While AGSA claims to reduce the human workload, the metrics used for the user study might not be the most relevant. In the paper, authors used Performance, Anxiety and Devotion. Performance is a good metric and easy to understand. However, anxiety and devotion are comparably abstract. From the explanation in Appendix C.3, the explanation for the questionnaire is very vague. There would be some better and quantitative metrics for the user study. (Minor issue (line 890): "Choces" should be "Choices").
- Absence of Alternative Feedback Models for Gating: AGSA’s gating mechanism relies on a binary, evaluative feedback model that may not capture the nuances of continuous human feedback. Incorporating or discussing alternative feedback types, such as graded or probabilistic feedback, could further enhance the framework's adaptability to real-world conditions where binary feedback may be insufficient.

### Questions
See above sections.

### Soundness
3

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
The paper presents an interactive imitation learning algorithm. The key idea is to learn a gating agent based on a ground truth intervention signals and a preference-based learned reward. The authors present experiments showing that their method achieves higher reward / success rates than comparison algorithms while being more convenient for human demonstrators.

### Strengths
Originality:
* The paper seems to be the first to use preferences in this context
* The reward formulation combining the ground truth intervention signal with the preference reward is interesting

Quality:
* The paper seems to achieve some impressive results 
* The paper presents ablation results which verify the components of their method

Significance:
* Human in the loop IL is an important research direction and

### Weaknesses
Unfortunately there are several key weaknesses in the paper that make me think it needs some more work before it is ready for publication.
* As I mention below there are some missing details in the experiment section that make it difficult for me to understand the results. Most notably, I'm not sure how much data any of the algorithms were trained on or where the $I(s_t)$ values came from for AGSA.
* I'm concerned that the experiment in Table 1, which is pretty important to the paper, is not fair. As far as I can tell AGSA is the only algorithm which has access to the intervention signals $I(s_t)$ and the preferences (and by extension the ground truth reward)
* Some of the writing is confusing. Algorithm 1 helps a decent bit but I think there are a lot of missing details in the main text.

I'm concerned about the paper's fit for ICLR. As far as I can tell, the paper's key contributions are not so much about the learning method, since no part of their pipeline is very novel. Instead, my takeaway from this paper is that asking for preferences and GT intervention signals does not place much additional burden on supervisors and can help to improve HL learning performance. I think this paper is maybe a better fit for a robotics conference such as CoRL, ICRA or RSS than a learning conference like ICLR. Ultimately I think this issue is better decided by the AC so I am not factoring this into my score, but I feel it's an issue to raise.

### Questions
The following questions were ultimately answered in Algorithm 1, but until I got there I didn't find a clear answer in the main text of the paper:
* It says that humans provide a signal $I(s_t)$ which indicates whether $s_t$ is worthy of an intervention. Does this happen online or do humans go back and relabel the rollouts?
* When the humans interact with the environment (line 213) is that happening online based on some gating signal or is that a post-processing step as well?
* Line 230: what do 'current' and 'previous' segment mean? Is this temporal (ie {s_1, s_}, {s_3, s_4}) or current and previous versions of the policy?
* I don't see anywhere that it says how the gating agent is trained. Is it binary cross entropy on the GT gating values? Is it equivalent to the gating policy $\pi_g$?

Another quick question:
* In algorithm 1, are Q_G and Q_g the same thing?

Experiment Questions:
* How do you get the $I(s_t)$ values from the RL experts?
* What are the parameters of the experiment going into Table 1? How much data do all the policies get? What kind of reward do the expert policies get?
* How do the low and medium experts do so well for RLIF and AGSA?
* It seems to me that your method has much more information available to it than comparison methods (demos + preferences + GT $I(s_t)$ values. Are there any possible baselines that can use a similar amount of data, even possibly ablations on your method?
* I'm confused about what "safety cost" is in the human experiment. It says "the number of potential dangers exposed to the agent" but that's pretty vague

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The problem domain of this paper is human-in-the-loop learning where the human provides feedback to an AI agent by providing interventions. However, as opposed to human-gated interventions where the human decides when to intervene, this paper considers the setup where an external AI agent decides when the human should intervene. In addition, the paper attempts to address the problem of imperfect human feedback, i.e., the human demonstrations when they intervene may be suboptimal. To this end, the paper introduces AGSA, an Agent-Gated Shared Autonomy framework. Through experiments in robotics and autonomous driving tasks, AGSA shows improved training efficiency, safety, and user-friendliness compared to prior methods, even with variable-quality human input.

### Strengths
- The proposed method achieves more efficient and user-friendly learning from interventions compared to existing works.
- The paper provides both theoretical analyses and empirical evidence supporting the effectiveness of the proposed method.
- The proposed method enables learning from humans with varying level of expertise.
- The paper includes a human subject study.

### Weaknesses
- The proposed method requires the human to compare an intervention segment with the previous intervention segment. This brings the assumption that two segments are comparable. Since their initial states will be different, this is difficult to guarantee. This should be discussed.
- The paper says RLHF has not been thoroughly investigated in HL for continuous control tasks. This is not true. Even the original RLHF paper (Christiano et al. 2017) uses continuous control tasks. Similarly, I recommend the authors to check the works by Daniel S. Brown, Scott Niekum, Erdem Biyik, Nils Wilde, Dorsa Sadigh for further studies of RLHF and preference-based learning for continuous control tasks.
- The paper says "uncertainty cannot be aligned with human instructions." It is not clear what this means. Plus, there is no reference or study that supports this statement. Some clarification is needed.
- In line 141, "compares" should be "compare".
- The theoretical analyses presented in the paper are nice but the bounds do not seem useful in practice as they are very loose bounds. This should be discussed as a limitation.
- Line 290 has a broken reference to an equation.
- What is eta in Theorem 3.1. Is it properly defined in the paper?
- I understand the motivation behind using preference data. I agree it is a good way to mitigate the problems due to human suboptimality. But the imitation data could still be used in addition to preferences.
- The human subject studies should report the number of subjects and demographics information.
- Table 5 reports variances but I suggest adding statistical significance tests as well.
- Why is future tense used in line 547?
- The paper reports authors have applied for IRB. Is it not approved yet? It is not acceptable to conduct human subject studies before IRB approval.

### Questions
Please see the questions in the weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
3
