# Ego-Foresight: Self-supervised Learning of Agent-Aware Representations for Improved RL

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Despite the significant advances in Deep Reinforcement Learning (RL) observed in the last decade, the amount of training experience necessary to learn effective policies remains one of the primary concerns in both simulated and real environments. Looking to solve this issue, previous work has shown that improved efficiency can be achieved by separately modeling the agent and environment, but usually requires a supervisory signal. In contrast to RL, humans can perfect a new skill from a small number of trials and often do so without a supervisory signal, making neuroscientific studies of human development a valuable source of inspiration for RL. In particular, we explore the idea of motor prediction, which states that humans develop an internal model of themselves and of the consequences that their motor commands have on the immediate sensory inputs. Our insight is that the movement of the agent provides a cue that allows the duality between the agent and environment to be learned.  To instantiate this idea, we present Ego-Foresight (EF), a self-supervised method for disentangling agent information based on motion and prediction. Our main finding is that, when used as an auxiliary task in feature learning, self-supervised agent-awareness improves the sample-efficiency and performance of the underlying RL algorithm. To test our approach, we study the ability of EF to predict agent movement and disentangle agent information. Then, we integrate EF with both model-free and model-based RL algorithms to solve simulated control tasks, showing improved sample-efficiency and performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Ego-Foresight (EF), a training method that employs an auxiliary task of predicting agent-specific features. The authors show that EF improves sample efficiency when applied to both model-free and model-based RL algorithms. They show that EF lifts limitations of supervised methods for representation disentanglement and enables agents to adapt to changes in their body-schema, such as during tool use.

The authors provide strong motivation for their method and the results are convincing. The contribution of a self-supervised approach for motor prediction is of interest to the RL community. I have some concerns regarding the clarity of the mechanisms underlying EF's performance improvements, but overall, I think this paper should be accepted.

### Strengths
EF demonstrates improved sample efficiency when integrated with both model-free and model-based RL algorithms. The performance differences and improved sample efficiency between DRQv2-EF/TD-MPC-EF and their baseline counterparts are convincing across the tasks presented. Additionally, reporting Rliable metrics in addition to the performance curves add robustness to the results since they report statistic uncertainty more rigorously.

EF successfully reconstructs future agent configurations and adapts to changes in body-schema such as tool use (Figure 4). This approach offers advantages over supervised methods, which are constrained by fixed body-schema and cannot adapt to changes such as tool use.

### Weaknesses
The authors propose that EF's improvements stem from (i) disentanglement of agent information allowing the RL algorithm to focus on agent control in early training stages and environmental interactions later, and (ii) regularization through imposing predictability in robot movements. However, there seems to be not enough evidence supporting this mechanism, particularly the claim about distinct training phases. While Figure 1 (left) shows some training progression, the predictive ability for agent-specific features appears to improve later in training rather than in the initial stages, which seems inconsistent with the proposed explanation.

I also have a slight concern about the use of Efficiency Normalized Score (ENS). ENS identifies the step at which 95% of maximum performance is reached and measures algorithm performance at that point. This metric may inadvertently favor algorithms with higher performance variance which could reach the threshold faster than more stable learners. So ENS may not be reflecting true sample efficiency accurately.

### Questions
**Questions for clarification:**

1. The claim that "predicting the sensory consequences of one's own movement is remarkably simpler, yet equally important" cites Wolpert and Flanagan (2001), but this source doesn't appear to directly support this claim. Could you clarify what you mean by "simpler" and "important" in this context, and provide additional citations if available?
2. You mention "both h_a and h_s have a similar influence in the predicted frame, with solid and static regions such as the background being quickly overfitted by the decoder," and mention the "regularizing effect on learning". Could you elaborate on what constitutes overfitting in this context and how you observe the regularizing effect?
3. You mention "the need for pixel-wise reconstruction of the scene, which could be addressed in the future by exploiting a contrastive loss." Could you elaborate on this idea and how a contrastive loss might address the current limitations?
4. You mention "the characteristic instability of baseline RL algorithms such as DDPG." Did you observe this instability in your experiments?

---

**Minor comments:**

- Figure 4 (left) could be clearer. It's difficult to distinguish the trajectory of the end effector because the end-effector is blurred. The trajectory was more distinguishable in the gif provided in the codebase, so maybe a different angle would be helpful for the static figure.
- The claim that "in terms of ENS, a significant improvement is achieved with the addition of our approach" would be strengthened by adding confidence intervals to Figure 1 to support statistical significance.
- In the ablation study you mention, "In terms of the dimensionality of the agent features h_a, the ablation study shows that larger sizes have a detrimental effect." This wasn't immediately apparent to me because the larger dimensionalities appeared to have similar performance curves to the default value of 32.

---

**Suggestions for improvement (not factored into score):**

1. Could evaluate EF with DreamerV3 to demonstrate generalization across model-based algorithms.
2. Include additional analysis demonstrating the proposed distinct phases where the model learns agent-specific versus environment-specific features.
3. Compare the performance boost from tool use versus no tool use, and compare this difference with the difference given by the supervised baseline (SEAR). Given your claims about supervised approaches being non-adaptable to body-schema changes, EF should demonstrate better performance gains on tool-use tasks compared to SEAR.
4. The ablation study performance curves on the Door Open task show high variance with similar results across conditions. Could consider conducting ablations on tasks with larger differences, such as bin picking or peg insertion, to better distinguish the impact of different hyperparameters.
5. You mention that sample efficiency is more important than computational efficiency, but it would still be helpful to quantify the wall-clock time tradeoff of EF.

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
This paper proposes Ego-Foresight (EF), a self-supervised method, where the core idea is to disentangle the agent’s representation from the environment’s representation by leveraging motor prediction. The model learns to separate a feature vector h into agent-specific features ha  and scene-specific features hs. A recurrent model predicts future agent features based on unrolled action sequences, while the scene features hs are fast-forwarded from the context frame. The predicted agent features and the original scene features are then used to reconstruct the predicted frame to train the encoder. This method avoids the need for explicit supervision required by prior work like SEAR. The authors demonstrate that adding EF as an auxiliary loss to model-free (DrQ-v2) and model-based (TD-MPC2) algorithms improves performance and sample efficiency on Meta-World and DMC tasks.

### Strengths
- The core idea of using motor-prediction as a self-supervised signal for agent-environment disentanglement is interesting. It provides an alternative to supervised methods that rely on segmentation masks, which are often unavailable in real-world settings.
- The proposed EF module is model-agnostic and can be integrated as an auxiliary loss into both model-free and model-based RL algorithms, as demonstrated by its successful application to DrQ-v2 and TD-MPC2.
- The method shows clear improvements in sample efficiency and asymptotic performance against strong baselines (DrQ-v2, TD-MPC2) and is competitive with its supervised counterpart (SEAR) on several tasks.

### Weaknesses
- Figure 4 (“Door Open” and "Hammer" task) seems to be central to the claims made in the paper and tries to show that the door is part of the environment whereas the hammer becomes part of the body schema. However, the distinction is not so clear to me. the hammer has to be picked up first, just like the agent has to grip the door handle. Once the contact is made the door could be moved to and fro just like the hammer. The illustrations in the figure also does not really help. It seems like the gripper disappears in the door opening task, i can't even see whether the door is open or close in the picture...

- The authors state that “horizon length does not significantly impact the results”. However, the “Prediction Horizon” graph in Figure 8 clearly shows that H=10 and H=40 (peak reward ~4500) perform significantly better than H=5 and H=20 (peak reward ~4000). A ~500 reward difference (or ~12.5%) is not insignificant. It is similiar to the difference between β=0  (no EF module) and other β where it is claimed that the difference is important. This difference in treatment of the same kind of error difference seems concerning wrt to the paper's claims.

 Minor issues:

- Presentation: The paper's structure could be tightened. Section 2 ("Background") is a single, long subsection that could be broken up. Similarly, for Section 4.1 ("Experimental Setup"), the special subtitle "Environments" is not necessary.

- Typo: On Line 238, the feature vector is referred to as $h_{t_c}$. To be consistent with the paper's notation defined on Line 191, this should be $h^{t_c}$.

- Clarity: In Figure 2, the "action" label is a bit vague. It seems strange to use a single variable $x$ to represent the observation, and use a vector $[a^0, \cdots, a^M]$ to represent the action.

### Questions
## Questions
1. Following Weakness #1: Could the authors please clarify the role of $h_s$? Is it intended to be static? Why does the gripper disappear in the Figure 4 prediction, and what does this imply about what $h_s$ is actually encoding? 

2. Following Weakness #2: Could the authors justify their claim that a ~500 reward gap in the horizon ablation is "not significant"? It appears to be one of the more significant factors in the study.

3. The qualitative results in Figure 4 show predictions for relatively small movements. How does the model's predictive quality and disentanglement hold up when the agent performs large-scale actions or when predicting over a longer horizon?

4. in Fig. 5 and 6, the authors do not integrate their approach into the Dreamer architecture, but only in the other two approaches. Why is that?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proposes idea of self-supervised learning of consequences of RL agent's actions in an environment for inducing agent-aware representations. The motivation behind the work is grounded in how humans foresee the effects of their actions while executing them which allows them to perform the tasks more effectively. The paper proposes use of convolutional encoder-decoder architecture that predicts future frames given the visual features and sequence of planned actions. A self-supervised loss is applied on the predictions that allows us to learn visual features that enable easy prediction of the future, and hence instill agent-awareness. The authors extend pre-existing RL algorithms in robotics with proposed ego-foresight loss that allows for improved performance.

### Strengths
1. The paper is very well written with the introduction making connection of the technique with human locomotion and various other foresight instances. 
2. The approach proposed in this work seems intuitive and should be easy to integrate into existing RL pipelines learning from visual features.
3. The results demonstrate consistent improvement when the technique is used in conjunction with prior frameworks. 
4. The choice of the research questions studied is apt for this work and the paper goes into detail answering each of them.

Overall, I find the quality of the write-up, the experiments and the results high and recommend accepting the work.

### Weaknesses
1. The paper uses recurrent neural networks, while the overall efficacy of transformers in sequence generation is well-proven in the domains of natural language and vision.
2. Episode partitioning uses a fixed constant H for defining the future window. This choice could have been made more dynamic to remove any bias due to burden of predicting next H frames every time.

### Questions
Are these results also applicable in non-robotics environments like standard control tasks (e.g. CartPole) in RL? It would be nice to have an auxiliary loss that allows the neural network to predict the next states to align the representations used for deciding the policy. It is only a minor suggestion.

### Soundness
4

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
This submission proposed Ego-Foresight (EF), a self-supervised method using motion prediction to disentangle agent information from environment, aiming at improving sample efficiency for deep RL. The method uses an encoder-decoder architecture with a recurrent module that predicts future agent configurations from actions, with a bottleneck forcing focus on agent-specific features. The authors integrate EF with DrQ-v2 and TD-MPC2, and compare with SEAR and Dreamer on Meta-World and DMC benchmarks, showing improved sample efficiency particularly on tasks requiring tool use.

### Strengths
1. The proposed self-supervised framework for learning agent-aware representation, as opposed to supervised training that requires agent masks in previous work.
2. The proposed method can be integrated with both model-free (DrQ-v2) and model-based (TD-MPC2) algorithms, showing flexibility.
3. The evaluation is comprehensively done on 26 tasks across two benchmarks and baseline methods, including Dreamer (model-based), SEAR (previous supervised agent-aware learning method). Results are presented with proper statistical analysis with RLiable metrics, and the ablation studies are useful for dissecting the proposed model.
4. The paper is generally well-written with good use of figures, including the model diagram.

### Weaknesses
1. The results on performance improvement does not seem to be consistent across the tasks. In figures 5 and 6, SEAR and Dreamer seem to perform equally well, and sometimes even better, than EF. This performance gap seems to be more prominent in figures 11 and 12. This inconsistency is not adequately explained. Also, for efficiency normalized score presented in figure 1, could you please provide standard deviation/ errors to properly assess the improvement? 
2. Since the proposed EF algorithm is a self-supervised algorithm for image based deep RL, why are other self-supervised image-based RL algorithms not included as comparison baselines, such as CURL (Srinivas, Laskin, et al, 2020) and SPR (Schwarzer, Anand, et al, 2021)? This and the performance inconsistency are perhaps the main weaknesses of the current draft.
3. The ablation study is performed on the DoorOpen task, which is not part of the MetaWorld tasks that require tool use as presented in figure 5. Since you highlighted EF should be better in tasks that require tool use, why is the ablation study done on DoorOpen?
4. As learning efficiency is emphasized, it’d be good to provide comparisons on computational overhead (wall-clock time, memory, etc.), which is mentioned but not quantified.
5. Maybe I missed something, but in terms of the algorithm (neuroscience inspiration aside), Intuitively, why would learning disentangled representation lead to improved sample efficiency?

### Questions
1. Can you explain more what Fig 4 is implying? Perhaps using the same visualization for Fig 1 to highlight what’s actually being learned by the agent feature?
2. In the ablation study, larger agent feature dimension seems to have a detrimental effect, which is attributed to “reducing the ability to disentangle agent information.” Might an alternative explanation be just because when the total feature dimension is too large, the policy is harder to learn as the state space is now larger?
3. For tasks where EF doesn't help (e.g., Walker tasks), what is the failure mode? Can you characterize when EF is expected to help?
4. How good is the learned agent feature? Since EF is proposed to disentangle agent from the environment, can you quantify how much overlap there is between the learned agent feature with the supervised agent masks if available in some tasks?
5. Tool embodiment is mentioned as an interesting capability, and is perhaps when EF might be helpful the most. Does this adaptation happen within a session? Say in the hammer task, does the agent feature starts to include the hammer only after the hammer is being picked up? Or does the agent feature already include the hammer from the beginning?

### Soundness
2

### Presentation
3

### Contribution
2
