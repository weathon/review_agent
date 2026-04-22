# Dejavu: Post-Deployment Learning for Embodied Agents via Experience Feedback

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 8

## Abstract
Embodied agents face a fundamental limitation: once deployed in real-world environments to perform specific tasks, they are unable to acquire new useful knowledge to enhance task performance. In this paper, we propose a general post-deployment learning framework called Dejavu, which employs an Experience Feedback Network (EFN) and augments the frozen Vision-Language-Action (VLA) policy with retrieved execution memories. EFN automatically identifies the most contextually successful prior action experiences and conditions action prediction on this retrieved guidance. We adopt reinforcement learning with semantic similarity rewards on EFN to ensure that the predicted actions align with past successful behaviors under current observations. During deployment, EFN continually enriches its memory with new trajectories, enabling the agent to exhibit “learning from experience” despite fixed weights. Experiments across diverse embodied tasks show that EFN significantly improves adaptability, robustness, and success rates over frozen baselines. These results highlight a promising path toward embodied agents that continually refine their behavior after deployment. We provide the code and demo on our anonymous website https://dejavu2025.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Dejavu proposes a post-deployment learning framework that enables frozen VLAs to improve after deployment without gradient-based updates. It introduces experience feedback network, a lightweight residual module that refines the base policy's output by reusing stored past experiences from an experience memory bank.

### Strengths
This paper addresses an important challenge in the VLA literature, achieving post-deployment adaptation without retraining large embodied policies. It presents a practical framework that demonstrates memory-driven self-improvement on both simulated and real robotic environments, highlighting the potential of non-gradient adaptation in embodied AI.

### Weaknesses
1. The main limitation of the paper lies in its simplified problem formulation, and the proposed solution is equally naive, offering little conceptual novelty beyond straightforward retrieval and residual correction.

2. The framework implicitly assumes the existence of successful trajectories or identical tasks already exist in the memory to provide strong supervision for the agent, which constitutes an overly strong assumption and fundamentally limits its capability to generalize to truly novel or unseen scenarios. 

3. Although the paper emphasizes that the proposed framework performs post-deployment learning without any gradient-based updates, this claim is somewhat misleading. In practice, the agent continually updates its experience memory, which effectively serves as an external, non-parametric knowledge update mechanism. Thus, while the policy parameters remain frozen, the system's behavior still changes through memory accumulation, making it not truly gradient-free, but rather a memory-driven adaptation process.

4. The framework requires maintaining a large experience bank to achieve noticeable improvements. Even expanding the bank size from 100 to 1000 yields only marginal improvements (Table 1). This indicates poor scalability and raises concerns about the method's practically in real-world deployments with limited memory or compute resources. 

5. The framework performs retrieval at every timestep, introducing computational overhead and is arguably unnecessary, as adjacent frames within at trajectory are often highly correlated. A more efficient strategy could greatly reduce inference cost without sacrificing performance.

6. Although the proposed method is conceptually similar to RAG paradigms, the paper does not include any baselines that incorporate retrieval mechanisms for fair comparison.

### Questions
1. How significant is the inference-time overhead introduced by EFN, given that each step requires retrieval, similarity computation, and residual correction in addition to the base policy execution?

2. Why does the method require performing a retrieval at every timestep?
 
3. Are there any experiments evaluating on unseen or novel tasks to assess its generalization ability beyond memorized or previously encountered trajectories?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes DEJAVU, a novel framework for enabling embodied agents to continue improving their performance post-deployment in real-world environments without updating model parameters. At the core of DEJAVU is the Experience Feedback Network (EFN), which augments a frozen Vision-Language-Action (VLA) policy by retrieving prior successful trajectories and applying residual corrections to refine the base policy’s outputs. EFN is trained via reinforcement learning

using a semantic similarity–shaped dense reward, optimizing residual actions so that the predicted next observation resembles the next frame in a retrieved experience. This allows continual adaptation during deployment without gradient-based updates. Experiments on the LIBERO simulator and the AgiBot-G1 physical robot demonstrate consistent improvements over frozen baselines across multiple backbones (OpenVLA, UniVLA, and GO-1).

### Strengths
- Technical Advancement: The work tackles the challenging problem of post-deployment learning without parameter updates, introducing an inference-time, experience-conditioned residual correction mechanism. The design integrates ideas from retrieval-augmented RL, residual policy learning, and continual adaptation.
- Empirical Consistency: Across diverse backbones (OpenVLA, UniVLA, GO-1) and both simulated and real-world settings, EFN consistently improves success rates and reduces trajectory lengths. Notably, Tables 1 and 2 show monotonic performance gains as the size of the experience memory increases, demonstrating scalable benefits.
- Methodological Clarity: The paper clearly structures the method, covering the experience bank design, retrieval mechanism, similarity-based reward, and reinforcement learning optimization. Figures 2 and 3 effectively visualize the training and inference pipelines.
- Practical Relevance: Successful deployment on a physical robot supports the method’s real-world viability and generality.

### Weaknesses
Section 2 categorizes related work into three lines, Retrieval-Augmented RL, Residual Policy Learning, and Post-Deployment Adaptation, and claims that the proposed EFN embodies a compositional integration of these three paradigms. Specifically:

- (i) retrieves a task-relevant experience trajectory,
- (ii) predicts a residual action that refines the base policy’s output, and
- (iii) optimizes the residual via dense, similarity-shaped reinforcement signals comparing the observed next frame to the next frame in the retrieved trajectory.

Accordingly, EFN’s contribution could be further substantiated through more explicit component-level comparisons with recent representative works in each direction:

- (i) Compared with Retrieval-Augmented RL approaches [1,2], EFN leverages retrieval not merely as a reference but as an inference-time conditioning mechanism, achieving post-deployment adaptation without parameter updates.
- (ii) In relation to Residual Policy Learning [3,4], it would be valuable to assess how effectively EFN’s residual action prediction refines the base policy compared to standard residual control methods.
- (iii) Regarding Similarity-Shaped Reinforcement Learning, comparison with dense-reward methods such as world model-based prediction or goal-similarity rewards [5,6] could highlight EFN’s advantages in training stability and generalization.

[1] Retrieval-Augmented Reinforcement Learning. Goyal et al., 2022.

[2] Titans: Learning to Memorize at Test Time. Behrouz et al., 2024.

[3] Residual Reinforcement Learning for Robot Control. Johannink et al., 2019.

[4] Visual Reinforcement Learning with Residual Action. Liu et al., 2025.

[5] Mastering Diverse Domains through World Models. Hafner et al., 2025.

[6] Test-Time Offline RL on Goal-Related Experience. Bagatella et al., 2025.

Although the paper reports significant performance improvements on both simulated (LIBERO) and real-world (AgiBot-G1) benchmarks, it remains unclear to what extent these gains represent an advance over existing approaches.

The current experimental setup focuses mainly on internal ablation studies (removing or simplifying EFN components) and does not provide even partial or functional comparisons against representative methods such as retrieval-augmented, residual policy, or similarity-based RL.

Therefore, even without full end-to-end benchmarking, component-wise quantitative comparisons with these baselines would clarify which aspect of EFN yields the strongest performance advantage and by how much.

### Questions
- Does EFN’s retrieval mechanism serve merely as a memory lookup, or does it materially influence the inference-time decision process? Does the submission include a discussion of how this design differs theoretically and empirically from conventional approaches?
- How does the residual policy interact with the frozen VLA backbone to ensure stable correction? Does the submission include empirical validation of the claimed benefits of the gradient-free residual optimization against standard residual RL methods?
- Does the submission include an analysis of whether the similarity-shaped dense reward genuinely improves training stability compared to sparse rewards, or whether it mainly serves as a shaping mechanism?
- If EFN were directly compared to Retrieval-Augmented RL or Residual RL, what performance trends or trade-offs might we expect?
- Given the issues discussed in Appendix A, such as memory growth and retrieval ambiguity, what mechanisms would be required for EFN to evolve toward a truly continual learning framework?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Experience Feedback Network (EFN) to augment VLA policies with previous executions at test time. EFN identifies relevant trajectories and then predicts a residual action for the frozen VLA to encourage following the previous trajectory.

### Strengths
1. The framework does not modify the VLA. This helps maintain the strong generalization performance of the VLA from the large-scale pretraining.

1. The paper addresses the important problem of improving the model with test-time experience in interactive environments.

1. Results in Tables 1 and 2 confirm that EFN improves the performance of base VLA policies.

1. Ablations in Figure 6 confirm the importance of EFN components.

### Weaknesses
1. From my understanding, EFN encourages the policy to mimic previous successful evaluations at test time. This increases the robustness of successful behaviors by reinforcing them with the residual action prediction. This only enables EFN to learn from experience in already successful trajectories. For tasks where the policy outputs failed or suboptimal trajectories, EFN cannot improve the performance.

1. The framework assumes the task and setting in the selected experience are very similar to the current task. By rewarding EFN for matching per-step semantic features, the current state must match that in the selected context. This assumption likely holds when the starting state distribution is narrow, the language instruction is the same, and the environment dynamics are largely deterministic, as is the case for Libero. However, this will not hold for more complex environments with varied starting states, stochastic dynamics, or varied language instructions within a task.

1. The paper does not compare with sufficient baselines that also leverage the test time experience of successful trajectories in the same task. For example, rather than training a residual network, a baseline is to select the state from the buffer with the nearest neighbor visual feature and then select that action from that corresponding state. 

1. The paper only shows results on Libero and a custom real-world setup. Reporting results on additional benchmarks such as CALVIN or SimplerEnv would strengthen performance.

1. The paper uses SAC to train the EFN module, while the majority of recent works use variants of PPO or REINFORCE. This decision to use SAC is not justified.

1. The paper does not include the EFN RL training curves showing the reward over the number of updates. This is important for reproducing results.

1. The paper does not provide sufficient details about the real-world AgiBot experiments. How many trials are conducted per task? How are the numbers in Table 2 reported with three significant figures? Furthermore, what do these three tasks correspond to? How diverse is the starting state?

### Questions
1. How can EFN be extended to improve from experience in failed or suboptimal trajectories (see weakness 1)?

1. How does EFN perform with a more varied starting state distribution (see weakness 2)?

1. Why use SAC over variants of PPO or REINFORCE (see weakness 5)?

Minor:
1. What does the paper mean by "non-blank step" (L219)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a new framework, Dejavu, which incorporates a new "Experience Feedback Network", with frozen VLAs to improve performance using the retrieved "memories" from an experience database (embeddings of vision-action trajectories).

The experiences are retrieved based on the language-conditioned contextual similarity as guidance. The EFN network takes the current observation and the retrieved experience to predict a residual corrective action, which is added to base policy's output for the final action. The network is optimized using RL with dense cosine similarity reward between next observation and next state in the retrieved trajectory. The reward also incorporates progress and prevents idling at the same state. EFN is optimized using SAC algorithm.

The EFN experience bank uses compact keys (with mean-max fusion) for fast retrieval, language-conditioned similarity and allows addition of experiences to the bank during inference. During retrieval they use top-k sampling from the keys.
For inference, the EFN uses instruction-filtered candidate set for retrieval using the VLA encoding for the language instruction. They also promote shorter rollouts by normalizing the cosine similarity of the experiences with experience length.

The authors use OpenVLA, UniVLA and GO-1 as the base VLAs, and evaluate in both sim (LIBERO) and real (AgiBot G1). The tasks involve table-top manipulation. The authors also experiment with different experiment bank sizes and show that larger banks usually are better for success and efficiency. They perform ablations by replacing SAC with value-critic, removing anti-idling penalty and removing instruction-based filtering during inference, and show the importance of these different aspects of the approach.  The highest drop is observed after replacing SAC with V-critic.

### Strengths
1. The paper applies the concept of memories to VLA in a very methodical and well-thought and novel fashion. The creation of experience bank, fast retrieval, residual corrective actions are well-designed to handle the task of post-deployment VLA improvement.
2. The authors perform thorough evaluations and ablations, including real-world experiments, which shows that the method actually works in the real-world.
3. The authors show critical ablations showing that each aspect of their approach is important for the task.

### Weaknesses
1. While there is no training of the VLA required, the EFN network does need to be trained with the benchmark in use. This means that for each new task/benchmark, a new EFN network has to be trained, making the approach not easily scalable.
2. The authors should discuss the task and baselines in the abstract and the introduction, or anywhere in the experiments. Without prior context, it is hard to know which task (e.g. tabletop manipulation) are targeted.
3. Only one kind of task - table-top manipulation is considered, casting a doubt on applicabilty of this approach on other tasks (e.g. navigation, mobile manipulation, etc.)

Minor:
1. Incorrect grammar for the sentence: "This ability to draw... before"".
2. The first contribution is misleading - the EFN network does require gradient-based retraining. Would be good to clarify this.
3. Figure 4 is confusing - what is left, right and middle in the figure?

### Questions
1. What would happen if instead of residual action, the network also takes VLA action as input and predicts the new action? I ask this question out of curiosity and this could be a useful ablation perhaps.
2. How is success incorporated into the EFN reward? Or do we just care about getting similar trajectory? How is the experience bank filled during training?
3. How is the EFN trained for real-world deployment? How are the trajectories collected?

### Soundness
3

### Presentation
3

### Contribution
3
