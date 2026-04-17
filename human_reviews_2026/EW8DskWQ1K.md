# Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
The temporal lag between actions and their long-term consequences makes credit assignment a challenge when learning goal-directed behaviors from data. Generative world models capture the distribution of future states an agent may visit, indicating that they have captured temporal information. How can that temporal information be extracted to perform credit assignment? In this paper, we formalize how the temporal information stored in world models encodes the underlying geometry of the world. Leveraging optimal transport, we extract this geometry from a learned model of the occupancy measure into a reward function that captures goal-reaching information. Our resulting method, $\textrm{\textbf{Occupancy Reward Shaping (ORS)}}$, largely mitigates the problem of credit assignment in sparse reward settings. ORS provably does not alter the optimal policy, yet empirically improves performance by $\mathbf{2.2\times}$ across 13 diverse long-horizon locomotion and manipulation tasks. Moreover, we demonstrate the effectiveness of ORS in the real world for controlling nuclear fusion on 3 Tokamak control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Offline GCRL methods trained with sparse rewards struggles in especially long-horizon tasks, since learned value functions tend to be noisy. Task specific hand-crafted reward functions can address this problem, but the task-based reward design is challenging and not practical. It is stated that previous work mostly focuses on online GCRL and recent offline GCRL approaches employ graph based distance classifiers, which fail to scale to large datasets. In this paper, considering the importance of scalability and generalization, an occupancy measure based approach is proposed for offline GCRL. The main idea is that the occupancy measure over future states is learned via flow matching, and then a reward function is defined to guide the policy via occupancy measure matching. Overall the idea is interesting and experimental results show such an occupancy measure based algorithm improves the performance of offline GCRL in long-horizon settings.

### Strengths
- **Interesting Approach**: Learning the occupancy measure via flow matching and then using this similarity as the reward function is a novel and interesting approach.
- **Improved Non-monotonicity**: It is nicely presented that the proposed approach has lower non-monotonicity when compared to sparse rewards. 
- **Analysis of the Proposed Approach**: The experimental analyses are well-presented, clearly showing the effectiveness of ORS over sparse rewards.

### Weaknesses
- **Computational Overhead**: The computational overhead of the proposed approach is not discussed. ORS requires learning an occupancy measure before training the policy, making it appear computationally more complex than baselines. Therefore, the computational overhead should be discussed and compared with both graph-based and non-graph-based offline GCRL baselines.
- **Novelty**: Occupancy measure matching has already been employed by recent works for GCRL. The contribution over these methods is not discussed, which makes the novelty questionable. The benefits of ORS over recent literature (a-b) must be elaborated.
- **Baselines**: In the experiments, recent relevant works (a-c) are omitted from the comparisons.
- **Experimental Setting**: The experiments only cover long-horizon tasks. The applicability of the proposed method to short- and medium-horizon tasks should be discussed.
- **Experimental Results**: The environments used between Table 1 and Table 2 are not the same. It is not clear why some environments are included in Table 1 but omitted in Table 2. For a clear and fair evaluation of ORS, all environments in Table 1 must also be included in Table 2.
- **Vague Explanation**: The main assumption of the paper (in long-horizon tasks, the value function exhibits a high level of non-monotonicity) is evaluated under section 3.2, however it is not clear how $\hat{V}(s,g)$ is trained. It is not explained how the authors obtained $\hat{V}(s,g)$ in 3.2, or which algorithm was used and how $\hat{V}(s,g)$ was trained.

[a]: Sikchi, Harshit, et al. "Score models for offline goal-conditioned reinforcement learning." The Twelfth International Conference on Learning Representations. 2023.

[b]: Ma, Jason Yecheng, et al. "Offline goal-conditioned reinforcement learning via $ f $-advantage regression." Advances in neural information processing systems 35 (2022): 310-323.

[c]: Zhou, John Luoyu, and Jonathan C. Kao. "Flattening Hierarchies with Policy Bootstrapping." Workshop on Reinforcement Learning Beyond Rewards@ Reinforcement Learning Conference 2025.

### Questions
- How computationally complex is the proposed method? Can you please provide computational efficiency comparisons with baselines and graph-based solutions?
- Can you please elaborate on the benefits of the proposed approach over recent literature [a-b]?
- Can you please compare the proposed method with [a-c], in addition to the current baselines?
- Can you please elaborate on whether this approach would also be useful in medium or short-horizon tasks?
- Can you please clarify how $\hat{V}(s,g)$) was trained for the analysis in Section 3.2? What algorithm was used?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the problem of insufficient signals in offline goal-conditioned reinforcement learning (Offline Goal-Conditioned RL, GCRL) under sparse rewards and long-horizon tasks. It proposes Occupancy Reward Shaping (ORS), a learning-based reward shaping method grounded in the occupancy measure, which can capture temporal dependencies in long-horizon tasks. By integrating flow matching for fitting, ORS distills goal achievement information from the occupancy measure into a generalizable reward function.

### Strengths
1. The paper is clearly written, logically structured and well organized. Technical details are thoroughly presented.
2.The theoretical analysis is solid, providing proofs of convergence and an analysis of reward monotonicity. Starting from empirical evidence that sparse rewards lead to non-monotonic value functions, the paper proposes a reward shaping idea based on the occupancy measure, forming a complete logical chain.
3.Experiments cover both locomotion and manipulation tasks. The experimental design is rigorous, and the results consistently demonstrate strong performance.
4.The approach is compatible with existing offline goal-conditioned RL algorithms, and the analysis of the value function non-monotonicity is insightful. Experiments validate that ORS effectively alleviates this issue.

### Weaknesses
1.While multiple tasks from OGBench are used, all of them are simulated environments. There is no mention of testing on real-world data or environments with different physical properties. It remains unclear whether the algorithm remains stable under real-world physics or diverse visual conditions.
2.The theoretical assumptions are relatively strong. Discussions or brief experiments demonstrating robustness under stochastic dynamics are needed, or clarification on whether these assumptions still hold approximately in practice.
3.Ablation studies are insufficient. Although the text mentions “conduct detailed analyses and ablations,” the contribution of different ORS components to the final performance is not shown.

### Questions
1.The κ parameter in the ablation is highly sensitive. Could adaptive scheduling of κ improve stability?
2.Does the optimality guarantee in Theorem 1 depend on data coverage quality? In datasets covering only part of the optimal path, can ORS still maintain optimality?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to address the struggles of offline goal-conditioned RL on long horizon tasks via reward shaping. In particular, it propposes a novel reward shaping method, occupancy reward shaping, trained using flow matching to perform effective credit assignment as a reward function. Experiments show that it improves over prior Offline GCRL methods on long-horizon lcomotion and manipulation tasks in simulation.

### Strengths
- clear motivation and extensive discussion on background
- provides proofs for theoretical guiarantee

### Weaknesses
- It would be more convincing if results can also be demonstrated on real-world robotics tasks, where both data quantity and quality are lower
- There should be more discussion on other ways of computing dense reward information
- Results seem only to be marginally better, most of the gains over GO-FRESH are on 2 tasks
- how did the authors select the tasks in the benchmark? why are tasks like ant soccer and humanoid maze not selected?

### Questions
- What is the quality and quantity of data that is required to train such a reward model? Will there be circumstances where the reward model is not accurate? If so how is the performance affected
- In an offline RL setting, since there is no online interaction, why can't the reward be simply obtained using distance of current state to goal?
- Can you discuss the comparison of your approach against works like GoFar (Ma et al), where shaped reward is not used, and goal reaching behavior is direclty learned by minimizing divergence between policy and expert's  goal conditioned state occupancy?

### Soundness
3

### Presentation
2

### Contribution
2
