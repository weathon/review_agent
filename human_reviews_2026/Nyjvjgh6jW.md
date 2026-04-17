# Test-time Offline Reinforcement Learning on Goal-related Experience

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Foundation models compress a large amount of information in a single, large neural network, which can then be queried for individual tasks. There are strong parallels between this widespread framework and offline goal-conditioned reinforcement learning algorithms: a universal value function is trained on a large number of goals, and the policy is evaluated on a single goal in each test episode. Extensive research in foundation models has shown that performance can be substantially improved through test-time training, specializing the model to the current goal. We find similarly that test-time offline reinforcement learning on experience related to the test goal can lead to substantially better policies at modest compute costs. We propose a novel self-supervised data selection criterion, which selects transitions from an offline dataset according to their relevance to the current state and quality with respect to the evaluation goal. We demonstrate across a wide range of high-dimensional loco-navigation and manipulation tasks that fine-tuning a policy on the selected data for a few gradient steps leads to significant performance gains over standard offline pre-training. Our goal-conditioned test-time training (GC-TTT) algorithm applies this routine in a receding-horizon fashion during evaluation, adapting the policy to the current trajectory as it is being rolled out. Finally, we study compute allocation at inference, demonstrating that, at comparable costs, GC-TTT induces performance gains that are not achievable by scaling model size.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method called Goal-Conditioned Test-Time Training (GC-TTT) for dynamically fine-tuning pretrained goal-conditioned reinforcement learning policies during the test phase. The approach selects trajectory segments from an offline dataset that are relevant to the current state and optimal for the current goal, using them to perform minimal gradient updates on the policy. GC-TTT significantly improves policy performance across various high-dimensional movement and manipulation tasks, while maintaining manageable computational costs.

### Strengths
1.This work introduces test-time training to offline goal-conditioned reinforcement learning, breaking the traditional "train-freeze-infer" paradigm.

2.The method can be combined with various offline RL algorithms (e.g., GC-BC, GC-IQL, SAW) and does not rely on a specific algorithmic structure.

3.In multiple benchmark tasks, the success rate significantly improves, particularly in complex environments.

4.Under the same computational budget, GC-TTT is more effective than simply increasing model parameters.

### Weaknesses
1.Although the use of test-time adaptation in offline GCRL is both reasonable and novel, merely stating that it improves policy generalization and performance is insufficient. The motivation behind this approach is not sufficiently clear and explicit in the paper.

2.Sub-trajectory selection depends on the accuracy of the value function or distance function; inaccurate estimates may affect performance.

3.[1] also applies TTA to offline goal-conditioned reinforcement learning. Although it is part of the same body of work, I suggest the authors clarify the differences and provide a discussion in the related work section.

4.As I understand it, this paper presents a plug-and-play method for offline goal-conditioned reinforcement learning algorithms. I would like to ask the authors to explain why it has not been applied to more advanced GCRL algorithms such as QRL [2] and HIQL [3].

5.While the experimental section shows improvements in state-based environments, the performance in vision-based environments remains unclear.

6.Lack of theoretical foundation and interpretability: The paper is entirely experiment-driven, lacking a theoretical analysis of why the GC-TTT method is effective and when it might fail. 

7.Poor interpretability: GC-TTT resembles a "black-box" optimization process, making it difficult to understand what the policy specifically learns during testing. Which systematic errors or challenges from pretraining are addressed? The absence of visualization or analysis of the internal mechanisms reduces the method's reliability and credibility.

[1]Opryshko E, Quan J, Voelcker C, et al. Test-Time Graph Search for Goal-Conditioned Reinforcement Learning. arXiv preprint arXiv:2510.07257, 2025.

[2]Wang T, Torralba A, Isola P, et al. Optimal goal-reaching reinforcement learning via quasimetric learning. ICML, 2023.

[3]Park S, Ghosh D, Eysenbach B, et al. Hiql: Offline goal-conditioned rl with latent states as actions. NIPS, 2023.

### Questions
1.Could you provide a theoretical analysis or at least a heuristic justification for why test-time training is effective in the offline RL setting?

2.Under what conditions does GC-TTT guarantee performance improvement, and in what situations might it fail?

3.If the pretrained policy is already locally optimal, how does test-time training avoid overfitting to the small adaptation dataset?

4.What is the rationale behind choosing percentile q=0.2? How sensitive is the method to different values of q? What are the principles for choosing H? Can it be adapted dynamically?Could you provide a detailed comparison between H-step return, Monte Carlo return, and TD(λ)?

5.Could the authors provide a comparison of different TTT strategies in Table 2 under the offline GCRL setting?

6.Have the authors examined the impact of using different distance functions on the results?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a Test-Time Training (TTT) framework to improve the performance of goal-conditioned offline Reinforcement Learning (RL) agents. The authors argue that existing offline goal-conditioned methods often underfit individual goals and propose dynamically fine-tuning policies at test time to mitigate this issue. The authors propose an approach that leverages test-time policy updates by sampling goal-related experiences from an offline dataset. Specifically, the paper introduces a data selection methodology based on two key criteria: Relevance and Optimality of trajectories. This method allows for the dynamic fine-tuning of the pre-trained policy during evaluation using only a subset of the original dataset that is most pertinent to the agent's current state and target goal. Through extensive experiments, the authors demonstrate that applying this Test-Time Training (TTT) framework significantly enhances the performance of various goal-conditioned offline RL algorithms backbones.

### Strengths
1. Significant Performance Improvement: The paper demonstrates substantial performance gains over established backbones, indicating the effectiveness of the proposed approach.
2. Well-Justified Data Selection Methodology: The methodology for extracting goal-related experiences is thoroughly explored and empirically justified, supported by detailed experimental analysis.
3. Versatile Methodological Contributions: The work offers a versatile set of methodological contributions, including a value-based approach and a value-free alternative for scenarios where expert data is available, thereby broadening its applicability.

### Weaknesses
1. Concerns Regarding Practicality and Control Latency: The paper claims the proposed algorithm is practical and suitable for robotic control environments, citing an acceptable average control frequency. However, the algorithm's update mechanism involves $N$  updates occurring every $K$-horizon steps, which introduces episodic delays rather than a uniformly distributed average delay. In real-time control systems, such intermittent but significant delays can pose a greater challenge than a consistent average frequency might suggest. The validity of justifying the practicality solely based on average frequency is questionable, as it does not adequately address the non-uniform nature of these computational latencies, which could be critical in safety-critical applications.
2. Insufficient Justification for TTT in Goal-Conditioned Offline RL: The authors suggest (line 78) that GC-TTT improves performance by addressing a "systematic underfitting with respect to individual goals" in existing offline goal-conditioned methods. However, this claim appears to be contradicted by the empirical results in Figure 6, where increasing model scale—a common approach to mitigate underfitting—does not lead to performance improvements and, in some cases, even shows a decrease. If underfitting is indeed the core problem necessitating TTT, one would expect larger models to alleviate this issue. Further discussion is needed to reconcile why larger models do not reduce underfitting in this context, or to elaborate on the specific nature of the underfitting that TTT uniquely addresses, beyond what model scaling can achieve.

### Questions
1. The paper argues for "systematic underfitting," yet Figure 6 shows model scaling does not improve performance. Could the authors elaborate on the nature of the underfitting that GC-TTT aims to address, and explain why increasing model capacity fails to mitigate it?
2. Given that higher TTT frequency generally improves performance (Figure 6), why are K and N (TTT frequency) hyperparameters tuned differently and not uniformly high across environments in Table 5? What is the rationale for these task-specific settings?
3. A dedicated ablation study on Horizon K is missing. Could the authors investigate how varying Horizon K independently impacts performance and alignment with dataset trajectories, beyond its inverse relationship with TTT frequency?
4. The repeated re-initialization to pre-trained policy parameters at every K-horizon step raises concerns. Unlike offline-to-online or other test-time adaptive methods [1] that pursue continuous policy improvement, this mechanism appears to prevent sustained policy enhancement. Could the authors discuss the implications for long-term learning and potential solutions for cumulative improvement?

[1] Xu, Shoukai, et al. "Test-time Adapted Reinforcement Learning with Action Entropy Regularization." ICML2025.

### Soundness
2

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
3

### Summary
This paper proposes a test-time fine-tuning framework for goal-conditioned reinforcement learning, which adapts the policy during evaluation by updating it on a subset of data that is both relevant to the current state and optimal for the current goal. Specifically, the fine-tuning dataset consists of trajectory segments whose distance to the current state is within a certain threshold and whose estimated return exceeds a predefined threshold. Experiments on a wide range of goal-conditioned tasks demonstrate that the proposed method can be effectively combined with standard offline goal-conditioned RL algorithms, leading to performance improvements.

### Strengths
- The motivation of improving the performance on the current goal through test-time fine-tuning is intuitively sound and reasonable.
- The method is well described, simple, and easy to understand.
- The experiments are comprehensive, and the reported results are convincing.

### Weaknesses
- The main technical contribution lies in the data selection mechanism for fine-tuning, which appears somewhat limited in technical novelty.
- The choice of the key parameter $\epsilon$ and its effect on the algorithm’s performance are not discussed.
- A potential disadvantage of modifying model parameters during deployment is the risk of instability, especially in time-sensitive control tasks where gradient updates may cause the model collapse which cannot be validated before use.

Minor：
- Test-time training is conceptually related to meta-RL, and discussing this connection could make the paper more complete.

### Questions
- Table 1 and Figure 5 lack results for the *no-critic* variant on play datasets. Although the model doesn’t benefit from the optimality of sub-trajectories, relevance may still provide performance gains. It would be interesting to see whether performance improves and by how much.
- The performance gain on the *CubeSingle* environment is relatively small; It would be helpful to provide some explanations.
- As shown in the comparison among GC-BC, GC-IQL, and SAW, improvements during the training stage often lead to substantial performance gains, with SAW performing well across most environments. Would it be better to solve the underfitting issue with respect to specific goals by enhancing the training phase rather than relying on test-time training, which could help avoid its higher computational cost?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes goal-conditioned test-time training (GC-TTT) for offline reinforcement learning. Instead of freezing a pre-trained goal-conditioned policy at inference, the method repeatedly fine-tunes it during evaluation on a small batch of goal-relevant, high-quality sub-trajectories drawn from the original offline dataset. Relevance is defined by proximity to the current state and optimality by an H-step return estimate using a learned critic; a critic-free variant uses reward returns on expert data. The updates are applied in a receding-horizon loop that resets weights every K steps, yielding consistent gains across OGBench loco-navigation and a manipulation task on top of GC-BC, GC-IQL, and SAW backbones. The paper further studies compute allocation at inference and shows that spending compute on GC-TTT outperforms simply scaling model size at matched inference FLOPs.

### Strengths
* The core idea is simple, broadly applicable, and well-motivated by analogies to test-time adaptation in foundation models. The algorithmic design is clear: a principled data filter that combines state relevance with estimated return, plus a lightweight fine-tuning loop with periodic resets; the paper explains these choices and provides an ablation showing that both relevance and optimality are needed for the observed gains.

* Empirically, the method consistently improves success rates across pointmaze, antmaze, humanoidmaze, and cubesingle when layered on GC-BC, GC-IQL, and SAW, sometimes converting weak pre-trained policies into strong ones with only a few gradient steps. The critic-free variant is a practical bonus for pure imitation settings with expert data. The compute analysis is thoughtful, reporting control frequency, profiling overheads, and showing that increasing TTT frequency often buys more than enlarging the backbone under matched FLOPs.

### Weaknesses
* The approach assumes convenient access to the entire pre-training dataset at test time, plus fast retrieval by state proximity; the paper does not quantify memory and I/O costs or the latency of building and querying this index online.   

* Several comparisons could be tighter. The compute comparison with model scaling relies on simplifying assumptions about FLOPs and may understate benefits of width scaling with better regularization; sensitivity analyses are limited. Some backbones are already near-saturated on certain tasks, and the absolute gains there are modest. Results use three seeds and fixed goals; it would help to include broader goal sets and confidence intervals or statistical tests. Finally, performance depends on critic quality and hyperparameters like K, N, and the percentile threshold; while there are ablations, guidance for robust default settings under distribution shift is limited.

### Questions
* How is state proximity computed in practice for high-dimensional observations and what indexing structure is used for fast retrieval at test time? Please report memory footprint, average query latency, and end-to-end wall-clock overhead per episode on your hardware.
  
* How sensitive is performance to the percentile threshold q and distance threshold, especially when relevant data are scarce or when the current state lies outside the support of D? Do you have a fallback for the no-data case beyond widening the threshold?
  
* Can you clarify the stability of the critic during test-time adaptation? Since the actor is updated but the critic guides data selection, is there drift that degrades the H-step estimate over repeated loops, and would periodic critic refresh help?
  
* In the compute study, could you include alternative baselines that also spend test-time compute, such as nearest-neighbor action retrieval, dynamic evaluation on behavior-cloned policies, or short-horizon planning with learned dynamics, matched for wall-clock and FLOPs?
  
* What happens if you also incorporate freshly collected transitions into the fine-tuning buffer, possibly with conservative weighting, so GC-TTT becomes a hybrid test-time online learner? Any early results or pitfalls you observed?

### Soundness
3

### Presentation
3

### Contribution
3
