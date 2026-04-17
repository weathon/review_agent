# Peak-Return Greedy Slicing: Subtrajectory Selection for Transformer-based Offline RL

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
Offline reinforcement learning enables policy learning solely from fixed datasets, without costly or risky environment interactions, making it highly valuable for real-world applications. While Transformer-based approaches have recently demonstrated strong sequence modeling capabilities, they typically learn from complete trajectories conditioned on final returns. To mitigate this limitation, we propose the Peak-Return Greedy Slicing (PRGS) framework, which explicitly partitions trajectories at the timestep level and emphasizes high-quality subtrajectories. PRGS first leverages an MMD-based return estimator to characterize the distribution of future returns for state-action pairs, yielding optimistic return estimates. It then performs greedy slicing to extract high-quality subtrajectories for training. During evaluation, an adaptive history truncation mechanism is introduced to align the inference process with the training procedure. Extensive experiments across multiple benchmark datasets indicate that PRGS significantly improves the performance of Transformer-based offline reinforcement learning methods by effectively enhancing their ability to exploit and recombine valuable subtrajectories.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript proposes Peak-Return Greedy Slicing (PRGS), a framework designed to enhance Transformer-based offline reinforcement learning (RL). The method introduces a fine-grained, timestep-level trajectory segmentation mechanism that selects high-quality subtrajectories instead of relying solely on full trajectories. PRGS consists of three components: (1) an MMD-based return estimator that models the distribution of potential future returns for each state-action pair and yields optimistic estimates (Sec. 3.1, Eq. (2–4)); (2) a greedy slicing algorithm that recursively partitions trajectories based on peak optimistic returns (Sec. 3.2, Eq. (5)); and (3) an adaptive history truncation mechanism for evaluation, ensuring alignment between training and inference (Sec. 3.3, Eq. (6)). Experiments on diverse benchmarks (D4RL, BabyAI, AuctionNet) demonstrate consistent performance improvements, averaging 15.8 % gains over Transformer-based baselines (Sec. 5.1; Table 1).

### Strengths
- The writing and presentation of this paper are intuitive and clear. The paper effectively explains the motivation behind the method’s design. I can clearly understand the three components of the approach — the evaluation of the return distribution, the trajectory slicing mechanism, and the adaptive history truncation during deployment — as well as the rationale behind each of them.
- The experiments in the paper are thorough and align well with the authors’ claims. Applying PRGS to Transformer-based offline methods such as DT yields noticeable performance improvements, demonstrating the effectiveness of the proposed approach. Some ablation studies, such as examining the impact of the number of particles used in return distribution estimation, also correspond well with the authors’ design intentions.

### Weaknesses
- Although PRGS improves the performance of Transformer-based offline algorithms, its overall performance is not particularly outstanding. The compared RL-based baselines include only CQL, IQL, BEAR, and TD3+BC, which are not among the most recent RL-based methods. To my knowledge, the state-of-the-art performance in offline RL is significantly higher than what is reported in the paper.
- Intuitively, the effectiveness of the PRGS method is likely to depend on the diversity of the dataset. If the data coverage is narrow—for example, consisting entirely of expert demonstrations—the number of peak-return points should be relatively small. The paper also lacks experiments on random and expert tasks, and the authors do not show how different types of datasets affect the number of peak-return points.
- Because dataset coverage is often incomplete, value estimation in the offline setting typically suffers from significant bias and even overestimation. Although the paper estimates the return distribution rather than a single value, this issue is still likely to occur. The authors, however, do not appear to provide any analysis or mitigation of this problem.
- PRGS is somewhat more complex than standard Transformer-based methods, yet the authors do not provide any experimental comparison of its computational cost in terms of runtime or GPU memory consumption.

### Questions
- How does PRGS perform on datasets with narrow coverage, such as random or expert datasets?
- Why are the comparisons with QDT, EDT, and CGDT conducted only on the medium-replay and medium tasks, rather than across all datasets?
- How biased is the return distribution estimation? Could it lead to overestimation? Would optimistic estimation make the overestimation even more severe?
- How does PRGS perform in terms of time and GPU memory consumption? Could you provide a detailed comparison with other algorithms?
- What would the performance be if we do conventional Q estimation rather than estimating the return distribution?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Peak-Return Greedy Slicing (PRGS) to enhance Transformer-based Offline Reinforcement Learning. PRGS uses an MMD-based return estimator and greedy slicing to extract high-quality subtrajectories instead of relying only on complete ones. This method significantly improves performance by better exploiting and recombining valuable data segments.

### Strengths
1) Well motivated. The authors propose to utilize high quality subtrajectories to improve performance of transformer based RL.
2) Results seem good. The proposed method shows strong improvement on reward.

### Weaknesses
1) The core strength of the proposed method appears to be its ability to capture and utilize peak intermediate rewards within subtrajectories, leading to the reported strong average reward improvements. However, the use of subtrajectories that might originate from ultimately failed full trajectories introduces a potential ambiguity. Since the primary metrics presented in the paper are based on reward, it would significantly enhance the clarity and robustness of the results if the authors could provide the success rate for the respective tasks.

2) As depicted in Figure 3, the training procedure involves masking low quality tokens, leading the Transformer to learn intermediate high quality tokens conditioned on low quality segments. A potential concern is that this mechanism might inadvertently encourage reward hacking: the policy may learn to exploit the high intermediate reward signal without consistently leading to final task success. Specifically, the model might optimize for peak reward segments from trajectories that ultimately terminated in failure. How about to implement a filtering mechanism to exclude subtrajectories (state-action pairs in the env) that can not reach the final success?

### Questions
See weaknesses,

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a framework to enhance the performance of transformer-based offline RL methods. Peak-Return Greedy Slicing, helps enable discovery of high quality subtrajectories, and in turn do BC on those subtrajectories. The method first trains a Q function with MMD, which approximates the distribution of Q values. Then, a trajectory is partitioned into subtrajectories by recursively determining peak Q values, and then do BC on each subtrajectory. During inference time, the history is truncated whenever the value increases.

### Strengths
- the paper is very well written, and does a great job at describing the method and walking through the experiment results. The plots are well made to help with understanding the paper.
- The experiment section provides extensive number to validate PRGS, and shows the method works well across diverse benchmarks. There are also sufficient baseline to compare the method against.
- I appreciate the visualization in section 5.4, which help with intuition of the method

### Weaknesses
- It would be great to offer some discussion on the failure modes of this method. For example, consider a subtrajectory where there are multiple part where the values go down, but eventually the values rise to the “peak” of the episode. When we do BC on this trajectory, don’t we then inherently learn to “clone” suboptimal behavior? How does PRGS deal with this?
- it is unclear why we need optimistic Q values. Traditionally, offline RL Q function needs to be conservative to prevent overestimation. Figure 4 does show empirically that optimistic Q values seem better. But since this seemingly goes against traditional wisdom (though I can think of why it’s different for transformer-based offline RL), it would be nice to include some intuition and explanation in the paper.
- it is unclear to me why in Section 3.3, the authors chose to remove history when the value increases. During training, doesn’t the value of mostly increase for each subtrajectory? Then doesn’t this cause a train-test mismatch, when at training time history is kept when values increase and at test time history is removed when values decrease?
- nitpick: section 2.2 could have highlighted more the difference between transformer-based offline RL and traditional offline RL

### Questions
- why is the sum of return not discounted in the definition for “aligned optimistic return” at the bottom of page 4? The Q values are learned with discount, so doesn’t this favor returns toward the start of the trajectory?
- in section 5.3, how do you filter for the top 10% or 20% of the trajectories?

### Soundness
3

### Presentation
4

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
The paper presents Peak-Return Greedy Slicing (PRGS), a framework for improving the learning process of Transformer-based architectures in offline deep reinforcement learning. The main idea is to identify high-quality sub-trajectories and use them for higher quality training of the model. The sub-trajectories are chosen by identifying points in the trajectory where the overall reward (obtained+predicted) begins to decline. The authors use a mechanism to "reconcile" trajectory history with the sub-trajectories they use. Extensive evaluation on multiple datasets and environment types shows a generally consistent improvement.

### Strengths
1) The timestep-level sub-trajectory slicing approach is novel, simple and elegant. The analogy to a "human" way of learning is clear.
2) The approach is relatively model agnostic, and can be used in multiple offline methods.
3) The evaluation is comprehensive, consisting of multiple environments and algorithms.

### Weaknesses
1) There is no discussion of the computational complexity of the approach. The slicing and MMD estimator are likely to require non-trivial computational resources.
2) The subject of training trajectory quality is likely to be a key element in the success or failure of the proposed approach. The lack of discussion and analysis of this point is problematic.
3) There is a lack of analysis regarding to how often the proposed approach actually takes place: in what percentage of trajectories does slicing take place? what is the average length of a sliced trajectory? what is the impact of history truncation?

### Questions
I invite the authors to respond to the points I raised in the "weaknesses" section.

### Soundness
3

### Presentation
3

### Contribution
3
