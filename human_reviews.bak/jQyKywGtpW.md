# Offline-to-Online Reinforcement Learning with Prioritized Experience Selection

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Offline-to-online reinforcement learning (O2O RL) offers a promising paradigm that first pre-trains an offline policy and fine-tunes it with further online interactions. Nevertheless, the distribution shift between the offline and online phase often hinders the fine-tuning performance, sometimes even incurring performance collapse. Existing methods mitigate this by enhancing training robustness with Q-ensemble, training a density ratio estimator to balance offline and online data, etc. But they often rely on components like ensemble and have higher training costs. In this paper, we address this issue by establishing a concrete performance bound for the optimal policies between two consecutive online steps. Motivated by the theoretical insight, we propose a simple yet effective fine-tuning method, \textbf{P}rioritized \textbf{E}xperience \textbf{S}election (PES). During the online stage, PES maintains a dynamically updated priority queue containing a portion of high-return trajectories, and only selects online samples that are close to the samples in the queue for fine-tuning. In this way, the distribution shift issue can be mitigated and the fine-tuning performance can be boosted. PES is computationally efficient and compatible with numerous approaches. Experimental results on a variety of D4RL datasets show that PES can benefit different offline and O2O RL algorithms and enhance Q-value estimate. Our code is available and will be open-source.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed the Prioritized Experience Selection (PES) for offline-to-online RL. During the online stage, PES maintains a priority queue for high-return trajectories, and only selects online samples that are close to the samples in the queue for fine-tuning. PES shows strong performances in experiments on the D4RL benchmark.

### Strengths
- The writing is clear and the paper is easy to follow.
- The proposed method is simple and can be combined with different algorithms.
- PES shows promising results in the experiments.

### Weaknesses
- Background Section is too simple.
- Line 34: "learns the optimal policy from a previously collected dataset" is incorrect. We cannot learn an optimal policy from a low-quality offline dataset.
- Missing recent baselines: BOORL (https://github.com/YiqinYang/BOORL), RLPD (https://github.com/ikostrikov/rlpd), SO2 (https://github.com/opendilab/SO2), FamO2O (https://github.com/LeapLabTHU/FamO2O).
- PES needs to search a k-nearest neighbours in the queue when we have a new online transition, which increases the computational cost. Further, it also introduces extra hyper-parameters, i.e., the size of the queue and the neighbour size k.
- Insufficient details are provided for readers to understand the specifics of the two baselines in Section 3.1. 
- Theorem 3.1 is less practical. Firstly, the difference of empirical MDP for two consecutive steps are usually negligible in practice. Further, a more close related theory analysis should focused on the distribution shift between the online policy and the data in the replay buffer. How does distribution shift influence the value & policy learning is not analyzed.
- Line 185: "and the sample efficiency can be boosted (since only high-quality samples are selected)". The description is misleading. We use less samples during the training, but we still have the unused samples during the rollout. From the perspective of environmental steps, we still have the same number of total samples.
- Line 186: "the computational load is also significantly reduced ..." is also inaccurate. PES mainly saves memory usage, but the computational cost (FLOPs) is not reduced.
- The effectiveness of PES is not stable. Sometimes adding PES make the performance worse and sometimes PES can not improve the performance.

### Questions
- In the abstract, "In this way, the distribution shift issue can be mitigated...". The queue only contains high-return trajectories, which corresponds to a narrower state distribution. The distribution shift issue is mitigated only when the online policy can generate high-return trajectories as in the queue. However, during the offline-to-online fine-tuning the online policy is dynamic and it is not always a high-performing one. I think the distribution shift issue is not guaranteed to be mitigated.
- Line 67: "we make sure that the samples used for fine-tuning stay close to the previously encoutered samples". We can also achieve this by using a smaller FIFO buffer to only store online samples. So there is a missing baseline: fine-tune the agent with a smaller FIFO buffer that only stores online samples. 
- What's the detail of IQL-Expert baseline in Section 3.1? I did a quick experiment to fine-tune an IQL agent, which is first pretrained on the halfcheetah-medium-v2 dataset, with the halfcheetah-expert-v2 dataset for 1e6 steps. The curve is very different from IQL-Expert in Figure 1(right). In my experiment, the normalized score increases from 47 to 94. Also, there is a severe distribution shift issue in my experiment. However, this does not prevent the agent from improving its performance from 47 to 94. Therefore, the main point of this work on the problem of distribution shift is not complete. The claim in Section 3.1 is not correct as well. Distribution shift is not always harmful. We can still train a high-performing policy even with the distribution shift issue. A deeper discussion about when it hurts and when it doesn't would increase this paper's contribution. 
- If we use pixel-based inputs, will the computational cost to find k-nearest neighbors be large? How does the proposed method perform with pixel-based inputs, i.e., final performance and computational cost?
- Line 71: "we adapt the selection threshold throughout the online phase to ensure data diversity". How can we prove this point? Any experiments to support this claim?

### Soundness
2

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
This paper introduces Prioritized Experience Selection (PES) to address the distribution shift challenge in offline-to-online reinforcement learning (O2O RL). PES selectively uses high-return samples from online interactions to fine-tune the policy, mitigating performance degradation and enhancing learning efficiency. Unlike prior methods that rely on complex models or ensembles, PES is lightweight and adaptable to various algorithms. Experiments demonstrate that PES consistently outperforms traditional approaches across multiple O2O RL benchmarks, achieving stable and improved performance.

### Strengths
- By selectively using high-return samples, PES helps prevent performance degradation (unlearning) and enhances overall learning stability.
- PES doesn’t require complex models or ensemble methods, making it lightweight and computationally efficient compared to other approaches.
- PES is adaptable and can be easily integrated into various offline-to-online reinforcement learning algorithms, improving their effectiveness without altering core structures.
- By focusing on the most relevant online samples, PES allows policies to quickly adapt to new data and dynamic environments.

### Weaknesses
- The sample selection criteria for PES appear somewhat naive, based solely on the distance (difference) between samples in the priority queue. It seems unlikely that a distance-based comparison can consistently guarantee the selection of good samples. Potentially promising new samples might be filtered out due to this distance criterion, which could lead to missed learning opportunities through exploration, ultimately limiting long-term performance.
- The base score of Cal-QL [1] appears unusual. When examining the Cal-QL graph from the original Cal-QL paper, both CQL and Cal-QL achieve scores above 80 in the Antmaze Large environment after 1M finetuning, which aligns with my experience running the official Cal-QL code. Could there be a reason why the Cal-QL score for Antmaze is so low?
- It’s promising that this algorithm performs well even without using an ensemble. However, I wonder if performance could be significantly enhanced by substituting only the balanced replay with PES in configurations that already use both an ensemble and balanced replay. With advancements like JAX accelerating RL training, computational demands might reduce over time. Although achieving strong performance without an ensemble is beneficial, assessing whether a small ensemble combined with the proposed method could improve overall results would ultimately be valuable for the community. From this standpoint, it’s worthwhile to consider integrating with offline-to-online algorithms like Off2on [2], which, despite some computational demands, achieve high performance.
- Furthermore, studies like ReBRAC [3] demonstrate that integrating multiple techniques can yield strong performance, particularly in environments like Antmaze, even without dedicated offline-to-online methods. I am curious about how effective PES might be within these latest algorithms.
- In Figure 3, the IQL graph shows a relatively meaningful improvement with IQL+PES over IQL, but the results in Table 1 don’t convey an overall impactful difference. Although some gains are observed, there are also declines, making it challenging to determine if the proposed method offers a clear advantage.

(Minor)

- The first paragraph of the introduction appears to contain an excessive amount of information. Breaking it down into 2–3 shorter paragraphs could enhance readability.


&nbsp;

[1] Nakamoto, Mitsuhiko, et al. "Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning." Advances in Neural Information Processing Systems 36 (2024).

[2] Lee, Seunghyun, et al. "Offline-to-online reinforcement learning via balanced replay and pessimistic q-ensemble." Conference on Robot Learning. PMLR, 2022.

[3] Tarasov, Denis, et al. "Revisiting the minimalist approach to offline reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).

### Questions
- I am curious about the reason for the low base score of Cal-QL.
- What would the results be if only the BER in Off2on were replaced with PES?
- What would the outcome be if the proposed method were applied to ReBRAC or other recent SOTA algorithms?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
To address the distribution shift issue in offline-to-online reinforcement learning (O2O RL), this paper proposes an O2O RL method called Prioritized Experience Selection (PES). The main idea of PES is to maintain a priority queue that contains a selection of high-return trajectories and only select online samples that are close to the samples in the queue for fine-tuning. Additionally, to avoid being overly conservative, the priority queue is updated each time a trajectory is sampled from online interactions. Finally, to evaluate the effectiveness of PES, the authors provide empirical comparisons between various offline RL methods and their corresponding O2O methods, which integrate PES into each offline method.

### Strengths
PES successfully addresses the distribution shift issue in O2O RL.

### Weaknesses
1. The experimental results provided do not include empirical results for other O2O RL methods (e.g., [1] and [2]).
2. Although PES consists of five steps, no theoretical explanation is provided.
3. PES appears to be sensitive to the hyperparameters (N, $\alpha$, $\eta$).

- References:

    [1] Tarasov, Denis, et al. "Revisiting the minimalist approach to offline reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).

    [2] Wang, Shenzhi, et al. "Train once, get a family: State-adaptive balances for offline-to-online reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).

### Questions
1. Compared to other O2O RL methods, what is the main advantage of PES?
2. Is there any comparison between the resulting policy of offline RL methods (after Step 1 in Algorithm 1) and PES? If not, could you provide such a comparison to assess the effect of PES?
3. In Step 2 of Algorithm 1, could you provide an empirical evaluation of the quality of the priority queue? I am curious whether the performance improvement by PES is related to the initial quality of the priority queue.

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
This paper studies the offline-to-online (O2O) RL problem from the perspective of experience replay. Specifically, this paper proposes prioritized experience selection (PES), which maintains a priority queue containing high-return trajectories and selects only similar online samples for fine-tuning to mitigate distribution shift. Such similarity is enforced via finding k-nearest neighbors. Moreover, PES can be integrated with various offline and O2O RL methods for online fine-tuning. Experimental results on various tasks in D4RL demonstrate that PES (with IQL as the base algorithm) can achieve good fine-tuning performance and can serve as a general add-on for various offline RL base algorithms. Some ablation studies are also provided to justify the design of the prioritization experience selection scheme.

### Strengths
- The proposed method is conceptually simple and quite general in the sense that it can be integrated nicely with various offline RL algorithms.
- The proposed PES archives fairly strong performance in several locomotion and maze environments (despite that some O2O baselines are not included, please refer to the comments below).
- Overall the paper is well written and easy to follow.

### Weaknesses
- On the high level, I find the idea behind PES to be rather similar to Balanced Replay (Lee et al., 2022), which also manages to select nearly on-policy samples for fine-tuning based on the density ratios. As a result, it remains unclear to me why PES can significantly outperform Balanced Replay in most of the tasks in FIgure 3. Moreover, in Figure 3, the scores of Balanced Replay appear to be much lower than those reported in the original paper (Lee et al., 2022). It would be helpful to clarify the discrepancies in the experiments and provide the intuition / justification why PES is better than Balanced Replay.

- Regarding the experiments, there are several recent baselines that are missing in the comparison. For example, RLPD (Ball et al., 2023) is a pretty strong O2O baseline on D4RL. Moreover, Cal-QL (Nakamoto et al., 2023) learns a calibrated value function initialization and also serves as a very competitive benchmark method in the context of O2O. Online decision transformer (Zheng et al., 2022) also achieves fast fine-tuning progress on both MuJoCo and AntMaze tasks. FamO2O (Wang et al., 2023) address the distribution shift in O2O without relying on Q ensembles. It would be helpful to include the above baselines into the comparison to showcase the effectiveness of PES.

- While I can appreciate the empirical results in Section 4.2, which shows that PES can be integrated with various offline RL methods, I do find the comparison to be somewhat not very conclusive. With the additional online data, it is not surprising that PES can achieve improved normalized scores compared to the vanilla offline methods. To show that PES is indeed a better add-on technique for O2O RL, baselines (such as Balanced Replay + vanilla base algorithm) shall be included in Table 1.

- One major component of PES is to find $k$ nearest neighbors for selecting the online samples. This design implicitly assumes that the L2 distance directly reflects the similarity between state-action pairs. This assumption may hold in some RL tasks (like the locomotion tasks in D4RL), but would not hold in other environments, (e.g., maze, where two state-action pairs with a small L2 distance can be hard to reach from each other). 

- Theorem 3.1 appears somewhat disconnected from the PES algorithm. The implication of Theorem 3.1 is that the optimal (simulated) total return performance under the two empirical MDPs would be similar if the two empirical MDPs are similar. However, this does not imply that the learned policies have similar performance in the true environment.
 
- Regarding the motivating example in Section 3.1: Figure 1 is meant to illustrate the severe distribution shift issue that can lead to performance degradation during online fine-tuning. That said, as far as I know, this issue has been pointed out by some prior works, such as (Nakamoto et al., 2023) and (Wen et al. 2023).

References:
- (Ball et al., 2023) Philip J. Ball, Laura Sith, Ilya Kostrikov, and Sergey Levine, “Efficient online reinforcement learning with offline data,” ICML 2023.
- (Nakamoto et al., 2023) Mitsuhiko Nakamoto et al., "Cal-QL: Calibrated offline RL pre-training for efficient online fine-tuning," NeurIPS 2023.
- (Zheng et al., 2022) Qinqing Zheng, Amy Zhang, and Aditya Grover, “Online decision transformer,” ICML 2022.
- (Wang et al., 2023) Shenzhi Wang, Qisen Yang, Jiawei Gao, Matthieu Lin, Hao Chen, Liwei Wu, Ning Jia, Shiji Song, and Gao Huang, “Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning,” NeurIPS 2023.
- (Wen et al., 2023) Xiaoyu Wen, Xudong Yu, Rui Yang, Chenjia Bai, and Zhen Wang, “Towards robust offline-to-online reinforcement learning via uncertainty and smoothness,” 2023.

### Questions
- As mentioned above, it would be helpful to clarify the discrepancies in the experiments and provide the intuition / justification why PES is better than Balanced Replay.
- Regarding the experiments, there are several recent baselines that are missing in the comparison.
- The design about $k$ nearest neighbors seems to already involve some underlying assumption on the geometry of the environment. Can the authors comment on this?
- The implication of Theorem 3.1 on PES appears unclear. 
- About the motivating example: It would be helpful to explain the connection with the prior works on the distribution shift issue.
- To show that the results of Table 1 are statistically significant, could the authors also report the error bars?

### Soundness
3

### Presentation
3

### Contribution
2
