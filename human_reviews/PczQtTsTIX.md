# CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity

- Decision: Accept (spotlight)
- Scores: 6, 8, 6, 8

## Abstract
Sample efficiency is a crucial problem in deep reinforcement learning. Recent algorithms, such as REDQ and DroQ, found a way to improve the sample efficiency by increasing the update-to-data (UTD) ratio to 20 gradient update steps on the critic per environment sample.
However, this comes at the expense of a greatly increased computational cost. To reduce this computational burden, we introduce CrossQ:
A lightweight algorithm for continuous control tasks that makes careful use of Batch Normalization and removes target networks to surpass the current state-of-the-art in sample efficiency while maintaining a low UTD ratio of 1. Notably, CrossQ does not rely on advanced bias-reduction schemes used in current methods. CrossQ's contributions are threefold: (1) it matches or surpasses current state-of-the-art methods in terms of sample efficiency, (2) it substantially reduces the computational cost compared to REDQ and DroQ, (3) it is easy to implement, requiring just a few lines of code on top of SAC.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present CrossQ, a variation of SAC that removes target networks and leverages Batch Normalization to improve learning efficiency while avoiding high update-to-data ratios. CrossQ presents a straightforward modification of SAC and is experimentally validated on a selection of tasks from the OpenAI Gym.

### Strengths
-	The changes over SAC are straight-forward and easy to implement, simplicity being a great asset in RL
-	The approach removes target network dynamics as a step towards decluttering RL
-	Improved learning efficiency over the baselines on the selected tasks
-	Several interesting ablation studies

### Weaknesses
- The evaluation looks promising but some experiments are missing to yield a well-rounded study. For example, CrossQ updates the Adam momentum and critic width compared to REDQ and DroQ. How do REDQ and DroQ compare to CrossQ when we set their (1) Adam momentum to 0.5, (2) critic width to 2048, (3) momentum to 0.5 and width to 2048? To reduce compute requirements, this could be studied for e.g., 5 seeds on Humanoid + HumanoidStandup + 1 other task
- The concept should be sufficiently general to combine with other baseline algorithms than SAC, so evaluation across multiple algorithms would further broaden applicability [i.e.are the results specific to SAC variations?]
- Generally, a more diverse set of environments would further strengthen the evaluation – this could include other domains (DeepMind Control Suite / MetaWorld / …) or visual control tasks [i.e.are the results specific to OpenAI Gym domains?]

### Questions
- How does CrossQ compare to REDQ and DroQ under the experiments mentioned above?
- Have you tried removing the double Q functions on top of removing the target networks?
- Have you tried keeping the target network and normalize with batch norm parameters of the regular network?
- Do you have an intuition for why CrossQ (black line) performs worse on HalfCheetah in Figure 7?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors address the efficiency of Soft Actor-Critic (SAC) and related deep RL methods for continuous control. Recent methods such as REDQ and DroQ have large update-to-data (UTD) ratios when training the critic (as high as 20). The authors introduce a new method called CrossQ that

1. reduces the critic's UTD to 1,
1. uses batch normalization,
1. removes target networks, and
1. increases hidden-layer widths.

It is shown that these changes are still able to achieve high sample efficiency in MuJoCo environments while being significantly cheaper to run. It is also hypothesized that two of the architectural changes (batch norm and no target networks) are what enable stable training with such a low UTD ratio.

**Score raised from 5 to 8 during rebuttal.**

### Strengths
- Good empirical results in MuJoCo. The tested baselines, REDQ and DroQ, are recent and strong baselines yet CrossQ still performs equally or better in the tested environments.
- Removes the target networks of SAC. This is a pain point in terms of both learning speed and implementation, and this is the first time (that I know of) that it has been removed without hurting performance. The authors provide an interesting and sensible hypothesis about the interaction between batch normalization and target networks in Section 3.2 to support their empirical observations. It would be exciting if this generalizes to other algorithms as well.
- The total computation and wall-clock execution time of the proposed method is significantly less than baseline methods, yet sample efficiency is not sacrificed. Along with its simplicity, the new method is very appealing.
- The paper is well written and the discussion of recent related papers is excellent.

### Weaknesses
- The paper claims it “show[s] the first successful application of BatchNorm in off-policy Deep RL.” However, it appears that DDPG [1] also used batch normalization 7 years ago in the exact same setting: off-policy deep RL for continuous control in MuJoCo. This means the titular contribution of the paper is not original.
- There is a lot of hyperparameter tuning that might be unfairly benefiting CrossQ. For instance, the network width for CrossQ is increased from 256 to 2048 units, an 8x increase. Adam’s momentum parameter was also reduced from 0.9 to 0.5. As far as I can tell, these hyperparameters were not tested for the baselines, so CrossQ has been over-optimized for MuJoCo. A fairer comparison would be testing the methods with the same network sizes and then selecting the best Adam hyperparameters for each of them.
- I feel that the claim of “state-of-the-art” performance is a bit strong given that only 6 MuJoCo environments were tested, and CrossQ is a clear improvement only in 2 of them. The method has not been evaluated in other continuous-control problems nor against other potential baselines, so I would not necessarily agree that it is state-of-the-art. Nevertheless, the empirical results are still great.

**Minor Edits**
- Inconsistent use of hyphens should be fixed. For example, “Q-function” and “Q function” both appear in the paper.
- Possible typo in Figure 2’s caption: “a **single double** Q function network”?
- As Equation 1 is currently written, it is not correct to refer to the Polyak averaging parameter $\tau$ as “momentum” since $\tau=1$ corresponds to a moving average with no inertia.
- References are not consistently formatted, e.g., some conference names are capitalized but others are not.

### Questions
1. At the end of the introduction, you say that your success with batch norm “contradicts” another paper [2] that did not find batch norm to work well. Why do you think you were able to achieve better results? Is it because you removed the target network, or is there another reason?
1. Figure 2’s caption says, “this removes the computational need for two individual forward passes through the same network” in reference to batching the observations and next observations together. But this doesn’t actually reduce computational cost, does it? The same number of forward passes are being done either way.
1. Could you explain the evaluation procedure in the Q-estimation bias experiment (Figure 6)?
1. Could you also explain what is meant by a policy delay of 3? How is this different from the UTD ratio?

**References**

[1] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. 2016.

[2] Takuya Hiraoka, Takahisa Imagawa, Taisei Hashimoto, Takashi Onishi, and Yoshimasa Tsuruoka. Dropout Q-functions for doubly efficient reinforcement learning. 2021.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes very simple adaptations to the well known SAC algorithm to make it more data and especially more compute efficient. While recent works focused on increasing UTD ratios to make more use of the collected data and thus becoming more data efficient at the cost of compute resources, the authors go the opposite way and show that even with UTD=1 competetive data efficiency is possible with only a fraction of the compute. The main ingredients for the new method are the addition of BatchNormalization, the removal of widely believed necessary target networks, different activation functions as well as wider critic networks.

### Strengths
The proposed method achieves competitive sample efficiency while needing only a fraction of the compute, as demonstrated in figures 1 & 5. The paper showcases some interesting findings to enable these advantages - target networks are widely believed to be necessary for stable learning, however the authors found a way around them by using BatchNorm in a novel fashion. Since target networks necessarily slow down learning, this is one part that enables the computationally more efficient learning process. The paper is well written and the major points come across clearly.

### Weaknesses
The paper makes very broad claims in terms of proposing the new state of the art in both sample and compute efficiency - they might be justified, however only relatively few baselines (DroQ, REDQ, SAC) are used for comparison. For statements like this, a broader comparison might be required. 

Further, even though I am no expert on the the achievable performance in the environments used, I know that at least for some of them higher final performances are possible - e.g. on HalfCheetah I have seen policies do much better than 10k (interestingly the D4RL paper even reports 12k for SAC, whereas in Fig 4. it looks like you only get about 7.5k) - even if other algorithms use more data / compute: if the superior efficiency were only attainable for suboptimal performances, that would be quite limiting and needs to be examined. Please clarify

If you could show some evidence on the suspicion that the reason for prior approaches being unable to use BatchNorm effectively is the OOD-ness of the actions from the policy that are not sampled from the batch in the target networks, that would be great as well since it is I think one of the key insights (i.e. visualize the distributions of sampled and policy actions).

### Questions
See weaknesses

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents CrossQ, an algorithm for deep reinforcement learning, aiming to address the crucial issue of sample efficiency. CrossQ achieves state-of-the-art sample efficiency, reduces computational costs significantly compared to existing methods (REDQ and DroQ), and simplifies the implementation process by eliminating the need for advanced bias-reduction schemes and high update-to-data (UTD) ratios. The core innovations of CrossQ include the removal of target networks, strategic use of Batch Normalization, and wider critic networks.

### Strengths
1. Well-written paper, easy to follow. The paper is well-structured and clearly articulates the motivation, methodology, and experimental results. The use of figures and code snippets aids in understanding the algorithm's implementation, making it accessible to readers with varying levels of expertise.

2. CrossQ's approach stands out in the field of deep reinforcement learning by challenging the trend of high UTD ratios. The innovative removal of target networks, combined with Batch Normalization and wider critic networks, is a departure from conventional methods, leading to improved sample efficiency and reduced computational complexity.

3. The paper demonstrates a rigorous empirical analysis of CrossQ's performance through ablation studies, clearly showcasing the algorithm's effectiveness. The use of Batch Normalization in a novel manner, coupled with wider networks, highlights a high-quality exploration of the algorithm's design space.

### Weaknesses
1. While the empirical results are compelling, the paper lacks theoretical analysis or justification for the success of CrossQ. Providing insights into why the proposed modifications lead to improved performance would enhance the paper's depth and contribute to a more comprehensive understanding of the algorithm.

2. The experiments are only conducted in one continuous control domain (MuJoCo). It is dangerous to draw conclusion of sample efficiency only from one experimental domain.

3. Also, this paper does not include image-based settings. There are tons of efficient RL works recently on image-based settings and some of the techniques (state augmentation [1, 2] and auxiliary losses [3, 4]) can be also applied to state-based settings. I think it is worth discussing the recent progress of efficient RL in the related work session or adding baselines for comparison.

[1] Laskin, M., Lee, K., Stooke, A., Pinto, L., Abbeel, P., & Srinivas, A. (2020). Reinforcement learning with augmented data. Advances in neural information processing systems, 33, 19884-19895.

[2] Yarats, D., Fergus, R., Lazaric, A., & Pinto, L. (2021). Mastering visual continuous control: Improved data-augmented reinforcement learning. arXiv preprint arXiv:2107.09645.

[3] Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., & Bachman, P. (2020). Data-efficient reinforcement learning with self-predictive representations. arXiv preprint arXiv:2007.05929.

[4] He, T., Zhang, Y., Ren, K., Liu, M., Wang, C., Zhang, W., ... & Li, D. (2022). Reinforcement learning with automated auxiliary loss search. Advances in Neural Information Processing Systems, 35, 1820-1834.

### Questions
1. Can you elaborate on the specific scenarios or domains where CrossQ might outperform other algorithms significantly? Are there any limitations or challenges in which CrossQ may not be the most suitable choice?

2. The paper focuses on empirical results; however, are there any insights or intuitions on why the chosen modifications (removal of target networks, Batch Normalization, wider networks) lead to the observed improvements? Providing additional context on the underlying mechanisms would enhance the paper's impact.

3. To me, I think using wider critic networks played a pivot role in the performance boost, especially in humanoid-based environments. Can you draw SAC+wider networks to Figure 7?

I would be happy to raise my scores if my concerns are addressed.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
