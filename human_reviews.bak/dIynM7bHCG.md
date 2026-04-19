# Beyond Conservatism: Diffusion Policies in Offline Multi-agent Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3, 3

## Abstract
We present a novel Diffusion Offline Multi-agent Model (DOM2) for offline Multi-Agent Reinforcement Learning (MARL). Different from existing algorithms that rely mainly on conservatism in policy design, DOM2 enhances policy expressiveness and diversity based on diffusion model. Specifically, we incorporate a diffusion model into the policy network and propose a trajectory-based data-augmentation scheme in training. These key ingredients make our algorithm more robust to environment changes and achieve significant improvements in performance, generalization and data-efficiency. Our extensive experimental results demonstrate that DOM2 outperforms existing state-of-the-art methods in all multi-agent particle and multi-agent MuJoCo environments, and generalizes significantly better to shifted environments (in $28$ out of $30$ settings evaluated) thanks to its high expressiveness and diversity. Moreover, DOM2 is ultra data efficient and requires no more than $5\%$ data for achieving the same performance compared to existing algorithms (a $20\times$ improvement in data efficiency).

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes DOM2, a decentralized training and execution Offline MARL approach using diffusion policies. Their algorithm combines CQL (critic) and DPM-Solver (actor) as well as data augmentation to increase the dataset size with good trajectories. DOM2 allows for better generalization to slightly shifted environments as well as improved data efficiency. DOM2 is evaluated on MPE and HalfCheetah-v2 and shows improved performance across all datasets, and even shows good performance on random datasets where all other previous methods fail.

### Strengths
1. The experimental results are impressive and it is especially surprising to see DOM2 perform well in random datasets across all tasks, which is challenging for conservatism-based offline RL algorithms.

2. While DOM2 uses decentralized training without any consideration of the non-stationarity in MARL (see Weaknesses), I find it interesting that decentralized training is able to perform well across all datasets and even exhibit multi-modal behavior.

### Weaknesses
1. Since the policy training loss in Eq. 5 is decentralized, there is no guarantee that DOM2 maximizes the global Q-function for the overall Dec-POMDP. The $Q(o_j, a_j)$ is just an individual utility function which ignores the non-stationarity of the environment due to the other agents’ policies being updated during training. Since many environments require policy dependence to be considered (see e.g. [1]), there should be some clear insight as to (a) what kinds of environments and (b) what specific aspect of DOM2/diffusion models in general can allow for the community to consider decentralized training.

2. The paper appears to be a relatively simple combination and application of existing work, namely CQL for the critic loss, DPM Solver for the diffusion policy. This is not a problem in and of itself but there is not insight or deeper analysis of the specific properties of DOM2/diffusion policies which makes it suitable for offline MARL. 

3. While the results on MAMuJoCo and MPE are impressive, DOM2 should be tested on more complex environments requiring agents to coordinate at a higher level e.g. Google Research Football, SMACv2.

4. The main contribution or key insight of DOM2 from the diffusion model perspective is not clear. The analysis below Algorithm 2 only refers to the architectural differences compared to other algorithms.

5. The claim that DOM2 is ultra data efficient is not convincing to me as data augmentation is included in the DOM2 algorithm but it seems the same augmentation technique could be used for other baselines.

[1] Revisiting Some Common Practices in Cooperative Multi-Agent Reinforcement Learning (Fu et, al. ICML 2022)

### Questions
1. If a Dec-POMDP is considered, why does the reward $r_j^t$ index on agent ID? Are different reward values given for each agent in the environment during the experiments as well?

2. Is there any insight regarding why decentralized training is enough for DOM2 perform well? For instance, it could be the case that (1) the environments considered are too simple or (2) some specific property of using diffusion policies make it such that there is some implicit dependence among policies or (3) continuous control environments in practice require less  dependence among policies. 

3. If the critic is trained using the CQL loss, then it seems that the Q values will just be conservative to OOD actions. This means that in order to produce policy diversity, the dataset must already contain the diverse behavior. Is my understanding correct here? I also considered the possibility that data augmentation helps with behavior diversity but Figure 6 suggests that it is not crucial. 

4. As far as I can tell, the data augmentation technique seems orthogonal to the DOM2 algorithm itself. If that is the case, shouldn’t all baselines also include the data augmentation technique? Is it really possible to say DOM2 is “ultra” data efficient without a fair comparison?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Recent works in offline Reinforcement Learning (RL) rely on conservatism. The paper presents Diffusion Offline Multi-agent Model (DOM2), which improves policy design and diversity using a diffusion model. DOM2 utilizes a diffusion model in the policy network and makes use of a trajectory-based data augmentation scheme during offline training. The data augmentation technique trains DOM2 agents on a replay buffer wherein trajectories with higher rewards are duplicated. Ablation studies and experiments demonstrate the empirical effectiveness of proposed design choices.

### Strengths
* The paper is well-written and organized.
* Empirical evaluation provided by the authors is comprehensive.

### Weaknesses
* **Claims on Conservatism:** My main concern is the claims on conservatism used within the DOM2 model. Authors argue that different from prior works they do not rely on conservatism for policy design. The paper also states that learning policies and values with conservatism is inefficient. However, DOM2 model significantly relies on Conservative Q Learning (CQL) to train Q values and hence the policy $\pi$. Furthermore, CQL forms the key part of DOM2 as it is the only ingredient used for policy improvement. This can be validated from ablation studies wherein the conservative policy improvement scheme contributes to most gains in the performance of DOM2. Thus, claims on conservatism severely contradict the paper's central idea.
* **Data Augmentation:** The proposed data augmentation scheme prioritizes highly rewarding trajectories while downweighing lower ones. DOM2 agents, thus, have access to privileged data samples rather than augmented samples as the datapoints itself have not been modified in any way (eg- shifting, scaling, transformed, etc.). In this view, the training process appears to be biased resulting in a near-expert dataset for DOM2 agents and sub-optimal dataset for baseline agents. Note that other baselines do not have access to privileged data samples but only the orignal dataset. This leads DOM2 to outperform prior methods as a result of dataset selection and not algorithmic modifications.
* **Choice of Baselines:** While the empirical evaluation provided in the paper is comprehensive, authors only compare DOM2 to multi-agent baselines. It would be worthwhile to consider other offline RL algorithms in multi-agent settings which have demonstrated cutting edge performance. The paper could compare DOM2 to independent IQL learners [1] or BRAC agents [2] in the multi-agent setting. Similarly, authors could assess the choice of policy improvement scheme using a different offline RL algorithm such as BEAR [3]. This would help validate the claims of conservatism and evaluate the importance of CQL during training.
* **Differences from Prior Work:** I struggle to understand the central contribution of DOM2 within the offline RL literature. Using diffusion models for learning policies is a common practice in offline RL. In addition, the data augmentation scheme corresponds to a top-k sampling strategy wherein trajectories with higher rewards are sampled. It is thus unclear as to what is the novel contribution of DOM2 within multi-agent offline RL literature. It would be worthwhile if authors could highlight the differences between DOM2 and recent algorithms such as Diffuser [4], EDP [5], MADIFF [6], OMAC [7] and OMIGA [8] explicitly. Additionally, authors could discuss the benefits or design choices which are not found in standard multi-agent learning algorithms.

[1]. Kostrikov et. al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022.  
[2]. Wu et. al., "Behavior Regularized Offline Reinforcement Learning", arxiv 2019.  
[3]. Kumar et. al., "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction", NeurIPS 2019.  
[4]. Janner et. al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022.  
[5]. Kang et. al., "Efficient Diffusion Policies for Offline Reinforcement Learning", arxiv 2023.  
[6]. Zhu et. al., "MADIFF: Offline Multi-agent Learning with Diffusion Models", arxiv 2023.  
[7]. Wang et. al., "Offline Multi-Agent Reinforcement Learning with Coupled Value Factorization", AAMAS 2023.  
[8]. Wang et. al., "Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization", arxiv 2023.

### Questions
* Why is learning policies and value functions with conservatism inefficient? Can you please explain the reliance of DOM2 on CQL for conservatism?
* Does trajectory-based augmentation provide high-quality samples only to DOM2? What if other baselines are trained with a similar scheme? Were any samples modified/augmented using shifting, scaling , etc. during training?
* How does DOM2 compare with other offline multi-agent RL baselines such as IQL or BRAC? How effective is the usage of CQL for policy improvement? Can the policy improvement scheme be replaced/compared with another offline RL algorithm such as BEAR?
* How is DOM2 different from Diffuser [4], EDP [5], MADIFF [6], OMAC [7] and OMIGA [8]? Can you please discuss some recent related works comparing DOM2 with offline RL and multi-agent RL literature?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents Diffusion Offline Multi-agent Model (DOM2), an offline MARL algorithm that is based on diffusion policy. DOM2 first augments the dataset by replicating the high-return trajectories. Then, each agent is trained independently by the Diffusion-QL-style learning method while using CQL loss for the critic. In the experiments, DOM2 outperforms the baselines in diverse MARL benchmarks including standard and shifted environments.

### Strengths
1. The paper is well-written and easy to follow.
2. The empirical performance of the proposed DOM2 is strong. It outperforms the baselines in various domains and in shifted environment setting.

### Weaknesses
1. While the paper claims "beyond conservatism", it still relies on conservatism both in value learning and policy learning, both in critic learning (i.e. using CQL loss) and actor learning (i.e. using BC loss).
2. The novelty is limited. It seems the proposed DOM2 is a straightforward extension of Diffusion-QL (Wang et al., 2023) with additional data augmentation. Since the overall training is done in a fully decentralized way, it seems there is no additional/special consideration in the algorithm for the 'multi-agent' setting. 'Why diffusion model for multi-agent RL' is not well-motivated in the paper.
3. Given that each agent is trained independently (decentralized training, rather than centralized training), it may be suboptimal even in a very simple domain (e.g., like OMAR in XOR-game as described in [1]). Can DOM2 solve simple XOR-game-like domains?
4. The proposed data augmentation (section 4.3) does not seem doing actual data augmentation. It is not generating novel data samples, but rather just replicating the existing data samples in the dataset. It just corresponds to changing the 'data sampling distribution' (uniform -> non-uniform depending on the trajectory return).

[1] Matsunaga et al., AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation, NeurIPS 2023

### Questions
Please see the weaknesses section above.
- What is the difference between DOM2 with Diffusion-QL, except for the data augmentation? Also, could you elaborate on the core contribution of DOM2 to solve 'multi-agent' RL?
- Why does DOM2 show better generalization performance than other baselines? Is it due to using diffusion policy, or from other factors?
- I am also curious about the offline single-agent RL performance of DOM2.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed to incorporate diffusion-based policy into multi-agent offline reinforcement learning, which is a straightforward extension of the diffusion-based policy from single-agent setting into the multi-agent counterpart. Most of the techniques are known to the community, but empirical results are quite strong.

### Strengths
* Very strong empirical results.

### Weaknesses
* I don’t find any new insights from this paper. Most of the techniques are from the existing work, and I don’t get any intuitions on why we should do that.

### Questions
* What are the unique hardnesses of the multi-agent setting, compared with the single-agent setting? I feel there are no differences between the algorithm for single-agent setting and multi-agent setting, except that authors replace the state with the observation that can contain other agents’ information.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes DOM2, which applies the Diffusion QL to cooperative multiagent settings following the independent learning paradigm. Extensive experiments on multi-agent particle and multi-agent MuJoCo environments show the superiority of the proposed method.

### Strengths
* The writing of the paper is clear.
* Extensive experiments on multi-agent particle and multi-agent MuJoCo environments are conducted.

### Weaknesses
* **The contribution of the paper is minor.** The proposed method DOM2 seems to be a simple application of Diffusion QL [1] to cooperative MARL. Besides, DOM2 just follows the independent learning paradigm (more like single-agent learning problems). 
* There are very few differences between DOM2 and Diffusion QL [1]. Replacing the DDPM-based diffusion policy with a faster first-order DPM-Solver should not be the main contribution of the paper. 
* **The proposed method DOM2 has little to do with multiagent.** Since the conservatism-based approaches in single-agent RL have limitations, why not directly apply the diffusion-based method to single-agent domains? As the proposed DOM2 is a decentralized training and execution framework (i.e., independent learner), evaluating the method in the single-agent domain is more straightforward.
  * MA-DIFF (Zhu et al., 2023) have done some special designs to apply diffusion models to MARL under the CTDE paradigm, while DOM2  is a straightforward application of Diffusion QL [1].
* The description of the motivating example shown in Figure 1 is not clear. 
* Since MA-SfBC (the extension of the single agent diffusion-based policy SfBC) is compared, MA-Diffusion QL (the extension of the single agent Diffusion QL) should also be compared.


References

* [1] Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning.

### Questions
Please see the weaknesses above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor
