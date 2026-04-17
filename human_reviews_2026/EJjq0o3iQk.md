# FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning

- Decision: Reject
- Scores: 2, 8, 4, 2

## Abstract
Flow Matching (FM) has shown remarkable ability in modeling complex distributions and achieves strong performance in offline imitation learning for cloning expert behaviors. However, despite its behavioral cloning expressiveness, FM-based policies are inherently limited by their lack of environmental interaction and exploration. This leads to poor generalization in unseen scenarios beyond the expert demonstrations, underscoring the necessity of online interaction with environment. Unfortunately, optimizing FM policies via online interaction is challenging and inefficient due to instability in gradient computation and high inference costs. To address these issues, we propose to let a "student'' policy with simple MLP structure to explore the environment and be online updated via RL algorithm with a reward model. This reward model is associated with a "teacher'' FM model, containing rich information of expert data distribution. Furthermore, the same "teacher'' FM model is utilized to regularize the "student'' policy's behavior to stabilize policy learning. Due to the student’s simple architecture, we avoid the gradient instability of FM policies and enable efficient online exploration, while still leveraging the expressiveness of the teacher FM model. Extensive experiments shows that our approach significantly enhances learning efficiency, generalization, and robustness, especially when learning from suboptimal expert data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to incorporate the benefits of online exploration into flow matching training by building on an inverse reinforcement learning framework. First, a teacher flow-matching model is trained on a static offline dataset. To enhance the expressiveness of the reward model, the teacher model’s flow-matching loss on agent-generated trajectories is used to measure the discrepancy between the agent distribution and the expert distribution. To avoid the instability of updating flow-matching policies through backpropagation through time or policy gradients, the method adopts a distillation objective that jointly integrates the rewards and teacher behavior into a simpler student policy. Experimental results show that, with additional online interactions, the proposed approach can mitigate the potential sub-optimality of offline datasets and outperform standard behavior cloning models trained solely on static data.

### Strengths
1. The paper is clearly written and well structured.

2. Extending the capabilities of flow-matching or diffusion policies on out-of-distribution area using online interatctions is an important research direction, since distributional shift is hard to be addressed by simply scaling static offline data. It is necessary to using additional online interatctions of model rollouts to mitigate some corner cases.

3. Improving the RL training stability for flow-matching or diffusion policies is important. Most current methods like directly maximizing differentiable rewards or policy gradients can be unstable.

### Weaknesses
1. `Lack of Novelty`. 

This paper appears to be a straightforward combination of three existing ideas: using diffusion losses as rewards [1], applying distillation methods to optimize flow-matching policies via reinforcement learning [2], and inverse reinforcement learning [3]. Although the authors claim novelty in being the first to use flow-matching loss as rewards, flow matching and diffusion models are essentially two sides of the same coin. Therefore, this contribution does not strike me as genuinely novel.

Furthermore, the idea of using reinforcement learning to optimize a distilled, simpler policy is already well studied [2][4][5]. As a result, using distillation to enhance the stability of flow-matching RL training does not appear novel either. Taken together, these factors make the paper resemble a naïve “A + B + C” combination without a clearly original contribution.

2. `Weak Motivation`. 

If the authors had provided strong motivation for why the “A, B, C” components should be integrated, I could have acknowledged the contribution despite the limited novelty. Unfortunately, the current manuscript fails to do so.

The authors identify the potential suboptimality of offline datasets in traditional flow-matching training as the core challenge, and aims to introduce additional online interactions to address this. However, the final objectives still primarily fit the static offline dataset. For example, rewards are higher when the agent’s behavior resembles that of the offline dataset, which only encourages in-distribution behavior. In addition, to improve training stability and mitigate inaccurate reward estimation on out-of-distribution areas, the authors introduce a distillation loss that encourages the student policy to mimic the teacher flow. This, again, is essentially behavior cloning.

In my view, the proposed method reformulates the “mimic offline dataset” objective into “A + B + C jointly mimic the offline dataset.” This reformulation does not fundamentally address the challenge that the authors themselves emphasize. For this reason, I would recommend rejection.

3. `Marginal Improvements`. 

The experimental results show only marginal improvements over baseline models, which further limits the paper’s contribution.

[1] Diffusion-reward adversarial imitation learning. 2024

[2] Flow q-learning. 2025

[3] Generative adversarial imitation learning. 2016

[4] Score Regularized Policy Optimization through Diffusion Behavior. 2024

[5] Diffusion Policies creating a Trust Region for Offline Reinforcement Learning. 2024

### Questions
Please see weaknesses for details.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an offline imitation learning framework in which a student policy learns from a reward model based on Flow Matching (FM). The authors begin by noting that the absence of an online FM policy learning mechanism limits the policy's generalization capability. The point of this paper is not to train a FM policy. Inspired by adversarial inverse reinforcement learning, this work leverage FM to develop an enhanced discriminator. A student policy is implemented as a simple MLP. The FM-based discriminator is trained to fit expert data while distinguishing it from the behavior generated by the student policy.

The authors also observe that while several prior attempts have integrated online RL with diffusion models, these methods often suffer from training instability. I would recommend that the authors also discuss the relevant work DACER [1] in this context, but I generally agree with this statement based on my own experience. Furthermore, the paper elaborates on the inherent challenges of training FM policies online. Its key contribution lies in leveraging a powerful generative model to "infuse" knowledge into a simple policy, while the simple policy also learns online to prevent overfitting.

Overall, this paper is interesting to me. However, I still have some concerns about the comprehensiveness of the technical details, which lower my confidence.

[1] Wang, Yinuo, et al. "Diffusion actor-critic with entropy regulator." *Advances in Neural Information Processing Systems* 37 (2024): 54183-54204.

### Strengths
This paper presents a well-motivated and novel approach, supported by a clear and logical structure. The proposed method demonstrates significant performance improvements over baselines. The authors also provide comprehensive discussion in the appendix, including answers to some possible questions, which greatly aids in understanding the methodological rationale.

### Weaknesses
I am not clear about whether the framework is easy to implement effectively or it requires tricks and careful hyperparameter tuning. Also, I am not sure about how the authors made the comparison fair (e.g., using common hyperparameters or network architectures, or fine-tuning each algorithm one-by-one). I noticed that the authors claim that the code will be made open-source, but I would appreciate some explicit discussion about such details.

### Questions
1. Is there a learned value function or advantage estimation for the student MLP policy? Regarding the student policy loss (Equation (11)), the first term is the expected return. Does this mean that the student policy learning is identical to REINFORCE with adversarial rewards when $\beta = 0$?
2. How is the MLP student policy implemented? For example, PPO usually outputs a mean vector and uses a fixed scale to represent a Gaussian distribution, while SAC typically outputs both a mean vector and a per-dimension scale vector. There is also a policy class called amortized actors [2,3,4] which, though structurally an MLP, can express a multi-modal decision distribution.
3. The teacher FM can represent a multi-modal data distribution, but the student policy probably cannot (if it is Gaussian). In cases where the expert data is highly multi-modal (for example, the scenario discussed in Figure 1 of the DQL paper (Wang et al., 2022)), would the "infusing" encounters challenges?
4. Are there any tricks or hyperparameters not covered in the appendix? For example, only disc_lr is listed in Table 2. Is this a global learning rate that also applies to the student policy? What is the detailed network architecture of FM teacher?
5. How did you made the comparison with baselines fair? 
6. What is $p_\theta$ and $T$ in equation (1)? They don't seem to have been explained.

[2] Haarnoja, Tuomas, et al. "Reinforcement learning with deep energy-based policies." *International conference on machine learning*. PMLR, 2017.

[3] Messaoud, Safa, et al. "S$^ 2$AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic." *12th International Conference on Learning Representations, ICLR 2024*. 2024.

[4] Wang, Ziqi et al. "Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization." *Advances in Neural Information Processing Systems*. 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Flow Matching (FM) for RL is emerging as a strategy of Imitation Learning, which, however, defaults to offline learning, which lacks an exploration mechanism and is upper-bounded by expert demonstration performance. This paper proposed a method to use FM for reward shaping and regularization in online RL, which involves a teacher-student style learning. The teacher FM model shapes a reward for the agent model learning and action regularization during the online RL phase.

Empirical experiments on SOTA locomotion and navigation tasks showed that their method achieves higher generalization of learned policy and robustness to sub-optimal expert demonstrations in certain tasks.

### Strengths
+ This paper aims to tackle two fundamental challenges in online AIL: 1) where expert demonstrations are noisy or suboptimal ,and 2) traditional FM cannot adapt to an online setting
+ Preliminary work is clearly introduced to facilitate the introduction of the proposed method
+ The appendix offers an interesting theoretical discussion on why FM may offer advantages over conventional IRL reward models

### Weaknesses
Vague contribution scope:

* It looks to me that the main algorithmic novelty of this work is the integration of an FM model to replace the traditional IRL reward shaper, which feels incremental. Since the action regularization and the reward shaping methods can somewhat be traced back to prior work.   Especially, the discriminator training objective is identical to GAIL. 
* The motivation for introducing FM into online IRL is relegated to the appendix rather than the main text. Although the argument there is compelling, the empirical evidence presented in the main paper does not strongly support these theoretical claims.

* Limited empirical improvement:
  - Reported performance improvements are marginal on several benchmarks (e.g., Hopper, Maze, Ant-goal) 
  - For noisy initial and goal state settings, all experiments are evaluated only on a single task, Hand-rotate.
  - The experimental result in Table 1 does not support the claim that FM-IRL overcomes the limitation of suboptimal expert data, as they are approximately or often below the expected return of demonstration data.

### Questions
- Can a traditional  IRL discriminator be viewed as a special case of a  Flow Matching model, perhaps implicitly defining a probability flow between expert and policy distributions?
- What would be the algorithmic robustness in noisy settings for other locomotion/navigation tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an adversarial imitation learning (AIL) method, termed FM-IRL, using flow-matching (FM) models. The main idea is to combine the benefits of adversarial imitation learning (which leads to better generalization and policy performance than behavior cloning) and FM-based policies (which are more expressive than MLP policies with a Gaussian action distribution). Since directly optimizing FM-based policies using AIL in an online manner is difficult, they first train an FM-based policy to clone the expert, and then train an MLP policy using a combination of (i) AIL reward derived from the FM-based policy, and (ii) regularization to stay close to the FM-based policy.

### Strengths
- The paper addresses an important research gap (i.e., how can we leverage flow-based policies for adversarial imitation learning)
- FM-IRL design choices (using a smart parameterization for the FM discriminator model, and a regularization term to stay close to the teacher FM policy) are clearly presented
- They show competitive performance against various AIL baselines, and report much better generalization to noise in the initial and goal states of the tasks

### Weaknesses
- Loss of multimodality in the trained policy: The main motivation of the paper is to combine the expressiveness of FM-based policies (i.e., their ability to represent multimodal policies) with AIL. However, since the student policy trained by FM-IRL is actually a unimodal MLP policy, it is unclear if the full potential of multi-modal policies is leveraged. 
- Regularization in Eq. 11 may be mode-averaging: As a related point, I suspect the regularization term in equation 11 could lead to poor performance since it would encourage the unimodal MLP policy to spread its probability mass over multiple modes of the FM policy. It would be helpful to add an ablation study to test the benefit of the regularization.
- Unfair comparison to FM-based baselines: The comparison in Section 4.3 is great to have, but it might be unfair due to different rewards being used. I believe you are comparing baseline methods that use the sparse reward of the environment with FM-IRL, which uses a discriminator-based reward obtained using expert trajectories. Could you run the FM-A2C, FFM-PPO, and FPO baselines with the discriminator-based reward?
- Terminology (IRL vs. AIL): Even though the AIL problem is the same as IRL with a convex regularization, I would expect an IRL paper to have more empirical evaluation of the quality of the recovered reward, e.g., train a policy on the recovered reward and examine this policy's performance compared to the expert. In the absence of these experiments, I would recommend updating the name of the method to FM-AIL.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3
