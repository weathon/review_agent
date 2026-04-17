# Action-Free Offline-To-Online RL via Discretised State Policies

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6, 4

## Abstract
Most existing offline RL methods presume the availability of action labels within the dataset, but in many practical scenarios, actions may be missing due to privacy, storage, or sensor limitations. We formalise the setting of action-free offline-to-online RL, where agents must learn from datasets consisting solely of $(s,r,s')$ tuples and later leverage this knowledge during online interaction. To address this challenge, we propose learning state policies that recommend desirable next-state transitions rather than actions. Our contributions are twofold. First, we introduce a simple yet novel state discretisation transformation and propose Offline State-Only DecQN (OSO-DecQN), a value-based algorithm designed to pre-train state policies from action-free data. OSO-DecQN integrates the transformation to scale efficiently to high-dimensional problems while avoiding instability and overfitting associated with continuous state prediction. Second, we propose a novel mechanism for guided online learning that leverages these pre-trained state policies to accelerate the learning of online agents. Together, these components establish a scalable and practical framework for leveraging action-free datasets to accelerate online RL. Empirical results across diverse benchmarks demonstrate that our approach improves convergence speed and asymptotic performance, while analyses reveal that discretisation and regularisation are critical to its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies a novel and practically motivated setting in reinforcement learning (RL): action-free offline-to-online RL, where the agent must learn from datasets containing only tuples of the form state-reward-next state datasets, without action labels. Such a setting arises naturally in domains like healthcare, finance, and robotics, where action logs may be unavailable due to privacy, storage, or sensor constraints. The paper asks: Can an agent learn useful knowledge from such datasets and transfer it to online learning?
To address this, the authors propose a two-stage framework built around a new offline algorithm, Offline State-Only DecQN (OSO-DecQN), that learns state policies rather than action policies. Instead of predicting actions, the algorithm predicts discretised state differences (i.e., the direction of state change) and uses these predictions to guide online RL. 
Empirically, the authors show that OSO-DecQN pre-trained on action-free datasets can accelerate online learning and improve final performance across a range of continuous- and discrete-control tasks (D4RL, DeepMind Control Suite), outperforming existing action-free baselines (e.g., AF-Guide by Zhu et al., 2023). Ablation studies further highlight the importance of discretisation and regularisation, and theoretical results (Theorem 1–2) provide a discretisation-dependent value approximation bound.

### Strengths
The paper clearly identifies a real-world gap between existing offline RL methods (which assume full action observability) and practical domains where actions are missing. The authors articulate why action-free learning is challenging and outline an elegant conceptual framework that combines discrete state prediction with conservative regularisation and online guidance. This direction feels both original and valuable for RL research, especially as large unlabelled state datasets become more common.
The proposed OSO-DecQN is conceptually simple yet technically grounded. Overall, the algorithm is a clean extension of DecQN into an action-free paradigm.
The authors provide a series of formal results bounding the approximation error introduced by state discretisation. In particular, Theorem 1 and Theorem 2 derive the difference between the original and discretised optimal value functions, showing that the discretisation granularity controls the value loss. The proofs (Appendix B) are rigorous and grounded in classical contraction and KL-divergence arguments, enhancing the credibility of the approach.

### Weaknesses
Although the integration of discretisation, conservative regularisation, and decoupled Q-learning is novel in the action-free context, each component individually builds on well-known ideas (DecQN, CQL-style penalties, inverse dynamics models).
The contribution is therefore more conceptual unification than a fundamentally new RL mechanism.
While the paper claims the IDM is lightweight, its reliability is central to the online guidance mechanism. The IDM must map predicted state differences back to executable actions accurately; errors in this translation could affect results. Although the authors show robustness across architectures (Table 10–11), a more direct comparison of IDM accuracy versus final online performance would better quantify this dependency.
The theoretical analysis focuses entirely on the offline discretisation error.
There is no corresponding guarantee or convergence analysis for the guided online learning phase (e.g., stability of β-switching or IDM-based guidance).
Some formal justification, perhaps connecting it to off-policy improvement or safe exploration, would make the work more complete.

### Questions
1.	Since the annealed beta schedule significantly affects results (Fig. 3), analysing it in terms of exploration-exploitation balance could clarify its effect.
2.	It would be better to show a correlation between IDM prediction error and final online performance to support the claim that IDM design is not a major factor.
3.	It would be better if the paper includes comparison of total pretraining + online runtime versus baseline methods to substantiate claims of scalability.

### Soundness
2

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
This paper addresses learning from datasets that lack action labels, containing only (state, reward, next-state) tuples. It proposes OSO-DecQN, which learns policies over discretized states offline and uses them to guide online learning through an inverse dynamics model (IDM).

### Strengths
1. The paper tackles practical scenarios where data has no action labels by proposing an offline policy training method that treats a discretized state change as an action and applies offline Q-learning, and demonstrates its effectiveness through various D4RL tasks.

1. The work provides a solution to further improve models pretrained with offline data via online learning by incorporating an IDM (inverse dynamics model), which maps a state change to an action.

### Weaknesses
1. Limited comparison with other action-free methods. It seems that in Figure 2, it only compares with the case without pre-training on offline datasets (TD3) and the action-free method (Zhu et al., 2023 [1]). However, given there are more action-free offline training methods [2,3,4], it would be better to compare with all these baselines to fully validate the effectiveness of the proposed action-free training based on state discretization.

2. The presentation of Figure 2 is confusing because although it is for comparing with another action-free offline method (Af-guide), it doesn't directly compare the proposed one and Af-guide. Instead, it is comparing with another baseline (TD3) and indirectly claims that the proposed one is better than AF-guide because it is better than TD3, which I found confusing. It would be better to put all algorithms to be compared in the figure. Also, I wonder if the training/evaluation setup for your method, Af-guide, and TD3 is the same for all experiments.

3. The state discretization seems coarse, which could lose a lot of meaningful state transition information depending on state representation and tasks.

[1] Deyao Zhu, Yuhui Wang, J ̈urgen Schmidhuber, and Mohamed Elhoseiny. Guiding online reinforcement learning with action-free offline pretraining. arXiv preprint arXiv:2301.12876, 2023.

[2] Bohan Zhou, Ke Li, Jiechuan Jiang, and Zongqing Lu. Learning from visual observation via offline pretrained state-to-go transformer. Advances in Neural Information Processing Systems, 36: 59585–59605, 2023.

[3] Hao Luo, Bohan Zhou, and Zongqing Lu. Pre-trained visual dynamics representations for efficient policy learning. In European Conference on Computer Vision, pp. 249–267. Springer, 2024.

[4] Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with actionfree pre-training from videos. In International Conference on Machine Learning, pp. 19561–19579. PMLR, 2022.

### Questions
1. Zhu et al. 2023 [1] (AF-guide) seems to adopt different approaches for both offline and online training. In Figure 2, is the online and offline training of Af-guide entirely following Zhu et al. 2023? If so, can you provide results for the case of Af-guide (offline) + your method (online) and your method (offline) + Af-guide (online)? This would give a more fine-grained performance comparison of each training process (online/offline).

[1] Deyao Zhu, Yuhui Wang, J ̈urgen Schmidhuber, and Mohamed Elhoseiny. Guiding online reinforcement learning with action-free offline pretraining. arXiv preprint arXiv:2301.12876, 2023.

### Soundness
3

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
This paper proposes a new offline-to-online reinforcement learning (RL) framework designed for action-free offline datasets. The authors introduce a discretized state representation that serves as an action surrogate, enabling the training of a discretized Q-function purely from state transitions. During online learning, an inverse dynamics model (IDM) is trained to map the predicted next-state differences into executable actions. A policy-switching strategy blends the IDM-based actions with the online policy, while a regularization term is used to stabilize and constrain the offline Q-function.
Experiments on D4RL and the Action-Factorized DeepMind Control Suite demonstrate faster convergence and better asymptotic performance for both TD3 and DecQN baselines. Ablation studies further show that both discretization and regularization are crucial for performance.

### Strengths
1. The proposed method is conceptually clear and technically sound, addressing a problem where offline datasets lack action labels.
2. The methodology is well-designed, combining existing offline RL techniques with novel discretization and regularization components in a coherent way.
3. The ablation studies are comprehensive and clearly demonstrate the contribution of each component.

### Weaknesses
1. The motivation for the action-free offline data setting is not clearly justified. The paper should better explain when and why such data would realistically occur.
2. The examples in the introduction are not entirely convincing. The most plausible application would be robotics from video, but the experiments only consider relatively small state spaces (up to 78 dimensions), which limits the realism of the claim.
3. The baseline coverage is limited — the paper primarily compares against a single online RL algorithm (TD3) and lacks comparison with stronger modern offline-to-online or action-free RL baselines.

### Questions
1. Are there plans to extend this framework to visual or multimodal inputs, where the state-space dimension exceeds 1000?
2. Would the method still be effective if the offline dataset and the online environment come from slightly different dynamics distributions?
3. How many offline data samples are used in training, and how does the amount of offline data influence the overall training stability and performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors formalise the setting of action-free offline-to-online RL, where agents must learn from datasets consisting solely of (state, reward, next state) tuples and later leverage this knowledge during online interaction. To address this challenge, they propose learning state policies that recommend desirable next-state transitions rather than actions. First, the authors introduce a simple yet novel state discretisation transformation and propose Offline State-Only DecQN (OSO-DecQN), a value-based algorithm designed to pre-train state policies from action-free data. OSO-DecQN integrates the transformation to scale efficiently to high-dimensional problems while avoiding instability and overfitting associated with continuous state prediction. Second, they propose a novel mechanism for guided online learning that leverages these pre-trained state policies to accelerate the learning of online agents. Together, these components establish a scalable and practical framework for leveraging action-free datasets to accelerate online RL. Empirical results across diverse benchmarks demonstrate that their approach improves convergence speed and asymptotic performance, while analyses reveal that discretisation and regularisation are critical to its effectiveness.

### Strengths
1. The motivation for action-free RL is strong, while the explanation for the motivation is clear in the paper.
2. The authors propose an algorithm with novel technical designs.
3. The experimental results are generally good. The ablation study part is helpful.

### Weaknesses
1. The paper directly decouples the Q function. Under the case where the real Q function depends on the correlation between different dimensions, could this approximation perform well? Meanwhile, for the argmax of the action, I am wondering what is the choice when $\mathcal{A}$ is not a product space. For instance, if $\mathcal{A}$ is a unit ball, how to define the argmax on each dimension?

2. For the training loss of IDM, could you please provide some motivation for that? Does the success of such approach implicitly depend on the Lipschitzness of the Q value function over actions?

### Questions
Please see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an algorithm for learning value functions from action-free datasets in an offline manner and using these value functions to improve the learning speed in a subsequent online phase. The key parts are discretization method for the state space to assist learning in the offline phase. Experiments with standard offline RL datasets (removing action information) demonstrate the utility of the method.

### Strengths
The problem setting is interesting, action-free datasets would seem to be more common in practice and understudied.
The discretization technique for the state space and converting to actions using an IDM is simple and surprisingly effective.

### Weaknesses
Generally, there are no glaring issues but some experimental and design choices are unclear to me.
I have included these questions below in the "Questions" section.
Many of these are clarification questions or ablation suggestions and I am happy to increase my score after having more information.

### Questions
- Why is DecDQN specifically chosen tobe used rather than a simpler DQN variant given that the method seems to be agnostic to the value-based learning algorithm?


- For the IDM's loss, the $L_1$ loss is used. Why not use the Huber loss if robustness is an issue? 

- How does the performance of the method compare to offline-to-online methods that do have access to action information in the offline dataset? We would expect worse performance of course but it would be interesting to see how large the gap really is, it may turn out to be quite small.


- Concerning the discretization step for the state space, it has been found for value learning that regression and predicting a continuous output is inferior to using classification-style losses (the histogram loss) [1]. Have you tried using the histogram loss instead? The loss converts continuous regression targets into a classification target. Then predictions can be done by outputting probabilities over discrete outputs in the support and taking an average. It can be used with a variable number of support points.
This method could also potentially be used for the inverse dynamics model.


- Line 277: Why do we need to use generated actions from the policy to relabel transitions in the offline dataset with the current policy to train the IDM? This would seem to induce a feedback loop since the IDM is trained on its own actions. Could an ablation be run on including this part or not?


- Is the $\epsilon$ hyperparameter in the discretization important? In the appendix, it indicates it is set to be 1e-4 which seems quite small but an ablation showed that this was important to include for some environments.

- How important is the policy-switching strategy? How sensitive is it to the $\beta$ parameter? 

- Does the $Q(s, \Delta s)$ value network also get updated during the online phase?


- It is fairly surprising that an inverse dynamics model would be able to use the $\Delta s$ discretized state change effectively since there would be many actions that could map to the same discretized state difference. Why do you think this is effective? Is this exploiting some structure specific to simulated robotics tasks? 
An experiment that would help clarify this is to test whether following the action given by the IDM actually leads to the expected dicretized state change. Verifying that the IDM can do this would be interesting.


Minor points (not impacting the score):
- The notation for the discretized state is introduced as $\delta(s,s')$ but in other places it is $\Delta s$ that is used. 

- In Theorem 1, M is used to denote the number of bins but it is also used to denote the MDP. It would also be clearer to include the definition of $M$ and $H$ in the theorem statement. 

- Table 1: The font is a bit small. 

[1] Stop Regression: Training Value Functions via Classification for Scalable Deep RL

### Soundness
2

### Presentation
2

### Contribution
3
