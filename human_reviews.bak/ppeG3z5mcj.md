# ASOR: Anchor State Oriented Regularization for Policy Optimization under Dynamics Shift

- Decision: Reject
- Scores: 3, 3, 5, 8

## Abstract
To train neural policies in environments with diverse dynamics, Imitation from
Observation (IfO) approaches aim at recovering expert state trajectories. Their
success is built upon the assumption that the stationary state distributions induced
by optimal policies remain similar despite dynamics shift. However, such an
assumption does not hold in many real world scenarios, especially when certain
states become inaccessible during environment dynamics change. In this paper,
we propose the concept of anchor states which appear in all optimal trajectories
under dynamics shift, thereby maintaining consistent state accessibility. Instead of
direct imitation, we incorporate anchor state distributions into policy regularization
to mitigate the issue of inaccessible states, leading to the ASOR algorithm. By
formally characterizing the difference of state accessibility under dynamics shift,
we show that the anchor state-based regularization approach provides strong lower-
bound performance guarantees for efficient policy optimization. We perform
extensive experiments across various online and offline RL benchmarks, including
Gridworld, MuJoCo, MetaDrive, D4RL, and a fall-guys like game environment,
featuring multiple sources of dynamics shift. Experimental results indicate ASOR
can be effectively integrated with several state-of-the-art cross-domain policy
transfer algorithms, substantially enhancing their performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The imitation from observation approaches are built on the assumption that stationary state distributions induced by the optimal policies are similar. However, under the dynamic shift of the environments, this assumption is normally violated. The paper addresses this issue by proposing the concept of anchor states and incorporating the anchor state distributions into the policy regularization. Some theoretical results are also provided, trying to provide some lower-bound guarantee. Finally, the proposed method is demonstrated on several common environments.

### Strengths
The proposed method seems technically sound.

The experiments consider most of the commonly used benchmarks, which seem extensible.

### Weaknesses
**1. Motivation.** It is not clear whether the violation of the similar stationary state distributions for imitation from observation approaches is critical for performance. No empirical evidence has been provided, so it is hard to tell whether this paper is well-motivated or not.

**2. Methodology.** As far as I can tell, the main idea for addressing the dynamic shift is to focus on the shared accessible states across all environments, defined as the so-called anchor states. However, I could not figure out the rationale for such an idea. Is the inaccessible state detrimental to policy learning and its generalization across all dynamics? I am very confused about the motivation behind it.

**3. Scope of Dynamic shift.** This paper only considers the Hidden-parameter MDP, which significantly limits the scope of dynamic shifts to be considered. My feeling is that although different parameters can be initialized by drawing samples from a certain distribution, the classes of MDP typically share similar structures or have a large overlap with each other. I could not tell the reward function needs to remain the same, until I find the authors state this point in the end. I think this limitation should be clearly stated at the beginning. That being said, the proposed method can only consider the a potentially small class of transition dynamic shift (by Hidden parameter MDP), while failing to incorporate the shift in the reward function. 


**4. The theoretical results are limited and questionable.** Firstly, what is the relationship between Theorems 2 and 4? It is less convincing to consider the neural distance, which is typically considered in generative models. After I read the proof in the appendix, I found the proof highly relies on the assumption from Xu et al. 2020, assuming the reward function has a linear space structure. This assumption has not been clearly emphasized in Theorem 4.4 since the proof in the appendix highly depends on the proof from Xu et al. 2020. Thus, I do not think the theoretical results are sufficient to provide insightful results.

Another point that I am concerned about is the linear horizon dependency, which suggests the improved lower bound by authors. There is no direct evidence to demonstrate, or at least empirically, that the derived bounds are sharp or not vacuous to align with practice. The common RL theory normally promotes some regret analysis, sample complexity analysis, or small-error bound. However, it is not clear the linear horizon dependency (compared with quadratic one) can be widely accepted. The authors could provide some papers that also show a similar proof technique. I am not sure about this point, so feel free to correct me if I am wrong. 


**5. Clarity and Writing.** After reading this paper, I am afraid I can not understand the logic behind it. What critical issues is this paper really focused on addressing? Is this issue really important in the literature? The clarity needs to be significantly improved. Some statements are confusing, such as KL divergence, which has a poor generalization (in Line 318). The author should be very careful about this statement. 

**Minors**. How do you select the best \pho in Line 270? The experiments look fine to me, although I do not think there is any notable conclusion I can find. The paper has a large overlap with [1], which affects the novelty and technical contribution of this paper. Some detailed comparisons are also needed in the future.


[1] Zhenghai Xue, Qingpeng Cai, Shuchang Liu, Dong Zheng, Peng Jiang, Kun Gai, and Bo An. State regularized policy optimization on data with dynamics shift. In NeurIPS, 2023a.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates a problem in Imitation from Observation (IfO) that occurs when certain states become inaccessible due to changes in the dynamics between the expert and the agent. The authors introduce a reward augmentation algorithm called ASOR (Anchor State Oriented Regularization) to tackle this issue. ASOR works by excluding reference states that change in accessibility during training and by defining anchor states that remain accessible across all dynamics.

### Strengths
The paper points out a challenge in cross-domain imitation learning from observation, where some actions in the expert demonstrations are not realizable by the agent.

### Weaknesses
1.	The motivation of the paper is not entirely convincing. Even in in-domain imitation from observation—where the experts and agent operate in the same environment—optimal policy trajectories are often diverse due to randomness or differing initial conditions. As a result, the state sets of these trajectories are unlikely to fully overlap; therefore, most of these states won't fit within the anchor states introduced in the paper, which are indeed realizable by the agent.
2.	The empirical results show limited improvement over baseline methods, suggesting that the proposed ASOR algorithm may not offer a significant performance advantage.

### Questions
The core assumption behind ASOR seems to be that anchor states exist and can be consistently identified across environments. However, in many real-world settings, this assumption seems unrealistic (see weakness 1). Is there a practical strategy for dynamically identifying anchor states in such complex environments? The current approach assumes these states remain fixed across all trajectories, which could be overly simplistic.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper recognizes a limitation in existing RL under the dynamics shift method. They assume stationary state distributions induced by optimal policies remain similar despite the dynamics shift. This limitation does exist in many off-dynamics RL/RL under dynamics mismatch. Thus, they propose a regularized policy optimization algorithm that regularizes the 'anchor state' instead of the 'all-state' stationary distribution. The anchor state is defined as the accessible state across all the optimal trajectories in all the MDP.  Further, they provide a lower bound for the performance.

### Strengths
1, The method is well-motivated by the limitation of existing RL under dynamics shift. And regularizing the 'anchor state' distribution is novel.  

2, The experiments is comprehensive, demonstrating the effectiveness of the method.

### Weaknesses
1, the problem is not formally defined. Can you be more specific about policy optimization under dynamics shift? Some paper, such as off-dynamics RL, focus on training a policy in the source (simulation) environment(s) and deploying to the target (real world) environment. This can also be called as policy optimization under dynamics shift. It looks like the focus is on training a policy on multiple MDPs with different transitions and performing simultaneously well in all the environments.

2, section 3.3 and 3.4 is hard to follow. How to connect eq(3) with the eq(2) and how do you get eq(3) with Random Network Distillation (RND). What is the loss function?  Also, more details of ESCP should be included, and the notation of the ESCP should be defined formally instead of just mentioning it in the algorithm 1. And the notation is a bit complicated.

### Questions
1, What if the anchor state doesn't exist? Is there any assumption you need? It can be the case that the optimal trajectories in different MDP won't intersect or the intersection state is limited. 

2, See weakness 1(b) about section 3.2. 

3, Why using an arbitrary $T_0$ in the loss, how do you choose $T_0$ how it affect the results? Does the anchor state in $T_0$ cover all the anchor states across different MDPs? If yes, can you justify it? If not, the the Eq(2) seems different from the motivation. 

4, how do you get the anchor state during training?

I would be happy to raise my score if the questions are answered.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces ASOR (Anchor State Oriented Regularization), a novel approach for policy optimization in reinforcement learning under dynamics shift. In real-world scenarios, environmental dynamics can change significantly, making certain states inaccessible and rendering traditional Imitation from Observation (IfO) techniques, which assume identical optimal state distributions, ineffective. ASOR addresses this by focusing on "anchor states"—states accessible across all dynamics—which it incorporates into policy regularization to improve robustness and adaptability. The paper also demostrates the versatility of the method as an add-on for state-of-the-art policy optimization methods across different settings. ASOR is validated across various online and offline RL benchmarks, demonstrating its efficacy in scenarios with dynamic changes.

### Strengths
**Strengths**

- Conceptual Intuition and Motivation: The paper introduces a method with strong intuitive appeal by addressing the limitations of state-only policy transfer algorithms, which often assume identical state distributions across environments. The authors effectively motivate the need for a refined approach, introducing anchor states as a way to target consistently accessible states. This innovation weakens the assumption of identical state distributions and proposes a structured method to identify and match distributions on anchor states, which is a valuable contribution.
- Theoretical Analysis: I didn't follow all the theoretical analysis. But the paper claims to be obtaining stronger performance lower bounds while making weaker assumptions compared to the baselines. 
- Compatibility with Existing Algorithms: ASOR is modular and integrates smoothly into existing algorithms by modifying the reward function. The paper demonstrates ASOR’s effectiveness as an enhancement to state-of-the-art RL algorithms like PPO, MAPLE, and ESCP across different RL settings, which increases its potential applicability.
- Empirical Validation Across Diverse Environments: ASOR is validated across a range of environments, including Minigrid, D4RL, MetaDrive, and a high-dimensional Fall Guys-like game, covering both online and offline RL settings. The method consistently improves performance, showcasing ASOR’s versatility and impact across environments with dynamics shifts.

### Weaknesses
**Weaknesses**
- Writing and Background Accessibility: At points, the writing was challenging to follow for me. This may be partly due to my fading memory of some of the background needed for the work. Particularly in Section 3.3, which references previous work with minimal explanation.  Including foundational information or more detailed background in the appendix would make the paper more accessible, especially to readers less familiar with some of these topics.
- Anchor State Identification: Anchor state identification is central to the  approach, yet it is only briefly discussed in Section 3.3. A deeper examination of the considerations, challenges, and potential tradeoffs in anchor state identification would strengthen the method’s presentation and provide clarity on this crucial component.
- Limited Ablations on Policy Regularization Choices: While it makes sense to regularize policies to maintain consistent distributions on anchor states, the decision to explicitly incentivize divergence on non-anchor states is less straightforward. This is particularly surprising given that the anchor states are identified using learned policy distributions and state values. More ablations or additional exposition on the rationale behind this choice would provide valuable insights and a clearer understanding of the intended impact of this aspect of the method.

### Questions
same as weaknesses

### Soundness
4

### Presentation
2

### Contribution
4
