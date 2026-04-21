# TOP-ERL: Transformer-based Off-Policy Episodic Reinforcement Learning

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
This work introduces Transformer-based Off-Policy Episodic Reinforcement Learning (TOP-ERL), a novel algorithm that enables off-policy updates in the ERL framework. In ERL, policies predict entire action trajectories over multiple time steps instead of single actions at every time step. These trajectories are typically parameterized by trajectory generators such as  Movement Primitives (MP), allowing for smooth and efficient exploration over long horizons while capturing high-level temporal correlations. However, ERL methods are often constrained to on-policy frameworks due to the difficulty of evaluating state-action values for entire action sequences, limiting their sample efficiency and preventing the use of more efficient off-policy architectures. TOP-ERL addresses this shortcoming by segmenting long action sequences and estimating the state-action values for each segment using a transformer-based critic architecture alongside an n-step return estimation. These contributions result in efficient and stable training that is reflected in the empirical results conducted on sophisticated robot learning environments. TOP-ERL significantly outperforms state-of-the-art RL methods. Thorough ablation studies additionally show the impact of key design choices on the model performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presented a transformer-based episodic reinforcement learning method for enabling online updates in an off-policy RL framework. Combined by a trajectory segmentation technique, the transformer architecture is employed to construct a critic that evaluates the values of segments of state-action trajectories. As a result, the proposed method achieves an off-policy update, improving update efficiency compared to previous episodic RL methods using on-policy training. Empirical evaluations on control tasks show that the proposed method improves performance.

### Strengths
- The paper is well motivated.
- The writing is quite good. 
- The main experiments are conducted on the large-scale RL benchmark.
- Analysis of many design choices are presented.

### Weaknesses
- Some used techniques, such as trust region constraints, layer normalization, greatly affect the performance of the proposed algorithm (see Fig. 5). The ablations of TOP-ERL, which do not use these techniques, actually achieved worse performance than previous methods.  Someone may conclude that the performance improvement of TOP-ERL over baselines is achieved by using all these techniques, rather than the usage of the transformer architecture or others. 

- Layer normalization and trust region constraints is important for TOP-ERL achieving good performance. But the paper lacks a detailed discussion about the importance of these two components.

### Questions
-  Designing additional experiments to verify the individual contribution of the transformer-based critic network, such as TOP-ERL without the transformer, is suggested. However, I have no idea about the specific details about the exp.  
- Explaining why layer normalization and trust region constraints are critical in Section 5.3 are suggested.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Transformer-based Off-Policy Episodic Reinforcement Learning (TOP-ERL), an interesting combination of episodic RL with off-policy updates and the transformer architecture. 
The main contribution is the way to use the transformer architecture in this setting to learn an off-policy critic.
The authors leverage N-step returns, Trust Region Projection Layer, Probabilistic Movement Primitives and Target Networks. 
Empirical results on robot learning environments (Meta-World, etc.) show that TOP-ERL outperforms state-of-the-art RL methods, both in terms of policy quality and sample efficiency. The paper also includes thorough ablation studies to analyze the impact of key design choices.

### Strengths
I find the paper very well written, also well structured, motivated and explained.

The idea of using a transformer such that s_0 maps to V(s), a_0 maps to Q(s, a_0), a_1 maps to Q(s, a_1), ... is very natural and creative.
To the best of my knowledge, it is the first approach that proposes to do it this way.

I also find it interesting that V and Q can be modeled by the same unique transformer architecture without having dedicated weight just for V or just for Q.

The contributions of this paper are good for the field of RL with transformers, although I believe ERL is slightly less used than standard RL.

### Weaknesses
The authors claim that off-policy algorithms are often more sample efficient than on-policy counterparts. However, PPO is still often the way to go when you face a new RL problems because of better stability and the need for less hyperparameter tuning. I am not fully convinced this approach won't have the same issue.

I believe the explanations of why importance sampling is not necessary could be improved.

### Questions
1. It is not completely clear to me if the transformers can attend only the current observation s^k_0 or also the previous ones during training?

2. I am wondering how important are the probabilistic MP in the performance, what if the policy was also a transformer?

3. I am also wondering if the positional encoding of two different states s^k_0 and s'^k_0 would always be the same (even if one is more in the future that the other one)?

4. Is a target network necessary at all?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new off-policy episodic reinforcement learning framework that, especially, improves sampling efficiency over traditional episodic-based and step-based baselines. A main novelty of the work is using a Transformer-based architecture to train a critic network that predicts value and Q-functions for a sequence of actions from an initial state. This critic network is then adapted and applied in an SAC-like algorithm for off-policy episodic reinforcement learning training.

### Strengths
1.	A critic network for action sequences using a traditional Transformer architecture is proposed to model N-step returns.
2.	A SAC-like algorithm is adapted for off-policy episodic reinforcement learning using the trained critic network.
3.	The proposed methodology is shown to improve sampling efficiency and also is shown to stabilize training over other baseline methods.

### Weaknesses
1.	In the current writing, it is hard to identify the novel parts that are proposed by the author and parts that are kept from other works.
2.	Section 4.4 seems very important to understand how to adapt SAC to episodic reinforcement learning but it seems not clear.
3.	Given the resemblance of the critic training to the prediction of rewards that can be used for more effective credit assignment compared to dense and sparse rewards, additional ablation studies or baselines could be discussed.

### Questions
Comments:

1.	In Eq. (8), the future return after N-step is modeled by the critic network during training? It is interesting that if this is the case, the training is stable.
2.	In the reviewer’s view, it is hard to grasp the need for the enforcement of the initial condition in section 4.3.1 as s_0^k is already a condition to generate the trajectory in the policy. Some additional intuition about how this is related to the division in segments, or because of the variable length used, or the sampling of trajectories would clarify this part to the reader.
3.	Another critical part is Section 4.4.  It is not clear to the reviewer how SAC is adapted to the new policy objective during training. 
4.	In Fig.4, why for (c) and (d), the TOP-ERL training is stopped before the other baselines?
5.	The impact of choosing random segments seems to be crucial to performance. Any intuition regarding this result?
6.	Given the resemblance of the critic training to the prediction of rewards that can be used for more effective credit assignment compared to dense and sparse rewards, the reviewer thinks what is the relation between the proposed methodology and using other networks that predict future rewards for more effective credit assignment in performance for the robotic tasks in Section 5.

Minor Comments:

1.	Line 62: Typo “Transformer”
2.	Lines 124-130: The writing in these lines is very confusing. It could be rewritten more clearly.
3.	Title Section 4.5: Is it “summarize” the right word here?
4.	Line 425: Typo “empirical”
5.	Line 439: Typo “Noteably”
6.	Line 471: Typo “sprase”

### Soundness
2

### Presentation
3

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
This paper presents TOP-ERL, Transformer-based Off-Policy Episodic RL, which leverages the Transformer as a critic to predict the value of action sequences, splits it into smaller segments and inputs them into the Transformer for value prediction, with an off-policy update scheme.

### Strengths
This paper is easy to follow. Given that research on episodic RL is relatively limited, TOP-ERL shows its strengths in sample efficiency and overall performance.

### Weaknesses
1. The code repository is incomplete, lacking the environment configuration files and key training files. Additionally, several important functions, such as ValueFunction, have not been implemented.

2. Figure 4(a) only presents the average results for the 50 tasks, without showing the individual results for each task. While I understand that the main text has space constraints, I strongly recommend including the results for all 50 tasks in the appendix.

3. The methodology seems to lack novelty, as important components follow the design of previous episodic RL methods. I did not observe other core contributions aside from the reported strong performance. In the experimental section, it would be beneficial to see experiments and analyses related to the gaussian policy and movement primitives module.

### Questions
1. The paper claims to be "the first off-policy episodic reinforcement learning algorithm," but this seems to be an overclaim. To my knowledge, there are already several related works, such as:
Liang D, Zhang Y, Liu Y. "Episodic Reinforcement Learning with Expanded State-reward Space." arXiv preprint arXiv:2401.10516, 2024.

2. Regarding the statement about "enforcing action initial conditions," can this approach effectively address the mismatch between the state and the corresponding action sequence? In addition to the ODE explanation, could you design experiments to validate this claim?

3. Why does using random segments yield better results than most fixed segments? Could you provide a more intuitive explanation? The correlation presented in Appendix D does not seem to clarify this adequately.

### Soundness
2

### Presentation
3

### Contribution
2
