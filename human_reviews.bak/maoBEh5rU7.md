# FlipNet: Fourier Lipschitz Smooth Policy Network for Reinforcement Learning

- Decision: Reject
- Scores: 6, 1, 6, 3

## Abstract
Deep reinforcement learning (RL) is an effective method for decision-making and control tasks. However, RL-trained policies encounter the action fluctuation problem, where consecutive actions significantly differ despite minor variations in adjacent states. This problem results in actuators' wear, safety risk, and performance reduction in real-world applications. To address the problem, we identify the two fundamental reasons causing action fluctuation, i.e. policy non-smoothness and observation noise, then propose the Fourier Lipschitz Smooth Policy Network (FlipNet). FlipNet adopts two innovative techniques to tackle the two reasons in a decoupled manner. Firstly, we prove the Jacobian norm is an approximation of Lipschitz constant and introduce a Jacobian regularization technique to enhance the smoothness of policy network. Secondly, we introduce a Fourier filter layer to deal with observation noise. The filter layer includes a trainable filter matrix that can automatically extract important observation frequencies and suppress noise frequencies. FlipNet can be seamlessly integrated into most existing RL algorithms as an actor network. Simulated tasks on DMControl and a real-world experiment on vehicle-robot driving show that FlipeNet has excellent action smoothness and noise robustness, achieving a new state-of-the-art performance. The code and videos are publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new policy network architecture called Fourier Lipschitz Smooth Policy Network (FlipNet), with the aim to address the action fluctuation problem in deep RL. Through identifying  With a continuous-time analysis, two causes of action fluctuation are identified, i.e., policy network smoothness and observation noise. FlipNet then adopts two techniques, Jacobian regularization and Fourier filter layer, to deal with the two causes respectively. The proposed method is evaluated in a classic linear quadratic control task, four DMC tasks, and four Mini-vehicle driving tasks, showing a great reduction of action fluctuation with a slight score improvement at the same time.

### Strengths
- The paper is well written and organized. Illustrations are well used to describe the ideas and methods.
- The motivation is almost clear. The transition to the proposal of the two techniques is smooth. The differences between the proposed method and the previous methods are presented clearly.
- Three types of environments are used for evaluation. Other results for the ablation study, and hyperparameter analysis are provided in the appendix. I recommend the authors adjust the content and move the ablation study from the appendix to the main body, because I believe it is important for the audience to understand the effect of the proposed method.

### Weaknesses
- The motivative analysis in Equation 5 is good. However, taking the second term at the right hand side of the equation as the observation noise needs more discussion and justification. A natural question here is, whether the second term should be related to the transition dynamics (as well as the actions selected by policy). This concern is not well discussed and addressed in Section 3.1.
- As the observation noise is one of the main courses identified in this work. More discussion and analysis are needed. What are the noise models? In Section 4.1, a uniform distribution is used as the model of noise. For DMC, Table 9 presents the uniform distributions. And for Mini-vehicle driving, the noise model used is not clear in the main body and also in the appendix L.
- FlipNet introduces three new hyperparameters, i.e., $\lambda_k, \lambda_h$ and $N$. According to Table 6, it seems that the backward computation time of FlipNet is about 10 times of MLP. Therefore, it remains questionable whether the proposed method can be useful and feasible in other problems.
- Four tasks from DMC are a bit insufficient to evaluate the performance of the proposed method thoroughly. I suggest the authors add 4 more tasks, e.g., dog, quarduped, hopper, ant. The choice of noises for the noisy environments is not well discussed and justified (see my question below).
- Some experimental details are missing. Please refer to my questions below.

&nbsp;

### Minors

- In Line 51, “MLP-SN suffer” should be “MLP-SN suffers”.
- In Line 124, “Take DDPG as an example again” should be “Taking DDPG as an example again”.

### Questions
1. In Equation 5, is the second term at the right hand side of the equation also determined by transition dynamics?
2. $j$ is not defined in Equation 8. What is definition and meaning of it? Should it be $i$, i.e., the imaginary unit?
3. The trainable filter matrix $H$ depends on the dimensionality of the observation $D$. What if $D$ is high as commonly seen in real-world problems? Will it increase the training difficulty and cripple the performance of the proposed method? I would like to see more discussion or additional experiments on this point.
4. According to Table 6, it seems that the backward computation time of FlipNet is about 10 times of MLP. I would like to know the training time of FlipNet, as the back-propagation is different a lot from regular policy networks. A comparison regarding wall-clock training time between FlipNet and regular MLP will help a lot.
5. In Line 269, the authors mentioned “By choosing H as a complex matrix instead of real matrix, the Fourier filtering layer can not only alter frequency amplitudes but also perform feature extraction”. Can the authors provide more explanation for this?
6. How many seeds are used for Figure 9, Table 10, Table 11?
7. How are the observation noise amplitudes in Table 9 chosen?
8. Why are there no error bars in Figure 10 and Figure 13?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper focuses on action fluctuation ratio and tries to minimize the consecutive action change between two temporally adjacent states. The submission claims to prove the Jacobian norm is an approximation of Lipschitz constant and proposes to add a Fourier filter layer to deal with the action change when there is observation noise. Some experiments in DeepMind Control Suite are provided. However, most of the experiments are conducted in a non-standard environment called “Double Integrator” and in a non-standard robotic application.

### Strengths
Enhancements in reinforcement learning would be of interest to the machine learning community.

### Weaknesses
Authors write in the submission in page 2 that:

*“The code is publicly released to facilitate the implementation and future research.”*

Again the authors write that at page 6

*“The code is publicly available at.”*

These are false statements. In the link the authors provide there is a GitHub repository and it states that:

*“Code and full article will be released after review stage.”*

If the authors did not want to share their code, they should honestly state in their paper as well that the code is not going to be shared. 


The main theoretical contribution of the submission is not stated rigorously, you cannot just state that X is an approximation of Y in a formal theorem statement. Furthermore, the relationship between the norm of the Jacobian and the Lipschitz constant is a well-known fact that can be found in any multivariable Calculus textbook. The authors cannot claim that these results are their contributions. 

The submission states as a main contribution that they identify the two fundamental reasons that cause action fluctuation. However, this is rather something obvious and has been clearly discussed before. Authors keep mentioning throughout the paper in the same manner as these “fundamental reasons”. I find this kind of writing quite misleading and unnecessary. 

One problem I have is the action oscillation ratio metric itself that is introduced in [1]. It is not clear at all why we should measure this or try to minimize this metric. In high dimensional action and state space there can be differences in actions even though somehow the $\ell_p$-norm distance between two consecutive states is smaller does not mean the $\ell_p$-norm distance between the actions between these two states must also be smaller. It is not at all clear that this is a straightforward good approach. As far as I can see this action oscillation only appears in two papers in the past 5 years. I am not sure the submission's claims on how much researchers want to solve this extremely important point is accurate.

Furthermore, the action fluctuation ratio in Equation 4 that the submission states is a different metric than what the original paper states as action fluctuation ratio [1]. This should be also clearly stated here.

[1] Chen Chen, Hongyao Tang, Jianye Hao, Wulong Liu, Zhaopeng Meng. Addressing Action Oscillations through Learning Policy Inertia, AAAI 2021.


In section 4.2 in the Deepmind Control Suite results why is only the TD3 algorithm tested? There should be more baselines in the main training environment. SAC, TRPO and PPO should also be included in these results. Furthermore, in the DeepMind Control Suite it is usually tested in more environments including Hopper and Ant.

The caption for the bar graph in Figure 9 should not be covering the actual results. The results in the DeepMind Control Suite demonstrates that the total average return obtained by the FlipNet is within one standard deviation with the total average return obtained by MLP. From these results it is not clear FLipNet performs better. In particular, the total average return in the noise free Cheetah environment of MLP is 816$\pm$30 and 829$\pm$15 of FlipNet. The total average return in the noise free Reacher environment of MLP 981$\pm$10 and 988$\pm$10 for FLipNet. All of these results throughout the entire table are within one standard deviation. 

It is also insufficient to only compare one method [1] in the main results.

[1]  Xujie Song, Jingliang Duan, Wenxuan Wang, Shengbo Eben Li, Chen Chen, Bo Cheng, Bo Zhang, Junqing Wei, Xiaoming Simon Wang. LipsNet: A Smooth and Robust Neural Network with Adaptive Lipschitz Constant for High Accuracy Optimal Control, ICML 2023.

### Questions
See above.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates the issue of action fluctuations encountered in policy training for reinforcement learning, explicitly stating that this problem arises from two factors: policy non-smoothness and observation noise, and addresses these two causes in a decoupled manner. For policy non-smoothness, a Jacobian regularization technique is introduced to enhance the smoothness of the policy network. To address observation noise, a Fourier filter layer is incorporated to improve noise robustness. This paper is the first to decouple the study of the causes leading to action fluctuations, providing new insights for further research on this issue.

### Strengths
1. This paper clearly identifies the two fundamental causes of action fluctuations and resolves them through a decoupled approach.  
2. This paper conducts rigorous formula derivations and proofs.  
3. This paper implements good encapsulation of the network, facilitating further research and use.  
4. This paper conducts experiments in various simulation and real-world environments, achieving state-of-the-art performance and promoting the application of reinforcement learning in the real world.

### Weaknesses
1. The compared methods should be consistent across different experimental environments. For example, methods (MLP-SN,    LipNet-G) should be compared in double integrator environment.
2. The figure in Section 3.4 lacks the figure number and caption.

### Questions
1. The compared methods should be consistent across different experimental environments. 
2. In Equation (5), the second term $\frac{do_t}{d_t}$ reflects the change rate of $o_t$.  This change rate is not equal to level of observation noise. Why does this term reflect the level of observation noise?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces a novel neural network architecture designed to address action fluctuation in reinforcement learning tasks. The proposed approach decomposes the problem into two distinct components: observational noise and policy smoothness. These challenges are addressed independently through a Fourier-based filter layer and a Lipschitz constant regularizer on the policy, respectively. Experimental results demonstrate that their method achieves superior performance compared to MLP-SN and Lipsnet across multiple tasks.

### Strengths
1. The theoretically grounded separation of observational noise and policy smoothness as distinct challenges to solve when approaching action fluctuation provides valuable insights for the community.
2. The paper includes clear and concise plots
3. The result that the jacobian norm can be used as a lipschitz constant estimate is an important discovery, and the proofs are rigorous
4. The sensitivity analysis is thorough and convincing, substantiating the claim that unlike some existing methods, their method is not significantly sensitive to the choice of hyperparameters.
5. The experimental section is thorough, including a real world test, and ablation studies, and comparisons against existing methods on standard benchmarks.

### Weaknesses
1. The authors present that one their main contributions is that Flipnet can be used just like an mlp in standard RL algorithms, making it seem like a drop in replacement for existing RL algorithm implementations based on pytorch. However, this does not seem possible to me as the fourier transform layer specifically requires the history of observations to be stored. Most standard implementations do not store the history of observations as they assume that the environment is a markov decision process, and often rely on this assumption to make their implementations efficient.
2. The code is also defined as publicly available but it is not provided until after the review process is complete, which means I can't verify their previous claim.
3. The backwards pass is approximately 3x slower than Lipsnet and 10x slower than a standard MLP, and forward computation is twice as expensive as MLP-SN.
4. Lack of any mention of the limitations of the method in the main paper (for example comutational time is burried in the appendix). They dismiss alternative methods as needing substantial changes to existing RL algorithms, thereby not needing to compare against them, but many of these methods can also be cast as part of the neural network architecture, for example CAPS and L2C2 can be defined in the backwards pass as part of their loss functions.

If the authors resolve the limitations representation issue presented above, I will substantially improve my recommendation for the acceptance of this paper and my soundness score. It is otherwise strong enough to be publishable regardless of the performance against the other related methods and computational time concerns.

### Questions
1. Have the authors considered using more efficient filtering methods, such as Butterworth filters, to reduce computational overhead, since the motivation is real time applications?
2. How does the architecture perform in environments requiring rapid reaction times?
3. Can the authors provide more justification for using complex matrices in feature extraction? As it seems to go against the notion of separation of concerns.

### Soundness
2

### Presentation
4

### Contribution
4
