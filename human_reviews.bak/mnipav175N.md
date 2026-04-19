# Open the Black Box: Step-based Policy Updates for Temporally-Correlated Episodic Reinforcement Learning

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Current advancements in reinforcement learning (RL) have predominantly focused on learning step-based policies that generate actions for each perceived state. While these methods efficiently leverage step information from environmental interaction, they often ignore the temporal correlation between actions, resulting in inefficient exploration and unsmooth trajectories that are challenging to implement on real hardware. Episodic RL (ERL) seeks to overcome these challenges by exploring in parameters space that capture the correlation of actions. However, these approaches typically compromise data efficiency, as they treat trajectories as opaque black boxes. In this work, we introduce a novel ERL algorithm, Temporally-Correlated Episodic RL (TCE), which effectively utilizes step information in episodic policy updates, opening the 'black box' in existing ERL methods while retaining the smooth and consistent exploration in parameter space. TCE synergistically combines the advantages of step-based and episodic RL, achieving comparable performance to recent ERL methods while maintaining data efficiency akin to state-of-the-art (SoTA) step-based RL.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an approach for episodic reinforcement learning. In such a setting, the policy outputs a high-level trajectory instead of 
a single action per state. This idea has been extensively explored (e.g., hierarchical reinforcement learning). The main technical innovations of this work are (1) optimizing over trajectory chunks instead of the whole trajectory and (2) modeling the correlation between states observed during the chunks. The paper is extensively evaluated in multiple benchmarks.

### Strengths
The approach is technically sound. While the change from previous work (Otto et al. 202; Li et al. 2023) is relatively limited, the proposed innovations are novel. 
The extensive experiments show both the advantages and limitations of the approach.

### Weaknesses
The main weaknesses I see in the approach are:
1. Integrating over multiple timesteps breaks the markov assumption from which the learning objective in Eq. 8 is derived. This is because, given the multi-state parametrization, a state s_t depends on more than s_{t-1}. This is true not only for the proposed approach but also more generally for every method that introduces a dependency between states (e.g. smoothing). Previous work even noticed that such smoothing can hurt performance (see Smith et al., A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning).
2. The dependence on a hand-picked segment length K. While having some parts of the algorithm heuristically tuned is generally not a problem, I find this parameter to possibly be challenging to pick (due to its dependence on the task) and time-varying. My intuition is that the problem does not appear in the selected experiments due to their simplicity. I guess it will be much more challenging as soon as the time derivative of the reward drastically changes over time.  A cue in this direction is the failure case on the table tennis task, which shows these characteristics (when the ball is close to the end-effector, the robot needs very high-resolution updates, while a longer horizon is sufficient in the waiting times. This problem is not unusual in most robot tasks: for example, in visual navigation or locomotion, the interdependence between states is rarely constant (e.g., when a sudden obstacle appears, one needs to react fast). I think the paper would benefit from studying this aspect in more detail. Could there be a connection between the derivative of the value function and the segment length K? Can it be learned with the task? 

A second limitation is in the experimental setup. Specifically:
a) I find some of the results to be difficult to interpret. In Fig. 6b-c, all methods appear to perform similarly in the dense and sparse setting, which I find surprising. Even more so, almost all approaches (including gSDE, which achieves the best performance) perform better with sparse rewards than dense ones. Why is this case? At least for step-wise methods, dense should do better. Why does this happen? 
b) I think that results should be reported for a longer number of interactions. In Fig. 5,6, and 7, the methods do not appear to be fully converged. Showing results up to convergence (as in Fig. 5a) would give a better intuition about the approach's strength in comparison to the baselines.

Other relatively minor limitations:
1. The baselines are not defined (e.g., what paper is BBRL?)
2. The comparison of trajectory-based to state-based is not entirely apple to apple since the trajectory methods require an extra low-level controller, possibly specifically tuned to the robot/task. Their data efficiency could be attributed to this prior. In addition, such a low-level controller is difficult to get in some cases, and a simple PD controller won't do (e.g., locomotion). I would mention this disclaimer to the experimental setup.
3. There is a typo in Eq. 5.

### Questions
I would like to see a detailed derivation of the objective in the proposed multi-step setup. In addition, I would appreciate a discussion and possibly experiments on the value of K and a more in-depth analysis to justify the experimental setup.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This is an empirical paper that proposes a segment (sub-trajectory) based reinforcement learning method Temporally-Correlated Episodic RL (TCE).

This method adopts the trajectory-based policy representation using Probabilistic Dynamic Movement Primitives (ProDMP) [Li et. al 2023]. It parameterizes the trajectory as a full Gaussian distribution, whose parameters are predicted by the policy network. This enables temporally coherent exploration by sampling trajectories from the Gaussian distribution.

Subsequently, they divide the trajectory into segments, and run trust-region policy gradient based on the segment-wise advantage function estimate and the projection technique in [Otto et al. 2021], to update the trajectory parameter. To simplify the computation of the likelihood, they utilize the pairwise technique from Li et. al (2023). 

The efficacy of this approach is demonstrated through a number of robotics manipulation tasks.

### Strengths
- Good empirical performance across a variety of tasks and included negative cases like the table tennis with sparse rewards
- The approach is straight-forward and bridges the gap between the extremes of step-based and trajectory-based reinforcement learning.

### Weaknesses
1. The terminology “episodic RL” could potentially lead to confusion, as it bears similarity to the concept of “episodic task” contrasted with “continuing task” [Sutton and Barto, 2018]. A more fitting term, such as “sub-trajectory based RL,” may better capture the essence of the work.
2. The paper would benefit from a more organized exposition of its contributions. It appears that applying policy gradient to segments is the primary novel component. And this work integrates techniques from Li et al. (2023) and Otto et al. (2021). A clearer differentiation from these prior works would enhance the clarity of the paper.
3. While hyperparameters are listed in the appendix, the rationale behind their selection remains opaque. Expanding on this would add depth to the methodology.
4. I feel that the current experimental setup does not sufficiently justify certain design choices. Consider incorporating ablation studies to address the following:  
    a. The necessity of the differentiable policy projection, as compared to alternatives like those used in PPO.  
    b. The benefits of employing full covariance over diagonal variance in the trajectory representation, especially in light of the increased computational cost. 
5. Formatting:
    Certain pages in the main pdf are images, making it impossible to do a text search in the pdf and hindering readability
6. In Section 2.2, the claim "The most effective episodic RL approaches directly explore in parameter space" is presented without supporting references.

### Questions
Is there a feasible approach to directly represent each segment, rather than representing the entire trajectory?

### Soundness
3 good

### Presentation
2 fair

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
By using parametric trajectory generators such as Movement Primitives (MP), episodic reinforcement learning techniques tackle robot trajectory generation issues by rephrasing them as contextual optimization. Although these methods are efficient in producing smooth trajectories and detecting certain movement correlations, they do not make use of the temporal structure present in trajectories, leading to less sample efficiency. To overcome these problems, the authors provide the Temporally-Correlated Episodic RL (TCE) technique. By sampling multi-second trajectories in a parameterized space, TCE improves exploration efficiency and ensures high-order trajectory smoothness and movement correlation capture. TCE divides the whole trajectory into smaller parts for policy changes, assessing each part according to its unique benefits.

### Strengths
1. Figure 3 efficiently summarizes the entire learning framework.

2. The code base and config files are provided for double-blind review, which enhances the credibility of the study.

### Weaknesses
1. I suggest the authors to spend more time re-organizing this paper. The overall description is very messy at present. There is no good connection between the various parts, making it difficult to follow the storyline.

2. Only previous works and their limitations are discussed in the BACKGROUND AND RELATED WORKS section without explaining how they relate to your own work. What's more, there are a lot of claims or conclusions without adding corresponding references, especially in the EPISODIC REINFORCEMENT LEARNING section.

3. The writing of this paper should be improved. The current expression is too redundant, especially in the LIKELIHOOD COMPUTATION OF A SAMPLED TRAJECTORY SEGMENTS section. The author should reasonably remove some content to refine it.

4. The simulation experiment results do not verify the effectiveness of the method. Compared with strong step-based methods, the proposed method does not demonstrate the obvious final performance improvement. Although the variance of the proposed method is the lowest in the CONTACT-RICH MANIPULATION section, the gSDE approach achieves better performance in 2 of 3 tasks. Moreover, the BBRL Cov. algorithm achieves a 20% higher success rate with faster convergence in the HITTING TASK.

### Questions
The authors are suggested to experiment with some real scenes. The simulation environments now set by the authors can not provide a big enough challenge to verify the robustness of the algorithm.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a framework called Temporally-Correlated Eposodic RL, which is able to generate smooth trajectories and capture the movement correlation. In particular, the contributions are:
A. Change the action space to the parameter space of the smooth trajectories.
B. Reconstruct the temporal correlation using pair information.
C. Complete trajectory sampling.
D. Exact likelihood computation for trajectory segments.

### Strengths
1. The new parameterization of the trajectory seems to guarantee smoothness by construction.

### Weaknesses
1. The novelty of the algorithm is incremental. It is a direct application of Li et al. 2023 to the RL regime, by (1) changing the action space to the parameter space of trajectories, and (2) enlarging the timestep to the segmentation scale.

2. Many terms in the paper are described with words, which makes it hard to understand. I strongly suggest the authors provide mathematical definitions. For example, 

- a. Since this is still an RL approach, define the Markov Decision Process first. What is the action space? What is the state space? Now you can explain how your approach differs from traditional RL more clearly, by providing us the details in this different MDP. A tricky question here is that since MDP is Markovian, why do you need the temporal correlation?

- b. It would be more helpful to define the dimensions of $y$, $w$, $\Sigma$ and $s$.

- c. What are d_mean and d_cov in Equation (4)? Are they Frobenius distances?

3. As a reader, I feel many unimportant texts are taking too many places, which makes it not easy to understand the paper. For example, Sections 3.1, 3.2, and 3.4 don't need so many places to mention, since they are merely (a) complete-trajectory sampling, (b) GAE, and (c) TRPO-like. Instead, providing the pseudo-code for the whole learning process would be more valuable as a complement to Figure 3. Currently, there are too many technical things and curves in Figure 3, which makes it hard to understand the learning procedure from this single figure.

4. The result is not sufficient to show the significance of the proposed method. In particular,

- a. After checking Appendix C.2, I found that the baselines perform generally well when TCE works well. However, there exist some situations where TCE significantly fails and the other baselines still work well, and these situations are not rare, including LeverPull, StickPull, Hammer, ButtonPressTopdown, and ButtonPressTopdownWall. 

- b. Furthermore, the curves for SAC and PINK are terminated after 10M and 27M interactions respectively, whereas the other methods terminate at 50M interactions, which is an unfair comparison. It is hard to show the efficiency of this algorithm unless these baselines are finished. The reason is: in many cases, these baselines would raise their success rates nonlinearly during the training process, and we cannot conclude by only inspecting their intermediate performances.

- c. Several key ablation studies are missing to test the soundness of each proposed component. See the questions.

Finally, there are some typos in the paper. For example, in Equation 5, the left curly brace is missing. In Equation 6, the (|s) looks strange, it might be (s) instead.

### Questions
1. What does black box in the title stand for? Can you help me accurately define it? Does it mean the dynamics is black-box, or the policy approximator (neural network) is black-box? Though this term is repeated several times in the paper, there is no accurate definition in the paper. Furthermore, since the title says 'open the black box', is your algorithm a white-box algorithm?

2. Is Figure 1 a real experiment, or just an illustrative explanation?

3. It seems that Section 3.1 is just a description of whole-trajectory sampling. Is there any novelty in terms of the exploration here? Doesn't normal RL actor also sample using their own stochastic policy?

4. 
> Notably, the off-policy methods SAC and PINK were trained with less samples than used for on-policy methods due to their limitations in parallel environment utilization.

      What does the limitation refer to here? Why cannot SAC and PINK be deployed to parallel training? (Also, 'fewer' samples, not 'less' samples).

5. In terms of the ablation studies:

- a. The PPO and SAC in this paper are only used to generate low-level actions. However, both of them are general RL frameworks, which should also be applicable even if the action space is changed to the parameter space of the trajectory. And these versions in the parameter space of trajectory are missing as the baselines. You can still claim that these new baselines are just some naive version of TCE since they do not really calculate per-segment credit assignment and there's no exact probability calculation. But, showing how these frameworks work in this new parameter space is still important, since now we can evaluate how this different action space could improve the sample efficiency by itself.

- b. Smoothness comparison. Define a metric for smoothness, and evaluate how all the final trajectories for all the methods perform under this metric. Since you claim that TCE helps the smoothness, then only showing the success rate is not sufficient.

- c. The significance of the exact probability calculation. If you abandon this component and just use the normal p(w) instead of p(y), how would the performance change empirically? 

- d. One thing I am wondering is how this temporal correlation helps the sampling complexity. Can you provide the final Sigma for each DoF? Ideally, this matrix should be far from the diagonal matrix, otherwise, we can just use the step-wise algorithm.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
