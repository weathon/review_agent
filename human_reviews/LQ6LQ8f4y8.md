# CCIL: Continuity-Based Data Augmentation for Corrective Imitation Learning

- Decision: Accept (poster)
- Scores: 5, 6, 6, 6

## Abstract
We present a new technique to enhance the robustness of imitation learning methods by generating corrective data to account for compounding error and disturbances. While existing methods rely on interactive expert labeling, additional offline datasets, or domain-specific invariances, our approach requires minimal additional assumptions beyond expert data. The key insight is to leverage local continuity in the environment dynamics. Our method first constructs a dynamics model from the expert demonstration, enforcing local Lipschitz continuity while skipping the discontinuous regions. In the locally continuous regions, this model allows us to generate corrective labels within the neighborhood of the demonstrations but beyond the actual set of states and actions in the dataset. Training on this augmented data enhances the agent's ability to recover from perturbations and deal with compounding error. We demonstrate the effectiveness of our generated labels through experiments in a variety of robotics domains that have distinct forms of continuity and discontinuity, including classic control, drone flying, high-dimensional navigation, locomotion, and tabletop manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to enhance the robustness of imitation learning methods by generating corrective data to account for compounding error and disturbances. Their work is based upon utilizing the local continuity in the environment dynamics. The paper augments the original expert's dataset with generated corrective labels within the neighborhood of the demonstrations but beyond the actual set of states and actions in the dataset. The authors' argue that this augmentation helps the agent to recover from perturbations and deal
with compounding error.

### Strengths
- Their problem is well-defined, and methods are explained clearly.
- Related work covers basic prior work.
- The data augmentation part where the authors utilize the local continuity of dynamics model helped them achieve better performance than the basic Behavioral Cloning Algorithm.

### Weaknesses
- While the writing was clear and easy to understand, the paper lacked substantial content. I didn't find any need to pause and think while reading and I skimmed through the paper rather quickly .
- Performance comparisons of their work are only done with basic Behavior Cloning and NoiseBC algorithms that are basic Imitation Learning (IL) Algorithms. Comparison with state-of the-art IL methods are missing.
-  I would recommend the authors to include experiments to compare the sample efficiency with other state of the art algorithms in terms of trajectories needed as that is also an important metric in IL paradigm.
- There are many Offline IL algorithms proposed recently in literature that have the same settings where they don't make any new interaction with the environment or the expert. Comparisons with them would be interesting to see.
- I would recommend the authors to also report results on Humanoid environment from Mujoco.

### Questions
Please check the Weaknesses part.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes the data augmentation method for behavioral cloning (BC) utilizing the local Lipschitz constraint. To train the forward dynamics from expert data, the proposed method (CCIL) minimizes mean-squared error with the regularization that is computed from the local Lipschitz constraint. Then, two techniques are proposed to generate transition triplets that can be used as expert data.  Once the dataset is augmented, naive BC is applied to find a policy. CCIL is evaluated on various tasks and outperforms BC and NoiseBC.

### Strengths
1. Although the proposed idea is simple, the experimental results show that CCIL is very powerful even if the environmental dynamics is not globally continuous. 
2. The authors evaluated CCIL on various tasks, and it suggests that the proposed method is appealing to practitioners. 
3. The manuscript is written well and easy to follow and understand.

### Weaknesses
1. My major concern is that the proposed method has to solve relatively complicated optimization problems. For example, Equation (3) contains two complicated terms: Lipschitz constraint and L0 norm. How to deal with the max operator in the Lipschitz constraint term is unclear. 
2. The proposed method assumes a deterministic transition function. I am curious when the proposed method is applied to stochastic systems.

### Questions
1. The proposed method is formulated in a discrete-time state transition model, whereas the corresponding true system operates in continuous-time. Therefore, the proposed method implicitly applies a time discretization. In this case, the time interval is critical, and I think the Lipschitz constant depends on the time interval. How did the authors determine an appropriate Lipschitz constant? Or, are there any assumptions on the time discretization in the proposed method? 
2. I do not fully understand the major differences between the techniques of the proposed method and Data as Demonstrator (DaD) proposed by Venkatraman et al. (2015)? I think the core idea is similar; therefore, it is worth discussing the advantages of the proposed method.  
3. Two augmentation techniques are proposed, but I am unsure whether either would be equally useful. Is it possible to conduct an additional ablation study where one of the techniques is removed? 
4. In the paragraph above Definition 4, the authors introduce $\mathrm{Support}(d^\pi)$, but it is not defined. Is $d^\pi$ a stationary distribution induced by $\pi$? 
5. The first paragraph on page 5: $\hat{f}(s_t, a_t) \to s_{t+1}^* - s_t^*$ should be $\hat{f}(s_t^*, a_t^*) \to s_{t+1}^* - s_t^*$. 
6. Is $\bar{\lambda}$ in Equation (3) is an average of $\{ \lambda_j \}_j$? 
7. Please define $f'$ in Equation (3) explicitly.
8. Regarding the technique 1 (Backtrack label), what does "xlabel" mean?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This study is dedicated to enhancing the robustness of imitation learning through the generation of corrective data, compensating for compounding errors and disturbances. Numerous experiments have been executed on an array of tasks, ranging from drone navigation and locomotion to robot manipulation, to validate the effectiveness of the proposed approach.

### Strengths
1. This work offers a detailed theoretical analysis, providing evidence that the quality of the generated label is bounded under specific assumptions related to the dynamics.

2. Various tasks ranging from drone navigation to locomotion and robot manipulation have been extensively experimented and analyzed in the study.

### Weaknesses
1. The proposed method is only compared to vanilla BC and noisy BC. The proposed method declares it constructs a dynamics model for policy learning and has used implementation with a model-based RL framework; therefore, it would be more robust to also include a comparison with other model-based RL methods. As model-based RL also constructs a dynamic model first before planning the most effective actions.

2. There is a lack of clarity in important implementation details. The process of generating corrective labels is discussed in Section 4 but the paper does not make it clear how these labels are employed in later stages. The additional corrective labels could be used to train the imitation learning agent, presumably a neural network? However, if this is the case, further details on the network's implementation could be discussed.

### Questions
The reason why noise BC underperforms compared to vanilla BC is not clear. If we reduce the noise added to the BC, noiseBC's performance should align more closely with that of vanilla BC. Nonetheless, in multiple tasks, noiseBC exhibits significantly poorer results. This could potentially be attributed to the fact that the added noise has not been not carefully chosen and thus, an excessive amount of noise has been injected into the system?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work presents a method for augmenting imitation learning data by learning locally lipschitz-continuous dynamics models and then generating additional labels by perturbing the action to find noisy states as well as tracing states that would lead to the current state with the current action according to the learned dynamics model. Experiments on a diverse set of simulated tasks demonstrate the effectiveness of the proposed method.

### Strengths
Offline data augmentation is an important area of research that could lead to more robust  policies. This paper proposes an intuitive solution by generating additional data around existing data points by querying a locally smooth dynamics model.
The proposed solution categorizes two types of data augmentation: one by perturbing action labels and finding states that would land in the next state given this noise label, and the other by tracing states that would land in the current state given the current action.
This work presents thorough evaluation of the proposed method by experimenting with diverse task settings ranging from controlling a drone to manipulation tasks that have discontinuous dynamics.

### Weaknesses
The theoretical and algorithmic contribution is novel and exciting but the empirical results are not as impressive.

It would be great if the authors could test augmenting the training data with ground truth dynamics models (isn’t it deterministic -> computable given low-dimensional state representations?) to showcase the full potential of data-augmentation-based methods and situate the performance of the proposed method: i.e. help the audience understand if the performance gain/no-gain attribute to additional data or quality of the dynamics model.

This work could also benefit from additional experiments with varying number of demonstrations in a particular domain to show how much data is needed to learn a good dynamics model and at the same time could still benefit from additional augmentation data. 

This work only conducted experiments in simulation, where dynamics models are deterministic and different from real applications. The authors should comment more on what challenges there would be to apply the proposed method in the real world and if one can benefit more or less from this paradigm of data augmentation.


----- Edit ------
The authors presented additional results during rebuttal that address some of my concerns about the evaluation. However, I do think a real-world experiment is practical and valuable for the true impact of this paper, given the proposed method is fully offline.

I am happy to raise my evaluation to weakly accept.

### Questions
See weakness for major concerns.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
