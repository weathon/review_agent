# Privileged Sensing Scaffolds Reinforcement Learning

- Avg Score: 8.50
- Decision: Accept (spotlight)
- Scores: 10, 8, 8, 8

## Abstract
We need to look at our shoelaces as we first learn to tie them but having mastered this skill, can do it from touch alone. We call this phenomenon “sensory scaffolding”: observation streams that are not needed by a master might yet aid a novice learner. We consider such sensory scaffolding setups for training artificial agents. For example, a robot arm may need to be deployed with just a low-cost, robust, general-purpose camera; yet its performance may improve by having privileged training-time-only access to informative albeit expensive and unwieldy motion capture rigs or fragile tactile sensors. For these settings, we propose “Scaffolder”, a reinforcement learning approach which effectively exploits privileged sensing in critics, world models, reward estimators, and other such auxiliary components that are only used at training time, to improve the target policy. For evaluating sensory scaffolding agents, we design a new “S3” suite of ten diverse simulated robotic tasks that explore a wide range of practical sensor setups. Agents must use privileged camera sensing to train blind hurdlers, privileged active visual perception to help robot arms overcome visual occlusions, privileged touch sensors to train robot hands, and more. Scaffolder easily outperforms relevant prior baselines and frequently performs comparably even to policies that have test-time access to the privileged sensors. Website: https://penn-pal-lab.github.io/scaffolder/

## Human Reviews

## Human Reviewer 1

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose Scaffolder, a model-based RL method that can leverage privileged information at training time. Here, priviledged information means an MDP where:

* There is some true state $s$ with observations $o^+$, where $o^+$ includes privileged info we do not want to assume is available at inference time. (i.e. ground truth object state). This could be used to train a privileged policy $\pi^+$ that cannot be used as-is for inference.
* There is an observation $o^-$ for unprivileged / impoverished target observations, which our final policy $\pi^-$ will depend on.
* We would like the best $\pi^-$ possible while leveraging information in $o^+$.

This problem has been studied in a number of recent works, often in a model free manner. This paper aims to leverage privileged information in a model-based manner.

To do so, the authors train 2 worlds models. The world model subroutine used is DreamerV3. One models privileged information $o^+$ and the other models target information $o^-$. In this summary I'll call them WM+ and WM-.

A "latent translator" is learned to translate WM+ into an observation that WM- can use in its rollouts. Specifically: WM+ has internal latent state $z^+$. We fit a prediction model $p(e^-|z^+)$, where $e^- \approx emb(o^-)$. Part of DreamerV3 is learning a posterior $q(z_{t+1}|z_t,a_t,e_{t+1}=emb(o_{t+1}))$ that infers latent state from history and current observation. By replacing the impoverised $e^- = emb(o^-)$ with a prediction driven by privileged latent $z^+$, we can channel some privileged information into the rollout of $z^-$, assuming that privileged information is eventually observable in unprivileged information.

This latent translator lets us use $\pi^-(a|z^-)$ to rollout both WM+ and WM-, giving a sequence of latents $(z^+,z^-)$ from both world models. The learned reward function is then defined as $R(z^+,z^-)$ to allow observing privileged information in the critic.

We additionally fit a $\pi^+$ directly in the privileged world model, using this solely to generate additional exploratory data (it is possible that some exploration behaviors are easier to learn or discover from privileged information). Last, a decoder is trained to map $z^-$ to $o^+$. To me this seems the least motivated, in that not all parts of $o^+$ should be predictable from $z^-$ in the first place, but it seems to empirically be effective.

The evaluation of Scaffolder is done in a variety of "sensory blindfold" tasks, mostly robotics based, where some sensors are defined as privileged and some are not. The method is compared to DreamerV3 on just target information, a few variants of DreamerV3 based on only fitting one world model with decoding of privileged information, and some model free baselines like slowly decaying use of privileged information, asymmetric actor critic, or using BC to fit an unprivileged policy to a privileged one.

### Strengths
The paper provides a good overview of previous methods for handling privileged information, proposes an evaluation suite for studying the problem of privileged information, and proposes a modification of DreamerV3 that handles the information better than Informed Dreamer. There is a significant amount of machinery around Scaffolder, but it's mostly clear why the added components ought to be helpful for better world modeling and policy exploration. The model-free baselines used are pretty reasonable, and it is shown that Scaffolder still outperforms these model free methods even when the model free methods are given significantly more steps.

Finally, the evaluation suite covers a wide range of interesting robot behaviors and the qualitative exploratory methods discovered to handle the limited state (i.e. spiraling exploration behavior) are quite interesting. The S3 suite looks like a promising testbed for future privileged MDP work, separate from the algorithmic results of the paper.

### Weaknesses
In some sense, Scaffolder requires doing 2x the world model fitting, as both the z^- and z^+ models need to be fit for the approach to work. In general, this is "fair" for model-based RL, which is usually judged in number of environment interactions rather than number of gradient steps, but it very definitely is a more complex system and this can introduce instability.

A common actor-critic criticism is that the rate of learning between the actor and critic needs to be carefully controlled such that neither overfits too much to the other. Scaffolders seems to take this and add another dimension for the rate of learning between the privileged world model and unprivileged one, as well as the learning speed of the privileged exploratory actor $\pi^+$ and target actor $\pi^-$.

### Questions
In general, the paper focuses on the online, tabula rasa case where we are in an entirely unfamiliar environment. How adaptable is this method to either the offline case, or the finetuning case where we have an existing policy / world model?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The learning process known as "sensory scaffolding" involves novice learners using more sensory inputs than experts. This principle has been applied in this study to train artificial agents. The researchers propose "Scaffolder," a reinforcement learning method that utilizes privileged information during training to optimize the agent's performance.

To evaluate this approach, the researchers developed a new "S3" suite of ten diverse simulated robotic tasks that require the use of privileged sensing. The results indicate that Scaffolder surpasses previous methods and frequently matches the performance of strategies with continuous access to the privileged sensors.

### Strengths
This paper delves into a critical question within the field of reinforcement learning: how can we effectively use privileged information as a 'scaffold' during training, while ensuring the target observation remains accessible during evaluation? This question takes on an added significance in robotic learning, where simulation is a major data source.

While there has been considerable research in this area, as detailed in the related work, this paper adds value to the existing body of knowledge, even without introducing novel methods. The proposed method may not be groundbreaking, but it offers a comprehensive examination of this issue from four perspectives: model, value, representation, and exploration.

This research serves as a valuable resource for those looking to deepen their understanding of the field. The excellent writing and presentation of this paper further enhance its contribution. Overall, despite the lack of methodological novelty, the paper is worthy of acceptance due to its systematic exploration and clear articulation of the subject matter.

### Weaknesses
1. Increasing the clarity around the Posterior and detailing how it is used to transition from the privileged latent state to the non-privileged latent state would greatly enhance understanding of the method.
   
2. The related work section could be expanded to include research papers that leverage privileged simulation reset to improve policy. These works also seem to align with the scaffolding concept presented in this paper. Papers such as [1][2] could be added for reference.

3. In the experimental design, the wrist camera and touch features don't appear to be excessively privileged or substantially different from the target observations. It would be beneficial to experiment with more oracle-like elements in the simulator as privileged sensory inputs. For instance, the oracle contact buffer or geodesic distance could be considered.


[1] DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills

[2] Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation

### Questions
1. More clarification on the posterior and embedding component in the method part. 
2. More clarification of other scaffolding methods in the related work, no need for any experiments.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes to utilize privileged sensory information to improve every component of model-based reinforcement learning, including world model, exploration policy, critic, and representation as well. This work provides extensive evaluation over 10 environments including different kinds of sensory data, showing the proposed method outperform all representative baselines. This work also provide detailed ablation study over all environments showing the

### Strengths
1. This work provides systematic analysis over different components in the “sensory scaffolding” setting, and proposes corresponding scaffolding counterparts  of every component in MBRL, except the policy during deployment. 
2. This work provides a promising evaluation comparison with multiple representative baselines, demonstrating that with the proposed pipeline, privilege information improves the sample efficiency as well as the final performance over wide-range of tasks.
3. Through ablation study, this work shows different components in the system boost the performance in a different way, providing additional insight on how privileged information can be used in the future work.
4. Experiment details are well presented in the Appendix, including runtime and resource comparison over different methods on different environments.
5. The overall presentation of the work is good, considering the complexity of the system and amount of information delivered.

### Weaknesses
1. For scaffolded TD error comparison, it’s not clear why the comparison is conducted on Blind pick environment, since the gap between the proposed method and the version without scaffolded critic is much larger (at least in terms of relative gap) on Blind Cube Rotation environment. Also it would be great to see whether the estimate is close for tasks like Blind Locomotion (since the gap is small on that task). It seems there is some obvious pattern in the Figure 9, that the scaffolded TD is worse at 5, 10, 15 epoch and performs best on 7, 12, 18 epoch, it would be great to have some explanation for that.
2. For some claims made in the paper, it’s actually not quite convincing. For “In other words, much of the gap between the observations o− and o+ might lie not in whether they support the same behaviors, but in whether they support learning them.”, some additional visualization like trajectory visualization might be helpful to strengthen the claim, since the similar reward score does not necessarily result in similar behavior. 
3. For runtime comparison, since the speed of given GPUs varies a lot, it might be better to compare the wall-time with similar system configuration, assuming the wall-time is consistent across different seeds.

### Questions
1. Refer to weakness. 
2. Regarding some technical details, is a bit confusing:
* In section C1, it says “We launch 4-10 seeds for each method”, what’s the exact meaning of using different number seeds across methods or across environments?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes Scaffolder, a MBRL method that extends DreamerV3 with privileged information in its modules. Scaffolder uses privileged world models and exploration policies to roll-out trajectories to train a better target policy. To ensure consistency between target and privileged latent, Scaffolder proposes to predict target latent from privileged latent, bottlenecked by target observation. Scaffolder outperforms baselines on the newly proposed S3 benchmark.

### Strengths
+ The paper is well written and motivated. The presentation is clear.
+ Strong empirical performance.

### Weaknesses
- I agree that it makes sense to evaluate the proposed method on the newly proposed benchmark, for motivations mentioned in the paper. However, the paper would still benefit from evaluating extra existing benchmarks, just for reference. 
- One major benefit of privileged information reinforcement learning is to train the target policy with privileged information in simulation, and deploy it in the real world where there is no privileged information. However, all experiments in the paper are purely in sim. Can the authors comment more on how well the presented approach will work in real-world applications?
- In addition to the number of frames being the x-axis for figure 6, please also include one where x-axis is the wall-clock time. This way the community will have a better understanding of how the proposed method and baselines work on this particular environment set.

### Questions
- Real-world applications. Please see details in the weaknesses section above.
- I am curious about one particular design choice. Why do the authors choose to predict target latent from privileged latent, bottlenecked by target observation. Why not usethe same latent, shared by both the privileged and target modules?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
