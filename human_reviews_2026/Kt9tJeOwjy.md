# RFS: Reinforcement learning with Residual flow steering for dexterous manipulation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 2

## Abstract
Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions.
However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors. We propose Residual Flow Steering (RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy.
We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning both in simulation and in real-world settings when adapting pretrained base policies. Project website: https://weirdlabuw.github.io/rfs/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Residual Flow Steering (RFS), a novel reinforcement learning framework for fine-tuning flow-matching policies in dexterous manipulation.
RFS adapts a pre-trained policy by learning to modulate both the initial latent noise (for global exploration) and output a residual action (for local refinement).
The method is evaluated in both simulation and real-world settings, demonstrating improved performance over several baselines. The core idea is intuitive and addresses a relevant challenge in policy adaptation. However, the paper could be strengthened by a more comprehensive related work section, clearer methodological explanations, and more competitive baseline comparisons.

### Strengths
1. This paper proposes a novel RL method for adapting a policy pre-trained with flow-matching to dexterous manipuation tasks.
The adaptation is achieved by training a policy to output an initial noise and a residual action, which correspond to the input and output of the flow policy.
2. The overall ideas and intuition of the method are clear.
3. The paper demonstrate the performance of the proposed method with both simulation and real-world experiments.

### Weaknesses
1. A more comprehensive related work section will enhance the paper.
For example, recent papers on residual RL are not included, such as policy decorator[1].
And some baselines compared in this paper are not introduced in the related work.
2. Some details about the method are not clearly explained. Please check the questions in the following section.
3. The chosen baselines, while reasonable, do not fully demonstrate the advantage of RFS over the state-of-the-art.
A more compelling comparison would be against: (1) Recent RL methods specifically designed for fine-tuning diffusion/flow policies (beyond the ablated DSRL) (2) State-of-the-art residual RL methods that finetune a base policy, to better isolate the contribution of the combined approach.

Minor:
1. The title "Imitation-Bootstrapped Reinforcement Learning" in related work might be slightly misleading. As defined in works like IBRL[2], the term has a specific meaning. The papers [3][4] cited under this heading are more broadly categorized as "RL with Demonstration". It would be helpful to refine this terminology for precision.

[1] Zhiyuan Yuan, et al. "Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model." ICLR 2025.

[2] Hengyuan Hu, et al. "Imitation Bootstrapped Reinforcement Learning." RSS 2024.

[3] Ashvin Nair, et al. "Overcoming exploration in reinforcement learning with demonstrations." ICRA 2018.

[4] Aravind Rajeswaran, et al. "Learning complex dexterous manipulation with deep reinforcement learning and demonstrations." RSS 2018.

### Questions
1. In section 5.2, it is mentioned that base policy actions $a_b$ and human corrections $a$ are recorded and transformations are applied to them to obtain $(a_0, a_r)$ for training. Why is the initial latent noise $a_0$ not recorded directly?
2. The RFS policy $\pi_{RFS}$ outputs $a_0$, which is passed through the "Push" function (an ODE solver involving multiple evaluations of $v_\theta$). Could you clarify if gradients are backpropagated through the entire "Push" operation?
If so, there will be a huge computational cost for the gradient calculation during training and it is unstable. If not, how are the gradients for $a_0$ obtained?
By the way, could you mention the number of steps used for the "Push" operation?
3. The policy $\pi_{RFS}$ is trained to output both $a_0$ and $a_r$. I am curious about the results if we separate the current loss into two terms. For example, detaching $a_r$ when calculating gradients for the $a_0$, or vice-versa, to see if it stabilizes training or improves performance?
4. The process of using RFS for simulation data generation could be explained more clearly.
I got the following questions when reading this part:
First, are VR-teleoperated demonstrations real-world data or simulation data?
Second, the high-level policy $\pi_{RFS}$ is trained with the same demonstrations using an offline RL method or trained in simulation using online RL method?
I can find the answer in Section 6.1: simulation data, online RL method (PPO) while it will be easier for the reader to understand if the Section 5.1 can be rephrased.
5. What is the specific advantage of applying RFS to create a simulation policy first, rather than directly applying RFS to the original base policy $v_{\theta}(a_t, s, t)$ using human-collected real-world data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper focuses on offline-to-online reinforcement learning (RL) setting.
Specifically, the paper proposes to adapt generative policies with reinforcement learning by unifying residual policy learning and diffusion steering into a new class of algorithms, residual flow steering (RFS).
The paper illustrates, through simulated and real-life experiments, that RFS outperforms compared algorithms.

### Strengths
- The unification between residual RL and latent-noise steering is interesting, and can open up avenues for various choices of $f$ and $g$.
- In the offline RL setting, RFS appears to generalize to unknown objects better in the real-life setting. The extra robustness experiments are helpful in demonstrating RFS' benefits.

### Weaknesses
- In section 5.2, the paper proposes to collect extra human correction data $a$---this seems to be a strong limitation, similar to applying the DAgger algorithm. It is possible that I have totally misunderstood this process:
	- The whole trajectory $((o_1, s_1), (o_2, s_2), \dots)$ is generated using the correction action $a$, as opposed to below.
	- First sample the trajectory $((o_1, s_1), (o_2, s_2), \dots)$ using the base policy actions, then obtain the correction actions based on the already collected trajectory.
	- Nevertheless, perhaps the paper can clarify this on lines 296-298.
- Experiments
	- The experimental setting is confusing. In section 5.2 it is mentioned that the RFS is finetuned via offline RL, but in the experiment, specifically section 6.1.2, the setting is now in the online RL setting.
		- In 6.1.2, does the learner obtain any human-correction data for finetuning, in addition to the PPO trajectories?
		- Secondly, if the setting is originally offline-to-online RL, I think it's unfair to compare the learning curve of Tabula-rasa RL, and instead it should be compared against algorithms such as Cal-QL, AWAC-like algorithms, or RLPD.
		- Instead, the rationale on the chosen baselines on action-space reduction and action codebooks are unclear. Is the intuition to reduce the sample efficiency through easier exploration with smaller action space?
	- Likewise, for section 6, the choice of compared algorithms can be strengthened with offline RL algorithms like CQL and IQL.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the problem of fine-tuning pretrained generative (flow matching) policies for reinforcement learning. The contribution of this work is to introduce a general framework that incorporates two types of preexisting fine-tuning methods: input modulation (flow steering) and output modulation (residual learning). The authors propose a specific instance of this framework: Residual Flow Matching (RFS) which learns both an initial latent noise distribution and a policy that outputs residual actions, which is added to the push forward action produced by the pretrained model (pushed forward from the learned noise). The authors conduct simulated and real-life experiments in the dextrous manipulation robotics domain, and the experiments demonstrate RFS achieving favorable performance.

### Strengths
1. The paper integrates two previous methods (flow steering, residual learning) with complementary benefits and drawbacks to get the best features of both.
2. The framework the paper introduces is broad enough to be applicable to many important reinforcement learning applications, even outside the domain of manipulation, especially given the wide adoption of generative policies in various RL applications.

### Weaknesses
1. **Some Baseline choices lack motivation/clarity:**  I do not understand the use of the VQ-VAE and PCA baselines in Section 6.1.2/Table 1/Fig. 4. From my understanding, both are methods to get different state-action representations, whereas the focus of the paper is finetuning. If my understanding is correct, these results are comparing PPO finetuned with RFS with PPO trained for two separate state-action representations (made by PCA, VQ-VAE). Why is such a comparison meaningful?

2. **Seeds for experiments:** For the simulation experiments, it is really not valid to make claims based on only 3 random seeds, especially given that the paper is mostly empirical in nature. 
 
3. Unless I am mistaken, residual RL is not a baseline for `w/ hand` in Table 1. Given that the authors propose a method that is essential residual RL+flow steering, I feel residual RL should be included for the `w/ hand` comparisons.

### Questions
Besides addressing my points in the weaknesses section, please make the following changes:
1. The notation for math in the paper is a bit cluttered. Specifically, please do not use the symbol $a$ whenever you write $a=a_r+a_b$, as we also have $\pi(a|s)$ (for example in Equation 4).
2. Grammatical mistakes:

a. In the abstract the sentence "Doing so allows policies to perform both local (residual actions) and
global exploration (latent noise), data-efficient adaptation." seems grammatically incorrect.

b. Please revise In table 1: absolute joint psose -> absolute joint pose.

### Soundness
3

### Presentation
2

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
The paper proposes Residual Flow Steering (RFS), an RL fine‑tuning scheme for flow‑matching / diffusion policies that jointly (i) steers the initial latent noise for global changes and (ii) applies a residual action for local corrections—without modifying base policy parameters. The method is instantiated for dexterous grasping: pretrain in simulation (Leap Hand + Franka), distill a visuomotor policy, then fine‑tune in the real world with offline RL (TD3+BC). Experiments show improved success over action‑representation baselines and ablations (residual‑only or steering‑only) in both pinch and power grasps.

### Strengths
RFS is a clean unification of residual RL (output modulation) and latent steering (input modulation), formalized via a modulation policy.

Strong, consistent improvements over baselines and over residual‑only / steering‑only ablations in simulation and better real‑world success vs. zero‑shot and supervised fine‑tuning.

The paper is well written. The introduction and the method are easy to follow and communicate the contributions and the implementation well.

### Weaknesses
Limited novelty. The algorithm mainly combines two widely used adaptation strategies (residual action learning and latent steering).

Baseline coverage. Real‑world evaluation lacks comparisons to other RL fine‑tuning approaches for diffusion/flow policies (e.g., recent flow‑RL fine‑tuners); most comparisons are ablations or action‑space baselines.

Task scope. Validation centers on grasping; the paper claims broader applicability, but no additional manipulation tasks are shown.

### Questions
Why is “changing the action representation” (absolute/relative joints, PCA, VQ‑VAE) considered a primary baseline for the first task? What does that comparison communicate about the contribution of the paper?

Could you add head‑to‑head comparisons against other RL fine‑tuning methods for diffusion/flow policies to strengthen the claim that RFS is preferable beyond its own ablations?

Do you have preliminary results (even in sim) on more complex tasks (e.g., non‑prehensile skills or multi‑stage manipulation) to support generalization claims?

### Soundness
3

### Presentation
3

### Contribution
2
