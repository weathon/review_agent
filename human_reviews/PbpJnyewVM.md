# Zero-shot Cross-task Preference Alignment for Offline RL via Optimal Transport

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
In preference-based Reinforcement Learning (PbRL), aligning rewards with human intentions often necessitates a substantial volume of human-provided labels. Furthermore, the expensive preference data from prior tasks often lacks reusability for subsequent tasks, resulting in repetitive labeling for each new task. In this paper, we propose a novel zero-shot cross-task preference-based RL algorithm that leverages labeled preference data from source tasks to infer labels for target tasks, eliminating the requirement for human queries. Our approach utilizes Gromov-Wasserstein distance to align trajectory distributions between source and target tasks. The solved optimal transport matrix serves as a correspondence between trajectories of two tasks, making it possible to identify corresponding trajectory pairs between tasks and transfer the preference labels. However, direct learning from these inferred labels might introduce noisy or inaccurate reward functions. To this end, we introduce Robust Preference Transformer, which considers both reward mean and uncertainty by modeling rewards as Gaussian distributions. Through extensive empirical validation on robotic manipulation tasks from Meta-World and Robomimic, our approach exhibits strong capabilities of transferring preferences between tasks in a zero-shot way and learns reward functions from noisy labels robustly. Notably, our approach significantly surpasses existing methods in limited-data scenarios. The videos of our method are available on the website: https://sites.google.com/view/pot-rpt.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Preference Optimal Transport (POT), which aims to establish a correspondence between the same human preference in different reinforcement learning tasks. It solves a question: if a human user prefers trajectory x in the source domain, what trajectory y would the same user prefer in the target domain? Specifically, POT aligns two sets of trajectories by solving an optimal transport matrix under Gromov-Wasserstein distance. POT also incorporates uncertainty in human preferences by using distributional rewards.

### Strengths
1. Clear identification of the problem (user preference transfer between RL task domains)
2. Clear presentation of the POT algorithm.

### Weaknesses
In Table 1, we can see that scripted labels are in general providing better results than transferred labels, which is expected. Therefore, transferred labels should only be used as a substitute when scripted labels are expensive. The paper has yet to discuss applicable scenarios of using transferred labels - when should we compromise success rate for cheaper labels?

### Questions
Please respond to the weakness above. If is addressed appropriately, I am willing to improve my rating.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces an algorithm for zero-shot preference-based RL, aiming to address the human preference-guided RL challenge in a transfer learning context. Specifically, the authors utilize optimal transport theory to transfer human preference labels from one RL task to another. This transfer procedure involves computing a coupling matrix using the Gromov-Wasserstein distance, and this matrix yields the transferred preference labels. Additionally, the reward function is modeled as a distribution, facilitating learning from noisy labels and is refined using a process that employs Gaussian distributions instead of scalar rewards. The proposed approach is assessed on robot manipulation tasks in the Meta-World and Robomimic environments. Comparative experiments against baselines and thorough ablation studies further validate the design decisions presented in the paper.

### Strengths
- Overall, this paper is well-written, and the proposed idea is presented comprehensively.
- The method proposed uses the Gromov-Wasserstein distance to learn a coupling and subsequently transfers the preference label. This approach allows for the creation of a reward function for a target task using preference labels from source tasks, without the need for preference labels specific to the target task.
- The authors have enhanced the robustness of the RL objective by introducing Gaussian noise.
- Comprehensive experimental results are provided to validate the proposed method.

### Weaknesses
- The contribution could be the weakness of this work. It seems that the proposed method is rather incremental since the authors incorporate the original preference-based reinforcement with off-the-shelf Gromov-Wasserstein distance method. In addition, the uncertainty module also utilizes the existing reparameterization trick. Since the Gromov-Wasserstein distance is a concrete way to measure the difference between different distributions, is it possible for the authors can provide generalization bounds (even in the toy example)? 
- Some RL tasks can be difficult to generalize. The authors haven't clarified how their optimal transport technique might be effective across such different scenarios. 
- The authors may better justify the usage of preference in RL settings. The simulation environments used in this work are not designed to validate human preference, they are more suitable for goal-conditioned RL. The RL task displayed in this work are reply on the end-effector of the robot arm. The authors may want to clarify what is the underlying benefit of transferring human performance.
- It seems that the total number of trajectories is 4, and the authors use a K-means clustering to separate them further. I am wondering about the necessity of doing this. Are you labeling the preference among clusters or within clusters?

### Questions
- Task Similarity: It appears that different tasks represent variations of the same RL task, with only differing goals. This seems a restrictive setting. Can the authors discuss the method's versatility?

- Preference Matrix Clarification: Including the original preference matrix would enhance comprehension. Consider the source samples matrix:
​[[/, 1, 0], [0, /, 0], [1, 1, /]], and using this and the coupling matrix: [[1/6, 1/6, 0], [1/6, 1/6, 0], [0, 0, 1/3]]. Could the authors detail the preference transfer?

- Preference Transfer Properties: What are the inherent properties of the proposed method? A deeper analysis using the aforementioned matrices could shed light on this.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In Preference-based Reinforcement Learning, matching rewards with human intentions typically demands a significant amount of labels provided by humans. Moreover, the costly preference data from previous tasks often isn't reusable for future tasks, leading to repeated labeling processes. This paper introduces a zero-shot cross-task preference-based RL method that employs preference labels from labeled data to deduce labels for other data, thus bypassing the need for additional human input. The authors employ the Gromov-Wasserstein distance to map trajectory distributions across tasks. Yet, relying solely on these transferred labels could lead to potentially imprecise reward functions. Addressing this, we present the Robust Preference Transformer. It estimates both the average and variance of rewards by representing them as Gaussian distributions. The proposed methodology, when tested on Meta-World and Robomimic, demonstrates superior performance compared with baselines.

### Strengths
The paper is well-written and easy to follow.

### Weaknesses
I don't feel the simple addition of labels can work. Let me provide a toy example. If the authors pointed out my error, I would be glad to change my score.

Assume the pair-wise labels of source $\{x_1, x_2, x_3\}$ is $Z_s = \[\[/, 0, 0\], \[1, /, 1\], \[1, 0, /\]\]$.

Let's assume the target is exactly the same as the source, i.e., $\{y_1 = x_1, y_2 = x_2, y_3 = x_3\}$. Then the transportation matrix is $\boldsymbol T = \[\[1/3, 0, 0\], \[0, 1/3, 0\], \[0, 0, 1/3\]\]$

Let's say we want to compute the label of $(y_2, y_3)$

$\boldsymbol A^{23} = \[\[0, 0, 0\], \[0, 0, 1/9\], \[0, 0, 0\]\] $

$z(y_2, y_3)=\frac{1}{9}z(x_2, x_3)=1/9$, while we know the ground truth is $z(y_2, y_3)=z(x_2, x_3)=1$.

In addition, some notations are ambiguous, e.g., $\mathcal S$ for state space and source task at the same time.

### Questions
* In Figure 3b, why is RPT+POT even better than Oracle PT? If it is due to the variance, error bars should be provided in all figures.

### Soundness
1 poor

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
This paper first introduces a zero-shot cross-task transfer algorithm Preference Optimal Transport (POT) for preference-based offline reinforcement learning. The trajectories between the source and target tasks are aligned via the optimal transport method and generates pseudo preference labels based on the alignment matrix. Additionally, the paper introduces the Robust Preference Transformer (RPT) to model the uncertainty of preference labels, enabling robust learning in the presence of transfer noise. Experimental results demonstrate that the proposed algorithm exhibits significant advantages in both zero-shot and few-shot preference.

### Strengths
1. The paper addresses a highly meaningful scenario that could have a positive impact in practical applications. 

2. The writing logic of the paper is very clear, and it is almost devoid of difficulty in understanding.

### Weaknesses
1. There are issues with the problem formulation in the paper. The assumption of identical action space alone is not sufficient to guarantee the alignment of trajectories and preference labels between source and target tasks. The fundamental reasons for the success of Preference Optimal Transport (POT) are not adequately explained.
2. Transferring the preference labels from the source task to the target task undoubtedly involves negative noise or uncertainty. This problem becomes even more severe in the context of zero-shot learning, where there is no corrective information available. While the paper acknowledges modeling uncertainty as variance, it doesn't eliminate this negative impact on the downstream tasks, which is unreasonable. On the contrary, the success of the target task appears to depend on such uncertainty since RPT+POT > PT+POT.

### Questions
1. The paper proposes that uncertainty should approach a predefined value $\mu$ during training. What will happen if the uncertainty is directly set to $\mu$ without training?
2. Unclear expression：$\mathcal{S}$ represents state space and source task simultaneously and the expression of $\mathcal{T}$ is also a little confusing in **problem setting**; PT+Dis and PT+Sim in experiments.
3. What is value of $u_i$ and $v_j$ and does the calculation of $A$ need the value of $u_i,v_j$?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
