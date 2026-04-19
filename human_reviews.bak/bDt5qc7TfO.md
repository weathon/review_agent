# Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
In safe offline reinforcement learning, the objective is to develop a policy that maximizes cumulative rewards while strictly adhering to safety constraints, utilizing only offline data. Traditional methods often face difficulties in balancing these constraints, leading to either diminished performance or increased safety risks. We address these issues with a novel approach that begins by learning a conservatively safe policy through the use of Conditional Variational Autoencoders, which model the latent safety constraints. Subsequently, we frame this as a Constrained Reward-Return Maximization problem, wherein the policy aims to optimize rewards while complying with the inferred latent safety constraints. This is achieved by training an encoder with a reward-Advantage Weighted Regression objective within the latent constraint space. Our methodology is supported by theoretical analysis, including bounds on policy performance and sample complexity. Extensive empirical evaluation on benchmark datasets, including challenging autonomous driving scenarios, demonstrates that our approach not only maintains safety compliance but also excels in cumulative reward optimization, surpassing existing methods. Additional visualizations provide further insights into the effectiveness and underlying mechanisms of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper develops a novel approach to safe offline reinforcement learning; a task that aims to maximize rewards and minimise safety constraint violations during training on an offline dataset. The proposed methodology leverages conditional variational autoencoders (CVAEs) to model latent safety constraints, effectively learning safety bounds. The proposed methodology can then be encompassed into two further methodologies that the authors evaluate upon: LSCP-S which prioritizes selecting the safest actions, and LSPC-O, which aims to maximize the reward within these latent safety bounds. In the evaluation, it is seen that LSCP-S effectively reduces cost and LSCP-O manages a nice balance between reducing cost and maximising reward.

### Strengths
- A promising approach to capture complex, non-linear safety constraints. Results clearly back this up and I was impressed with the methodology and theoretical analysis.
- The literature review is comprehensive and up-to-date.
- The broad range of benchmarks that the methodology is evaluated upon is robust and impressive.
- Authors are transparent about the methodology's limitations and future work, showing openness and foresight.
- Figure 3, in particular, is visually engaging and provocative for the reader. It enables further transparency into the proposed method and I really appreciated it.
- Code is available and open.

### Weaknesses
- There is little to no discussion, apart from minor comments in the related work section, regarding comparative methods and hyperparameters which strongly hinders the assessment of Table 1. For example, in the metadrive experiments, CPQ doesn't seem to learn anything. More detail into this method or the hyperparams used at the start of the the evaluation section could have provided me with further insight. Currently, I am left thinking that the implementation wasn't tuned correctly. Also, there is no standard RL baseline against which to assess all methods.
- An analysis point, I would like/liked to have seen a discussion on surrounds the comparison between LSPC-S and LSPC-O. In Table 1, there seems to be a disproportionate increase in cost between the -S and -O models. I would have expected, within reasonable deviation, to have seen a reasonably proportionate increase in reward to increase in cost between the -S and -O models. But, in the hard metadrive experiments, for example, the reward increased +0.03 on average whilst the cost increased +0.24 on average. I am not concerned at all about this, I would just like to see some discussion surrounding this interesting trade-off that the -O model has to try to balance.
- Another analysis point, which slightly draws parallels from my previous point, is in Figure 2 it seems that LSCP-O observes large oscillations in terms of cost in comparison to other methods. The small increase in reward is seen from the -S to the -O model but a larger cost is also observed alongside some stronger oscillation. Could this oscillation be due to the model trying to find a trade-off balance or is this some hyperparam tuning that is needed?
- Computational/Time Taken/Machine experimented on details are needed in the appendix. Without these, I personally would draw the conclusion that the proposed methods are computationally more intensive than comparative methods. Personally, I do not computation as a major weakness, but I would appreciate details about it to be transparent.

Minor weaknesses
======================
- Figures 2 and 3 have very small text and need to be increased. The legends are also in a poor place, personally, and could be better placed above/outside the axis.
- Tables could also benefit from rows having alternating highlights (white to light gray) to make them easier to read.
- Please put boldface and blue colour reasoning in the table caption for readability.

### Questions
Can the authors comment on the following:
- The disproportionate increase in rewards to the large increase in costs from the -S to -O models. A point that is then observed again in Figure 2.
- Are the oscillations observed in Figure 2 a minor weakness with the method or simple hyperparam tuning that is needed?
- Can you add experiment insights (computation/time taken/machine experimented on) into the appendix to improve transparency?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Traditional RL learns optimal policies through interactions with the environment, but this approach can pose safety risks during the learning process. In industrial settings such as autonomous driving and robotics, even minor errors can lead to significant costs or safety hazards. To address these challenges, offline reinforcement learning has been introduced. However, offline RL, which relies solely on fixed datasets, risks generating unsafe policies when encountering new situations. This study proposes a new framework called Latent Safety-Prioritized Constraints (LSPC), designed to learn offline policies that optimize rewards while adhering to safety constraints, demonstrating its effectiveness across various high-risk tasks.

### Strengths
1.  Balance of safety and performance: Designed to optimize rewards while satisfying safety constraints.
2.  Efficiency of offline learning: Learns policies without further interactions with the environment by using fixed datasets.
3.  Theoretical stability guarantee: Offers theoretical bounds on policy performance and sample complexity.
4.  Applicability in various environments: Demonstrates strong performance in complex tasks, including autonomous driving and robotic manipulation.

### Weaknesses
1.  Difficulty in latent space configuration: Finding the optimal latent space configuration requires empirical tuning.
2.  Dependency on dataset quality: Safe policy learning heavily relies on the quality of the dataset.
3.  Limitations in real-time application: Offline RL struggles to respond immediately to dynamic changes in real-time environments.

### Questions
## Theoretical Concerns

1.  Setting safety constraints in the latent space
    
    LSPC uses a conditional variational autoencoder (CVAE) to learn safety constraints in the latent space. However, it is unclear whether the latent space can fully capture real-world safety constraints. If the safety constraints are not properly learned, the model may fail to ensure safety in unforeseen situations.
    
2.  Reliability of constrained optimization
    
    LSPC performs constrained reward optimization to maintain a balance between rewards and safety. However, it raises concerns about whether the optimization in the latent space will remain valid for actions not encountered during training. The paper lacks guarantees that the policy will behave appropriately for out-of-distribution state-action pairs.
    
3.  KL divergence-based safety assurance
    
    While safety is ensured by minimizing the Kullback-Leibler (KL) divergence between the learned policy and behavior policy, subtle behavioral differences, even with small divergences, could lead to significant risks in complex environments. Theoretical minimization of KL divergence may not fully capture all possible real-world scenarios.
    

----------

## Experimental Concerns

1.  Gap between simulation and real-world environments
    
    The paper evaluates performance primarily in simulated environments. However, simulations may not fully account for real-world uncertainties, such as noise, sensor errors, or dynamic changes. This gap poses a risk that LSPC’s safety and performance may fall short in real-world applications.
    
2.  Dataset limitations and bias
    
    Since LSPC relies on static datasets, there is a risk that the policy will not perform well in situations not included in the dataset. Furthermore, if the training data is biased toward specific patterns, the policy may make incorrect decisions.
    
3.  Interpretation of safety-performance trade-offs
    
    While the paper claims to maintain a balance between safety and rewards, it may not have thoroughly analyzed cases where this balance breaks down. For example, certain tasks may allow minor safety violations to achieve higher rewards, but the paper lacks a clear discussion of these scenarios.
    
4.  Stability of optimization
    
    The complex optimization process involving CVAE and implicit Q-learning can lead to convergence issues, resulting in unstable learning or local optima. The paper does not provide adequate solutions to address these potential issues.
    

----------

## Comprehensive Concerns

-   Are the safety constraints in the latent space sufficient? It remains unclear whether CVAE can effectively capture all necessary safety constraints.
-   Lack of generalization to real-world scenarios: Reliance on static datasets raises concerns about performance degradation in new situations.
-   Trade-offs between safety and performance: While the framework claims to balance safety and rewards, it may fail to guarantee absolute safety in specific scenarios.
-   Complexity of the optimization process: Additional analysis is required to address potential convergence issues and instability during learning.

These concerns highlight important considerations for evaluating the applicability and reliability of LSPC. Future research should focus on validating its generalization performance in real-world environments and ensuring that the balance between safety and rewards is consistently maintained across various scenarios.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel Latent Safety-Prioritized Constraints (LSPC) framework aimed at addressing the balance between reward maximization and safety constraints in safe offline reinforcement learning. The authors model potential safety constraints using Conditional Variational Autoencoders (CVAE) and demonstrate experimental results across multiple benchmark tasks, indicating that the proposed method effectively enhances cumulative rewards while ensuring safety.

### Strengths
Originality:

The LSPC framework offers a fresh perspective on managing the trade-off between safety and rewards, particularly within the context of offline reinforcement learning, demonstrating significant practical and theoretical value.

Significance:

The paper provides a detailed and persuasive theoretical analysis of policy performance, ensuring the scientific foundation of the proposed method.

Quality:

Through systematic experimentation across various benchmark tasks, the effectiveness of LSPC is showcased, with significant advantages over existing methods.

### Weaknesses
Please refer to questions

### Questions
- Although the experiments primarily focus on specific tasks, there is a lack of in-depth discussion on the application and effectiveness of the LSPC framework in multi-task learning. How will you address the variations in safety constraints and reward structures across different tasks?

- When comparing with several benchmark methods, the analysis does not adequately demonstrate the specific advantages of LSPC across different tasks. Is it necessary to provide a more detailed discussion of the comparison methods, especially regarding their performance under different safety constraints?

- When using Conditional Variational Autoencoders (CVAE) to construct a latent safe space, how do you ensure that the generated samples comprehensively cover all possible safety constraints in the environment? Is it necessary to validate the effectiveness and sufficiency of this coverage?

- Decoupling reward maximization and safety constraints may lead to failures in safety assurance. How does the decoupled strategy effectively guarantee safety under different safety constraint conditions?

- The thoretical analysisi is dense, it's better to provide more context and examples to help readers understand the impact and theoretical basis of this assumption.

### Soundness
3

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
This paper aims to solve the offline safe RL. To balance safety constraints and distribution shifts in offline safe RL, they use the CVAE to model the policy. Since the CVAE is trained on the offline dataset, this paper expects the CVAE architecture to achieve a balance between safety constraints and distribution shifts. To ensure the CVAE achieves such balance, they propose two methods to use the CVAE. The first method, LSPC-S, penalizes the high-cost state based on the cost function learned by IQL. The second method,  LSPC-O, focuses on reward-maximizing under the CAVE-based policy. This paper demonstrates the effectiveness of their methods through empirical evaluation and theoretical analysis.

### Strengths
- The authors provide a theoretical analysis of their method.
- The proposed method outperforms baselines in safe RL tasks.

### Weaknesses
- The method proposed by this paper cannot strictly keep safe constraints. Since a state-action pair is safe iff $ V_c^\pi(s)\leq \kappa $. Equation 10 only serves as a regularizer to penalize the high-cost actions and cannot guarantee $ V_c^\pi(s)\leq \kappa $.
- Theory insights are one of the contributions of this work. However, the authors do not derive an exact sample complexity; they only claim that the sample complexity for obtaining an $ \epsilon $-optimal policy is about $ \mathcal{O}(1/\epsilon^4) $and put the sample complexity results in the appendix.
- Lack of ablation study.

### Questions
1. Why choose CVAE? Many other generative models, such as the flow and diffusion models, have been shown to outperform VAEs in many fields. Could other generative models achieve better results? 

2. Regarding Lemma 2. Theorem 2 in Cen 2024 [1] is bounded by $ \frac{2R_{max}}{(1-\gamma)}D_{TV}[d^D(s)||d^*(s)]+e_N $. I am unable to understand why Lemma 2 is tighter than Cen 2024 since the relationship between $ D_{TV}[d^D(s)||d^*(s)]+e_N $and $ E_{s\sim d^{\pi^*}}[D_{TV}(\pi|\pi^*)] $is unclear. And it's surprising that the upper bound for safe offline RL is tighter than that for offline RL, as offline Safe RL is a more challenging task. 

3. I noticed that this paper is similar to [2], as both use CVAE to model policies. What is the difference between the two?

References:

[1] Cen, Zhepeng, et al. Learning from Sparse Offline Datasets via Conservative Density Estimation. 

[2] Xu, Haoran, Xianyuan Zhan, and Xiangyu Zhu. "Constraints penalized q-learning for safe offline reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.

### Soundness
2

### Presentation
2

### Contribution
3
