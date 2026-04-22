# Regularized Latent Dynamics Prediction is a Strong Baseline For Behavioral Foundation Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Behavioral Foundation Models (BFMs) have been recently successful in producing agents with the capabilities to adapt to any unknown reward or task. In reality, these methods are only able to produce near-optimal policies for the reward functions that are in the span of some pre-existing _state features_. Naturally, their efficiency relies heavily on the choice of state features that they use. As a result, these BFMs have used a wide variety of complex objectives, often sensitive to environment coverage, to train task spanning features with different inductive properties. With this work, our aim is to examine the question: are these complex representation learning objectives necessary for zero-shot RL? Specifically, we revisit the objective of self-supervised next-state prediction in latent space for state feature learning, but observe that such an objective alone is prone to increasing state-feature similarity, and subsequently reducing span of reward functions that we can represent optimal policies for. We propose an approach, RLDP, that adds a simple regularization to maintain feature diversity and can match or surpass state-of-the-art complex representation learning methods for zero-shot RL. Furthermore, we demonstrate the prior approaches diverge in low-coverage scenarios where RLDP still succeeds.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to use regularized latent dynamics prediction as an objective to learn representations for behavior foundation models. The main idea is to learn a state representation using a latent dynamics prediction objective, where the encoded latent states $\phi(s_t)$ are trained to regress the unrolled predicted latent states $h_t := g(h_{t-1}), a_t, h_0 = \phi(s_0)$. This objective alone suffers from collapse (shown in Figure 1), so the paper proposes to use an orthogonal regularization to mitigate it. The state representation is then used as the backward feature in the Forward-Backward (FB) representation framework. The paper shows theoretically that optimizing the latent dynamics objective also reduces prediction error in successor measures (i.e. the FB objective). Empirically, they evaluate the method on the offline zero-shot RL benchmark ExoRL, where the method matches or outperforms other behavior foundation model baselines. Their method also achieves online zero-shot RL in a Humanoid environment and is capable of learning representations from low-coverage datasets. The key design choice of orthogonal regularization is justified by ablation experiments. Overall, the paper proposes a simple yet effective method for zero-shot RL.

### Strengths
1. The proposed method is simple yet effective, performing favorably to baselines across multiple settings while suffering less from feature collapse.
2. The theoretical justification is sound.
3. The paper conducts thorough analysis on the feature collapse phenomenon and the effect of orthonogal regularization.

### Weaknesses
1. The theory and practice are somewhat mismatched. Theory says that learning successor measure on top of the latent representations improves as the representation learning objective improves. However, in practice, the forward feature is learned from scratch, and only the backward feature is carried over. 
2. The method does not improve much compared to baselines in the ExoRL benchmark.

### Questions
1. Can you explain the mismatch between theory and practice as outlined in Weakness 1?
2. Can you try an experiment where you learn a Universal Successor Feature on top of the state features, and do zero-shot RL with linear regression (assuming the features linearly span rewards)? This can potentially substantiate the broad applicability of the learned representations.

References:

[1] Diana Borsa, André Barreto, John Quan, Daniel Mankowitz, Rémi Munos, Hado van Hasselt, David Silver, Tom Schaul. Universal Successor Features Approximators. ICLR 2019.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work presents Regularized Latent Dynamics Prediction (RLDP), a representation learning approach for zero-shot reinforcement learning that replaces complex, reward-dependent objectives with a simple, self-supervised latent dynamics prediction task. The method augments next-state prediction with an orthogonality regularization term that mitigates feature similarity and prevents collapse in the learned state representations. Empirical evaluations across multiple benchmarks—including DeepMind Control Suite, D4RL, and high-dimensional humanoid control—show that RLDP achieves performance comparable to or exceeding state-of-the-art Behavioral Foundation Model baselines such as FB, PSM, and HILP. In particular, RLDP maintains stable performance in low-coverage datasets.

### Strengths
The paper makes a simple yet effective case for introducing latent dynamics into zero-shot offline RL. 

RLDP achieves performance on par with or exceeding that of more complex methods, demonstrating that simple latent dynamics prediction can yield strong generalization.

Policy-independent formulation avoids instability from Bellman backups, leading to more reliable learning in challenging or low-coverage environments.

### Weaknesses
Evaluation is performed on simulated continuous control benchmarks. Real-world data or transfer is not evaluated.

Empirical results show comparable but not universally superior performance, suggesting that the method’s advantages depend on task characteristics and data diversity.

### Questions
What challenges do the authors suspect when adapting RLDP to real-world domains?

How would the approach perform in settings where the dynamics themselves evolve or where the environment exhibits non-stationarity—common in human-in-the-loop or adaptive control scenarios?

### Soundness
4

### Presentation
3

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
This paper introduces **RLDP (Regularized Latent Dynamics Prediction)** as a pretraining objective for zero-shot reinforcement learning.  
The main contributions include:

1. Proposing RLDP, a method to pretrain state representations for zero-shot RL via latent next-state prediction with regularization.  
2. Theoretically proving that training in the latent space can approximate training in the original state space, preserving the predictive power of successor measures.  
3. Conducting extensive experiments under both online and offline settings, showing that RLDP achieves competitive performance.  
4. Demonstrating the advantage of RLDP under low-coverage conditions, where other methods tend to fail.

### Strengths
1. Comprehensive experimental settings, including both online and offline environments.  
2. Theoretical guarantees that connect the latent training objective to successor-measure consistency.  
3. Rich intuition and ablation studies that help interpret how RLDP improves representation quality.

### Weaknesses
There are no major flaws, but several aspects could be improved for clarity and consistency.  

1. Structure and exposition could be better organized:
   - The paper does not clearly specify which parameters are optimized for each loss.  
     The training process is only briefly introduced around line 295, and it’s unclear which components receive gradients.  
     A concise summary of the full optimization pipeline (e.g., which modules update under each loss) would make the method section more readable.  
   - In lines 180–190 describing policy improvement, it is unclear whether a separate policy network is maintained.  
     In Equation (4), π(s) is written as if it equals a Q-value—perhaps “max” should be “argmax”?  
     Equation (5) is understandable but would be more precise if an expectation operator **E[…]** were added.  
   - The variable $\bar{M}$ is first defined in *Lemma 4.3* but appears earlier in Equation (3).  
     This may indicate an input mismatch: it should likely take $\Phi(s′)$ instead of $s′$ as argument.

2. The claim in line 198—“these representations minimize the prediction error for successor measures for any policy”—is not directly demonstrated.  
   Subsequent quantitative analyses focus on returns rather than direct prediction errors.  
   Since this is primarily a representation learning method, showing explicit results for prediction errors (e.g., L2 distance between predicted and ground-truth successor measures) would greatly strengthen the argument.

### Questions
1. In Figure 1, the paper mentions a “mild form of collapse.”  
   However, the cosine similarity range of 0.4–0.6 does not obviously indicate collapse. Why is this value considered high?  Providing reference baselines (e.g., expected cosine similarity for random embeddings) would make this claim more convincing.

2. The paper evaluates RLDP on D4RL’s medium and medium-expert datasets.  
   Why not also include medium-replay? Those datasets have broader coverage, which could provide further insight into RLDP’s robustness under high-variance data.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel representation learning objective for unsupervised RL, where rewards are not given in the data, focused on enabling downstream Behavioral Foundation Models to perform effective zero-shot RL. Their objective broadly encompasses two components: a next latent state prediction loss motivated by theory, and an orthogonal regularization loss motivated through empirical observation. Through a series of experiments in robotics simulators, they validate their method, and show that their simple framework is a strong baseline for learning BFMs in a variety of settings including offline zero-shot RL, online zero-shot RL, and low coverage datsets.

### Strengths
- Their proposed method is well-motivated both theoretically and empirically, and the authors present strong experimental evidence for their framework relative to other works in its area.
- The experiments in their paper suggests that their proposed method can produce competitive results in the zero-shot RL setting, all the while being both simple to understand and implement compared to existing methods. Thus, this work is an important entry point for new researchers in the field of BFMs for zero-shot RL, and an important reference for existing researchers as well.
- Their method is further studied beyond final performances, as the authors provide ablation studies for almost all components of their framework in either the main text or the supplementary material.

### Weaknesses
- It would be nice to further contextualize RLDP with the existing representation learning methods, by either providing the exact objectives of the baselines used in the paper(Laplace, FB, HILP, PSM) or being more explicit about the different assumptions used. For instance, the authors also suggest that their method is simpler in implementation and lacks some assumptions made by prior methods such as a prior class of policies. Showing the objectives used in the other works would clarify these differences

### Questions
- Aside from ease of implementation, what other benefits arise from the simplicity of the proposed methods? Can we also any significant improvements in compute costs or sample efficiency?
- As remarked in [1], next latent state prediction is generally a strong objective for representation learning in RL. However, in some tasks, other objectives such as observation reconstruction can also be beneficial. Due to the similar nature of all the environments in the paper (robotics physics-based simulators), it is possible that the proposed objective is only a strong baseline in environments of that class. For example, [1] remarks that while observation reconstruction is not beneficial in MuJoCo tasks, it can be good in low-dimensional clean environments such as grid worlds. Have you considered the benefit of other simple and popular representation learning objectives, and possibly experimenting with other environments such as pixel-based environments?


### Minor Comments
- Typo in line 287: "Lemma 4.33 implies that minimizes".
- Typo in line 370: "Due to **the** exploratory challenge [...]"
- Typo in line 412: "a class of **policies** to learn [...]"
- The text is not consistent with its paragraph titles. For example, paragraph titles are sometimes bolded with a period (e.g. lines 320, 347, 353, 368 ...), bolded with a colon (e.g. lines 385, 423, 431), or italicized (e.g. lines 441 and 448).
- The text is not consistent with line breaks between paragraphs. There looks to be missing line breaks in line 368, 375, 385, 423, 430, 440. 
- Figure 3 does not show any variance statistics for the "average return" datapoints. It can be difficult to parse whether the stated improvement from the orthogonality regularization is statistically significant.
- It could be benficial to further contextualize the results by adding one or two naive baselines that are not state-of-the-art. For example, what is the performance of zero-shot RL on a random representations?
- The results in Figure 2 are difficult to parse both in detail and at a glance. It could be beneficial to aggregate these results by normalizing the scores and providing an IQM [2]


### References
[1] Ni et al., 2023 -  https://arxiv.org/pdf/2401.08898
[2] Agarwal et al., 2022 - https://arxiv.org/pdf/2108.13264

### Soundness
4

### Presentation
3

### Contribution
3
