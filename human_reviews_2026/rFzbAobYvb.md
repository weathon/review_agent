# PB²: Preference Space Exploration via Population-Based Methods in Preference-Based Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Preference-based reinforcement learning (PbRL) has emerged as a promising ap-
proach for learning behaviors from human feedback without predefined reward
functions. However, current PbRL methods face a critical challenge in effectively
exploring the preference space, often converging prematurely to suboptimal policies
that satisfy only a narrow subset of human preferences. In this work, we identify
and address this preference exploration problem through population-based meth-
ods. We demonstrate that maintaining a diverse population of agents enables more
comprehensive exploration of the preference landscape compared to single-agent
approaches. Crucially, this diversity improves reward model learning by generating
preference queries with clearly distinguishable behaviors, a key factor in real-world
scenarios where humans must easily differentiate between options to provide mean-
ingful feedback. Our experiments reveal that current methods may fail by getting
stuck in local optima, requiring excessive feedback, or degrading significantly when
human evaluators make errors on similar trajectories, a realistic scenario often
overlooked by methods relying on perfect oracle teachers. Our population-based
approach demonstrates robust performance when teachers mislabel similar trajec-
tory segments and shows significantly enhanced preference exploration capabilities,
particularly in environments with complex reward landscapes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PB², a population-based preference-based reinforcement learning (PbRL) framework designed to enhance exploration in the preference space. By training a diverse population of policies guided by a performance-constrained diversity bonus, PB² enables the generation of more informative and distinguishable preference queries, leading to better reward model learning and improved robustness against noisy human feedback. Experimental results demonstrate superiority over several state-of-the-art PbRL methods across various benchmarks.

### Strengths
1. The manuscript is clearly written and well-structured, the idea is intuitive.

2. The paper provides an insightful analysis of exploration deficiencies in existing PbRL methods, particularly highlighting how ensemble-based and single-policy approaches fail to capture sufficient behavioral diversity.

### Weaknesses
1. The paper lacks a formal theoretical analysis to support the claim that the proposed method effectively mitigates exploration challenges or addresses uncertainty, limiting the depth of its empirical contributions.

2. The proposed approach introduces substantial computational overhead due to the maintenance and training of multiple policy networks and a discriminator, which may hinder scalability to more complex or real-time environments.

3. The method depends on several critical hyperparameters (e.g., the diversity coefficient $\lambda$ and performance threshold $\alpha$) that are not adaptively tuned. This reliance may require domain-specific calibration, potentially limiting the method’s generalizability and ease of deployment in practice.

4. The set of baseline comparisons, while representative, could be further expanded to include more recent or diverse PbRL methods, including RIME [1], direct alignment methods like CPL [2].

[1] Cheng J, Xiong G, Dai X, et al. Rime: Robust preference-based reinforcement learning with noisy preferences. ICML 2024.

[2] Hejna J, Rafailov R, Sikchi H, et al. Contrastive preference learning: learning from human feedback without rl. ICLR 2024.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
PB2 attempts to solve the core problem of preference space exploration. The authors point out that existing PbRL methods often suffer from premature policy convergence to suboptimal local optima. PB2 (Population-Based Preference-Based RL) is a population-based method. Its core idea is to explicitly maintain a population of agents with diverse behaviors, compelling the population to generate a series of high-return and behaviorally distinguishable trajectories to achieve efficient exploration.

### Strengths
• The paper accurately identifies a critical flaw in existing PbRL methods: query similarity. This is a very significant problem in real-world applications.    
• The introduction of a similarity threshold, $\epsilon$, to simulate human inconsistency when facing similar trajectories is highly practical. The experiments demonstrate that PB2's robustness in high-noise (high $\epsilon$) environments far exceeds that of baseline methods.    
• The experiments in the maze environment are very intuitive, showing how PB2 leverages its population diversity to successfully explore and find the globally optimal path.

### Weaknesses
• Compared to single-agent methods like QPA, PB2's implementation complexity and the number of hyperparameters are significantly increased. It requires maintaining N policy networks, N corresponding Q-value networks, and an additional discriminator network $q_{\psi}$. This leads to higher computational and memory overhead.        
• In the grand scheme, PB2 can be seen as a diversity-based exploration strategy. I am convinced that it performs better in toy examples. However, in the DMC experiments, the variance and mean of most methods have severe overlap, making it difficult to discern a clear performance advantage for PB2.         
• Appendix A.3 mentions several "tricks" to "ensure stable and effective diversity guidance." However, traditional methods like PEBBLE and QPA already perform stably without much tuning, suggesting that this added complex design exacerbates training instability.

### Questions
1. I am curious, compared to PEBBLE and RUNE, how would PB2 perform if it were used only as an auxiliary exploration strategy for them?     
2. Does the PB2 mechanism allow me to drastically improve exploration efficiency and final performance by simply increasing the population size (e.g., to N=20)? Or would the discriminator's training become the new bottleneck at that point? Furthermore, in environments like DMC, is exploration itself truly the bottleneck?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
PB² proposes a population-based method for PbRL to address insufficient exploration of the preference space and query ambiguity. The approach trains an anchor policy to exploit the current learned reward and multiple diverse policies encouraged via a discriminator-based mutual-information bonus under a performance constraint, yielding more distinguishable trajectories for preference queries. Experiments across navigation and DMControl tasks show improved feedback efficiency, greater robustness to noisy or inconsistent human labels, and the ability to escape local optima compared to single-agent baselines and a naïve posterior-sampling ensemble.

### Strengths
Maintaining behavioral diversity during the query stage can markedly increase the discriminability and stability of human comparisons, thus learning rewards more effectively and reducing feedback needs. The central problem addressed by this paper is indeed important.

The discriminator’s mutual-information objective directly optimizes for “distinguishable behaviors”, which aligns with the goal of making human comparisons easier and provides a principled mechanism for boosting information content and robustness. The design is reasonable and clear.

### Weaknesses
1. As acknowledged by the authors, the computationalcost of the design is significant, which may limit practical usability.  
2. Other relevant approaches deserve discussion. For example, PPE [1] manages data/distribution-side coverage and evaluation reliability; and PPE appears to report stronger results than the present method. The authors should provide fuller theoretical analysis and empirical comparisons. The SENIOR paper [2] is also highly relevant.  
3. The mathematical presentation lacks clarity and rigor in several places, with notable inconsistencies that should be carefully checked, including but not limited to:  
   a) **Algorithm 1 vs. main equations**: Algorithm 1 Line 12 uses  $ r_\phi(\tau) + \lambda \cdot q_\psi(i), $   whereas the main method and Eqs. (3)/(4) use the **log-probability** conditioned on trajectories/states, e.g.,   $ \log q_\psi(i \mid \tau). $ 
   b) **Granularity of \(q_\psi\) inputs**: Algorithm 1 does not specify whether \(q_\psi\) takes entire trajectories \(\tau\) or single states \(s\); Algorithm 2 uses a state-based information-gain form,   $ \log q_\psi(i \mid s) - \log p(i). $   The paper should unify the convention in the main text and annotate Eq. (4) accordingly.  
4. There are typos and minor writing issues: “one of the main claim” → “one of the main claims”; “We also the naive implementation …” (incomplete sentence); “collect diverse trajectory” → “collect diverse trajectories.” These should be carefully corrected.  
5. Stronger Bayesian query selection/uncertainty-modeling baselines could and should be compared against the proposed method.


The method’s performance advantages appear limited relative to its higher computational cost, which raises concerns about practical utility. The paper would benefit from more comprehensive validation to demonstrate effectiveness. In parallel, the manuscript requires careful, rigorous proofreading to resolve notational inconsistencies and typographical issues.


[1] Zhu, Y., ... . (2024). Optimizing reward models with proximal policy exploration in preference-based reinforcement learning. In NeurIPS 2024 Workshop on Behavioral Machine Learning. 

[2] Ni, H., ... . (2025). SENIOR: Efficient query selection and preference-guided exploration in preference-based reinforcement learning. arXiv preprint arXiv:2506.14648.

### Questions
See above.

### Soundness
3

### Presentation
2

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
This paper introduces PB², a novel population-based framework for preference-based reinforcement learning (PbRL) aimed at addressing the lack of behavioral diversity during user feedback collection. Traditional single-policy PbRL methods often converge to local minima in the preference space, limiting exploration and leading to suboptimal alignment with human preferences.

PB² tackles this by training multiple distinct policies simultaneously, each encouraged to explore different behavioral modes through an explicit diversity bonus. These diverse policies generate varied trajectories, which are then compared using human preference feedback to train a shared reward model. A discriminator module maintains population diversity while ensuring that learned behaviors remain consistent with user preferences.

Experimental results across DMControl locomotion and navigation tasks show that PB² produces more diverse and distinguishable behaviors, improves reward learning efficiency, and remains robust under noisy or inconsistent feedback. Additionally, the paper reveals that neural ensemble models fail to capture preference uncertainty effectively, offering little improvement over deterministic baselines.

### Strengths
1. The idea of separating a reference policy and a diverse policy to serve as different purposes is insightful and reasonably novel in preference-based RL literature. 
2. The use of discriminator to differentiate between anchor and diverse incorporates the idea of adversarial learning into preference-based RL algorithms.
3. Experimental results demonstrate performance gains in both sample and feedback efficiency in different types of teachers, which are more aligned with real world scenarios.

### Weaknesses
1. Although the use of discriminator is different from previous work that only focuses on single-agent preference-based RL algorithms, are there implementation and experimental challenges as a result?

### Questions
1. Will the policy used for evaluation the same as the reference policy $\pi_{\text{ref}}$? Is it possible to somehow combine reference policy and diverse policy to see what are resulting behaviors, since exploration may also benefit a task-specific policy?
2. It seems that choice of $\alpha$ is based on heuristic, i.e. ideally the diversity-based exploration should contain meaningful behavior at least some portion of anchor policy. How sensitive is performance with different values of $\alpha$?
3. Is the discriminator trained to maximize mutual information between all trajectories collected by the diverse policy? Or to distinguish between the anchor policy and the remaining from diverse policy?

### Soundness
3

### Presentation
3

### Contribution
2
