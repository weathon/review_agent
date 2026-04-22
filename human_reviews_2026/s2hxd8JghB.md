# Bayesian Ensemble for Sequential Decision-Making

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Ensemble learning is a practical family of methods for uncertainty modeling, particularly useful for sequential decision-making problems like recommendation systems and reinforcement learning tasks. The posterior on likelihood parameters is approximated by sampling an ensemble member from a predetermined index distribution, with the ensemble’s diversity reflecting the degree of uncertainty. In this paper, we propose Bayesian Ensemble (BE), a lightweight yet principled Bayesian layer atop existing ensembles. BE treats the selection of an ensemble member as a bandit problem in itself, dynamically updating a sampling distribution over members via Bayesian inference on observed rewards. This contrasts with prior works that rely on fixed, uniform sampling. We extend this framework to both bandit learning and reinforcement learning, introducing Bayesian Ensemble Bandit and Bayesian Ensemble Deep Q-Network for diverse decision-making problems. Extensive experiments on both synthetic and real-world environments demonstrate the effectiveness and efficiency of BE.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new ensemble learning method for sequential decision making, including contextual bandits and reinforcement learning (RL), called Bayesian Ensemble (BE). BE updates the ensemble index distribution using Bayesian inference on observed rewards. The authors validate their BE approach using extensive experiments on synthetic and real-world datasets.

### Strengths
* The contributions appear to be novel and technically sound.
* A clear explanation of the BE approach is presented, for both the bandit and RL settings.
* Bounds on the variance of the BE-DQN approach for RL provided in two theorems, both of which appear to be theoretically sound.
* The experimental results on synthetic and real datasets, which include regret, wall-clock run time, and cumulative reward, are convincing.

### Weaknesses
* Table 1 in the paper shows that the BEB method incurs a computational overhead of approximately 20% for hypermodel, due to the use of variational inference. A more thorough study of the scalability of the BE approach should be provided, as well as some options for mitigating scalability issues.
* The RL experiments are somewhat limited, since they only include experiments run using the MiniGrid environment.

### Questions
It would be helpful for the authors to respond to the weaknesses pointed out above.

### Soundness
3

### Presentation
3

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
This paper has proposed Bayesian Ensemble (BE), a modified framework of the existing framework of using ensemble for uncertainty modeling. The main novelty of Bayesian Ensemble is that **the index distribution of the ensemble is also updated**, in addition to the parameters of the ensemble model. Preliminary analysis results are presented in Section 5, and experiment results are demonstrated in Section 6.

### Strengths
- The core idea of this paper, i.e. updating the index distribution of an ensemble in addition to its parameters, is both novel and natural.

- The experiment results of this paper seem to be extensive.

- Overall, this paper is well written and easy to follow. This paper has also done a good job of literature review.

### Weaknesses
- My main concern is that the necessity, importance and advantage of using "Bayesian ensemble", i.e. also updating the index distribution, need to be better explained and justified. In theory, even with a fixed index distribution, by appropriately updating the model parameters, we can still achieve a good approximation of the posterior. Thus, we need a better justification of Bayesian ensemble. I think the theoretical justification of this paper is relatively weak. In particular, the analyses (Theorem 1 and 2) are limited to a specific case. Also, the conclusions on variances are only vaguely connected to the "exploratory behavior". I recommend that the authors significantly improve the theoretical results of this paper: first, please analyze under a general setting rather than a specific case; second, please present the analysis results in term of regret bounds rather than the variance. One possible result is that the authors might prove a better regret bound of Bayesian ensemble compared to the standard ensemble with a fixed index distribution. This will significantly strengthen the paper.

- The experiment results of this paper look interesting and strong, but it is unclear to me if this paper has done sufficient hyper-parameter sweep for the baseline agents. For instance, for ensemble+, how are the prior_scale, learning_rate, and $L_2$ weight decay tuned? Please explain. Similar questions for all other agents. 

- It seems that this paper has only considered Beta index distributions in the experiments, why? In addition to conjugacy, are there any other advantages of Beta distributions?

- Minor comments:
  - in equation 2, $r=r_i$ should be $r=R_i$
  - Please make images in Figure 3 larger, they are hard to read
  - If possible, please include Algorithm 1 and 2 in the main body of the paper to make it more self-contained

### Questions
Please address the weaknesses listed above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a new approach to Bayesian ensembles in sequential decision making; the main contribution to this front is to explicitly model a distribution over member indices, i.e., explicitly treat model assignment as a bandit problem. This results in the "Bayesian ensemble bandit (BEB)" framework, which unifies and generalizes a few existing approaches for Thompson-based sampling in the literature. A version of deep Q-learning using Bayesian ensembles (BE-DQN) is similarly introduced. Both BEB and BE-DQN are evaluated on a set of synthetic and real-world datasets, which show modest improvements over baselines.

### Strengths
- Optimizing over the index makes a lot of sense, and the proposed approach is both general and simple. In conjugate scenarios, it comes at essentially no computational overhead, and would intuitively improve results.
- The empirical results overall paint a positive picture, of minimal computational overhead and marginal performance improvement. 
- Variance analysis of BE-DQN, even in somewhat restricted settings, is a nice sanity-check to suggest nice behavior.

### Weaknesses
- One concern is the reliance on {0, 1} rewards. This makes sense for updates, as the Bernoulli-Beta conjugacy is nice to exploit, but it remains unclear how implementation or performance would work in the more general case.

- A well-known concern in Bayesian ensembles is collapse to a single model -- is this observed empirically, and is there any way to avoid this? 

- The experiments are in some ways lacking. For example, what is the effect of the number of members in an ensemble? How does ensemble+ perform on a subsample of the Yahoo dataset, if the entire dataset is too large? The BE-DQN results are also somewhat limited, being tested only on MiniGrid.

- As a minor note, in several places "enemble" is written instead of "ensemble" (lines 406, 412, 414, 415).

### Questions
To restate questions from weaknesses:

1. How is a more general reward space handled?

2. Do the Bayesian ensembles tend to collapse? How can this be alleviated, if so?

3. How does ensemble+ perform on a smaller version of the Yahoo dataset?

4. How does performance grow/stagnate as the number of ensemble members changes?

### Soundness
2

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
4

### Summary
This paper proposes **Bayesian Ensemble (BE)**, a framework designed to enhance existing ensemble-based methods for sequential decision-making. The core idea is to move beyond the standard approach of sampling ensemble members from a fixed, uniform distribution. Instead, BE treats the selection of an ensemble member as a bandit problem itself, dynamically updating a sampling distribution over the members using Bayesian inference based on observed rewards.

The authors instantiate this framework in two settings:
1.  **Bayesian Ensemble Bandit (BEB):** This applies the BE layer to contextual bandit problems, showing how it can "supercharge" existing methods like `ensemble+` (using Beta distribution updates for a discrete index) and `hypermodel` (using variational inference on a Gaussian distribution for a continuous index).
2.  **Bayesian Ensemble Deep Q-Network (BE-DQN):** This applies the framework to deep reinforcement learning. It maintains a distribution over an ensemble of K Q-networks and updates this distribution based on rewards, similarly to Thompson sampling.

The paper provides theoretical analysis for BE-DQN, proving that its Q-value variance is bounded between that of standard DQN and Ensemble DQN. Empirically, the authors demonstrate the effectiveness of their methods on synthetic benchmarks (Neural Testbed, Mushroom, MiniGrid) and a real-world dataset (Yahoo!R6B), showing improved regret and cumulative rewards over baselines.

### Strengths
* **Novelty and Simplicity:** The core idea of treating ensemble member selection as a bandit problem and updating the selection distribution via Bayesian inference is simple, intuitive, and novel. It provides a principled way to prioritize more "useful" ensemble members over time.
* **Generality:** The framework is shown to be flexible, adapting to both discrete index spaces (enhancing `ensemble+`) and continuous index spaces (enhancing `hypermodel`). Its application to both bandit and full RL problems demonstrates its broad utility.
* **Strong Empirical Results:** The method shows consistent performance improvements over its corresponding baselines across multiple domains:
    * **Bandits:** Lower regret in Neural Testbed and Mushroom environments.
    * **Real-World Bandits:** Higher cumulative clicks on the large-scale Yahoo!R6B dataset.
    * **Reinforcement Learning:** Stronger average performance across several MiniGrid environments compared to DQN, E-DQN, and RE-DQN.
* **Theoretical Justification:** The paper provides a theoretical analysis for the RL variant (BE-DQN), proving that its Q-value variance is bounded, lying between the variance of Ensemble DQN (lower bound) and standard DQN (upper bound). This adds a layer of soundness to the method's stability claims.

### Weaknesses
1.  **Missing Related Work:** The related work section and subsequent discussions overlook some highly relevant and recent papers. Contextualizing the BE framework against these works is crucial:
    * **Li et al. (2025, "Scalable Exploration via Ensemble++"):** This paper also focuses on scalable exploration using ensembles. The authors should discuss how their work differs from or relates to this version of Ensemble++.
    * **Li et al. (2024, "Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent"):** Given that a key baseline and contribution of this paper is the enhancement of `hypermodel`, the authors must discuss "HyperAgent," which also uses hypermodels for scalable posterior sampling in RL.
    * **Li et al. (2022, "HyperDQN: A Randomized Exploration Method for Deep Reinforcement Learning"):** This paper also explores randomized exploration in deep RL, which is highly relevant to the BE-DQN contribution. The authors should position their method relative to this work.
2.  **BE-DQN Design Justification:** The design of BE-DQN has an interesting asymmetry: it selects actions using the member with the *maximum probability* ($j=arg~max_{k}p_{k}^{i-1}$), but it computes the target value using a *weighted average* of all members ($\sum_{k=1}^{K}p_{k}Q(...)$). The paper justifies this as mitigating overestimation, but this design is not fully ablated or motivated. It's unclear why a more standard Thompson Sampling approach (sampling an index $j \sim p$ for action selection) was not used.
3.  **Theoretical Limitations:** As noted, the variance analysis for BE-DQN is performed in a zero-reward MDP. The authors rightly state in the appendix that these bounds may not hold in environments with high reward stochasticity. This is a significant limitation that should be more prominently acknowledged in the main paper, as it bounds the applicability of the theoretical stability guarantees.
4.  **Experimental Gaps:** The exclusion of the `ensemble+ (BEB)` method from the real-world Yahoo!R6B experiment due to computational cost is understandable but leaves a gap in the evaluation. This was one of the two main bandit instantiations, and its performance on a large-scale problem remains unknown.

### Questions
1.  **Missing Citations:** Could you please discuss the relationship between your proposed BE framework and the following highly relevant works:
    * Li et al. (2025), "Scalable Exploration via Ensemble++"
    * Li et al. (2024), "Q-Star Meets Scalable Posterior Sampling... via HyperAgent"
    * Li et al. (2022), "HyperDQN: A Randomized Exploration Method..."
    How does your work compare or contrast, especially regarding the `ensemble++`, `hypermodel`, and DQN enhancements?
2.  **BE-DQN Design:** Can you provide more justification for the asymmetric design of BE-DQN, where action selection uses the `arg max` member while the target calculation uses a weighted average? Have you experimented with sampling the action-selection member $j$ from the distribution $p$ (i.e., $j \sim p$), which would seem more aligned with Thompson Sampling?
3.  **`ensemble+ (BEB)` Sampling:** Could you please clarify the *exact* sampling mechanism for the `ensemble+ (BEB)` variant? Is the member index $z$ at each step chosen by sampling $w_i \sim Beta(\alpha_i, \beta_i)$ for all $i$ and then setting $z = arg~max_i w_i$? If so, how does this relate to the goal of Thompson sampling, and have you considered simply sampling $z$ from a categorical distribution whose parameters are derived from the Beta posteriors?
4.  **Practicality of Variance Bounds:** Your theoretical analysis is based on a zero-reward environment. How do you expect the variance of BE-DQN to behave in practical environments with significant reward variance? Does this limitation on the theory affect the practical stability of the algorithm?

### Soundness
3

### Presentation
3

### Contribution
2
