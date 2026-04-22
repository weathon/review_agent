# Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 8, 4

## Abstract
Model-based offline reinforcement learning (RL) aims to enhance offline RL with a dynamics model that facilitates policy exploration. However, model exploitation could occur due to inevitable model errors, which degrades algorithm performance. Adversarial model learning offers a theoretical framework to mitigate model exploitation by solving a maximin formulation, and RAMBO provides a practical implementation with model gradient. However, we empirically observe that severe Q-value underestimation and gradient explosion can occur in RAMBO with only slight hyperparameter tuning, suggesting that it tends to be overly conservative and suffers from unstable model updates. To address these issues, we propose RObust value-aware Model learning via Implicitly differentiable adaptive weighting (ROMI). Instead of updating the dynamics model with model gradient, ROMI introduces a novel robust value-aware model learning approach. This approach requires the dynamics model to predict future states with values close to the minimum Q-value within a scale-adjustable state uncertainty set, enabling controllable conservatism and stable model updates. To further improve out-of-distribution (OOD) generalization during multi-step rollouts, we propose implicitly differentiable adaptive weighting, a bi-level optimization scheme that adaptively achieves dynamics- and value-aware model learning. Empirical results on D4RL and NeoRL datasets show that ROMI significantly outperforms RAMBO and achieves competitive or superior performance compared to state-of-the-art methods on datasets where RAMBO typically underperforms.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes an alternative approach to model-based offline reinforcement learning, building on the idea of value-aware model learning. A model is constructed that minimizes an adversarial value difference error as a weighing to a log likelihood reconstruction loss. The authors show the efficacy of their method on standard offline RL tasks, in addition to providing some theoretical insights into the method.

### Strengths
While the overall method is slightly complex, the paper presents the motivation and contribution of the work clearly. The proposed integration of conservative value-aware model learning into offline model-based RL is reasonable and shows some promising results. In addition, known issues about the generalization abilities of value-aware model learning are appropriately avoided with the bi-level optimization scheme.

### Weaknesses
I genuinely do not have any major issues with the paper, so the following are excessively nitpicky and mostly comments on the writing.

I believe the writing of the paper could be strengthened slightly by defining the method less in contrast to RAMBO, as I believe the method conceptually stands on it's own.

Line 175: I don't believe this is a proper example of an adversarial loss, as there is no min/max formulation with competing networks.

It would be polite to cite Farahmand for coining the term value-aware as soon as it appears. Since the Farahmand paper is cited, I don't believe this is an egregious error. Similarly, the method is somewhat conceptually similar to Voelcker et al., as a reweighted regular MLE loss is considered. Again, prior work is acknowledged, so this is not really an issue.

### Questions
How is the state distance metric defined? This seems to be a crucial implementation detail.

How are the uncertainty intervals in the empirical section computed? Are bolded results statistically significant? Why do aggregated results and baseline lack uncertainty intervals?

Can the method work for higher dimensional problems as well? Pessimism over a state uncertainty set could conceivably be harder to tune for larger more high-dimensional problems, and all those tested here are rather low-dimensional.

How was hyperparameter tuning and model selection conducted? Was there a separate offline validation set, or where hyperparameters tuned by testing in the online environment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ROMI, a new method for model-based offline reinforcement learning (RL) that addresses instability and over-conservatism found in previous approaches like RAMBO. ROMI uses a robust value-aware model learning strategy and a bi-level optimization scheme to better control conservatism and improve generalization. Experiments show ROMI outperforms RAMBO and matches or exceeds other state-of-the-art methods on standard offline RL benchmarks.

### Strengths
- Proposes a novel, robust approach that addresses instability and over-conservatism in model-based offline RL. The bi-level optimization scheme appears a promising solution for adaptive conservatism, improving both stability and performance
- Demonstrate strong empirical results on MuJoCo datasets, outperforming or matching state-of-the-art baselines
- Proposed framework appears to be less sensitive to hyperparameters and more generalizable across tasks, as shown e.g. in Fig. 3

### Weaknesses
- Even though robustness has clearly gone up, the approach may still require careful tuning of certain parameters - i.e. in Fig. 3 we see that while 0.01-1 are close together, setting the parameter to 10 yields very unstable results. Since all evaluations are in a relatively similar task domain (MuJoCo hopper, walker and halfcheetah), it is unclear whether the range of stable parameters is representable for other tasks as well, i.e. for other environments tuning the parameter may again become an issue.
- More generally speaking, while the authors clearly demonstrate performance gains on a set of well established offline RL benchmarks, the distribution of these tasks is quite narrow - it would be interesting to see how the algorithm performs on a broader selection
- Another downside I see is, if I am not mistaken, that the degree of uncertainty that governs the training needs to be set a priori and cannot be adjusted at runtime. Prior methods, such as [1,2,3,4] (could also help your related work section) have developed policies that can be adapted immediately when it is observed that the behavior is too far OOD or too conservative, which can be an advantage over having to go back and retrain. Of course, this point does not necessarily have to be addressed, the proposed method has other demonstrated merits which justify publication.
- The method introduces additional complexity through the bi-level optimization, which may increase implementation difficulty and be a burden in practical application

[1] Ghosh, D., Ajay, A., Agrawal, P., & Levine, S. Offline RL Policies Should be Trained to be Adaptive. Proceedings of the 39th International Conference on Machine Learning (ICML), 2022.

[2] Hong, J., Kumar, A., & Levine, S. Confidence-Conditioned Value Functions for Offline Reinforcement Learning. International Conference on Learning Representations (ICLR), 2023.

[3] Swazinna, P., Udluft, S., & Runkler, T. User Interactive Offline Reinforcement Learning. International Conference on Learning Representations (ICLR), 2023. 

[4] Zhang, Y., Liu, J., & Wang, Y. Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

### Questions
Am I correctly assuming that the parameter governing the size of the uncertainty set has to be set a priori?

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
3

### Summary
This paper looks into an instability noticed in RAMBO with underestimated Q-values and proposes a new method called ROMI to fix this. Their method aims to prevent model exploitation by training the model to balance predicting a close-by state with the lowest value, as well as getting the dynamics correct.

### Strengths
Interesting finding in RAMBO, seemingly strong results, includes ablations, and an interesting method of balancing competing objectives that I have not come across in RL before.

### Weaknesses
**W1.** There are lots of moving parts. It seems significantly more complicated to implement and understand than RAMBO.

**W2.** The authors say they compare all methods at 1M steps for fairness. However, since some baselines like MOBILE were tuned for 3M steps, it would be important to see if ROMI's performance holds up relative to MOBILE etc when trained for the full 3M steps.

**W3.** Larger compute requirement.

**W4.** They solve the problem of tuning lambda, but introduce more new hyperparameters: xi, H, N.

### Questions
**Q1.** Where are the results from for each method for both the D4RL and Neo-RL benchmarks? If they are from the authors own implementations, how did they ensure they were implemented correctly and tuned fairly?

**Q2.** How was Wasserstein distance chosen. Did the authors try or consider any other distance metrics?

### Soundness
2

### Presentation
2

### Contribution
2
