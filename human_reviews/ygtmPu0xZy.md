# Scalable Exploration via Ensemble++

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
Scalable exploration in high-dimensional, complex environments is a significant challenge in sequential decision making, especially when utilizing neural networks. Ensemble sampling, a practical approximation of Thompson sampling, is widely adopted but often suffers performance degradation due to ensemble coupling in shared layer architectures, leading to reduced diversity and ineffective exploration. In this paper, we introduce Ensemble++, a novel method that addresses these challenges through architectural and algorithmic innovations. To prevent ensemble coupling, Ensemble++ decouples mean and uncertainty estimation by separating the base network and ensemble components, employs a symmetrized loss function and the stop-gradient operator. To further enhance exploration, it generates richer hypothesis spaces through random linear combinations of ensemble components using continuous index sampling. Theoretically, we prove that Ensemble++ matches the regret bounds of exact Thompson sampling in linear contextual bandits while maintaining a scalable per-step computational complexity of $\tilde{O}( \log T)$. This provides the first rigorous analysis demonstrating that ensemble sampling can be an scalable and effective approximation to Thompson Sampling, closing a key theoretical gap in exploration efficiency. Empirically, we demonstrate Ensemble++'s effectiveness in both regret minimization and computational efficiency across a range of nonlinear bandit environments, including a language-based contextual bandits where the agents employ GPT backbones. Our results highlight the capability of Ensemble++ for real-time adaptation in complex environments where computational and data collection budgets are constrained. \url{https://anonymous.4open.science/r/EnsemblePlus2-1E54}

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents Ensemble++, an ensemble sampling method that enhances the computational efficiency of Thompson sampling. Recently, Ensemble+ was developed to tackle this challenge; however, it faces an issue with ensemble coupling that adversely affects its performance. To address this problem, Xu et al. later introduced Langevin Monte Carlo Thompson Sampling, which, unfortunately, incurs high computational costs. The aim of this paper is to improve the approach to ensemble coupling while minimizing computational expenses. The proposed approach, Ensemble++, addresses this by implementing decoupled optimization and lifted index sampling, enhancing exploration and uncertainty estimation. Theoretically, the paper shows that it achieves the same regret bounds as exact Thompson sampling in linear contextual bandits, with a per-step computation complexity of $\tilde{O}(\log T)$.
Empirical results in the performance of  Ensemble++ are presented.

### Strengths
The paper tackles an important issue: scalable exploration poses a significant challenge in sequential decision-making tasks, including reinforcement learning (RL) and contextual bandits, especially relevant in high-dimensional practical environments.

### Weaknesses
The paper lacks details in many instances, which hinders understanding the contributions of the paper.

### Questions
1. The two key innovations in the proposed model are noted as (1) a variance-aware discretization method that prevents the exponential growth in ensemble size and (2) a reduction to sequential random projection techniques. However, the current presentation of the paper does not clearly explain how these innovations are applied. It is not clear how the two proposed changes improve upon the shortcomings of the existing approaches. Additionally, the solution approach lacks clarity without the algorithm.

    a) Can you provide a high-level description of how the variance-aware discretization method works and how it prevents exponential growth in ensemble size?

    b) Explain the key steps in applying sequential random projection techniques to their problem.

    c) Include a pseudocode description of the Ensemble++ algorithm to clarify the solution approach.

    d) Explain how the proposed modifications address the shortcomings of existing approaches like Ensemble+.

2. The total computational complexity is $O(d^3 \log T)$. However, it is not clear from the main paper how this result was concluded. Can you provide a proof sketch and key insights into how the specific approach resulted in this computational complexity and how it overcame the bottleneck in the existing approaches?

3. The primary contribution is the computational advantage in high-dimensional settings. To validate the effectiveness of the proposed approach, experiments showcasing the variation of regret with respect to computational time will be beneficial. 

    a) Can you include a plot showing regret vs. computation time for Ensemble++, Ensemble+, and the Langevin Monte Carlo approach?

    b) Provide a detailed comparison with the Langevin Monte Carlo approach in  Xu et al., highlighting both performance and computational efficiency.

    c) Discuss any trade-offs between regret and computation time for each method.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an improved ensemble exploration method called Ensemble++, which incorporates decoupled optimization and lifted index sampling for more efficient exploration and uncertainty estimation. The architecture outperforms existing methods and is scalable in terms of computational cost.

### Strengths
(1) As demonstrated in the paper, Ensemble++ is computationally efficient, which is highly beneficial in practice. 

(2) Ensemble++ addresses the ensemble coupling issue by utilizing the stop-gradient operator.

### Weaknesses
In my understanding, the primary weakness of the paper lies in the theoretical analysis in Section 4. Specifically, the impact of the stop-gradient operator on regret performance is unclear, which raises concerns about whether the regret analysis fully supports the main idea.  However, if I am missing something, please let me know, and I will reconsider my evaluation.

### Questions
More details should be provided on the stop-gradient operator in the network. Is it solely intended to block gradient flow? If so, how to do the ensemble process? I think the current explanation may not be sufficient for readers to fully understand this aspect.

### Soundness
3

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
4

### Summary
Scalable exploration is challenging in high-dimensional bandit settings. Ensembling is one (computationally expensive) way to try to approximate thompson sampling; ensembling can be made computationally cheaper by sharing one feature extractor between ensemble members, but can underestimate uncertainty due to the ensemble coupling issue, where ensemble members are too similar. 

Authors propose Ensemble++, which uses stop gradients and a high-dimensional ensemble index. Empirically, authors find Ensemble++ outperforms existing methods in bandit settings. Authors analyze regret and computation for Ensemble++ in linear contextual bandit settings; Ensemble++ achieves the same regret bounds as exact TS, and has $\tilde O(\log T)$ per-step computation complexity.

============================
I appreciate the author's response to my questions and comments and have accordingly raised my score.

### Strengths
The motivation seems strong, the method is largely straightforward and well-motivated (see questions for exceptions), the paper seems to be overall well-written and well-organized, and the experiments seem promising.

### Weaknesses
* **Insufficient discussion of related work** While there is a mention and empirical comparison to epinet in the paper, I felt there was insufficient discussion of how the proposed method is methodologically different from epinets, as they also use index functions.

* **Unclear references to posterior approximation** There are references made to things approximating a posterior, but it was not clear what kind of an approximation that is (e.g. the way MCMC sampling is a posterior approximation, vs VI is a posterior approximation, vs a closed-form posterior for a different but similar problem is an approximation). 


    * **Proposition 1** Proposition 1 is described as a closed-form update; while an earlier sentence mentions posterior approximation, from the Appendix, Proposition 1 is deriving the closed-form minimizers of the loss (2) with respect to $A$ and $\mu$. Is this supposed to also be a posterior update? If so, what are the assumptions on the Bayesian model? 

    * **Lemma 1** This result assumes $\Sigma_t$ is the true posterior variance, but it is not clear why that should be the posterior variance. It seems like there are some assumptions and/or definitions missing. 

    * **Experiments** There are no experiments that compare Ensemble++ vs other methods to a ground truth posterior, despite claims about Ensemble++ better approximating the posterior. (The first setting in 5.1 does appear to compare with ground-true TS, if I am understanding correctly, but this is the only example.)

* **Sequential dependency** I still don't understand what exactly is the problem in regular ensembles and how it is mitigated in Ensemble++.

* There are claims that the stop gradient prevents ensemble coupling; it would help if this could be empirically measured (beyond better bandit regret). 

* I think it could help for readability to have an algorithm box to summarize the method. 

Also see questions for areas that were confusing.

### Questions
* In the abstract (and elsewhere in the paper), how is ensemble sampling a computationally efficient approximation of Thompson sampling? It seems that it is in specific settings, e.g. the Gaussian setting in Lemma 3 in [1], and the linear setting in this paper where $P_\xi$ is zero-mean and isotropic. If I am understanding correctly, then this limitation in the connection between ensembles and posteriors needs to be specified. 
* In line 195-196, it is mentioned that there are $M$ ensemble components, but $A$ is simply a matrix of size $d\times M$; are these $M$ ensemble components separate in any way? 
* How is the variance of the $Z_{s,m}$ perturbations (139-140) chosen? Are these heuristics, or is there a principled reason for their existence? 
* The abstract / intro mention computational complexity but I don't see it derived anywhere. Is this supposed to be in the paper? 

[1] Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep reinforcement learning. Advances in Neural Information Processing Systems, 31, 2018.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Scalable exploration in sequential decision-making is challenging, especially in high-dimensional environments. Ensemble sampling, an approximation of Thompson sampling, is widely used but can suffer from performance degradation due to ensemble coupling in shared-layer networks. The Ensemble++ architecture is proposed to overcome this limitation by introducing decoupled optimization and lifted index sampling. Empirical results show that Ensemble++ outperforms existing methods in regret minimization across various tasks.

### Strengths
(1) The paper proposes the Ensemble++ architecture, which achieves scalability with O(d³log T) per-step computation. This closes a long-standing gap in scalable exploration theory.

(2) The paper empirically validates the scalability and efficiency of Ensemble++ through experiments in bandit tasks, including language-input contextual bandits using a GPT backbone.

### Weaknesses
I appreciate the motivation and approach presented. However, I find that the methods, analysis, and experiments are inconsistent. The motivation of this method is primarily based on ensemble neural networks, which suffer from gradient coupling issues. Yet, the analysis and experiments focus on linear contextual bandits, where gradient descent is not commonly used. Additionally, an important related work, "neural contextual bandits [1,2,3]," has been overlooked. I believe that the analysis from neural contextual bandits could be adapted to this method.


[1] Neural Contextual Bandits with UCB-based Exploration
[2] Ee-net: Exploitation-exploration neural networks in contextual bandits.
[3] Federated Neural Bandits

### Questions
(1) May include additional analysis or experiments on neural network models to demonstrate how Ensemble++ addresses the gradient coupling issue in that setting.

### Soundness
2

### Presentation
3

### Contribution
3
