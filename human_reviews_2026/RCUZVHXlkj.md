# Long-term Fairness with Selective Labels

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Long-term fairness algorithms aim to satisfy fairness beyond static and short-term notions by accounting for the dynamics between decision-making policies and population behavior. Most previous approaches evaluate performance and fairness measures from observable features and a label, which is assumed to be fully observed. However, in scenarios such as hiring or lending, the labels (e.g., ability to repay the loan) are _selective labels_ as they are only revealed based on positive decisions (e.g., when loan is granted). In this paper, we study long-term fairness in the selective labels setting, and analytically show that naive solutions do not guarantee fairness. To address this gap, we then introduce a novel framework that leverages both the observed data and a label predictor model to estimate the true fairness measure value, by decomposing into the observed fairness and bias from labels predictions. This allows us to derive the sufficient conditions to satisfy true fairness from observable quantities by using the confidence on the predictor model.  Finally,  we rely on our theoretical results to propose a novel reinforcement learning algorithm for effective long-term fair decision-making with selective labels. In semisynthetic environments, the proposed algorithm reached comparable fairness and performance to an agent with oracle access to the true labels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies long-term fairness in a sequential decision setting where only the labels of admitted samples are available. The paper first proves that only ensuring fairness in the observed samples cannot guarantee overall fairness. Motivated by this, the authors proposed a RL-based algorithm with an added $L^{Renyi}$ term to upper bound the fairness divergence. Experiments on synthetic datasets demonstrate the effectiveness of the algorithm.

### Strengths
1. The motivation is reasonable, and the selection biases do exist in real-world settings.
2. The decomposition of the disparity and the upper bound derivation seem to be correct.
3. The algorithm design is clear, and experimental results support the claims.

### Weaknesses
1. The method heavily depends on the accuracy of the label predictor $\phi$. But $\phi$ itself is only correct under the overlap assumption that accepted and rejected samples share enough support. In real-world settings such as loan application, it is reasonable to believe some applicants will never be accepted,i.e., some features will never be covered. 

2. $\phi$ is also implemented as a simple logistic regression model, and the generalization to complex, high-dimensional data is uncertain.

3. Synthetic experiments may not reflect the applicability in real-world settings.

4. Minor: I feel that this work is closely related to fairness in sequential strategic classification and performative prediction settings, while some related works are missing (e.g., [1,2])

[1] Xie, Tian, and Xueru Zhang. "Automating data annotation under strategic human agents: Risks and potential solutions." Advances in Neural Information Processing Systems 37 (2024): 127436-127482.

[2] Somerstep, Seamus, Ya'acov Ritov, and Yuekai Sun. "Algorithmic fairness in performative policy learning: Escaping the impossibility of group fairness." Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency. 2024.

### Questions
I feel that $c$ can be very important for the performance of the algorithm. Did the authors explore how the cost affects the results?

### Soundness
3

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
This paper studies the problem of satisfying long-term fairness in scenarios with selective labels, that is, where action feedback is only observable based on positive decisions. The authors argue that this specific setting can create a flawed objective where optimal policies can learn to minimize $\Delta^{A=1}_t$ without necessarily minimizing disparity within the rejected population. They propose a framework for approaching these problems and an algorithm to solve it based on advantage regularization of PPO. Finally, they conduct experiments on 2 case studies and show that they are able to achieve higher reward and lower disparity than an oracle baseline.

### Strengths
* The paper studies an important problem in considering long-term fairness with partial observability on rejected candidate labels. The context the authors study this in is high-stakes and can have a significant effect on people's lives.
* The proposed method is fairly simple to implement since it is just regularizing the PPO advantage. 
* The experimental results support the claims made by the authors and their proposed framework/algorithm.

### Weaknesses
* There seems to be some highly related works [1, 2] not mentioned in the paper. Could the authors please give a comparison with these works?
* It would be a more compelling paper to include at least one more case study (ideally studying Accuracy Parity now, since the other two proposed fairness formulations are studied by the two given case studies), and more baseline methods, to compare with SELLF. For example, [1] considered the tasks of criminal justice, health care, and insurance. Another example could be long-term exposure fairness in recommendation [4]. For baselines, [1] provided a contraction method to address this problem. Another potential baseline is an constrained RL approach such as FOCOPS [2] or CPO [3] and treat the disparity as an expected cost to minimize.

I am happy to raise my score if my concerns above are addressed.

Some minor Typos (probably run a typo checker upon revision):
* Line 104: selection -> select
* Line 211: cofounded -> confounded
* Line 228 -> depends -> depend
* Line 286: Labes -> Labels?
* etc.


[1] Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., & Mullainathan, S. (2017). The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 275–284. Presented at the Halifax, NS, Canada. doi:10.1145/3097983.3098066

[2] Zhang, Y., Vuong, Q., & Ross, K. W. (2020). First Order Constrained Optimization in Policy Space. arXiv [Cs.LG]. Retrieved from http://arxiv.org/abs/2002.06506

[3] Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). Constrained Policy Optimization. arXiv [Cs.LG]. Retrieved from http://arxiv.org/abs/1705.10528

[4] Mansoury, M., & Mobasher, B. (2023). Fairness of Exposure in Dynamic Recommendation. arXiv [Cs.IR]. Retrieved from http://arxiv.org/abs/2309.02322

[5] Chang, T., & Wiens, J. (07 2024). From Biased Selective Labels to Pseudo-Labels: An Expectation-Maximization Framework for Learning from Biased Decisions. Proceedings of Machine Learning Research, 235, 6286–6324.

[6] Yu, E. Y., Qin, Z., Lee, M. K., & Gao, S. (2022). Policy optimization with advantage regularization for long-term fairness in decision systems. Proceedings of the 36th International Conference on Neural Information Processing Systems. Presented at the New Orleans, LA, USA. Red Hook, NY, USA: Curran Associates Inc.

### Questions
Please see weaknesses. Also, some additional questions:
* I notice you use a linear predictor architecture. Does increasing the number of layers have any effect on your performance?
* Line 136: Should the beta distribution $Be(\cdot)$ require 2 parameters, $\alpha, \beta$, instead of just the one you provided?
* Is there a reason why you put $L^{\text{Renyi}}$ into the objective function in Line 109, rather than creating an additional regularization term in the advantage? I am wondering if placing this penalty term in the objective vs advantage will incur any reward hacking issues as seen in [6]. 
* In Line 326, what is semisynthetic about the environments?

### Soundness
2

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
This paper tackles the critical problem of achieving long-term fairness in sequential decision-making systems where the true outcome (label) is only observed for selections. This "selective labels" setting is common in real-world scenarios like lending (where repayment ability is only known if a loan is granted) or hiring (where job performance is only known for hired candidates). The authors first demonstrate formally that naive approaches, such as measuring fairness only on the sub-population with observed labels, are insufficient and can fail to guarantee fairness for the overall population. Then the paper introduces a novel theoretical framework that uses a label predictor to impute the labels for the "rejected" population. The core of their theoretical contribution is a decomposition (Theo. 3.1) that precisely links the true disparity to the observed disparity calculated using the predictor's imputed labels. This decomposition shows that the observed disparity is confounded by the policy's rejection rate and the predictor's error on the rejected group.

To address the problem, the authors propose a new reinforcement learning algorithm, SELLF (SElective Labes in Long-term Fairness). SELLF is based on PPO and incorporates the paper's theoretical insights through two key mechanisms: (1) It penalizes the observed disparity; and (2) It introduces a novel regularization term ($L^Renyi$) that penalizes the Renyi divergence. This new loss term directly corresponds to the theoretical bounds and encourages the policy to take actions that reduce the predictor's error and improve the confidence of its estimates.

### Strengths
- The paper's primary originality lies in its formal problem formulation to model the intersection of long-term fairness and selective labels. 

- The authors provide a clear, formal progression from demonstrating the failure of naive methods to a full decomposition of the observed disparity in Theorem 3.1 and to actionable, observable conditions in Theorem 3.4.

- The mathematic framework of this paper is rigorous. The authors have shown how did they identify the disparity, and its error bounds directly motivates the design of the proposed learning algorithm.

### Weaknesses
- The motivation for choosing Inverse Propensity Weighting (IPW) to estimate the predictor's error on the rejected population ($D_R^i$) using data from the accepted population ($D_A^i$) feels abrupt. This justification is critical because the entire framework and algorithm are now built upon IPW, which is notoriously unstable and suffers from high variance, especially when acceptance probabilities are low. And I would suggest the authors to provide some implication on Assumption 1.
- The paper's core premise of *selective labels* is a potentially problematic and imprecise way to frame the problem. A more accurate conceptualization would be data selection bias. The issue is not necessarily that the label $Y$ is selectively realized only upon a positive action $A\_t=1$. Rather, a true, latent qualification $Y$ should be assumed to exist for all individuals, and a separate selection variable $S$ (in this work, the policy's action, $A_t$, which is based on the model's predictions) merely determines whether $Y$ is observed by the decision-maker. The data selection is standard in missing data and causal inference literature. 
- The introduction of the $L^\text{Renyi}$ loss in Eq. (6) is confusing. The loss is justified as a practical way to control the theoretical error bound from Theorem 3.3. However, this relies on the unsubstantiated assertion that this specific Renyi divergence term "will be dominated" by other terms in the bound. This makes the $L^\text{Renyi}$ term feel like a complex and indirect proxy, obscuring the direct connection between the algorithm's objective and the actual goal of minimizing fairness disparity. 
- The framework depends on a label predictor ($\phi$) to estimate the unobserved error ($\epsilon_t^i$). However, this predictor is itself trained on the same selectively-labeled, biased data (Eq. 7). The paper does not fully address how errors or biases in this predictor (resulting from unstable IPW training) might in turn corrupt the disparity estimate ($\tilde{\Delta}$) and the error bounds ($\bar{\epsilon}^i$).
- Although the paper is framed as a "long-term" fairness study, the objective (Eq. 1) is to satisfy a static fairness constraint $|\Delta_t| \le \omega$ at every timestep $t$. This formulation does not fully capture the dynamics of fairness over time, such as how unfairness at one step might be permissibly traded for greater fairness at a later step. The experiments show the results of fairness over time, but the problem's objective remains a per-step constraint rather than a truly holistic long-term objective.

### Questions
1. Line 50:  presents great in impact in sequential decision-making -> presents a great impact on sequential decision-making
2. Line 52: make it not trivial to obtain -> make it non-trivial to obtain
3. Line 62: *presents* conditions
4. Line 104: The decision-maker will *select* actions to maximize a reward function.
5. In Definition 2, where does $\mathcal{F}$ of $\mathcal{F}$-MDP come from? And what is the notation of $Be(\cdot)$?
6. As shown in Figure 1, the selection variable $A$ is only dependent on the features $X$ and $Z$. Should not this also depend on the true qualification $Y$?
7. Line 228: as they depends on -> as they depend on
8. Line 256: In the first sentence of Theorem 3.3, Let $d < \infty$ be the psuedo-dimension

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
This paper studies long-term fairness by addressing the selective labels problem, where outcomes are only observed for positively decided cases. The authors show that naive approaches to measuring fairness under selective labels fail to guarantee true population-level fairness. They introduce a framework that uses a label predictor to impute missing labels and derive theoretical conditions under which observed disparity bounds translate to true disparity bounds. Experiments are conducted on lending and school admission environments.

### Strengths
1. The combination of long-term fairness and selective labels is both relevant and, to the best of my knowledge, previously understudied. The authors correctly point out the gap and address it to an extent.
2. The theoretical contributions are strong (disclaimer: I have not thoroughly verified the proofs).
3. The proposed method has a principled algorithmic design grounded in the theoretical results and can handle multiple fairness notions.

### Weaknesses
1. Assumption 1, that every feature combination with non-zero rejection probability must also have non-zero acceptance probability, is very strong and potentially unrealistic. In practice, certain subpopulations may be systematically excluded. The authors should better acknowledge these limitations. While I understand these assumptions are needed for the theoretical results, the authors could improve the paper’s practical relevance by experimentally evaluating cases where these assumptions do not hold. The stationarity assumption of the F-MDP is only briefly discussed at the end; further investigation of its implications in practice would also be valuable.

2. Some aspects of the algorithmic design are unclear. The paper moves from bounding predictor error to minimizing it using Renyi divergence as a proxy. In Line 307, the authors state that “in practice, the bound from Theo. 3.3 will be dominated by the divergence term,” but no proof or empirical validation supports this claim. Why not directly optimize for the prediction error?

3. The baseline selection is quite limited and POCAR is the only fair RL algorithm compared against. I suggest including at least two additional baselines from the list below. To my knowledge, Xu et al. (2023) is a particularly strong comparison.



4. Some citations are missing; I suggest the authors include the following (the list is not exhaustive):
* Jabbari, Shahin, et al. "Fairness in reinforcement learning." International conference on machine learning. PMLR, 2017.
* Satija, Harsh, et al. "Group fairness in reinforcement learning." Transactions on Machine Learning Research (2023).
* Xu, Yuancheng, et al. "Adapting static fairness to sequential decision-making: Bias mitigation strategies towards equal long-term benefit rate." arXiv preprint arXiv:2309.03426 (2023).
* Rezaei-Shoshtari, Sahand, et al. "Fairness in Reinforcement Learning with Bisimulation Metrics." arXiv preprint arXiv:2412.17123 (2024).
* Deng, Zhihong, et al. "What hides behind unfairness? exploring dynamics fairness in reinforcement learning." Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence. 2024.
* Frauen, Dennis, Valentyn Melnychuk, and Stefan Feuerriegel. "Fair off-policy learning from observational data." Proceedings of the 41st International Conference on Machine Learning. 2024.

### Questions
Please answer my questions above.

### Soundness
2

### Presentation
2

### Contribution
3
