# Continuity-Regularized Flow Matching for Offline Reinforcement Learning

- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Flow-matching policies have recently emerged as a powerful class of generative models for offline reinforcement learning (RL), capable of capturing complex, multi-modal action distributions from static datasets. However, standard training objectives are largely agnostic to the global properties of the generative path, permitting learned vector fields that are irregular and unstable, which can hinder performance. In this work, we introduce PDE-regularized Q-Learning (PQL), a novel algorithm that addresses this limitation by imposing a principled structure on the entire probability flow. PQL makes two synergistic contributions: first, a partial differential equation based regularizer derived from the continuity equation enforces global smoothness and stability on the flow. Second, to solve the complex optimization problem introduced by this regularizer, we propose a Beta-distributed timestep sampling strategy that focuses learning on the critical trajectory segments where the trade-off between imitation and smoothness is most acute. Through extensive experiments, we demonstrate that by structuring the generative journey and not just its destination, PQL achieves state-of-the-art performance on a wide range of challenging offline RL tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work puts forth PQL i.e. PDE regularized Q-learning for offline RL setting towards improving flow-matching policies. To this end, the paper identifies that standard flow-matching methods often ignore the trajectory path, while just ensuring the starting point and the target point of the generative process are correctly accounted for. PQL attempts to ameliorate the path inconsistency issue with a PDE assisted regularizer that essentially penalizes the learnt vector field to maintain smoothness and stability. A beta-distributed timestep sampler is introduced to specifically focus on the intermediate trajectory where the conversion from pure noise to a very meaningful action while 
balancing between imitation and smoothness is critical at training time. Experimental evaluations are performed on relevant offline RL and IL task environments with comparison against benchmark algorithms and ablation studies.

### Strengths
1. PDE based regularization is an innovative approach towards enforcing generative path stability in flow matching policy algorithms.

2. The adaptive timestep sampling strategy while presented in the context of handling the PDE regularizer, appears to be a principled way to improve solution tractability with flow-based constrained optimization setups.

3. The practical utility of PQL is supported by improvements seen in benchmarking experiments. Hyper-parameter tuning analysis and ablation studies further strengthen credibility.

### Weaknesses
1. It is not clear computational overhead added by the Jacobian-based regularizer and how does that component ultimately trade-off in performance gains.

2. The Jacobian penalty while handles deformation of the flow, appears to be state agnostic. It deserves a separate study whether PQL would over-smoothen policies in critical intermediate states where complex, sharp actions were actually optimal.

3. There are multiple typos and grammatical mistakes throughout the paper that need to be corrected.

### Questions
In addition to the weakness comments, I request authors' response to the following questions :

1. How does PQL's wall-clock training time compare to standard flow-matching algorithms ?

2. The beta distributed adaptive seems like the intuitive next step beyond a uniform sampler. Is it worth investigating whether other distribution classes might work here ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a regularizer into the flow-matching-based policy to improve training stability and proposes a Beta-distributed time sampling strategy that enables stable and efficient optimization.

### Strengths
1. The experiments are thorough and well-executed, demonstrating solid empirical results.

2. The paper provides sufficient theoretical justification and proofs.

### Weaknesses
This paper appears to have been written somewhat hastily. In addition to several obvious typos and inconsistencies, there are also issues in understanding and experimental setup, as detailed below:

1. There are several typos. For example, in the Introduction, the last item of the listed contributions does not include the method name, which looks like a placeholder from a template.

2. The paper’s writing is sometimes difficult to follow, and the presentation could be improved.

3. The authors claim that the introduced regularizer improves training stability, but in Section 5.2, it is not clearly shown that the model without the regularizer is unstable.

4. The Introduction mentions that the Beta-distributed time sampling strategy also contributes to training stability, yet there is no direct evidence or analysis of stability in the experiments. Moreover, this Beta-distributed time sampling strategy is not discussed in the Method section.

5. No code

### Questions
1. In Section 5.2, the results do not clearly show that the version without the regularizer is unstable—could the authors clarify this?

2. Why is the Beta-distributed time sampling strategy mentioned in the Introduction and Appendix, but not described in the Method section?

### Soundness
3

### Presentation
2

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
The paper proposes PDE-regularized Q-learning (PQL), an extension of Flow Q-Learning (FQL) [1] by regularizing the Frobenius norm of the Jacobian $\nabla_a u_\theta(s, a, t)$ of the action policy $u_\theta(s, a, t)$. The intuition is that by controlling the Frobenius norm, the 2-Wasserstein distance between the true marginal probability $\rho_t^*(\cdot|s)$ and $\rho_t^\theta(\cdot| s)$. It provides a theoretical upper bound, provided the boundedness of the action policy. Furthermore, it introduces Hutchinson's trace estimator with JVP autodifferentiation to make memory-efficient computation. PQL is validated on D4RL, Adroit, and OGBench and demonstrates performance improvement in comparison with FQL.

[1] Park, Seohong, Qiyang Li, and Sergey Levine. "Flow q-learning." arXiv preprint arXiv:2502.02538 (2025).
[2] Hoffman, Judy, Daniel A. Roberts, and Sho Yaida. "Robust learning with Jacobian regularization." arXiv preprint arXiv:1908.02729 (2019).

### Strengths
1. Improvements in performance in different benchmarks
2. Theoretical guarantees for path stability are provided
3. Experiment validation is quite thorough
4. The method is straightforward to implement

### Weaknesses
1. The idea of doing Jacobian regularization is not new e.g. [2]
2. While the performance improvement exists, it seems to be only a marginal improvement.
3. RL is always sensitive to hyperparameters. Introducing new regularizers likely increases the search space for tuning hyperparameters.
4. Jacobian regularization does not guarantee the boundedness of the Lipschitz constant. The assumption might not be correct. Furthermore, if $J$ becomes large, the exponential term in the bound will make the bound too loose.

### Questions
1. How large is the actual Lipschitz constant of the final policy? Could you quantify this empirically to show the correlation?
2. It seems that we are just learning a robust policy that is robust to perturbation in actions rather than improving the training stability. Is this fixing training dynamics or just learning smoother policies?
3. In line 073, you did not modify the name of your method.
4. In Table 4, why do you think that beta sampling is improving the performance significantly?
5. What is the overhead of computing the Jacobian?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
To address the issue in previous Flow-matching methods, where point-to-point optimization often neglects the global properties and smoothness of the generative path, this work introduces a regularizer based on partial differential equations (PDEs) to constrain the learning process. Additionally, a Beta-distributed time sampling strategy is proposed to improve the optimization efficiency of this regularizer. Experimental results on multiple offline RL benchmarks, including D4RL, Adroit, and OGBench, demonstrate the effectiveness of the proposed approach. Furthermore, the authors provide theoretical justification for the method.

### Strengths
1. The manuscript exhibits a coherent and logical structure, employs precise and formal language, and presents a clearly defined motivation that is readily understandable.
2. The study includes a wide range of rigorous experiments, offering insightful analysis of the outcomes and thorough evaluation of the model’s structure and hyperparameters.
3. The paper is grounded on a solid theoretical foundation, and the provided theoretical analysis offers principled support for the effectiveness of the proposed model.

### Weaknesses
1. The experimental baselines do not include comparisons with diffusion-based methods such as Decision Diffuser, Diffuser, or Diffuser-Lite, which would provide a more comprehensive evaluation.
2. The manuscript does not discuss the limitations of the proposed approach, which is important for understanding its scope and potential drawbacks.
3. A placeholder remains in the Introduction: the third point summarizing the contributions still contains “[Your Method Name]” and should be properly updated.

### Questions
1. Could the authors provide the performance results of their method on the Kitchen environment?
2. The proposed approach introduces a relatively large number of hyperparameters. While the experiments analyzing hyperparameters in the paper are appreciated, could the authors provide practical guidelines or recommendations for selecting hyperparameters to facilitate rapid adaptation to new tasks?
3. Have the authors considered alternative strategies to reduce the model’s sensitivity to hyperparameter choices?

### Soundness
4

### Presentation
3

### Contribution
3
