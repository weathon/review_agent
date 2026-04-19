# MoMA: Model-based Mirror Ascent for Offline Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Model-based offline reinforcement learning methods (RL) have achieved state-of-the-art performance in many decision-making problems thanks to their sample efficiency and generalizability. However, prior model-based offline RL methods in the literature either demonstrate their successes solely through empirical studies, or provide algorithms that have theoretical guarantees but are hard to implement in practice. To date, a practically implementable algorithm for model-based offline RL with theoretical guarantees is still lacking. To fill this gap, we develop MoMA, a model-based mirror ascent algorithm with general function approximations under partial coverage of offline data. Iteratively, MoMA conservatively estimates the value function by a minimization procedure within a confidence set of the transition model in the policy evaluation step, then updates the policy with general function approximations instead of commonly-used parametric policy classes in the policy improvement step. Under some mild assumptions, we establish theoretical guarantees of the proposed algorithm by proving an upper bound on the suboptimality of the returned policy. The effectiveness of the proposed algorithm is demonstrated via numerical studies.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a model-based offline reinforcement learning algorithm with general function approximation. It aims to fill the gap between the theoretical guarantees and the practical implementation. The algorithm proposes a model-based mirror ascent algorithm with general function approximation under partial coverage of offline data. The paper shows the theoretical guarantees of the algorithm, which is supported by numerical studies.

### Strengths
The paper provides a solid study on the model-based offline reinforcement learning algorithm. The main contribution comes from two side: 

1) Comparing to previous model-based offline RL algorithms, this paper provides an practically implementable algorithm and provides an empirical study on it.

2) Comparing with model-free offline RL algorithms, the paper studies a case with weaker assumption and shows a faster suboptimality.

The theoretical and the empirical study in the paper is solid. Moreover, the results of the paper is presented in a

### Weaknesses
My major concern on the paper comes from the contribution of the paper. Algorithm 1 is similar to Uehara and Sun. While Algorithm 2 is a more interesting part, its idea is still similar to previous work, especially on the conservative estimation and the partial coverage. Moreover, I have several concerns on the implementation of the algorithm:
- My first concern is on the PD algorithm (3) and (4) for finding the estimation of the transition model to construct the conservative estimation of the Q-function. In general, the parameterization of the transition kernel is non-convex, and the loss function is estimated empirically. Thus, it could be hard to find the global optimizer, or find a good estimation of the global minimizer.
- The above problem also exists in (6) and (7) for estimating $\beta$.

### Questions
1. Is there a theoretical guarantee on the PD algorithm in (3) and (4), especially in the case of general function approximation and empirical estimation of the loss function?
2. The paper presents the comparison with several other offline RL in Section 7. For table 1, is there also a comparison of MoMA and other SOTA algorithms (like in Table 2) on the synthetic dataset? 
3. For table 2, there seems to be a large difference on the scores of the different algorithms. Is there any insight on why such a large difference occurs?
4. The definition of $\mathcal{P}$ seems missing.
5. The error rate in Theorem 1 involves $\epsilon_{est}$, which, however, depends on $n$ as defined in (8). What is the dependence on $\epsilon_{est}$ in terms of $n$?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper:

(1) proposes a model-based mirror descent offline RL algorithm and its corresponding practical version via Lagrangian multiplier and function approximation.

(2) provides theoretical guarantee of the proposed algorithm.

(2) conducts some empirical experiments to validate the performance of the proposed algorithm.

### Strengths
(1) The proofs are rigid.

(2) Apart from theoretical guarantees, there are also numerical results.

### Weaknesses
Overall I feel this work is not novel or significant enough:

(1) For Algorithm 1, the idea of constructing a confidence set over the model and use pessimism for policy improvement has been applied in the literature of offline RL, such as [1].

(2) For Algorithm 2, there also have been papers [2] utilizing Lagrangian multipliers to get rid of the constraint of confidence set in the theoretical algorithms and get computationally efficient algorithms. The method of function approximation in Algorithm 2 also has been applied in [3].

(3) The empirical performance of the proposed algorithm is not impressive. In Table 2, RAMBO, IQL and CQL all get better performance (averaged over all tasks) and thus I think the proposed algorithm does not have much significance in the empirical studies either.

In summary, I think this work simply combines the existing techniques and results in the literature and is kind of incremental. 

[1] Uehara, Masatoshi, and Wen Sun. "Pessimistic model-based offline reinforcement learning under partial coverage." arXiv preprint arXiv:2107.06226 (2021).

[2] Rigter, Marc, Bruno Lacerda, and Nick Hawes. "Rambo-rl: Robust adversarial model-based offline reinforcement learning." Advances in neural information processing systems 35 (2022): 16082-16097.

[3] Guanghui Lan. Policy optimization over general state and action spaces. arXiv preprint arXiv:2211.16715, 2022.

### Questions
(1) The paper claims that it does not require Bellman completeness and only needs model realizability. However, the literature have shown that when a model class that contains the true MDP model is given, value-function classes that satisfy a version of Bellman-completeness can be automatically induced from the model class [4]. This implies that model realizability is even stronger than Bellman-completeness.

(2) The paper claims that the size of the function can be arbitrarily large. However, in Theorem 1 (the performance of Algorithm 2), the sample complexity clearly depends on the size of the function class $|\mathcal{F} _ {t,i}|$. In addition, I believe both the performance of Algorithm 1 and 2 depends on the size of the model class, and model classes can be even larger than function classes.

(3) The paper does not need a parametric policy class. However, in general the computation complexity of the optimization problem Equation 7 will depend on the size of the state space, which can be infinite.

(4) The authors claim that Theorem 1 characterizes the performance of Algorithm 2, but it seems not. Theorem 1 assumes the existence of the confidence set of models and picks the most pessimistic model from such a confidence set while in Algorithm 2 you simply run primal-dual gradient descent-ascent and do not have a confidence set.

[4] Chen, J. and Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning, pages 1042–1051. PMLR.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an offline RL algorithm, MOMA,  that offers practical implementability alongside theoretical guarantees.  MoMA uses a model-based mirror ascent approach with general function approximations, operating under partial coverage of offline data. The algorithm iteratively and conservatively estimates value functions and updates policies, moving beyond the conventional use of parametric policy classes. With mild assumptions, MoMA’s effectiveness is theoretically assured by establishing an upper bound on the policy’s suboptimality, and practical utility is confirmed through numerical studies.

### Strengths
This paper proposes a new offline RL algorithm with pessimism.

### Weaknesses
1. The contribution of this work seems exaggerated. In particular, the authors claim that the algorithm enjoys theoretical guarantees and practical implementation. But in fact, the algorithm with the theoretical result is **different** from the practically implemented version. In particular, the theoretically analyzed algorithm involves pessimistic model learning, while the practical version uses a regularized form. With significantly different algorithms, the claim does not hold. 

2. In terms of the theoretically sound version (Algorithm 1), the theoretical novelty is limited. The analysis is based on pessimistic model learning combined with policy mirror descent. The combination seems quite direct given existing works. In particular, in the proof of Theorem 2, the regret decomposition in (15)--(17) is standard analysis of a pessimism-based algorithm. The first term of (18) is standard analysis of policy mirror descent, and the second term of (18) is an application of simulation lemma. Similar analysis has also appeared in the RL literature, e.g., Provable Benefits of Actor-Critic Methods for Offline Reinforcement Learning. 

3. The practically implemented version of algorithm is also similar to some existing algorithms. For example, "Bayesian Optimistic Optimization: Optimistic Exploration for Model-based Reinforcement Learning" also studies a regularized version of policy optimization algorithm, but for online RL. In addition, MOPO and MoRel has practical implementations of pessimism-based algorithms. 

4. The implemented version of algorithm is only tested on three D4RL tasks. It would be great to have more extensive experiments.

### Questions
1. How to implement the policy update when there is a normalization factor? 
2. How do you update the multiplier $\lambda$ in the experiments? 
3. Is it possible to unify these two different versions of algorithms?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a conservative and practical model-based offline RL algorithm that alternates between pessimistic value estimation and policy update via mirror ascent. Theoretical guarantees are provided for the algorithm under partial data coverage and some experimental evaluations are provided.

### Strengths
- The paper studies an important problem, which is practical and theoretically-founded model-based offline RL with general function approximation.
- The proposed algorithm is new, implementable, and enjoys theoretical guarantees.

### Weaknesses
- One of the main contributions is stated as “In contrast to model-free RL literature which heavily relies on the assumption of Bellman completeness, we propose novel theoretical techniques for offline model-based policy optimization algorithms that are free of the assumption”  In general, model-based methods (unless additional variables are introduced) do not require Bellman completeness assumption because assuming a realizable model is stronger; see e.g. [1], which shows that model realizability subsumes completeness assumptions.
- It is stated that “To our knowledge, this work is the first study that provides a practically implementable model-based offline RL algorithm with theoretical guarantees under general function approximations.” There are prior works that offer implementable model-based offline RL algorithms with optimal statistical rates such the model-based version of the algorithm in [2]. The mentioned algorithm also does not require the difficult step of minimizing within a confidence set of transition models. Another example algorithm is ARMOR [4].
- Although the work is focused on general function approximation, it requires the concentrability definition with bounded policy visitation and data distribution ratio for every state and action. This is a stronger assumption compared to the Bellman-consistent variant such as in [3].
- Theory does not seem to be particularly challenging and/or offer new insights or techniques.
- The experimental section is weak. In particular, the synthetic data experiments only compare MoMA with standard natural policy gradient and model-free FQI, none of which include any form of conservatism/pessimism, and the results are expected. It would be good to see comparison with other pessimistic model-based methods. Additionally, comparison with the work of Uehara & Sun 2021 when combined with the Lagrangian approach of this work would be useful. For the D4RL benchmark, comparison is only provided for a small subset of datasets and only baseline model-free offline RL methods. No comparison is provided with ARMOR and ATAC.

**References:**

[1] Chen, Jinglin, and Nan Jiang. "Information-theoretic considerations in batch reinforcement learning." In International Conference on Machine Learning, pp. 1042-1051. PMLR, 2019.

[2] Rashidinejad, Paria, Hanlin Zhu, Kunhe Yang, Stuart Russell, and Jiantao Jiao. "Optimal Conservative Offline RL with General Function Approximation via Augmented Lagrangian." In The Eleventh International Conference on Learning Representations. 2022.

[3] Cheng, Ching-An, Tengyang Xie, Nan Jiang, and Alekh Agarwal. "Adversarially trained actor critic for offline reinforcement learning." In International Conference on Machine Learning, pp. 3852-3878. PMLR, 2022.

[4] Bhardwaj, Mohak, Tengyang Xie, Byron Boots, Nan Jiang, and Ching-An Cheng. "Adversarial model for offline reinforcement learning." arXiv preprint arXiv:2302.11048 (2023).

### Questions
- Comparison with the references above, both in terms of technique and empirical performance.
- Clarifying challenges and technical contributions.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
