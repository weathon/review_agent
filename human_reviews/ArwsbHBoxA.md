# Nearly Optimal Algorithms for Contextual Dueling Bandits from Adversarial Feedback

- Decision: Reject
- Scores: 5, 6, 5, 5

## Abstract
Learning from human feedback plays an important role in aligning generative models, such as large language models (LLM). However, the effectiveness of this approach can be influenced by adversaries, who may intentionally provide misleading preferences to manipulate the output in an undesirable or harmful direction.
To tackle this challenge, we study a specific model within this problem domain--contextual dueling bandits with adversarial feedback, where the true preference label can be flipped by an adversary. We propose an algorithm namely robust contextual dueling bandits (\algo), which is based on uncertainty-weighted maximum likelihood estimation.  Our algorithm achieves an $\tilde O(d\sqrt{T}+dC)$ regret bound, where $T$ is the number of rounds, $d$ is the dimension of the context, and $  0 \le C \le T$ is the total number of adversarial feedback. 
We also prove a lower bound to show that our regret bound is nearly optimal, both in scenarios with and without ($C=0$) adversarial feedback. To the best of our knowledge, our work is the first to achieve nearly minimax optimal regret for dueling bandits in the presence of adversarial preference feedback.
Additionally, we conduct experiments to evaluate our proposed algorithm against various types of adversarial feedback. Experimental results demonstrate its superiority over the state-of-the-art dueling bandit algorithms in the presence of adversarial feedback.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors study contextual dueling bandits under adversarial feedback. They propose an algorithm, RCDB, which utilizes uncertainty-weighted maximum likelihood estimation (MLE). The algorithm achieves a regret of $\tilde{O}(d\sqrt{T} + dC)$, where $C$ is the total number of adversarial feedback instances. They also provide a regret lower bound, showing that the achieved regret bound is nearly optimal. Finally, they conduct numerical experiments to demonstrate the superiority of the proposed algorithm over state-of-the-art dueling bandit algorithms.

### Strengths
1. They study dueling bandits under adversarial feedback, which has not been studied before.
2. The achieved regret bound is near optimal.
3. They demonstrate their algorithm using synthetic datasets.

### Weaknesses
1. The weighted version for the estimator of dealing with corruption was first proposed in He et al. 2022. Their method is highly rely on this method.

2. For the unknown number of adversarial feedback, it requires the information of upper bound of C.

3. There seems to be a lack of novelty in theoretical analysis.

### Questions
1. The main concern is regarding theoretical contribution. Could you explain the nontrivial theoretical contribution?
2. Could you provide a more detailed formulation for $\sigma$?

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
3

### Summary
The paper studied contextual dueling bandits with adversarial feedback and proposed an uncertainty-weighted MLE algorithm. The authors also provided theoretical upper bound and lower bound for regret. And they also run experiments to verify their findings.

### Strengths
It is good to provide both upper bound and lower bound for the problem of contextual dueling bandits with adversarial feedback. And they also provide both theoretical and experimental results.

### Weaknesses
See questions for more details.

### Questions
1. In Assumption 3.1., you assume the reward is a linear model. Can your method extend to non-linear cases?
2. How to choose the value of the regularization parameter in algorithm 1 specifically?
3. In the experiment setup, you choose the sigmoid function for the preference model. How about the other preference model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a new approach for contextual dueling bandits under adversarial feedback by proposing the Robust Contextual Dueling Bandits (RCDB) algorithm. RCDB incorporates uncertainty-weighted maximum likelihood estimation (MLE) to handle adversarial interference in preference labels, aiming to minimize regret even in adversarial scenarios. The algorithm achieves a near-optimal regret bound that is robust to adversarial feedback, showing both a baseline optimal performance in uncorrupted cases and effective scaling with adversarial interference. Extensive experiments validate RCDB’s superiority over existing dueling bandit algorithms under various adversarial conditions, underscoring its potential for applications that rely on preference-based learning models.

### Strengths
+ The problem is timely and important
+ The proposed algorithm achieves (nearly) optimal performance bounds
+ Experiments are also provided

### Weaknesses
- Some technical novelty needs to be clarified

### Questions
1. It seems to me that the main algorithm is a somewhat extension of the non-dueling counterpart in He et al 2022, i.e., the same uncertainty weight and the same update rule of the weights. Except for the possible non-linearity in dueling (logistic) bandit (which can be handled by existing techniques), it seems to me that the proposed algorithm and its analysis largely follow from prior work. 

2. As the dependence of \kappa is a key factor in the logistic bandit, can the authors comment on the optimality of \kappa in the regret upper bound (for both non-corrupted and corrupted terms)?

3. If I understand it correctly, there are two commonly used corruption models. One is the total budget model used in the current paper and the other is the strong corruption model from robust statistics (see Definition 3.1 in [R1). In general, these two models are not comparable. But, given the fact that under the dueling bandit setting, the label is only 0 or 1, it seems to me that these two models are somewhat comparable, especially given the fact the current paper considers the strong adversary model where the adversary can observe the actions. It would be good to see some comments on the comparison of the two commonly used corruption models, i.e., if they are equivalent or one implies the other one. This kind of discussion is useful, since it might make the current results even stronger in the sense that it can be used to talk about some results in another corruption model. 



[R1] Zhang, Xuezhou, et al. "Robust policy gradient against strong data corruption." International Conference on Machine Learning. PMLR, 2021.

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
This paper studies the contextual dueling bandit problem with adversarial preference feedback in which adversaries may intentionally provide misleading preference feedback. The goal is to design an algorithm that is robust to these adversarial manipulations and has sub-linear regret, i.e., the sum of the difference between two times the maximum reward and total reward of a selected pair of arms.

The authors propose an algorithm named RCDB (Robust Contextual Dueling Bandits) for this problem, which uses uncertainty-dependent weights in the maximum likelihood estimator. They have shown that RCDB achieves a nearly optimal regret bound as long as the number of adversarial feedback, $C=O(\sqrt{T})$. Finally, the authors have shown that RCDB outperforms existing algorithms with various adversarial strategies in different synthetic scenarios, validating its robustness and effectiveness.

### Strengths
**The following are the strengths of the paper:**
1. This paper considers a contextual dueling bandit problem with adversarial preference feedback, which is motivated by real-world applications like training LLMs using RLHF.

2. The authors propose a contextual dueling bandi algorithm RCDB that handles adversarial manipulations and then show that RCDB achieves a nearly optimal regret bound.

3. Finally, the authors have demonstrated the different performance aspects (comparing the regret bound of existing dueling bandit algorithms and regret vs the amount of adversarial feedback) of the proposed algorithms on synthetic problem instances.

### Weaknesses
**The following are the weaknesses of the paper:**
1. The proposed algorithm RCDB assumes the reward function is linear, which may limit its applicability to real-world settings where the reward functions are non-linear. Even though one can assume a feature map ($\phi$) exists for which reward is a linear function. However, a feature map must be known upfront (Assumption 3.1), which may not always be possible in real-world applications. 

2. The performance of RCDB depends on accurately tuning the tolerance threshold $\alpha$ and confidence bound $\beta$, which may vary depending on the number of adversarial feedback. In practice, knowing the amount of adversarial feedback upfront may be impossible (as they are adversarial).

3. It is unclear how to distinguish among the following cases:

    1. Two actions have similar rewards; hence, the preference probability is close to 0.5.
    2. A weak labeler who gives bad quality preference feedback, which may be common in practice.
    3. Adversarial labeler who manipulates the feedback

4. It is not clear why setting weights as defined in Eq. (4.3) is a good approach to dealing with adversarial feedback because it is possible that there is no adversarial feedback in data and still has a high uncertainty.

### Questions
Please address the weaknesses of the paper. I have a few more questions/comments:
1. Is the maximum likelihood estimator $\theta_t$ biased due to the small weights of some data without no adversarial manipulation, which is possible as $ ||\phi_i|_ {\Sigma_i^{-1}}$ may be large in the initial rounds.

2. Is there a way to set the weights $(\alpha)$ without knowing the amount of adversarial feedback?


**Minor comment:**
1. Line 248 - Line 293 (expect algorithm description) may not be needed as this is not critical, but it can n create confusion due to the use of Taylor approximation. 

2. The authors can check the following paper that considers non-linear reward functions in contextual dueling bandits:
Verma et al., [Neural Dueling Bandits](https://arxiv.org/pdf/2407.17112).

I am open to changing my score based on the authors' responses.

### Soundness
3

### Presentation
2

### Contribution
3
