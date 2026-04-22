# BNPO: Beta Normalization Policy Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Recent studies, including DeepSeek-R1 and Kimi-k1.5, have demonstrated that reinforcement learning with rule-based, binary-valued reward functions can significantly enhance the reasoning capabilities of large language models. These models primarily utilize REINFORCE-based policy optimization techniques, such as REINFORCE with baseline and group relative policy optimization (GRPO). However, a key limitation remains: current policy optimization methods either neglect reward normalization or employ static normalization strategies, which fail to adapt to the dynamic nature of policy updates during training. This may result in unstable gradient estimates and hinder training stability. To address this issue, we propose Beta Normalization Policy Optimization (BNPO), a novel policy optimization method that adaptively normalizes rewards using a Beta distribution with dynamically updated parameters. BNPO aligns the normalization with the changing policy distribution, enabling more precise and lower-variance gradient estimation, which in turn promotes stable training dynamics. We provide theoretical analysis demonstrating BNPO's variance-reducing properties and show that it generalizes both REINFORCE and GRPO under binary-valued reward settings. Furthermore, we introduce an advantage decomposition mechanism to extend BNPO’s applicability to more complex reward systems. Experimental results confirm that BNPO achieves state-of-the-art performance among policy optimization methods on reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a dynamic, beta-distribution based normalization technique for the Advantage function that is used in a policy gradient context. Their method is specifically designed for binary reward settings. The authors call this method BNPO and use it train LLMs which are then evaluated on several math benchmarks.

### Strengths
- The empirical results show promise, however, as mentioned in the weaknesses they need to be presented more precisely
- The authors give an extensive background section, making even a reader that is unfamiliar with the topic able to follow the contents of the paper. However, in case you might need extra space to properly include the reviewers feedack, I think you could prune the beta-distribution section or move it to the appendix. I think that that is more or less basic math knowledge and anyone unfamiliar with it, could quickly read it up in a math textbook.
- I quite like the general idea and it is quite elegant that one is able to unifying REINFORCE and GRPO in one framework.

### Weaknesses
- Insufficient related work: Briefly touch in a broader scope of the application of normalization in RL. E.g. observation, reward normalization (i.e. before being processed by the training algorithm) and for example SAC (Soft Actor-Critic Algorithms and Applications, Haarnoja) dynamically normalizes its entropy coefficient.
- Missing Confidence Intervals for Empirical Results
- I am not convinced that the average in Table 1 is sufficient to say that BNPO is best overall. At the end of the day, BNPO is only best only in AMC23 and the difficulty of squeezing out an improvement differs from dataset to dataset. For example, the 0.2 gain of REINFORCE over BNPO in AIME2025 is percentagewise as significant as the lead GRPO has over BNPO in MATH500, however it almost does not contribute to the average value. For transparency I think one would also have to include different metrics aside from the average such as avg % increase.

### Questions
- Please be more clear about your hyperparamter choice in "Metrics" you say that they are based on the best average performance? Of which method? Then later you say that for fairness all methods use the same hyperparamter which would not be fair if they are tuned for one specific method
- Could you include runtime comparisons? BNPO uses Monte-Carlo samples to compute its parameters alpha and beta which might hamper the runtime. Please elaborate.
- While you give a good intution that p(q) follows a beta-distribution, to substantiate this claim, it would be good to actually plot the empirical distribution of some ps and show that it can be well-fitted with some beta-distribution
- I do not get Lines 256-259. To reduce to RLOO, F_N would have to be parametrisable to be any constant, however the only constant F_N can be is 1?
- Are there any empirical validations for the claim in Lines 297
- Training stability: Isnt it possible to just directly compute the policy variance instead of using a proxy?

Minors:
- Missing space in Line 134 before sigma and after 'end'
- Missing formal problem description. I think a multi-armed bandit would be the problem formalization here. At least mention it.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper examines reward normalization methods for REINFORCE-type policy gradient algorithms, primarily in the context of binary reward problems. The paper proposes  Beta normalization, which normalizes the center reward $R - p(q)$ by a Beta distribution PDF with adapted parameters, where $p(q) = E[R|q]$ is the success probability for question $q$. Some conceptual justification is given, and experiments using small language models on MATH are conducted to show the performance of the new normalization method.

### Strengths
1. The paper proposes a complete workflow for normalizing the reward for policy gradient with a Beta distribution, including estimating the distribution parameters.
2. It is interesting that with a different parameter setup, Beta normalization recovers REINFORCE and GRPO.
3. The proposed approach is empirically evaluated on multiple datasets and multiple models, with both pass rate and gradient norm.

### Weaknesses
The conceptual aspect of the proposed method, in many aspects, is somewhat flawed, which limits the method in general RL problems. 

1. The correctness of the gradient estimator is questionable. After the new normalization, the gradient estimator is no longer unbiased to the exact policy gradient. There is no conceptual justification that the new estimator would achieve a solution close to the optimal policy of the original problem for general problems.

2. The theorem in this paper rests on an unrealistic (even erroneous) independence assumption and is vacuous. The core variance-reduction justification assumes $\nabla_\theta \log \pi(o|q)$ is uncorrelated with $\frac{R(q,o) - p(q)}{f_N(p(q); \alpha, \beta)}$. This is generally false in policy gradient because both terms have explicit dependence on $\pi_\theta$. Therefore, the proof does not convincingly justify the claimed minimum variance and stability.

3. The modeling of success probability $p(q)$ over the population of questions with a single Beta distribution is fragile, especially when the quantity is non-stationary and heavily depends on the policy $\pi_\theta$. There is no justification, both theoretically and empirically, for how the modeling is correct, and what would happen if the modeling fails. It is largely unclear why using a Beta distribution density function to normalize would result in a better performance, and under what circumstances it will and will not.

4. Normalizing the reward by a density function could lead to instability. The estimated $p(q)$ may correspond to a small density, which amplifies the noise in advantage estimation. Therefore, the gap between this possibility and the reported stability in empirical evaluation is unclear.

The empirical evaluation is also limited to demonstrating the performance of the proposed method in general applications.

1. The empirical evaluation misses the core evaluation of the value in the proposed Beta normalization. The authors did not demonstrate how the empirical $p(q)$ fits the beta distribution across training periods, or how the curves of gradient variance vary across time compared to baselines, or how sensitive the performance is to mis-specified $(a,b)$ in estimation, especially when the sampled $p(q)$ gives a small density. 

2. The reported score of the proposed method is only marginally higher than some benchmarks, and not the best in several individual benchmarks. Moreover, with Beta normalization, the method still performs quite poorly in slightly more complex tasks such as AIME. It is unclear whether the normalization benefit (if there is any) could scale to more complex IMO-style tasks, or modern large models with many more parameters, or even to more diverse reasoning tasks, such as coding and science reasoning. The marginality and inconsistency do not translate to a convincing motivation for using this method in reality for general RL problems.

### Questions
see weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Beta Normalization Policy Optimization (BNPO), a novel method for policy optimization in reinforcement learning (RL). The proposed method addresses a critical limitation in current RL policy optimization techniques—namely, their inability to dynamically adjust reward normalization as the policy evolves during training. BNPO utilizes a Beta distribution to adaptively normalize rewards, improving gradient stability and reducing variance in gradient estimates. Theoretical analysis and experimental results demonstrate that BNPO performs better than existing methods, including REINFORCE, GRPO, and REINFORCE++, especially on reasoning tasks.

### Strengths
1. BNPO introduces a dynamic reward normalization technique using the Beta distribution, offering a novel solution to reward normalization issues in reinforcement learning. This is particularly important as existing methods use static normalization, which doesn’t adapt as training progresses.
2. The paper provides a solid theoretical analysis showing that BNPO effectively reduces gradient variance and enhances the stability of training. The detailed proof and derivations offer a clear understanding of the method's benefits.
3. The experiments on large-scale language models and reasoning tasks clearly show that BNPO outperforms other state-of-the-art optimization methods, including REINFORCE, GRPO, and REINFORCE++.
4. The advantage decomposition mechanism introduced in the paper allows BNPO to handle more complex reward systems, extending its applicability beyond binary-valued rewards.
5. Experimental results demonstrate that BNPO leads to more stable training dynamics compared to other methods, as evidenced by consistent gradient norms and low gradient variance, which is crucial for large model training.

### Weaknesses
1. While BNPO works well for binary-valued rewards, the extension to multi-valued or continuous rewards is mentioned but not thoroughly explored in the paper. More detailed analysis and experiments on this extension would strengthen the claim of BNPO's general applicability.
2. The adaptive nature of BNPO, while theoretically sound, may introduce additional computational overhead in estimating the parameters of the Beta distribution during training. The paper could discuss the trade-off between the benefits of the method and its computational costs more explicitly.
3. The paper compares BNPO primarily with methods like REINFORCE, GRPO, and REINFORCE++. It would be valuable to see comparisons with a broader range of reinforcement learning algorithms or real-world scenarios beyond reasoning tasks.
4. There are no detailed ablation studies presented to demonstrate the individual contributions of different components of BNPO, such as the Beta normalization and advantage decomposition mechanisms. This would provide clearer insights into the importance of each part of the method.
5. The method’s performance on larger, more complex datasets and models is promising, but scalability for extremely large-scale models or highly complex environments needs further investigation.
6.  Although BNPO shows strong results on reasoning tasks, its generalization to other domains or tasks with different reward structures and dynamics could be further tested. This would provide a broader validation of the method’s robustness.

### Questions
1. How does BNPO perform when applied to non-binary reward structures, especially in tasks with multi-modal or continuous reward functions? Could there be challenges in adapting the Beta distribution in such cases?
2. Can BNPO maintain its performance and stability with significantly larger models, such as those with billions of parameters, without encountering computational bottlenecks?
3. How do the hyperparameters in BNPO, particularly the Beta distribution parameters (α, β), interact with other optimization hyperparameters like learning rate and batch size? Is there a risk of overfitting or sensitivity to these parameters?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes reducing the variance of RL gradients by introducing a novel normalization, normalized by a beta distribution. The parameters of the beta distribution are dynamically computed according to the reward mean and variance, in order to minimize the gradient variance. The modelling unifies the existing methods, including GRPO and REINFORCE, by showing them taking different distribution parameters. Additionally, it extends to multiple-reward problems through an advantage decomposition mechanism, which normalizes each reward, such as format reward and accuracy reward, separately. The algorithm is tested on four math benchmarks, showing comparable performance to baselines.

### Strengths
The paper addresses a practical problem aimed at improving the stability of policy gradient by reducing gradient variance. Meanwhile, it provides an elegant theoretical derivation of the algorithm.

### Weaknesses
One of the baselines, ReMax, already provides a straightforward method for computing a baseline and stabilizes the training as shown in Figure 2. The proposed method requires the estimation of two parameters, which can introduce extra estimation biases and variances.

The empirical testing is limited to four MATH benchmarks and Qwen base models only. Also, experimental analysis on the advantage decomposition mechanism is missing.

Li, Z., Xu, T., Zhang, Y., Lin, Z., Yu, Y., Sun, R., & Luo, Z. Q. (2023). Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505.

### Questions
1. What is the variance and bias of estimating the parameters $\alpha$ and $\beta$? How will it influence the gradient estimator? As shown in Figure 3, the estimation of $\alpha$ is very unstable.

2. Are there any experiments on the advantage decomposition mechanism?

### Soundness
3

### Presentation
3

### Contribution
2
