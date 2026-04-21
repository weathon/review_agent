# Aligning With Human Values Without Revealing Human Judgements

- Avg Score: 4.50
- Decision: Reject
- Scores: 3, 6, 6, 3

## Abstract
With the increasing ubiquity of large language models it has become crucial to ensure guarantees for models trained to be aligned with human values to avoid leaking information on the human judgements that have been provided to the algorithm. To target this issue we focus on the problem of alignment via reinforcement learning from human preference rankings, subject to the constraint of 
not revealing any information on the human data used to align the model. To achieve this, we analyze $(\epsilon,\delta)$-DP for both the Bradley-Terry-Luce (BTL) model and the Plackett-Luce (PL) model. We introduce a theoretically founded algorithm for learning rewards from human rankings that achieves this objective without leaking the human rankings. We further demonstrate that the privately learned rewards can be used to train policies achieving statistical performance guarantees that asymptotically match the best known algorithms in the non-private setting, which are in some cases minimax optimal. Strikingly, our analysis and our results reveal that it is possible to obtain the same model performance without any trade-off on the protection of the human judgments, and our paper provides the first algorithms that can achieve provable privacy of human judgements, while still producing aligned models with optimal performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper investigated RHLF from K-wise comparisons under central differential privacy. The authors designed private algorithms for the pessimistic method and provided the upper bounds for the sub-optimality gap.

### Strengths
The paper gives some theoretical results for RLHF under DP.

### Weaknesses
1. The literature review is not comprehensive enough, see Questions part for more details.
2. There are some typos in important equations and some concepts are not clear, see Questions part for more details.
3. There are no lower bounds and experiments.

### Questions
1. In lines 110-111, the notation is confusing for $\sum_{j=1}^K \sum_{j=k+1}^K$. There are two "j" indexes.
2. The authors claim that the paper provides the first algorithms that can achieve provable privacy of human judgments. However, [1] studied RLHF under both local DP and central DP. And the authors didn't cite the paper and didn't compare the results with theirs.
3. The concept of DP in section 2.1 is not clear for the corresponding case of RLHF. What is the output of algorithm $\mathcal{A}$? Why did you choose approximate DP, not pure DP? Why did you choose central DP, not local DP?
4. The authors should cite earlier work like [2] for the pessimistic method.
5. In algorithm 1, why did you use objective perturbation in Line 4 and output perturbation for DP together? See section 5.2 in [1] for objective perturbation to guarantee DP. Why did you choose the regularizer parameter $\alpha$ to be related to private parameter $\epsilon$.
6. What is the dependence on $B$ and $L$ in your final suboptimality gap?

[1]. Chowdhury, Sayak Ray, Xingyu Zhou, and Nagarajan Natarajan. "Differentially private reward estimation with preference feedback." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

[2]. Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In International Conference
on Machine Learning, pages 5084–5096. PMLR, 2021

### Soundness
2

### Presentation
2

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
The paper proposes a method for aligning large language models with human values while satisfying differential privacy requirements. The authors achieve minimax-optimal performance, ensuring the security of private data. Their results demonstrate that even under privacy constraints ($(\epsilon,\delta)$-DP), model alignment with human values is effective in contextual bandits, maintaining performance within $O(\sqrt{d/n})$ of the optimal policy when compared with non-private methods. This approach is then extended to general MDPs, where human preferences are collected over trajectory pairs. The algorithm preserves privacy while matching the performance of non-private methods.

### Strengths
Please find the strengths below:
1. The paper introduces differential privacy to model alignment while preserving similar performance with respect to $d$ and $n$.
2. The paper is not limited to $K=2$ but also provides results for general $K$, which is closer to the setting of RLHF.
3. The results are also extended to general MDPs, matching previous performance levels without satisfying differential privacy constraints.

### Weaknesses
Please find the weaknesses below:
1. The paper does not include any numerical experiments or experiments on real datasets. While the theoretical results for both contextual bandits and MDPs are asymptotic to the optimal policy, it would be valuable to observe the overhead of differential privacy in practice, especially when $d$ is small.
2. Although the performance matches that of methods without differential privacy in both contextual bandits and MDPs settings, the paper does not provide a lower bound. It would be beneficial to include a lower bound with respect to $\epsilon$ and $\delta$.
3. The paper is limited to the Plackett-Luce model; it would be interesting to explore whether similar results could be derived for other comparison models, such as the Thurstone-Mosteller model.

### Questions
Refer to the weaknesses section above, and find the additional questions below:
1. In Section 4.2, the paper discusses the case of $k$-wise comparison. Is it possible for agents to make comparisons involving different numbers of items?
2. Is $O(\sqrt{d/n})$ optimal for both contextual bandits and MDP settings with and without differential privacy? If so, it would be helpful to mention these results in the main paper.
3. In the MDP setting in Section 5, is there a specific reason why the results focus on $K=2$ rather than a general $K$?

### Soundness
3

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
The paper considers the problem of differentially private alignment via reinforcement learning from human feedback, motivated by the practical concern of trying to preserve privacy of the annotators preferences. The authors consider two main settings in this general framework: the first is a linear contextual bandit setting, and the second is a more general finite horizon MDP setting. The human feedback in the former setting consists of complete orderings over the different actions, assumed to have been drawn from the Plackett-Luce model, whereas the feedback in the latter setting is assumed to consist of a pairwise choice between two trajectories, assumed to have been drawn from the Bradley-Terry-Luce model, which is a special case of the Plackett-Luce model for pairwise comparisons. The objective is to first learn the underlying parameters of these two models (i.e. estimate the underlying human preferences) in a differentially private manner (from given offline data consisting of these trajectories and corresponding ranked preferences), and then use these estimated parameters to then learn a policy that achieves reward that is comparable to to the one achieved by a policy that knew the underlying preference vector exactly. In this setting, the authors show that it is possible to compute an $(\epsilon,\delta)$-differentially private estimator for the underlying preferences (as well as the data-covariance matrix) that for any constant $\epsilon$ and inverse polynomial $\delta$, is such that the pessimistic policy optimization algorithm given these estimated, private parameters as input, achieves reward that is minimax optimal. In other words, the additional loss in reward incurred due to enforcing $(\epsilon,\delta)$-DP is order-wise identical to that achieved by the non-private MLE estimators. These results follow by a fairly direct application of the Gaussian mechanism to the MLE estimation of the PL parameters, and data-covariances. It is important to note that these results only hold in the central model of differential privacy, where there is a central trusted curator that holds the non-private data.

### Strengths
In my opinion, the paper is quite well written and easy to follow. The problem of privacy more generally is an important area of consideration, so the research is also of practical significance. The algorithms are pretty simple which should make them applicable in practice. The results are also order wise optimal, and show that it is possible to achieve differential privacy without much asymptotic loss in reward.

### Weaknesses
Other than the fact that these results only hold assuming a linear reward function (which to be fair, the authors have also acknowledged), the only other concern I have is novelty. Differential privacy is not my primary area of work, but it seems to me that this result is obtained via a pretty straightforward application of the gaussian mechanism which I believe to be one of the standard methods in the DP world. That being said, I am no means an expert in this area, and will defer to other more knowledgeable reviewers judgement regarding the novelty and impact of this work.

### Questions
If the authors can elaborate what makes this result novel, and not a direct consequence of a straightforward application of the gaussian mechanism, that would help me evaluate this work more fairly. 

Moreover, I find the last bit in the conclusion regarding the importance of privacy and how it can prevent data leaks by LLMs to be very much an oversell in this paper. The impact of this paper is limited to keeping the preferences of annotators who are involved in aligning these models, private. This by itself cannot and will not prevent data leaks from LLM responses due to the very nature of the way LLMs are trained. I would strongly prefer it if you could remove that statement/ reword it so as to not send a wrong message to the readers, especially ones who come from a more applied background and who may not be aware of how differential privacy works and what its limitations are.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the problem of aligning language models with human values while preserving the privacy of human judgments used in training. By leveraging differential privacy (DP), the authors ensure that human preference data remains confidential, even during alignment through reinforcement learning from human feedback (RLHF). They adapt the Bradley-Terry-Luce and Plackett-Luce models to achieve privacy-preserving reward learning, which performs comparably to non-private methods.

### Strengths
+ The study of privacy protection in LLM alignment is important and timely

### Weaknesses
- Some result lacks rigor in the proof
- Some important references are missing
- The technical novelty is not clear

### Questions
1. My first concern is that this paper needs a careful literature review. In particular, several important related works are missing. For example, [R1] studied a similar private LLM alignment problem as the current paper, considering both the local and central model of DP. Moreover, several works on private RL are missing, including [R2-R5]. 

2. The current proof of privacy guarantee is not grounded (i.e., Theorem 4.1 and in particular Lemma A.10). The key issue is that the privacy proof in the original paper of Kifer et al 2012 has a gap, which has been mentioned in [R1, R6, R7]. As a result, since the current proof of Lemma A.10 directly uses the result in Kifer et al 2012, it needs a careful check. It seems to me that this issue has already been addressed in [R1]. 

3. The utility upper bound seems to be straightforward, given the standard techniques in private linear bandit and RL. Moreover, no lower bound is established, which makes it hard to evaluate the tightness. 

4. Finally, it would be good to see some basic experiments to justify the theoretical results.





-- 

[R1] Chowdhury, Sayak Ray, Xingyu Zhou, and Nagarajan Natarajan. "Differentially private reward estimation with preference feedback." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

[R2] Qiao, Dan, and Yu-Xiang Wang. "Near-optimal differentially private reinforcement learning." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

[R3] Liao, Chonghua, Jiafan He, and Quanquan Gu. "Locally differentially private reinforcement learning for linear mixture markov decision processes." Asian Conference on Machine Learning. PMLR, 2023.

[R4] Chowdhury, Sayak Ray, and Xingyu Zhou. "Differentially private regret minimization in episodic markov decision processes." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 6. 2022.

[R5] Zhou, Xingyu. "Differentially private reinforcement learning with linear function approximation." Proceedings of the ACM on Measurement and Analysis of Computing Systems 6.1 (2022): 1-27.

[R6] Agarwal, Naman, et al. "Differentially private and lazy online convex optimization." The Thirty Sixth Annual Conference on Learning Theory. PMLR, 2023.

[R7] Redberg, Rachel, Antti Koskela, and Yu-Xiang Wang. "Improving the privacy and practicality of objective perturbation for differentially private linear learners." Advances in Neural Information Processing Systems 36 (2023): 13819-13853.

### Soundness
2

### Presentation
2

### Contribution
2
