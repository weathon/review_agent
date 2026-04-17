# AI Alignment with Provable Protection of Human Judgements

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Reinforcement learning from human preference rankings forms the basis for training language models to be helpful and value-aligned. As these powerful AI systems are trained for increasingly high-stakes tasks, the risk of leaking sensitive human training data increases. However, the problem of protecting human preference data is complicated by the fact that reinforcement learning from human feedback is a multistage pipeline involving learning a reward function from human preferences, and subsequently training a language model policy from the learned rewards. To address these issues, we design algorithms for the task of alignment from preference feedback that provably avoid leaking human preference data in both the Bradley-Terry and Plackett-Luce models. Our algorithms satisfy $\epsilon$-DP while matching the minimax optimal sample complexity for the task of aligning a policy to human preference rankings. These results demonstrate that there is no inherent tradeoff between protecting the privacy of human preferences and efficient alignment with human values.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studied the RLHF under the central differential privacy model. The authors investigated the problem of private RLHF on both pairwise and K-wise comparisons for the human preference model in RLHF. And they also consider RLHF via the contextual bandits view and the Markov decision process view. They designed a private pessimistic MLE algorithm for the reward model by achieving DP via the objective perturbation and output perturbation. And they provided suboptimality bounds for their algorithm.

### Strengths
1. The paper studied an interesting and important problem of private RLHF.
2. The paper presented algorithms and derived theoretical guarantees for their design.

### Weaknesses
1. The statement for writing in the paper is unclear. For example, in the abstract, the authors stated they aim to protect the privacy of human preferences. However, the algorithm design is not privacy-preserving for preference information.
2. The algorithm and technical parts are incremental. They are mainly building on [1], combining with existing DP techniques.
3. The theoretical results are missing some important factors. For example, it is well-known that the exponential term of $e^B$ is unavoidable due to the BT model, but I didn't find it in their final bounds.
4. The literature review is not comprehensive. Please see [2-4] for related work.
5. Lack some empirical results.

[1]. Zhu, Banghua, Michael Jordan, and Jiantao Jiao. "Principled reinforcement learning with human feedback from pairwise or k-wise comparisons." International Conference on Machine Learning. PMLR, 2023.

[2]. Zhan, Wenhao, et al. "Provable offline preference-based reinforcement learning." arXiv preprint arXiv:2305.14816 (2023).

[3].Yuan, Hongyi, et al. "Rrhf: Rank responses to align language models with human feedback." Advances in Neural Information Processing Systems 36 (2023): 10935-10950.

[4].Song, Feifan, et al. "Preference ranking optimization for human alignment." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.

### Questions
1. In your algorithm 1, by using the objective function, refer to section 5.2 of [1], this already achieves DP for parameter estimation. Why do you still need output perturbation in step 6?
2. Compare your results with the results in [1] and discuss them. And what is the advantage of your methods and results compared with theirs?
3. You claim your results are minimax optimal sample complexity. What is the evidence for minimax optimal such as lower bounds?

[1].Chowdhury, Sayak Ray, Xingyu Zhou, and Nagarajan Natarajan. "Differentially private reward estimation with preference feedback." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores the leakage of human preference data in Reinforcement Learning from Human Feedback (RLHF). The authors design algorithms that satisfy $\epsilon$-DP while still aligning to the preference data.

### Strengths
1. The paper clearly states the assumptions, which are reasonable.
2. The paper considers a extensive set of settings with both Contextual Bandits and general MDPs both in the pairwise and K-wise comparison setting.
3. The findings of this paper are quite strong, demonstrating that the proposed algorithms satisfy $\epsilon$-DP, while having a small sub-optimality gap.

### Weaknesses
1. The writing of the paper needs some improvement. The introduction and conclusion section seem to bring up unrelated examples such as the generation of harmful content. It is not fully clear how these examples relate to the topic of the paper.
2. While I like the theoretical results of the paper, it does not include any empirical results. While I understand that the focus of the paper are the theoretical results,  I believe given the practical implications of the results, at least a small toy-box experiment would be highly beneficial.
3. I’m not convinced by the relevance of the considered topic. Preference data doesn’t seem to sensitive data and I do find the examples given in the introduction to be quite artificial. I do believe the authors should include more practical examples of why the leakage of preference data would be a problem.

### Questions
1. Is there any research on whether it is possible to extract preference data?
2. Why would the leakage of preference data be problematic?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a framework for performing Reinforcement Learning from Human Feedback (RLHF) with formal differential privacy guarantees, ensuring that human preference data remains confidential while maintaining statistical efficiency. The authors design algorithms for learning from preference rankings under the Bradley–Terry and Plackett–Luce models that achieve differential privacy and match the minimax-optimal sample complexity of non-private RLHF. They provide theoretical results for both contextual bandit and general MDP settings, demonstrating that private versions of maximum-likelihood estimation and pessimistic policy optimization yield near-optimal suboptimality bounds $O(\sqrt{d/n})$.

### Strengths
1. Ensuring data privacy is a crucial aspect of reinforcement learning, and the authors present an algorithm that achieves differential privacy guarantees without sacrificing performance.
2. The manuscript is well-organized, clearly presented, and easy to understand.
3. This paper addresses both the contextual bandit scenario and the general MDP setting.

### Weaknesses
1. The level of technical difficulty and originality. Algorithm 1, 2 and 3 seem to be derived straightforward using the gaussian mechanism.
2. The paper only provides upper bounds without corresponding lower bounds to demonstrate their tightness.
3. The paper lacks experimental results. While this is acceptable for a theoretical study, RLHF’s primary objective in the context of LLMs is practical alignment in real-world applications. Without empirical validation, the contribution of the paper appears weaker and less convincing in demonstrating its real-world relevance.
4. Typo: in page 2 the definition of $\sum_{D}$, the inner summation should start from $k=j+1$.

### Questions
How does the result of this paper compare with those in other DP-RHLF studies? For example, in [Chowdhury et al., 2024], a similar bound is achieved on a randomized response strategy.

[Chowdhury et al., 2024] Chowdhury, Sayak Ray, Xingyu Zhou, and Nagarajan Natarajan. "Differentially private reward estimation with preference feedback." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the privacy risks in Reinforcement Learning from Human Feedback (RLHF). While RLHF effectively teaches models to follow human-preferred behaviors, it relies on human preference data—potentially revealing sensitive information from human data. To mitigate this, the authors propose differentially private algorithms for aligning policies using human preference rankings under Bradley–Terry and Plackett–Luce models. Their key contribution is proving that privacy-preserving RLHF can match the minimax optimal sample complexity of non-private algorithms for the leading terms, meaning there is no inherent tradeoff between privacy and statistical efficiency.

### Strengths
1. The paper’s originality lies in integrating differential privacy (DP) directly into the theory of RLHF.  Prior work on privacy in RL assumes rewards are private but known, while this work uniquely handles the learning of rewards from human rankings.
2. The theoretical development is rigorous, well-grounded in statistical learning and DP theory, and connects clearly to recent results on RLHF sample complexity (Zhu et al. 2023).

### Weaknesses
1. While the theoretical contributions are solid, the paper lacks any empirical or simulated demonstration of the proposed algorithms. Even small-scale synthetic experiments (e.g., in 2D linear bandits) could illustrate the impact of privacy noise on performance and validate the asymptotic analysis.
2. The results hinge on the assumption that rewards are linearly parameterized with known feature maps. This assumption, while standard in theoretical RL, limits applicability to real-world LLM alignment tasks where reward models are deep neural networks. 
3. The paper does not provide or discuss any lower bound in the context of DP-RLHF itself. To me, the techniques to derive the upper bounds in this paper are relatively direct given the literatures in both RLHF and DP.
4. Since this is a theory driven paper, it is necessary to discuss more on existing literatures on the theory of RLHF. References include [1,2,3,4,5], etc.

**References:** 

[1] Zhan, Wenhao, Masatoshi Uehara, Nathan Kallus, Jason D. Lee, and Wen Sun. "Provable Offline Preference-Based Reinforcement Learning." In The Twelfth International Conference on Learning Representations.

[2] Liu, Zhihan, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. "Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer." Advances in Neural Information Processing Systems 37 (2024): 138663-138697.

[3] Cen, Shicong, Jincheng Mei, Katayoon Goshvadi, Hanjun Dai, Tong Yang, Sherry Yang, Dale Schuurmans, Yuejie Chi, and Bo Dai. "Value-Incentivized Preference Optimization: A Unified Approach to Online and Offline RLHF." In The Thirteenth International Conference on Learning Representations.

[4] Xie, Tengyang, Dylan J. Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Hassan Awadallah, and Alexander Rakhlin. "Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF." In The Thirteenth International Conference on Learning Representations.

[5] Huang, Audrey, Wenhao Zhan, Tengyang Xie, Jason D. Lee, Wen Sun, Akshay Krishnamurthy, and Dylan J. Foster. "Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization." In The Thirteenth International Conference on Learning Representations.

### Questions
1. Could the authors provide even a small-scale empirical study (e.g., synthetic linear bandit or toy MDP) to illustrate the tradeoff between privacy level $\epsilon$ and suboptimality? This would make the paper more accessible to the broader ICLR audience.
2. Can the method extend beyond linear setups to general function approximations?

### Soundness
2

### Presentation
2

### Contribution
1
