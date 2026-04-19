# Neural Dueling Bandits: Preference-Based Optimization with Human Feedback

- Decision: Accept (Poster)
- Scores: 6, 8, 5, 6

## Abstract
Contextual dueling bandit is used to model the bandit problems, where a learner's goal is to find the best arm for a given context using observed noisy human preference feedback over the selected arms for the past contexts. However, existing algorithms assume the reward function is linear, which can be complex and non-linear in many real-life applications like online recommendations or ranking web search results. To overcome this challenge, we use a neural network to estimate the reward function using preference feedback for the previously selected arms. We propose upper confidence bound- and Thompson sampling-based algorithms with sub-linear regret guarantees that efficiently select arms in each round. We also extend our theoretical results to contextual bandit problems with binary feedback, which is in itself a non-trivial contribution. Experimental results on the problem instances derived from synthetic datasets corroborate our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies dueling bandits based on neural networks. UCB and TS algorithms are suggested, and sublinear regret bounds are achieved for both algorithms. The theoretical analysis is robust as well.

### Strengths
This work provides theoretical results of two mainstream algorithms, UCB and TS, for neural dueling bandits. These results are significant contributions to bandits.

### Weaknesses
Overall, this paper is difficult to follow due to the use of numerous quantities and notations that are not explicitly defined. Given the long-standing history and inherent complexity of neural network analysis, it is understandable that some notations and intermediate reasoning steps are abbreviated. However, the reviewer believes that enhancing clarity would greatly improve readability for uninitiated readers.

L59-63: Could you be more specific?

L85-94: Since many readers may be unfamiliar with dual bandits, the description should be more detailed. For instance, the authors should clarify that the learner selects two arms from the action set with multiple arms, plays both arms, and receives feedback from both, which is captured as the preference y_t. It would be helpful to emphasize how this differs from standard contextual bandits. Additionally, the authors introduce the context-arm set X_t. As the reviewer understands, this is no different from setups where either an action or context set is considered as long as a context and action pair is in R^d. Is there any technical distinction between the two? If not, this setup seems to introduce unnecessary complexity.

L95-111:  The authors stated that the logistic function is considered as the link function, but stated in Assumption 1 that a class of link functions satisfying two conditions is considered in this work. For the reviewer, this description is confusing. 

L112-120: As these regrets are not common in other literature, regret should be better explained with the basic ideas of regret.

Could you provide more specific practical applications of dueling bandits?

### Questions
The reviewer notes that while the suggested setup differs somewhat from previous works (Zhou et al., 2020; Zhang et al., 2021), the analysis appears to be quite similar, based on the main manuscript. Although certain distinctions are mentioned in the introduction, the reviewer finds it challenging to identify the specific differences. Could you clarify more explicitly the technical novelty of this work compared to the existing literature?

As the authors provide two algorithms, the reviewer would like to know some comparisons between these two. While some comparative simulation results are provided, could you also include a more qualitative analysis, incorporating some mathematical insights, to highlight the differences between the two algorithms?

For regret analysis, what kind of regret is analyzed between average instantaneous regret and weak
instantaneous regret?

For TS, what do we do if the first and second arms are the same?

Could the authors define the matrix H in any way? What are the differences between H and H’?

In Assumption 2, [x]_k = [x]_k+d/2, [] is used for multiple purposes, making some confusion. It should be defined clearly.

L191: Is \phi is g? Could the authors please explain why a different notation is needed if they are the same?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper titled *Neural Dueling Bandits: Principled Preference-Based Optimization with Non-Linear Reward Function* tackles the challenge of dueling bandits in contexts where the reward function is non-linear—a common scenario in real-world applications like recommendation systems and search ranking. The authors propose leveraging neural networks to model these non-linear reward functions based on preference feedback. They introduce two algorithms, *NDB-UCB* and *NDB-TS*, which are extensions of the Upper Confidence Bound and Thompson Sampling frameworks, respectively. Both algorithms come with sub-linear regret guarantees. The paper also extends the analysis to binary feedback settings and discusses implications for reinforcement learning from human feedback (RLHF). Theoretical results are supported by experiments on synthetic datasets.

### Strengths
**Originality**: The paper makes a significant contribution by extending the dueling bandit framework to accommodate non-linear reward functions using neural networks. This advances the field beyond the limitations of linear assumptions prevalent in prior work, allowing for more realistic modeling of user preferences.

**Theoretical Rigor**: The authors provide rigorous theoretical analysis, deriving sub-linear regret bounds for the proposed algorithms under standard assumptions. Extending these results to binary feedback settings enhances the robustness and applicability of their approach.

**Practical Relevance**: By addressing non-linear reward functions, the work is directly applicable to practical domains such as recommendation systems and web search, where user preferences are inherently complex. This increases the algorithms' relevance and potential impact on real-world applications.

**Clarity and Structure**: The paper is well-organized and clearly written. It presents the problem, proposed solutions, and theoretical foundations in a logical and accessible manner, making it easier for readers to comprehend the complex concepts involved.

**Impact on RLHF**: The exploration of reinforcement learning from human feedback broadens the impact of the work. By aligning AI systems more closely with human preferences, the paper contributes to an important area of AI research focused on ethical and user-aligned AI.

### Weaknesses
**Experimental Validation**: The empirical evaluation is limited to synthetic datasets. Testing the proposed algorithms on real-world datasets would strengthen the claims regarding their practical utility and effectiveness.

**Computational Complexity**: The paper lacks a detailed discussion on the computational demands of the algorithms, particularly concerning the training of neural networks in an online setting. Insights into scalability and efficiency are essential for practical deployment.

**Baseline Comparisons**: While comparisons are made with linear dueling bandit algorithms, the paper does not benchmark against other non-linear methods (e.g., those using Gaussian processes or kernel methods). Including such comparisons would provide a more comprehensive performance analysis.

### Questions
**Real-World Dataset Evaluation**: Have you considered evaluating your algorithms on real-world datasets, such as those from recommendation systems or search logs, to demonstrate their practical applicability and effectiveness?

**Computational Efficiency and Scalability**: Can you provide insights into the computational complexity of your algorithms? How do they scale with the number of contexts and arms, and are there strategies (e.g., model compression, online learning techniques) to mitigate computational overhead?

**Comparison with Other Non-linear Methods**: How does your approach compare to other non-linear bandit algorithms, particularly those leveraging Gaussian processes or deep kernel learning? Including comparisons with these methods could strengthen your empirical evaluation.

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
In this paper, the authors extend the contextual dueling bandit formulation to neural bandit settings, where the reward function is arbitrary and does not assume linearity. To address this task, they propose two methods based on the existing NeuralUCB and NeuralTS algorithms. Both theoretical and empirical analyses on synthetic datasets are provided.

### Strengths
The application of neural bandits to solve real-world dueling bandit problems is intuitive, as existing linear contextual dueling bandit algorithms may fail to capture complex reward relationships.

Additionally, the analysis of neural bandits with binary feedback may be of particular interest to the bandit research community.

### Weaknesses
Firstly, my concerns are with the problem formulation and theoretical analysis. I will list a few below. For instance:

If my understanding is correct, the problem formulation heavily relies on the link function $\mu$, whose properties significantly influence the regret bound, particularly through terms such as $\kappa\_{\mu}$ and $L\_{\mu}$. Furthermore, unlike conventional stochastic bandit works, this paper assumes the reward feedback associates with the sub-Gaussian noise, which in this case, depends on the chosen link function. Typically in stochastic bandit settings, sub-Gaussian noise is independent of the learning formulation and is determined solely by environmental settings, which makes this formulation unnatural.

Additionally, in Assumption 1, $\mathcal{X}$ is not clearly defined. Does $\mathcal{X}$ represent the entire arm context space with $\mathcal{X}\_t \subset \mathcal{X}$? If so, how would the learner possess prior knowledge of $\mathcal{X}$ to select the optimal $\kappa\_{\mu}$ value for exploration (i.e., the $\beta\_T$ term in Equation 3)? A similar issue also exists for $\tilde{d}$, as it is unrealistic to assume the learner has prior knowledge of this parameter.

The theoretical analysis appears to rely heavily on the proof framework of NeuralUCB and NeuralTS, adapting assumptions like the positive definite NTK matrix for arm context. The authors also mention the $c_0$ term, representing gradient differences between arm contexts in Theorem 2. This term should be clearly shown in the regret bound, as its data dependence could lead it to grow with $T$ and should not be hidden by big-O notation.

All experiments are conducted on synthetic datasets with predefined configurations. While the authors claim the paper is motivated by dueling bandit applications in LLM tuning, no experiments are included to support this claim. Furthermore, the theoretical analysis may not align with real RLHF settings, particularly given the strong over-parameterization assumption, which may not be realistic. In practice, each arm could be a generated response from an LLM, with high-dimensional vector representations and complex reward mappings. While RLHF methods like DPO employ powerful neural architectures, the analysis in this paper assumes an MLP, which may be insufficient. Therefore, I believe real dataset experiments relevant to LLM tuning, as indicated by the authors, are necessary.

### Questions
Please refer to "Weaknesses" above.

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
The paper studies the challenge of modeling dueling bandit problems where the reward function is non-linear. Traditional algorithms assume a linear reward function, which can be restrictive in real-life applications like online recommendations or web search rankings. The authors propose using neural networks to estimate the non-linear reward function using noisy preference feedback. They introduce upper confidence bound (UCB) and Thompson sampling (TS) based algorithms with sub-linear regret guarantees for efficient arm selection. Their theoretical results extend to contextual bandit problems with binary feedback, provide meaningful discussion with LLMs and experimental results on synthetic datasets support their theoretical findings.

### Strengths
The paper introduces neural network-based methods to estimate the non-linear reward function using preference feedback observed for previously selected arms. The paper proposes two algorithms: one based on UCB and another based on TS. Both algorithms are equipped with sub-linear regret guarantees, ensuring their effectiveness in selecting the best arm over time. Moreover, the authors provide the interesting discussion with contextual bandits with binary feedback and RLHF.

### Weaknesses
However, for the regret analysis, the authors still require the NTK gram matrix to be positive definite. This assumption can be a limitation in real-world applications, as it can be easily violated by repeating observed arm contexts, thereby hindering the practicality of the proposed methods.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
