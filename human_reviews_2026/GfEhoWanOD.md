# Activity-Driven Quantile Optimization: Dynamic Exploration and Exploitation in Recommender Systems

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Recommender systems, as core components of modern digital platforms, leverage reinforcement learning (RL) paradigms to optimize long-term user experience through exploration-exploitation trade-offs. While existing studies implement differentiated policies via coarse-grained user grouping, they face critical challenges in dynamically evolving scenarios: how to capture fine-grained user state transitions and establish precise exploration-exploitation balancing mechanisms. Empirical analysis on existing dataset shows that over 
40% of users experience activity level transitions within four weeks, highlighting the need for dynamic optimization. To address this, we propose Activity-Driven Quantile Optimization (ADQO), which integrates a general value critic network for user activity modeling and a quantile critic network to finely characterize the distribution of recommendation values, capturing the stochastic of user feedback. A dynamic policy implements high-potential exploration for low-activity users and low-risk exploitation for high-activity users: optimizing the upper quantiles for low-activity users to uncover latent interests, and the lower quantiles for high-activity users to mitigate risks. We further introduce two alignment losses to enhance training stability and consistency. Experiments demonstrate ADQO's superior performance across three datasets, effectively converting low-activity users to higher states and retaining high-activity users, validating its practical applicability. Our data analysis and training code are shared at https://anonymous.4open.science/r/ADQO-6DC9/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Summary:

To solve this, the paper proposes Activity-Driven Quantile Optimization (ADQO), a novel RL framework. ADQO utilizes a dual critic architecture:A general value critic estimates the user's state value ($V(s)$), which serves as a proxy for their long-term activity level. A quantile critic network (based on distributional RL) models the complete probability distribution of future returns ($Z(s,a)$) for a recommendation, capturing the inherent uncertainty in user feedback.The core idea is to use the activity level from the value critic to dynamically guide the policy's optimization objective on the value distribution. For low-activity users (low $V(s)$), the policy optimizes the upper quantiles of the return distribution, promoting "high-potential exploration" to discover latent interests.For high-activity users (high $V(s)$), the policy optimizes the lower quantiles, encouraging "low-risk exploitation" to avoid poor recommendations and maintain engagement. The framework is formalized by mapping the user's activity score $V_{\omega}(s)$ to a target quantile $\tau(s)$:$$\tau(s)=\tau_{high}-(\tau_{high}-\tau_{low})\cdot\frac{V_{\omega}(s)-V_{min}}{V_{max}-V_{min}}$$ The policy is then trained to optimize a combination of this activity-driven quantile and the expected return. Two additional alignment losses are introduced to enhance training stability and consistency. Experiments on three public datasets (KuaiRand, ML-1M, and RL4RS) demonstrate that ADQO significantly outperforms existing baselines, effectively converting low-activity users to higher states and improving the retention of high-activity users.

Contribution:
* Problem Formulation: It provides strong empirical evidence for the dynamic nature of user activity in recommender systems, motivating the need for adaptive policies beyond coarse-grained user grouping.
* ADQO Framework: It introduces a novel dual-critic RL framework that dynamically tailors the exploration-exploitation trade-off to individual users based on their activity level.
* Activity-Driven Optimization: The core mechanism of using a general value critic to model user activity and then using that score to select a specific quantile (potential-seeking or risk-averse) from a distributional critic for policy optimization.
* Training Stability: It proposes two novel alignment losses: an Exploration-Weighted Dual Critic Alignment and a Guidance-Based Policy Feedback Alignment , to improve the consistency and stability of the complex dual-critic training process.

### Strengths
1. Strong Empirical Motivation: The paper begins with a clear analysis of the KuaiRand dataset, convincingly demonstrating that user activity is not static, which grounds the paper's central problem in a real-world phenomenon.

2. Novel and Intuitive Method: The ADQO framework is a novel integration of standard value-based RL and distributional RL. The core idea of mapping high activity to risk-aversion (low quantiles) and low activity to potential-seeking (high quantiles) is both intuitive and elegant.

3. Comprehensive Evaluation: The authors evaluate ADQO against a wide array of 10 baselines, including classic RL (A2C, DDPG), distributional RL (IQN), and state-of-the-art RS-specific RL methods (DEHAC, UOEP).

4. Statistically Significant Results: The method shows superior performance across three distinct datasets in terms of both total reward and interaction depth, with results reported as statistically significant. The analysis also directly confirms that ADQO achieves its goal of converting low-activity users and retaining high-activity users.

### Weaknesses
1. High Complexity: The proposed system is complex, requiring the training of a policy network, two separate critic networks (value and quantile), and the computation of two additional alignment losses. This could pose a significant challenge for implementation, tuning, and computational overhead in a large-scale production environment.
2. New Hyperparameters: The method introduces several new hyperparameters, most notably the quantile boundaries $\tau_{low}$ and $\tau_{high}$ and the margin $\epsilon$ for the critic alignment loss. The paper does not provide a deep analysis of the model's sensitivity to these parameters.
3. Implicit Activity Proxy: The user's "activity level" is proxied by the learned state-value function $V_{\omega}(s)$. While the paper argues the state value captures long-term engagement, it is an implicit, black-box measure. It's unclear how this learned state value is more effective comparing to discounted sum of heuristic metric values (e.g., recent click-through rate) for driving the quantile selection.
4. Assumption: The main contribution is a more adpative exploration and exploitation balance RL mechansim. Although the design is clever and makes sense in assumption, we don't know if user activity is always in high transiton frequency in reality data, as the paper claimed.

### Questions
1. On the Policy Loss: The policy loss in Equation 8 is a sum of the activity-driven quantile $q_{\theta}(s,a;\tau(s))$ and the mean expected value $\mathbb{E}[Z(s,a)]$. The paper states that optimizing $q_{\theta}(s,a;\tau(s))$ alone causes instability. Could you elaborate on the nature of this instability and why adding the expected value term stabilizes the training?
2. On the Simulator: The evaluation relies on a simulator based on a pre-trained user model. How can we be sure this simulator accurately models the very phenomenon you are trying to influence—namely, the transition of a disengaged user to an engaged one?
3. On Real-World Deployment: Given the model's complexity, what are the primary challenges you foresee in deploying ADQO in a large-scale, live recommender system serving millions of users?

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
3

### Summary
This paper introduces Activity-Driven Quantile Optimization (ADQO), a reinforcement learning method that adapts exploration and exploitation based on user activity. Using two critic networks, a value critic for user engagement and a quantile critic for reward distribution, ADQO adjusts recommendations dynamically for different users. Experiments run on multiple datasets show improved long-term rewards and user retention.

### Strengths
1) The ADQO automatically adjusts the balance between exploration and exploitation based on each user's activity level, offering a more flexible and personalized recommendation strategy than static or rule-based methods.
2) The use of both a value critic (for user activity) and a quantile critic (for reward distribution) allows the model to capture both engagement potential and uncertainty, leading to more stable and effective policy learning.
3) As shown across multiple benchmark datasets, ADQO consistently outperforms existing reinforcement learning baselines in long-term rewards, user conversion, and retention, showing strong practical promise.
4) The authors provide open-source code and detailed experimental settings, which enables reproducibility and allows others to extend their work.

### Weaknesses
1) The framework is well-motivated but largely heuristic. Core ideas such as dual critics, activity-driven quantile mapping, and alignment losses lack formal theoretical grounding i.e. there’s no convergence or optimality analysis. The contribution feels more engineering-focused than theory-driven, which makes it less suited to ICLR’s scope and better aligned with applied venues like RecSys, WSDM, or SIGIR.
2) The ADQO integrates known components from Distributional RL (IQN), user activity modeling, and alignment regularisation. While the combination is effective, it's not an conceptual innovation. It's also unclear whether simpler risk-sensitive baselines could achieve similar results.
3) All results come from offline datasets (KuaiRand, MovieLens-1M, RL4RS). There is no real-world A/B testing or deployment evidence, which weakens the practical claims about improving user engagement in live systems.
4) Although the ADQO consistently outperforms baselines, the margins are relatively small and inconsistent across datasets. This makes the added architectural complexity less convincing in terms of real benefit.
5) Improving engagement for low-activity users is one of the paper's main motivations and a major justification for the proposed framework. While Figure 5 provides some subgroup analysis showing ADQO's advantage on low-activity and users at risk, this evidence is limited to a single dataset and lacks statistical depth or transition-rate analysis. Given how central this user group is to the paper's scope, the current evaluation does not demonstrate sufficiently that ADQO revives inactive users in a consistent and generalisable way.

### Questions
Some suggestions to the authors:
1) Provide theoretical analysis or formal justification, even under simplified assumptions, to strengthen the quantile mapping and alignment losses.
2) Include small-scale online testing to support real-world applicability.
3) Since low-activity users are central to the paper’s motivation, the current evidence in Figure 5 is not sufficient. Consider adding more targeted evaluations such as activity transition rates, temporal performance curves, or results on additional datasets, to demonstrate better stability and robustness gains for this user group.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides an interesting perspective on decision making using a distributional actor-critic algorithm, which is specifically adapted to handle both high-activity and low-activity users through specially designed loss functions. I think this is an interesting and meaningful problem worth exploring. The paper proposes a novel actor-critic design with two critics (one distributional critic and one value critic) and an actor to update the policy, specifically adapted to address this problem. Overall, it is an inspiring work, and the simulations are extensive and clearly demonstrate the performance of the system.

### Strengths
1. The paper’s idea and real-world motivation are well established, and the overall framework is well constructed.
2. The simulations are extensive, covering multiple settings with comprehensive comparisons to existing works in the literature.
3. I particularly appreciate the authors’ effort in providing clear illustrations of the actor-critic structure and various visualization plots.

### Weaknesses
1. The related work section is a bit short. I would expect more discussion on literature that focuses purely on distributional RL (e.g., Nam et al., 2021), as well as works that explore similar actor-critic extensions.
2. Some of the loss constructions lack theoretical justification or related citations. For example, in Equation (7), is \tau(s) defined mainly for ease of implementation? For real practitioners who may operate in different application settings, should they consider alternative mappings? Why was this particular form chosen, and is there any related work that supports this choice?
3. Following point 2, in Equation (10), while I understand the motivation to keep the distributional and expected critics “close” to each other, I am curious whether this loss term is proposed here for the first time or adapted from an existing paper.
4. Overall, the paper feels more application-driven, with a specifically designed structure to both explore high-potential rewards for low-activity users and reduce risk for high-activity users. I initially thought this would be supported by real-world online testing, but it turns out the evaluation is based only on simulations. While I appreciate the diversity of simulation settings, it is somewhat hard to assess the generalizability of the proposed system since several components (e.g., activity score definition, mapping back to quantiles via a seemingly non-unique function form in Equation 7, joint updates of two critics and one actor) appear quite specific. The paper would benefit from more clarity on which parts of the design are generalizable, and from presenting the framework in a more standardized way to help other readers or practitioners replicate the system.
5. A minor point: it might be better to include standard errors (or empirical confidence intervals) in Figure 5 to better compare the performance of different methods.

Reference:
Nam, D. W., Kim, Y., & Park, C. Y. (2021). GMAC: A distributional perspective on actor-critic framework. In International Conference on Machine Learning (pp. 7927–7936). PMLR.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the challenge of dynamic user activity in Reinforcement Learning (RL)-based recommender systems. The authors present empirical analysis on the KuaiRand dataset showing that over 40% of users experience transitions in their activity levels over a four-week period. To address this, they propose Activity-Driven Quantile Optimization (ADQO). ADQO utilizes a dual-critic approach: a general value critic to model user activity levels and a distributional (quantile) critic to model the uncertainty of recommendation values.


The core mechanism involves dynamically mapping the estimated user activity to specific target quantiles for policy optimization. The policy optimizes upper quantiles (optimistic exploration) for low-activity users to uncover latent interests, and lower quantiles (risk-averse exploitation) for high-activity users to prevent churning. The framework further incorporates two alignment losses: an exploration-weighted dual critic alignment and a guidance-based policy feedback alignment to ensure training stability.

### Strengths
Originality: The idea of directly linking user activity levels to the target quantiles of a distributional critic is a novel and intuitive approach to unifying exploration and exploitation within a single policy network. This avoids the complexity of maintaining separate actor populations for different user groups, as seen in baselines like UOEP.

Significance: The paper tackles a well-motivated real-world problem. The initial analysis of user activity transitions (Figure 1) clearly establishes the need for dynamic policies.

Quality: The experimental validation is comprehensive, utilizing three standard datasets (KuaiRand, ML-1M, RL4RS) and comparing against a wide range of relevant baselines, including recent methods like DEHAC and UOEP. The ablation studies (Table 2) effectively justify the need for the proposed alignment losses.

Clarity: The paper is well-written, and Figure 2 provides a clear overview of the proposed framework.

### Weaknesses
Heuristic Mapping Function: The mapping from user activity value $V_\omega(s)$ to the target quantile $\tau(s)$ is defined as a simple linear interpolation in Equation (7)12. While functional, this linear relationship is a strong heuristic. The paper does not theoretically justify why a linear mapping is optimal compared to non-linear alternatives.

Hyperparameter Complexity: The method introduces several new hyperparameters, including $\tau_{high}$, $\tau_{low}$, and the learning rates for the two new alignment losses ($\eta_C$, $\eta_F$). The sensitivity analysis in Figure 8 indicates that performance can degrade if these (particularly $\eta_F$) are not carefully tuned14141414.

Simulation Dependence: Like many RL4Rec papers, the evaluation relies heavily on an online simulator constructed from offline data. While standard practice, it is always worth noting that simulators may not perfectly capture the complex, long-term stochasticity of real user behavior, potentially overestimating the effectiveness of complex exploration strategies.

### Questions
Regarding Equation (7), did the authors explore non-linear mappings between the activity value $V_\omega(s)$ and the target quantile $\tau(s)$? For example, a sigmoid or piecewise function might better capture distinct user states.

In the sensitivity analysis (Appendix F, Figure 7), only the impact of $\tau_{low}$ is shown17. How sensitive is the model to the choice of $\tau_{high}$, and what is the optimal "gap" between these two bounds?

How does the model perform for users who exhibit highly volatile activity (frequent major transitions)? Does the TD-learning based value critic adapt quickly enough to these rapid shifts?

### Soundness
3

### Presentation
3

### Contribution
3
