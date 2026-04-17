# Why DPO is a Misspecified Estimator and How to Fix It

- Decision: Accept (Oral)
- Scores: 6, 8, 6

## Abstract
Direct alignment algorithms such as Direct Preference Optimization (DPO) fine-tune models based on preference data, using only supervised learning instead of two-stage reinforcement learning with human feedback (RLHF). We show that DPO encodes a statistical estimation problem over reward functions induced by a parametric policy class. When the true reward function that generates preferences cannot be realized via the policy class, DPO becomes misspecified, resulting in failure modes such as preference order reversal, worsening of policy reward, and high sensitivity to the input preference data distribution. On the other hand, we study the local behavior of two-stage RLHF for a parametric class and relate it to a natural gradient step in policy space.  Our fine-grained geometric characterization allows us to propose AuxDPO, which introduces additional auxiliary variables in the DPO loss function to help move towards the RLHF solution in a principled manner and mitigate the misspecification in DPO. We empirically demonstrate the superior performance of AuxDPO on didactic bandit settings as well as LLM alignment tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the misspecification of the parametric reward model in the DPO framework. 
The standard DPO framework assumes that the LLM is expressive enough to be able to fit the ideal solution characterized by RLHF.
The authors raise critical questions that the LLM models can be misspecified with a concrete example (Section 3.1). This example shows that such misspecification may lead to undesirable behavior. 
Based on the local analysis of the geometry, they propose to add additional optimization variables which allow enough slackness to the model to fit to the desired RLHF solution. 
The experiment results show clear advantage of the proposed method AuxDPO.

### Strengths
This paper proposes an interesting idea to further improve the quality of DPO, based on principled local information geometry.
The proposed technique is not only principled but also effective in practice.

### Weaknesses
The only downside of the paper is that it is very notation heavy and not easy to follow in the first read.

### Questions
- I cannot understand the implication of Remark 4. Do the authors want to argue that Song et al. (2024)'s argument is wrong? If the authors can end up with a different conclusion, what is the difference? I believe that the difference comes from the misspecification, but I appreciate if the authors can clarify.
- I have a question about the notation in Section 4. The authors use $\mathcal{N}(A)\subset\mathbb{R}^m$ to denote the nullspace of $A$. Given this, I do not know how to parse $\\|\\mathcal{N}(A)\\delta\\|^2$, and how it can manifest as in the final objective function. I think this paragraph deserves a further expansion.

#### **Suggestions**
- The notation for dataset is a bit weird. For example, in line 138, consider $\mathcal{D}=\\{(\text{tuple}\_i)\\}_{i=1}^n$.
- The shorthand b/w and w.r.t. read weird. Please expand them in the final version.

---

Overall, I find the core idea compelling and the paper a solid contribution. However, a thorough revision to streamline and simplify the notation would greatly improve the manuscript’s readability and its accessibility to a broader audience.

### Soundness
4

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
This paper first demonstrates that DPO implicitly projects the weights of the optimal solution to the manifold of possible reward functions under the given policy class. The authors prove that this can lead to mis-specifications where the reward function does not actually obey preferences.

### Strengths
* the paper is mathematically rigorous in demonstrating its claims. 
* the work does a good job demonstrating why the mis-specification is a problem through a useful example and follow-up points.
* the results demonstrate strong performance in both in distribution and out of distribution settings.

### Weaknesses
* the derivation of the aux DPO objective makes sense, but is justified using a local approximation. While this makes sense, it could be worth pointing out the the AuxDPO solution (at least to my understanding) holds under these approximations only. 
* The paper is a bit hard to follow at times as it is particularly dense. I think at various points in the manuscript having more motivation and explanation would be helpful. Why do we want to do a first order approx? What is the meaning of the A matrix? 
* In a similar manner, the actual instantiation of AuxDPO is a bit unclear. AuxDPO adds variables in the null space of the A matrix, but how are these variables actually represented in code? Is $\delta$ just predicted as another head of the LLM? 
* I understand why the authors used MMLU etc. as a preference learning benchmark, but it (in spirit) does not seem exactly like one since its more of a QA dataset. While DPO can be used in this case I think more benchmarks actually based on human preferences woudl be preferred.

### Questions
See weaknesses section.

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
4

### Summary
The paper investigates a fundamental limitation of Direct Preference Optimization (DPO), a widely used direct preference alignment method. It shows that when using parametric policy classes (as opposed to the tabular assumption underlying DPO's derivation), DPO suffers from a misspecified statistical estimation problem. This misspecification arises when the true reward function that generates preferences cannot be represented by the chosen policy class, leading to undesirable outcomes. To address this, the authors propose AuxDPO, which augments the DPO loss with auxiliary variables to mitigate misspecification. They demonstrate the empirical effectiveness of AuxDPO on LLM preference alignment tasks.

### Strengths
The paper is well written and clearly structured.

Provides an insightful theoretical analysis of DPO and RLHF via Taylor approximation, revealing:
- the local geometry of DPO under parametric policies,
- the local geometry of RLHF optimization, and  
- the relationship between RLHF equivalence classes and DPO linearization.

Proposes a novel and principled solution (AuxDPO) to address the identified misspecification issue.

### Weaknesses
Could oversampling or undersampling preference pairs to balance frequencies before DPO training mitigate the misspecification issue and yield comparable performance?

There is no discussion or experimental analysis of AuxDPO's sensitivity to its core hyperparameters ($\lambda$ and $n$). How should these values be chosen in practice?

### Questions
(Related to the constructive example in Proposition 3): If preference data are balanced (uniform) across $(s, a, a')$, would DPO still exhibit the preference reversal issue shown in the example?

For the datasets used in the experiments, were the preference frequencies balanced or imbalanced?

### Soundness
3

### Presentation
3

### Contribution
3
