# HessianGrad: Optimizing AI Systems with Hessian-Aware Textual Gradients

- Avg Score: 3.67
- Decision: Reject
- Scores: 3, 5, 3

## Abstract
Recent advancements in large language models (LLMs) have significantly enhanced the ability of LLM-based systems to perform complex tasks through natural language processing and tool interaction. However, optimizing these LLM-based systems for specific tasks remains challenging, often requiring manual interventions like prompt engineering and hyperparameter tuning. Existing automatic optimization methods, such as textual feedback-based techniques (e.g., TextGrad), tend to focus on immediate feedback, analogous to using first-order derivatives in traditional numerical gradient descent. However, relying solely on first-order derivatives can be limited when the gradient is either very small or fluctuates irregularly, which may slow down or stall optimization. To address these limitations, better adaptation in regions with small or fluctuating gradients is necessary. Second-order gradient methods, which incorporate the Hessian matrix, offer a promising solution by enabling more precise adjustments. Inspired by this, in this paper, we introduce HessianGrad, a novel optimization method that leverages textual feedback and tracks the iterative evolution of LLM systems responses across iterations, leading to more dynamic and adaptive optimization. We evaluate the effectiveness of HessianGrad on three tasks: prompt optimization, solution optimization, and code optimization. Experimental results demonstrate that HessianGrad consistently improves performance across all three tasks, achieving a **7.8%** improvement in prompt optimization, a **20.72%** gain in solution refinement, and a **29.17%** increase in code optimization compared to baselines, highlighting its adaptability and effectiveness in optimizing LLM-based systems.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a new LLM-based textual optimization method that takes the evolution of LLM systems responses across iterations into account. Improvements are achieved on multiple textual optimization tasks.

### Strengths
1. The paper writing is clear and easy to understand.
2. The proposed optimization method achieves considerable improvements on a variety of tasks.

### Weaknesses
1. Although the authors have written about the difference between momentum-based methods and HessianGrad, the novelty of transferring the focus from feedback similarity to response similarity is somewhat weak. The authors should include more convincing ablation experiments to verify that tracking the dynamics of responses is more effective.

2. The second-order Hessian formulation can not provide sufficient theoretical support for the optimization framework. The relationship between tracking feedback and tracking responses is not comparable to first-order and second-order optimization.

### Questions
Please respond to the concerns in the "Weaknesses" part.

Q1. The motivation (line 223~226) of introducing the similarity function does not match well with the second-order optimization theory. How can the similarity of the responses provide second-order information?

Q2. The concrete definition of the similarity function on line 244 is meant to connect with the formulation of second-order optimization. However, this definition is contrary to the motivation in line 226 ("more gradual and thoughtful evolution of the response over multiple iterations"), since larger similarity means larger fluctuation between successive steps, according to this definition. Moreover, this definition is actually focusing on changes in feedback ($L(r(p_t))$) instead of response ($r(p_t))$), and this is a point that contradicts the motivation of this paper. Please clarify how the definition aligns with the stated motivation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a new method for optimizing system prompts through gradient descent. The authors outline the limitations of the classical method "TextGrad", and present solutions to mitigate these issues. The second-order derivative (i.e., HessianGrad) is introduced into "TextGrad", thereby reducing the likelihood of the system prompt getting trapped in local minima. The authors conduct empirical experiments on three tasks: prompt optimization, solution optimization, and code optimization. The conclusion is that the proposed new method outperforms the naive TextGrad method and the Momentum-Enhanced TextGrad method across four mainstream models.

### Strengths
This paper is well written with clear presentation of the new algorithm, and the experiments are rigorous with repeatability.

### Weaknesses
Overall, this paper may have an algorithmic contribution, but supplemental details are required on both theoretical and experimental aspects.

1. Theoretical aspect:
- When defining the similarity $\mathcal{S}$ of the prompt before and after each iteration, you do not multiply it by any hyperparameters and directly add it to the original TextGrad to obtain the HessianGrad. Consider adding two hyperparameters, $\beta_1$ and $\beta_2$, to the two terms of HessianGrad, similar to Adamw. Could you conduct an ablation study on the effect of adding these hyperparameters, or provide justification for why they were not included in the current formulation?
- You use the $L_1$ norm to define $\mathcal{S}$, however, when measuring semantic similarity, cosine similarity is more commonly used as it signifies that the loss $\mathcal{L}$ before and after iteration is closer in direction, while $L_1$ primarily signifies that $\mathcal{L}$ is closer in numerical value. It is recommended to provide a rationale for choosing the $L_1$ norm as a similarity metric. (Simple reasons are acceptable, such as "$L_1$ norm is easier to compute"). Could you compare the performance of your method using L1 norm versus cosine similarity, or provide empirical justification for why L1 norm was chosen over other common similarity metrics?

2. Experimental aspect:
- Please provide the loss curves for HessianGrad, M-TextGrad, TextGrad, and CoT of their iterative processes for a representative example from each of the three tasks (prompt optimization, solution optimization, and code optimization) to demonstrate the effect of "HessianGrad Escaping Local Optima" as shown in Figure 1.
- Calculating HessianGrad typically requires more computational resources. Could you provide a detailed comparison of computational resources (GPU memory, runtime) for HessianGrad versus the baseline methods across all three tasks.

### Questions
1. In general computational frameworks, HessianGrad is computed by adding a small quantity in the iteration direction and recalculating the gradient once, followed by taking the finite difference of the gradients. From a practical standpoint, can the direct finite difference version of similarity $\mathcal{S}(r(p_t), r(p_{t-1})) = \frac{|| \mathcal{L}(r(p_t)) - \mathcal{L}(r(p_{t-1}))||}{||p_t - p_{t-1}||}$ save computational costs (since $\mathcal{L}(r(p_{t-1}))$ and $p_{t-1}$ are both values computed from the previous iteration) and achieve similar effects?
2. There is a typo in Equation (3) in Section 3. The second-order partial derivative should be denoted as $\frac{\overset{\sim}{\partial}^2\mathcal{L}}{\overset{\sim}{\partial}p_t^2}$ rather than $\frac{\overset{\sim}{\partial}^2\mathcal{L}}{\overset{\sim}{\partial}^2p_t}$.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Existing automatic optimization methods only focus on immediate feedback, which can be easily trapped by the local optima. HessianGrad is analogous to second-order derivative methods by taking into account the feedback over multiple iterations. Experimental results show consistent gains of HessianGrad in prompt optimization, solution refinement, and code optimization.

### Strengths
1. The core idea addresses a critical limitation of iterative/reflective methods that only focus on immediate feedback. 

2. This work covers a wide array of recent literature, and the presentation is clear.

### Weaknesses
1. Besides the analogy provided in Eq. 2 and Eq. 3, the core implementation of HessianGrad is to include a meta-prompt that encourages LLM to reflect over multiple turns as shown in Pg. 12. However, this is arguably analogous to the real Hessian matrix that the work is trying to deliver. Moreover, whether LLM is able to capture the second-order phenomenon is also questionable and lacks justification in this work. 

2. The actual technical contribution of this work is to provide a more refined meta-prompt to the original TextGrad’s meta-prompt. First, the contribution of the refined meta-prompt appears to be limited. Second, the OPRO work [1] has also included similar meta-prompts to reflect over multiple turns by feeding the iterative optimization trajectory into the context of LLM. Therefore, the novelty of this work is limited and appears more as an incremental improvement on TextGrad.

3. The selected baselines in main experiments are also questionable. Most baselines are TextGrad and M-TextGrad. However, for each task, competitive baselines, e.g. ProTeGi [2] in prompt optimization, are not compared.

### Questions
1. Could you provide justifications for the difference between your optimizer prompt and OPRO’s meta-prompt [1] for prompt optimization?  

[1] Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., & Chen, X. (2023). Large Language Models as Optimizers (No. arXiv:2309.03409). arXiv. http://arxiv.org/abs/2309.03409

[2] Pryzant, R., Iter, D., Li, J., Lee, Y. T., Zhu, C., & Zeng, M. (2023). Automatic Prompt Optimization with “Gradient Descent” and Beam Search (No. arXiv:2305.03495). arXiv. http://arxiv.org/abs/2305.03495

### Soundness
2

### Presentation
3

### Contribution
2
