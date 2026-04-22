# Boosting Multi-Domain Reasoning of LLMs via Curvature-Guided Policy Optimization

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Multi-domain reinforcement learning (RL) for large language models (LLMs) involves highly intricate reward surfaces, posing significant challenges in finding parameters that excel across all domains. Recent empirical studies have further highlighted conflicts among domains, where gains in one capability often come at the expense of another. However, approaches to mitigate such conflicts and enhance multi-domain reasoning remain largely underexplored. To address this challenge, we propose **C**urvature-**G**uided **P**olicy **O**ptimization (**CGPO**), a principled and scalable training framework to advance the multi-domain reasoning of LLMs. Inspired by Newton's method, CGPO exploits the geometric structure in the reward surface, while sidestepping the prohibitive cost of Hessian computation. At each update, CGPO processes domains in random order, preconditioning their gradients with curvature information from other domains to foster richer cross-domain interactions. This mechanism further promotes implicit gradient alignment by maximizing inter-domain inner products in expectation, steering the parameters toward regions that jointly enhance multi-domain performance. Extensive experiments on a mixed dataset covering math, coding, science, and creative writing, evaluated across seven widely-used benchmarks, show that CGPO significantly outperforms all baselines in terms of faster reward improvement and stronger multi-domain capability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript introduces a multi-domain post-training algorithm for LLMs. This method is motivated by Newton's method, where the Hessian matrix can reshape the objective landscape and accelerate convergence. In practice, the authors propose that sequentially training on each domain in a randomized order will naturally simulate the cross-domain Hessian-gradient interaction, and thus is better than joint training on all the domains.

### Strengths
* The algorithm is simple in practice and grounded in well-motivated theory.
* The paper is well written; the authors clarify their ideas with a thorough background and intuitive discussion.

### Weaknesses
* The author has some concerns about Equation (5) and the discussion following it. The authors claim that "since $\sigma$ is randomized, in expectation every pair will contribute equally". However, for each **specific** training iteration, the permutation is fixed, which means half of the $(i,j)$ pairs will not be present. Therefore, it is not an unbiased estimation of the Hessian matrix, and it is not clear that this effect will cancel out across multiple iterations. The reviewer would like to hear more analysis and discussions on this point. 
* Regardless of the above point, let's accept the proposal that the training mechanism in the manuscript can simulate the cross-domain Hessian-gradient interaction. However, it is still unclear why joint-training on all domains cannot. Of course, there are no such interactions for a mini-batch, but there might be some cross-domain interaction stemming from two consecutive mini-batches. To convince the readers of the superiority of the training mechanism proposed in this paper, the authors should provide some analysis of the trivial joint-training mechanism in addition to experimental results.

### Questions
* Please refer to the weaknesses.
* I wonder whether the proposed mechanism can be utilized for pre-training of LLMs, because typically the pre-training corpus is also multi-domain. The reviewer would love to hear some discussions on this point, and what is even better, some experimental results.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
1．	This paper presents Curvature-Guided Policy Optimization (CGPO), a novel training framework designed to enhance the multi-domain reasoning capabilities of Large Language Models (LLMs) using reinforcement learning (RL). 
2．	The core challenge addressed is the presence of cross-domain conflicts, where improvements in one domain (e.g., coding) can lead to degradation in another (e.g., creative writing). 
3．	CGPO is inspired by the geometric principles of Newton's method but avoids its prohibitive computational cost by using a randomized sequential update mechanism. This mechanism implicitly approximates Hessian-gradient interactions across domains, fostering gradient alignment and steering the model towards parameters that perform well across all domains simultaneously. 
4．	The method is evaluated on a diverse dataset spanning math, coding, science, and creative writing, and demonstrates superior performance over several strong baselines.

### Strengths
1.  The core insight of leveraging the geometric structure of the reward landscape (inspired by Newton's method) to mitigate multi-domain conflicts is both innovative and principled. The motivation is clearly laid out, connecting the limitations of existing first-order and gradient-manipulation methods to the potential benefits of second-order information.
2.  A significant contribution is the design of a method that captures the benefits of curvature information without explicitly computing the Hessian. The proposed mechanism of randomized sequential updates is elegant and computationally lightweight, introducing only negligible overhead compared to standard joint training. This makes CGPO highly practical and scalable for large-scale LLM RL training, a critical consideration in this field.
3.  The paper provides a solid mathematical foundation for the proposed method. The derivations (in the main text and appendix) convincingly show how the sequential update scheme approximates cross-domain Hessian-gradient products and, in expectation, encourages gradient alignment. This moves beyond a purely empirical contribution.
4.  The choice of four distinct domains (math, code, science, creative writing) effectively demonstrates the method's ability to handle varied types of reasoning and reward signals.
The comparison includes a representative set of baselines (joint learning, curriculum learning, gradient balancing), showing that CGPO advances the state-of-the-art.
5．Evaluating on both 3B and 7B parameter models strengthens the claims, showing that the benefits are consistent and potentially scale with model size. Using seven established benchmarks provides a robust assessment of multi-domain capability.
6. The ablation studies on domain order randomization and the mixing coefficient `α` are crucial for validating the design choices. The analysis of computational overhead directly addresses a key potential concern for adoption.

### Weaknesses
1.  The role and intuition behind the final interpolation step (`θ_new ← φ_0 + α(φ_K - φ_0)`) could be explained more clearly in the main text. While the ablation study shows its importance, a more intuitive explanation of why this interpolation stabilizes training would be helpful. Is it primarily a form of trust region enforcement or a step-size damping mechanism?
2.  The paper would be strengthened by a more explicit discussion of its limitations. For instance, the creative writing reward is based on a single LLM-as-a-judge (Qwen2.5-72B). The potential biases of this judge and how they might influence the overall multi-domain optimization are not discussed.
3.  While the baselines are well-chosen, the related work section mentions other second-order approximation methods like Shampoo, K-FAC, and SOAP. A discussion on why CGPO is a more suitable path for the RL-for-LLMs setting compared to adapting these existing approximate second-order methods would provide valuable context and further justify the novelty. The argument is implicit but could be made more explicit.

### Questions
1.How sensitive is CGPO to the number of domains (K)? Does performance plateau or degrade as K becomes very large?
 2. The method relies on multiple gradient steps per "outer" iteration. While the overhead is shown to be low, are there scenarios (e.g., with extremely large models) where even this could become a bottleneck compared to a single-update method?

### Soundness
3

### Presentation
2

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
This paper studies multi-domain optimization in RL. The authors propose a simple training design: at each (mini-batch) step, prompts are randomly sampled from all tasks. A random permutation of the task order is generated, and the model performs sequential updates on each task according to this permuted order. They show that the sequential update scheme has a theoretical connection to Newton's method where sequential updates induce cross-domain Hessian–gradient interactions, which encourages cooperation across domains  and steers optimization toward coordinated improvements over all tasks.

### Strengths
* Originality: The authors establish a novel connection between task-wise sequential updates within a mini-batch in GRPO and Newton's method and Hessian approximation. This insight and perspective are neat and enlightening. 
* Quality/Clarity: The entire paper is very well written, clearly structured, and easy to follow. Overall, it demonstrates high technical and presentation quality.

### Weaknesses
* The paper focuses on optimizing the average task accuracy. But in practice, some tasks may be more critical than others, and the proposed method does not have a mechanism to account for task-specific importance or weighting. Extending the framework to handle non-uniform task importance could make the approach more broadly applicable.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2
