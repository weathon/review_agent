# LoRA Meets Second-Order Optimization: Towards Optimal Low-Rank Updates

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Low-rank fine-tuning is widely applied for the effective adaptation of large models. Most existing methods rely on low-rank matrix factorization, whose performance is limited by the condition number of the associated Jacobi operator. Although these methods are computationally efficient, their performance still falls short compared to full fine-tuning. To address this, we propose SoLoRA, which leverages an adaptive metric to find a low-rank approximation of the full fine-tuning gradient. This low-rank approximation can be viewed as an approximation of Hessian, effectively incorporating second-order information to achieve faster convergence and higher optimization efficiency. Furthermore, the low-rank approximation in SoLoRA is computationally simple and easy to implement, achieving a close approximation to the performance of full fine-tuning with almost no additional computational overhead. We conduct fine-tuning experiments on large language models and diffusion models, and the results consistently demonstrate that SoLoRA achieves superior performance advantages over state-of-the-art low-rank fine-tuning methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The proposed method, SoLoRA, extends LoRA-Pro by introducing a metric defined as a weighted average of gradients in the same spirit of AdaGrad and Adam. The approach is evaluated through fine-tuning experiments on GPT-2 and Mix-of-Show models.

### Strengths
The proposed approach generalizes LoRA-Pro and demonstrates improved performance in the evaluated cases.

### Weaknesses
W1. Conceptually, what motivates the introduction of the gradient-based metric to modify LoRA’s updates? If the intent is to capture second-order information, why not directly employ established second-order optimization methods (e.g., SOAP or Shampoo)? Moreover, comparing the results in Figs. 1 and 2 shows that the improvement under AdamW is considerably smaller than under SGD, suggesting that AdamW may already serve a similar role to the proposed approach.

W2. The specific design choices behind SoLoRA are not sufficiently justified. Why is the $\ell_1$ norm adopted in Eq. (4)? There are several alternative and reasonable options, such as using an $\ell_2$ norm or omitting normalization altogether.

W3. In terms of novelty, the proposed approach appears to reduce to LoRA-Pro when setting $L_t = R_t = I$. Moreover, the idea of using a moving average to adaptively precondition gradients is well established in the literature. As a result, the conceptual distinction between SoLoRA and existing methods remains unclear. The authors are encouraged to clarify the new insights or mechanisms introduced by the proposed metric beyond LoRA-Pro, and to demonstrate concretely how it results in different optimization dynamics or empirical improvements.

W4. Several recent works closely related to this paper are not cited or discussed (e.g., [1, 2]). Both methods are more computationally efficient than the proposed approach, yet the paper does not acknowledge or compare against them. A discussion on how SoLoRA differs from or improves upon these methods would strengthen the contribution.

W5. The experimental validation appears relatively small in scope. Only GPT-2 Small and GPT-2 Medium are tested, which are modest in size by today’s standards. This makes it difficult to assess whether the proposed method scales effectively to larger models. Moreover, the evaluation is conducted on a single dataset (E2E), leaving open the question of how the approach generalizes to other tasks or domains.

W6. The performance gain under AdamW is quite limited compared to baseline methods such as LoRA-Pro, further weakening the empirical support for the proposed design. 

W7. The paper does not explicitly report the runtime or memory consumption of the proposed method. Since SoLoRA aims to improve training efficiency, such metrics are essential to substantiate its practical benefits. A quantitative comparison with baseline methods in terms of computational cost and memory usage would make the evaluation more complete.

W8. The writing can be improved significantly.
- The description in Section 2.1 contains inaccuracies. For instance, the statement that “LoRA was designed to avoid the expensive SVD computation at each training step” (lines 95 - 96) is not historically correct. LoRA predates GaLore, and was not motivated by avoiding SVD computations. The authors are encouraged to revise this section for historical and conceptual accuracy.
- Line 94: "leading to time-consuming"
- Line 242 - 243: By the definition of eq.(4), $L_t$ and $R_t$ are clearly not rank-1.
 

[1] Zhang Y, Li B, Giannakis GB. RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models. arXiv preprint arXiv:2505.18877. 2025.

[2] Yen JN, Si S, Meng Z, Yu F, Duvvuri SS, Dhillon IS, Hsieh CJ, Kumar S. LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization. arXiv preprint arXiv:2410.20625. 2024.

### Questions
See above.

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
4

### Summary
The paper proposes SoLoRA, a low-rank fine-tuning method that uses an adaptive metric to approximate the full fine-tuning gradient, incorporating second-order information to improve efficiency and convergence. 

SoLoRA achieves performance comparable to full fine-tuning with minimal additional computational cost and demonstrates superior results compared to existing low-rank methods on large language and diffusion models.

### Strengths
The paper addresses an important problem of improving computational efficiency of fine-tuning methods.

Convergence rates of the proposed SoLoRA were analyzed in detail.

In some of the experimental results, SoLoRA converges faster compared to the baseline LoRA.

### Weaknesses
There are a few major issues with the paper.

First, one of the main claims of the paper is improving computational efficiency. However, this claim has not been justified experimentally.

Second, in some results, e.g. in Fig. 3, SoLoRA converges to a larger loss compared to the other LoRAs. To support the proposed SoLoRA better, additional analyses utilizing different models in different tasks in comparison with the other state-of-the-art LoRA methods should be given.

### Questions
Have you compared convergence rate of your proposed method and the other LoRA variations using other LLMs/VLMs on additional benchmarks?

Could you please provide memory consumption and running times (during training and inference) in comparison with other state-of-the-art LoRA methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Overview

This paper proposes SoLoRA, a second-order low-rank adaptation method that uses an adaptive weighted metric to approximate the full fine-tuning gradient while incorporating Hessian information. The authors demonstrate performance improvements over existing LoRA variants on GPT-2 and diffusion model fine-tuning tasks.

### Strengths
Strengths

The paper provides a clear theoretical framework connecting low-rank fine-tuning to optimization on Riemannian manifolds. The analysis of existing methods through the lens of condition number dependence and the explicit formulation of the Jacobian operator offers useful insights into why standard LoRA methods underperform compared to full fine-tuning.

The proposed adaptive metric based on diagonal approximations of the Hessian is computationally efficient with O(mn + (m+n)r² + r³) complexity per iteration and O(m+n) memory overhead. This makes the method practical for large-scale applications while still incorporating second-order information.

Experimental results consistently show improvements across multiple settings. The GPT-2 experiments demonstrate superior performance on the E2E benchmark across different model sizes and ranks, while the diffusion model experiments show better FID scores and comparable or better CLIP scores.

The closed-form solutions for optimal low-rank updates (Theorems 3.1 and 3.2) are mathematically rigorous and provide clear guidance for implementation. The paper includes thorough supplementary materials with detailed proofs and additional experimental results.

### Weaknesses
Weaknesses

The theoretical contribution is somewhat incremental, primarily combining existing ideas from AdaGrad/SOAP-style adaptive metrics with the LoRA-Pro framework. While the weighted metric approach is sensible, the novelty over simply applying preconditioned optimization to LoRA factors is limited.

The experimental evaluation has several limitations. The GPT-2 experiments use only the E2E dataset with a single evaluation protocol, making it difficult to assess generalization. The performance gains, while consistent, are often modest (e.g., 70.0 vs 69.8 BLEU for rank 16). Statistical significance testing and error bars are absent throughout.

The comparison to LoRA-Pro appears incomplete since the paper claims LoRA-Pro requires solving a Sylvester equation at each step with "extremely high computational costs," yet reports runtime comparisons showing LoRA-Pro is only marginally slower than other methods (Figures 1-2). This suggests either an unfair implementation comparison or an overstatement of LoRA-Pro's computational burden.

The adaptive metric construction using diagonal approximations Lt and Rt is heuristic rather than principled. While inspired by AdaGrad/SOAP, the specific choices (element-wise summation, normalization by l1-norm, decay factors β₁ and β₂) lack theoretical justification for why this particular formulation provides a good Hessian approximation for the LoRA optimization problem.

The claim that the low-rank approximation "can be viewed as a rank-1 approximation of the Hessian" (page 5) is not rigorously justified. The connection between the Kronecker product structure of Lt and Rt and the actual Hessian of the loss function needs more careful analysis, especially given that the Hessian is with respect to the factors B and A, not the weight matrix W.

The paper introduces multiple hyperparameters (β₁, β₂, β₃, learning rates) that appear to require careful tuning based on Tables 3-4 and 6. The sensitivity analysis is limited to learning rate variations in Figure 3, and the paper does not provide clear guidance on how to set these hyperparameters in new applications.

The diffusion model experiments use relatively small-scale personalization tasks with only two characters (Harry Potter and Hermione Granger). The qualitative improvements in image diversity mentioned for lower CLIP scores at scaling factor 0.7 are not convincingly demonstrated through systematic evaluation or user studies.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies LLM fine-tuning with LoRA, which receives great attention recent years due to its efficacy and efficiency for fine-tuning LLMs to adapt to downstream tasks. Although LoRA works well in general, there is still some potential gap compared to the performance of full parameter fine-tuning. Motivated from the previous paper, LoRA-Pro, this work proposes to let LoRA updates match with the updates using an approximation of Newton's method on the full model. The new algorithm, termed SoLoRA, achieves better empirical performance on fine-tuning GPT-2.

### Strengths
The paper provides a good summary of related works on LoRA using their framework. The derivation of the proposed algorithm is explained in details and mathematically consistent. The studied topic is well-motivated and is interesting to the general community.

### Weaknesses
1. Although the authors call their algorithm as SoLoRA, it is not a second-order method. There are lots of approximations happening in the derivation, and the Hessian is replaced by $L$ and $R$. The authors say in the paper that such rank-1 approximations of Hessian are optimal w.r.t. the generalized KL divergence and refer to the AdaFactor paper. This statement of the authors is wrong and very misleading. Indeed, what AdaFactor paper says is that $L$ and $R$ are optimal rank-1 approximations of the second moments of the gradients $G_t^2$, **but not the Hessian matrix**. There is still a gap between $G_t^2$ and the true Hessian. In this regard, the algorithm is approximating preconditioned GD with $G_t^2$ but not Newton. Therefore, it should not be called as second-order. Although there exists some work trying to approximate Hessian with Fisher information, when and why this approximation makes sense remain unclear for LLMs.

2. The major weakness of the algorithm is that the full gradients $G_t$ are always required. In Algorithm 1, $G_t$ is used to update $l_t$ and $r_t$. In Algorithm 2, $G_t$ is required to maintain the momentum $M_t$. In fact, the computation of $G_t$ is exactly what LoRA tries to avoid. It increases the memory for optimizer states and also activation storage for backpropagation. By only updating using gradients of $A$ and $B$, the savings of LoRA are made possible. In fact, according to the motivation that there is still a gap between LoRA and full parameter fine-tuning, if $G_t$, **the full gradient, is already computed, why not just do full parameter fine-tuning using it**? Moreover, the storage of the momentum $M_t$ also increases memory compared to the original LoRA, from $(m+n)r$ to $mn$.

3. The experiments are too limited. Only one model, GPT-2, is tested on a single task, E2E. What about other model family? How about even larger models? What about other tasks? The current experiments do not suffice to show that the new method achieves good performance. Also, the paper claims that their method adds no overhead, which is not verified. What are the memory consumptions compared to LoRA, say for example on the considered GPT-2 with E2E, and other larger models with different tasks?

### Questions
Other questions not asked in weaknesses:

1. In the original LoRA-Pro derivation, there is a constraint $dL\leq0$. Can authors provide some details why this is dropped in this paper?

2. Why are $B^\top L^{1/2}B$ and $AR^{1/2}A^\top$ invertible? In particular, since $B_1=0$, what is $(B_1^\top L_1^{1/2}B_1)^{-1}$?

3. Are there typos in line 8 of Algorithm 2? Should be $(1-\lambda\eta) B$ and also for A?

### Soundness
2

### Presentation
2

### Contribution
2
