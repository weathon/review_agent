# Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Sharpness-Aware Minimization (SAM) enhances generalization by minimizing the maximum training loss within a predefined neighborhood around the parameters. However, its practical implementation approximates this as gradient ascent(s) followed by applying the gradient at the ascent point to update the current parameters. This practice can be justified as approximately optimizing the objective by neglecting the (full) derivative of the ascent point with respect to the current parameters. Nevertheless, a direct and intuitive understanding of why using the gradient at the ascent point to update the current parameters works superiorly, despite being computed at a shifted location, is still lacking. Our work bridges this gap by proposing a novel and intuitive interpretation. We show that the gradient at the single-step ascent point, when applied to the current parameters, provides a better approximation of the direction from the current parameters toward the maximum within the local neighborhood than the local gradient. This improved approximation thereby enables a more direct escape from the maximum within the local neighborhood. Nevertheless, our analysis further reveals two issues. First, the approximation by the gradient at the single-step ascent point is often inaccurate. Second, the approximation quality may degrade as the number of ascent steps increases. To address these limitations, we propose in this paper eXplicit Sharpness-Aware Minimization (XSAM). It tackles the first by explicitly estimating the direction of the maximum during training, and addresses the second by crafting a search space that effectively leverages the gradient information at the multi-step ascent point. XSAM features a unified formulation that applies to both single-step and multi-step settings and only incurs negligible computational overhead. Extensive experiments demonstrate the consistent superiority of XSAM against existing counterparts across various models, datasets, and settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new variant of SAM optimizer, namely, XSAM that provides a more faithful and effective estimation of the gradient with respect to the current parameters. The authors argue that the gradient at the single-step ascent point offers a better approximation of the direction from the current parameters towards the maximum within the local neighborhood than the local gradient. And they further justify this observation under a second-order approximation, theoretically. Extensive experiments are also presented to validate the efficacy of the proposed optimizer.

### Strengths
This paper is well-motivated and easy to follow. Actually, SAM optimizer has attracted a lot of attention and many works have been devoted to improve the generalization performance of SAM. In this work, the authors identified a critical issue in estimating the gradient descent direction and proposes a variant of SAM that unifies one-step and multi-step versions.

### Weaknesses
While the authors have presented their work in a good shape, I still have serveral questions:

 - In line 22, the authors claim that **thereby enabling a more direct escape from the maximum within the local neighborhood**. Actually, I do not catch why is this case. As Figure 1a suggests, descending along $g_0$ quickly steers the optimization process towards the low-loss regions,  then $g_1@\theta_0$,  and $g_1$ gradually mitigates this issue. Frankly speaking, I do not know how it relates to the maximum of the local neighborhood.
 
 - In line 72, the authors states that **...eveal that the approximation by the gradient at the single-step ascent point is often inaccurate**. Possibly, here might be a mistake. Please recall that Figure 1a is visualized by spanning $g_0$ and $g_1$. And, the direction of $g_1$ is heavily dependent on the perturation radius $\rho$.  Therefore, the loss peak might vary significantly. It would be much better to visulize the true gradient with respect to the current parameters as well. Moreover, what's the value of $\rho$?
 
 - On the theoretical respect, my biggest concern is that the proof of Proposition 1 is too hand-wavy. In line 810,  **for sufficiently large $\rho_m$, the $\rho_m^2$ term dominates**. I believe here deserves further investigation because the taylor-expansion holds only for small $\rho_m$ and the remainding terms cannot be simply removed. Moreover, the assumption that Hessian $H$ is positive definite also requires careful handling and experimental verification. Idealy, I would suggest to plot $L(\theta_0+\rho_m\frac{g_1}{\|g_1\|})$ versus $L(\theta_0+\rho_m\frac{g_0}{\|g_0\|})$ for different combinations of $\rho$ and $\rho_m$.

- In Experiments section, more baselines should be included, such as ASAM, FisherSAM, etc. Particularly, line 16 of Algorithm 1 rescales the descent direction with $\|g_k\|$. I am wondering how it performs when rescaled with the base gradient $\|g_0\|$? The result of a recent study [1] seems to suggest it works. Moreover, how XSAM performs on corrupted dataset such as CIFAR-10/100-C should be reported as well.

[1] Tan et al., Stabilizing sharpness-aware minimization through a simple renormalization strategy, JMLR, 2025.

### Questions
please see Weaknesses.

### Soundness
2

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
This paper revisits Sharpness-Aware Minimization (SAM) and provides a new interpretation of its mechanism: the gradient at the ascent point more accurately approximates the direction toward the worst-case loss in a neighborhood than the local gradient. Motivated by visualization and theoretical analysis showing that this approximation is often inaccurate and deteriorates under multi-step ascent, the authors propose XSAM, which explicitly estimates the ascent direction within a two-dimensional subspace via spherical interpolation. XSAM dynamically adjusts the interpolation factor and updates parameters in the opposite direction to escape local maximum more effectively. Extensive experiments validate the effectiveness of the proposed method.

### Strengths
- The paper is generally well-written and easy to follow.
- The authors perform visualization studies to show how single-step SAM gradient directions better approximate ascent directions within the neighborhood, while multi-step SAM may degrade. These visualizations ground the theoretical intuition in empirical phenomena.
- Despite the additional probing steps, the runtime overhead remains negligible. The method is compatible with SAM , making it practical and easy to integrate into real-world training pipelines.

### Weaknesses
- The underlying motivation for using $-v(\alpha^*)$ as the final gradient descent direction remains unclear. Following the direction of $-v(\alpha^*)$ appears to encourage moving away from a local neighborhood maximum. However, this does not necessarily guarantee convergence toward a flatter minimum. Additional clarification and theoretical justification would strengthen the argument.
- In the experimental section, the paper primarily compares the proposed method against standard SAM and SGD. Given that numerous SAM variants have been discussed in the related work, including stronger SAM-based baselines would provide a more rigorous and convincing empirical evaluation of the proposed method's effectiveness.

- More representative SAM-based methods, such as GSAM [1], GAM [2], ImbSAM [3], CC-SAM [4], and Focal-SAM [5] should be included in the related work for a more comprehensive review.

-----

[1] Surrogate Gap Minimization Improves Sharpness-Aware Training, ICLR 2022

[2] Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization, CVPR 2023

[3] ImbSAM: A Closer Look at Sharpness-Aware Minimization in Class-Imbalanced Recognition, ICCV 2023

[4] Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition, CVPR 2023

[5] Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification, ICML 2025

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors study how the sharpness-aware minimization (SAM) algorithm works and how to improve it. 
They first show that the single-step ascent used in SAM, when applied to current parameters, is a better approximation of the maximum over a ball, compared to the naive gradient. This shows why the SAM algorithm is superior in practice, as it is indeed minimizing the maximum loss over a ball. 
Indeed, they show that the approximation of max loss with naive gradient in a single step is often inaccurate, and the quality gets worse if we consider multi-step ascent, a phenomenon observed in practice. 

 
In Proposition 1, they show that while SAM provides a better approximation of the maximum loss around parameters (up to a ball), linear combinations of SAM updates and naive gradients can be even better. They thus propose XSAM: 
They probe in the hyperplane of the naive gradient and the SAM gradient, using spherical linear interpolation (Eq. 6). It needs some maximization over a parameter alpha.
The proposed method proposed only a small additional computational overhead.
They conclude the paper with experiments supporting their proposed method.

### Strengths
- The theoretical work on explaining the success of the SAM algorithm in minimizing the SAM objective, compared to the naive gradient, is of potential community interest. 


 - The paper is probing between the SAM and naive gradient method, and it shows that it will strictly improve the performance, which is of potential interest to the community. The experiments also support this.

### Weaknesses
- Long sentences, hard to follow. The paper would benefit from better writing, focusing on short and clean sentences.

### Questions
This is an interesting paper, and I believe it makes a good contribution to our understanding of why SAM works and how to improve it. 

My only concern is that the paper lacks good writing. I appreciate that the authors provided an explanation of how they used LLMs just for polishing sentences, but the fact that there are long sentences in the paper is quite annoying (sometimes LLMs suggest them as a way to polish the paper, but they indeed make reading the paper harder).  For example, Line 19-24 of abstract, the main question of the paper, is introduced in a very, very long sentence, hard to follow. Please modify the abstract and other parts of the paper, and make sure the message is delivered clearly. The paper, in my opinion, needs major changes. 


- (minor) Line 11 in the algorithm: Where is the definition of v(alpha) within the algorithm?

### Soundness
3

### Presentation
2

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
This paper revisits Sharpness-Aware Minimization (SAM) and introduces eXplicit Sharpness-Aware Minimization (XSAM), which improves upon SAM by explicitly estimating the direction toward the local maximum loss within a neighborhood, rather than relying on the potentially inaccurate gradient at the ascent point. XSAM consistently outperforms SAM and its variants across various models and datasets in both single-step and multi-step settings. The authors provide both theoretical justification and empirical visualizations to explain why SAM’s approximation can be suboptimal and how XSAM addresses these limitations.

### Strengths
1. XSAM provides a more accurate and adaptive estimation of the direction toward the local maximum, leading to better generalization.

2. The method is unified across single-step and multi-step settings and shows consistent improvements over SAM with minimal computational overhead.

3. The paper offers a clear theoretical and intuitive explanation of SAM’s limitations, enhancing understanding of sharpness-aware optimization.

### Weaknesses
1. XSAM introduces additional hyperparameters (e.g., search range for α and update frequency), which may complicate hyperparameter tuning.

2. Although the overhead is small, XSAM still requires extra forward passes for direction estimation, slightly increasing computational cost.

3. Although the paper critiques multi-step SAM’s degradation, the comparison with existing multi-step variants (like MSAM or LSAM) could be more extensive to fully establish XSAM’s superiority in that regime.

### Questions
1. Your 2-D search is restricted to the plane spanned by v_{0} and v_{1}.
What theoretical or empirical evidence guarantees that the true worst-case perturbation   \delta^{*} (i.e., the arg-max of L(\theta+\delta) inside the \rho-ball) actually lies in this plane, and how might the method behave if the Hessian is highly anisotropic or the loss landscape is strongly non-quadratic?

### Soundness
4

### Presentation
3

### Contribution
3
