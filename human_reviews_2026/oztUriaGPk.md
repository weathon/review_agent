# Bernoulli-LoRA: A Theoretical Framework for Randomized Low-Rank Adaptation

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Parameter-efficient fine-tuning (PEFT) has emerged as a crucial approach for adapting large foundational models to specific tasks, particularly as model sizes continue to grow exponentially. Among PEFT methods, Low-Rank Adaptation (LoRA) [Hu et al., 2021] stands out for its effectiveness and simplicity, expressing adaptations as a product of two low-rank matrices. While extensive empirical studies demonstrate LoRA's practical utility, the theoretical understanding of such methods remains limited. Recent work on RAC-LoRA [Malinovsky et al., 2024] took initial steps toward rigorous analysis. In this work, we introduce Bernoulli-LoRA, a novel theoretical framework that unifies and extends existing LoRA approaches. Our method introduces a probabilistic Bernoulli mechanism for selecting which matrix to update. This approach encompasses and generalizes various existing update strategies while maintaining theoretical tractability. Under standard assumptions from non-convex optimization literature, we analyze several variants of our framework: Bernoulli-LoRA-GD, Bernoulli-LoRA-SGD, Bernoulli-LoRA-PAGE, Bernoulli-LoRA-MVR, Bernoulli-LoRA-QGD, Bernoulli-LoRA-MARINA, and Bernoulli-LoRA-EF21, establishing convergence guarantees for each variant. Additionally, we extend our analysis to convex non-smooth functions, providing convergence rates for both constant and adaptive (Polyak-type) stepsizes. Through extensive experiments on various tasks, we validate our theoretical findings and demonstrate the practical efficacy of our approach. This work is a step toward developing theoretically grounded yet practically effective PEFT methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Bernoulli LoRA, which is a novel theoretical framework for extending and understanding LoRA methods. Moreover, this paper derives convergence rates under different algorithm such as GD, SGD, and federated GD with variant assumptions. Finally, the authors support their theoretical findings with numerical results.

### Strengths
**Comprehensive theoretical results** This paper studied Bernoulli LoRA, and proved the convergence results under different assumptions and algorithms (see Table 1) which is solid.

**Framework to analyze LoRA** Theoretical analysis of the optimization process of LoRA is difficult, and this paper proposes a novel idea to address this issue. It seems their framework can theoretically cover almost all fine-tuning problems under regular assumptions.

### Weaknesses
**Poor writing quality** Section 2: Problem Statement seems to be misplaced. The transition from Section 1 to Section 2 is non-smooth. Moreover, it seems to me that Section 1 and Section 3 are more coherent.

**Missing important features of fine-tuning** Bernoulli LoRA makes the theoretical analysis of LoRA tractable but it also misses some important features of the fine-tuning problems, such as how does the pre-trained model, relation between pre-training data and fine-tuning data affects the convergence. The results of this work lacks such interpretation.

**Weak experimental results**  The experiment section (Section 7) contains two parts: linear regression and MLP for MNIST. The size of the dataset and network architecture are too toy. Moreover, the experiments do not show or support the superiority of Bernoulli LoRA, and it makes me wonder why we study it in the first place? If the only reason is that Bernoulli LoRA is theoretically tractable, then the results of the paper seem less interesting the lack practical relevance.

**Missing literature reviews**  This paper misses discussion of several studies on LoRA optimization, such as follows.

[1] Jang, Uijeong, Jason D. Lee, and Ernest K. Ryu. "LoRA training in the NTK regime has no spurious local minima." arXiv preprint arXiv:2402.11867 (2024).

[2] Xu, Ziqing, et al. "Understanding the Learning Dynamics of LoRA: A Gradient Flow Perspective on Low-Rank Adaptation in Matrix Factorization." arXiv preprint arXiv:2503.06982 (2025).

[3] Kim, Junsu, Jaeyeon Kim, and Ernest K. Ryu. "LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly (But it Probably Won't Fail)." arXiv preprint arXiv:2502.09376 (2025).

[4] Dayı, Arif Kerem, and Sitan Chen. "Low-rank fine-tuning lies between lazy training and feature learning." Proceedings of Machine Learning Research vol 291 (2025): 1-57.

### Questions
**Question 1** What is the motivation for Bernoulli LoRA (in Algorithm 1)? Section 5.1 suggest Bernoulli LoRA can be interpreted as projecting the gradient onto a random subspace (see equation 8). However, I don't see the intuition why such approach will make the algorithm work better. Besides,  $H_{B}^{t}, H_A^t$ (in equation 8) misses definition.

**Question 2** How to interpret the dependency of $p$ in the bounds in Theorem 1 and Theorem 2? It seems as one sets $p\to 1$, the bounds get better, i.e., $1/\lambda_{min}^p, (\frac{\lambda_{max}}{\lambda_{min}})^p$. However, based on the interpretation in equation 8, $p=1$ suggests that one should only update $B$ and keep $A$ as fixed, which seems strange.

**Question 3** What is the definition of $b$ in Theorem 3, and $q$ in Theorem 4, $\omega$ in Theorem 5?

**Question 4** Why in Section 7.1, the fine-tuning loss is defined with the regularization $\sum_{j=1}^d \frac{x_j^2}{1+x_j^2}$ instead of the standard $L_2$ regularization such as $\|\|x-\tilde x^{\ast}\|\|^2$?

### Soundness
2

### Presentation
1

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
This paper introduces Bernoulli-LoRA, a theoretical framework that generalizes Low-Rank Adaptation (LoRA) methods for parameter-efficient fine-tuning (PEFT). The key idea is to use a Bernoulli random mechanism to decide whether to update one of the two low-rank matrices A or B during each iteration. The authors establish convergence guarantees under standard non-convex optimization assumptions (smoothness, Lipschitz continuity, PŁ condition) and extend the analysis to the non-smooth convex case with both constant and adaptive step sizes. Empirical results on linear regression and MNIST classification demonstrate that Bernoulli-LoRA performs comparably to RAC-LoRA and COLA, while offering a cleaner and more general theoretical foundation.

### Strengths
1. Introducing stochastic binary masks within LoRA’s low-rank structure is original and intuitively appealing.

2. The paper provides rigorous convergence theorems for multiple Bernoulli-LoRA variants, including SGD, variance reduction (PAGE, MVR), and FL settings with compression and error feedback. 

3. Experimental evidence  aligns with theoretical convergence predictions。

### Weaknesses
1. Some theoretical assumptions are  idealized for real-world applications. In particular, the convergence proofs rely heavily on Lipschitz smoothness and positive expected projection conditions that may not hold under real LoRA parameterizations, where f(W_0 + BA) is non-smooth and non-Lipschitz (as the paper itself admits). There is no empirical check of these assumptions.

2. Experiments are limited to small-scale tasks (linear regression, MNIST). No evaluation on modern large-scale or multimodal models is provided.

### Questions
1. How does Bernoulli-LoRA behave in the presence of non-i.i.d. data in Federated Learning?

2. Since f(W_0 + BA) loses Lipschitz smoothness, how realistic are the assumptions used in Theorem 1–8 for deep networks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Bernoulli-LoRA, a novel LoRA-based parameter-efficient fine-tuning (PEFT) method. At each training iteration, Bernoulli-LoRA conducts a Bernoulli trial to decide whether to optimize the $A$ or $B$ module, keeping the other fixed. Building on this framework, the authors propose seven variants of Bernoulli-LoRA tailored to different optimization scenarios, thereby addressing a broad spectrum of fine-tuning challenges. Furthermore, the paper provides a theoretical convergence analysis for Bernoulli-LoRA and all its variants, offering formal justification for the proposed approach.

### Strengths
The main strength of this paper lies in its theoretical contributions. Overall, the manuscript is well-written and presents a rigorous theoretical development. I have carefully examined all the proofs and confirm that they are correct. Moreover, the analytical results have the potential to be extended to a broader class of LoRA-based methods, opening up promising directions for future research on the theoretical understanding of convergence in PEFT frameworks. In addition, the proposed algorithm is straightforward to implement and practically applicable.

### Weaknesses
The reviewer is skeptical about the contribution of the paper, both practically and theoretically. 

+ **About the theoretical contributions:** 

- It appears that Bernoulli-LoRA is a relatively straightforward modification of RAC-LoRA [1]. Specifically, rather than deterministically alternating between the left and right sketches, Bernoulli-LoRA introduces stochasticity by performing a Bernoulli trial at each iteration to decide which module to update. Consequently, the theoretical results presented in the paper seem to be direct extensions or adaptations of prior analyses, including those from RAC-LoRA. Moreover, the theoretical advantages of this stochastic modification are not clearly justified. A detailed comparison of the convergence properties between Bernoulli-LoRA and RAC-LoRA would substantially strengthen the paper. I recommend that the authors include a remark or a subsection discussing the theoretical limitations of RAC-LoRA’s design and explaining how Bernoulli-LoRA potentially addresses these weaknesses.

- The paper frequently asserts that Bernoulli-LoRA provides a unifying framework for existing update strategies. This claim is ambitious but currently lacks rigorous theoretical or empirical substantiation. The authors are encouraged to explicitly identify the PEFT methods that fall under this framework and to formally articulate the theoretical mechanism through which Bernoulli-LoRA generalizes them.

+ **About the experimental results:** 

- Although the paper proposes multiple variants of Bernoulli-LoRA for different fine-tuning scenarios, the experimental evaluation is limited in scope and scale, which weakens the demonstration of the framework’s practical effectiveness. To better substantiate the empirical claims, it is recommended that the authors extend the experiments to more comprehensive and widely recognized benchmarks commonly used to assess LoRA-based methods—such as natural language understanding and generation tasks [2] or vision benchmarks [3].

### Questions
My main concerns are presented in the Weaknesses section. If these concerns are adequately addressed, I am willing to adjust my evaluation accordingly. 

**References:**

[1] Randomized Asymmetric Chain of LoRA: The First Meaningful Theoretical Framework for Low-Rank Adaptation. arXiv, 2024.

[2] LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

[3] V-PETL Bench: A Unified Visual Parameter-Efficient Transfer Learning Benchmark. NeurIPS, 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes the convergence of LoRA fine-tuning in the context of non-convex optimization.

### Strengths
- The theoretical results are reasonable because they are extensions of the standard results in convex and non-convex optimization.
- The study and analysis about PEFT model fine-tuning techniques are necessary.

### Weaknesses
- The setting of the paper is a simplification of the practical use of LoRA. More specifically, it considers $f(W^0 + \Delta W)$ where LoRA is applied to only one matrix. However, in practice, LoRA is applied to many matrices (e.g., key, query, value matrices of self-attention layers) across many layers of a transformer.
- The setting of the optimization problem in this paper is unclear. Particularly, it is unclear what parameters are optimized in the target optimization problem.
  - If $\Delta W$ in (1) is optimized, it refers to full fine-tuning. The objective is to prove that the LoRA updates can converge efficiently to the full fine-tuning. However, it is weird that in experimental results in Table 3, standard LoRA attains only 86% of full-parameter fine-tuning.
- The theoretical results are not surprising because they are too standard in the field.
- It would be better if the experiments are conducted with foundation models (ViT, LLMs, SDMs) because this is a common use of PEFT.

### Questions
- What is the matrix $H$ in Eq. (8)?
- Can you suggest more insights or understanding from your theories regarding how to use LoRA more efficiently in real-world applications?

### Soundness
3

### Presentation
2

### Contribution
2
