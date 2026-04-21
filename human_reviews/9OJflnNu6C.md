# Controllable Unlearning for Image-to-Image Generative Models via $\epsilon$-Constrained Optimization

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
While generative models have made significant advancements in recent years, they also raise concerns such as privacy breaches and biases. Machine unlearning has emerged as a viable solution, aiming to remove specific training data, e.g., containing private information and bias, from models. In this paper, we study the machine unlearning problem in Image-to-Image (I2I) generative models. Previous studies mainly treat it as a single objective optimization problem, offering a solitary solution, thereby neglecting the varied user expectations towards the trade-off between complete unlearning and model utility. To address this issue, we propose a controllable unlearning framework that uses a control coefficient $\epsilon$ to control the trade-off. We reformulate the I2I generative model unlearning problem into a $\epsilon$-constrained optimization problem and solve it with a gradient-based method to find optimal solutions for unlearning boundaries. These boundaries define the valid range for the control coefficient. Within this range, every yielded solution is theoretically guaranteed with Pareto optimality. We also analyze the convergence rate of our framework under various control functions. Extensive experiments on two benchmark datasets across three mainstream I2I models demonstrate the effectiveness of our controllable unlearning framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors study the problem of machine unlearning (MU) in image-to-image (I2I) generative models. Unlike prior studies, this approach diverges from a single objective to better consider the tradeoff between unlearning completeness and model utility, offering more flexibility for varying user needs. Specifically, the authors first reformulate the bi-objective MU problem into a constrained optimization problem and then propose a gradient-based algorithm to find Pareto optimal solutions. The proposed algorithm comes with a theoretical guarantee for convergence. Additionally, empirical results show that the proposed method provides a good balance between the two objectives, performing competitively among baselines.

### Strengths
1. The proposed algorithm is well-motivated and comes with a theoretical guarantee for convergence, laying a solid theoretical foundation for application.
2. The authors identify an overlooked issue in MU for I2I generative models in previous works: the failure to cater to varying user expectations in the real world, $i.e.,$ lack of controllability. Based on this observation, they derive a novel solution to this new bi-objective problem, which has practical significance for improving I2I generative models.
3. The empirical findings align with the theoretical results, demonstrating that the solutions found by the proposed algorithm achieve good performance in terms of both objectives, $i.e.,$ unlearning completeness and model utility.

### Weaknesses
1. The definition of unlearning completeness in the paper is problematic. The paper uses the KL divergence between distributions of forget data and reconstructed data to evaluate the completeness of unlearning, ultimately approximating it with the L2 loss. However, both losses are not ideal criteria for assessing unlearning performance, as they are defined in pixel space and disregard the original image manifold. Generative models can exploit this by outputting inconsistent pixel values when operating on the forget set, leading to suboptimal unlearning results. This is evident in the artifacts in reconstructed image examples from the forget set in Appendix F, $e.g.,$ in the inpainting task.
2. The proposed algorithm doubles memory usage, as it requires storing two separate model gradients. It also involves model-level gradient operations (as described in Algorithm 1, Line 8), making it more complex than other baselines. This can be less practical for larger models. A detailed computational complexity comparison and a discussion on memory usage would be helpful.
3. Confusing evaluation. In Table 1, the Inception Score (IS) appears in both columns for the forget set and retain set. It is unclear why IS should be "the less the better" (if that is the meaning of the down-arrow) for the forget set. If unlearning completeness is linked to low-quality generation, then the objective becomes trivial—a simple classifier to detect forget data would suffice. Echoing my previous point in W1, the generative model should at least produce a natural or similar image, even if the input is out-of-distribution. This requirement is completely overlooked here.
4. Minor typos. For example, in Equation 1, $I_\theta=D_\phi(E_\gamma(\mathcal{T}(x)))$ should be $I_\theta=D_\phi(E_\gamma(x))$ to be consistent with the rest of the text.

### Questions
Is the proposed method applicable to text-guided I2I generative models, such as image editing models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submission formulates the controllable I2I unlearning problem as a $\epsilon$-constrained problem, which differs from the prior objective. By reformulating the problem as a $\epsilon$-constrained bi-objective function, two Pareto optimal solutions and the valid range of the control coefficient $\epsilon$ can be obtained. Furthermore, the authors provide a theoretical analysis of the convergence of the proposed method under various control functions used to govern the direction of parameter updates. The experimental results on two well-known benchmarks show the effectiveness over the mentioned baselines.

### Strengths
- The proposed method is sound. The proposed method reformulates the I2I unlearning problem by integrating the $\epsilon$-constrained method which is widely used in multi-objective optimization. This integration makes the unlearning degree controllable and brings a few theoretical merits, such as convergence analysis.
- This submission is well written and organized, which reduces the difficulty in reading and comprehending.

### Weaknesses
- This could be an improvement of [1] based on $\epsilon$-constrained method. Technically, please provide the specific design of $\epsilon$-constrained optimization for the I2I unlearning problem. And why $\epsilon$-constrained method is required to integrate with the I2I unlearning problem?
- Some claims are not evaluated. For instance, in line 70,  how the challenge ``First and foremost, this approach offers a solitary resolution,..’’ is addressed? 
- Evaluation of different crop sizes should be conducted. In practice, not only the degree of forgetting but also the size of crop area is defined by users.

[1] Machine unlearning for image-to-image generative models, ICLR 2024

### Questions
- Why the results of Composite Loss is different from those reported in [1]? Please provide more details of implementation differences about it.
- According to Fig.4, why the visualization of the retained set of MAE is changed after unlearning? This is quite different from [1].
- Can you provide experimental results to demonstrate the proposed enjoys better unlearning efficacy than other methods? The theoretical results sometimes are different from real practice.

[1] Machine unlearning for image-to-image generative models, ICLR 2024

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a machine unlearning approach for generative image-to-image models. Machine unlearning algorithm in generative domain aims to make the model forget a specific subset samples (for e.g. defined by classes) while retaining its generalization capability on the other samples in order to address issues related to privacy and biases. The paper proposes a controllable unlearning algorithm flexible enough to balance between quality/degree of unlearning concepts and the model’s generalization capabilities. The approach uses a gradient based method to solve a constraint optimization objective where the constrain is to forget a certain specified set while retaining its reconstruction quality on remaining samples. The paper also shows theoretical analysis of its approach using Pareto optimality. The paper shows quantitative and qualitative results on in-painting/out-painting tasks to demonstrate the efficacy of the proposed approach.

### Strengths
The paper explores an unlearning approach for generative image-to-image models that uses gradient based method to solve a constrained optimization objective.
The paper explains the issues present in the current machine unlearning domain and address these issues using a controllable optimization where the users have control over the unlearning optimization (model unlearning while maintaining model generalization). The proposed framework shows better results on ImageNet-1k and Places-365 dataset for in-painting tasks compared to other baseline unlearning approaches. The paper provides detailed ablation experiments and theoretical analysis to explain its proposed algorithm. The paper is well-written, easy to follow and contains a pseudocode that explains the methodology clearly.

### Weaknesses
It would helpful for the reader to see some discussions around the robustness of the concepts removal. For example is it possible to use some attack that resurfaces the forget set, for example as shown in paper Petsiuk, Vitali, and Kate Saenko. "Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models." arXiv preprint arXiv:2404.13706 (2024).


It would be helpful for the readers if some more related unlearning papers are added as references:


[1] Petsiuk, Vitali, and Kate Saenko. "Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models." arXiv preprint arXiv:2404.13706 (2024)

[2] Kumari, Nupur, et al. "Ablating concepts in text-to-image diffusion models." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Questions
It would be helpful if the paper can answer/comment on the following question/suggestion:

1. Is it possible to formulate the unlearning objective that simply in-paints with background content ( i.e. instead of predicting a gaussian type patch in the image for in-painting task, the model predicts the background and does not generate the subject that is to be forgotten). Does this require modification in the formulation that uses Divergence(P_Xf | N(0, sigma)) as condition.

### Soundness
3

### Presentation
4

### Contribution
3
