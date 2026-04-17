# Review

## Summary
This paper provides both theoretical and experimental evidence of MGDL’s computational advantages. The authors establish convergence guarantees for gradient descent (GD) applied to MGDL, demonstrating greater robustness to learning-rate choices compared to SGDL. In the case of ReLU activations with single-layer grades, MGDL reduces to a sequence of convex optimization subproblems. For more general settings, the authors analyze the eigenvalue distributions of Jacobian matrices from GD iterations, revealing structural properties underlying MGDL’s enhanced stability. The authors benchmark MGDL against SGDL on image regression, denoising, and deblurring tasks, as well as on CIFAR-10 and CIFAR-100, covering fully connected networks, CNNs, and transformers.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The authors provide a rigorous convergence analysis of gradient descent for SGDL and MGDL, offering deeper insight into MGDL’s computational advantages. 
2. The authors prove that if each grade of MGDL employs a single hidden ReLU layer, the originally nonconvex optimization problem decomposes into a sequence of convex subproblems. 
3. Extensive experiments on image regression, denoising, deblurring, CIFAR-10, and CIFAR- 100 classification, including fully connected networks, CNNs, and transformers, demonstrate that MGDL consistently outperforms SGDL with greater stability. 
4. The authors analyze the impact of learning rate, showing that MGDL is more robust than SGDL. 
5. The authors study a linear approximation of GD dynamics and the eigenvalue distribution of the associated Jacobian to explain MGDL’s convergence and stability advantages.

## Weaknesses
1. The paper lacks a detailed discussion on the computational complexity of MGDL, especially regarding the overhead introduced by the multi-grade structure compared to SGDL. While the paper mentions that MGDL’s memory cost is lower than that of a single deep network since each grade trains only a shallow model, a quantitative analysis of the computational cost in terms of runtime and memory usage for each experiment would strengthen the evaluation.
2. The paper primarily compares MGDL with SGDL. It would be beneficial to include comparisons with other advanced optimization techniques, such as momentum-based methods or adaptive learning rate schedules, which are commonly used to improve training stability and convergence speed in SGDL.

## Questions
1. What is the additional computational overhead introduced by the multi-grade structure in MGDL compared to SGDL? Could the authors provide quantitative data on runtime and memory usage for each experiment?
2. Have the authors considered combining MGDL with other optimization techniques, such as momentum-based methods or adaptive learning rate schedules? If so, how did these methods perform compared to SGDL?
3. The paper mentions that MGDL reduces to a sequence of convex optimization subproblems when each grade uses a single ReLU layer. How does this convex reformulation impact the convergence speed and stability of the training process? Could the authors provide more insights into how this convex formulation contributes to MGDL's improved performance?
4. The authors analyze the eigenvalue distributions of Jacobian matrices from GD iterations. How do these eigenvalues evolve during training, and what is their impact on convergence behavior? Could the authors provide a more detailed analysis of how the eigenvalue distribution changes over time and how it relates to the stability and convergence of the training process?
5. The experiments cover tasks such as image regression, denoising, deblurring, and classification. How does MGDL perform on more complex tasks, such as object detection or segmentation? Have the authors considered these tasks in their evaluation?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4