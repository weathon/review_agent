# ParaSolver: A Hierarchical Parallel Integral Solver for Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
This paper explores the challenge of accelerating the sequential inference process of Diffusion Probabilistic Models (DPMs). We tackle this critical issue from a dynamic systems perspective, in which the inherent sequential nature is transformed into a parallel sampling process. Specifically, we propose a unified framework that generalizes the sequential sampling process of DPMs as solving a system of banded nonlinear equations. Under this generic framework, we reveal that the Jacobian of the banded nonlinear equations system possesses a unit-diagonal structure, enabling further approximation for acceleration. Moreover, we theoretically propose an effective initialization approach for parallel sampling methods. Finally, we construct \textit{ParaSolver}, a hierarchical parallel sampling technique that enhances sampling speed without compromising quality. Extensive experiments show that ParaSolver achieves up to \textbf{12.1× speedup} in terms of wall-clock time. The source code is publicly available at https://github.com/Jianrong-Lu/ParaSolver.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a framework that generalizes the sequential sampling process of diffusion models as solving a system of banded nonlinear equations. Applying the Newton-Raphson method to solve the nonlinear equations then yields a corresponding parallel sampling algorithm for diffusion models. By utilizing the unit-diagonal structure of the banded nonlinear equations' Jacobian matrices, the authors further simplified the updating rules of the parallel algorithm. Extensive numerical experiments were also conducted to show that the ParaSolver algorithm proposed in this paper can indeed accelerate the inference time of diffusion models compared to existing implementations based on parallel sampling.

### Strengths
1. This paper has provided a complete literature review of related work on accelerating diffusion models via parallel sampling. Also, both the theoretical and algorithmic results in this paper are presented in a relatively clear way to follow. 

2. A complete set of large-scale numerical experiments on the Imagenet and LSUN-Church datasets are included to justify the acceleration achieved by the proposed ParaSolver algorithm.

### Weaknesses
1. The authors mentioned in lines 366-367 that the ParaTAA algorithm proposed in [1] needs to be implemented for comparison as it has yet to be integrated into the Diffusers library. However, given that there are only a few empirical works on combining parallel sampling with diffusion models, the reviewer thinks it would be essential for the authors to implement ParaTAA and use it as one extra baseline. Moreover, it might also be necessary for the authors to compare ParaSolver with approaches that accelerate diffusion models from other aspects, such as knowledge distillation [3-5], restart sampling [6], and self-consistency [7]. Furthermore, the authors should consider releasing the code used for implementing the ParaSolver algorithm. 

2. There are some minor issues regarding the presentation of the paper. For instance, the phrase "to fast construct a set of more precise initial values that conform to the Definition 1" in lines 296-297 doesn't seem quite right. It can be possibly rephrased as "to construct a set of more precise initial values that conform to the Definition 1 quickly". Moreover, the authors might also consider adding a few figures to illustrate the ParaSolver algorithm more vividly, just as what has been done in previous work [2].

### Questions
1. The reviewer's main question about the design of the ParaSolver algorithm is the claim in lines 244-247 of the paper. Specifically, the authors proposed to approximate the Jacobian term $\frac{\partial}{\partial \hat{X}^{(k)}_{t_n}}\Phi(t_{n+1}, t_n, \hat{X}^{(k)}_{t_n})$ with the identity matrix in the original update rule (11). Could the authors discuss which specific parts in the cited papers on Jacobian-free backpropagation (lines 245-246) actually used similar techniques? Furthermore, would it be possible for the authors to provide some mathematical intuitions on why the identity matrix should work here? Is it possible to derive some error bounds via numerical analysis?

References:

[1] Tang, Z., Tang, J., Luo, H., Wang, F. and Chang, T.H., 2024, January. Accelerating parallel sampling of diffusion models. In Forty-first International Conference on Machine Learning.

[2] Shih, A., Belkhale, S., Ermon, S., Sadigh, D. and Anari, N., 2024. Parallel sampling of diffusion models. Advances in Neural Information Processing Systems, 36.

[3] Luhman, E. and Luhman, T., 2021. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388.

[4] Meng, C., Rombach, R., Gao, R., Kingma, D., Ermon, S., Ho, J. and Salimans, T., 2023. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14297-14306).

[5] Salimans, T. and Ho, J., 2022. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512.


[6] Xu, Y., Deng, M., Cheng, X., Tian, Y., Liu, Z. and Jaakkola, T., 2023. Restart sampling for improving generative processes. Advances in Neural Information Processing Systems, 36, pp.76806-76838.

[7] Song, Y., Dhariwal, P., Chen, M. and Sutskever, I., 2023. Consistency models. arXiv preprint arXiv:2303.01469.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors present an interesting extension of previous work for inference in DPMs. The general idea is to formulate the solution to the ODE or SDE not as a sequential integration, but instead look at it as solving a set of nonlinear equations, done either via fix-point iteration or utilizing root finding algorithms. While this class of approaches does not improve the computational effort per se, it can lead to reduced wall-clock time by using less evaluation points compared to what is necessary when sequentially integrating the differential equation.

The paper proposes a unified framework that encompasses previous approaches as extreme cases. This results in a set of banded nonlinear equations. One key insight of the authors is to realize and proof that the banded system posses a unique and unbiased solution. They then further utilize the Newton method of root finding to accelerate the fix-point iterations. For this, one needs to calculate the Jacobian matrix. This, in general, is computationally prohibitive. An approximation scheme is proposed, where only the diagonal of the Jacobian is used, and the off-diagonal terms are set to unity. This results in an only modest increase in function evaluations over a sequential solution, indicating in addition to the reduced wall-clock time only a small increase in computational cost.

The achieved scores are on par with previous methods. A sizeable speed up in terms of wall-clock time is achieved leading to a better user experience. This is done without an massive increase in computational cost.

### Strengths
I believe this paper is generally well written and makes an relevant contribution to the field.

All claims are well supported by experiments, and the analyses appear sound.

### Weaknesses
- I believe the differences to the established methods utilizing fixed-point iterations for DPMs and their advances such as utilizing the Anderson acceleration used in previous work could be made clearer. It is currently not clearly mentioned that ParaTAA utilizes an conceptually similar idea. Albeit of course the approaches are different, they share common ideas which do not become clear without reading the literature carefully. I would encourage the authors to authors to rework the related work section and mention the differences to the other works more clearly.
- It would be interesting to see by how much the number of necessary iterations to reach the threshold decreases by utilizing the Newton method. I recommend the authors include an ablation study showing how the number of iterations and convergence are affected by (1) using the Newton method vs. fixed-point iteration, and (2) approximating vs. fully computing the Jacobian, on a toy problem.

### Questions
I have only a few minor comments:
- Please improve Figure 1 by using higher resolution, adding axis labels, and using a consistent font and style with the other figures in the paper.
- Figure 5: There is, to me, no discernible difference between the images for different N. Can the authors comment on this? Why do we see such a clear difference for DDPM, but not for DDIM? It would be good if the authors either (1) provide a quantitative analysis of the differences between results for different N, if they exist, or (2) explain why DDIM results are less sensitive to N compared to DDPM.
- In the Table 1 & 2, the results are ordered as DDPM, DDIM, DPMSolver, whereas in the figures the order is DDIM, DPMSolver, DDPM. I would appreciate some reordering to make it consistent.

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
3

### Summary
The authors present an approach to accelerating the inference of diffusion probabilistic models (DPMs). They transform the problem of sequential sampling of DPMs into one of solving banded nonlinear equations. The Jacobian of the nonlinear system, required by Newton's method for rootfinding (aka Newton-Raphson) is unit block-lower-banded (1 on the diagonal, bands below), allowing for efficient parallel solution through a simple recurrence relation. The authors also present an initialization procedure that accelerates convergence. Finally, they combine this framework with a sliding window technique to conduct parallel iterations only a subset of the points. The combined approach is then evaluated on StableDiffusion-v2 and the LSUN Church pixel-space diffusion model, and demonstrates large speedups on inference without a loss in visual quality.

### Strengths
There has been a surge of recent interest in fast parallel sampling of diffusion models. The state of the art for parallel sampling, to the best of my knowledge, appears to use Picard iterations to solve the nonlinear system of equations. The authors of this work make a few important contributions, all of which serve to accelerate convergence: (1) they use Newton's rootfinding method, which converges quadratically to the root for smooth enough functions; (2) they leverage the banded structure of the Jacobian to accelerate their solver; (3) they come up with a good initialization for Newton so it in fact converges; (4) they batch their parallel sampling and denoising so that it only happens within a sliding window.

### Weaknesses
1. Newton's method for rootfinding converges rapidly only if the function one is rootfinding on is sufficiently smooth. The authors should discuss the smoothness properties of the nonlinear system and how it impacts the convergence of the Newton solver, and also comment on the theoretical guarantees and limitations of their approach in this context.

2. If the nonlinear residual for the nonlinear system has a complicated landscape, Newton can easily get stuck. The state of the art in optimization is to either use trust-region Newton methods or use quasi-Newton. The authors skirt around this issue altogether and count on their results and experiments to drive their point home. It would be useful to see the loss landscape as a function of, say, two of the most "important" unknowns (determined for instance by PCA) or the eigenvalues of the kernel matrix of the neural tangent kernel to determine if Newton is the right choice for this problem. Alternatively, if the authors could justify why these failure modes don't occur in PDMs, that would also suffice.

3. How are Equation 12 and 13 justified? If the Jacobian term in the paragraph below Equation 11 is expensive to compute, why not approximate it? Newton's convergence rate requires at least an estimate for the Jacobian. Using the identity matrix instead effectively reverts Newton to a first-order method. Did the authors experiment with alternatives? Please provide theoretical/empirical justification for using the identity matrix approximation and discuss any experiments you conducted with alternatives.

4. Rootfinding can be inherently unstable. Did the authors investigate other alternatives, such as optimization-based methods? Why did the authors choose one over another?

5. This is minor, but I would've picked a less generic name for the paper. "ParaSolver" could imply a large number of things, but this is mainly a Newton-based parallel solver for PDMs. Consider a name change.

### Questions
1. See the Weaknesses section above. These must be addressed.

2. The language in the paper hinders the presentation occasionally. For instance, the second paragraph of the related work section (Section 2) was challenging to read, primarily due to strange use of passive voice. There are similar issues throughout the paper. I suggest reframing to active voice wherever possible to improve clarity.

3. Section 4.2, below equation (9): What is the "reverse of Jacobian matrix"? Do the authors mean the inverse? 

4. The authors separately explore tolerance and speedup in the results. I'd like to know which tolerance leads to the best speedup without compromising visual results. The authors should add a new graph with this extra information.

### Soundness
4

### Presentation
3

### Contribution
3
