# Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel

- Decision: Accept (poster)
- Scores: 6, 5, 8, 5

## Abstract
We propose conditional flows of the maximum mean discrepancy (MMD) with the negative distance kernel for posterior sampling and conditional generative modelling. This MMD, which is also known as energy distance, has several advantageous properties like efficient computation via slicing and sorting. We approximate the joint distribution of the ground truth and the observations using discrete Wasserstein gradient flows and establish an error bound for the posterior distributions. Further, we prove that our particle flow is indeed a Wasserstein gradient flow of an appropriate functional. The power of our method is demonstrated by numerical examples including conditional image generation and inverse problems like superresolution, inpainting and computed tomography in low-dose and limited-angle settings.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces conditional flows of the Maximum Mean Discrepancy (MMD) with the negative distance kernel for posterior sampling and conditional generative modeling. The joint distribution of the ground truth and the observations is approximated using discrete Wasserstein gradient flows, and an error bound for the posterior distributions is established. it is proven in the paper that the particle flow within our method indeed functions as a Wasserstein gradient flow of an appropriate functional. The paper's efficacy is demonstrated through various numerical examples, encompassing applications such as conditional image generation and the resolution of inverse problems, including superresolution, inpainting, and computed tomography in low-dose and limited-angle scenarios.

### Strengths
* The proposal of MMD flows with a "generalized" kernel kernel which is also known as energy distance or Cramer distance is new. 
* The paper can prove that the particle flow with the generalized MMD is indeed a Wasserstein gradient flow of an appropriate function.
* The paper uses the MMD flows in the setting of sampling from the posterior which is interesting and new.
* Experiments are conducted on class-conditional image-generation (MNIST, FashionMNIST, and CIFAR10) and inverse problems with medical images.

### Weaknesses
* There is no quantitative comparison in class-conditional image-generation with previous works e.g., score-based generative modeling (without using labels). Similarly, score-based generative models can also be used in medical image inverse-problem [1].
* There is no comparison with Sliced Wasserstein Gradient flows e.g., with JKO scheme. [2]
* Considering discrete flows is quite restricted. 

[1] Solving Inverse Problems in Medical Imaging with Score-Based Generative Models.
[2] Efficient Gradient Flows in Sliced-Wasserstein Space

### Questions
* Standard Sliced Wasserstein is not optimal, there are other variants e.g., [3],[4]. Is standard SW preferred in this setting?
* Can the proposed MMD flows be seen as a debiased version of Sliced Wasserstein gradient flow in the setting of discrete flows?

[3] Generalized Sliced Wasserstein Distances
[4] Energy-Based Sliced Wasserstein Distance

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes conditional MMD flows with the negative distance kernel for posterior sampling and conditional generative modelling. By controlling the MMD of the conditional distribution using the MMD of the joint distribution, the paper provides a pointwise convergence result. In addition, the paper shows that the proposed particle flow is a Wasserstein gradient flow of a modified MMD functional, and hence provides some theoretical guarantee for [1]. Finally, the paper experiments on several image generation problems and compares with other conditional flow methods.

[1] C. Du, T. Li, T. Pang, S. Yan, and M. Lin. Nonparametric generative modeling with conditional slicedWasserstein flows. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (eds.), Proceedings of the ICML ’23, pp. 8565–8584. PMLR, 2023.

### Strengths
1. The paper is well-written and clearly-organized.
2. The paper proves that the proposed particle flow is a Wasserstein gradient flow of an appropriate functional, thus providing a theoretical justification for the empirical method presented by [1].
3. Abundant generated image samples are shown in the experiments.

[1] C. Du, T. Li, T. Pang, S. Yan, and M. Lin. Nonparametric generative modeling with conditional slicedWasserstein flows. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (eds.), Proceedings of the ICML ’23, pp. 8565–8584. PMLR, 2023.

### Weaknesses
1. The novelty of the proposed method appears to be limited, since it is mainly the Generative Sliced MMD Flow [1] method applied to conditional generative modelling problems. Additionally, the proof of Theorem 3 partially follows [2].
2. The theoretical comparison with different kernels (Gaussian, Inverse Multiquadric and Laplacian [1]) and discrepancies (KL divergence, W_1 [2] and W_2 [3] distance) in Theorem 2 is insufficient.
3. The numerical results of image generation lack comparison with other methods like Generative Sliced MMD Flow in [1]. It would be better to compare the FID scores for different datasets and various methods like [1], since the proposed method adopts the computational scheme of Generative Sliced MMD Flow. It would be beneficial to compare with Conditional Normalizing Flow in the superresolution experiment and with WPPFlow, SRFlow in the computed tomography experiment.


[1] J. Hertrich, C. Wald, F. Altekrüger, and P. Hagemann. Generative sliced MMD flows with Riesz kernels. arXiv preprint 2305.11463, 2023c

[2] F. Altekrüger, P. Hagemann, and G. Steidl. Conditional generative models are provably robust: pointwise guarantees for Bayesian inverse problems. Transactions on Machine Learning Research, 2023b.

[3] F. Altekrüger and J. Hertrich. WPPNets and WPPFlows: the power of Wasserstein patch priors for superresolution. SIAM Journal on Imaging Sciences, 16(3):1033–1067, 2023.

### Questions
1. The paper states that MMD combining with the negative distance kernel results in many additional desirable properties, however it lacks convergence rate or discretization error analysis because “the general analysis of these flows is theoretically challenging”. Regarding this problem, what is the advantage of MMD over other discrepancies like Kullback–Leibler divergence or the Wasserstein distance especially for conditional generative modelling problems?
2. Is it possible to provide a discretization error analysis between discrete MMD flow and the original continuous MMD flow?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, conditional MMD flow with negative distance kernel is introduced.
The model's stability is proven by bounding the expected approximation error of the posterior distribution.

Through theoretical justification, the authors obtain convincing results by neglecting the velocity in the y-component in sliced Wasserstein gradient flows.
Then, the power of the method is also demonstrated by numerical examples including conditional image generation and inverse problems.

### Strengths
1. The theoretical justification of the proposed method is clear and detailed.
2. Several experiments are conducted to prove the power of the method.
3. Introducing negative distance kernel to MMD is a good idea and contributions are well-described.

### Weaknesses
As mentioned by the authors, the proposed approach has some limitations:

1. The model is sensitive to forward operator and noise type.
2. Lack of meaningful quality metrics to evaluate the results.
3. Realism of the computed tomography experiment results can not be guaranteed.

### Questions
1. Except computed tomography experiment, only visulization results of other experiments are given in the paper, however, it is difficult to quantitatively evaluate the result and to compare with other method. Hence, evaluation metrics need to be introduced or self-defined.

2. The related work: Neural Wasserstein gradient flows for maximum mean discrepancies with Riesz kernels, proposed similar method, what is the strength and advantage over it? and what about the performance difference?

3. Why chosing UNet? Is there a significant difference in the effect of choosing other models such ResNet and transformer.

4. As Fig.7c shows, inpainting results of CIFAR are not good enough, the generated images differ from each other greatly at the unobserved part, what is the reason? and are there any solutions to improve it.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a conditional flow of the MMD with the negative distance kernel, which can be further implemented by conditional generative neural networks with application in image generation, inpainting, and super-resolution. The authors derive the convergence of the posterior under some certain stability conditions, and relate it to a Wasserstain gradient flow. Those results extend previous investigation for sliced Wasserstein flow. The work is relatively theoretical and lacks a thorough comparison with other generative models.

### Strengths
The paper presents some interesting theories, and extends the analysis on sliced Wasserstein gradient flow.

### Weaknesses
1. It would be better to elaborate on the pros and cons of using a negative distance kernel (efficiency, sample complexity, etc).

2. The contribution is not entirely clear. Could the author comment on the effectiveness/efficiency/novelty/difficulty of the proposed method?

3. A highlight of the proof techniques used by the authors to address gradient flows with respect to MMD with negative distance kernel without mean-field approximation would help to improve the importance of this work.

### Questions
1. In Equation 4, $T$ is defined, however $T_\sharp$ is not defined.


2. Is it possible to validate the error bound via numerical experiments somehow?


3. Could the author comment on the difference between the proposed analysis and sliced Wasserstein flow, as the implementation is still based on the sliced version of it?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
