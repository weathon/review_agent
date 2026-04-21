# On Error Propagation of Diffusion Models

- Avg Score: 7.50
- Decision: Accept (poster)
- Scores: 8, 8, 6, 8

## Abstract
Although diffusion models (DMs) have shown promising performances in a number of tasks (e.g., speech synthesis and image generation), they might suffer from error propagation because of their sequential structure. However, this is not certain because some sequential models, such as Conditional Random Field (CRF), are free from this problem. To address this issue, we develop a theoretical framework to mathematically formulate error propagation in the architecture of DMs, The framework contains three elements, including modular error, cumulative error, and propagation equation. The modular and cumulative errors are related by the equation, which interprets that DMs are indeed affected by error propagation. Our theoretical study also suggests that the cumulative error is closely related to the generation quality of DMs. Based on this finding, we apply the cumulative error as a regularization term to reduce error propagation. Because the term is computationally intractable, we derive its upper bound and design a bootstrap algorithm to efficiently estimate the bound for optimization. We have conducted extensive experiments on multiple image datasets, showing that our proposed regularization reduces error propagation, significantly improves vanilla DMs, and outperforms previous baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a theoretical framework to quantify per iteration error in variational inference within the backward pass of difussion models.
The authors introduce metrics to theoretically quantify these errors and convincingly demonstrate their increase with iteration number. 
Consequently, the model's performance deteriorates as the number of iterations increases. 
To mitigate this issue, the authors propose to add the error as a regularization term upon the original loss function. 
Empirical experiments are conducted to validate the effectiveness of this proposed methodology.

### Strengths
- The paper is very well written and I really enjoys reading this paper.

- The concept of error quantification presented by the authors is both innovative and well-conceived.

### Weaknesses
- Modular Error Definition: The modular error is defined as the expected KL divergence from $p_{\theta}$ to $q$. However, the KL divergence is not symmetric. Could the authors please explain why the modular error is defined in such a way? Why not using the reversed KL divergence as in the original loss function?


- Assumptions in Theorem 3.1: The assumptions regarding the output distribution of the neural network (NN) following a standard Gaussian distribution and the entropy reduction with iteration number appear quite strong. Can the authors justify these two assumptions empirically?
    

- Technical Issues in the Proof:

    - In the proof of Proposition A.1 in Appendix A, equation (19) holds only in the limit as $T\to\infty$. The current proof falls short of demonstrating Proposition A.1 for finite values of $T$.

    - Several typographical errors in equation (24) in Appendix B raise concerns about the validity of the proof. Specifically, the $p_{\theta}$ term in both the numerator and denominator should be consistent and denoted as $p_{\theta}(x_{t-1}|x_{t})$ in order for equation (29) to hold. Additionally, the first term in the second line of (24) appears to be missing a logarithmic notation, and the third line of (24) appears inconsistent with the current notation.

- Hyper-Parameter Selection: The authors should provide further clarification on the process for selecting hyper-parameters. Notably, the weight assigned to the regularization term is a critical factor that requires elucidation.
How to select the hyper-parameters? One such important factor is the weight of the regularization term, could the authors please clarify?

- Minor Issues:

    - The notation for conditional expectation should be corrected throughout the paper to $E_{X_{t}|X_{t+1}}$.
    
    - In Section 3.2, second paragraph, "get ride of" should be corrected to "get rid of".
    
    - The proof for (14), establishing the boundedness of the KL divergence by the MMD, rests on the assumption that $\log(1-x/4)$ is well-defined, which only holds when $x<4$. The authors should address and clarify this point.

### Questions
Please refer to the questions in the previous section.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper analyses the error propagation/accumulation in DMs across iterations. The authors prove and empirically verify that the error in DMs cumulatively increases. To minimize this cumulative error as regularization, the authors prove tractable estimates which tightly bound this error, which is then used as a proxy. The authors empirically show the proposed method reduces the cumulative error, and increases generation quality across multiple datasets.

### Strengths
1. The theoretical framework and the bounds on error propagation through DMs are useful for analyzing robustness of DMs.
1. The proposed method results in strong significant improvements across a range of datasets.
1. The proposed method successfully decreases cumulative error in DMs

### Weaknesses
1. The proposed method requires significant compute overhead, so gains need to be weighed against this increase in compute.

### Questions
1. Contemporaneous work[1] (released after submission deadline) also analyses the sensitivity of DMs to error propagation, and aims to bound this error by scaling the long skip-connections. While the method in [1] is sufficiently different to the proposed method, could the authors comments on this? It would appear that [1] also bounds the error, without the computational overhead.
2. The proposed method has significant computational overhead (Figure 4). How does the comparisons to  baselines (Table 1) change at equal wall-clock time?



[1] https://arxiv.org/abs/2310.13545

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work analyze the error propagation of diffusion models by introducing the modular error (KL divergence between the reverse conditional distribution at each step), the cumulative error (KL divergence between the marginal distribution at each step). And then introduce an regularization loss based on MMD estimation for the cumulative loss to reduce the cumulative error and improve the sample quality of the trained diffusion model.

### Strengths
- The proposed method is easy to understand and the writing is clean and easy to follow.
- The error propagation of diffusion models is an important question and worth to be studied. The topic is important.

### Weaknesses
Major:

- The assumption in the core theorem is absolutely wrong.
  - "suppose that the output of neural network $\epsilon_\theta$ follows a standard Gaussian", which cannot be true. Because the noise-pred model corresponds to the denoising score matching loss, it is proved that the ground truth of such model is propotional to the score function of the distribution, i.e., $\nabla_{x_t} \log q_t(x_t)$. For a small $t$, such score function is quite complex and cannot be a simple and single-mode Gaussian distribution, and is far different.
- Remark 3.2 is not rigorous. The proof requires $T$ goes to infty, but it is not true in practice.
- Lack of detailed settings of experiments: what is the sampling algorithm for obtaining the FID results? What is the detailed network structure (e.g., layer structure and number of hidden neurons) and amount of parameters?

### Questions
1. Please address and fix the proof of the main theorem.

2. Please add more detailed descriptions of the experiment settings to ensure reproducibility.

=====================

I've carefully read the authors' responses to other reviewers and AC. I deeply appreciate the authors' efforts to address the concerns, and now I don't have more questions. I think it is a quite interesting and useful technique for improving the training procedure of diffusion models, with the validation of the fine-tuning experiment results and the EDM results. So I raise the score to 6.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors investigate error propagation in diffusion models. They develop a framework to define error propagation for diffusion and connect error propagation to generation quality. This enables them to use the measured error as a regularization term during the diffusion model training, improving the generation results of models in small-scale experiments.

### Strengths
- The proposed method is a novel approach to measuring error propagation in diffusion models and offers a new perspective on diffusion model training. The authors argue that apart from making the denoiser network more accurate, which has been the main focus of the literature so far, it is also important to regularize such that the denoiser is also robust to errors in the input during inference. This could have significant impacts on the broader diffusion generative model community.

- The presented methodology is principled and well-explained. The experiments clearly demonstrate the success of the proposed solution in mitigating error propagation in diffusion models.

### Weaknesses
- The authors briefly address the trade-off between increased training time and reduced error propagation (resulting in better FID) for the 32x32 images of CIFAR and ImageNet but do not mention their CelebA experiments on 64x64 images. It is not clear if the benefits scale with the image sizes without an increased overhead as it is possible that the error estimate requires more samples or a larger sampling length $L$.

### Questions
- Would it be possible to fine-tune pre-trained diffusion models with the regularization term to mitigate this error propagation a-posteriori? If the training time of adding the regularization makes training larger models prohibitive it would be interesting to explore whether it is possible to tune the denoiser network after having trained with just the $L^{nll}$ loss.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
