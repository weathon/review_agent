# Consistency Model is an Effective Posterior Sample Approximation for Diffusion Inverse Solvers

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Diffusion Inverse Solvers (DIS) are designed to sample from the conditional distribution with a pre-trained diffusion model an operator and a measurement derived from an unknown image. Existing DIS estimate the conditional score function by evaluating operator with an approximated posterior sample. However, most prior approximations rely on the posterior means, which may not lie in the support of the image distribution and diverge from the appearance of genuine images. Such out-of-support samples may significantly degrade the performance of the operator, particularly when it is a neural network. In this paper, we introduces a novel approach for posterior approximation that guarantees to generate valid samples within the support of the image distribution, and also enhances the compatibility with neural network-based operators. We first demonstrate that the solution of the Probability Flow Ordinary Differential Equation (PF-ODE) yields an effective posterior sample with high probability. Based on this observation, we adopt the Consistency Model (CM), which is distilled from PF-ODE, for posterior sampling. Through extensive experiments, we show that our proposed method for posterior sample approximation substantially enhance the effectiveness of DIS for neural network measurement operators (e.g., in semantic segmentation). The source code is provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors propose a new approach using CMs to generate realistic image samples in Diffusion Inverse Solvers, improving the application of complex, non-linear neural network operators like those in  semantic segmentation, room layout estimation, image captioning, and image classification. Unlike traditional methods that produce low probability images, the incorporation of CMs is expected to maintain sample realism, resulting in more accurate posterior approximations, particularly when neural network-based operators are involved.

### Strengths
The idea of using CMs in the process of inverse problems/posterior approximation seems novel and interesting.

The proposed method outperforms baseline approaches, particularly when compared to straightforward extension

### Weaknesses
Although being interesting, the novelty of the presented work is marginally above acceptation threshold since the only contribution seems to be the use of CMs in order to compute $x_0$ from $x_t$.

The manuscript could benefit from improved clarity and organization, as certain sections are challenging to follow. See further remarks.

Further remarks: 

In the presented algorithm, the updates are stated as $\zeta_t \Delta\left(f(x_0 \mid t), y\right)$, where $\Delta$ is defined as some distance. The update of $x_t$ however should be the gradient of that distance.

In order to enhance the readability of the work, the authors should think about introducing a clear distinction between the posterior $p(x_0|x_t)$ and the posterior $p(x|y)$, given y is the observation.

The abbreviation DPS (Diffusion posterior sampling) is used but never introduced.

### Questions
The presented manuscript often states likelihood terms as $p_\theta(y|x)$. Please elaborate on why it contains theta, as the likelihood in general is not a learned function with the same parameters as the learned data distribution $p_\theta(x)$.

Furthermore, I would be interested as to which extend the PF-ODE solution differs from the MAP solution?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper is about solving inverse problems using a pre-trained denoising diffusion prior. Posterior sampling with diffusion models requires an estimate of the score $\nabla _{x_t} \log p_t(x_t) + \nabla _{x_t} \log \int p(y | x_0) p _{0|t}(x_0 | x_t) d x_0$. While the first term is  estimated using the pre-trained score, the second term is usually very difficult to estimate accurately. A common approximation used in the literature involves using $\nabla _{x_t} \log p(y | E[X_0 | X_t = x_t])$ where  $E[X_0 | X_t = x_t]$ can also be estimated using the pre-trained score via Tweedie's formula. This approximation results in many efficiencies well documented in the literature and this paper tries to circumvent them using by using a sample from the PF-ODE as a replacement. Specifically, the authors use consistency models to speed up the process of sampling from the PF-ODE and ensuring that the differentiation is not costly.

### Strengths
The idea is quite original wrt to the literature. The paper is quite illustrated and clear. The experiments are interesting and extensive; the authors compare to many existing methods, on both pixel space and latent space diffusion.

### Weaknesses
- The most obvious weakness of the method is the use of consistency models since these are quite difficult to train and pre-trained CMs are not widely available. 

- In my opinion the justification for the method is rather weak. The paper argues that using a sample from the PF-ODE is valid because the sample has zero density and that furthermore, for the Gaussian mixture example considered, the PF-ODE sample has non-zero density under the posterior $p(x_0 | x_t)$ with high probability. At the same time it is also easy to find examples of a likelihood function $p(y|x_0)$ such $\int p(y|x_0) p(x_0 | x_t) d x_0 > 0$ for all $x_t$ but $p(y|\Phi(t, x_t)) = 0$ for $x_t$ in a set of positive Lebesgue measure. As a result, I'm not totally convinced that the argument is strong.

### Questions
The presentation of Proposition 3.3 is a bit misleading. The authors should state in the assumption the condition on $\sigma$ that they use in the proof in order to get a lowerbound independent of the dimension, or remove this claim after the proposition.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents an interesting approach for posterior sampling using $p(x_0|x_t)$ being approximated via consistency model. Results show improvement over baselines such as DPS

### Strengths
1. Easy to follow
2. The idea of using CM to approximate the PF-ODE solution is interesting

### Weaknesses
1. Using CM over the existing inverse problem solving adds a lot of computational burden, while the benefit is not clearly visible. Even though the $x_0|x_t$ may be off the manifold of ground truth image distribution, this does not imply a sacrifice in reconstruction quality as demonstrated in this work [1]. The quantitative results do not show a significant improvement over DPS.
2. CM itself can be used as a good prior for solving inverse problems: see this work [2]. This paper needs to compare with more recent works in inverse problem solving
3. More DIS baselines are desired such as DDNM [3]




[1]. Wang, Hengkang, et al. "DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models." arXiv preprint arXiv:2405.16749 (2024). NeurIPS 2024

[2]. Zhao, Jiankun, Bowen Song, and Liyue Shen. "CoSIGN: Few-Step Guidance of ConSIstency Model to Solve General INverse Problems." ECCV 2024

[3]. Wang, Yinhuai, Jiwen Yu, and Jian Zhang. "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model." The Eleventh International Conference on Learning Representations.

### Questions
1. More DIS baselines are desired such as CoSIGN [2], DDNM [3]

I am open to change my rating if authors could address my concerns (how clearly a CM model could benefits with inverse problem solving comparing to strong baselines like DDNM)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes using consistency models (CMs) to solve inverse problems with diffusion models. In diffusion inverse problems, common approaches need to go from xt to x0 at every iteration to be able to compute a measurement-guided gradient with respect to the measurement y from x0. The majority uses the expectation from Tweedie's formula to compute x0 from tx, which may not result in a good example for complex multi-modal distribution. The paper proposes to replace this step with a CM. They show improvement upon prior work.

### Strengths
The paper considers an important problem: how diffusion models can be used to solve complex inverse problems, such as giving a segmented image to reproduce the underlying image. The paper acknowledges a known limitation on one assumption that prior work makes on the distribution of the data (which allows to use the Tweedie's formula), and proposes to address it by consistency models.

### Weaknesses
I find the contribution of the contribution incremental, and the presentation can be improved. The main weaknesses are as follows: the mathematical formulation is confusing and the approach to training the CM is not fully fair compared to the baselines.

1. The paper trains a CM to be used for diffusion-based inverse solvers. It's not clear why such mathematical details (some are not fully proper) are included, which I do not think is the main contribution of the paper. It's unclear why in (1) the authors formulate the Markov chain following a VE formulation. Could the author explain this choice? The conventional Markov chain for diffusion models is the popular one based on VP, which has a different mean and variance such that the structure is destroyed, but the energy of the process remains the same.

2. Solving inverse problems with diffusion mostly occurs in the regime where the diffusion model is trained unconditionally without the knowledge of the measurement operator f(.) (see the DPS used for vision problems such as deblurring). However, the paper in Section 3.4 discusses that the CM is overfitting to f(.) and proposes an approach to make the framework robust. So this brings the following: the CM going from xt to x0 seems to be trained with the knowledge of the measurement. If f(.) is involved during the training, then the trained framework is not general anymore (it's problem specific). Hence, the comparison of the proposed framework to models such as DPS, where the model is not trained based on the measurement operator, is not fair. Please provide more information and clarity if this is not the case. The fair comparison would be a scenario where both methods are trained a similar condition (e.g., not having the knowledge of the forward operator).

Here are some questions concerning this:

- Is CM trained with knowledge of f(.), or if this is a misunderstanding?
- If the CM is trained with f(.), please explain and justify comparing it to methods like DPS that don't use this information?

More comments are below:


Lack of thorough literature

- The limitations of DIS are not fully explained in the intro. Indeed, the mean-based approximation is one challenge. A few others are related to whether methods such as DPS are doing posterior sampling or using the measurement to guide the process onto likely solutions (see [1]).


The paper needs improvement in presentation. Here are a few examples

- While the notations such as Xt and X0 are known to the reader with knowledge of diffusion models, these are used in the abstract and intro without introducing them. Hence, I suggest re-writing the abstract without these notations and introducing the diffusion in the introduction before using x0, xt, etc.

- Consistency models are not defined and introduced in the intro, but the authors explain that they are used to improve performance. I suggest the authors to provide a brief definition or explanation of consistency models in the introduction.

- How the results are generated for Table 1; this appears abruptly without proper explanation. I suggest to include a brief explanation of the methodology used to generate the results in Table 1.

Some terms within the manuscript are not precise and clear. Please provide clarifications.

- Section 1: with "neural network operators"? Does this refer to the measurement operator or the score function of the diffusion? I suggest to say "measurement operators" instead of "operators". Please clarify "neural network operators"?


[1] Wu, Z., Sun, Y., Chen, Y., Zhang, B., Yue, Y., & Bouman, K. L. (2024). Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors. arXiv preprint arXiv:2405.18782.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2
