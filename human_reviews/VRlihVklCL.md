# MaD-Scientist: AI-based Scientist solving Convection-Diffusion-Reaction Equations Using Massive PINN-Based Prior Data

- Decision: Reject
- Scores: 5, 5, 8, 6

## Abstract
Large language models (LLMs), like ChatGPT, have shown that even trained with noisy prior data,  they can generalize effectively to new tasks through in-context learning (ICL) and pre-training techniques.
Motivated by this, we explore whether a similar approach can be applied to scientific foundation models (SFMs). Our methodology is structured as follows: (i) we collect low-cost physics-informed neural network (PINN)-based approximated prior data in the form of solutions to partial differential equations (PDEs) constructed through an arbitrary linear combination of mathematical dictionaries; (ii) we utilize Transformer architectures with self and cross-attention mechanisms to predict PDE solutions without knowledge of the governing equations in a zero-shot setting; (iii) we provide experimental evidence on the one-dimensional convection-diffusion-reaction equation, which demonstrate that pre-training remains robust even with approximated prior data, with only marginal impacts on test accuracy. Notably, this finding opens the path to pre-training SFMs with realistic, low-cost data instead of (or in conjunction with) numerical high-cost data. These results support the conjecture that SFMs can improve in a manner similar to LLMs, where fully cleaning the vast set of sentences crawled from the Internet is nearly impossible.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper introduces an in-context learning architecture for scientific foundation models. The proposed method incorporates Bayesian inference into the prediction by pre-training on PINN-based low-cost and noisy approximated data, which is referred to as the PINN-prior. The PINN is initialized as a general PDE with derivatives up to a certain degree. Initializing the PINN with a general PDE helps to overcome the fact that the underlying governing equations of physical systems are often unknown; instead, only the observations of the underlying systems are available. At the base, the method uses a transformer architecture, where a self-attention mechanism is used to capture relationships among known data points, and a cross-attention mechanism is used to extrapolate and infer solutions for unseen points without fine-tuning.

### Strengths
1. This paper integrates several non-trivial concepts of the scientific foundation model, approximated prior data, Bayesian inference, and pre-training that are already well-established in scientific machine learning. 
2. The concept of using PDE dictionaries in the PINN-prior for pre-training is original. 
3. This paper tries to address the challenges of the inherent noisiness of the data used for training.

### Weaknesses
According to me, some parts of the paper need further clarification. Please see the comments below:
1. The underlying concept of using approximated prior data is a concept that has been introduced previously in scientific machine learning. This concept already exists in multi-fidelity learning, where a low-fidelity solver is first trained on a large number of available low-fidelity data, and the approximate predictions of low-fidelity solver are used in conjunction with the small high-fidelity data to train a fast high-fidelity emulator [1,2]. Except for the difference between the purely data-driven nature of the multi-fidelity learning methods and the partial physics-constrained nature of the proposed framework, I do not see any significant improvement.  Further clarification is required to understand the novelty of the proposed method.
2. The authors mention the "scientific foundation model" at several places in the paper. The purpose of the foundation models is to pre-train on a large number of datasets, which in this case is analogous to multiple PDEs, and then efficiently fine-tune or directly predict on unseen PDEs without sacrificing prediction accuracy. However, the authors consider only a simple CDR equation for pre-training and special cases of the CDR equation like convection, diffusion, and reaction equations for testing. This raises concerns about whether the methods can genuinely aid in foundation modeling, highlighting the need for more rigorous case studies to properly assess the effectiveness of the proposed PINN-prior framework. To be able to clearly appreciate the method's potential as a foundation model, in addition to the CDR equation, it is necessary to simultaneously pre-train on a few of more complex PDEs like the Kuramoto–Sivashinsky equation, Schrodinger equation, Korteweg–De Vries equation, Navier-Stokes equation, Burgers' equation, etc. and test on remaining PDEs.
3. In the PINN dictionary, the authors consider the exact derivative terms of the CDR equation. This is unlikely to be available in a practical scenario, which is also mentioned by the authors in the motivation. Thus, the performance of the PRIOR in the presence of a large number of dictionary functions needs to be inspected to understand further the potential of the proposed framework for scientific foundation modeling. The authors need to conduct additional experiments with a larger and more diverse set of dictionary functions to better simulate real-world scenarios where exact terms may not be known.
4. Many scientific foundation models have already been developed for pre-training and testing on multiple physical systems [3-6]. However, the authors do not mention them in this paper. Given the existing literature on foundation models, the authors should consider comparisons with a few of the existing scientific foundation models. This will further improve the contents of the paper.

[1] Meng, Xuhui, and George Em Karniadakis. "A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems." Journal of Computational Physics 401 (2020): 109020.\
[2] Howard, Amanda A., et al. "Multifidelity deep operator networks for data-driven and physics-informed problems." Journal of Computational Physics 493 (2023): 112462.\
[3] McCabe, Michael, et al. "Multiple physics pretraining for physical surrogate models." arXiv preprint arXiv:2310.02994 (2023).\
[4] Tripura, Tapas, and Souvik Chakraborty. "A foundational neural operator that continuously learns without forgetting." arXiv preprint arXiv:2310.18885 (2023).\
[5] Hao, Zhongkai, et al. "Dpot: Auto-regressive denoising operator transformer for large-scale pde pre-training." arXiv preprint arXiv:2403.03542 (2024).\
[6] Herde, Maximilian, et al. "Poseidon: Efficient Foundation Models for PDEs." arXiv preprint arXiv:2405.19101 (2024).

### Questions
Please see below questions on the paper content:
1. line 078. The authors claim that their goal is "to predict solutions from observed quantities, such as velocity and pressure, without relying on governing equations". However, any data-driven ML model learns solutions from observed quantities without relying on governing physics. Therefore, why use a PINN with approximate PDE dictionaries while robust methods like neural operators are already there? 
2. line 269. In Eq. (12), define $\tilde{u}(x_T^{(j)},t_T^{(j)})$. Is it the same as $\tilde{u}(\alpha)$, or does it denote the observed solution field?
3. line 271. "...our model only needs observed quantities". The correct derivatives, as in the original CDR equation, are considered in Eq. (8). Thus, this sentence seems to be vague to me. Consider a large number of derivatives and their polynomials in Eq. (8) and then generate prior data to see the performance of the proposed method. 
4. line 302. it is written $D$ requires the solution u. Is $u$ the true observed quantity or the PINN-approximated data?
5. line 323. What value of $m$ is taken in the examples?
6. line 372. Is it supposed to be section 4.2.1?
7. Table 3. For 100% PINN-prior, the error in the convection equation is as high as 16%. This raises questions about the framework's ability to generalize effectively over a large number of unseen systems. 
8. Is there any reason similar to the above why the "TIME DOMAIN EXTRAPOLATION" study is not carried out on diffusion and reaction equations?
9. Fig. 3. This is a continuation of comment 7. See that the time extrapolation results for the convection equation is more than 50%, which again raises questions about the framework's generalization ability. 
10. Fig. 4. The solution field on the right is too smooth to draw a conclusion about the extrapolation ability.
11. line 480. Why is the $\beta$ = $\nu$ = 0 taken? I think a non-zero value will make the problem more challenging.
12. line 483. It is not fair to say "it can generalize to unseen PDEs" since the model is pre-trained on the CDR PDE, which is nothing but a mixture of convection, diffusion, and reaction.
13. line 487. How many training samples are taken? 
14. Table 4. A central goal of the paper is to achieve zero-shot inference for predicting PDE solutions. However, when the model is pre-trained on a mixture of reactions but tested on different reactions separately, the error increases significantly beyond 6% in most of the cases. This indicates that the proposed framework may also need fine-tuning or few-shot learning.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a scientific foundation model to solve convection-diffusion-reaction equations in a zero-shot setting. Its main novelty is that the PDE solution data used to train the model is itself generated by a PINN solver, and is therefore subject to noise and measurement error. The authors demonstrate the the model can outperform baselines and make accurate predictions under parameter distribution shifts and time-domain extrapolation.

### Strengths
1. This work validates the hypothesis that foundation models are able to generalize even under noisy prior data, which has been thoroughly demonstrated in NLP but has been largely unexplored in scientific ML.

2. The model performs well even under substantial distribution shifts, which suggests that it is really learning to learn as opposed to memorizing the distribution of the training data.

### Weaknesses
1. The authors provide very few details on their architecture, other than that they use a transformer with self-attention and cross-attention. This makes it very difficult to meaningfully compare their model with other baselines. I would suggest adding a detailed description of the architecture to the appendix.

2. The class of PDEs considered is quite restrictive, since it is a family of 1D equations parameterized by relatively few variables. It doesn't seem too surprising that a sufficiently complex foundation model can learn this family of equations.

Typos: In Theorem 2.1, I think the first inequality should say that $\mathbb{E}[KL(p_{\widehat{\theta}}(\cdot | x, D_n) \parallel \pi(\cdot | x, D_n))] < \epsilon$ rather than $\widehat{\theta} < \epsilon$.

### Questions
1. Can you clarify specifically what you mean by 'zero-shot inference'? In Figure 2 (Inference Phase) it looks like you prompt your model with pairs $(x_i,t_i,u_i)$, and thus require data from the operator to be inferred in order to make predictions.

2. Can you explain your rationale for using the L^2 norm as an evaluation metric, as opposed to the H^1 norm or a more PDE-specific metric?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a scientific foundation model to solve PDEs with scarce data by training on low-cost data. This methodology is proven to be amenable to zero-shot learning with high performance, as evidenced by benchmark experiments across various kinds of PDEs.

### Strengths
- The paper is well organized and written.
- Encouraging results across datasets.
- To have a genearlizable zero-shot PDE solver trained from low-cost data would have high impact on the physical modeling community.

### Weaknesses
- Figure 2. used a lot of page real estate to demonstrate concepts not unique to the paper. I would perhaps highly the innovation of this paper here.
- I feel like to brand this as a _scientific_ foundation model is slightly overselling it. To solve PDEs is hardly equal to _science_.

### Questions
- I feel like the idea behind Theorem 2.1 is rather intuitive. And it is a little out-of-place here. I don't see it referenced at all in the text. Could you motivate this theorem a bit more?

### Soundness
3

### Presentation
3

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
This paper introduced MaD-Scientist, a novel approach to solving partial differential equations (PDEs) using Transformer-based models trained on Physics-Informed Neural Network (PINN) generated prior data. The key innovation is demonstrating that scientific foundation models (SFMs) can learn effectively from approximate, low-cost PINN-generated data, similar to how large language models (LLMs) learn from noisy internet data. The authors focus on convection-diffusion-reaction (CDR) equations and show impressive zero-shot inference capabilities.

### Strengths
- The parallel drawn between LLMs learning from noisy internet data and SFMs learning from approximate PINN-generated data is insightful and well-motivated. Similarly the zero-shot inference approach without requiring knowledge of governing equations is equally important. 
- Clear mathematical formulation of the problem and the methodology. 
- Extensive experiments covering multiple scenarios. The baselines are sound, and the results are excellent. Especially notable is the "superconvergence" phenomenon observed (Remark 4.1)

### Weaknesses
- The scope of the paper is very limited. Focus solely on 1D CDR equations may limit the broader applicability. Furthermore, no real-world applications are thoroughly explored (albeit a brief mention of semi-conductors). 
- Limited discussion of computational efficiency and scaling properties, with not much variations of the transformer architecture explored. 
- The appendices contain important information that should be in the main text like efficiency comparison (above point), training algorithm and some PINN failure modes. 

Based on the questions/suggestions below, I am willing to increase my score.

### Questions
- Can the authors investigate the method's performance on higher-dimensional PDEs?
- How does the computational cost scale with problem complexity?
- Will including physical constraints in the Transformer architecture improve performance?
- A discussion of potential real-world applications would strengthen the paper

### Soundness
3

### Presentation
3

### Contribution
2
