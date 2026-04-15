# Enhancing Solutions for Complex PDEs: Introducing Translational Equivariant Attention in Fourier Neural Operators

- Decision: Reject
- Scores: 3, 5, 6, 6

## Abstract
Neural operators extend conventional neural networks by expanding their functional mapping capabilities across various function spaces, thereby promoting the solving of partial differential equations (PDEs). A particularly notable method within this framework is the Fourier Neural Operator (FNO), which draws inspiration from Green's function method to directly approximate operator kernels in the frequency domain. However, after empirical observation and theoretical validation, we demonstrate that the FNO predominantly approximates operator kernels within the low-frequency domain. This limitation results in a restricted capability to solve complex PDEs, particularly those characterized by rapidly changing coefficients and highly oscillatory solution spaces. To address this challenge, inspired by the attentive equivariant convolution, we propose a novel \textbf{T}ranslational \textbf{E}quivariant \textbf{F}ourier \textbf{N}eural \textbf{O}perator (\textbf{TE-FNO}) which utilizes equivariant attention to enhance the ability of FNO to capture high-frequency features. We perform experiments on forward and reverse problems of multiscale elliptic equations, Navier-Stokes equations, and other physical scenarios. The results demonstrate that the proposed approach achieves superior performance across these benchmarks, particularly for equations characterized by rapid coefficient variations.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces the Translational Equivariant Fourier Neural Operator (TE-FNO), an enhanced model for solving complex Partial Differential Equations (PDEs) by improving upon the standard Fourier Neural Operator (FNO). FNOs are known to perform well in learning operators for PDEs, but they tend to focus on low-frequency components, which limits their performance for problems with rapidly changing coefficients. TE-FNO addresses this by incorporating an equivariant attention mechanism that allows the model to capture both high- and low-frequency features. This enables more accurate predictions for challenging PDEs, such as multiscale elliptic equations and the Navier-Stokes equations.

### Strengths
1. The paper identifies a critical limitation in Fourier Neural Operators (FNO), specifically its bias towards low-frequency components. However this is not entirely new.

2. The paper demonstrates that TE-FNO achieves superior performance across various benchmarks, including forward and inverse problems in multiscale PDEs, with consistent improvements over existing state-of-the-art methods like FNO, U-NO, and HANO. The experimental results are comprehensive, showing the model's robustness in handling noise and its ability to generalize across different problem settings.

3. The paper provides a theoretical analysis for the use of equivariant attention mechanisms.

### Weaknesses
1. While the proposed method builds on existing work, such as FNO and attention-based mechanisms, the overall novelty may appear incremental. The core idea—enhancing FNO by capturing high-frequency features using attention mechanisms—shares similarities with other recent developments in operator learning. Similar ideas and also multilevel architectures are also presented in LSM (Wu et al., 2023), HANO (Liu et al., 2022). The performance is only marginal.

2. The equivariant attention is also not new. A more detailed comparison with related works like https://proceedings.mlr.press/v202/helwig23a/helwig23a.pdf and [Helwig et al., 2023] would help clarify the contributions of this work. 

3. The claim that "we propose a novel Translational Equivariant Fourier Neural Operator (TE-FNO) which utilizes equivariant attention to enhance the ability of FNO to capture high-frequency features" is claimed in the abstract but not explained. The motivation is not clear. Unfortunately, I think this paper is only a combination of several concepts from exsiting works.

### Questions
Please see the weaknesses.

### Soundness
2

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
5

### Summary
The paper proposes a novel method, Translational Equivariant Fourier Neural Operator (TE-FNO), designed to solve complex partial differential equations (PDEs) with high-frequency features. TE-FNO builds on the Fourier Neural Operator (FNO) but addresses its limitations in capturing high-frequency components by introducing an equivariant attention mechanism and convolutional-residual Fourier layers. This hierarchical structure allows TE-FNO to capture local and global features for challenging multiscale and inverse PDE problems, such as Navier-Stokes equations and elasticity equations.

### Strengths
- TE-FNO captures high-frequency features, which is a known limitation in standard FNO, making it suitable for complex PDEs with rapid coefficient variations.

- Equivariant Attention Mechanism for neural operators seems to be novel.

### Weaknesses
- Previous works have already studied ways to remedy FNO's limitations in capturing high-frequency information, e.g. [1]. The authors should clearly mention this paper and state the advantages of TE-FNO compared to this method. 

See questions for more.

[1] Neural Operators with Localized Integral and Differential Kernels, Miguel Liu-Schiaffini et al.

### Questions
- In Table 2, for FNO, did you run your own experiments for viscosity constants $1 \times 10^{-3}$ and $1 \times 10^{-4}$, or are the numbers taken directly from the FNO paper? The results are identical to those in the FNO paper. The FNO architecture was updated over two years ago, leading to improved performance.

- Were the experiments conducted only once, or were they averaged over multiple runs? The paper does not explicitly mention whether the experiments were averaged over multiple runs, nor does it provide standard deviations for the reported results.

- What is the motivation for using an **equivariant** attention mechanism? Why is equivariance desired? Which groups are of interest: the translation group, the Euclidean group, the orthogonal group, or more general Lie groups?

- By "we replaced the fully connected residual layers with a convolution layer (line 272)," do you use fixed-size local convolution kernels (e.g., Conv2D in PyTorch)?

- Related to the previous question on Equation 9, what does $\operatorname{Conv}(\boldsymbol{v}^k)$ mean?

- "The input and output of the convolutional-residual Fourier layer at the $k$-th scale are denoted as $\boldsymbol{v}^k$ and $\tilde{\boldsymbol{v}}^k$, respectively (line 273)." By this, do you mean the coefficients of the $k$-th Fourier mode?

- To my understanding, the hierarchical structure is similar to that in a U-Net, which can already capture multi-scale features [1, 2]. Why is equivariant attention needed to achieve this?

- In Sec. 3.5, you mention that the evaluation metrics are N-MSE, but why is MSE reported in Table 1?  

- In Section 4.5, the ablation study seems to suggest that equivariant attention itself does not improve performance, as stated: "It is discovered that all components of our model are essential for solving multiscale PDEs after removing experiments. (line 468)." If this is the case, what is the actual contribution of equivariant attention? Other components are borrowed from existing papers, and it appears that you have mainly combined them. Can you clarify the specific role and impact of equivariant attention in this work?

- What are the limitations of this work? I do not see any discussion in the paper.

Suggestions:

- Consider improving the presentation of this work for greater clarity and readability.

----
[1] Towards Multi-spatiotemporal-scale Generalized PDE Modeling; Jayesh K. Gupta, Johannes Brandstetter; 2022

[2] Convolutional Neural Operators for robust and accurate learning of PDEs; Bogdan Raonić, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, Emmanuel de Bézenac; 2023

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors look at an extension to the Fourier Neural Operator (FNO) by adding translationally equivariant attention. It is noted that FNO struggles with high frequency information, and the proposed attention mechanism improves FNO’s ability to learn in high frequencies. Experiments for both the forward and inverse problem are performed on a variety of systems with a set of recent neural operator architectures.

### Strengths
- The method is mathematically sound.
- The compared baselines are comprehensive and compelling.
- Ablations are thorough, showing translational attention's contribution to performance..

### Weaknesses
- The training details of the baselines are not clear. What measures were taken to provide a fair comparison? (E.g., parameter count, hyper parameters, training time)
- There are a few cases where the proposed method is suboptimal (Pipe, NS $10^{-4}$). No explanation or exploration of the limitations of translational equivariant attention are included.
- The related works should expand further on the differences to existing attention mechanisms (e.g., compare to [1] and [2]).

### Questions
- In the ablation, what is the *add hier*? Please expand on the details and how this is different from the hierarchical architecture used through the paper.
- How does performance compare when using the same architecture with a different attention mechanism? In other words, *rep Att -> self-attention, Fourier / Galerkin [1], or adaptive token mixing [2]*.

**Minor Typos:**
- Line 264: “proved to the” -> “proved to be an”


[1] Shuhao Cao. (2021). Choose a Transformer: Fourier or Galerkin.

[2] John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, & Bryan Catanzaro. (2022). Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers.

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
This paper proposes a new architecture, the Translational Equivariant Fourier Neural Operator, which enhances FNOs ability to capture high frequency features when solving partial differential equations. It does so by introducing an equivariant attention in combination with convolutional-residual layers, in an overall hierarchical structure. The authors then benchmark TE-FNO against state-of-the art models for a number of physical problems including the Darcy and Navier Stokes equations, as well as inverse problem solving on a multiscale elliptic PDE.  Finally, an ablation study justifies the relevance of each element of the architecture.

### Strengths
- The contribution proposes a novel architecture, usable for a wide variety of PDEs and based on sound theoretical grounds.
- Extensive benchmarking on various challenging problems shows that this architecture improves over state-of-the-art ML-based solvers
- An ablation study shows that all the components of the proposed architecture are relevant, and that removing or replacing them leads to a degradation of the results.

### Weaknesses
The paper does not offer an insight into the computational cost of this new method, in comparison with FNO.  While FNO and other neural operator based models were applied for large scale problems, this contribution doesn't allow to say whether TE-FNO would also scale well; in particular because of the attention mechanism that is used.

### Questions
The authors should discuss in more detail the computational and memory trade-offs, in particular the impact of adding the attention layer

### Soundness
3

### Presentation
4

### Contribution
4
