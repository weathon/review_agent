# Score-Based Neural Processes

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 3, 5

## Abstract
Neural processes (NP) are a flexible class of models that generate stochastic
processes by operating on finite-dimensional marginal distributions. NPs are
designed to maintain exchangeability and marginal consistency, which are necessary
to define a valid stochastic process. However, NP variants can come with drawbacks
such as limited expressivity, uncorrelated samples, and consistency sacrifices. To address the issues of previous NPs, we introduce score-based neural processes, \emph{scoreNP}, which incorporate a score-based generative model within the neural process paradigm. This score-based approach enhances expressivity, allowing the model to capture complex non-Gaussian distributions of functions, generate correlated samples, and maintain marginal consistency. Previously, no NP variant has been able to maintain conditional consistency. We show that using \emph{guidance} methods from conditional diffusion sampling, \emph{scoreNP} is the first NP is able to satisfy conditional consistency. Empirically, \emph{scoreNP} performs well qualitatively and quantitatively well across a range of unconditional and conditional functional generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The paper aims to develop more expressive neural process models by integrating score-based generative models, claiming enhanced handling of non-Gaussian stochastic process and the first achievement of conditional consistency within this framework. However, the submission is flawed with unsubstantiated claims, lack of rigorous theoretical support, and narrow experimental validation limited to 1D function regression tasks that do not conclusively outperform existing baselines.

### Strengths
Combining diffusion models with neural processes is a promising avenue in the study of neural processes.

### Weaknesses
* This paper contains many incorrect claims. Crucially, the development of the methodology and the associated proofs are fundamentally flawed. See detailed comments in the Questions section. 

* Almost all figures are blurry and barely readable. Even significant zooming does not aid in understanding the content.

* The experiment design is far from satisfactory.  Only 1D function regression is used in this paper. Even for the 1D regression result, the proposed method doesn’t outperform existing baselines

* The presentation of the paper is highly problematic. The structure is disorganized and the language is unclear. For example, in the introduction part, the narrative chaotically oscillates between discussions on neural processes and diffusion models without a clear connection or transition. Additionally, the placement of the related work section after the experimental results is unconventional and disrupts the logical flow of the paper

### Questions
* The claim in line 18 that the proposed method is the first NP variant to maintain conditional consistency is incorrect. The first NP to achieve this was the Markov Neural Process (MNPs, Xu et al., 2023). The authors need to clarify how their approach to conditional consistency differs from or improves upon that of MNPs both theoretically and experimentally. 

* Line 21, the paper fails to demonstrate that scoreNP outperforms existing neural process models based on the experimental results provided. 2D regression problems should be incorporated in the main text. Additionally, the emphasis should be on regression performance rather than generative capabilities. 

* The organization of the presentation is disordered. The introduction starts with neural processes, shifts to diffusion models, discusses neural diffusion processes, and then returns to diffusion models.  Moreover, the relevance of Figure 1 to the function regression discussion is unclear.  The placement of the related work section after the experimental results is unconventional and disrupts the logical flow of the paper

* The claim in line 58 that operations directly on function space compromise consistency due to discretization is incorrect. Recent advances have shown that direct regression on function space can yield superior performance over existing NP baselines (Maronis et al., 2023; Shi et al., 2024).


* Lines 120-124 acknowledge that insights from MNPs inspired the development of an infinite chain of NPs via a denoising model. However, the assertion on line 229, which maps the point evaluation of an unknown stochastic process to $p^1(y_{1:n}|z, x_{1:n})$, namely $\mathcal{N}(0,I)$. This is fundamentally problematic because white noise doesn’t exist in L2 space.  You cannot define a stochastic process in L2 space such that joint distribution of a collection of points from this stochastic process is $\mathcal{N}(0,I)$. This issue has been extensively discussed in the literature on functional diffusion models (Lim et al., 2023; Kerrigan, 2023), where the Gaussian Process with trace-class operator is used instead of white noise.  Moreover, the proof of Theorem 1 is incorrect; the author incorrectly extends a finite-dimensional diffusion model defined on $\mathbb{R}^d $ to function space (specifically L2 space) by naively taking $d$ to infinity. This approach is flawed because density functions used in the finite-dimensional diffusion model do not exist in function space where the unknown stochastic process defined. A rigorous proof should start from the standpoint of measure theory. 

* Line 591-591 “each step can be considered a functional markov transition operator of MNPs” is wrong.  In MNPs, the model uses normalizing flows, viewed as pointwise operators, applicable to stochastic processes observed at any position. Conversely, finite-dimensional diffusion model is restricted to $\mathbb{R}^d $ and cannot act as a valid operator in function space. To demonstrate that the proposed method is a valid operator, the author needs to provide related proof as well as zero-shot super-resolution experiments similar to those in functional diffusion models. (Lim et al., 2023; Kerrigan, 2023). 

* The regression experiment is inadequate, with only 1D function regression tested against other baselines. Even within the 1D function results, there is no convincing evidence that the proposed method outperforms existing baselines, less alone the additional computation overhead required. More experiments are clearly needed, see the experiments in Dutordoir et al, 2023 for reference

## Reference
Jin Xu, Emilien Dupont, Kaspar Martens, Tom Rainforth, and Yee Whye Teh. Deep Stochastic Processes ¨ via Functional Markov Transition Operators, May 2023

Juan Maroñas, Oliver Hamelijnck, Jeremias Knoblauch, and Theodoros Damoulas. Transforming Gaussian Processes With Normalizing Flows. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, pp. 1081–1089. PMLR, March 2021. URL https://proceedings.mlr.press/ v130/maronas21a.html. ISSN: 2640-3498

Yaozhong Shi, Angela F Gao, Zachary E Ross, and Kamyar Azizzadenesheli. Universal functional regression with neural operator flows. arXiv preprint arXiv:2404.02986, 2024.

Jae Hyun Lim, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, and Anima Anandkumar. Score-based Diffusion Models in Function Space, November 2023. URL http://arxiv.org/abs/2302.07400. arXiv:2302.07400 [cs, math, stat].

Gavin Kerrigan, Justin Ley, and Padhraic Smyth. Diffusion Generative Models in Infinite Dimensions, February 2023. URL http://arxiv.org/abs/2212.00886. arXiv:2212.00886 [cs, stat].

Dutordoir, Vincent, Alan Saul, Zoubin Ghahramani, and Fergus Simpson. "Neural diffusion processes." In International Conference on Machine Learning, pp. 8990-9012. PMLR, 2023.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes Score-Based Neural Processes (scoreNP), integrating score-based generative models with Neural Processes (NPs) through joint diffusion in latent-data space. The authors claim two main contributions: (1) enhanced expressivity while maintaining marginal consistency through score-based modeling, and (2) achieving conditional consistency using guidance methods from conditional diffusion sampling. The method is evaluated on several 1D function regression tasks and image datasets.

### Strengths
* The combination of score-based models with NPs is an interesting direction. However, this combination has been explored in several works [1, 2, 3], the authors should discuss the advantages and differences of this work over previous ones.
* The joint latent-data space diffusion provides a theoretically motivated approach for handling both unconditional and conditional generation.
* The proposed solution for conditional consistency through guidance seems novel.


[1] Dou, H., Lu, J., Yao, W., qian Chen, X., & Deng, Y. Score-based Neural Processes.

[2] Dutordoir, V., Saul, A., Ghahramani, Z., & Simpson, F. (2023, July). Neural diffusion processes. In International Conference on Machine Learning (pp. 8990-9012). PMLR.

[3] Mathieu, E., Dutordoir, V., Hutchinson, M., De Bortoli, V., Teh, Y. W., & Turner, R. (2024). Geometric neural diffusion processes. Advances in Neural Information Processing Systems, 36.

### Weaknesses
The paper is still in very incomplete shape and requires substantial revision, I will list a few points:
* The paper is overall not well-written, with important details scattered across different sections, the language seems informal in places and the presentation is rushed and incomplete. Many technical terms are used without proper definitions, e.g., "Predictive NPs" and "Generative NPs", "Encoder collapse", "NFEs", "NELBO". Several equations contain unexplained variables and notations.
* The paper's theoretical development lacks rigor and clarity, particularly in explaining how score-based modeling integrates with Neural Processes. And Theorem 1's proof is incomplete.
* Critical implementation details are missing or insufficiently explained, e.g., appendix E.3 is empty; network architectures are vaguely described; Fourier feature encoding specifications are missing.
* The conditional sampling procedure's stability issues are not properly discussed.
* The experimental section is inadequate, which only runs some experiments on 1D synthetic functions, Appendix D.10 which should show results on 2D functions is empty. And there is a very limited comparison with state-of-the-art NPs. The failure cases on MNIST are concerning and not adequately analyzed.
* The font size in some figures is way too small. Figure 3 is too blurry, you should use vector graphics.
* The Related Work section is inadequate and misses several important connections and recent developments in NPs.
* The computational requirements section is vague.

I encourage the authors to enhance overall clarity and organization, add missing implementation details, and improve experimental validation.

### Questions
* What does encoder collapse mean?
* Can you give a more general introduction about "conditional consistency"?
* What causes the poor performance on periodic functions?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors introduce score-based neural processes (scoreNPs), a novel member in the neural process family that utilizes score-based generative modelling for enhanced expressivity. In comparison to other NPs, scoreNPs retain exchangeability and marginal consistency making them a valid stochastic process, and do not sacrifice consistency upon conditioning. The authors evaluate their method on several 1D regression tasks with mixed results.

### Strengths
- The paper proposes a promising method for the neural process family which definitely should be relevant to the community.
- The theoretical results are intriguing.

### Weaknesses
## Major
- The experimental section in the main text is extremely thin in comparison to the rest of the manuscript (some parts of the main text could in my opinion easily be moved to the appendix without hurting exposition in lieu of a more exhaustive experimental section).
- The method is in my opinion empirically not very convincing, e.g., it does not outperform the baselines on simple 1d regression tasks.
- The method seemingly fails to work on more complex tasks, such as unconditional generation of geofluvial data or conditional generation of MNIST.

## Minor
- The paper contains extremely many typos and grammatical errors, and sometimes misses entire words.
- The quality of the figures is low (pixelated, unreadable axes, ...).
- Some sections in the appendix seem to be empty, e.g. E.3.

### Questions
- The authors state: "The variational inference makes the learnability a difficult task with a couple of difficult
balancing hyperparameters."  Could they please elaborate to which parameters they are referring to and why they are difficult to tune?
- What is the advantage of this method is in comparison to, e.g., Markov NPs (leaving consistency aside)?
- The authors state (Appendix E.4.): "[G]enerating a batch of 128 samples took a couple of minutes." Is the method generally computationally as demanding?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents Score-Based Neural Processes (scoreNP), a framework that enhances the expressivity and consistency of neural processes by integrating score-based generative models. scoreNP can model complex non-Gaussian distributions and generate correlated samples while maintaining both KET conditions.

### Strengths
The integration of score-based generative models within the NP framework is a novel approach. The theoretical foundation is robust- the proofs and methodology enhance trust in the results. The VLB objective is also a novel extension of likelihood re-weighting.

### Weaknesses
The paper only shows competitive performance for 1-D regression (that too only in certain cases) which raises questions about the scalability of the approach. Appendix D.10 (qualitative results sections) is empty. Although MNIST results for unconditional are competitive, the samples when conditioned on the context points are not good.

The authors acknowledge that tuning the architecture and hyperparameters gets very difficult even from 1-D to 2-D. This raises serious concerns about the scalability of this work.

### Questions
- Could the authors elaborate on how scoreNP would perform in higher-dimensional functional spaces? What modifications, if any, would be necessary to scale the approach?
- How sensitive is the model performance to the choice of hyperparameters?

### Soundness
2

### Presentation
2

### Contribution
2
