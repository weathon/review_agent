# Data-driven estimation of gradient fields in the weak sense

- Decision: Reject
- Scores: 2, 4, 8, 6

## Abstract
Estimating the gradient of a function is crucial in various fields, including data assimilation, reinforcement learning, and generative modeling. These issues are particularly challenging because of the high-dimensional spaces involved. Furthermore, direct gradient computation is often infeasible due to the intrisick lack of the regularity of the underlying function or simply because the function's explicit form is unavailable. In this work, we introduce a novel framework for the offline estimation of gradient fields inspired by score matching. The core idea is to employ an integration-by-parts formulation to replace the direct mean squared error loss with a term involving the divergence of a neural network, which can then be efficiently approximated using Hutchinson’s estimator. Numerical experiments include offline gradient estimation where we evaluate the fidelity of estimated gradient for different methods and an application to implicit regression problems involving the two-dimensional Navier–Stokes equations. The results demonstrate the practical potential of the approach for various tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a method for estimating gradient fields of functions whose gradient functional cannot be directly accessed or easily computed. The key insight is that a trick similar to score matching enables rewriting the loss to learn such functionals without requiring access to the target gradient. The authors empirically compare their method to existing benchmarks, demonstrating advantages in most cases.

### Strengths
The paper is well motivated. The main technique relies on an elegant connection to score matching, and its application to this setting is compelling. The proposed method is sensible and shows promise for tackling the problems identified by the authors.

### Weaknesses
There are several important areas where the authors could strengthen the paper. I am confident these can be addressed, but they require substantial changes that prevent acceptance at this venue.

- **Insufficient empirical verification.** The paper would benefit from additional ablation studies to understand how different parameters affect performance—for example, the number of Hutchinson vectors, the control variate $c$, and the choice of architecture for parametrizing $v_\theta$. Additionally, there is a notable gap between the practical problems where such a gradient field estimator would be valuable and the current shallow application of the proposed WGM method. Including one or two experiments in practically relevant settings would substantially strengthen the paper.
- **Structural issues.** While generally well written, the paper sometimes dwells on technical details and lacks clear motivation. For instance, Section 2.1 could be moved to the appendix, as most of its content is not essential to the main text. Section 2.3 needs better integration with the surrounding material. Section 4, in its current form, reads as a collection of equations that are difficult to parse without prior familiarity. The paper would also benefit from intuition about the control variate choice. Finally, Sections 4 and 5 lack proper motivation and clear connections to the rest of the paper.

I want to emphasize that I believe this will be a strong paper once these issues are addressed. I encourage the authors to incorporate this feedback and resubmit their work to a future conference.

### Questions
-

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces WGM that estimates the gradient of a scalar field without requiring explicit differentiability. The method is inspired from score matching and extend to gradient of general scalar fields, with requirement that gradient of log likehood being known. Experiment results show superior performance of the proposed method compared to prior arts including finite difference or autodiff on scalar field surrogate models. Detailed derivation of the WGM loss is also provided.

### Strengths
- (Originality) This work proposes WGM as an indirect method for accurate estimation of gradient fields. The application of divergence theorem on top of score matching is novel and not studied in other works to the best of the reviewer's knowledge.

### Weaknesses
- (Significance and quality) The proposed WGM loss is tested only on two problems: Learning the gradient of MSE reconstruction error on MNIST dataset, and searching for the initial conditions of 2D NS equations with a non-differentiable physics solver. For the first problem, the practical meaning of constructing the GMM and estimating the gradient field of MSE reconstruction error is unclear. Basically, the proposed method seems to fit only for a very narrow class of problems and lacks general applicability (significance). It is very uncommon to have a need to predict the gradient field rather than the scalar field itself, while not having access to the gradient ground truth, is very uncommon.
- (Clarity) The paper is rather hard to follow as the structure doesn't follow the typical flow of deep learning research paper: Intro, Background, Method, Results, etc. The results section discusses a lot of method content, with actual experimental results shown in both section 3 and 5. Also, more experimental results should be included to support the claim.

### Questions
- The experiments discussed in this paper are more of domain specific problems with narrow significance. Even for pure comparison purpose, two case studies is not sufficient to prove the effectiveness of the proposed method. Is it possible to include at least one additional case, ideally with higher significance for comparison?
- There were papers using weak formulation in loss function such as ***wPINNs: Weak physics informed neural networks for approximating entropy solutions of hyperbolic conservation laws***, ***Weak-formulated physics-informed modeling and optimization for heterogeneous digital materials***, and ***The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems***. Can you elaborate more on the fundamental difference between your approach and theirs?
- The framework requires prior knowledge of score values. This can be a major limitation in many problems. How can this be resolved generally? Also it's unclear how this is resolved in the 2D NS case study.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposed a novel framework for estimation of gradient fields. The method is inspired by score matching and the goal is to learn $$\min_{\theta}\mathbb{E}_p \Vert  v(x) - \nabla J(x)  \Vert ^ 2$$

By integrate-by-part technique, the above problem is converted into learning the gradient fields in weak sense. The authors called the method, weak gradient matching (WGM). One technical detail is to use Hutchinson estimation of the divergence to estimate $\nabla \cdot v_{\theta}(x)$, a term emerging from integrate-by-part (see equation (5)).

Experiments include Gibbs energies, random CNNs, MNIST, Navier-Stokes. Results show that the proposed method has certain potential in various scenarios.

### Strengths
**Originality** The work is well-established from a novel perspective in learning gradient fields. The motivation is well presented. The method is established in a skillful way, e.g., the application of Hutchinson Estimation of divergence.

**Quality** The paper is presented in a way that methods are well referenced and related, e.g., Poincaré inequalities, Sobolev training and the works from score matching and continuous normalization of flows.

**Clarity**  The flow of the paper is smooth. From motivation to implementation, a novel framework is established.

### Weaknesses
1. My main concern is regarding experiments. It seems that the advantage of WGM is not apparent. See Table 2 and 3.
2. The inference procedure (Algorithm 1) may not be optimal. Instead one may seek $y$ such that $v_{\theta}(y)=\mathbf{0}$.

Typo:
1. Fig. 1, Left, right -> Top, bottom
2. Fig. 7, row 3,4,5 -> column 3,4,5?

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed an algorithm to estimate the gradient flow of a function which is not directly tractable (computational cost, blackbox etc).
It is based on the Green's identity and Hutchinson Estimation, yielding a weak gradient estimate that needs an explicit knowledge of the log probability. This assumption can be effectively weakened empirically by being replaced with a mixture of Gaussians.
The paper then carefully review the statistical estimates and convergence of the method.

### Strengths
Clear, sound and principled mathematical presentation of the subject.
Each hypothesis is carefully assessed and studied.

### Weaknesses
The overall structure is somewhat difficult to follow (different problems are presented one after the other, without it being clear exactly how they are related to each other).
The same applies to the presentation of the algorithm: it is somewhat difficult to follow what relates to the presentation of the algorithm, its adaptation to a particular case, etc.
The choice of experiments also seems somewhat arbitrary (at least it is not very clear whether these are classic problems or examples chosen by the author. In any case, a more in-depth discussion would not hurt). 
Finally, I think there should be a more in-depth discussion with PINNs on the one hand, but above all with CMA-ES type methods, the purpose of which is precisely to estimate a gradient statistically.

### Questions
- In figure 2 top and bottom seems inverted (at least in caption).
- In section B.2 p17 (but this applies more generally) : use \big,\Big,\bigg,\Bigg for parenthesis, otherwise it is hard to identify each block between parenthesis.
- In B.3.2 p. 22 after polynomial regression, you have a reference missing.
- In algorithm 2 p 27 you have a typo : v or \upsilon is missing after Integrate.

### Soundness
3

### Presentation
2

### Contribution
3
