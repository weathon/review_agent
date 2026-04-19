# Efficient PDE Solutions using Hartley Neural Operators in Physics-Informed Networks: Potentials and Limitations

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
In this work, we introduce novel differentiable architectures for solving partial differential equations (PDEs) using the well-known Discrete Hartley Transform. We incorporate Hartley Neural Operators (HNO) into Physics-Informed Neural Operator Networks (PINOs). Our analysis concentrates on two pivotal PDEs: 1. the one-dimensional diffusion equation, which holds significance not only in machine learning but also across a broad spectrum of physical sciences and engineering disciplines; and 2. the one-dimensional Thermodynamic Energy Equation that is commonly used in weather data analysis. Our implementation of HNO that employs real-valued linear transforms into the PINO architecture results in significant run-time improvements. We show that reconstruction loss is lower than other recently introduced operators that may be used for the above PDEs. Importantly, we find that HNOs naturally satisfy the governing physical laws and equations specific to the PDEs under consideration. However, our empirical observations suggest that the benefits of HNO diminish in certain scenarios where the underlying physical conditions at the boundary are less tractable and involve complex numbers. As an example of a potential failure mode we illustrate that in the case of the one-dimensional Burger's equation, traditional Fourier Neural Operators outperform their Hartley counterparts. Our results indicate that a combination of neural operators including Fourier and Hartley transforms may be better to effectively address the specific type, and/or context of the physical problem at hand.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work considers the problem of solving partial differential equations with a physics-informed neural networks (PINNs) approach. The authors propose a novel neural operator architecture, based on the Discrete Hartley Transform, as an alternative to the popular Fourier neural operator, which is based on the Fast Fourier Transform. The main advantage of the Hartley transform is that it allows for a fast convolution (similar to the FFT) but enforces the resulting function to be real as opposed to complex-valued. In a second part of the paper, the authors evaluate the architecture on two partial differential equations: Burger's equation and a diffusion equation.

### Strengths
- The approach of the authors that incorporate the Hartley transform into neural operators appear to be novel.
- The Hartley transform has same computational complexity than the FFT, which allows for a fast evaluation of the neural operator.

### Weaknesses
- One of the main weaknesses of the paper is the motivation for the Hartley transform. The authors mention that the main limitation of the FNO is that it ``provides suboptimal solutions'' because the resulting functions can be complex-valued. However, I do not see any evidence supporting that this could be a limitation in practice as one could simply take the real part of the network since the output will likely have a small imaginary part if the loss function is small after training on real data.
- The authors only consider problems where the PDE is known so there is little motivation for using a PINO approach over PINN (or even simply a traditional numerical method like finite-difference or finite elements). The experiments must include a comparison against a PINN technique.
- The numerical experiments in the paper do not demonstrate that the proposed architecture improves the performance of the Fourier Neural Operator, as acknowledged by the authors for Burger's equation and observed in Fig. 1.

### Questions
- In the 2nd paragraph, what do the authors mean by suboptimal solutions?
- The last paragraph on p.3 has citations not rendered.
- Eqs. (5-7) should include the domain on which the PDE is defined, as well as a proper definition of the variables, and initial/boundary conditions.
- The loss functions in Eq. (8) should be mathematically defined.
- The discussion regarding the activation function in Section 3 requires references for claiming that GELU is preferred in physics-informed deep learning.
- In the last paragraph of section 3, it is inexact to state that the numerical scheme provides exact solution. What is the advantage of using the present approach over the numerical techniques, which are much faster and have convergence guarantees?
- In Figure 1, 4th column, why is the absolute error negative?
- Table 2: these results should be repeated over several training runs and contain deviation errors. Is it statistically significant to mention the ``reduced reconstruction loss" in the first paragraph of p. 7 if the difference only appears on the 4th digit?
- In the 2nd paragraph of the discussion, the author mention that their architecture achieve a ``more random trajectory'', this is a vague term which should be make precise.
- In the last sentence of the discussion: the authors mention that the activation function made no difference. I am wondering why it is discussed in the paper, which is about using the Hartley transform for neural operators.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, an application of the Hartley transform to Physics-Informed Neural Operator (PINO) is discussed. The Fourier Neural Operator is one of the typical neural operators. This method is derived by replacing the Fourier transform used in this neural operator with the Hartley transform. The proposed method is applied to some specific problems and the performances of the methods are discussed.

### Strengths
When a Fourier transform is performed, complex numbers usually appear in the computation process. In contrast, the Haretley transform has the advantage that it can be computed using real numbers only, so the computation is expected to be accelerated.

### Weaknesses
The authors do not give any particular theoretical analysis for their proposed method. Therefore, the main contribution of this paper is considered to be numerical experiments. The authors do not give any particular theoretical analysis for their proposed method. Therefore, the main contribution of this paper is considered to be numerical experiments. However, the problem considered is limited to one-dimensional problems. Practically important multidimensional problems are not considered. I do not believe that this paper has made a sufficient contribution for publication.

### Questions
As mentioned in Future Work by the authors, I believe that application to multi-dimensional problems and the introduction of sophisticated architecture such as the AHNO attention mechanism are important and needed for publication. I encourage the authors to resubmit this paper after these improvements.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a novel neural operator structure using the Hartley transform and suggests combining this neural operator structure with Physics-Informed Neural Networks (PINNs) to better incorporate the PDE loss. Overall, the idea is worth publishing; however, the paper's writing and formatting require substantial improvement. The comparisons in the experiments are also quite inadequate, hence I recommend the authors to revise the paper meticulously before submitting it to the next venue.

### Strengths
The integration of the Hartley transform with neural operators and PINNs is a novel idea proposed in this paper.

### Weaknesses
1. There is significant room for improvement in the paper's writing and formatting. Many reference links in the paper are represented as "?" symbols, tables in the paper are not centered, the best results are not highlighted, and there's an entire page (page 6) with a large figure without any accompanying text. These issues give the impression that the paper was hastily prepared to meet a deadline without sufficient revisions.
2. The paper provides scant explanations on why the integration with PINNs is essential, and what advantages does the Hartley transform-based Neural Operator have when combined with PINNs compared to the Physics-Informed Neural Operators (PINO)? Moreover, why is the use of complex numbers in Fourier neural operators considered a limitation? These essential inquiries regarding the necessity of the ideas presented were not well-addressed in the paper. Additionally, in many experimental results, we do not observe a significant improvement of the Hartley Neural Operator (HNO) over the Fourier Neural Operator (FNO). Despite the novelty of the idea, the paper has ample room for improvement in terms of writing and experimental validation.

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The propose to use the Discrete Hartley Transform (HNO) rather than the FFT in the Fourier Neural Operator for a more efficient scheme. In particular, they use the HNO in a PINNs framework within PINO, which is a Neural Operator that adds the PDE to the loss function as a soft constraint. They study two PDEs, the linear parabolic and smooth heat equation and thermodynamic energy equation

### Strengths
- The authors study an important problem of improving deep learning models for solving PDEs, which are prevalent in science and engineering problems.
- It is nice that the authors consider adding PDE information into data-driven Neural Operator methods although this has already been covered with PINO (https://arxiv.org/abs/2111.03794) with mixed results.
- The main novelty is in proposing to use the Hartley Transform basis and its computational efficiency, which I'm not convinced is significant enough novelty.
- The authors identify and highlight an important problem with FNO and operators in general that is the curse of dimensionality. In particular, these methods are not scalable in 3D space and time since the resulting tensor is 6D and the operations are quite expensive. In fact, this real-world 3D case is the motivation for developing DL methods since the curse of dimensionality also impacts the state-of-the-art numerical methods.
- Nice use of numerical techniques such as the fourth order RK4 for the time-stepping rather than the commonly used first order Forward Euler.

### Weaknesses
- Very related work such as the Multi-wavelet Neural Operator (Gupta et. al, "Multiwavelet-based Operator Learning for Differential Equations", NeurIPS 2021) needs to be cited, which proposes using a multi-wavelet instead of Fourier basis and added as an additional baseline.
- The test cases of only 2 simple PDEs, e.g., heat and thermal is too simple and far less than the large number of test cases in FNO (Li et. al, https://arxiv.org/abs/2010.08895, ICLR 2021) and boundary-constrained NO (Saad et., al, https://arxiv.org/abs/2212.07477, ICLR 2023). In addition, Hansen et. al, "Learning Physical Models that Can Respect Conservation Laws", ICML 2023 provides a PDE benchmarking framework from "easy" to "hard" PDEs, where the heat equation studied here is a simple case. I would like the method tested on harder problems with shocks, e.g., Stefan in this paper or hyperbolic conservation laws.
- In addition Saad et. al, ICLR 2023 shows that adding the PINNs loss to Neural Operators can actually be more unstable than the data-driven training and this work should also be cited especially since the authors note their are issues with their proposed approach near the boundary. It seems like future work is needed to resolve this, such as the mixed results on Burgers' equation.
- Rather than starting the introduction with a focus on Neural Operators, I think providing background on PDEs, their importance in the scientific community and numerical methods would be good to add.
- Also references to other state-of-the-art SciML methods, such as MeshGraphNets (Pfaff et. al, https://arxiv.org/abs/2010.03409, ICLR 2021), MP-PDE (Brandstetter et. al, "Message Passing Neural PDE Solvers", ICLR 2022 (Spotlight)) are missing and the main focus of the literature review is Neural Operators.
- The novelty is in using the Hartley Transform basis and not the physics-informed part since that is applying PINNs out-of-the-box to Neural Operators as it was done with PINO.
- Problem definition section where the PDE is defined with initial condition and boundary conditions and the assumptions is missing.
- Some of the preliminaries on Discrete Hartley Transform, e.g., Eqn. 3 can be moved to an appendix, similarly with the long comparison to wavelets (where Gupta et. al should be cited) and discrete cosine transforms (this is interesting to further investigate.) and discussion of PINNs etc should be moved to background appendix sections.
- Listing viscous Burger's in Eqn. 5 seems out of place and should be moved to experiment section or a separate equations section and similarly for diffusion and themodynamic energy to make more room for the description of the proposed method.
- Several of the references seem quite dated.
- References have ? in "Operating on the Diffusion Equation, a key equation in Machine Learning. F"
- Ignoring the initial conditions " (since their GitHub implementation has this commented out in their code anyway)" is not an acceptable answer as to why they are ignored and much too informal in the writing.
- Paper organization is quite poor since there is no dedicated method section. It is mixed with background on these equations.
- The reference to PINO in discussion section is incorrect and should be Li et. al (https://arxiv.org/abs/2111.03794), 2021
- The plots are hard to read the title, x and y-axis labels and are not informative on the quality of the method.
- The work is a nice initial investigation but not ready for publication.

Minor
- Title is a bit long and simpler title may be better for the main message of the paper with too many keywords of Neural Operators and Physics-Informed Networks.
- Best not to start the introduction with bold paragraph header.
- Parenthesis around references is not consistent such as with FNO on the first line.
- Floating equations should have punctuation following it. Commas after equations are missing.
- "salable" instead of scalable at the end of the first paragraph.
- Matérn repeated in "Using GRF For Initial Conditions" paragraph

### Questions
- The method seems effective for PDEs with real-valued solutions but what happens with PDEs with complex-valued output? Will the method will work?
- Why are the authors arguing that the linear diffusion equation is key in ML? This is the simplest PDE that numerical methods can solve to high order.
- Is PINO compared to, i.e., FNO + PINNs loss? The experiments should compare to data-driven FNO and FNO + PINNs loss.
- How are the number of Hartley and Fourier layers chosen? It seems too hand-tuned and weakens the contributions isnce Fourier layers are still needed in some cases.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
