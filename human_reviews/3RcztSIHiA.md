# PDE-GAN for solving PDE optimal control problems more accurately and efficiently

- Decision: Reject
- Scores: 3, 3, 3, 5, 5

## Abstract
PDE optimal control (PDEOC) problems aim to optimize the performance of physical systems constrained by partial differential equations (PDEs) to achieve desired characteristics. Such problems frequently appear in scientific discoveries and are of huge engineering importance. Physics-informed neural networks (PINNs) are recently proposed to solve PDEOC problems, but it may fail to balance the different competing loss terms in such problems. Our work proposes PDE-GAN, a novel approach that puts PINNs in the framework of generative adversarial networks (GANs) “learn the loss function” to address the trade-off between the different competing loss terms effectively. We conducted detailed and comprehensive experiments to compare PDE-GANs with vanilla PINNs in solving four typical and representative PDEOC problems, namely, (1) boundary control on Laplace Equation, (2) time-dependent distributed control on Inviscous Burgers' Equation, (3) initial value control on Burgers' Equation with Viscosity, and (4) time-space-dependent distributed control on Burgers' Equation with Viscosity. Strong numerical evidence supports the PDE-GAN that it achieves the highest accuracy and shortest computation time without the need of line search which is necessary for vanilla PINNs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The aim of this work is to use neural networks to solve PDE-constrained optimal control problems. The main contribution of this work is to introduce the GAN style to train the PINN to solve optimal control problems. The GAN style to train PINN is the previous work.

### Strengths
This paper is not difficult to follow. The proposed method uses the PINN framework to solve the parametric optimal control problems, which can be used to solve high-dimensional problems. The training style is inspired by GAN. Based on such training style, the different terms in the loss can be balanced without tuning by hand.

### Weaknesses
1. The PDE-constrained optimal control problems considered in this work only involve the equality constraint, but in practice, the inequality constraints are typical, e.g., the box constraint. 

2. As stated above, if there exist inequality constraints, the proposed method in this manuscript cannot be applied directly. There are some literature that have already resolved this issue, but this manuscript did not mention, e.g., P. Yin, G. Xiao, K. Tang, and C. Yang, AONN: An adjoint neural network method for all-at-once solutions of parametric optimal control problems, SIAM Journal on Scientific Computing (2024). In this literature, the authors handle more general parametric optimal control problems with complex constraints. Since the AONN inherits the scheme of direct adjoint looping, it does not require tuning the penalty parameter. At the very least, the author should discuss AONN in related work because its key point has a strong correlation with this manuscript.   

3. The experimental results are not so convincing. The loss behavior during training is not shown. Only the final error is reported. However, the training procedure of GAN is unstable. It is hard to say that the performance is better than the baseline.

### Questions
1. The symbol of weight $w$ in line 97 is not consistent with equation (3). 
2. The main point of this work is to remove the hand-picking of the penalty parameter $w$. $w$ is just one hyperparameter, but the discriminator is a network, and it has a lot of hyperparameters, such as the depth and the width. Moreover, PDE-GAN (the proposed method in this manuscript) needs four discriminators. Tuning one hyperparameter is easier than many hyperparameters. 
3. Also, did you try to set $w$ to be learned? 
4. As I said above, this work only considers the equality constraints, but the inequality constraints are common in practice. How do you generalize the proposed approach to more complex constraints?  
5. The numerical experiments did not show the stability of PDE-GAN. Can you show the whole results (e.g., the loss during training) of $D_c$ and $D_u$ to demonstrate the stability of PDE-GAN?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents a GAN-based approach to address the dual optimization problem for solving PDEs with an unknown control function. The method integrates PINNs-like objective functions and loss structures, targeting both forward and inverse problems. In this setup, the generators are tasked with predicting both the control function and the corresponding solution function. Meanwhile, the discriminators are designed to differentiate between valid solutions and zero-valued outputs.

### Strengths
- Originality: The use of GANs under the framework of PINNs is interesting.
- Significance: The problem tackled in this paper is inherently challenging due to the complexity of solving inverse problems under strict physical constraints. The authors’ approach demonstrates a promising direction in addressing these difficulties effectively.

### Weaknesses
- Choice of the Discriminator:
  - The current approach computes discriminators in a point-by-point manner. However, in traditional settings with discrete images, the entire image is typically used as input instead of individual pixels. The authors should provide a clear rationale and experimental results for this design choice.

- Lack of Comprehensive Baseline Comparison:
  - The paper lacks a comparative analysis with relevant methods such as bi-level optimization techniques. While these methods are mentioned in the related work, the absence of a thorough experimental comparison is not adequately justified.
  - Furthermore, there is no comparison with existing approaches like Physics-informed DeepONet (Wang et al., 2021), which address similar challenges. A direct comparison would help contextualize the proposed method’s performance in relation to established approaches exploring similar ideas.

- Complexity of Addressed Problems:
  - The paper does not sufficiently communicate the complexity or importance of the problems it addresses, making it challenging for readers to assess the novelty and significance of the proposed solution.
  - For example, Mowlavi and Nabi (2022) explore a range of equations, from simpler Laplace problems to more complex 2D Navier-Stokes equations, in their study of PDE-based optimal control (PDEOC). Including results for similarly challenging equations in this work would strengthen the paper’s validation and impact.

- Readability and Clarity:
  - The submission requires revisions to enhance readability and clearly communicate the main ideas. Key areas for improvement include:
    - Unifying the notations for the generator, solution function, and control function.
    - Organizing and presenting the definitions of different components in a clearer and more cohesive manner.

References:
- Mowlavi and Nabi. (2022). *Optimal control of PDEs using physics-informed neural networks.*
- Wang et al. (2021). *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets.*

### Questions
Please refer to Weaknesses.

### Soundness
2

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
This paper introduces PDE-GAN a framework for solving PDEs optimal control problem with a PINN and an adversarial loss. The framework is an extension of the hard-constraint PINNs, which imposes constraints directly on the PINN solution instead of enforcing them  through loss penalties. In the optimal control configuration, the pde and cost objectives are balanced by a weight $w$. Existing implementations require for a search of the best $w$ to find a compromise between the two loss terms. The adversarial loss aims at mitigating the need for searching for the optimal weight value. The authors conducted experiments on Laplace and Burgers equation with several control setups.

### Strengths
* The method seems to work and obtains good experimental results on the different problems.
* The overall running time is less than of the Soft-PINNs and Hard-PINNs baselines when linear search is taken into account.

### Weaknesses
* The motivations of the paper are not well-founded to my view. The authors do not explain why the adversarial approach is needed to balance the different loss terms and never discuss nor test possible alternatives.
* The paper has some writing issues and suffers from a lack of clarity. Sections 3.1 and 3.2 should be within a separate background section. The notations introduced in Section 3.3 are difficult to read, especially RHS and LHS which are not explicitly detailed. I suggest using several examples to improve clarity.
* The running time is greater than that of a single PINNs. 
* The importance of linear search for the other methods is not explained properly.
* The results are only marginally better than Hard-PINNs except for the second equation.
* The authors do not discuss their architectural choices, especially the adversarial loss and the noise injection.
* I suggest changing the name of the framework to Adversarial-PINNs as it is more faithful to the core idea fo the paper.

### Questions
* Why would the adversarial loss help for solving PDE optimal control problems ?
* Have the authors tried other techniques to try balancing the two losses ?
* Have the authors tried without the hard constraints ?

### Soundness
1

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
4

### Summary
This paper proposes to combine PDE constraints with generative adversarial training as a method to solve PDE optimal control problems. The method is a GAN-based analogue to PINNs and outperforms the latter in some numerical experiments.

### Strengths
The paper proposes a new method for PDE-constrained control problems and demonstrates its superiority to PINNs in several numerical experiments.

### Weaknesses
The paper is far from well-written. First, it contains many grammatical and spelling errors that distract from the overall contribution. Beyond this, the writing (especially in Section 3) is unclear and it is difficult to understand the authors' reasoning.

In addition, this paper lacks any comparison to classical techniques for solving PDE-constrained optimal control problems. The proposed method is only compared to PINNs, but PINNs are not exactly state-of-the-art methods and can be quite easy to beat in many circumstances. As such, I am not convinced that PDE-GAN is the best method for solving these problems.

### Questions
1. Why do we need two generators and two discriminators, what are their respective purposes?
2. What unit is used in Table 2? That is, are results presented in wallclock time?  Such information is relevant for anyone to make a fair comparison to this work.
3. Have the authors considered comparing their algorithm to classical techniques (instead of only PINNs)?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a novel method PDE-GAN, which integrates PINNs into the GANs framework to solve the PDEOC problems. The authors address the limitations of traditional PINN approaches in balancing competing loss terms and reducing computational time, particularly by eliminating the need for exhaustive line search in weight tuning. They validate their method on four representative PDEOC problems, including linear and nonlinear PDEs, and various types of control (boundary, spatio-temporal domain, and time-domain distributed equations) and compared with soft-PINNs and hard-PINNs.

### Strengths
The integration of PINNs into the GAN framework is a new approach for solving PDEOC problems. This allows to use two additional
discriminator networks to adaptively adjust the loss function, allowing for the adjustment of weights between different competing loss terms. Compared to Soft-PINNs and Hard-PINNs, PDE-GAN can find the optimal control without the need for cumbersome line search, offering a more flexible structure, higher efficiency, and greater accuracy.

### Weaknesses
The paper lacks a theoretical analysis explaining why integrating PINNs into a GAN framework results in improved performance. Theoretical insights or proofs would strengthen the paper, espeically without any line search, the comprehensive evaluations of the results could be beneficial, however, using the experimental results to address its advantages is the main weakness.

### Questions
In Algorithm 1, why do you just limit the number of epochs 500?  

It seems that the algorithm updates the generator and discriminator together without any condition, why?

How do you properly set Bound1 and Bound2?

Table 2 shows the running time for PDE-GAN, which is the total? the mean? Does it include the training time for GAN?

### Soundness
2

### Presentation
2

### Contribution
3
