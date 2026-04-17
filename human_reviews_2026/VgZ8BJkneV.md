# LieDynNet: Learning Lie Symmetries from Spatiotemporal Data

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Continuous symmetries of dynamical systems—transformations that map solution trajectories or spatiotemporal fields to new, valid solutions—are powerful tools for analysis, reduction, and control. Prior work on symmetry discovery broadly falls into two categories: methods that prioritize Lie-algebraic structure but operate on static datasets rather than dynamical systems, and methods that discover symmetries for dynamical systems but often do not enforce algebraic structure. Across both threads, most approaches also neglect the infinitesimal invariance condition (IIC)—that prolonged generators annihilate the governing equations. To fill this gap, we introduce LieDynNet, which learns Lie symmetry generators directly from data by pairing neural ODE/PDE models with two families of constraints: dynamical validity, enforced both via IIC (via generator prolongations) and under finite flows; and algebraic soundness, enforcing closure, antisymmetry, and the Jacobi identity so the generators form a Lie algebra. The framework is model-agnostic and applies to both ODEs and PDEs without hand-crafted priors. On canonical dynamical systems, LieDynNet recovers symmetry algebras and associated invariants from data, showing that learned symmetries can be simultaneously algebraically consistent and dynamically faithful. These results provide a practical, data-driven route to discovering the symmetry structure of complex dynamical phenomena.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses discovering the Lie symmetries of dynamical systems. Based on the infinitesimal criterion of Lie point symmetry, the authors propose training a surrogate for the governing differential equation and a vector field that annihilates the surrogate model upon prolongation to given derivative orders. In addition, several regularization terms are added to the training objective to ensure the vector field forms a valid Lie algebra. Experiments include several low-dimensional ODE/PDE systems, where the proposed method discovers infinitesimal transformations that, upon integration, preserve the overall shapes of the solutions, while the residual of the differential equations, as direct evidence of the validity of discovered symmetry, is not shown.

### Strengths
The paper identifies some shortcomings of existing methods for discovering symmetry in dynamical systems, such as the lack of a guarantee of a valid Lie algebraic structure in the algebra spanned by *multiple* infinitesimal generators, and the proposed method clearly addresses these problems. The description of the methodology is detailed and easy to follow. The experiment of finding the 8 generators of the simple harmonic oscillator is an interesting challenge, but more evaluation needs to be included to assess the significance of the results.

### Weaknesses
## Contribution

My main concern is the novelty of this paper. Part of the contributions claimed in this work are already present in existing papers. For example, Forestano et al have introduced the *closure* loss (the 4th item in your list of learning objectives), and Ko et al have introduced the flow-based loss (the 2nd item in your list).

Also, the authors mentioned that the method is *prior-free*, as opposed to Ko et al, which requires the governing equations to evaluate the symmetry validity loss. However, as stated in L206-207, your method trains a PINN as a surrogate, which also involves the differential equation. While the subsequent step of learning vector fields does not require this prior knowledge, this is not true for the entire method with all components combined.

## Soundness

The method for learning the vector fields is basically to parameterize them using neural networks and introduce different loss terms to ensure they meet certain conditions. However, certain design choices and the motivations behind them remain elusive to me. Specifically:
* According to the infinitesimal criterion of Lie point symmetry, the finite group symmetry is equivalent to the infinitesimal invariance $v(F)=0$. Is it necessary to use both infinitesimal invariance and flow-based validity then? What would happen if only one of them were included in loss?
* How are the structure constants $c_{ij}^k$ computed? From the fact that you introduced a constancy loss, I suppose they are outputs of some neural net that takes points in jet space as input. However, structure constants, as the name suggests, should be constant regardless of where the vector fields are evaluated. Why not just make them learnable constants? Or compute them differentiably as the solution to minimizing the closure loss? What are the pros and cons of these different approaches?
* For PDEs, the proposed method trains a PINN to directly fit the solutions. Can you do the same thing for ODEs? What is the advantage of fitting the vector field $f_\theta$ instead?

## Experiments

* In some dynamical systems considered in the experiments, only 1 infinitesimal generator is trained. In this case, those additional loss terms such as closure become unnecessary. Also, for those experiments, the infinitesimal generators approximately correspond to the time translation symmetry, which can be readily recovered from the equation or the surrogate model itself.
* The paper does not provide sufficient evaluation of the trained infinitesimal generators. The figures only show the trajectories of transformed solutions and claim that the *shapes* of the transformed solutions remain similar, but this does not guarantee that the transformed solutions are still solutions to the same equation at all. I'd suggest including the residual error from substituting the transformed solutions into the differential equations to show whether they are indeed the symmetry of the differential equations by definition. Similarly, the vector fields are visualized for the simple harmonic oscillator, but readers might not be able to interpret the meaning of these vector fields just by looking at those plots. Do they match the ground truth 8-dimensional symmetry of this system? What are the structure constants of the discovered generators? Are they closed under Lie brackets? All of these are required to assess if the discovered symmetry is correct or not.
* There is no baseline comparison, which is also a critical issue.

## Presentation
* The methodology and the experiment sections can be organized better. For example, I'd suggest moving L301-320 to the method section and focusing on showing and explaining results in the experiment section.
* Math writing needs to be more clear and precise. For example, write out the actual formula for the text in the equation $L_\text{flow}$; define the $\lambda_{\text{max}}$ and $\lambda_\text{min}$ explicitly in L256; check for typos such as the missing subscript in the vector field in L376.
* Table 3 is not very informative.

### Questions
see Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LieDynNet, a prior-free framework for learning Lie point symmetries of unknown ODE and PDEs directly from spatiotemporal data. The key innovation is a unified objective that couples dynamical validity and algebraic soundness. The method first trains a differentiable neural surrogate and then learns a set of infinitesimal generators whose exponentials form a connected Lie symmetry group. The pipeline is model-agnostic and applies to both ODEs and PDEs.
The paper’s main contributions are:
1. A prior-free symmetry discovery framework that learns continuous symmetries without templates, canonical coordinates or physics priors.
2. A practical objective that jointly enforces infinitesimal and finite-flow invariance while imposing Lie algebra consistency which yields algebraically sound, dynamically valid symmetries.
3. Validation on canonical benchmarks, showing recovery of known symmetry families and solution-to-solution preservation under learned $\varepsilon$-flows.

### Strengths
1. The paper presents a prior-free, model-agnostic framework that learns Lie point symmetries directly from data. The approach applies uniformly to both ODEs and PDEs, indicating the method can be broadly applicable across dynamical systems.
2. The proposed LieDynNet overcomes the limitations of prior work with a practical objective so that the learned symmetries are both $dynamically \text{ } valid$ and $algebraically \text{ } sound$.
3. The paper is well-structured with clear explanations of the proposed method. It is validated across five canonical systems, demonstrating the recovery of known symmetry families and solution to solution preservation under learned flows.

### Weaknesses
1. It is recommended to compare LieDynNet across diverse differentiable surrogates under identical data and clarify the comparison in the paper to substantiate the model-agnostic claim.

2. While the symmetry-learning stage equation-agnostic by design, the PDE surrogate is trained with PDE residuals and IC/BC penalties. Please clarify this point and if possible, include a comparison of surrogates trained solely on data without residuals for Burgers PDE.

3. Please make the prolongation order $k$-selection rationale explicit; define a quantitative plateau criterion for ‘stabilizing’ $L_{inv}$, and include sensitivity plots of $L_{inv}$ versus $k$. The trade-off with differential noise will become more clear.

4. From your supplementary codes, the loss weights in $L$ (e.g., $w_{anti}$, $w_{jac}$, and so on) appear to be set to 1.0 or to similar values. You should briefly justify this choice; If the loss weights were not selected with sensitivity to each term’s scale, include a sensitivity study and present the recommended ratios among the weights. 
5. The mathematical setup allows $p$ independent and $q$ dependent variables and seems to be applicable to higher dimensional settings, all experiments are one dimensional (time for ODEs, 1D space for PDE). If feasible, include at least one $p>1$ case, or provide implementation details for $p$>1 to demonstrate applicability beyond 1D.

### Questions
I hope the authors can clarify my concerns about the above weakness. But if I missed some critical points, please let me know in the rebuttal.

### Soundness
2

### Presentation
2

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
The paper introduces LieDynNet, a framework for learning continuous Lie point symmetries of unknown ODE and PDE systems directly from data. The approach first fits a neural surrogate to approximate the underlying dynamics and then learns infinitesimal generators that (i) satisfy the infinitesimal invariance condition (IIC) via prolongations, (ii) preserve solutions under finite $\epsilon$-flows, and (iii) enforce the Lie-algebraic structure (e.g., closure, antisymmetry, Jacobi, …) The method is explicitly prior-free, requiring no equation templates, symmetry catalogs, or physics priors. It is validated on canonical ODE benchmarks and the one-dimensional Burgers equation, demonstrating that the recovered generators form a consistent Lie algebra and maintain dynamical validity on the learned surrogate.

### Strengths
- The motivation of the paper is clear. Symmetry discovery in dynamical systems is a highly active area with significant potential impact in scientific machine learning.

 - To the best of my knowledge, this is the first framework that jointly learns surrogate dynamics and discovers symmetries that form valid Lie group structures in a fully prior-free setting, yielding symmetry algebras and invariants directly from data for both ODEs and PDEs. The method leverages the formalism of Lie theory and implements it end-to-end through neural surrogates and carefully designed loss functions. 

- Given the mathematical depth of the work, the paper is overall well written and accessible.

### Weaknesses
- Discovering symmetries without any physical prior is inherently risky. As the authors acknowledge, the symmetry discovery is performed on a neural surrogate dynamics, rather than the underlying system. Thus, what is actually identified is the symmetry group of the surrogate, not necessarily that of the true data-generating process. There is no guarantee that these coincide, especially under noise or imperfect surrogate fitting.

- Ensuring both algebraic soundness and dynamical validity requires jointly optimizing multiple coupled objectives. This design is principled but increases optimization complexity and computational cost. The paper outlines practical training schedules and JVP-based implementations, yet it omits runtime or memory benchmarks. Including a compute summary table would strengthen the empirical analysis.

- The paper lacks experimental comparison with contemporary approaches. For example, although Ko et al. is based on known PDE priors, it could serve as a useful baseline to demonstrate the advantage of the proposed architectural and objective constraints under the prior-free formulation.

- Moreover, the paper omits a closely related and highly relevant reference: Hu et al., “Explicit Discovery of Nonlinear Symmetries from Dynamic Data,” ICML 2025, which also employs surrogate ODE/PDE modeling (though based on symbolic libraries) combined with Olver’s prolongation and IIC for Lie symmetry discovery. Discussing or benchmarking against these works would help clarify the paper’s relative strengths and limitations.

### Questions
Please refer to Weaknesses section. My main concerns are:

- Because this paper is prior-free, it likely inherits a fundamental identifiability challenge: whether the proposed method can recover the true symmetry group solely from data. In other words, under what assumptions would the recovered symmetry converge to that of the true underlying system (given infinitely many data samples)? 

- Additionally, including a comparison with similar yet prior-based approaches could further clarify the advantages of this work.

### Soundness
3

### Presentation
3

### Contribution
3
