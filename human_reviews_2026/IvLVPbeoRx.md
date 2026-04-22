# Physics-informed learning under mixing: How physical knowledge speeds up learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
A major challenge in physics-informed machine learning is to understand how the incorporation of prior domain knowledge affects learning rates when data are dependent. Focusing on empirical risk minimization with physics-informed regularization, we derive complexity-dependent bounds on the excess risk in probability and in expectation. We prove that, when the physical prior information is aligned, the learning rate improves from the (slow) Sobolev minimax rate to the (fast) optimal i.i.d. one without sample-size deflation due to data dependence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript presents a theoretical analysis of physics informed machine learning with data are dependent. The work is an attempt to address the open challenge why incorporating physics prior can benefit data-drive learning.

### Strengths
1. This paper addresses the open challenge: the theoretical soundness of physics informed ML, beyond empirical evidence and intuitive understanding that the prior physics knowledge is conductive to learning. 
2. Theoretical analysis appears to be rigorous (but I didn't check all the proofs).

### Weaknesses
1. The organization and exposition of the manuscript can be improved. Without loss of theoretical rigor, it would be great if the intuitive explanations can be provided for deep learning practitioners the meaning of the theoretical results in practice. 
2. Related to the first question, intuitive understanding of the importance of conditions related to key variables such as $T$, as in Theorem 5.2 and $\lambda_T$ as in theorem 5.1, would greatly benefit the reader.
3. It's understandable this is a theoretical paper, but the only numerical experiment does add the weight to the paper. A well thought-off experiments to demonstrate the conditions related to $T$ and $\lambda_T$ would also greatly benefit the readers, see the concern above.

### Questions
1. Minor formatting issue: all equations should be numbered.

### Soundness
3

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
This paper develops a rigorous statistical-learning theory for physics-informed learning under dependent (mixing) data. It studies regularized empirical risk minimization where the regularizer encodes known physical laws through a linear differential operator, and the data arise from a stochastic dynamical system $X_{t+1}=f_{\star}(X_t)+W_t$. Using tools from Sobolev-space analysis, the small-ball method, and martingale offset complexity, the authors prove complexity-dependent excess-risk bounds showing that when the physical prior is aligned with the ground-truth dynamics (i.e., PDE residual of $f_{\star}$ is nearly zero), the learning rate acclerates from the traditional Sobolev minimax rate $O(T^{-d})$ to the fast i.i.d optimal rate $O(1/T)$, even when samples are correlated. A simple unicycle-dynamics experiment empirically confirms the predicted speed-up, demonstrating that properly aligned physics-based regularization can provably improve sample efficiency in learning dynamical systems.

### Strengths
The paper addresses an important topic at the intersection of physics-informed learning and statistical learning under dependence. The authors aim to provide a theoretical foundation for when incorporating physical structure can mitigate the challenges of temporally correlated data. In this context, the paper offers several notable positive aspects:

1. The central idea of using elliptic differential operators to encode physical priors and recover i.i.d.-like learning rates (under suitable alignment) despite sample correlation is novel and insightful.

2. The theoretical analysis is mathematically sound. The assumptions are clearly stated, and the main theorems follow logically from the lemmas and proof techniques provided. The paper carefully extends existing theory to accommodate a Sobolev-based physics regularizer in the presence of Markovian data, which is nontrivial. While I did not look into the proofs in detail and some proof components rely on established techniques, the overall combination is coherent and technically competent.

3. The results contribute to a better theoretical understanding of physics-informed learning, particularly in settings where data dependence is unavoidable, such as system identification and scientific modeling. Showing that incorporating correct physical structure can mitigate the adverse effects of temporal correlation is a useful insight, although demonstrated under idealized assumptions.

### Weaknesses
While the paper makes a meaningful theoretical contribution and is clearly written, several limitations temper its overall impact and practical relevance. Most of these relate to the idealized nature of the assumptions and the gap between the theory and empirical applicability. The following points highlight areas where the work could be strengthened or where additional clarification or experimentation would improve the contribution:


1. The fast-rate $O(1 / T)$ convergence is achieved only under an idealized knowledge alignment condition, namely when $\|\|D\left(f_{\star}\right)\|\|_{L^2} \approx 0$. In practice, the physical operator $D$ is rarely known with such precision, and even modest mismatch can revert the rate to the slower $O\left(T^{-d}\right)$ regime. The paper does not provide a quantitative robustness analysis that would clarify how sensitive the rates are to partial or imperfect alignment, which limits the applicability of the theoretical claims in realistic settings.

2. The theoretical guarantees depend on $\lambda_T$ choices tied to latent problem quantities (e.g., $\Psi(f_{\star}), \sigma_W^2$ ). While the authors note that cross-validation could be used in principle, the paper offers limited practical guidance or empirical validation for tuning $\lambda_T$ in realistic settings.

3. The single toy experiment (unicycle) is supportive but narrow (one setting, small MLP, no real-world data/baselines), so the robustness of the phase transition across architectures/noise regimes remains unclear. Even in this controlled synthetic setting, implementing the physics-informed regularization term $\Psi(f)= \|\|D(f)\|\|_{L^2}^2$ may require evaluating higher-order derivatives and Sobolev norms, which can be computationally demanding in higher dimensions. In addition, the paper does not discuss how discretization, numerical differentiation, or instability in PDE solvers or neural approximators would affect the performance or validity of the theoretical bounds, leaving a gap between the continuous theory and practical implementation.

4. The framework relies on $D$ being a known, linear elliptic operator. Many modern scientific machine learning applications involve unknown or nonlinear physics, or operators that must be learned jointly with the model (e.g., operator learning, neural PDE surrogates, and PINNs). As the current analysis does not extend to such settings, it is unclear how the insights would generalize to applications where the governing equations are only partially known or inherently nonlinear.

### Questions
In relation to the weaknesses stated above, please see my questions/comments below:

1. The paper presents a two-term bound combining the $T^{-d}$ and $T^{-1}$ rates, but it is unclear how to interpret the intermediate regime. Could the authors clarify whether the transition between the two regimes is smooth or abrupt, and provide any threshold conditions under which the fast term becomes dominant? In addition, it'd be great if the authors could shed some light on the following: how robust this behavior is to imperfect knowledge alignment? For example, if $\|\|D\left(f_{\star}\right)\|\|_{L^2}$ is small but nonzero, does the convergence rate degrade gradually (and remain faster than $T^{-d}$ ) or does it collapse sharply to the slower regime?

3. Since $\Psi(f)=\|\|D(f)\|\|_{L^2}^2$ may involve higher-order derivatives, do the authors foresee computational or numerical challenges when scaling beyond low-dimensional synthetic problems? Any guidance on discretization or numerical stability when implementing this regularizer in practice would be helpful.

3. The analysis assumes that $D$ is a linear and elliptic operator, which can be restrictive. Could the authors comment on whether any part of the analysis may extend to mildly nonlinear or non-elliptic operators, or if linear ellipticity is fundamentally required for the proof techniques used?

4. The paper imposes $s \geq 2 d_X$ to ensure the burn-in term vanishes. Is this threshold believed to be intrinsic to the problem, or could the two-phase rate behavior persist under weaker smoothness assumptions (e.g., $s>d_X$ or $s>3 d_X / 2$ ) with possibly different constants?

5. Could the authors provide any preliminary numerical results or intuition on how large $T$ must be for the asymptotic behavior to manifest in practice? Even a brief discussion of the finite-sample regime would help readers assess when the theoretical rates become observable.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper derives complexity-dependent bounds that formally characterize the impact of aligned prior domain knowledge on learning rates in physics-informed machine learning settings. This is approached by focusing on bounds of the excess risk in regularized ERM under dependent data generated from non-linear dynamical systems. Results under the combined assumptions of physics-informed regularization and non-IID data have yet to be shown in existing literature, and the derived bounds reveal new insights into the relationship between statistical learning theory and physics-informed ML.

### Strengths
- The paper is well-organized and has a presentation that's easy to follow from start to finish. I appreciated the clear groups in Section 3 and the visual distinction for all stated assumptions.
- The authors address an important theoretical challenge in the physics-informed ML space, with a fundamental connection to all PIML architectures and practitioners looking to tackle scientific problems. Characterizing the impact of well-aligned domain knowledge takes a clear step toward helping researchers and practitioners alike when it comes to justifying additional time spent refining assumptions and calibrating inductive biases.
- Relaxing the IID assumptions from prior work (e.g., Doumèche et al. 2024) appears to be an important adjustment that broadens the reach of the resulting bounds. In most PIML settings, one is working with heavily dependent sequential data, so this seems a prudent step toward establishing useful bounds that mirror the real world.

### Weaknesses
- Assumption 5 is quite strong, limiting the applicability of the results in many real-world physics-informed modeling scenarios. For instance, unless additional constraints are applied, popular methods like physics-informed neural networks would presumably violate this assumption, and in these cases it's unclear how one should think about the applicability of the results. I understand details on approximations were stated as out of scope, but it would be nice to include some discussion/analysis of the likely ways practitioners may violate assumptions in practice and what elements of the original bound still apply (if any). 
- Similar to the above point, Assumption 4 also excludes common practical scenarios that leverage non-linear PDE priors. Discussion on what remains of the bound in these cases would be helpful for position the paper's analysis in a broader context.
- This is somewhat tangential to the core theoretical aims of the paper, but it would be instructive to include a more holistic case study. In particular, the impact of using misaligned/incomplete prior physics knowledge and how this empirically threads between the "with knowledge" and "without knowledge" scenarios. The sample sizes at which various levels of knowledge alignment outperform others, if only for short ranges of $T$, would also be helpful to characterize insofar as they relate to the paper's analysis of sample size.

### Questions
- Are there any promising avenues for achieving the stronger burn-in requirement while relaxing $s\ge 2d_X$ toward the standard $s\ge d_X/2$, or are there clear reasons to believe this is a necessary tradeoff? Can factors/assumptions be loosened in its place while maintaining the original bound, giving another route to the same result?
- In the numerical experiment and Figure 2, does the reported slope fit account for burn-in (if even needed in this case)? It would be instructive to see how this shows up here, and/or when burn-in exceeds realistically attainable sample sizes in practice.

### Soundness
4

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
This paper presents a theoretical analysis of physics-informed learning in the presence of dependent data. The authors analyze empirical risk minimization with a physics-informed regularizer that encodes known physical priors in the form of elliptic PDE constraints. Using the small-ball method and martingale offset complexity, they derive complexity-dependent excess risk bounds both in probability and expectation. The main result shows that when the physical prior regularizer is aligned with the true dynamics, the convergence rate improves from the slow Sobolev minimax rate to the optimal i.i.d. rate, even with dependent samples. The theoretical framework is supported by a unicycle dynamics experiment illustrating the empirical benefit of incorporating physics-informed regularization.

### Strengths
1. The paper rigorously extends the small-ball method and offset complexity analysis to dependent data settings with physics-informed regularization, which is technically novel.

2. The paper establishes a clear theoretical connection between physical priors (encoded as elliptic PDE constraints) and improved learning rates, addressing a long-standing gap in understanding the benefits of physics-informed models.

3. The proofs and appendices appear detailed, well-grounded in functional analysis, Sobolev space theory, and dependent process theory.

4. The  experiment, though simple, can demonstrate the theoretical prediction that physical priors accelerate convergence.

### Weaknesses
1. The analysis assumes elliptic linear PDE operators; it is unclear whether the results generalize to non-elliptic, nonlinear, or mixed-type operators often seen in physics-informed neural networks (PINNs). The gap or difficulty is not well addressed. 

2. The optimization error's influnene is not discussed in the theoretical part and the numerical example. Note that the physics-informed regularizer will increase the stiffness of the Hessian matrix and increase difficulties for the optimization in PINNs, which is not aligned with the main result (convergence rate improves with physics-inforemed regularizer added)

3. Too many assumptions may not hold for complex real-world systems, a more interpretable or verifiable condition would strengthen the practical relevance.

4. Only one low-dimensional example is provided. Additional tests on nonlinear PDE systems or stochastic dynamical systems would reinforce the claims.

### Questions
1. see in the weakness

2. The phsyics-informed regularizer is limited to linear elliptic PDEs. However, the data is dependent, which is always occured in dynamical systems. Is this two conflicting? The numerical example is an ODE, not elliptic PDEs. How the convergence rate behaves for common PINNs problems such as Poisson equation or Darcy flow problem? How can the proposed theory be connected to the optimization landscape of neural networks trained with PINNs?

3. Does the empirical rate in the numerical example persist when neural architectures differ from MLPs, or with stochastic training?

4. Can the presentation be more friendly to the general ICLR audience without additional intuition or graphical explanation of the key proof mechanisms? The current form is mathematically dense and more suitable to journals like JMLR, not general top conferences.

### Soundness
3

### Presentation
2

### Contribution
3
