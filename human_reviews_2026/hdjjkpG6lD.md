# U-RankMOEA: Learning to Optimize High-Dimensional Expensive Multi-Objective Problems

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
We present U-RankMOEA, a unified framework that integrates rank-based screening, uncertainty decomposition, and history-shaped acquisition for high-dimensional and costly multi-objective optimization under limited evaluation budgets. A calibrated Bayesian classifier first produces epistemic uncertainty over non-domination ranks, enabling inexpensive yet reliable generation of promising candidates. Deep Gaussian Process surrogates then disentangle reducible from irreducible uncertainty for each objective, providing accurate means and risk-aware variances. Finally, a lightweight acquisition network trained online distills past hypervolume gains into a predictive score that guides expensive evaluations toward regions expected to balance convergence and diversity. Staged screening, adaptive sampling, and amortized surrogate updates reduce computational overhead without sacrificing fidelity. Across widely-used DTLZ and ZDT families and a subsurface energy extraction task, U-RankMOEA consistently outperforms representative surrogate-assisted and classifier-augmented baselines in terms of Pareto front proximity and hypervolume dominance, while exhibiting robust sensitivity behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a framework of multi-objective optimization called U-RankMOEA, which is a hybrid type approach of the Bayesian modeling and evolutionary strategies. U-RankMOEA combines a variety of components such as rank-based classifier of nondominant sorting, complexity reduced deep GP, and a fitted acquisition function.

### Strengths
The paper constructs a practical multi-objective optimization algorithm by combining a variety of component techniques, such as rank-based classifier, evolutionary procedure, deep GP, and fitting of historical acquisition function values.

### Weaknesses
The proposed method is a collection of approximation strategies, and overall, each of components does not have particular technical novelty. 

In my understanding, in the end, the proposed method is an approximate EHVI. In this sense, recent EHVI based methods should have been in baselines. Gradient-based optimization of EHVI has been studied and therefore many (time consuming) expected volume computations may not be required. Further, in the case of the bi-objective case (for which many results are provided), the volume computation is not so difficult compared with higher output dimension case.

For the rank-based model, the model itself is directly from an existing method, and so, a contribution may be in uncertainty quantification, but a rationale behind the uncertainty quantification is not clear.

Although a theoretical analysis is shown in appendix, it is quite sudden (nothing is explained in the main text). The technical descriptions in the analysis are too messy by which I couldn't follow the detail (in my current understanding, the complete proof is not provided). Currently, I don't think the analysis is reliable.

In H.2, the assumption (A5) seems too strong (though as mentioned above, I couldn't fully follow the proof).

Minor:
- 'K nondomination rank categories' is not defined. Predicting this nondomination rank is probably directly from an existing study, but to be self-contained, more detailed definition should have been provided.
- About the theoretical analysis, even for the exact EHVI, the theoretical guarantee has not been widely studied (I only know the case of a scalarized acquisition). Since no relationship with existing analysis is revealed for the analysis, I do not find how the analysis is interpreted in the context of related theoretical analysis. 
- The proposed method selects a next point by an approximate EHVI and evolutionary strategies are in the acquisition function optimization. Then, it seems the proposed method should be called Bayesian optimization rather than an evolutionary algorithm (EA). I thinks the name U-RankMO'EA' is confusing.

### Questions
I don't understand why mutual information (7) can be seen as an epistemic uncertainty. Further, (7) is not mutual information.

The first term of (7) is entropy of \bar{p}, and the second term is the average entropy of p^s for each s \in [S(x)]. Therefore, both of them is seemingly not the uncertainty derived by MC dropout (e.g., it should reflect the variation of p^s over different s). This makes interpretation of u_ep^clf unclear and unreliable.

I don't understand technical novelty of the uncertainty decomposition of complexity-reduced deep GP. What is the technical difficulty to decompose the uncertainty in deep GP?

How to estimate \hat{s}_div is not written. 

How is K selected?

In Table 4 (ablation study), the proposed U-RankMOEA shows best performance. However, some components in U-RankMOEA are introduced for reducing computational complexity, not for improving sample-efficiency (iteration-efficiency) in my understanding. Why the results are improved? (I assume the authors employed the 300 max evaluations setting described in Sec4.1).

In SecH.2, what does 'surrogate+acquisition fidelity' in (A5) mean?

The analysis in SecH seemingly does not depend on rank-conditioned offspring generation. Why?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes U-RankMOEA, a framework for high-dimensional and expensive multi-objective optimization. Such problems occur in various disciplines, for instance, in engineering, where one aims to optimize airfoil designs for power output and weight. Various approaches, in particular multi-objective Bayesian optimization methods, tackle the problem setting of expensive multi-objective problems of moderate dimensionality. However, these approaches often struggle when the number of decision variables lies in the dozens, hundreds, or more. This paper proposes an approach that is grounded in Bayesian optimization but addresses high-dimensional problems through the following three contributions. First, the paper proposes a Bayesian classifier that uses Monte-Carlo dropout to produce uncertainty estimates. This classifier provides relatively cheap inference and, therefore, can be used to screen many candidate solutions. Second, a deep Gaussian process surrogate model is employed that, per objective, provides a predictive mean and estimates of the epistemic and aleatoric variance. Third, these values are used in a neural-network-based acquisition function to predict the expected hypervolume and a diversity score that are used to steer the optimization.

The proposed method is benchmarked against several other evolutionary methods on several synthetic problems, indicating better performance throughout the benchmark according to two metrics. Furthermore, the authors test their approach in a geothermal reservoir optimization case study.

### Strengths
The problem addressed by this paper is relevant, and the need for scalable methods is well-motivated in the introduction.

The general approach is reasonable. Screening a large number of candidates by a cheaper classification-based approach helps manage the computational cost, and the other contributions are also reasonable for the problem setting. 

The method is evaluated on a wide range of synthetic benchmarks of varying dimensionality.

### Weaknesses
One problem I see in this paper is that some parts are relatively vague while others are overly detailed. For instance, Section 3.2 states that the method uses sparse and low-rank GP approximations, including inducing points and RFF features, without giving additional details on this important design decision. Similarly, Section 3.4 lacks important details and/or uses confusing notation. The symbols $\theta_{\ell,m}$, $\sigma_m^2$, and $\text{Var}_{\text{ep},m}$ are not properly introduced. Finally, the construction of the acquisition network (Section 3.5) is also very brief. 

While some parts are overly detailed (for example, the network structure in Eqs. (1)-(4) could go into the appendix), the paper fails to give a high-level intuition for the overall approach. I would expect the high-level description in Appendix A to go into the main text. At the same time, some design decisions deserve a more complete motivation. Why are Deep GPs necessary? Why is the acquisition function network a good design decision? And how do these design decisions relate to more traditional design approaches for multi-objective Bayesian optimization?

Something is off in the bibliography. It seems that the authors added notes to the bibiliography items (“cite specific paper used”) did not revise the bibliography before submitting this work. At least one reference (Müller et al., 2023) does not exist under this title. Some relevant references are missing. [1] should be cited when mentioning log-space formulations to improve numerical stability. Section 2.4 should cite foundational works on deep Gaussian processes. Section 3.2 should cite relevant sources for RFF and Nyström features, e.g., [2].

The empirical evaluation, particularly in the main text, is not benchmarking against standard Bayesian optimization techniques for multi-objective optimization. Appendix F.1 features such an evaluation, but it is limited to 500-dimensional problems and 5 objectives. It would be good to have the GP-EI, RF-EI, and GP-HV baselines in every experiment (especially the lower-dimensional ones). Furthermore, it would be good to also show the hypervolume or IGD per iteration to be able to assess not only the final performance of each method.
The paper should discuss limitations in more detail. Deep GPs and neural acquisition functions are more difficult to train than more basic techniques. How do these problems limit the applicability of the proposed method?

[1] Ament, Sebastian, et al. "Unexpected improvements to expected improvement for Bayesian optimization." Advances in Neural Information Processing Systems 36 (2023): 20577-20612.

[2] Rahimi, Ali, and Benjamin Recht. "Random features for large-scale kernel machines." Advances in neural information processing systems 20 (2007).

### Questions
- Do you model correlations between objectives? 
- How do you benchmark against GP-EI in App F.1 if it’s not multi-objective?
- Why is Section F.1 benchmarking against GP-HV or GP-EI only on 500-dimensional problem instances?
- What exact setup do you use for GP-EI and GP-HV?
- In how far are Müller et al. using non-standard acquisition functions? (referring to the statement in line 127, ff.)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of high-dimensional, expensive multi-objective optimization (HE-MOO), where objective function evaluations are extremely limited. The authors propose U-RankMOEA, a unified, uncertainty-aware framework that "co-designs" three main components: (1) a Bayesian neural classifier with calibrated epistemic uncertainty for rank-guided generation, (2) Deep Gaussian Process (Deep GP) surrogates that disentangle epistemic and aleatoric uncertainty, and (3) a history-aware acquisition network trained online to predict hypervolume gains. The core contribution is the principled integration of these components, unified by a careful management of uncertainty. The method is validated on challenging DTLZ/ZDT benchmarks with up to 200 variables, and a 160-variable geothermal optimization task, demonstrating significant state-of-the-art performance in terms of IGD and Hypervolume under a budget of only 300 evaluations.

### Strengths
The validation is a fortress. They include: (a) direct comparisons to the correct SOTA baselines from two different fields; (b) a thorough ablation study justifying each major component; (c) a real-world, high-dimensional application (geothermal) proving it's not just a benchmark toy; and (d) a high-scale (D=500) "stress test."

The originality lies in identifying "component isolation" as the key problem. The paper's solution is a holistic, end-to-end system where calibrated uncertainty is the unifying currency. This is a far more mature approach than simply bolting on a better surrogate model.

### Weaknesses
The system is incredibly complex, combining a Bayesian NN (with adaptive MC-Dropout and temp scaling), a Deep GP stack (with sparse VI, RFF, and amortized updates), and another online-trained NN for acquisition. This is a "kitchen sink" of advanced techniques, which raises two red flags:Over-engineering: Is all this complexity truly necessary? The ablations say "yes," but it feels brittle.

This system will be a nightmare to reproduce. It must have a vast number of hyperparameters (network architectures, K ranks, $M_ind$ points, buffer size B, etc.). The lack of a hyperparameter sensitivity analysis is a significant omission. This is my primary concern.

I am highly skeptical of Module 3. It's an NN trained online in a regime with a total budget of 300 points. The history buffer will be laughably sparse, especially at the start. It's far more likely to overfit disastrously than to "learn" a meaningful policy. The ablation shows it helps, but is this due to the learning or simply because it's a non-linear function of the rich features it receives? The paper fails to disentangle these two possibilities.

The paper claims overhead is "negligible" and provides a time breakdown in the appendix. However, training Deep GPs is notoriously expensive. The algorithm states "Fit Deep GP surrogates" at each iteration. What does "fit" mean? Is this a full re-training to convergence, or just a few gradient steps? This ambiguity is crucial for assessing the method's practical (wall-clock) usability.

### Questions
Given the system's immense complexity, reproducibility is the single biggest barrier to its impact.(a) Can you commit to releasing a high-quality, documented implementation?(b) More importantly, please detail your hyperparameter tuning strategy. How were the many hyperparams (NN architectures, $M_ind$, $S_max$, $B$, etc.) selected? Were they tuned per-problem? If so, how can this be practical under an "expensive" budget?

My primary skepticism lies with Module 3. To convince me that the "online learning" is the key, could you compare it to a simpler, non-learning baseline that uses the exact same rich feature vector ($feat(x)$)? For example, a static, weighted-sum heuristic based on surrogate mean, epistemic uncertainty, etc. This would isolate the value of the "learning" itself.

What exactly is the training protocol for the Deep GP surrogates at each iteration? Are they trained from a warm-start to convergence, or just for a small, fixed number of epochs?

### Soundness
2

### Presentation
3

### Contribution
2
