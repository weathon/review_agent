# Shuffling the Data, Extrapolating the Step: Sharper Bias In Constant Step-Size SGD

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
From adversarial robustness to multi-agent learning, many machine learning tasks can be cast as finite-sum min–max optimization  or, more generally, as variational inequality problems (VIPs). Owing to their simplicity and scalability, stochastic gradient methods with constant step size are widely used, despite the fact that they converge only up to a constant term. Among the many heuristics adopted in practice, two classical techniques have recently attracted attention to mitigate this issue: *Random Reshuffling* of data and *Richardson–Romberg extrapolation* across iterates. Random Reshuffling sharpens the mean-squared error (MSE) of the estimated solution, while Richardson-Romberg extrapolation acts orthogonally, providing a second order reduction in its bias. In this work, we show that their composition is strictly better than both, not only maintaining the enhanced MSE guarantees but also yielding an even greater cubic refinement in the bias.
To the best of our knowledge, our work provides the first theoretical guarantees for such a synergy in structured non-monotone VIPs. Our analysis proceeds in two steps: (i)
we smooth the discrete noise induced by reshuffling
and leverage tools from continuous-state Markov chain theory to establish a novel law of large numbers and a central limit theorem for its iterates; and (ii) we employ spectral tensor techniques to prove that extrapolation 
debiases and sharpens the asymptotic behavior even under the biased gradient oracle induced by reshuffling. Finally, extensive experiments validate our theory, consistently demonstrating substantial speedups in practice.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies constant–stepsize stochastic methods for finite-sum VIPs and shows that composing random reshuffling (RR1) with Richardson–Romberg extrapolation (RR2) cancels all lower-order bias terms, yielding $O(\gamma^3)$ asymptotic bias for quasi-strongly monotone smooth VIPs. The analysis blends an epoch-level Markov chain view of reshuffling with a refined bias expansion and a spectral/tensor argument for the extrapolation step. Experiments on saddle-point games support the theory. I am satisfied with this manuscript and believe it will make a valuable contribution to the community.

### Strengths
- Modeling one reshuffled pass as a time-homogeneous Markov kernel (with Harris recurrence and LLN/CLT) gives clean asymptotic control of per-epoch iterates, a technically elegant way to handle the permutation-induced bias. This framing clarifies why RR1 can sharpen bias/MSE and provides a principled scaffold for the outer-loop extrapolation analysis. 
- The drift/minorization arguments (via Lyapunov–Foster) yield uniqueness of an invariant distribution and geometric convergence, which in turn legitimizes using stationary expansions for bias cancellation in the RR2 step. The explicit LLN/CLT for scalar observables further explains the empirical stability of epoch-averaged statistics. 
- The paper isolates that RR1 alone yields a linear-plus-cubic bias expansion and then shows RR2 cancels the linear term, leaving $O(\gamma^3)$. This surpasses known rates for either heuristic in isolation and is obtained under quasi-strong monotonicity. 
- The debiasing proof uses a spectral view of the full-pass operator (akin to multi-step extragradient) to bound eigenstructure under reshuffling noise, enabling a refined Taylor-type expansion where all sub-cubic terms vanish for the composed algorithm. 
- Algorithm 1 aligns with practical pipelines (RR1 inside, RR2 outside), and the plots consistently show the RR2⊕RR1 curve settling to a smaller bias neighborhood than baselines across condition numbers. This match between theory and experiment strengthens the paper’s practical message.

### Weaknesses
* **[Major] Scope of Assumption 2.2.** The λ-weak μ-quasi strong monotonicity condition, while weaker than strong monotonicity, still excludes many ML-relevant VI problems, e.g., GANs, adversarial training, and multi-agent RL (typically modeled as nonconvex–nonconcave games). However, these problems are mentioned in Introduction section as VIPs. To avoid misleading readers, I recommend softening the tone in the Introduction and adding a dedicated **Limitations** section that clearly delineates where the guarantees apply/do not apply.
  - Jin, Chi, Praneeth Netrapalli, and Michael Jordan. "What is local optimality in nonconvex-nonconcave minimax optimization?." International conference on machine learning. PMLR, 2020. 
  - Han, Andi, et al. "Nonconvex-nonconcave min-max optimization on Riemannian manifolds." Transactions on Machine Learning Research (2023). 
  - Kim, Beomsu, and Junghoon Seo. "Semi-Implicit Hybrid Gradient Methods with Application to Adversarial Robustness." International Conference on Artificial Intelligence and Statistics. PMLR, 2022. 
  - Bukharin, Alexander, et al. "Robust multi-agent reinforcement learning via adversarial regularization: Theoretical foundation and stable algorithms." Advances in neural information processing systems 36 (2023): 68121-68133.

* **[Major] Wall-clock comparisons missing.** Figures 1 and 3 plot error vs. epochs, but SGDA–RR2⊕RR1 likely incurs higher per-iteration computation than SGDA/SGDA-RR1/SGDA-RR2. To substantiate practical gains, please add plots against **wall-clock time** (or FLOPs), ideally with breakdowns for inner/outer steps, so readers can judge time-to-accuracy fairly. 
* **[Minor] Baseline coverage.** Given Loizou et al. (2021) and related work, two additional methods seem relevant and it would be good for them to be included as baselines:
  - [SCO] stochastic consensus optimization (SCO): Mescheder, Lars, Sebastian Nowozin, and Andreas Geiger. "The numerics of gans." Advances in neural information processing systems 30 (2017). 
  - [SHGD] stochastic Hamiltonian gradient descent: Loizou, Nicolas, et al. "Stochastic hamiltonian gradient methods for smooth games." International Conference on Machine Learning. PMLR, 2020.
  - Loizou, Nicolas, et al. "Stochastic gradient descent-ascent and consensus optimization for smooth games: Convergence analysis under expected co-coercivity." Advances in Neural Information Processing Systems 34 (2021): 19095-19108.
  A short ablation with these would contextualize the gains of RR2⊕RR1.
* Line 1587: (C^{\circ}) appears undefined in context. It likely should be (C^{c}) (the complement)

### Questions
1. **About (c_2) (Line 1494).** You state $(c_2 \in (0,1))$. I see (c_2) defined around Line 1475, but I could not follow how the admissible range is derived from that definition. Could you clarify the step that enforces (c_2\in (0,1))?
2. **Practical problem classes/benchmarks.** Beyond the synthetic/quadratic games, are there data-level benchmarks that exactly satisfy your assumptions? Even if the assumptions are only approximately met, are there realistic tasks (or public benchmarks) where SGDA–RR2⊕RR1 shows clear empirical gains over baselines?

### What would be change my opinion?
At the very least, if the revision includes a well-developed Limitations section or any equavalent clarification, I would be inclined to vote for acceptance of this paper. If additional comparison baselines are included or the method’s effectiveness is demonstrated on other ML problems/benchmarks, I will argue even more strongly for the strength of this paper’s contributions.

### Soundness
3

### Presentation
3

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
The article explores random permutations and Richardson–Romberg extrapolation for constant step-size SGD/SGDA in variational inequality problems (VIPs). It states that the combination of these two techniques reduces the bias order of the stationary distribution to $O(\gamma^3)$ for quasi-strongly monotone operators. The authors develop a detailed theoretical framework using the analysis of Markov chains at the epoch level, Foster–Lyapunov drift arguments, and tensor–spectral expansions, and confirm their conclusions empirically on synthetic quadratic problems.

### Strengths
1. **Clear and well-motivated question.**
It is important to understand how the combination of random reshuffling with Richardson–Romberg extrapolation affects bias. This is a theoretically complex issue.

2. **Technically sophisticated analysis.**
The paper presents a nontrivial combination of tools $-$ epoch-level Markov modeling, spectral tensor expansions, and moment control. The analytical pipeline is rigorous and, to my knowledge, novel for this combination of RR and extrapolation.

3. **Theoretical results.**
Lemma 3.5 and Theorem 3.6 provide explicit asymptotic bias rates up to third-order terms in $\gamma$, which is a novel rate.

### Weaknesses
1. **Comparison to work [1] and convergence rate concerns.**
While the paper claims an $O(\gamma^3)$ bias improvement, it is not immediately clear that this is conceptually distinct from previous works, e.g. Stochastic Extra-Gradient with Random Reshuffling (SEG-RR) by Emmanouilidis et al. [1].
Their work already provides higher-order dependence on $\gamma$ in convergence bounds for affine and strongly monotone VIPs. The difference lies in the metric (stationary bias vs. convergence rate) and in the algorithmic structure (plain SGD + extrapolation vs. SEG). The convergence estimate itself also raises a concern. In the theoretical analysis of convergence, the factor $\gamma^2$ appears in the bias term instead of the classical $\gamma$, but an additional factor $\frac{L}{\mu}$ also emerges. This is a problem, since such an extra factor indicates a deterioration of the bound, especially in real-world problems where $\frac{L}{\mu}$ is large. Moreover, it seems that the improvement in the estimate of bias is achieved precisely due to such a deterioration in the condition number.
2. **Unclear practical relevance of $O(\gamma^3)$.**
The bias order is $O(\gamma^3)$; however, the allowed stepsize $\gamma_{\text{max}}$ vary depending on task parameters $n, L_{\text{max}}, \mu$. It is possible that the “bias improvement” manifests only at unrealistically small $\gamma$. The paper should provide numerical examples of typical acceptable values of $\gamma$ to clarify whether this is only a theoretical improvement or a practically relevant regime.

3. **Dependence on Gaussian pre-processing.**
The algorithm’s PREPROCESS step introduces Gaussian smoothing, but the paper does not convincingly argue for its necessity in practice.

4. **Limited empirical validation.**
Experiments are purely synthetic (quadratic min–max tasks). There are no tests on realistic ML or game-theoretic problems (e.g., GANs or robust regression).
This makes it unclear whether the $O(\gamma^3)$ bias scaling provides any tangible benefit beyond theoretical neatness.

### Questions
See my weeknesses. I would be grateful if the authors would resolve my doubts about the additional factor $\frac{L}{\mu}$ in the estimate, and about the small values of stepsize. Can the authors provide an experimental confirmation of the revealed effect on realistic ML tasks?

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
This work studies variant of stochastic gradient methods with fixed step-size for weak quasi-strongly monotone variational inequality problems, which generalizes optimization and min-max problems, showing that combining Random Reshuffling with Richardson–Romberg extrapolation removes the usual bias of $\gamma^{3/2}$ (for SGD-RR2), $\gamma^{2}$ (for SGD-RR1) and achieves a cubic-order asymptotic refinement $\gamma^3$. The paper provides the first theoretical analysis of this combination in quasi-strong monotone VIs, using Markov chain and spectral tensor tools. Experiments confirm the theoretical predictions, demonstrating significant empirical speedups.

### Strengths
Overall, paper is well written, have strong theoretical contribution which can be applied in practice. 
1. Authors proposed to use both RR1 and RR2 simultaneously to reduce the bias of SGD for solving finite-sum VIs.
2. The main result in Theorem 3.6 provides tightens the neighborhood of convergence from previously known $\gamma^2$ to $\gamma^3$.
3. The proof technique seems to be new and not trivial. 
4. Their experimental results support the theory.

### Weaknesses
I did not find any major weakness in the paper and list minor corrections below:

1. Please define PreProcess. It is possible to guess from the next paragraph that it is a Gaussian perturbation, but it would be nice to define it similarly to StochOracle in line 137.

2. The assumptions on the operator of the VI problem are relatively strong. In particular, all $F_i$ need to be Lipschitz. Would the analysis be more involved under the assumption of $F$ being Lipschitz? Would it be possible to relax the quasi-strong monotonicity assumption?

3. I think Theorem 3.3 (lines 346–352) should be explained in more detail, especially the functions $l(x)$ and $L_l(x)$ should be defined for easier readability.

4. I believe it would also be interesting to see plots for a single run of SGD, SGD-RR1, SGD-RR2, and SGD-RR1+RR2, since this is more common in practice. Do you observe that SGD-RR1+RR2 outperforms vanilla, RR1, and RR2 for a single run?

### Questions
Please see the weaknesses section. Additionally:
1. Regarding Thm 3.6, the results are a little bit counterintuitive. Under quasi-strong monotonicity, one can expect a linear rate to a neighborhood of the solution. It is also known that in VIs, average iterates usually converge faster (at least in the monotone deterministic case) than the last iterate, while in the second result of Thm 3.6 the rate for the average iterate is sublinear. Can you please elaborate on this?
 2. The only assumption you make is on the existence of the solution. Does Theorem 3.6 hold for all solutions $x* \in X*$ , or only for the projection of $ x_k $ onto $ X^* $?

### Soundness
3

### Presentation
3

### Contribution
3
