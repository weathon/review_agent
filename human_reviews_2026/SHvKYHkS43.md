# Q-STRONG: Quantum-Statistical Robustness with Noise-Guarded Dynamics for Learning

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
State-of-the-art learners remain fragile under heavy-tailed noise, adversarial perturbations and decoherence. We propose \emph{Q-STRONG}, a quantum–statistical framework for certified robust learning that uses the spectral structure of a learned state representation as a stability signal. Inputs are embedded into a normalized quantum state space, and a task-aligned Hamiltonian induces a low-energy representation whose spectral gap $\Delta_\theta(x)$ quantifies local stability. This gap steers both training and certification: during optimization, robust losses and quantile-based clipping reduce gradient tail effects; at inference, a gap-adaptive randomized smoothing scheme chooses the noise level $\sigma(x)=\kappa \Delta_\theta(x)^{-\beta}$, yielding larger certified $\ell_2$ radii exactly where the representation is stable. We provide non-asymptotic guarantees for quantile-clipped robust SGD, stability-based generalization bounds with improved effective smoothness, and gap-adaptive extensions of randomized-smoothing certificates tied to $\Delta_\theta(x)$. Empirically, Q-STRONG attains a favorable accuracy–robustness frontier on MNIST and CIFAR-10 under label noise and common corruptions, and on synthetic manifolds that stress intrinsic dimension and outliers, while adding modest overhead and thus offers a practical, theoretically grounded route to certified, noise-resilient learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a robustness framework that encodes inputs as unit-norm complex quantum states and uses the Hamiltonian spectral gap as a stability signal for training and certification. It combines robust M-estimation with gradient clipping, and introduces gap-adaptive randomized smoothing for larger certified $\ell_2$ certificates. The paper proves convergence of clipped SGD, tighter generalization for the method, and a certification theorem that inherits the guarantee of Cohen et. al with the gap-dependent noise. The quantum aspect is, to my reading, primarily the state-space formalism.  Experiments on MNIST/CIFAR-10 show improved certificates with similar accuracy.

### Strengths
-- The paper combines robust M-estimation, adaptive gradient clipping, and randomized smoothing and gives proofs of convergence, stability, and certification

-- The idea of linking the randomized-smoothing noise to a quantum spectral gap seems to be an original connection (though I am not an expert)

-- Empirically the method achieves larger certified $\ell_2$ robustness without accuracy loss, supporting the theory claims

### Weaknesses
-- The paper is a bit of a mess in its current sate. For example lines 234 - 241 I think are intended to be a tex algorithm, but the authors messed up the latex. The leading [t] [1] suggests this.

-- On line 125, I think you meant $\Delta_\theta(x) = \lambda_1(x) - \lambda_2(x)$. This should be fixed.

-- I find the paper hard to read. It reads like a bunch of standalone passages. 

-- The experiments are rather limited overall. There are limited comparisons to stronger certified robustness methods.

-- The quantum and robust statistics connection is not clearly spelled out. I believe this needs a dedicated section.

For these reasons, I opt to reject the paper in its current state. Though I am open to changing my score if the authors can correct these points.

### Questions
-- Can you compare to newer certified training methods e.g. AdvSmooth [1] and TRADES [2]?

-- Could the spectral gap be replaced by a regular stability proxy such as Jacobian singular values?

-- In what sense is this framework quantum-statistical if implemented entirely on classical hardware? In general I believe the quantum connection needs to be spelled out more

[1] https://arxiv.org/pdf/1906.04584
[2] https://arxiv.org/pdf/1901.08573

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
3

### Summary
The paper introduces Q-Strong, a framework that combines randomized smoothing, robust M-estimation, and gradient clipping. The authors investigate the convergence and robustness of the framework and evaluate it on the MNIST and CIFAR datasets.

### Strengths
- The proposed framework combines state-of-the-art techniques, and the empirical performances are promising.

- The framework comes with non-asymptotic guarantees

### Weaknesses
- The paper is a combination of existing techniques already employed in the literature. While this is not necessarily a weakness, I miss seeing what the main non-incremental contributions of the paper are compared to the literature. For instance, it is not clear if the theoretical framework is simply a straightforward consequence of existing results.

- In my opinion, the empirical evaluation should be improved. In fact, if the goal of the paper is to propose a method that improves the tradeoff robustness/accuracy, then this should be compared with state-of-the-art techniques and possibly on more datasets.

### Questions
- What are the main technical contributions in the theoretical analysis? 

- Have you compared the proposed tool with state-of-the-art techniques for adversarial training?

### Soundness
3

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
1

### Summary
This paper proposes a quantum–statistical framework designed to enhance robustness against heavy-tailed noise, adversarial perturbations, and intrinsic stochasticity. In particular, the framework, Q-STRONG, is proposed for near-term quantum processors (NISQ devices). The approach integrates three key components: (i) robust M-estimation, (ii) quantile-scheduled gradient clipping (DynClip), and (iii) gap-adaptive randomized smoothing. Inputs are encoded as quantum states, and the spectral gap of a task-aligned Hamiltonian serves as a stability signal to guide adaptive noise injection and certification.

### Strengths
The mathematical analysis is rigorous and technically sound. The non-asymptotic convergence guarantees for clipped SGD under weakly smooth robust objectives, as well as the stability-based generalization bound, are well presented and theoretically meaningful. The integration of the spectral gap as a stability indicator is a novel approach. The connection between quantum representations and robustness is an appealing direction. The derivations for bounded-influence losses (e.g., Huber) and dynamic clipping are particularly interesting and appear consistent with the theoretical claims.

### Weaknesses
The major concern is the empirical validation, which is currently insufficient to support the broad theoretical claims. The experiments are primarily limited to MNIST, with only minimal results reported on other datasets. Although the authors mention experiments on CIFAR-10, I could not find any corresponding results in the paper.

Given the ambitious theoretical framework and the claim of hardware-agnostic applicability (classical or quantum), evaluations on more diverse benchmarks, such as Fashion-MNIST, SVHN, and CIFAR, are essential. Such experiments would help demonstrate the generality of the proposed method and bridge the gap between the theoretical assumptions (e.g., Lipschitz smoothness, bounded spectral gaps) and observed empirical behavior.

### Questions
Refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The submission proposes Q STRONG, a framework that combines three ideas (1) Robust M estimation, (2) Quantile‑scheduled gradient clipping, and. (3) Gap adaptive randomized smoothing at inference. Empirically, the paper reports ablations on MNIST and CIFAR 10 (with label noise and CIFAR-10-C) comparing CE, Huber, DynClip, and Dyn+Smooth, with Dyn+Smooth improving certified radii at modest accuracy cost.

### Strengths
1.	The paper’s idea is novel and is practically motivated. It presents a tidy framework where robust losses (bounded influence), quantile‑based clipping, and randomized smoothing reinforce each other and are all modulated by a single, interpretable quantity (the spectral gap).

2.	It has ablation studies to shows the effectiveness of DynClip and Dyn+Smooth. The direction is promising for practitioners for certified robustness.

### Weaknesses
1.	The construction of the error Hamiltonian Hθ(x) (how it depends on θ and x), the precise procedure to estimate Δ(x), and the statistical concentration of this estimator are not specified with enough operational detail to reproduce results.

2.	CIFAR 10(+C) results are described but not fully shown; the paper would be stronger with complete tables/plots for CIFAR 10 and severity sweeps on CIFAR 10 C, plus variance across seeds. Also it is suggested to show results on more complex tasks.

3.	Comparisons to stronger robustness baselines (e.g., adversarially trained smoothed models, label noise robust methods) are missing.

### Questions
1.	What exact Hamiltonian do you use in experiments? How is it parameterized, how often is Δ(x) estimated, and what is the computational cost relative to forward/backward? Please include an algorithmic box with pseudocode.

2.	The text mentions mean certified radius 0.666 for MNIST, while Tables 1–2 show 0.30–0.41. Which is correct? Also, why do the tables say Digits10 rather than MNIST?

3.	Could you add comparisons to stronger robustness baselines (e.g., adversarially trained smoothed classifiers) and a sensitivity study over the quantile α schedule, k, and beta?

### Soundness
3

### Presentation
2

### Contribution
3
