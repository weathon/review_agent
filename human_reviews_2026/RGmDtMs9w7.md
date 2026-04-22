# Modeling Training Dynamics and Error Estimates of DNN-based PDE Solvers: A Continuous Framework

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Deep neural network-based PDE solvers have shown remarkable promise for tackling high-dimensional partial differential equations, yet their training dynamics and error behavior are not well understood. 
This paper develops a unified continuous-time framework based on stochastic differential equations to analyze the noisy regularized stochastic gradient descent algorithm when applied to deep PDE solvers. 
Our approach establishes weak error between this algorithm and its continuous approximation, and provides new asymptotic error characterizations via invariant measures. 
Importantly, we overcome the restrictive global Lipschitz continuity loss gradient, making our theory more applicable to practical deep networks. 
Specifically, our study focuses on general second-order elliptic PDEs; however, the proposed framework is not limited to this specific form and can be extended in principle to broader classes of PDEs.
Furthermore, we conduct systematic experiments to reveal how stochasticity affects solution accuracy and the stability domains of optimizers. 
Our results indicate that stochasticity can have varying impacts on the stability of solutions near different local minima; therefore, in practical training, strategies should be dynamically adjusted according to the local optimization landscape to enhance robustness and stability of neural PDE solvers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper claims to propose a continuous-time theoretical framework for neural PDE solvers that removes the need for the global Lipschitz continuity assumption. It introduces a noisy high-order regularization to ensure well-posedness of an SDE corresponding to discrete SGD, proving local weak convergence and providing long-term error estimates for DNN-based PDE solvers. The theoretical contribution is novel, but relies on strong assumptions. Its practical impact is also quite unclear.

### Strengths
- Theoretical novelty: proposes a continuous-time analysis framework for neural PDE solvers without global Lipschitz assumptions.
- Provides long-term error estimates and Laplace-type approximations that may be useful for understanding training dynamics.
- Establishes local weak convergence from discrete SGD to a regularized SDE.

### Weaknesses
While theoretically interesting, the strong assumptions and limited experimental validation reduce the practical impact of the work. The main concerns is listed as following
- Very limited baseline comparisons. Modern optimizers like AdamW or other recent related literatures (for example **Newton Informed Neural Operator for Solving Nonlinear Partial Differential Equations**) are not compared in the experiments. Based on the results in the manuscript, it's difficult to claim better convergence performance.
- The theoretical derivation has too many restrictions which can significantly reduce the applicability of the proposed method.

### Questions
- The framework relies on a $|θ|^{2s}$ (s≥10) term to guarantee well-posedness. Such assumption may not be desirable in many application conditions. 
- Convergence is guaranteed only in bounded regions and depends on the probability of the process staying within that region. Its global convergence in high-dimensional or steep PDE loss landscapes is not addressed properly. 
- Too simple test cases: Only 2D ODEs / low-dimensional PDEs are tested with very small networks (2 layers, width 10). Performance on complex, nonlinear, high-dimensional PDEs or real physical systems is unknown.
- Experiments compare only SGD and GD, without modern optimizers (Adam, AdamW, L-BFGS, etc.), making the practical advantage of the proposed approach unclear.
- Only L2 errors and loss curves are reported. While convergence speed, training time, generalization, and robustness are not evaluated thoroughly.

### Soundness
1

### Presentation
3

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
This paper develops a continuous-time framework based on SDEs to analyze the training dynamics of deep neural network-based PDE solvers, specifically PINNs. The authors introduce a "noisy regularized SGD" algorithm that adds both a high-order regularization term and Gaussian noise to standard SGD. They establish weak convergence between this discrete algorithm and its continuous SDE approximation, removing the restrictive global Lipschitz assumption common in prior work. The paper also characterizes asymptotic error via the SDE's invariant measure using WKB approximation and the Laplace method. Experiments on a simple 1D ODE reveal that stochasticity narrows the stability regime of learning rates and degrades solution accuracy compared to gradient descent, though SGD can outperform GD near sharp minima.

### Strengths
1. The most significant theoretical contribution of this paper is to establish weak convergence results without requiring global Lipschitz continuity of the loss function. As the authors note, "our theory more applicable to practical deep networks" since standard neural networks violate this assumption. The local Lipschitz approach using stopping times (Definition 2) and the decomposition in Theorem 1 is technically sound.
2. The perspective of analyzing long-term error through the invariant measure of the SDE (Proposition 4) and its asymptotic approximation via WKB methods is interesting. The connection between the maximizers of $S_0$ and the expected error in Proposition 5 provides a different lens on understanding SGD behavior beyond standard optimization/generalization decompositions.
3. The experiments in Section 4 provide valuable empirical insights. The observation that SGD has a much narrower stability domain than GD near low-sharpness minima (Figure 1) and that stochasticity degrades accuracy even within the stable regime helps explain why PINNs often struggle with precision. The comparison between two regimes with different sharpness levels is well-motivated.
4. The paper is generally well-written with careful definitions and the proofs appear rigorous.

### Weaknesses
1. The practical applicability of the modified algorithm is a bit limited. The noisy regularized SGD requires an extremely high-order regularization term to ensure theoretical guarantees. While the authors acknowledge this in Remark 4, the gap between theory and practice can be concerning. The experiments in Appendix I show the algorithm works on some problems, but the regularization parameter and noise level appear highly problem-dependent. It's unclear how practitioners should set these hyperparameters or whether the theoretical insights transfer to standard SGD.
2. The uniform moment bounds in Assumption 3 are crucial for Proposition 3 and Theorem 1, yet the authors simply assume this holds without proof or empirical verification. They acknowledge it "remains open in many settings," which significantly weakens the main result. Without this, the weak convergence is only established for processes that stay in $B_R$, and the exit probability bounds don't apply uniformly in $\eta$.
3. The experimental validation focuses exclusively on a simple 1D ODE with width-10 networks where the exact solution can be represented. While this enables precise analysis, it's far from the high-dimensional PDEs that motivate DNN-based solvers. The experiments in Appendix I on 2D problems (Helmholtz, Fisher-KPP, Allen-Cahn) show the algorithm can work but don't validate the theoretical predictions about stochasticity effects or compare against standard methods systematically.
4. The paper promises "actionable guidance for practical training" but the main takeaway is not too surprising and doesn't lead to clear recommendations. The suggestion that "adaptively switching optimizers and step sizes...can be beneficial" is vague and not demonstrated.

### Questions
1. Can the authors provide any theoretical or empirical evidence to support Assuption 3? At minimum, is it possible to verify whether it holds for the experimental settings in Section 4?
2. The WKB approximation assumes $B_0 \in C^2$, but is this regularity guaranteed, given that the loss landscape of neural networks is typically not smooth?
3. In regime 2, SGD outperforms GD even when using a learning rate that causes GD to diverge. Does this contradict the stability analysis?
4. This paper https://proceedings.mlr.press/v235/chen24ad.html proposes a method that the author claims to solve Allen-Cahn equation (and other equations) towards machine precision. Can the authors comment on how the framework proposed in this paper is consistent with the method in https://proceedings.mlr.press/v235/chen24ad.html ? Is the framework not applicable in their setting at all? If not, can a similar framework be developed?
5. Are the theoretical results sensitive to the choice of $s=10$ in the regularization? The proof is for $s\ge10$, but is there an intuition for why?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work contributes to a deeper understanding of the optimization procedure with stochastic gradient descent (SGD) using tools from stochastic differential equations (SDEs) in the case of training feed-forward neural networks for a class of partial differential equations. The estimation error of the iterative SGD algorithm compared to its continuous counterpart is characterized in the weak form. Computational experiments are used to investigate the effect of stochasticity on the stability of the optimization and provide a comparison to classical gradient descent.

### Strengths
The paper makes use of established connection of SGD and SDE via stochastic modified equations (SME) (Li et at. 2017, 2019) and investigates the consequences of modified assumptions. So far this community has not focused much on neural networks for partial differential equations, which is a part of the novelty of this work. For the optimizer setting the authors consider, including the new assumptions, they find that the optimizer weakly converges to its continuous SDE version. An important result is the relaxation of Lipschitz continuity assumption on the network, which was used by previous works.
The experimental results are consistent with existing literature on the dynamics of SGD optimizers. To the best of my knowledge, the results are novel for the considered noisy regularized SGD optimizer.

### Weaknesses
1) The authors position this contribution in the realm of learning methods for general partial differential equations (PDEs), but it seems that the focus is on specific, second-order elliptic PDEs, as stated in section 2.1.. I suggest making this clear at least in the abstract and the title. The main results (theorem 1) heavily rely on this specific PDE type, with assumptions (1) for the coefficient terms.
2) "... a continuous framework" is also too generic for the title (use "continuous-time" instead?); it is also challenging to read about "continuous time" frameworks while the PDEs in question are not time-dependent, yet the introduction and abstract make no distinction about this.
3) The authors claim that "..our results readily extend to more general equations and deeper network architectures", but do not prove any of this, or demonstrate it experimentally. The same seems true for "all results extend directly to the empirical setting" (l155), where it is not clear at all why choosing a finite-size data set (with e.g. only 10 evaluation points, in higher base-space dimensions... etc) would result in "direct" extension from the exact L^2 error.
4) Similarly, in the title and the main text, "dnn-based PDE solvers" are mentioned, but in the setting (e.g. eq. 2), only shallow, two-layer networks are considered. Merely stating that "the results readily extend to deeper architectures" is not enough to use a broad statement in the title about DNNs. Shallow networks behave very differently from deep networks, and have very different properties (see e.g. "Poole, Ben, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. 2016. "Exponential Expressivity in Deep Neural Networks through Transient Chaos." In Advances in Neural Information Processing Systems 29, edited by D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett. Curran Associates, Inc. http://papers.nips.cc/paper/6322-exponential-expressivity-in-deep-neural-networks-through-transient-chaos.pdf".)
5) The computational experiments do not show the performance of SGD on a PDE, but just an ODE. There is no discussion why this is a good test case for behavior on PDEs, beyond "it has a closed form solution".
6) For many PINNs the optimizer starts with Adam iterations, followed by additional iterations with L-BFGS (a quasi-Newton method), this is not acknowledged and should be mentioned either as a limitation or future work, especially as the abstract mentions "adaptively switching optimizers".
7) The manuscript lacks a reflection on the limitations of the current work.

8) Minor remarks:
- The text in the figures is too small
- It seems that the same notation $|\cdot|$ is used both for the $L^2$ vector norm as well as for the determinant. Ideally two different notations would be used (e.g. $\|\cdot\|$ for norm, and normal lines $|$ for determinant, or use $\det(\cdot)$).
- Several times "continuous modeling" is used (even in the title, and l.053, for example), while I presume the authors meant "continuous-time modeling". This must be changed.
- The explanation of regularization of powers 20 is not sufficient (l240). It indeed seems extremely artificial, and it is not clear at this point in the paper why the power 20 is the barrier between unbounded and bounded growth. There should at least be a link to a discussion later in the paper, or the proof where this is clarified.

### Questions
1. Other work (Li et al. 2017, 2019) considers not only SGD but also SGD with momentum and Nesterov's method. Is an extension of your work possible for these variants of SGD? 
2. The loss function for PINNs often includes additional terms (e.g. boundary/initial conditions, additional physical constrains, enforcing symmetry), what kind of impact would these terms have on the SGD dynamics, or what assumptions should be made in regard to them?

### Soundness
1

### Presentation
3

### Contribution
2
