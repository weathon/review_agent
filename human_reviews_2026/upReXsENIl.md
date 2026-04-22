# An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 6

## Abstract
When compressing continuous data, some loss of information is inevitable, and this incurred a distortion upon reconstruction. The Rate–Distortion (RD) function characterizes the minimum achievable rate for a code whose decoding permits a specified amount of distortion. We exploit the connection between rate-distortion theory and entropic optimal transport to propose a novel stochastic-control formulation for the former, and use a classic result dating back to Schrodinger to show that the tradeoff between rate and mean squared error distortion is equivalent to a tradeoff between control energy and the differential entropy of the terminal state, whose probability law defines the reconstruction distribution. For a special class of sources, we show that the optimal control law and the corresponding trajectory in the space of probability measures are obtained by solving a backward heat equation. In more general settings, our approach yields a numerical method that estimates the RD function using diffusion processes with a constant diffusion coefficient. We demonstrate the effectiveness of our method through several examples.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a method to compute the _rate-distortion function_ $R(D)$, which is the smallest information rate needed to achieve a given level of distortion $D$. The approach focuses on the case where the data source is continuous, and the average distortion is quadratic, $D(X, \hat{X})=\tfrac{1}{2} \mathbb{E}[\| X - \hat{X} \|^2]$. Then, the problem reduces to an entropic optimal transport problem, which, in the low distortion regime, can be mapped to a Schrödinger bridge problem. Through this chain of reasoning the task of determining $R(D)$ is converted to a free energy minimization problem, which the authors refer to as _terminal-entropy stochastic control_, wherein the optimal rate corresponds to a balance between control energy and terminal state uncertainty. This optimum can be determined by parameterizing the control with a neural network and scanning through the free-energy landscape, which R2D2 algorithm presented in the paper. Experiments with some mixture distributions and the CIFAR-10 dataset demonstrate the algorithm at work.

### Strengths
1. The paper leverages both established literature and new developments in stochastic control theory and entropic optimal transport to address their central problem.
2. The proposed solution is elegant, and the argument is generally well-presented.

### Weaknesses
1. The authors state that their approach is restricted to quadratic distortion, and the low-distortion regime. Why is this corner interesting from a practical standpoint? Or was it chosen because it makes the problem tractable? The paper would benefit from motivating this choice early on.

2. In its present state, the paper reads like a stochastic control paper which shows that the rate-distortion function is the minimum of a free energy functional. The numerical solution to the latter involves a neural network. Beyond this, the paper makes no wider connection to machine learning in general. See my second question below.

3. Per my understanding, the method does not scale well to higher dimensions. The main bottleneck is the estimation of $H(X_1)$, as explained in appendix B of the paper. If this is an actual limitation it should be highlighted earlier in the paper. See my third question below.

4. Not really a weakness, but the derivations in Sec. 3.2 appears to be mostly a special case of the one presented in [1]. Include this reference.

Minor comment about formatting: The captions for the subfigures in Figure 1 could be spaced apart more for readability.
Minor comment about references: I encourage you to cite [4], which is an English translation of Schrodinger's paper 'On the reversal of the laws of nature.'

### Questions
1. The core arguments of the paper bear a strong resemblance to the _adjoint-matching_ paper from last year [2]. Briefly, given samples of a distribution $p_{\rm base}$ that paper addresses the problem of sampling from the tilted distribution $e^{r(x)} p_{\rm base}$. It seems to me that this is also a form of distortion. While $r(x)$ is fixed in their case, making it free may allow their argument to be extended to some class of distortions besides the one addressed in your paper. Could this be an interesting direction for follow-up work?

2. The terminal entropy stochastic control picture in Sec. 5 looks a lot like max-entropy RL, which you mention in the introduction. Both problems extremize an expected cost (or distortion) while regularizing by an entropy. Does this speak to a deeper connection between compression and learning? Is it correct to say that _the optimal policy is the most efficient compressor of the admissible solutions at a fixed cost_? Sharpening such connections could make the paper much more compelling.

3. On the difficulty of determining $H(X_1)$ in high-dimensions, if you are simulating the trajectories for different $u_\theta$ in Algorithm 1, is it possible to use the samples at $t=1$ to compute that entropy, say with the approach from [3] you cited?

4. Around line 207, what does it mean when you write $W_0^\epsilon \sim \mathbb{P}_0$? If $\mathbb{P}_0$ denotes the source distribution, is $W_0^\epsilon$ the initial state under a small diffusion of it?

Overall I think this is a good paper that can be further improved with some minor changes. I would also appreciate if the authors can address my comments/questions.

[1] Pavon, M., “Stochastic control and nonequilibrium thermodynamical systems,” Applied Mathematics & Optimization, vol. 19, pp. 187-202, Jan. 1989.

[2] Domingo-Enrich, C., Drozdzal, M., Karrer, B. & Chen, R. T. Q. "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control," ICLR 2025.

[3] Franzese et al., “MINDE: Mutual Information Neural Diffusion Estimation,” ICLR 2024

[4] Chetrite, Raphaël; Muratore-Ginanneschi, Paolo; Schwieger, Kay. “E. Schrödinger’s 1931 paper ‘On the Reversal of the Laws of Nature’ [‘Über die Umkehrung der Naturgesetze’, Sitzungsberichte der preussischen Akademie der Wissenschaften, physikalisch mathematische Klasse, 8 Nº 9 144-153].” arXiv preprint arXiv:2105.12617 (2021).

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper exploits the connection between rate-distortion (RD) and entropic optimal transport to propose a novel stochastic control formulation, and use a classic result dating back to Schrodinger to show that the tradeoff between rate and mean squared error distortion is equivalent to a tradeoff between control energy and the differential entropy of the terminal state, whose probability law yields the reconstruction distribution. For a special class of sources, they show that the optimal control law and trajectory in the space of probability measures are given by solving a backward heat equation. In the more general case, their approach gives rise to a numerical solution method, estimating the RD function using diffusion processes with a constant diffusion coefficient.

### Strengths
They present a novel stochastic control formulation that is regularized by terminal uncertainty, and show that this formulation is equivalent to the RD problem. 
They characterize the optimal solution under some regularity conditions.
Found a closed-form solution for the reconstruction distribution of a Gaussian-mixture source.
Proposed a novel neural method for estimating the RD function and the reconstruction distributions, using a simple diffusion model.

### Weaknesses
In some figures, the font size look weird. This is in the Appendix.

### Questions
Section 3.3 seems to focus on mixtures. Will there be any non-mixture examples?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors consider the problem of probabilistic data compression while keeping the distortion rate at an acceptable level. They look at this problem through the lens of entropy-regularized stochastic optimal control and suggest the Terminal-Entropy Control (TEC) method aiming at minimizing the loss function with respect to the admissible control $u(x, t)$ and the terminal distribution $\mathbb P_1$. The authors elaborate on connections of the TEC formulation with the approach based on entropic optimal transport (Theorem 3.1). They also derive an explicit expression for the optimal control function $u^\star(x, t)$ (Theorem 3.2). Finally, the authors illustrate the performance of their procedure with numerical experiments on artificial and real-world data.

### Strengths
According to the numerical experiments, the method outperforms its competitors.

### Weaknesses
1. I do not see a reason why numerical experiments on one-dimensional Gaussian data and CIFAR-10 were not attached as a supplement. This raises concerns about reproducibility of the results reported in Section 5.

2. It seems that the authors missed a very relevant paper [Dai Pra, 1991], where the author studied the problem of stochastic optimal control. His findings were recently applied to generative modelling (see, for instance, [Rapakoulias et al., 2023] and [Puchkin et al., 2025]). I suppose that the proof of Theorem 3.2 in the present submission can be simplified if the authors took into account Theorem 3.2 from [Dai Pra, 1991], where the author derived an explicit expression of the optimal control function $u^\star(x, t)$ for a fixed $\mathbb P_1$ through Schrodinger potentials (see, e.g., [Korotin et al., 2024] for the definition). I would be grateful if the authors could elaborate on this point.

3. In the suggested algorithm, one has to optimize both the terminal distribution $\mathbb P_1$ and the control $u(x, t)$. However, for any fixed $\mathbb P_1$ the form of the optimal control $u^\star(x, t)$ is known (see Theorem 3.2 in [Dai Pra, 1991]). Moreover, Dai Pra [1991] derives the corresponding value of $(2\varepsilon)^{-1} \mathbb E \int_0^1 \|u^*(x, t)\|^2 \, dt$. Hence, the loss can be substantially simplified.

In view of the weaknesses 2 and 3, I would not recommend the paper for acceptance in its present form. I think that it will benefit from a revision, if the authors take into account the results of Dai Pra.


**Minor remark**

On page 4 the authors write: ``Recent developments (Gushchin et al., 2022) has drawn an equivalency (up to an additive constant, depends on $\mathbb P_0$ , $\mathbb P_1$ , $\varepsilon$) between SB and EOT, where the latter can be optimized via a game-theoretic formulation.'' The connection between SB and EOT was known far before (Gushchin et al., 2022). In particular, it was mentioned in the survey of Leonard (2013).

**References**

[Dai Pra, 1991] Paolo Dai Pra. A stochastic control approach to reciprocal diffusion processes. Applied Mathematics
and Optimization, 23(1):313–329, 1991.

[Korotin et al., 2024] A. Korotin, N. Gushchin, and E. Burnaev. Light Schrodinger bridge. In The Twelfth
International Conference on Learning Representations, 2024.

[Puchkin et al., 2025] N. Puchkin, I. Pustovalov, Y. Sapronov, D. Suchkov, A. Naumov, and D. Belomestny. Sample complexity of Schrodinger potential estimation. Preprint. ArXiv:2506.03043, 2025.

[Rapakoulias et al., 2024] G. Rapakoulias, A. R. Pedram, and P. Tsiotras. Go With the Flow: Fast Diffusion for Gaussian Mixture Models. Preprint. ArXiv:2412.09059v3, 2024.

### Questions
1. Can you simplify the proof of Theorem 3.2 in the present submission using Theorem 3.2 from [Dai Pra, 1991]?

2. Can you simplify the loss (12) using Theorem 3.2 from [Dai Pra, 1991]? Is it possible to exclude $u$ from the optimization problem?

### Soundness
2

### Presentation
2

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
This paper introduces Terminal-Entropy Control (TEC), a novel approach to stochastic control that establishes a link between rate–distortion (RD) and entropic optimal transport (EOT) for continuous data sources under mean squared error (MSE) distortion. The authors demonstrate that the classical RD tradeoff between rate and distortion is equivalent to a tradeoff between control energy and the terminal-state differential entropy, and they characterize the optimal solution to TEC under certain regularity conditions. The authors also propose R2D2, a novel neural method for estimating the RD function and the reconstruction distributions using a simple diffusion model. Theory yields closed-form reconstruction distributions for mixtures; experiments validate on Gaussian, Gaussian mixtures, and small CIFAR-10 patches.

### Strengths
- Originality:  The paper introduces a novel dynamic formulation of the rate–distortion problem through the Terminal Entropy Control framework, which reinterprets RD optimization as a stochastic control process.
- Theoretical depth: The work provides substantial mathematical analysis and formal proofs, including clear optimality conditions via the Backward Heat Equation (BHE) and score-based drift characterization.
- Practical method (R2D2): Straightforward training; works from samples; empirically outperforms NERD/WGD on 1D Gaussian and matches analytic mixture results; scales to small image patches.

### Weaknesses
- Limited comparison: Experiments benchmark primarily against NERD and WGD only on the 1D Gaussian case. There is no comparison on gaussian mixtures or CIFAR-10, nor a discussion of why prior estimators cannot be adapted to these settings. Including why R2D2 generalizes better would enhance empirical credibility. 
- Clarity and positioning: 
    - Although the theoretical connection between RD, OT, and Schrödinger bridges is elegant, the manuscript could more explicitly contrast TEC with prior entropic OT formulations (e.g. WGD) and clarify where its generality or computational benefits concretely exceed them.
    - While the paper claims a “closed-form solution” for Gaussian mixtures, the actual rate–distortion values R(D) are still obtained via Monte Carlo or neural estimation. Clarifying this distinction would strengthen the paper’s theoretical claims.

### Questions
1. Relation to prior work (WGD / NERD):
Could the authors clarify why comparisons to WGD and NERD are only held for the 1D Gaussian case?
Is there a theoretical or computational reason these methods cannot handle Gaussian mixtures or CIFAR-10 patches?
2. Stability:
The backward heat equation underlying TEC is known to be ill-posed and potentially unstable. Could the authors elaborate on the numerical stability of training, especially for empirical datasets? In particular, how sensitive is R2D2 to the choice of entropy estimator (negentropy vs. KNIFE) in high-dimensional settings, and were any regularizations used to maintain stability? How strong or restrictive are the regularity constraints in practice—for instance, when $p_0$ represents real, high-dimensional data such as CIFAR-10 patches rather than smooth analytical densities?
3. Experimental setup:
Why are experiments performed on grayscale CIFAR-10 patches instead of full RGB images?

### Soundness
3

### Presentation
3

### Contribution
3
