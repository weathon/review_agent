# Random Neural Network Expressivity for Non-Linear Partial Differential Equations

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Neural networks with randomly generated hidden weights (RaNNs) have been extensively studied, both as a standalone learning method and as an initialization for fully trainable deep learning methods. In this work, we study RaNN expressivity for learning solutions to non-linear partial differential equations (PDEs). 
To achieve this, we derive approximation error bounds for time-dependent Sobolev functions and obtain a dimension-free approximation rate $\frac{1}{2}$. 
Our results  imply that RaNNs are capable of efficiently approximating solutions to complex non-linear PDEs. When applied to Physics-Informed Neural Networks (PINNs), our bounds imply that with high probability, the physics-informed training error converges to $0$ with convergence rate free from the curse of dimensionality. Our theoretical analysis is supported by numerical experiments on two benchmark PDEs. These simulations validate the obtained convergence rate.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the expressivity and approximation capabilities of Random Neural Networks (RaNNs) for solving non-linear, time-dependent partial differential equations (PDEs). The authors aim to provide theoretical guarantees for RaNNs in the context of Physics-Informed Neural Networks (PINNs), showing that such models with randomly generated weights can achieve dimension-independent approximation rates under Sobolev norms.

### Strengths
- The paper establishes a theoretical result (Theorem 1) showing that RaNNs can approximate time-dependent Sobolev functions at a rate of $N^{-1/2}$, where $N$ is the number of random neurons.
Importantly, this bound is independent of the spatial dimension, suggesting that RaNNs can mitigate the curse of dimensionality in high-dimensional PDE problems.
- De Ryck et al. (2025) required strong smoothness assumptions, $ s \geq (d+9)/2$, to estimate $H^1$ and $H^2$ errors. This paper relaxes these regularity requirements substantially.
- Previous RaNN approximation results (including De Ryck et al., 2025; Neufeld & Schmocker, 2023) mainly addressed static function approximation, either spatial functions or solutions at a fixed time. This paper extends the theory to time-dependent Sobolev spaces $H^p_t H^q_x$, which are crucial for evolutionary PDEs (e.g., Navier–Stokes, Porous Medium).

### Weaknesses
- The authors only provide results for the Porous Medium Equation (PME), while the second application, the Compressible Navier–Stokes equation, remains purely theoretical.
- The networks used (single hidden layer PINNs with 2500, 5000, and 7500 nodes) are not common in the PINN literature, where deeper and narrower architectures are typically preferred for complex PDEs.
- The paper suffers from presentation issues. For instance, around line 471, “Figure 1” should likely refer to Table 1, and in Figures 1–2, the blue and orange colors are never explained. Moreover, $N$ is used in the introduction section, but is only formally defined in Section 2, which makes it difficult for readers to follow. 
- The paper ends abruptly after the experimental section without a proper Conclusion or Discussion. Moreover, there is no reflection on the limitations of the current analysis.

### Questions
- What is the difference between RaNN and ELM proposed by Huang et al. (2006)?
- What is the definition of convergence rate (in line 20)?
- In line 183, what is $\phi(x)$?
- Does equation (14) mean $u_N$ is an unbiased estimator of $u$? or just $\mathbb{E}(u_N) = u$?
- If you mean $\mathbb{E}(u_N) = u$ by "unbiased", does it hold for $\forall N\geq 1$?
- Theorem 1 assumes $t-$like distribution. However, you sampled the weights from the Gaussian distribution in the experiments. Is there a reason for this choice?
- It seems the minimum regularity in Theorem 1 is quite restrictive ($s_1>3/2, s_2>(d+2)/2$). Could you provide the benchmark problems satisfying these conditions?
- Is it possible to check whether the condition in line 345 is satisfied?
- Is it true that one can find such a sequence as in line 349? If so, how?
- The weights in the experiment are sampled from $\mathcal{N}(0, 100I_{d+1})$. What makes the covariance so high compared to the conventional initialization scheme? How did you choose the variance $100I_{d+1}$? How did you initialize the PINNs?
- The choice of PINNs with a single hidden layer and 2500, 5000, 7500 nodes is unrealistic. Could you replace them with typical architectures with a similar number of parameters? For instance, d-32-32-32-1 has about 2000 trainable parameters.
- In Table 1, could you provide the results for RaNN without Fourier features?
- Check the typos (e.g., line 471 Figure 1 -> Table 1).
- Could you provide detailed training settings (e.g., optimizer, learning rate)
- Could you provide more experimental results that support the theoretical claims?
- This paper is under review as a conference paper at ICLR 2026. It seems the paper is written with ICLR 2025 format.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
For random neural networks (RaNNs), the authors derive approximation error bounds for time-dependent Sobolev functions and obtain a dimension-free approximation rate 1/2, inidicating that RaNNs are capable of efficiently approximating solutions to complex nonlinear PDEs. When applied to Physics-Informed Neural Networks (PINNs), the bounds imply that with high probability, the PINN training error converges to 0 with convergence rate free from the curse of dimensionality. Numerical simulations show that RaNN-based PINNs achieve a similar accuracy to classical PINNs, while requiring significantly less training time, supporting the theoretical findings.

### Strengths
* (Relevance) A precise theoretical understanding of RaNN approximation capabilities is crucial because the reduced number of trainable parameters of RaNNs compared to fully trainable models and  a simpler training phase with reduced computational cost are sometimes obtained at the expense of lower expressivity. (Lines 72-75)
* (Technical contribution & Novelty) The derived bounds are tailored to time-dependent, non-linear PDE problems, which starkly contrast the paper from previous results, which require strict information on the solution or are applicable only for approximating PDE solutions at a fixed point in time. (Lines 84-91)
* (Technical cotribution) The analysis yields residual-loss bounds for the PME and 1D compressible Navier–Stokes, clarifying when randomized features can train efficiently without the curse of dimensionality.
* (Reproducibility) The code is provided.

### Weaknesses
* Discussion on the comparison with other RaNN-based PINNs: \[Nelsen & Sturat 2021; Gonon 2023; Jacquier & Zuric2023；Neufeld+ 2025\] is recommended to highlight the paper's contribution.
* The code is provided, but adding a brief readme file would be helpful.
* Results require smooth solutions ($s_1>3/2, s_2>(d+2)/2$) and boundedness of certain network norms

  for the high-probability residual bounds. Constants like $C\_\\Omega$, $L\_\\psi$, $C\_\\pi$ are intricate and not empirically calibrated. The rate is dimension-independent, but constants could possibly depend on $d$ via extensions/embeddings, to my understanding.
  * Could the authors quantify or upper-bound key constants (e.g., $C\_\\Omega, L\_\\psi, C\_\\pi$) for representative domains and p, q so users can anticipate practical sample complexity?
* Including scaling curves of error/residual and runtime vs. N would be benefricial to empirically support the theoretical convergence rate.

### Questions
- Do you foresee extensions to non-smooth or stiff solutions (e.g., shocks), or stochastic PDEs, perhaps via different norms?

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
1

### Summary
This paper studies RaNN expressivity for learning non-linear PDEs solutions. It derives approximation error bounds for time-dependent Sobolev functions and obtain a dimension-free approximation rate 0.5. The experiments in solving Porous Medium Equations (PME) and Compressible Navier-Stokes Equations shows similar accuracy to the PINN method, while requiring less training time.

### Strengths
The proposed method achieves similar performance with less training time and fewer number of parameters compared with PINN.

### Weaknesses
This paper lacks ablation test and thus it is difficult to know the robustness and the scalability. For example, compare with PINN with similar number of trainable parameters? If the neural network is scaled to similar size, does the method still require less training time? How does the training time and number of trainable parameters affect the final results? Does it match the theory?

### Questions
The proposed method seems to be efficient in model size. Does scaling it further up improve the results?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study the expressivity of Random Neural Networks (RaNNs), deriving RaNN approximation error bounds for time-dependent Sobolev functions. They show that RaNNs can approximate solutions to complex nonlinear PDEs with a dimension-free approximation rate (the generic Monte Carlo Rate, $N^{-1/2}$, with N being the number of neurons). The integral representation of the solution using a Ridgelet transform is used to derive the error bounds. The authors improve results in Theorem 3.9 from De Ryck et al. (2025) and extend results to time-dependent norms. The authors also demonstrate, in a computational experiment, that RaNNs are faster to train for similar accuracy compared to PINNs on Porous Medium Flow (PME) example.

### Strengths
1) The derivation of approximation error bounds including the dimension-free approximation rate for time-dependent Sobolev functions is a significant theoretical contribution that extends previous works. For random feature methods, the rate $N^{-1/2}$ is very common, but the special type of functions addressed here makes the result relevant for the PDE setting.
2) Considering time dependence in the approximation is important, as most related theory concerns (linear) static PDE.
3) The main contribution of the paper, the proof of theorem 1, is featured prominently and explained well. Using ridgelet transforms is an interesting and useful idea in this field.

### Weaknesses
1) It is not clear how time-dependence is special/important to the theoretical results. I might have missed it when reading the proofs, but the dependence on time is not different from any other spatial dimension (also when looking at the model in eq. 1, time is treated like this). The same can be said for "nonlinearity" of the PDEs involved, as the theoretical results are related to general Sobolev functions, not the specific PDEs that have them as solution.

2) The write-up seems partially incomplete: (1) Conclusion/discssion section with future work is missing, (2) limitations are not addressed explicitly.

3) A single benchmark is not enough to demonstrate the theoretical results, especially since many computational results for RaNNs exist already (see list in "questions"). The experimental section is quite weak compared to the typical benchmarking standards of the ICLR community. More examples (e.g., d > 3 for PME or compressible Navier-Stokes equations) would strengthen the claims; also, as the main result is the independence of the approximation rate w.r.t. dimension, a computational experiment with increasing input dimension (e.g. from 2-10) that demonstrates this independence would have been useful.

### Questions
1) In Figures 1 and 2, is the true solution given by the blue or orange line? 

2) Did you try Porous Medium Equation for dimensions beyond three? What happens for higher dimension beyond 3? 

3) Did you try experiments with compressible Navier-Stokes equations? The current setup sets p=0 and mu(rho)=rho, which is extremely simplistic. Solving NS in the turbulent regime presents challenges beyond the curse of dimensionality, more related to the issue of spectral bias of networks. Is the spectral bias also something that random networks can resolve better than the ones trained with gradient descent?

4) The empirical details are not fully clear. Which optimizer was used to compare the results with PINNs? If a first-order optimizer like Adam is used, a comparison with any second-order optimizer (generally considered state-of-the-art for PINNs) would be helpful. E.g., L-BFGS, Adam + L-BFGS, SSBryoden. 

5) RaNN-based PINN performance, including high-dimensional examples, has been demonstrated  extensively in computational experiemnts, which should be cited and discussed. Examples:
 - Sun, Jingbo, Suchuan Dong, and Fei Wang. 2024. “Local Randomized Neural Networks with Discontinuous Galerkin Methods for Partial Differential Equations.” Journal of Computational and Applied Mathematics 445 (August): 115830. https://doi.org/10.1016/j.cam.2024.115830.
 - Datar, Chinmay, Taniya Kapoor, Abhishek Chandra, et al. 2024. “Solving Partial Differential Equations with Sampled Neural Networks.” arXiv:2405.20836. Preprint, arXiv, May 31. http://arxiv.org/abs/2405.20836.
 - Nelsen, Nicholas H., and Andrew M. Stuart. 2021. “The Random Feature Model for Input-Output Maps between Banach Spaces.” SIAM Journal on Scientific Computing 43 (5): A3212–43. https://doi.org/10.1137/20M133957X.
 - Chen, Jingrun, Xurong Chi, Weinan E, and Zhouwang Yang. 2022. “Bridging Traditional and Machine Learning-Based Algorithms for Solving PDEs: The Random Feature Method.” arXiv:2207.13380. Preprint, arXiv, July 27. http://arxiv.org/abs/2207.13380.

6) For general random feature networks in the supervised learning setting, tighter bounds were proven already (albeit not in the Sobolev case), see here:
Rudi, Alessandro, and Lorenzo Rosasco. 2021. “Generalization Properties of Learning with Random Features.” arXiv:1602.04474. Preprint, arXiv, April 15. http://arxiv.org/abs/1602.04474.

Would it be possible to tighten the bounds proven in the present work using their methods?

### Soundness
3

### Presentation
2

### Contribution
2
