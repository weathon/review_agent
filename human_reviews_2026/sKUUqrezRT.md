# Redefining Neural Operators in $d+1$ Dimensions

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Neural Operators have emerged as powerful tools for learning mappings between function spaces. Among them, the kernel integral operator has been widely validated on universally approximating various operators. Although many advancements following this definition have developed effective modules to better approximate the kernel function defined on the original domain (with $d$ dimensions, $d=1, 2, 3...$), the unclarified evolving mechanism in the embedding spaces blocks researchers' view to design neural operators that can fully capture the target system evolution.

Drawing on the Schrödingerisation method in quantum simulations of partial differential equations (PDEs), we elucidate the linear evolution mechanism in neural operators. Based on that, we redefine neural operators on a new $d+1$ dimensional domain. Within this framework, we implement a Schrödingerised Kernel Neural Operator (SKNO) aligning better with the $d+1$ dimensional evolution. In experiments, the $d+1$ dimensional evolving designs in our SKNO consistently outperform other baselines across more than ten increasingly challenging benchmarks, ranging from the simple 1D heat equation to the highly nonlinear 3D Rayleigh–Taylor instability. We also validate the resolution-invariance of SKNO on mixing-resolution training and zero-shot super-resolution tasks. In addition, we show the impact of different lifting and recovering operators on the prediction within the redefined NO framework, reflecting the alignment between our model and the underlying $d+1$ dimensional evolution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes extending Neural Operators to an additional auxiliary dimension inspired by the Schrödingerisation method. By modeling operator evolution in this d+1-dimensional space, the authors design the Schrödingerised Kernel Neural Operator (SKNO), which combines spectral convolution and residual connections to capture both global and local dynamics. Although the theoretical foundation applies to linear PDEs, the model empirically performs well on nonlinear systems like Burgers and Navier–Stokes equations through stacked linear and nonlinear blocks. SKNO achieves state-of-the-art accuracy and resolution invariance across multiple PDE benchmarks.

### Strengths
- This paper is clearly structured, with a logical flow from theoretical motivation to model design, implementation, and experiments, making it easy to follow.
- The proposed approach is grounded in a solid theoretical framework based on the Schrödingerisation of linear PDEs, providing a principled explanation for the evolution mechanism within Neural Operators.
- The proposed SKNO achieves superior performance across ten diverse PDE benchmarks, demonstrating excellent accuracy, resolution invariance, and robustness compared to existing models.

### Weaknesses
- The introduction of Schrodingerisation is interesting, but the proposed model itself appears to simply add a one-dimensional input variable during lifting and then combine it with FNO and various existing ideas.
- The reason why the proposed method produces good results is somewhat unclear (See Questions).

### Questions
- Schrodingerisation is a technique for linear PDEs, and it seems that, inspired by this idea, a neural operator that can also be used for nonlinear PDEs was modeled. Is the reason why the proposed method performs well because it adds one dimension to the space to increase its expressiveness? Or is it because Schrodingerisation can be considered to incorporate the physical laws or mathematical structure from Hamiltonian system?

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
4

### Summary
This paper revisits the concept of Neural Operators (NOs) by formulating them explicitly in d + 1 dimensions, where the extra dimension represents temporal or parametric evolution. The authors argue that existing neural operators often neglect this intrinsic spatiotemporal coupling, which limits their ability to capture dynamic PDE behavior. The proposed framework offers a unified and physically grounded perspective that treats space and time consistently, leading to better representational power. Experiments on standard PDE benchmarks demonstrate notable improvements in accuracy over established baselines such as FNO and transformer-based models.

### Strengths
1) The paper provides a conceptually elegant and theoretically motivated redefinition of neural operators, naturally extending them to spatiotemporal systems.

2) It establishes a unified mathematical framework that connects spatial and temporal operator learning under a common perspective.

3) The proposed formulation achieves empirical improvements over strong baselines like FNO and Transformer-based architectures.

### Weaknesses
1) The paper is somewhat dense and difficult to follow in places — it took multiple readings to fully grasp the core formulation. Also it seems written in hurry as conclusion, future work, limitations aren't discussed.

2) Comparisons with newer operator paradigms, such as state-space models (e.g., Mamba Operator), are missing, which would provide a more comprehensive evaluation or consider citing them in related work.

3) The experimental scope is limited to low-dimensional PDEs, leaving scalability to 3D or real-world physics problems unexplored.

4) The computational cost and efficiency trade-offs of the proposed approach are not clearly discussed or quantified.

5) Some implementation details (e.g., architectural variations or training hyperparameters) could be presented more clearly for reproducibility.

### Questions
1) How does the proposed d + 1 formulation handle non-autonomous PDEs or systems with variable coefficients over time? Additionally, what happens if we extend this idea to even higher-dimensional formulations—would the dynamics become more linear in such limits (Koopman Operator), and have you experimented with this?

2) Have the authors evaluated the scalability of the approach on high-dimensional or 3D PDEs, and how does its computational cost grow with problem size?

3) Could the proposed framework be combined with latent-space modeling or physics-informed priors to enhance interpretability and regularization?

4) How does the model perform under zero-shot generalization to unseen temporal regimes or boundary configurations?

5) In Figure 3, it’s unclear why multiple global propagators and only one local propagator were used. Have different configurations been tested, and how do they affect performance?

**Missing References:**

1) Complex Neural Operator: https://arxiv.org/abs/2406.02597

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
5

### Summary
This paper introduces a novel framework for Neural Operators (NOs) by reformulating them in a d+1 dimensional space. The core motivation stems from the "Schrödingerisation" method in quantum physics, which transforms a d-dimensional PDE into a d+1 dimensional equivalent. The authors propose that this higher-dimensional perspective can elucidate and improve the design of NOs. Based on this framework, they develop the Schrödingerised Kernel Neural Operator (SKNO), a model that explicitly operates in this d+1 space. SKNO consists of three stages: a lifting operator to map the input to the higher-dimensional space, an evolution block that propagates information in both the original d dimensions and the new auxiliary dimension, and a recovering operator to project the result back to the original d-dimensional space. The authors conduct extensive experiments across ten PDE benchmarks, claiming that SKNO achieves state-of-the-art performance, superior resolution invariance, and better zero-shot super-resolution capabilities compared to existing NOs like FNO and Transolver.

### Strengths
The paper introduces a creative, physics-inspired viewpoint for designing neural operators, which could stimulate new research directions.

### Weaknesses
**Weaknesses***

1.  **Weak Theoretical Grounding:** The foundational analogy to Schrödingerisation feels more inspirational than formal. The paper lacks a rigorous derivation showing why this `d+1` formulation is a necessary or uniquely optimal way to construct neural operators, weakening the claim of having "redefined" them.
2.  **Unclear Computational Cost:** The paper does not adequately address the increased computational complexity of its approach. The use of a `(d+1)`-dimensional FFT is significantly more expensive than the `d`-dimensional FFT in FNO. A transparent comparison of parameter counts and FLOPs is missing, which is crucial for evaluating whether the reported accuracy gains justify the additional computational burden.
3.  **Incremental Architectural Novelty:** When stripped of its quantum physics motivation, the SKNO architecture can be seen as a multi-channel FNO variant where channels (the auxiliary `p` dimension) interact through an additional learned operator. The building blocks are well-established, making the contribution more of an effective architectural design than a fundamental paradigm shift.
4.  **Limited Justification for Design Choices:** The paper implements the evolution along the auxiliary dimension using a spectral operator. The rationale for this specific choice over other possible mechanisms (e.g., attention, simple MLPs) is not thoroughly explored, leaving it unclear if this is a critical component of the framework's success.

5. **Some baselines' performance[1] is great, but authors does't compared**

[1]. Turb-L1: Achieving Long-term Turbulence Tracing By Tackling Spectral Bias

### Questions
1.  Could you provide a more formal justification for the `d+1` framework? Beyond the analogy, is there a theoretical reason (e.g., from an approximation theory perspective) why adding an auxiliary dimension and evolving along it should lead to better performance for learning PDE solution operators?
2.  For a representative benchmark like the 2D Navier-Stokes or 3D Rayleigh-Taylor instability, could you please provide a direct comparison of the total number of trainable parameters and the estimated FLOPs per forward pass for SKNO, FNO, and Transolver with the configurations used in your experiments?
3.  How does the performance of SKNO change as you vary the resolution of the auxiliary dimension, `N_p`? The paper seems to fix `N_p` based on the baseline (e.g., Table 7). Is there a trade-off between accuracy and computational cost associated with this hyperparameter?
4.  The evolution along the `p`-dimension is a key component. Have you experimented with replacing the spectral operator (`F^-1 * A * F + b`) with a simpler or different type of operator? For instance, how would a simple MLP applied across the `p`-dimension "channels" perform? This would help isolate the benefits of the `d+1` structure from the specific choice of operator.

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
4

### Summary
This paper tackles the problem of Neural Operators from the Schrodingerisation method used in quantum simulations of PDEs. Hoping to address the use of discretization matrix in the current methods, it proposes a new operator, namely Schrodingerised Kernel Neural Operator (SKNO), which is claimed to be operating on a (d+1) dimensional space. This is claimed to be better aligned with the underlying evolution mechanism. Experiments are conducted on a range of datasets to show the efficacy of the method.

### Strengths
1. Showing the connection between  Schrodingerisation and NO is interesting. 

2. It is shown that SKNO is resolution invariant. 

3. Good experimental design.

### Weaknesses
1. There is no strong intuition for why Schrodingerisation should be the best choice for the NO (beyond empirical success).

2. Theorems 1 and 2 effectively restate the FNO results in (d+1) dimension rather than showing why this formulation is fundamentally different/better (in terms of convergence, scale, computations etc) than the exisisting methods. 

3. The architectural design seems incremental compared to FNO (except for the auxiliary dimension).

4. Computational complexity claims are not accurate (the additional log N_p factor seems to be downplayed). 

5. Looks like the method is over-sensitive to the hyper-parameter choices. 

6. Chosen baselines seem limited (see this for more datasets and methods https://arxiv.org/pdf/2310.01650). Further, transolver and SKNO use different levels of N_p. 

7. The paper uses very dense notations (both in the text and figures), which makes it a little hard to read.

### Questions
1. Can you think of a theorem showing that d+1 dimensional operators have greater expressiveness than d-dim ones for a given parameter budget? 

2. Is there a physical interpretation for the auxiliary dimension p? 

3. What happens if N_p is increased to match that of Transolver? 

4. Despite SKNO being designed for linear PDEs, how can it do well on highly non linear problems?

5. Can an iso-paramter/iso-flop comparison be shown? 

6. Is there a principled way of making choices in Tab 5, it seems arbitrary now. 

7. Are there scenarios where SKNO performs worse than baselines? When does d+1 framework not help? 

8. For 3D problems with finer resolution, how does memory scale with N_p?

### Soundness
2

### Presentation
2

### Contribution
2
