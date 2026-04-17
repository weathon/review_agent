# Spectral-inspired Operator Learning with Limited Data and Unknown Physics

- Decision: Reject
- Scores: 6, 2, 2, 8

## Abstract
Learning PDE dynamics from limited data with unknown physics is challenging. Existing neural PDE solvers either require large datasets or rely on known physics (e.g., PDE residuals or handcrafted stencils), leading to limited applicability. To address these challenges, we propose Spectral-Inspired Neural Operator (SINO), which can model complex systems from just 2-5 trajectories, without requiring explicit PDE terms. Specially, SINO automatically captures both local and global spatial derivatives from frequency indices, enabling a compact representation of the underlying differential operators in physics-agnostic regimes. To model nonlinear effects, it employs a $\Pi$-block that performs multiplicative operations on spectral features, complemented by a low-pass filter to suppress aliasing. Extensive experiments on both 2D and 3D PDE benchmarks demonstrate that SINO achieves state-of-the-art performance, with improvements of 1–2 orders of magnitude in accuracy. Particularly, with only 5 training trajectories, SINO outperforms data-driven methods trained on 1000 trajectories and remains predictive on challenging out-of-distribution cases where other methods fail.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel neural operator framework that learns to predict spatial derivatives and the combined right-hand side of partial differential equations (PDEs), which are then advanced in time using the RK4 scheme. The proposed method demonstrates superior data efficiency and predictive accuracy compared to baseline approaches on the 2D Kuramoto–Sivashinsky equation (KSE), 2D Navier–Stokes equations (NSE), and both 2D and 3D Burgers’ equations.

### Strengths
1. The paper is well structured and clearly presented. In particular, the details of SINO and its connection to spectral methods are thoroughly explained, accompanied by solid theoretical insights that enhance the overall understanding of the approach.
2. The effectiveness of SINO is comprehensively validated across a diverse set of benchmarks, including both 2D and 3D PDEs. Moreover, SINO exhibits strong robustness in low-data regimes and maintains high performance under out-of-distribution settings.

### Weaknesses
1. **Insufficient comparisons with baselines**: The integration of spectral methods with neural operators has been investigated in prior works such as SNO[1], LSM[2] and NMS[3]. A more comprehensive discussion of how spectral components are incorporated into model architectures and training strategies in these studies would further enrich the Related Works section. While NMS adopts a physics-informed training paradigm and may not be directly comparable, SNO and LSM share a more similar experimental setting and therefore serves as a suitable baseline for empirical evaluation.
2. **Experiments on more challenge tasks**: The authors state that the proposed method addresses challenges related to limited data and unknown physical models, which are common in scientific computing. However, concrete examples of such scenarios are not provided. I suggest that the authors include realistic examples of these scenarios to better illustrate the motivation and to evaluate the method’s effectiveness in practical settings. For instance, the practical design tasks in [4] could serve as a useful reference for such considerations.
3. **Scalability with respect to data**: Although the paper primarily focuses on the low-data regime, it is also important to assess how the proposed method scales with increasing amounts of training data. An analysis of data scalability would provide a more complete understanding of the method’s performance characteristics.


**Reference**

[1] Fanaskov, Vladimir Sergeevich, and Ivan V. Oseledets. "Spectral neural operators." Doklady Mathematics, 2023.

[2] Wu, Haixu, et al. "Solving High-Dimensional PDEs with Latent Spectral Models." ICML, 2023.

[3] Du, Yiheng, Nithin Chalapathi, and Aditi S. Krishnapriyan. "Neural Spectral Methods: Self-supervised learning in the spectral domain." ICLR 2024. 

[4] Wu, Haixu, et al. "Transolver: A Fast Transformer Solver for PDEs on General Geometries." ICML 2024.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the Spectral-Inspired Neural Operator (SINO), a novel architecture designed to learn PDE dynamics from extremely limited data (as few as 2-5 trajectories) without prior knowledge of the governing equations. Inspired by classical spectral methods, SINO uses modules to learn spectral multipliers for derivatives and nonlinear interactions (with de-aliasing). SINO is evaluated on 2D and 3D PDE benchmarks, outperforming several data-driven & physics-encoded baselines, and demonstrating robust out-of-distribution generalization.

### Strengths
- The OOD generalization and super-resolution experiments demonstrate that SINO is able to approximate the underlying operator using only a very limited number of trajectories.
- The architecture is motivated by first principles, resulting in a learnable (physics-agnostic) version of classical spectral solvers and providing a good inductive bias for low-data regimes.
- The paper provides ablation studies confirming the necessity of each key component.

### Weaknesses
There are two main concerns:

1. In almost all experiments, the parameter count of the baselines is significantly larger than the one of SINO. Together with the relatively small grid of hyperparameters used for each method, it seems that all these methods are overfitting to the limited amount of training data. For a fair comparison, each baseline should be evaluated with several configurations that lead to a comparable parameter count. Moreover, it is unclear how much samples from each trajectory are taken for each method, in particular, since Table 3 shows different time-step sizes for DOL & POL methods.
2. The proposed model is not a surrogate model for accelerating the numerical solution of PDEs. It incurs a similar inference time as a solver since it still relies on a RK4 integrator with sufficiently small steps. In this sense, there is no point in comparing to data-driven surrogate models. The focus of the paper should be on learning/discovering the underlying physical operator from limited data and comparing to other baselines in that domain. While the paper propose *operator distillation* as a workaround, this only shows that the underlying operator can be sufficiently well approximated by SINO such that it can act as a synthetic data generator. 

Other concerns:

- Apart from the RK scheme used for time-stepping, the design of SINO is very similar to variants of FNOs, in particular https://arxiv.org/abs/2403.12553 for nonlinear mixing in the channel dimension and https://arxiv.org/pdf/2111.13587, https://arxiv.org/pdf/2403.03542 for MLPs in the Fourier space.
- In its current form, the method is dependent on regular grids and periodic boundary conditions, excluding a vast number of real-world physics and engineering problems.
- It would be good to evaluate other metrics such as spectra.
- As mentioned above, only a few hyperparameter settings are tested for each baseline. In particular, this only includes a single training strategy (pushforward trick) with two learning rates.
- The universality result is relatively standard and it seems to also follow from https://arxiv.org/abs/2304.13221
- It is unclear how the low-pass filter is adapted in the super-resolution results.

### Questions
See Weaknesses above

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SINO (Spectral-Inspired Neural Operator), a neural architecture designed to learn PDE dynamics from a very small number of trajectories without explicit knowledge of governing equations. The model mimics classical spectral solvers by operating in the frequency domain through a learnable “Freq2Vec” module, modeling nonlinear terms via multiplicative Π-blocks, and enforcing numerical stability using the 2/3 de-aliasing rule and RK4 time integration. Experiments on periodic PDEs (KSE, NSE, Burgers 2D/3D) claim state-of-the-art accuracy, outperforming existing neural and physics-encoded operators by 1–2 orders of magnitude.

### Strengths
- The authors propose an interesting surrogate architecture that directly mimics spectral numerical solvers. The inclusion of classical ingredients such as the 2/3 de-aliasing rule and Runge–Kutta-4 integration demonstrates a clear attempt to build numerical stability into the rollout process, something that many prior neural PDE solvers neglect.

- By structuring the model around Fourier transforms and multiplicative nonlinear interactions, SINO incorporates strong physics-motivated priors (e.g., global coupling in the frequency domain, multiplicative nonlinearities akin to convection terms). These choices may help mitigate overfitting in low-data settings.

- Conceptually, the paper highlights an appealing direction: combining spectral-method heuristics with learnable neural representations. Even if the implementation details are debatable, the underlying idea of treating neural operators as extensions of traditional numerical solvers is valuable for future research.

### Weaknesses
- Clarity and Notation

The paper is poorly written and the notation is vague, making it difficult to follow the technical exposition.
Section 3.3 (Architecture components), in particular, is imprecise and lacks formal definitions for core modules.

- Architectural Novelty Is Overstated

The “Spectral Learning Block” is highly similar to the Fourier layer in the Fourier Neural Operator (FNO). Both perform Fourier transforms, apply learnable frequency-domain multipliers, and inverse-transform the results. In FNO, this is realized via a learned matrix  R(k) operating directly on spectral coefficients. In SINO, this role is played by the Freq2Vec MLP that maps frequency indices to multipliers  Hence, the claimed novelty of “automatically capturing spatial derivatives via frequency embeddings” largely duplicates FNO’s existing design, merely reparameterized via an MLP. This similarity is never acknowledged or discussed, which undermines the claimed originality.

- Experimental Fairness and Reproducibility

(a) No released code

The supplementary material lacks any code or configuration files, making it impossible to verify the experimental claims.
Given the unusually large reported performance gap (1–2 orders of magnitude), this lack of transparency is problematic.

(b) Likely unfair training setup

The baselines are trained under a setup that gives SINO a massive structural advantage. SINO is trained and integrated with a fine time step (Δt ≈ 1e−3–1e−2) and Runge–Kutta 4 (RK4) integration, while the data-driven baselines (FNO, FFNO, CNext, etc.) are trained with much coarser Δt (0.2–1) and single-step rollouts. This means that SINO performs 100–1000× more effective temporal updates per trajectory, producing thousands of additional training samples and much higher temporal resolution. Given only 2–5 trajectories in total, the data-driven baselines effectively see only a few unique training examples and rapidly overfit. Thus, the baselines are not trained in a setting consistent with their intended design. This leads to an unfair comparison that strongly favors SINO’s numerically stable formulation.

(c) Baseline training likely under-tuned

Although the paper mentions a small “grid search,” the sweeps are extremely limited (few values for channel count or Fourier modes only).
No optimization over learning rates, batch sizes, regularization, normalization, or data augmentation is reported.
In such a low-data regime, these hyperparameters are crucial; omitting them almost guarantees poor performance for FNO-type models.
The “NaN” and “>1” errors in Table 1 likely reflect training instability or overfitting, not inherent architectural limitations.

- Overly Smooth and Favorable Benchmarks

All benchmark PDEs (KSE, NSE, Burgers) are periodic, smooth, and globally coupled, which are exactly the scenarios in which spectral methods excel. Since SINO is explicitly designed as a spectral-inspired model with periodic boundary assumptions, these datasets do not test its robustness. No non-periodic, multi-scale, or irregular-domain tasks are included. The authors should have evaluated SINO on more challenging benchmarks such as PDE-Gym [1] or PDEBench [2], which include non-periodic or discontinuous dynamics.

- Inference Efficiency Defeats the Point of Neural Operators

The inference time of SINO is comparable to that of a traditional spectral solver (Figure 8). This undermines the core purpose of neural operators, whose value lies in being orders of magnitude faster than classical solvers at the cost of some accuracy. SINO is effectively a slow, learnable surrogate of a spectral solver, not a practically deployable fast operator. While the authors introduce a “distilled” SINO-FNO variant for speed, its training depends on the expensive SINO teacher, defeating the point of efficiency.

- Theoretical Analysis Is Hand-Wavy

The so-called “universal approximation theorem” (Theorem 1) is not rigorous. The argument establishes approximation only for a fixed input function u,  not uniformly over admissible functions. The constants C_j​ (lines 762-765) depend on u, so no operator-norm or uniform bound is shown. Hence, this is not a true universal approximation result for operators.

- Lack of Conceptual Rigor and Reproducibility

The description of training dynamics (warm-up steps, rollouts) is inconsistent across models. It is unclear whether identical normalization, boundary padding, or Fourier transform conventions were used for all methods. Reported “NaN” results for multiple baselines suggest missing stability controls or inconsistent preprocessing. Without the released code, these ambiguities make the empirical findings non-reproducible and potentially unreliable.

____

[1] Herde, M., Raonic, B., Rohner, T., Käppeli, R., Molinaro, R., de Bézenac, E., & Mishra, S. (2024). Poseidon: Efficient foundation models for pdes. Advances in Neural Information Processing Systems, 37, 72525-72624.

[2] Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). Pdebench: An extensive benchmark for scientific machine learning. Advances in Neural Information Processing Systems, 35, 1596-1611.

### Questions
How is the “Spectral Learning Block” fundamentally different from the Fourier layer in FNO, which also learns frequency-domain multipliers?

- Were all models trained with the same temporal discretization and integration scheme, or does SINO’s RK4 setup give it an implicit advantage in rollout stability?

- How do the authors ensure that data-driven baselines are not underfitting or overfitting given the extremely small number of training trajectories?

- Since all benchmarks are periodic and spectral in nature, how would SINO behave on non-periodic or multi-scale PDEs where spectral assumptions break down?

- Theorem 1 seems to establish approximation only for a fixed function u. How does this guarantee generalization across functions in operator space?

The authors should also review the Weaknesses section for additional (implicit) questions and points raised.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Spectral-Inspired Neural Operator (SINO), a neural PDE solver that learns complex physical dynamics from only 2–5 trajectories without requiring explicit PDE terms or residual supervision.
SINO combines several frequency-domain components: a Spectral Learning Block with Freq2Vec to learn frequency-wise multipliers, a Linear Operator Block to mix derivative-like channels, a Π-block for nonlinear multiplicative interactions with a 2/3 low-pass de-aliasing, and RK4 time-stepping for temporal evolution.
The model shows strong results on 2D and 3D PDE benchmarks, outperforming data-driven baselines trained on orders of magnitude more data, and demonstrating operator-level generalization to out-of-distribution (OOD) initial conditions.

### Strengths
The main strength of this paper lies in its data efficiency and physics-inspired inductive bias. By operating in the frequency domain and embedding derivative-like interactions through the pi block, the model implicitly encodes structural knowledge of PDE dynamics while remaining fully data-driven. This architectural design allows SINO to achieve remarkable performance using only a handful of training trajectories, outperforming other data-driven solvers that require hundreds or thousands of samples. Moreover, the model demonstrates impressive stability and accuracy over long time horizons, effectively suppressing compounding and high-frequency errors. The experiments show that SINO generalizes beyond the training distribution, accurately capturing phase and shape evolution of unseen initial conditions, suggesting a level of operator-level generalization. The methodology is clearly motivated and systematically described, and the experimental validation convincingly supports the paper’s central claims. Overall, the paper represents a meaningful advance in operator learning under data scarcity and physics-agnostic conditions.

### Weaknesses
Despite its strong empirical results, the paper has several limitations that should be addressed. First, the interpretability claims are primarily qualitative: while Figure 5 suggests that the Π-block features align with ground-truth differential operators, the paper provides no quantitative metrics (e.g., correlation or projection scores) to substantiate this claim. This weakens the argument that SINO learns physically meaningful spectral structures. Second, the 2/3 de-aliasing rule is fixed throughout the experiments, and the sensitivity of performance to the cutoff ratio is not explored. Understanding how different cutoff ratios affect accuracy and stability would clarify the robustness of the method across resolutions. Third, all theoretical and empirical results assume periodic boundaries and spectral grids, leaving it unclear whether the approach generalizes to non-periodic or irregular domains. Finally, while the ablation study compares RK4 and Euler integration, it does not analyze the effect of varying the RK4 step size itself, making it difficult to assess temporal stability more generally. These limitations do not invalidate the contribution but do restrict the generality and interpretability of the results.

### Questions
- If the RK4 integrator is retained, does a coarser time step maintain long-horizon stability? The paper shows that Euler integration reduces accuracy, but step-size sensitivity within RK4 is not reported.
- Are the pi-block and linear features consistently aligned with ground-truth operators across random seeds and resolutions, or is the alignment case-dependent?
- How sensitive is the model to the chosen 2/3 cutoff rule? Would alternative values (e.g., 0.5 or 0.8) change stability or accuracy significantly?
- Can the proposed method extend to non-periodic boundary conditions, or is the formulation inherently restricted to Fourier-periodic domains?

### Soundness
4

### Presentation
3

### Contribution
4
