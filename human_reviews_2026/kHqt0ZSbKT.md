# Random Controlled Differential Equations

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
We introduce a training-efficient framework for time-series learning that combines random features with controlled differential equations (CDEs). In this approach, large randomly parameterized CDEs act as continuous-time reservoirs, mapping input paths to rich representations. Only a linear readout layer is trained, resulting in fast, scalable models with strong inductive bias. Building on this foundation, we propose two variants: (i) Random Fourier CDEs (RF-CDEs): these lift the input signal using random Fourier features prior to the dynamics, providing a kernel-free approximation of RBF-enhanced sequence models; (ii) Random Rough DEs (R-RDEs): these operate directly on rough-path inputs via a log-ODE discretisation, using log-signatures to capture higher-order temporal interactions while remaining stable and efficient. We prove that in the infinite-width limit, these models induce the RBF-lifted signature kernel and the rough signature kernel, respectively, offering a unified perspective on random-feature reservoirs, continuous-time deep architectures, and path-signature theory.

We evaluate both models across a range of time-series benchmarks, demonstrating competitive or superior performance. These methods provide a practical alternative to explicit signature computations, retaining their inductive bias while benefiting from the efficiency of random features. Code is publicly available at: https://github.com/FrancescoPiatti/RandomSigJax.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors introduce a training-efficient framework for time-series learning by combining random features with CDEs. It uses randomly parameterized CDEs as continuous-time reservoirs, requiring training only a linear readout layer. Two variants are proposed: RF-CDEs, which lift inputs via random Fourier features, and RRDEs, which operate on rough-path inputs using log-signatures. In the infinite-width limit, these methods recover the RBF-lifted and rough signature kernels, offering a unified perspective on reservoir computing and path-signature theory. In addition to strong theoretical superiority, experiments on UEA datasets demonstrate that the proposed models achieve competitive or state-of-the-art performance.

### Strengths
- The idea of bridging the random CDE with reservoir computing is both novel and interesting. Under this general idea, the two proposed variants offer distinct and independent advantages.
- This paper presents solid theatrical contribution. the authors prove kernel convergence in the infinite-width limit, and the finite-dimensional computation remains efficient and scalable. 
- The paper is well-organized and clearly written, making the technical details accessible and the main ideas easy to follow.

### Weaknesses
- The connection between the proposed methods with reservoir computing should be strengthened.
- The datasets in UEA benchmarks are relatively small. Evaluating the proposed methods on larger-scale benchmarks such as the Long Range Arena (LRA) would strengthen the empirical claims and better demonstrate efficiency.
- Some baselines are lacking, a comparison with learning-based path signature approaches would be beneficial, such as [1] [2]

[1] Morrill, James, et al. "Neural rough differential equations for long time series." ICML, 2021.

[2] Walker, Benjamin, et al. "Log neural controlled differential equations: The lie brackets make a difference." ICML, 2024.

### Questions
1.	In the experimental part of paper “Random Fourier Signature Features”, the truncated Signature Kernel exhibits a strong performance, looks even better than the newly proposed two variants, could the authors clarify this?
2.	In line 240, should the term “homogeneous” be corrected to “inhomogeneous”? 


Typos:

1. Line 122: Remove the word “lift” before $x$

2. In Eq (4), $\omega_M$ should be $\omega_F$

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes random-feature reservoirs in continuous time built from controlled/rough differential equations, with only a linear readout trained. Two main variants are introduced: RF-CDE, which first lifts inputs with Random Fourier Features then evolves them through a random CDE; and R-RDE, which acts directly on geometric rough paths using a log-ODE discretization with log-signature inputs.

### Strengths
- The writing is high-quality and mathematically literate. The background sections on rough paths, signatures, and kernels are compact, accurate, and helpful for positioning the work.

- The paper cleanly recalls the R-CDE limit to the signature kernel and extends it with two variants whose limits are the RBF-lifted signature kernel (RF-CDE) and the rough signature kernel (R-RDE). The statements (Theorems 3.2 and 3.4) are explicit.

- The paper correctly frames the models as training-efficient reservoirs with a linear readout. It provides a clear asymptotic cost analysis that contrasts the linear-in-length feature extraction with the quadratic scaling of kernel Gram matrix approaches.

### Weaknesses
- The main theorems are in the infinite-width setting. The paper does not provide non-asymptotic approximation rates or generalization/error bounds for practical feature counts. 

- Only UEA classification is considered. There are no forecasting or irregular sampling experiments, despite the continuous-time claim. The asymptotic table is helpful, but there are no runtime or memory measurements on the UEA suite to validate the linear-in-length advantage or to quantify the cubic term in R-RDE.

- The algebraic mapping ($\Gamma_A$ and $\Pi_B$) definitions are dense. It is not clear how the chosen truncation level (m) and Hall/Lyndon basis size affect stability, feature variance, and compute in practice, or how these are tuned.

### Questions
- Do you have non-asymptotic approximation or error bounds that relate N, F, and signature truncation to excess risk or kernel approximation error?

- How sensitive are RF-CDE and R-RDE to $\sigma_A$, $\sigma_b$, $\sigma_0$, F, and log-signature truncation (m)? Please add systematic sweeps or ablation studies.

- Why not include trained Neural CDE and modern sequence models as baselines for classification (such as variants of Neural CDEs or stable SDEs)?

- Do these models maintain advantages on forecasting or irregular sampling tasks, where continuous-time structure should help? Have you considered classification with missing observations or features? 

- Can you report wall-clock time and peak memory across UEA for fixed accuracy targets, to validate Table 1's asymptotics?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a training-efficient family of time-series models that combine random reservoirs with controlled/rough differential equations (CDE/RDE). Two main variants are introduced:
RF-CDE: lift inputs with Random Fourier Features and evolve them through a random CDE; in the infinite-width limit the model induces the RBF-lifted signature kernel.
R-RDE: operate directly on rough paths via a log-ODE discretization; in the infinite-width limit the model induces the rough signature kernel.
Only a linear readout is trained (reservoir computing spirit), yielding fast fitting and linear-in-samples scaling.

### Strengths
1. Clean theoretical framing with meaningful limits. RF-CDE -> RBF-lifted signature kernel and R-RDE -> rough signature kernel in infinite width; the results connect random differential equations to signature kernels in a principled way.
2. Train only a linear head on top of fixed random dynamic, this is simple and fast to fit in practice
3. Clear accounting for feature-extraction costs; random-feature models scale linearly in sequence length l.

### Weaknesses
1. The main theorems characterize infinite-width limits. There are no non-asymptotic approximation or generalization bounds to explain when a few hundred features suffice
2. All evaluations are classification on UEA (16 datasets). No forecasting, imputation, irregular sampling, or robustness to noise/missingness. 
3. While RF-CDE averages best among random-feature models, R-RDE trails (avg. 0.708). Some difficult datasets (EigenWorms, Handwriting) show large gaps to SigPDE/RFSF.
4. Ablations are thin. Beyond doubling features (250→500), there’s little exploration of RFF count F, signature truncation/order M, or log-ODE level m. 
5. There is no one figure in the whole paper, which will increase the difficulty for the reader to understand and capture the key designs.

### Questions
1. May you provide non-asymptotic bounds?
2. How do RF-CDE and R-RDE handle missing data, time jitter, or heavy noise? Any controlled experiments varying p-variation or noise levels?
3. Are there stability issues for large m or rough drivers For R-RDE’s log-ODE?
4. can you analyze why and provide guidance on when to choose R-RDE over RF-CDE?
5. Any preliminary results on forecasting or imputation to bolster the practical case?

### Soundness
3

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
2

### Summary
The paper introduces random-feature-based continuous-time reservoir models (RF-CDEs and R-RDEs) for time-series tasks. Both approaches approximate continuous-time dynamics using random neural controlled differential equations (N-CDEs) as reservoirs, while training only a linear readout layer. In essence, this provides a mathematical/software realization of the physical reservoir computing (PRC) framework. The authors show that, in the infinite-width limit, these models converge to kernel methods: RF-CDEs to an RBF-lifted signature kernel, and R-RDEs to a rough signature kernel. This establishes a theoretical bridge between neural and kernelized path representations. Experiments on standard time-series benchmarks demonstrate competitive accuracy and high training efficiency.

### Strengths
- The paper bridges two compelling areas: reservoir computing and the infinite-width limit of neural CDEs. To me, this connection seems novel.
-  The work is technically sound and has solid theoretical grounding. It proves asymptotic kernel equivalence and establishes existence and uniqueness of the limiting dynamics. 
- The authors provide a computational complexity analysis; the proposed linear-readout-only training offers efficiency gains.
- Although the paper is primarily theoretical, it includes empirical benchmarks on standard time-series classification datasets. RF-CDE achieves the notable average performance among the baselines. 
- The manuscript is theory-heavy but well-organized.

### Weaknesses
I do not identify any critical drawbacks in this paper. However,

- The experimental section focuses mainly on ablations of the proposed models and their variants. This improves completeness, but practitioners may be unsure about the broader practical advantages.

- In particular, while the motivation is framed in terms of neural CDEs + PRC, the paper does not compare against neural CDEs or other end-to-end trained deep learning architectures.

### Questions
- How does the expressive capacity of RF-CDE and R-RDE compare to that of a learnable Neural CDE with an equivalent number of parameters? Is there any theoretical or empirical analysis addressing this? 

- The proposed models can be regarded as theoretical or software analogues of the PRC, which in principle requires only a linear output layer. However, practical hardware implementations of PRC often include a nonlinear output layer to achieve performance on par with NNs. Have the authors explored how introducing a small nonlinear readout layer would affect the performance of RF-CDE or R-RDE?

 - In the infinite-width limit, the RF-CDE and R-RDE converge to Gaussian processes with signature-kernel covariances. Could this be interpreted as placing a specific functional prior over neural CDE dynamics, analogous to the GP prior that emerges in the NTK or NNGP limits?

### Soundness
3

### Presentation
2

### Contribution
3
