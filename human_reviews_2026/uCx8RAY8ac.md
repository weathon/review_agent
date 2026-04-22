# Chaining Spectral Pearls: Ellipsoidal Forecasting Beyond Trajectories for Time Series

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Current long-term time-series forecasting (LTSF) benchmarks are dominated by noisy stochastic datasets and pointwise losses, so models that look strong on ETT-type tasks can behave unpredictably under deterministic chaos or controlled regime shifts. We argue that forecasters should be stress-tested on canonical chaotic systems and on synthetic benchmarks with precisely scripted non-stationarity, and that evaluation should focus on the geometry of predictive distributions, not just single trajectories. We present FERN (Forecasting with Ellipsoidal RepresentatioN), a geometry-aware forecaster that uses a bidirectional encoder and a per-patch local linear transport map, factored as translate--rotate--scale--rotate-back with explicit eigenvalues and eigenvectors. The network therefore "only'' learns to generate stable Jacobians, while users obtain spectral diagnostics of local stretching, volume change, and regime switches. Alongside MSE/MAE we report Wasserstein Distance (shape fidelity) and Effective Prediction Time (horizon stability). Across 21 synthetic systems (chaotic, stochastic, and switching) and cleaned ETT/Weather benchmarks, FERN is a strong all-round "safe'' model: it achieves the best or second-best MSE or SWD on 19/21 synthetic tasks, maintains geometric fidelity far beyond the Lyapunov horizon on Lorenz-63, and remains competitive on real-world LTSF. The codebase also releases our controlled-shock benchmark and data-cleaning protocol.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces FERN (Forecasting with Ellipsoidal Rep-resentatioN), a time-series forecaster that predicts future geometry (ellipsoids) rather than exact trajectories, which is robust under deterministic chaos. It employs local linear transport with explicit spectral factors (eigenvectors/eigenvalues) for interpretability. FERN is stress-tested on chaotic systems (Lorenz63, Rössler, Chua) using new metrics like Sliced Wasserstein Distance and Effective Prediction Time, and outperforms baselines significantly on these.

### Strengths
1. The paper proposes new metrics (Sliced Wasserstein Distance and Effective Prediction Time) for evaluating forecasting performance in chaotic systems.  
2. The evaluation involving real-world benchmarks enhances the practical relevance of the proposed method.

### Weaknesses
1. The Appendix is unstructured and requires more rigorous revision to meet the publication standard.
2. The literature review is insufficient, and many related works and baselines are not discussed, making it difficult to assess the novelty and effectiveness of the proposed method. For instances:
 - Koopman operator-based methods for modeling chaos: Cheng, Xiaoyuan, et al. "Learning Chaos In A Linear Way." The Thirteenth International Conference on Learning Representations.
 - Geometric distribution preserving methods: Li, Zongyi, et al. "Learning Dissipative Dynamics in Chaotic Systems." (NeurIPS 2021).
 - Chaos system Benchmarks: Gilpin, William. "Chaos as an interpretable benchmark for forecasting and data-driven modelling." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).
 - Reservoir computing methods for chaotic systems.
 - Diffusion/Flow based methods, e.g. Shysheya, Aliaksandra, et al. "On conditional diffusion models for PDE simulations." Advances in Neural Information Processing Systems 37 (2024): 23246-23300.

3. The paper lacks clarity in validating the method in how is the spectral factors in deployed; how the network is implemented and optimized with the Algorithm 1.

### Questions
1. How is future geometry defined formally? 
2. How is the model's performance compared to existing methods for chaotic time series forecasting as referenced above?
3. Interest in scalability of Rotation: for high-dimensional data, how does the fixed R=8 reflections sufficiently approximate the necessary rotation U(z), and is this rotation layer a bottleneck for datasets larger than the 21 variables in Weather?

### Soundness
2

### Presentation
1

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
The motivation of this paper is twofold: (1) it introduces Forecasting with Ellipsoidal RepresentatioN (FERN) for long-term time-series forecasting and (2) contributes a new evaluation protocol. The proposed FERN model is geometry-aware that applies per-patch local linear transport using explicit spectral factors (eigenvectors/eigenvalues). This method transforms a Gaussian base distribution into a chain of local ellipsoids to predict the future geometry rather than the exact trajectory. The proposed evaluation metrics include Sliced Wasserstein Distance (SWD) and Effective Prediction Time (EPT) to address the limitations of pointwise metrics on chaotic systems. Experiments demonstrate that FERN outperforms baselines on several chaotic systems. The model also remains competitive on standard ETT and Weather benchmarks.

### Strengths
- The evaluation protocol is a substantive and timely contribution that correctly diagnoses blind spots in current LTSF practice—namely, overemphasis on noisy, quasi-periodic data and pointwise metrics. Accordingly, the authors propose concrete solution: geometry-aware (SWD) and stability (EPT) metrics with stress-testing on chaotic systems.
- The experiment evaluation is thorough and transparent. The authors provide comprehensive ablations and clear setup details in the appendix . A particularly valuable contribution is the identification of "recency bias" in standard validation splits, which addresses a common pitfall in training time-series models.

### Weaknesses
- The paper aggregates concepts from chaos theory, Koopman operator theory, optimal transport, and normalizing flows, but the authors fail to integrate these complex ideas into a unified framework and make the presentation hard to follow.
- The “Scope and Distinctions” section notes that FERN borrows NF/OT/Koopman language while not constituting any of these. The framework scope is ambiguous and unclear. Additional clarification should be included to support the spectral transparency claim. 
- The technical description is frequently replaced with unhelpful analogies, which severely hinders the understanding of the paper.

### Questions
- What does the paper mean when it says it targets "conditional local geometry, not the dynamics"? But the proposed model is used for learning and forecasting the dynamics.
- There are many blank margins in the manuscript; please remove them to improve readability and length compliance.
- The paper’s organization needs attention. There is no conclusion, and the references are placed after the appendix.
- In Algorithm 1, the Ellipsoidal Transport layer uses a fixed, learnable $K$ matrix to *mimic complex value eigenvalues*. How is this $K$ matrix trained, and is it shared across all datasets or trained per dataset? What evidence supports the claim that this simple block structure is sufficient to capture "global components"  of diverse dynamical systems?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FERN, a geometry-aware forecaster that represents future time-series patches as locally linear ellipsoids. It introduces a new evaluation protocol using Wasserstein distance and Effective Prediction Time, arguing that existing long-term forecasting (LTSF) metrics overfit noise and miss chaos.

### Strengths
I like the theoretical explaination of this article, looks good

### Weaknesses
- Writing is heavy, overly theoretical, and sometimes reads like a position essay rather than a reproducible model paper.
- No phase-space or attractor plots to prove that geometry preservation actually happens. Figure 1 is schematic only.

### Questions
1. Add visual reconstructions of chaotic attractors showing ellipsoidal chains.
2. How does the model sensity to the initial condition? some demonstration would be greatful, it is not clear to me.

### Soundness
3

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
The authors propose FERN, a model for long-term time series forecasting that represents local dynamics via ellipsoidal approximations with symmetric positive semi-definite Jacobians. The paper proposes a new evaluation protocol stress-testing on chaotic systems using Wasserstein-2 distance and effective prediction time metrics. The methodology aims to predict local conditional geometry via first-order Taylor approximation coupled with spectral decomposition, along with a Koopman-inspired global operator layer. On chaotic benchmarks, FERN achieves significantly better performance while remaining competitive on standard ETT benchmarks.

### Strengths
The paper achieves strong empirical results on chaotic benchmarks, with FERN substantially outperforming baselines. This performance gain, regardless of the theoretical framing issues, demonstrates that the specific architectural choices have merit for certain types of time series.

The emphasis on stress-testing models on chaotic systems is valuable, even if not entirely novel. The paper makes a reasonable argument for why current benchmarks may reward overfitting to specific historical trajectories rather than learning generalizable dynamics.

The Takens' embedding discussion, while not directly justifying the proposed method, provides useful context for understanding why simple models succeed on some benchmarks and why patching works well.

### Weaknesses
- The core idea of the FERN model is conceptually interesting but it is presented as an ad-hoc combination of normalizing flow, optimal transport, and Koopman frameworks. The authors explicitly state they are "not actually" implementing any of these theories rigorously, suggesting design-by-analogy rather than derivation from first principles and it also raises the question of why these frameworks are invoked at all.

- I found the collaborative movie adaptation analogy in Section 3 unhelpful and it made the technical content hard to parse. My understanding is that the proposed FERN model consists of two components:
   - **Encoder:** Despite being framed as "adapted ANF," this is simply a bidirectional coupling network that iteratively refines x and z ~ N(0,I) as follows: For i = 1…5: $z \leftarrow s^*(x) \odot z + t(x)$, $x \leftarrow s^*(z) \odot x + t(z)$. This mapping lacks the fundamental properties of normalizing flows and I don't see any clear benefit from the ANF framing. 
  - **Ellipsoidal Transport:** The prediction is computed as: $y^* = U(z) K \Lambda (z) K^T U(z)^T y_0 + t(z)$, where $y_0 ~ N(0,I)$. This is a structured linear transformation of Gaussian noise, conditioned on z. The optimal transport framing does not provide any meaningful insight since there is no transport problem being solved, no source/target distributions, and no transport cost. The Brenier theorem citation is irrelevant since this is supervised learning with MSE loss, not measure transportation. The SPSD structure merely constrains the linear map's form but doesn't make it an OT map.

  In summary, FERN essentially learns a nonlinear mapping from input x to parameters of an affine transformation applied to Gaussian noise, trained with standard MSE loss. The extensive discussion of ellipsoids, optimal transport, and Koopman theory obscures rather than illuminates this simple architecture. The actual contribution appears to be the specific parameterization choices that is shown to work well empirically on chaotic systems, not any deep connection to the invoked mathematical frameworks. 

- I was unable to find a formal definition of local conditional geometry and it was not clear to me if the ellipsoids refer to conditional distributions or prediction sets or second-moment approximations.

- The paper does not include a conclusions section and ends abruptly with the numerical studies. 

- The results for chaotic systems are impressive. However, the discussion of results needs more depth. For instance, it was not clear what happens beyond a few Lyapunov times where pointwise prediction is theoretically impossible.

### Questions
- Please formally define what "local conditional geometry” means in the context of this work.
- Section 3.2 provides a W2 bound; however, here deterministic predictions are being made and not distribution matching. Can you clarify what distributions are being transported and how this relates to minimizing MSE on point predictions?
- Algorithm 1 shows U(z) is data-dependent. How do you parametrize U to maintain orthogonality during SGD? 
- How do you prevent rank collapse?
- How does eigendecomposition cost scale with patch size P, number of patches L, and horizon H? It will be useful to provide wall-clock time comparisons at varying scales (e.g., d = 1, 10, 50; T = 100, 1000, 10000).
- Can you please provide details of the data preprocessing protocol used for FERN in the numerical studies.

### Soundness
2

### Presentation
2

### Contribution
2
