# FedMC: Federated Manifold Calibration

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Data heterogeneity in Federated Learning (FL) leads to significant bias in local training. While recent efforts to introduce distributional statistics as priors have shown progress, they universally rely on a flawed global linearity assumption, failing to capture the nonlinear manifold structures prevalent in real-world data. This model-reality mismatch causes the calibration process to generate out-of-distribution (OOD) samples, which fundamentally misleads the model. To address this, we introduce a paradigm shift. We propose Federated Manifold Calibration (FedMC), a novel framework that learns and leverages the local, nonlinear geometry of data. FedMC employs local kernel PCA on the client side to learn fine-grained local geometries, and constructs a global "geometry dictionary" on the server side to aggregate and distribute this knowledge. Clients then utilize this dictionary to perform context-aware, on-manifold calibration. We validate our proposed method by integrating it with a wide range of existing FL algorithms. Experimental results show that by explicitly modeling nonlinear manifolds, FedMC consistently and significantly enhances the performance of these state-of-the-art methods across multiple benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper makes a significant conceptual and practical contribution to federated learning by shifting the paradigm from global linear to local nonlinear geometric modeling. The proposed FedMC framework is innovative, theoretically grounded, and empirically effective across diverse settings. It opens a new direction for geometry-aware FL and has the potential to influence future work on distribution calibration in decentralized learning.

### Strengths
1. This paper identifies a key flaw in FL calibration methods, namely the global linearity assumption. This insight is both theoretically sound and empirically validated, and it reframes how we should think about distributional priors in heterogeneous FL.

2. FedMC is a well-motivated, technically sophisticated framework that bridges manifold learning and federated optimization. The use of local kernel PCA, secure projection onto a shared anonymous basis, and dynamic geometry querying constitutes a coherent and elegant solution to the on-manifold calibration problem under privacy constraints.

3. The experiments are comprehensive. Multiple datasets and heterogeneity types (label, domain, mixed). Comparison against SOTA FPL methods and a tailored linear baseline.

4. The construction of the Global Anonymous Basis (GAB) with differential privacy and the secure representation of local geometry via projection onto GAB are thoughtful and align well with FL’s privacy requirements.

### Weaknesses
1. The pre-image reconstruction step (solving for \(x'\) in the original space) involves iterative optimization per sample, which may introduce computational overhead. However, the authors note this is manageable and performed locally, and the empirical results justify the cost.

2. While the focus on FPL is well-motivated (communication efficiency, bias in prompts), a brief discussion on applicability to other parameter-efficient FL settings (e.g., LoRA, adapters) could further broaden impact.

### Questions
1. In Section 4.3.1, you mention using a Gaussian kernel with γ = 1/d for KPCA. Could you clarify what "d" refers to here? Is it the dimension of the image embeddings? Also, was this value chosen empirically, or is there a theoretical reason behind it?

2. I’m curious about how often the Geometry Dictionary is updated. Is it rebuilt from scratch in every communication round, or is it initialized once and then incrementally updated? If it’s updated every round, does the dictionary size grow over time, or is there a mechanism to keep it fixed? 

3. In Table 1 and 2, GGEUR(FedVTP) serves as a strong linear baseline. Could you briefly explain how GGEUR works in practice? Specifically, how is its “global geometric prior” computed and applied during calibration? Is the main difference from FedMC simply that GGEUR uses standard PCA instead of kernel PCA?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FedMC (Federated Manifold Calibration), a novel federated learning framework that explicitly models the nonlinear manifold geometry underlying client data. The key idea is to move beyond the conventional global linearity assumption (e.g., global PCA-based calibration) by capturing local nonlinear geometries through kernel PCA at each client. The server then aggregates these local representations via a Global Anonymous Basis (GAB) and a Geometry Dictionary, enabling privacy-preserving, on-manifold calibration during local training. Extensive experiments demonstrate that FedMC consistently improves the performance of both Federated Prompt Learning (FPL) and general FL algorithms across diverse heterogeneity scenarios (label skew, domain skew, and combined skew). The framework is theoretically motivated, methodologically complete, and empirically validated.

### Strengths
1. The paper challenges the often implicit global linearity assumption in geometry-based FL and introduces a principled nonlinear manifold perspective.
2. The design—from local KPCA to GAB construction and geometry dictionary fusion—is coherent and well justified under privacy constraints.
3. FedMC yields consistent improvements across six benchmarks and multiple FL paradigms, demonstrating generality and robustness.
4. The paper is clearly written, well structured, and includes detailed appendices and pseudocode that enhance reproducibility.
5.  The idea of modeling federated data manifolds could inspire future work on geometric and representation-level calibration in distributed learning.

### Weaknesses
1. It would be useful to briefly comment on the **runtime and communication overhead** of FedMC compared to standard FL baselines, even qualitatively.
2. Since the framework involves hyperparameters (e.g., kernel bandwidth γ, cluster number m, basis size N_base), a short note on how these were chosen in practice would improve clarity.
3. The experiments use up to 10 clients. A short discussion on how FedMC might scale with more clients or unbalanced participation would be valuable.
4. In a few places (e.g., Eq. 12–14), the correspondence between Φ(x), β*, and v* could be made slightly clearer for readers less familiar with kernel methods.
5. If space permits, a brief ablation on the effect of each key component (GAB, geometry fusion, calibration) would further highlight their individual contributions.

### Questions
1. The pre-image optimization in the calibration step (Eq. 12–14) involves iterative updates. Could you share how stable and efficient this process is in practice?
2. The Global Anonymous Basis (GAB) is initialized once. Would updating it periodically during training further enhance adaptability to shifting data geometry?
3. Have you explored alternative kernels (e.g., polynomial or Laplacian) for local KPCA, and if so, do they affect performance notably?
4. Regarding privacy, how does the added geometric sharing interact with differential privacy guarantees? Does it influence the overall privacy–utility balance?
5. Could FedMC potentially extend to multimodal or cross-modal FL settings, where clients hold different modalities (e.g., image–text)?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a new FL framework, FedMC, to address data heterogeneity by leveraging the nonlinear manifold structures of client data to describe the true local geometries and then securely aggregating to form a global geometry dictionary. This global dictionary enables on-manifold calibration of local embeddings. The authors conducted extensive experiments across multiple benchmarks to validate the improvement achieved by FedMC for both Federated Prompt Learning (FPL) and conventional FL algorithms.

### Strengths
1. The motivation is clearly presented, and the paper provides an intuitive analysis of the limitations of the global linearity assumption, which emphasizes the importance of on-manifold calibration in federated learning.

2. This paper provided a formal derivation in the appendix to justify the server's geometric fusion strategy.

3. This paper explicitly considers privacy preservation. Specifically, the authors proposed two mechanisms: (1) adding differential privacy to the anonymized prototypes and (2) applying secure projection to the extracted local geometry.

### Weaknesses
1. While section 4.3.1 is clear to me, the notations and equations in sections 4.3.2 and 4.4 are too dense. Incorporating one diagram to help illustrate the key steps could significantly improve readability.

2. FedMC seems to introduce too much computational overhead on the client side, requiring many operations such as K-Means clustering, Kernel PCA, and pre-image reconstruction.

### Questions
1. Can the authors justify how to ensure the reconstructed x′ lies on the true data manifold?

2. In section 4.4, the proposed “dynamic geometry query” finds the most relevant template entry based on the Euclidean distance (Eq. 9). Given that the concerns raised by the authors in section 3 about the issues regarding the simple Euclidean shortcuts, why is geodesic distance or a manifold-aware metric not considered here?

3. Those kernel-based operations typically scale poorly with large datasets or many clients. Can the authors justify the scalability of FedMC in the settings where both the number of local data points and the number of clients may be large?

### Soundness
2

### Presentation
2

### Contribution
3
