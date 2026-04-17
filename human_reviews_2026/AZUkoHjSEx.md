# CloudNFMM: A Hybrid Hierarchical and Local Neural Operator Inspired by the Fast Multipole Method

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
The Fast Multipole Method (FMM) is an efficient numerical algorithm used to calculate long-range forces in many-body problems, leveraging hierarchical data structures and series expansions.
In this work, we present the Cloud Neural FMM (CloudNFMM), a new neural operator architecture that integrates the hierarchical structure of the FMM to learn the Green's operator of elliptic PDEs on point cloud data.
The architecture efficiently learns representations for both local and far-field interactions.
The core innovation is the local attention, a specialised local attention mechanism which models complex dependencies within a small neighbourhood of points.
We demonstrate the effectiveness of this approach, and discuss possible extensions and modifications to the CloudNFMM architecture.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes **CloudNFMM**, a neural operator that integrates the **hierarchical information flow of the Fast Multipole Method (FMM)** with a **local attention** module to learn Green’s operators on **unstructured point clouds**. Long-range (far-field) effects are handled by an FMM-style **upward/downward tree pass** in which classical translation operators are replaced by small MLPs; short-range (near-field) interactions are modeled by a **shared local attention** over a $(3\times3)$ neighborhood of spatial “boxes.” A simple preprocessing partitions points into boxes and pads to a fixed per-box size to enable hybrid tree/point operations; **RoPE** provides position encoding. The design aims for resolution/discretization agnosticism and near-linear scalability akin to FMM. On **WaveBench** (2D Helmholtz), CloudNFMM achieves lower relative $(L_2)$ error than FNO/U-Net/UNO at **~1.8–1.9M params**; on **PDEBench–Darcy**, results are weaker for small $(\beta)$ but competitive at **$(\beta{=}100)$**. The method currently targets **time-harmonic** problems; the authors discuss extending to time-dependent settings via a latent recurrent update and improving tree↔leaf coupling.

### Strengths
1. **Clear and comprehensive writing.**
   The paper is well-structured (motivation → FMM background → architecture → experiments), with enough implementation detail to follow the hybrid tree/local-attention design and training setup.

2. **Captures both local and global interactions.**
   The model cleanly separates **far-field/global** effects via an FMM-style hierarchical upward/downward pass and **near-field/local** effects via a dedicated local attention operator over neighboring boxes—an explicit local–global split that aligns with Green’s operator structure.  

3. **Operates directly on unstructured point clouds.**
   CloudNFMM is designed to be **resolution/mesh agnostic**, working on point clouds rather than regular grids, which broadens applicability to irregular domains compared to FFT-based operators.

### Weaknesses
1. **Narrow and relatively simple PDE coverage; mixed accuracy vs. SOTA.**
   Experiments focus mainly on time-harmonic Helmholtz (WaveBench) and Darcy—both comparatively simple. Even with a fairly complex hybrid (FMM + local attention), CloudNFMM is **not consistently SOTA**; in some settings, a simpler FNO matches or outperforms it. This weakens the claim that the hierarchical FMM flow brings clear accuracy advantages.

2. **Point-cloud claim vs. grid-based evaluation.**
   Although the method is positioned for **unstructured point clouds**, the benchmarks used are largely **regular-grid** problems at **small scales**. Without results on genuinely irregular point clouds (varying density, complex boundaries, real geometry), it’s hard to assess robustness to discretization changes—the core stated benefit.

3. **Limited analysis of component contributions and frequency behavior.**
   The experiments lack **ablations** isolating the roles of (i) the upward/downward FMM tree, (ii) the local attention, and (iii) the RoPE/box partitioning. There is also no analysis of **low- vs. high-frequency** error components to justify the local–global split. As a result, the source of gains and the necessity of each module remain unclear.

### Questions
1. **PDE breadth & SOTA positioning.**
   Can you add harder/diverse PDEs (e.g., variable-coefficient Helmholtz, elastostatics, time-dependent waves) and run **capacity/compute-matched** head-to-heads vs. strong baselines to clarify when CloudNFMM truly outperforms?

2. **True unstructured point clouds.**
   Since the method is pitched as point-cloud native, could you report results on **genuinely irregular point sets** (nonuniform density, complex/curved boundaries, different samplings of the same geometry), with **scaling** to larger sizes? Please detail robustness to resampling and boundary condition changes.

3. **Ablations & frequency analysis.**
   Please ablate (i) FMM tree passes, (ii) local attention, and (iii) RoPE/box partitioning, and provide a **frequency-resolved error** study (e.g., spectral energy error vs. wavenumber) to show which components capture low/high-frequency effects and why both local & global paths are necessary.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents CloudNFMM, a neural operator which is an extension of the neural fast multipole method for handling point clouds in a resolution invariant manner. The proposed architecture has two routes for processing information, as the authors term them, a "tree pass" and a "leaf pass." The tree pass is essentially just the NFMM approach, replacing each operator layer of the FMM with a simple learnable layer, which learns global dependences. The leaf pass is a local attention mechanism which learns short-range dependencies. Supporting elements of the design work to process both channels of information together. The authors claim this design achieves both scalability and discretization invariance, presenting some empirical results on two datasets.

### Strengths
- This work primarily fuses an established transform, the fast multipole method, within a neural operator framework. This is conceptually sound, as fast transform algorithms are the basis for many neural operators, and investigating a new approach is always interesting.
- Depending on the PDE, there may be primarily local or global dynamics. Ideally, a neural operator should handle both. It seems like this architecture was designed with this in mind when combining the tree pass and leaf pass.
- The model shows good parameter efficiency.

### Weaknesses
1. There is a mismatch between the motivation/exposition and the experimental studies. A great focus is placed on *point clouds* specifically, and yet both experiments are on a regular grid. 
2. To investigate the performance on general point clouds, a more thorough analysis and stronger baselines is required. In addition to the Geo-FNO, the FNO may be applied directly to point clouds [arXiv:2305.19663v4]. Alternatives like CORAL [2306.07266], GAOT [2505.18781], or Transolver [2402.02366] are also strong baselines for general geometries.
3. The baseline problems themselves are also quite limited. Moreover, while the authors say this approach is optimal for elliptic PDEs, there is no analysis on the performance across a broader class of PDEs. I see no issue with the NO being optimized for a particular application, but there is no evidence to suggest it actually does obtain optimal results here. The combination of a global + more powerful local operator would also suggest it should perform well on hyperbolic PDEs. Either more experiments or a strong mathematical justification are necessary.
4. Even on modest baselines, CloudNFMM underperforms in comparison to other models. 
5. No study of scaling with respect to data size or model size, and no ablations which present justification for the intuition behind the models performance.
6. The explanation of the model components is a bit dense, and I feel that more could be done to first provide the intuition behind this approach for reader.

### Questions
1. Could the authors please provide additional experiments, as outlined above? That is, broader investigation of datasets and comparison to stronger baselines.
2. How does this approach perform on resolutions unseen during training? I would like to see more study than what is presented in Table 7. 
3. Could making the local leaf pass a bit larger actually give the model stronger performance on datasets with local features?
4. The Fourier transform also captures global, long-range dependence. Is there any theoretical or explicit empirical justification for FMM to perform better than the FFT in this task?
5. Minor comment -- I would encourage the authors to use the Z. Li citation [2010.08895] for the original FNO paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose an architecture inspired by the fast multipole method and provide two benchmarks to a helmholtz problem and a darcy problem with discontinuous coefficients. The method is discretization invariant.

The premise is sound: FMM provides a natural extension of Fourier-based architectures with exploitable hierarchical structure and connections between fourier series and wavelet expansions. Unfortunately this is not a novel idea; many authors have considered this (some of which were mentioned). My assessment is that this amounts to an alternative set of choices for how to connect intermediate levels and apply attention.

In the absence of theory, this paper would require compelling experimental results for me to recommend publication. Unfortunately the accuracy is simply too low. For the first benchmark comparable accuracy to an FNO is achieved (albeit with 2-4x fewer parameters). For the second, orders of magnitude *worse* performance is observed.

### Strengths
Comparable performance is maintained to some FNO architectures with fewer parameters.

### Weaknesses
The experimental results don't make a compelling case for this architecture.

### Questions
No questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposed CloudNFMM, a neural operator architecture inspired by the Fast Multipole Method (FMM) to efficiently learn PDE solution operators on point clouds. It combines a hierarchical FMM-style global information flow with a local attention mechanism to capture both long-range and short-range interactions while remaining resolution-invariant. Benchmarks on elliptic PDEs show strong accuracy and parameter efficiency compared to other neural operators.

### Strengths
1. The paper draws inspiration from the Fast Multipole Method (FMM), a mathematically and physically grounded algorithm, into a neural operator framework. This provides a clear and well-justified motivation that connects classical numerical methods with neural networks.
2. The model introduces an effective hybrid design that combines an FMM-style hierarchical global module with local attention to handle both long-range and short-range interactions efficiently.

### Weaknesses
1. The performance on standard PDE benchmarks  is not consistently superior to existing models such as FNO or transformer-based operators.
2. While the paper’s stated motivation centers on unstructured data, its evaluation remains limited to structured domains, weakening the empirical support for this claim.

### Questions
1. How well would CloudNFMM generalize to genuinely unstructured domains? Are there any preliminary results or plans for such validation?
2. It would be helpful to see a comparison of computational efficiency (e.g., runtime or memory usage) between CloudNFMM and existing neural operator models.

### Soundness
3

### Presentation
4

### Contribution
2
