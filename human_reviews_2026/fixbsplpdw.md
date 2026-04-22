# Physics-Preserving Compression of High-Dimensional Plasma Turbulence Simulations

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
High-fidelity scientific simulations are now producing unprecedented amounts of data, creating a storage and analysis bottleneck. A single simulation can generate tremendous data volumes, often forcing researchers to discard valuable information. A prime example of this is plasma turbulence described by the Gyrokinetic equations: nonlinear, multiscale, and 5D in phase space. They represent one of the most computationally demanding frontiers of modern science, with runs taking weeks and resulting in tens of terabytes of data dumps.
The increasing storage demands underscore the importance of compression, however, compressed snapshots might not preserve essential physical characteristics after reconstruction. To assess whether such characteristics are captured, we propose a spatiotemporal evaluation pipeline which accounts for structural phenomena and multi-scale transient fluctuations. Indeed, we find that various compression techniques lack preservation of temporal turbulence characteristics. Therefore, we explore Physics-Informed Neural Compression (PINC), which incorporates physics-informed losses tailored to gyrokinetics and enables extreme compressions of over 100000x. This direction provides a viable and scalable solution to the prohibitive storage demands of gyrokinetics, enabling post-hoc analyses that were previously infeasible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Physics-Inspired Neural Compression (PINC), a method that integrates physics-informed losses to compress massive gyrokinetic plasma simulation data by up to 70,000×. It preserves key spatial and temporal turbulence characteristics that conventional compression methods may fail to maintain.

### Strengths
Significance
This paper compares compression methods for high-dimensional scientific simulations, including neural field and vector quantized approaches in plasma turbulence, and presents solid experiments in a field lacking established benchmarks.

### Weaknesses
Originality

1. The proposed Physics-Inspired Neural Compression (PINC) lacks methodological novelty, as similar physics-informed compression frameworks have been explored in prior works such as GINN [1] and VQ-VAE for tubulence [2]. The paper mainly adopts existing  method on the plasma turbulence dataset without introducing a new architectural or algorithmic contribution.

Quality and Clarity

2. The paper does not clearly explain how the proposed physics-informed loss differs from or improves upon prior physics-guided neural compression works [1][2].

Significance 

3. The experimental results (Table 1) indicate a trade-off between PSNR and L1 error when incorporating physics-informed losses, suggesting that improved physical fidelity comes at the cost of reconstruction accuracy.
4. While the evaluation pipeline is comprehensive, the method itself is incremental, and its advantages over previous approaches are not sufficiently quantified.
5. The novelty and generalizability of the proposed method are limited, which may reduce its long-term impact compared to its benchmarking contributions.

[1] Geometry-Informed Neural Networks  
[2] A Physics-Informed Vector Quantized Autoencoder for Data Compression of Turbulent Flow

### Questions
1. Have the authors considered combining VQ-VAE with an entropy codec (e.g., arithmetic or Huffman coding) for improved hybrid compression efficiency?
2. What is the codebook utilization of the VQ-VAE component, and how does it affect compression quality and efficiency?

### Soundness
3

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
2

### Summary
This paper addresses the storage and analysis bottleneck for high-fidelity 5D gyrokinetic plasma turbulence simulations, which generate terabytes of data. The authors claim that standard lossy compression methods fail to preserve essential physical characteristics, particularly transient turbulence dynamics.
The core contributions are:
1. A novel physics-informed loss function: This loss function is specifically designed for gyrokinetics and incorporates terms to preserve physical integrals heat flux, electrostatic potential, turbulence spectra, and monoticity.
2. A Proposed Evaluation Framework: The paper proposes and uses a set of metrics to evaluate both spatial/steady-state quantities (quantitatively) and transient turbulence dynamics (the latter qualitatively).
3. State-of-the-Art Compression: The PINC-VQ-VAE model achieves an extreme compression rate of 70,000x while maintaining significantly better physics fidelity than all baselines.

### Strengths
1. Significance: The paper tackles a real, and high-value problem in the scientific ML, where data can be incredibly high-dimensional and sparse.
2. Clever Loss Function: While the concept of physics-informed losses is well-established (e.g., PINNs), the specific formulation is a significant strength. The authors move beyond standard PDE residuals. The inclusion of losses on derived, non-local turbulence spectra—which can be difficult to compute—and the isotonic loss to enforce a physically-correct spectral shape is a non-trivial and highly effective application of this idea to the compression domain.
3. Rigorous Experiments: The evaluation is thorough and backed by an impressive 500GB dataset (although difficult to share and reproduce).

### Weaknesses
1. Limited Baselines: The related work section (Sec 2) mentions other relevant deep learning methods for scientific data (e.g., VAPOR, Anirudh et al., Cranganore et al.). However, the quantitative comparison in Section 4 is limited to traditional methods (ZFP, Wavelet, PCA, JPEG2000) and the authors' own non-PINC ablations. This makes it difficult to assess how PINC compares to other state-of-the-art learned compressors in this domain.
2. Overstated "Unified Evaluation Pipeline" Contribution: The paper claims to contribute a "unified evaluation pipeline". This is strong language for what is, in practice, a curated set of metrics. While valuable, it is not a new automated framework. Furthermore, the authors admit in their limitations (Line 521) that the most novel part of this "pipeline"—the evaluation of transient dynamics—remains purely "qualitative"  and is not a quantitative metric.
3. Reproducibility: In your statement, you note the dataset is too large to distribute. A far more effective solution for reproducibility would be to provide the compressed test set.

### Questions
1. On Baselines: Your related work review is thorough, but the experimental baselines are primarily traditional methods. Could you comment on why other learned compression methods (like VAPOR or others) were not included in the comparison?
2. On "Unified Evaluation Pipeline": In practice, you introduced a new curated set of metrics. Can this really be called a unified evaluation pipeline?
3. On Reproducibility: You state the 500GB dataset is too large to share. A much more practical solution for reproducibility would be to release the compressed test set (i.e., the latent codes), which would be negligibly small (MBs). This would allow anyone to reproduce your entire analysis pipeline (all tables and figures) without needing to re-run the GKW simulations. Would you be willing to add this to your supplementary materials?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Physics-Inspired Neural Compression (PINC), a framework for compressing high-dimensional plasma turbulence simulations while preserving essential physical properties. The authors address the challenge of storing large gyrokinetic simulation data by proposing a single evaluation pipeline that measures both spatial and temporal turbulence characteristics. They investigate two neural compression paradigms (autoencoders and neural implicit field) and augment them with physics-informed loss terms derived from gyrokinetic integrals and turbulence spectra. The resulting PINC models achieve great compression rates while maintaining key physical quantities. The paper provides detailed quantitative and qualitative analyses, demonstrating that PINC significantly improves physics preservation compared to traditional compression methods.

### Strengths
- The paper addresses an underexplored yet highly relevant problem: data compression for large-scale physics simulations, rather than accelerating the simulations themselves. This is a practical and impactful direction, as it targets a major bottleneck in scientific computing: data storage and accessibility.

- The proposed Physics-Inspired Neural Compression (PINC) is conceptually well-motivated, bridging neural compression and physics-informed learning in a principled way. The inclusion of physically meaningful loss terms for gyrokinetics demonstrates a strong understanding of the domain.

- The experimental evaluation is extensive and well-structured, including comparisons with traditional compression methods, ablations of individual loss components, and scaling analyses across compression ratios.

- The results are compelling, showing that the proposed approach achieves extreme compression (up to 70,000x) while maintaining physically relevant quantities, a feat that existing methods fail to achieve.

- The paper is clearly written and well organized, with solid theoretical grounding and reproducibility details (including code, configurations, and dataset description).

- The authors provide a balanced discussion of limitations and outline meaningful future directions, such as incorporating temporal consistency and extending the approach to other domains.

Overall, the work sets a new benchmark for physics-preserving compression and highlights the potential of neural networks as a viable alternative to traditional methods in high-dimensional scientific data management.

### Weaknesses
Paper is really strong, minor potential issues: 

- While the proposed PINC framework is convincing, it remains domain-specific, with the physics-informed losses tailored to gyrokinetic equations. It is unclear how easily the method generalizes to other scientific domains (e.g., fluid dynamics, astrophysics). 

- Are there insights from this study that could inform the design of neural operators or surrogate models, possibly using compressed representations as priors or initialization?

### Questions
- How sensitive is the proposed physics-inspired loss formulation to the specific weighting or scaling of the individual components?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the critical data storage and analysis bottleneck in high-fidelity scientific computing, specifically focusing on 5D gyrokinetic plasma turbulence simulations. A framework Physics-Inspired Neural Compression (PINC) is proposed to integrate physics-informed losses into neural fields and vector-quantized autoencoders. The core of PINC is a composite loss function, which penalizes deviations in key physical integrals and turbulence diagnostics. The authors also introduce a unified evaluation pipeline to assess both spatial and transient dynamics. Results demonstrate that PINC models outperform traditional baselines and standard neural models in preserving these physical metrics, achieving extreme compression ratios of up to 70,000x.

### Strengths
1.This paper seems to be novel and valuable. The paper tackles a highly relevant problem for the HPC and computational science communities. The core idea of applying physics-informed losses not just for solving PDEs, but for data compression.
2.The authors provide a strong empirical evaluation. The analysis in Section 4.2 and Figure 5 moves beyond static, time-averaged metrics.
3.The ablation study is conducted to validate the contribution of each module in the proposed framework effectively and the visualization shows the comparison of compression effects.

### Weaknesses
1.In Section 3.2, the paper explores two distinct neural compression paradigms: Neural Fields and Autoencoders. What are the respective advantages and disadvantages (e.g., encoding/decoding speed, precise storage costs, generalization) of these methods in a real-world application?
2.In Section 4.2, the analysis of the energy cascade in Figure 5 is qualitative. Why was a quantitative transient metric not used in the evaluation? For example, computing the spectral error (such as L1 or Wasserstein Distance) at each individual timestep during the transient phase and reporting its mean would seem to more directly and rigorously validate PINC's advantage in preserving transient dynamics.
3.The composite loss in Equation 6 contains six distinct terms, which suggests a heavy optimization burden. The stability of this multi-term optimization and its sensitivity to the relative weighting of these components are not sufficiently addressed.
4.The chosen baselines, while standard, are somewhat outdated. The paper would be strengthened by comparing against more recent, state-of-the-art learned compression methods.
5.In Equation 3, the paper lacks an explanation for the meanings of the symbols $\mathfrak{R}$ and $\mathfrak{S}$.

### Questions
See Weaknesses. The authors are strongly recommended to clarify the effectiveness of different tasks and applications of the proposed method.

### Soundness
3

### Presentation
3

### Contribution
3
