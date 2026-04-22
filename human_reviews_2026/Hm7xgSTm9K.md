# Efficient Generative Transformer Operators for Million-Point PDEs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
We introduce ECHO, a transformer–operator framework for generating million-point PDE trajectories. While existing neural operators (NOs) have shown promise for solving partial differential equations, they remain limited in practice due to poor scalability on dense grids, error accumulation during dynamic unrolling, and task-specific design. ECHO addresses these challenges through three key innovations. (i) It employs a hierarchical convolutional encode–decode architecture that achieves a 100× spatio-temporal compression while preserving fidelity on mesh points. (ii) It incorporates a training and adaptation strategy that enables high-resolution PDE solution generation from sparse input grids. (iii) It adopts a generative modeling paradigm that learns complete trajectory segments, mitigating long-horizon error drift.
The training strategy decouples representation learning from downstream task supervision, allowing the model to tackle multiple tasks such as trajectory generation, forward and inverse problems, and interpolation. The generative model further supports both conditional and unconditional generation. We demonstrate state-of-the-art performance on million-point simulations across diverse PDE systems featuring complex geometries, high-frequency dynamics, and long-term horizons.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents ECHO, a transformer–operator framework for generating million-point PDE trajectories. ECHO employs a hierarchical convolutional encode–decode architecture to compress regular or irregular spatio-temporal trajectories and adopts a flow-matching model to learn the trajectory distribution. The two parts are trained separately and the generative part enables forward and inverse problems, and interpolation. The proposed method demonstrates SOTA performance on million-point simulations on complex geometries.

### Strengths
Overall I think the proposed framework is elegant.

- The proposed method incorporates several reasonable technical designs: a hierarchical spatio-temporal encoder-decoder which can compress the high-dimensional input efficiently; a generative flow-matching framework which is capable of modeling complex distributions and generating high-fidelity spatio-temporal dynamics. The generative modeling framework naturally enables more types of tasks as well.
- The experiments in comparison to state-of-the-art baselines seem solid. The proposed method is evaluated on two types of tasks: reconstruction tasks and time-stepping tasks. The paper also presents evidence that baseline models are not able of handling million-point inputs.
- The paper is overall easy to follow.

### Weaknesses
My major concerns are about the experiments and writing.

- Although the generative framework enables interpolation tasks, the corresponding experiments are not included, and therefore this capability of the model is not well evaluated.
- After carefully reading through the paper, I feel that the paper is finished in a hurry. There are several typos: Line 88 "we leverate"; Line 317 "$\gamma$if available"; Line 333 the reference is missing; Line 450 the caption only highlights the reconstruction error, but the table contains both reconstruction error and time-stepping error.
- In table 3 you compare ECHO with several baselines and show that all baselines ran OOM. However, as far as I know, at least one of your baselines in Table 2-Transolver++ supports million-scale point inputs. Why not compare with it in this task?
- The experimental design is not clearly elaborated. For the time-stepping task, what is the model input and output? Do your model generate the whole trajectory all at once or step by step? What about your baselines? The lack of explanation makes it hard to evaluate the performance.
- Why do you include the large-scale Vorticity dataset in the irregular mesh section? Isn't it regular grid?
- There should be experiments comparing the efficiency of each model, for example, the inference time of each model to generate a full trajectory.
- What is the purpose of the reconstruction task? It is not a must for models to first encode the inputs into a latent space and then decode back. For example, Transolver++. However, the comparison with it is missing in this section.

### Questions
Please refer to weaknesses.

### Soundness
2

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
4

### Summary
This paper introduces ECHO, a transformer–operator framework for generating million-point PDE trajectories. It employs a hierarchical convolutional encode–decode architecture that leverages the strengths of generative models while supporting multiple tasks (commonly seen in video domains). The experiments are solid and comprehensive.

### Strengths
1. The chosen problem scope is broad, showing potential for ECHO to serve as a foundation model for PDE solving.
2. Extensive experiments are conducted, which appear quite thorough. The experiments at 1024-resolution are particularly impressive.
3. The two-stage encoding design is well-motivated, as standard VAEs fail to compress long trajectories at high resolution effectively.

### Weaknesses
1. Regarding the latent grid mapping: how does the use of continuous convolution differ from constructing a nearest-neighbor graph and applying a GNN? A clarification or comparison would strengthen the methodological justification.
2. Is the assumption of a uniform latent grid always reasonable? For example, if input points are distributed on a 3D spherical surface, a uniform grid would significantly increase the number of points to process. How does the method avoid unnecessary computations for points inside the sphere?
3. Lines 245–246: The concatenation of noise and conditions which have same shape as noise is a natural and commonly used conditioning approach, while AdaLN is relatively less common for such condition types. Presenting this as a contribution may be overstated.
4. Lines 252–254: The embedding of PDE parameters has already been discussed in UniSolver [1]. The authors should appropriately acknowledge this prior work.
5. Could the authors provide more details on the three-stage training? For example, what batch size was used, and what was the GPU memory used for every model?
6. Sampling Process: The details and rationale behind the sampling strategy are unclear. How does ECHO perform sampling (e.g., the number of sampling steps, sampler type)? What ablation studies were conducted to determine and validate this sampling strategy? The lack of this information makes it difficult to assess the efficiency and optimality of the generative process.
7. DrivAerNet++ is a relevant and challenging dataset with more realistic flow simulations around vehicles. Experiments on this dataset could further validate the method's effectiveness and generalization.

[1] Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers

### Questions
See Weaknesses

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
4

### Summary
The paper proposes a generative latent transformer for spatio-temporal PDEs that encodes irregular meshes into a hierarchical spatio-temporal latent grid and then uses flow-matching to generate trajectories (rather than standard temporal autoregressive rollouts). This design makes global attention tractable at million-point scales, reduces long-horizon drift, and supports conditional tasks (forward prediction, interpolation, inverse/temporal conditioning). Experiments on several PDE benchmarks show the competitive performance of the model.

### Strengths
1. The paper is clearly written, which makes the method and implementation easy to follow.

2. Consistently lower MSE compared against other models across diverse PDE benchmarks that contains relatively large-scale problem.

3. The paper indicates that hierarchical convolutional yields lower reconstruction error at a fixed latent size than non-conv alternatives (graph, INR, and transformer-AE).

### Weaknesses
1. While the system is well-executed, many components are established (conv encoder/decoder, regular latent grid, DiT-style transformer, flow matching/latent diffusion). Related efforts already explore latent generative PDE solvers—e.g., [1] conv AE + structured latent grid with flow-matching DiT, [2] latent diffusion generating full trajectories at once, [3] autoregressive latent video diffusion. 

A clearer exposition of the main differences would strengthen the paper’s contribution

[1] Li, Zijie, Anthony Zhou, and Amir Barati Farimani. "Generative Latent Neural PDE Solver using Flow Matching."

 [2] Zhou, Anthony, et al. "Text2pde: Latent diffusion models for accessible physics simulation."

 [3] Nguyen, Tung, et al. "PhysiX: A Foundation Model for Physics Simulations."

2. Generating the entire trajectory at once works well on cases like cylinder flow is not surprising, but it’s unclear how robust this design is for more turbulent regimes and beyond the training horizon. The appendix’s shallow-water example is reassuring; it would be helpful to see a similar long-horizon generalization test on a more turbulence case like the 2D vorticity problem in the benchmark to confirm the effect. holds.

### Questions
Lower MSE can reflect over-smoothing/blur, especially in turbulent fields. Could the authors report energy spectra  alongside MSE to verify how well the small-scale content is preserved?

### Soundness
2

### Presentation
3

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
The model adopts an encoder-decoder architecture, enabling efficient solving of large-scale partial differential equations with millions of points. A specialized encoder is designed to map irregular grids in physical space to regular grids in latent space, thereby simplifying the downsampling process for large-scale point sets. At its core, the model employs a DiT-based generative approach. After training, it possesses multi-task capabilities, allowing it to perform forward and inverse solving along the temporal dimension as well as data interpolation tasks simultaneously.

### Strengths
1. It employs a compression-decompression architecture, supporting the solution of PDEs with millions of points. 
2. The designed encoder can map irregular grids in physical space to regular grids in latent space. 
3. Using a DiT-based generative approach, the trained model can simultaneously support multiple tasks such as forward solving, inverse solving, and interpolation. 
4. By generating complete trajectory segments, it effectively mitigates long-term error accumulation.

### Weaknesses
1. One of the contributions of the paper is its support for predictions with millions of spatial points. However, the compression process of spatial points in the encoder and decoder is not described in detail. Appendix C.2 mentions that the authors adopted structures similar to (Hagnberger et al., 2025), (Yu et al., 2023), and (Koupaï et al., 2025). So, what are the innovative aspects of the authors' work in spatiotemporal compression and decompression that distinguish it from these references?  
2. In Table 1, the spatial dimensions of all datasets are 2D. Neural operators supporting millions of spatial points are highly suitable for tasks involving large-scale 3D point clouds, and most real-world PDE problems are 3D. Therefore, exploring the applicability of this operator to irregular 3D point clouds would be highly meaningful.  
3. The algorithm can handle multiple tasks such as forward solving, inverse solving, and interpolation. However, the interpolation task lacks dedicated experimental support in Table 2 or Table 4.
4. In Section 2.1 and Appendix C.1, the forward, inverse, and interpolation tasks are defined along the temporal dimension. In Table 4, when X_te=20%, the model performs an interpolation task in the spatial dimension. How spatial interpolation is achieved lacks a clear explanation. 
5. Descriptions of certain tasks are insufficient. In Table 4, what do Time-Stepping (TS) and Reconstruction (Rec) refer to, and how are they specifically executed? 
6. The Related Works section should be included in the main body of the paper rather than in the appendix.

### Questions
1.The description of the Encoder and Decoder methods lacks clarity. The formulas and figures should be supplemented to better illustrate the process. The dimensionalities of variables and how they change throughout the process are not clear. The authors should provide a more detailed explanation in the main text or the appendix.
2.In lines 289-299, the spatial downsampling rate during training is stated to be between 0.5 and 1.0. However, in Table 4, the spatial sampling ratio is 20%, which is less than 50%. The reason for this inconsistency in the ranges should be clarified.
3.In Line 160, "ODE" likely refers to "Ordinary Differential Equation." The full term should be written out upon its first appearance.
4.For the "Acoustic Scattering Maze" dataset in Table 1, the corresponding reference citation is missing and should be added.
5.The references for the AROMAF and CALM-PDE methods are currently cited as arXiv preprints. Since they have now been formally published, the citations should be updated to their final, officially published versions. The citations for the other methods should also be checked for accuracy and completeness.

### Soundness
2

### Presentation
2

### Contribution
2
