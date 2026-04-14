# Reduced-Order Neural Operators: Learning Lagrangian Dynamics on Highly Sparse Graphs

- Decision: Reject
- Scores: 6, 3, 5, 6, 6

## Abstract
We propose accelerating the simulation of Lagrangian dynamics, such as fluid flows, granular flows, and elastoplasticity, with neural-operator-based reduced-order modeling. While full-order approaches simulate the physics of every particle within the system, incurring high computation time for dense inputs, we propose to simulate the physics on sparse graphs constructed by sampling from the spatially discretized system. Our discretization-invariant reduced-order framework trains on any spatial discretizations and computes temporal dynamics on any sparse sampling of these discretizations through neural operators. Our proposed approach is termed Graph Informed Optimized Reduced-Order Modeling or \textit{GIOROM}.  Through reduced order modeling, we ensure lower computation time by sparsifying the system by 6.6-32.0$\times$, while ensuring high-fidelity full-order inference via neural fields. We show that our model generalizes to a range of initial conditions, resolutions, and materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The article introduces Graph Informed Optimized Reduced-Order Modeling (GIOROM), a framework designed to accelerate simulations of Lagrangian dynamics—such as fluid flows, granular flows, and elastoplasticity—using neural-operator-based reduced-order modeling. GIOROM addresses this by simulating physics on highly sparse graphs sampled from the spatially discretized system, achieving a reduction in input size by 6.6 to 32 times compared to full-order models.

To capture local spatial features of the discretized input, the authors define a graph-based neural operator called the Interaction Operator, which performs two key tasks:

1. It uses a discretization-agnostic adaptation of message passing to model interactions between points, regardless of the discretization.

2. It leverages a Graph Neural Operator (GNO) layer to project features onto a regular grid, facilitating the construction of a latent space upon which the time-stepper operates.

Authors use following scheme: Discretization-agnostic MP ->  GNO ->  point-wise MLP -> Neural operator transformer -> GNO -> discretization-agnostic MP. This scheme predicts acceleration field in Q-point discretization which is used for computing velocity in t_(n+1) and deformation in t_(n+1) (using explicit Euler scheme) and after that combined loss is calculated between ground truth derformations and velocities (in this discretization). Neural field is trained using reconstruction loss and used for full-order inference (to P-point discretization).Developed approach generalizes well across different initial conditions, velocities, discretizations and geometries.

### Strengths
1. Developed approach generalizes well across different initial conditions, velocities, discretizations and geometries.

2.  The paper includes a thorough ablation study, which evaluates the impact of different components of the model. This study helps in understanding how each part contributes to the overall performance, thereby validating the effectiveness of the proposed methods.

3.The authors provide strong scientific justifications for their approach, supported by rigorous experiments.

### Weaknesses
1. Only explicit time integrator is considered. Please add motivation for this choice to paper.
2. Sometimes units in tables are not presented.

### Questions
1. Please clarify units of duration for Table 1.
2. Please add information about numerical scheme used in nclaw. Is it the same scheme used for training model?
3. What about stability issues of numerical scheme used? How the time step is chosen?
4. How does the developed approach compare with implicit numerical schemes in terms of stability and time-stepping requirements?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces Graph Informed Optimized Reduced-Order Modeling, a neural-operator-based framework designed to accelerate simulations of complex Lagrangian dynamics, such as e.g. fluid flows. GIOROM intends to reduce computational costs by training on sparse graphs sampled from spatially discretized systems, allowing it to simulate temporal dynamics efficiently without depending on full spatial resolution.
The authors emphasize that the proposed framework is discretization-invariant, meaning it can generalize across various spatial discretizations and resolutions.  To achieve this, the framework uses a graph-based neural operator transformer to model temporal dynamics on sparse representations and leverages continuous reduced-order modeling with neural fields to reconstruct the full-order solution, enabling evaluation at any spatial point within the system.

### Strengths
This approach shows some innovation by integrating graph-based neural networks and continuous reduced-order techniques, distinguishing it from more conventional dense simulations and the results point out computational gains. This Acceleration while retaining high accuracy can benefit industries where large-scale physics simulations are essential, e.g. engineering.

### Weaknesses
The paper's novelty is either limited or not effectively communicated, as the work primarily appears to combine approaches from the cited prior studies [Li et al., 2024; Chen, 2024]. The rationale behind the specific design choices and the reasons for the approach's effectiveness are not clearly explained.

Significant revision is needed, particularly in Sections 3, 4 and 5.  Sections 3 and 4 introduce the methods used but fail to clarify how the proposed work differs from existing research or to explain how the components fit together cohesively. The introduction is challenging to follow, especially since it doesn’t clarify how this work advances beyond prior literature. A more logical approach could be to introduce the complete framework as depicted in Figure 1 and follow the flow of this diagram, highlighting any novel contributions.

While the experimental nature of the work in Section 5 and in the Appendix is valuable, the structure of the experimental section lacks a clear line. It consists of nine figures and tables with minimal explanation. To improve clarity, the authors should provide a more detailed rationale for each experiment, explaining the intentions behind the choices and the methodologies used. Tables and figures might benefit from rearrangement or relocation to the appendix if necessary.

Although the paper claims high fidelity across various initial conditions and resolutions, it does not address complex cases with extreme variability, such as materials undergoing phase transitions or highly turbulent flows. These more complex scenarios, which are generally more challenging to model with reduced-order methods, are not covered in the results presented.

The research in this work is quite interesting; however, I cannot recommend the paper for acceptance in its current form as major revision is needed.

### Questions
How does the model handle edge cases with high variability or extreme dynamics? It would strengthen the work to clarify how robust the model is for complex, highly dynamic systems, such as those with rapid phase changes or intense turbulence.

What are the limitations of sparse sampling in terms of capturing fine-grained details? Can you provide information of the trade-offs between computational efficiency and the accuracy of capturing  system details?

How does the model perform when interpolating or extrapolating to new materials or conditions not represented in the training data? It is mentioned that these cases are challenging but no empirical evidence is provided.

Can you provide a systematic study on the scaling performance with varying graph size for the same dynamical system?

Are there specific physics-based scenarios where this reduced-order approach might be less effective?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper develops a technique for simulating physical systems by combining graph-informed ROM and neural operators. First, the initial full-order mesh is coarse-grained to a sparser mesh using the farthest point sampling method. That smaller graph is encoded via an interaction network by performing several message passing, capturing local spatial interactions. Then, the encoded features are embedded into a regular grid, processed with a neural operator transformer and decoded with a second interaction network in order to perform the integration step. Each snapshot computed on the reduced space can be projected back to a full-order system by using a learnt linear basis transformation, which can be efficiently solved with least squares and evaluated at any arbitrary point. The method is applied to several examples in both solid and fluid mechanics.

### Strengths
* The method is very versatile, as it is able to handle different types of systems and material models in continuum mechanics problems. It has also very good results in generalization.

* The processing pipeline is discretization-agnostic, and the solution can be sampled in an arbitrary level of discretization.

* Solving the dynamics in the reduced space makes the algorithm very fast in inference time.

### Weaknesses
* The graph construction is computationally demanding.

* The method could be problematic with challenging geometries or singular phenomena, in which the farthest point sampling might omit relevant details of the domain.

* The pipeline is too complicated and has many hyperparameters, which might be a problem for learning larger-scale systems and more complicated physical behaviours.

### Questions
* It is very surprising that the method is very complicated on the encoding stage (going from full order to reduced order) but fairly simple on the decoder stage. In my opinion, the paper has not complitely justified why the use of a graph encoder + interaction network is better than other existing reduction methods. I would suggest the authors to perform an ablation study comparing the current graph encoder approach with a neural field encoding, so there is clear evidence that the use of a sparsified graph and an interaction network is better than a more simple approach. 

* Could the authors to provide a sensitivity analysis showing how performance and computational cost vary with different values of Q (number of reduced point discretization)? Which are the heuristics or guidelines used for selecting Q in practice? This would help readers understand the tradeoffs involved in selecting this parameter.

* Line 164: "This ensures an even distribution of points, [...]". This is an advantage from the computational perspective, but it might be problematic under certain conditions. For example, pressure discontinuities in fluid dynamics or stress concentration in solid mechanics. Those singular regions might not be correctly captured by just using a distance criterion in the farthest point algorithm, and might be advantegous a finer discretization in those areas. Can the authors provide with any discussion/limitations on this regard?

* Figure 2: The bunny and cow examples show self-contact between surfaces. Is there any specific condition to handle this situations from the graph generation perspective, and is it handled differently in solids and fluids? Could the authors discuss any limitations on this area?

* Figure 3: The figure is misleading. Why is the farthest point graph more dense than the full-order graph? I would expect something like the Delaunay Graph example.

* Table 3: Can the authors clarify what "randomizing the indices" mean and its significance in the reduction method?
My guess is that the authors refer to different ordering in snapshots and nodal indices. If that is the case, it is not obvious why that randomization should affect the final result in a relevant way. In fact, POD/PCA decomposition is constructed over the snapshot covariance matrix and it is invariant under any permutation transformation. 

* Table 3: Can the authors provide more details about the autoencoder baseline and the LiCROM projection, including parameter counts, training times and network topology? Could the authors include a more fair comparison with a closer method, such as its non-linear version (CROM)?

Final comment: This paper has very promising results and generalization capabilities, but the contributions are not well justified. The real contribution of the paper is the graph sparse reduction together with an interaction operator, as the use of neural operator transformers and LiCROM neural fields is not novel. In the current state of the paper, there is a lack of comparison with other state-of-the-art end-to-end reduction techniques. For this reason, my final rating is marginally below the acceptance threshold, but I would be open to raising it with enough justification of my mentioned points.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a neural operator approach to Lagrangian simulations. The core idea is to subsample the original point cloud, then transform it into/from a fixed-sized latent (grid) representation with Message Passing (MP) + Graph Neural Operator (GNO), and in the middle apply a transformer. This model is used to predict the acceleration at each node, which is then numerically integrated to evolve the system. While aligned with Graph Neural Networks (GNNs) for Lagrangian dynamics, the proposed method is discretization-independent and consistently speeds up simulations compared to GNNs.

### Strengths
- Extensive literature review. 
- Detailed analysis of graph subsampling and construction implementations.
- Many details and ablations of the proposed approach, including an extensive appendix.

### Weaknesses
- **W1: self-generated datasets**: Is there any reason why you didn't use existing 3D datasets, e.g., from Sanches-Gonzalez et al. (2020)? The only subset that is not there is a 3D multi-material one. Looking at the baseline results in your Table 4, GNS and the proposed model perform similarly. Comparability with the baseline numbers from the GNS paper would have been very useful.
- **W2: Farthest Point Sampling (FPS)**: You say on lines 351-352 that random sampling has a similar performance to FPS, but on L. 329, you say that you do use FPS after all. Having some experience with FPS, I know that it is a sequential process over the point cloud, with the number of iterations being the number of subsampled points. And this sequential nature becomes a bottleneck when working with large point clouds as the presented 3D ones. Why didn't you just use random sampling?
- **W3: POD and Autoencoders**: What do you mean by POD and an autoencoder in Table 3? Do you explain this somewhere? I don't think any of these two are representative baselines. In addition, Table 3 has multiple identical lines, all of which could be summarized in one sentence. Please consider removing this table altogether.
- **W4: DINo**: You cite the DINo paper (Yin et al., 2023) and a few similar ones, but I didn't see a discussion of how your approach improves on them. To my knowledge, you could have used the DINo encoder and decoder on the velocity field (and Euler-integrate once), and you would only have had to add an operator transformer in between. Am I missing something?

**Minor:**
- L. 310, "Ma et al. (2023)" -> "(Ma et al., 2023)"
- L. 412: please add some space between the captions of subplots (a) and (b). Currently, it is unclear that there are two subplots.
- Table 13: please reduce the font size of this table
- L. 1048: "our model is at least 5x faster than [GNNs]" doesn't match with the numbers in Tables 9 and 10. It is rather a 2-4x speedup. Please reformulate.

### Questions
- Which dataset do you use for Table 3?
- Table 3: Did you consider discussing "ROM baselines" (L.431) and "Discretization-agnostic ROM" (L.514) next to each other, so that Table 3 is close to both?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a learning-based reduced-order modeling framework for Lagrangian simulation. The proposed model comprises  message passing layer and operator transformer.  To reduce the computational complexity, the original discretized field is first down-sampled and then reconstructed via linear combination of neural field basis. Extensive numerical experiments on different Lagrangian simulation including fluids and sand are conducted to demonstrate the effectiveness of the proposed method.

### Strengths
1. Combining a continuous decoder with a neural operator in the latent space is technically sound. The numerical experiments showcase the proposed framework significantly reduces the computational cost while maintaining good accuracy. The continuous decoding strategy also makes it more flexible with different discretizations.

### Weaknesses
1. Many details about the experiments are vaguely described, which makes it difficult to interpret some of the results presented.
 For example, in equation 4, how is the coefficient q derived in practice? Is it predicted by another neural network, or is it optimized directly via least-squares? If it is optimized online, then during the inference stage, it will require extra optimization for every reconstruction from reduced mesh to full mesh.  In table 4, several baselines are listed but there is no formal definition or a brief introduction of them, for example I could not find the definition of what is IPOT. In table 3, what is the autoencoder? Is it a GNN or it's just a MLP? If it's a GNN then shouldn't it be permutation-equivariant?

2. As shown in Figure 1, the reduction part does not contain any learnable parts but rather just rely on non-learnable sampling method, which can potentially result in information loss.

3. (Relatively minor) The system considered in the numerical experiments are rather small-scale, all less than 100k nodes/particles.

### Questions
1. When applying FNO as part of the latent dynamics model, how do you handle the empty voxels/regions? 

2. The paper compares  with full-order model like GNS or other reduce-order model, how does the model compare to multi-grid model like Cao et al. [1]?

3. Is it necessary to do the sampling of full-order mesh at every timestep, and why not stay in the latent space with a fixed set of particles sampled in the beginning?


[1] Cao Y, Chai M, Li M, et al. Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN[J]. arXiv preprint arXiv:2210.02573, 2022.

### Soundness
3

### Presentation
2

### Contribution
2
