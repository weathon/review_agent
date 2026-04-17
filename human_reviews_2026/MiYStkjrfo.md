# PIBNet: a Physics-Inspired Boundary Network for Multiple Scattering Simulations

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The boundary element method (BEM) provides an efficient numerical framework for solving multiple scattering problems in unbounded homogeneous domains, since it reduces the discretization to the domain boundaries, thereby condensing the computational complexity. 
The procedure first consists in determining the solution trace on the boundaries of the domain by solving a boundary integral equation, after which the volumetric solution can be recovered at low computational cost with a boundary integral representation.
As the first step of the BEM represents the main computational bottleneck, we introduce PIBNet, a learning-based approach designed to approximate the solution trace. The method leverages a physics-inspired graph-based strategy to model obstacles and their long-range interactions efficiently.
Then, we introduce a novel multiscale graph neural network architecture for simulating the multiple scattering.
To train and evaluate our network, we present a benchmark consisting of several datasets of different types of multiple scattering problems. 
The results indicate that our approach not only surpasses existing state-of-the-art learning-based methods on the considered tasks but also exhibits superior generalization to settings with an increased number of obstacles.
Code available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes PIBNet, a method that approximates the boundary solution of the multiple scattering problem. The approach is based on the boundary element method, which solves the boundary integral equation and then computes physical values in the volumetric domain. The authors investigate a UNet-like architecture for GNNs that uses downsampling and upsampling of vertices and edges. The numerical experiments show that the proposed model achieves the highest accuracy among the considered baselines. The authors also investigated generalization regarding the number of obstacles and ablation, implying the effectiveness of the proposed approach.

### Strengths
1. The experimental results show the high accuracy compared with the considered baselines, although the comparison is not complete (See Weakness 5).

### Weaknesses
1. The problem setting is unclear. The authors cite the reference for the multiple scattering problem, but it should be clearly formulated in the paper, as it is a central problem addressed in the work.
2. The difficulty of dealing with multiple obstacles is unclear. Since GNNs can handle arbitrary domains, there is no fundamental difficulty with multiple obstacles. There may be marginal difficulty arising from the increased complexity of the boundary, but this is not clearly stated in the paper. If the difficulty of the multiple obstacles is different from that of a complex-shaped single obstacle, it should be stated clearly, too.
3. The novelty of the proposed approach is not clear. The paper discusses upsampling and downsampling of the graph, but the critical difference from existing research (e.g., MGKN [Li+ NeurIPS 2020]) remains unclear. The authors should clarify the shortcomings of the past approaches and the benefits of the proposed method.
4. The presentation of the proposed method is not clear enough. The paper says “$n_c$ candidate edges are proposed,” but does not explain how. Also, the rationale for calling the edge selection strategy “physics-inspired” is unclear. In addition, the explanation of how to predict physical values in the interior is missing —an inevitable part of the prediction process. The authors should elaborate more on their method.
5. The experimental evaluation of the method is incomplete. Since the method is based on BEM, the authors should include BEM-based methods in the baseline (e.g.,  BINN [Sun+ Comput. Methods Appl. Mech. Eng. 2023] and PIBI-Nets [Nagy-Huber and Roth J. Comput. Phys. 2023]). To demonstrate the effectiveness of the proposed graph up- and down-sampling approach, a comparison with graph-UNet approaches (e.g., MGKN [Li+ NeurIPS 2020] and Graph U-Nets [Gao and Ji ICML 2019]) should be provided. In addition, there is no baseline for classical solvers, making the practical contribution of the work unclear. Since machine learning methods involve some error, the authors should compare speed and accuracy across varying convergence thresholds and mesh resolutions with classical solvers.
6. The analysis of the generalization experiments is weak. The proposed method has the lowest error across all obstacle settings among the baselines, but the reason the error increases with the number of obstacles is not analyzed in depth. Since the proposed method also increases error, it is not a groundbreaking technology for the multiple scattering problem, whereas the method is dedicated to that problem. The authors should clarify the cause of the increase in error and the key breakthrough of the proposed method for the problem at hand.


Minor points:

* The domain of obstacles $\Omega$ should be a closed set because the domain for PDE $\mathbb{R}^3 \setminus \Omega$ should be open to have well-defined differentiation everywhere.
* Table 2: Some decimal points are written as commas instead of periods. These points should be written using periods for standard scientific English.

### Questions
1. Since the method is based on the boundary element method, the reviewer assumes the applicable domain is limited to linear or semilinear PDEs; is this correct? If so, it should be clearly written in the paper as a limitation.
2. The method considers only the Dirichlet boundary condition. Can it be applied to Neumann boundary and mixed boundary problems? If not, the authors should clarify the limitations of the boundary treatment.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, PIBNet is proposed to approximate the solution trace on the boundaries for solving multiple scattering problems with boundary integral. More specifically, a multi-scale graph is used to represent the obstacles with an octree. A UNet-like graph neural network is used to construct the surrogate solver. Experiments show the proposed method outperforms previous methods including MeshGraphNet, PTv3, Transolver (++), and Erwin, on the constructed datasets of exterior Laplace and Helmholtz problems.

### Strengths
- This work addresses neural boundary element method with multiple disjoint obstacles, which is an important and challenging task.
- The technical and experimental parts are very solid.
- The ablation study is very detailed and clear.

### Weaknesses
- The "Physics-Inspired" part of PIBNet is too weak, which is simply to choose the shortest edge among the candidates.
- The neural network design is very similar to previous UNet-like meshgraphnet papers.

Minor Issues:
- I think there is a typo in the title of Section 3.1 "LEARNING AND BOUNDARY ELEMENT METHODS"

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
PIBNet is a novel machine learning method designed to accelerate the computationally intensive Boundary Element Method (BEM) used for solving scattering problems governed by PDEs like the Helmholtz equation. Its core innovation is replacing the slow iterative numerical solution for the boundary trace. It achieves this by utilizing a Physics-Informed Neural Network (PINN) approach where the loss function is specifically defined to minimize the residual of the Boundary Integral Equation (BIE), thus embedding the physics directly into the model's training. This approach yields significant inference speedups, demonstrating factors up to $200\times$ faster than traditional BEM while maintaining accurate results.

### Strengths
1. Technically robust framework validated across three key PDEs (Helmholtz/Laplace), demonstrating high-fidelity prediction and a substantial inference time speedup of up to $200\times$.
2. It replaces the GMRES solver bottleneck in BEM by minimizing the Boundary Integral Equation (BIE) residual directly, creating a novel, physics-constrained acceleration method.

### Weaknesses
1. Generalization to Out-of-Distribution Inputs, different shapes not present in the training distribution (e.g., highly concave, asymmetric, or multiply connected domains).
2. Comparison to State-of-the-Art Accelerated BEM: The comparison is made against the standard, non-accelerated BEM (solving the dense BIE matrix). The authors should include a quantitative comparison against the gold standard for large-scale BEM: the Fast Multipole Method (FMM)-accelerated BEM. 
3. Overall Cost Justification (Training Data Expense): The paper rightly focuses on the inference speedup, but this gain is predicated on the initial investment of generating a large dataset by performing thousands of expensive, full BEM solves (the ground truth). The authors should quantify the total computational cost (data generation + network training)

### Questions
q1: The speedup of PIBNet relies on replacing the iterative GMRES solve. However, the BIE loss function still requires repeated evaluation of the Boundary Integral Operators, please clarify the computational cost of this BIE residual evaluation. Is this calculated analytically or numerically, and how does its complexity scale with the number of boundary elements ?

q2: The BEM for the Helmholtz equation is known to suffer from the non-uniqueness issue at interior resonant frequencies (the "fictitious frequencies" problem). Does PIBNet, by training on the standard BIE formulation, inherit this instability?

q3:  The paper demonstrates generalization across shapes and wavenumbers within the training distribution. Could the authors provide a more detailed analysis on the model's interpolation performance?

### Soundness
2

### Presentation
2

### Contribution
2
