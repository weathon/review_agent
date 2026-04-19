# MOFFlow: Flow Matching for Structure Prediction of Metal-Organic Frameworks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Metal-organic frameworks (MOFs) are a class of crystalline materials with promising applications in many areas such as carbon capture and drug delivery. In this work, we introduce MOFFlow, the first deep generative model tailored for MOF structure prediction. Existing approaches, including ab initio calculations and even deep generative models, struggle with the complexity of MOF structures due to the large number of atoms in the unit cells. To address this limitation, we propose a novel Riemannian flow matching framework that reduces the dimensionality of the problem by treating the metal nodes and organic linkers as rigid bodies, capitalizing on the inherent modularity of MOFs. By operating in the $SE(3)$ space, MOFFlow effectively captures the roto-translational dynamics of these rigid components in a scalable way. Our experiment demonstrates that MOFFlow accurately predicts MOF structures containing several hundred atoms, significantly outperforming conventional methods and state-of-the-art machine learning baselines while being much faster. Code available at https://github.com/nayoung10/MOFFlow.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes MOFFlow, a flow matching framework for the structure prediction of Metal Organic Frameworks (MOFs). The approach decomposes MOFs into blocks and jointly generates the entire lattice, along with the rotation and translation of each block, using flow matching on different Riemann manifolds. The results demonstrate the superiority of the proposed hierarchical flow matching framework over previous baselines.

### Strengths
1. The paper addresses the relatively novel problem of MOF generation and is the first to design an end-to-end framework for jointly generating both the lattice and block-level translations and rotations.
2. The authors construct benchmarks with easily calculable metrics for evaluation, and the results clearly showcase the model's effectiveness.

### Weaknesses
1. The discussion on periodicity in MOFs is insufficient. Periodicity is a crucial feature in crystallography. Apart from explicitly modelling the lattice parameters, previous works also adopt multi-edge connections [1] or the fractional coordinate system [2] to address this problem. Additional details on modeling periodicity would enhance the paper. For example:
- In Line 235-237, the authors mention the mean-free system to maintain translation invariance. However, constructing mean-free or center-of-mass (CoM) free systems for periodic boundary conditions (PBCs) differs from non-periodic systems [3] and warrants further discussion.
- The backbone model is based on a Cartesian coordinate system, which necessitates careful design of edge connections to capture intra-cell interactions.

2. The design of the backbone model lacks clarity. It is unclear whether the atom-level and block-level modules are updated sequentially or alternatively. Including a figure of the overall architecture would be beneficial.

[1] Fu, Xiang, et al. "MOFDiff: Coarse-grained Diffusion for Metal-Organic Framework Design."

[2] Jiao, Rui, et al. "Crystal structure prediction by joint equivariant diffusion." 

[3] Lin, Peijia, et al. "Equivariant Diffusion for Crystal Structure Prediction."

### Questions
1. How many different types of building blocks can be decomposed from the dataset?
2. By generating only translations and rotations instead of fine-grained structures, the proposed method assume building blocks as rigid bodies. Could this assumption potentially limit the model's performance?

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
The paper presents MOFFLOW, a deep generative model aimed at predicting the structures of metal-organic frameworks (MOFs), leveraging a novel Riemannian flow matching framework. This approach models the building blocks of MOFs as rigid bodies, significantly reducing the complexity of structure prediction. MOFFLOW operates in the SE(3) space, effectively capturing the roto-translational dynamics of the metal nodes and organic linkers. The authors claim that MOFFLOW outperforms both conventional crystal structure prediction (CSP) methods and state-of-the-art deep learning models in terms of accuracy and scalability, handling unit cells with up to thousands of atoms.

### Strengths
1. The use of a Riemannian flow matching framework to handle the inherent complexity and modularity of MOFs is a novel aspect of the work. This approach allows MOFFLOW to reduce the dimensionality of the problem and improve scalability.
2. MOFFLOW's ability to handle large structures with thousands of atoms and maintain high accuracy is a significant advantage, setting it apart from conventional methods that struggle beyond smaller systems.
3. The authors provide detailed information on hyperparameters, training setup, and make their code and model checkpoints available, enhancing the paper's transparency and reproducibility.

### Weaknesses
1. While the paper compares MOFFLOW against CSP methods and the DiffCSP model, the absence of comparisons with more recent and potentially relevant MOF-specific deep learning models could limit the assessment of state-of-the-art competitiveness.
2.  Although MOFFLOW incorporates multiple technical innovations (e.g., block-level embedding updates, MOFAttention), the impact of these individual components is not deeply dissected through ablation studies. Such an analysis would strengthen the understanding of the contributions of each module.
3. While MOFFLOW is stated to be computationally efficient compared to baseline methods, a more detailed analysis of its training and inference costs relative to comparable models would help contextualize its practical deployment feasibility.
4. (Minor point) The paper mainly focuses on properties relevant to structure prediction accuracy (e.g., match rate and RMSE). It would be beneficial to explore how MOFFLOW's predictions perform in real-world settings that are essential for the practical use of MOFs.

### Questions
1. I would suggest the authors add necessary baselines for a fair comparison, which include but are not limited to CDVAE, MOFDiff, FlowMM, DiffCSP++, etc.
2. Providing insights into the relative contributions of different components (e.g., block-level update layers, MOFAttention module) would add clarity.
3. Including a detailed comparison of computational resources (e.g., GPU hours, memory usage) against other models would be valuable.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces MOFFlow, a frame-based flow matching approach to predict the structure of Metal Organic Frameworks (MOFs). MOFFlow treats the MOF building blocks (Metal nodes, organic links) as frames, and generates the 3D crystal structure by modeling the orientation and translation of the building blocks separately. A novel MOFAttention module is also introduced to model the hierarchical structure of MOFs. MOFFlow is benchmarked against DiffCSP, random structure search, and evolutionary algorithm baselines. The paper shows a significant improvement compared with DiffCSP in generating MOF structure, demonstrating the advantage of the hierarchical modeling approach in generating large MOF structures containing hundreds of atoms.

### Strengths
- The hierarchical modeling approach to separate the translation and orientation clearly demonstrates an advantage in predicting the structure of MOFs. The advantage is most clearly demonstrated in Fig. 6, indicating MOFFlow’s ability to scale to large structures with more than 200 atoms.
- The introduction of the MOFAttention module, inspired by the IPA module from AlphaFold 2, is interesting and novel. 
- The authors show an improvement over the self-assemply algorithm from the MOFDiff paper. This demonstrates an advantage of learned methods over heuristic based methods.

### Weaknesses
- The individual pieces of the paper are not particularly novel. The dataset, evaluation, and the MOF decomposition method are similar to the MOFDiff work. The flow matching approach is similar to FrameFlow. The PCA axis to denote the orientations of the building block is similar to Gao & Gunnemann (2022). 
- The paper lacks ablation study and/or more extensive benchmarking to illustrate the key component of their model that lead to the improvements. From DiffCSP to MOFFlow, there are many changes: 1) the adoption of a hierarchical modeling approach; 2) change from diffusion to flow matching; 3) change of the architecture of the score network. Which component leads to the biggest improvement? Although the likely answer is 1), more comparisons would greatly benefit the community.

### Questions
- How did the authors obtain the initial structure of the individual building blocks? Do they use the conformation from the data? Or predict those with a specific algorithm? 
- Have the authors studied the number of steps required for the reverse process in flow matching? 
- Have the authors studied the scalability of MOFFlow and the self-assembly algorithm used in MOFDiff, similar to Fig. 6? 
- I would like to see more details on how Fig. 5 is produced. Since MOFDiff and MOFFlow models different types of problems, how do the authors perform a fair comparison in Fig. 5?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors tackle the problem of crystal structure prediction (CSP) of Metal-Organic Frameworks. Typically, CSP is performed using ab initio calculations using density functional theory (DFT). However, DFT is computationally expensive and scales poorly for large complex systems such as MOFs. MOFFLow presents a MOF-specific generative model for CSP that operates on secondary building units rather than isolated atoms for structure prediction.

### Strengths
* Outperforms existing deep learning-based structure prediction models by a significant amount. 
* Highly scalable.

### Weaknesses
* Limited applicability on domains.
* Crystal structures that cannot be decomposed into secondary structures cannot readily use these techniques

### Questions
* Awkward notation in equation 1. L is 3x3. k is 3 x 1. Lk is therefore 3x1. x_n in line 143 is  1 x 3. So, x_n + Lk is not proper. 
* Line 156: building blocks contain a set of “local” coordinates. What is the reference frame used for these coordinates?
* Section 5.2: Are the property evaluations done on predicted structures and compared with property evaluations on ground-truth structures?

### Soundness
4

### Presentation
4

### Contribution
3
