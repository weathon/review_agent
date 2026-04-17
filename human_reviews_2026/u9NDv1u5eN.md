# PETIMOT: A Novel Framework for Inferring Protein Motions from Sparse Data Using SE(3)-Equivariant Graph Neural Networks

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Proteins move and deform to ensure their biological functions. Despite significant progress in protein structure prediction, approximating conformational ensembles at physiological conditions remains a fundamental open problem. This paper presents a novel perspective on the problem by directly targeting continuous compact representations of protein motions inferred from sparse experimental observations. We develop a task-specific loss function enforcing data symmetries, including scaling and permutation operations. Our method PETIMOT (Protein sEquence and sTructure-based Inference of MOTions) leverages transfer learning from pre-trained protein language models through an SE(3)-equivariant graph neural network. When trained and evaluated on the Protein Data Bank, PETIMOT shows superior performance in time and accuracy, capturing protein dynamics, particularly large/slow conformational changes, compared to state-of-the-art diffusion and flow-matching approaches, as well as traditional physics-based models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes PETIMOT, a method to learn protein motion subspaces from sparse experimental structures using an SE(3)-equivariant GNN and embeddings from pretrained protein language models. The approach defines new subspace comparison losses (LS, SS, IS) and reports better accuracy than diffusion or physics-based models on PDB-derived datasets. However, the paper does not provide any code or data for reproducibility, which violates ICLR policy.

### Strengths
(1)	Addresses an interesting biological problem: modeling conformational flexibility from limited experimental data.
(2)	Combines structure-aware equivariant GNNs with protein language model embeddings in a creative way.
(3)	The proposed loss functions are mathematically clean and symmetry-aware.

### Weaknesses
(1)	No code or data are provided.
(2)	The evaluation is limited to PDB static structures, with no validation on dynamic or time-resolved data.
(3)	Theoretical explanations and derivations are shallow, and ablation studies are missing.
(4)	Possible redundancy or data leakage among homologous structures in the benchmark.

### Questions
(1)	Why wasn’t the code submitted at review time?
(2)	How do the authors ensure fair dataset splits without sequence redundancy?
(3)	Do the predicted motion subspaces correspond to known biological motions (e.g., open–closed transitions)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PETIMOT, a novel framework for inferring protein conformers from sparse experimental data in the PDB using SE(3)-equivariant graph neural networks. The proposed approach shifts focus from sampling full conformational distributions to learning compact linear subspaces of motions via a custom symmetry-aware loss function.

### Strengths
1.	The paper provides a novel formulation for the protein conformer sampling task. Given the practical importance of the task, the novel formulation is of significant value.
2.	The proposed method does not require any simulation data or physics-based guidance, being able to train utilising only sparse experimental data. The dependency on the simulated data is the severe practical limitation of existing approaches.
3.	The paper is well-structured and well-written.

### Weaknesses
1.	The need to introduce a novel SE(3)-equivariant architecture is unclear, given a wide range of existing equivariant graph neural networks, which could likely be adapted for this task without modifications or with minor modifications.
2.	Long-range dependencies are still approximated by random subsampling, which may fail to capture coordinated motions in large proteins; also, the process for selecting random residues lacks detail.
3.	Linear subspaces provide only limited expressibility in covering conformational ensembles, which potentially hinders the application of the method to families of proteins with highly non-linear dynamics, such as intrinsically disordered proteins or those undergoing allosteric transitions with flexible loops. At the same time, authors openly mention this limitation in the main text.
4.	Three main evaluation metrics are clear, but they seem very specific to the proposed method. Other methods like BioEmu use metrics such as relative free energy errors, flexibility correlation, and distributional similarity, while AlphaFlow includes precision, recall, and diversity via pairwise RMSD or lDDT.

### Questions
1.	I suggest authors firstly address weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors propose a new approach for inferring conformations of proteins as linear motions by utilizing a covariance matrix obtained after clustering pdbs by sequence similarity and extracting principle components from it. The work also proposes various loss terms to be utilized during training and evaluation.

### Strengths
1. Work presents an interesting approach for interpreting principle components of the defined covariance matrix as linear motions, the paper claims the conformational heterogeneity of proteins can be almost fully explained by these linear motions. Modeling protein conformations has important applications for drug discovery and enzyme engineering so this is an important research area and speed up in inferring conformational ensembles can be helpful in protein design pipelines. 
2.  Table 1 shows the improved performance of PETIMOT compared to AlphaFlow on a test set.

### Weaknesses
There are some questions and benchmarks lacking to fully evaluate the novelty of the contribution [see below]

### Questions
-  R in equation 2 - Wouldn’t the size of R in this case be m x 3N not 3N x m since W is 3Nx3N 
- Authors should show the linear motions mapped onto the pdbs where the PETIMOT performed poorly compared to AlphaFlow or NMA as an example (6JNA and 2HCB) and similarly for PDBs mentioned under the biological relevance to visualize the type of conformations PETIMOT does poorly on. 
- The paper mentions comparing to ATLAS MD as an independent test, however its missing benchmarks beyond the values of the losses defined in the paper, authors should consider metrics like RMSD, RMSF used by AlphaFlow for comparing their generated conformation to the MD ensemble. Almost all of the results use the loss terms proposed by the paper, authors should consider metrics more commonly used by the protein modeling community like above to evaluate the true performance of the method. 
- This is mentioned in the text of the paper, but for readability authors should consider including a table showing the inference time speedups. 
- Bioemu was mentioned earlier in the paper when talking about related work. Why wasn't that included as a method to benchmark against as well ?

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
This paper proposes a model of the different conformational configurations of proteins in the protein database. The model is based on a linearization of this subspace of possible conformations configurations that is based on PCA of the covariance matrix of conformations. The vectors that span this subspace of the PCA are estimated by approximately solving a linear least squares assignment problem.

### Strengths
Learning representations of the subspace of conformations of a protein is an important topic in proteon science.

### Weaknesses
The presentation of the work is not very clear. The problem is presented as a linear lead squared assignment problem (eq. 7). However it is not clear how the later described model architecture (Fig. 1) and the protein language model (line 276) relate to this task. The presentation of the paper and how the different mentioned aspects relate to each other needs to be improved.

Although I want to support protein science and diversity of applications of machine learning, it appears to be a very protein related topic with perhaps a limited audience in the ICLR community.

### Questions
How is the model described in section 4 used to solve the task described in section 3?

### Soundness
1

### Presentation
1

### Contribution
1
