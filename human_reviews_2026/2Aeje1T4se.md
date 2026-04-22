# Geometric Laplace Neural Operator

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Neural operators have emerged as powerful tools for learning mappings between function spaces, enabling efficient solutions to partial differential equations across varying inputs and domains. Despite the success, existing methods often struggle with non-periodic excitations, transient responses, and signals defined on irregular or non-Euclidean geometries. To address this, we propose a generalized operator learning framework based on a pole–residue decomposition enriched with exponential basis functions, enabling expressive modeling of aperiodic and decaying dynamics. Building on this formulation, we introduce the Geometric Laplace Neural Operator (GLNO), which embeds the Laplace spectral representation into the eigen-basis of the Laplace–Beltrami operator, extending operator learning to arbitrary Riemannian manifolds without requiring periodicity or uniform grids. We further design a grid-invariant network architecture (GLNONet) that realizes GLNO in practice. Extensive experiments on PDEs/ODEs and real-world datasets demonstrate our robust performance over other state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes GLNO, which introduces a pole–residue representation and learnable exponential bases in the Laplace–Beltrami spectral domain to achieve unified modeling of non-periodic transient signals and operators on arbitrary geometric manifolds. The main contributions lie in establishing a unified spectral framework for geometric frequencies, extending neural operators to non-Euclidean domains, and significantly improving the accuracy and efficiency of geometric modeling and PDE solving.

### Strengths
Innovative theoretical framework: Introduces the Generalized Laplace Basis and a new pole–residue formulation, enabling neural operators to handle diverse and challenging dynamic scenarios.
Extensive experimental validation: GLNO consistently outperforms baselines on a wide range of ODE/PDE and real-world geometric tasks, including pendulum, reaction–diffusion, Poisson, and RNA/human-body surface datasets.

### Weaknesses
The paper suffers from several weaknesses: its theoretical rigor is insufficient, with incomplete mathematical justification and somewhat confusing notation; the experiments are limited in scope and lack thorough ablation or comparative analysis; interpretability is weak, and the connection between theory and empirical results is not clearly demonstrated. In addition, the paper contains spelling and grammatical errors, vague expressions, and inconsistent mathematical formatting, which collectively hinder readability and clarity.

### Questions
1. In Sec. 5, line 354, “We use GLNO to perdict...” — should this be corrected to “predict”?
2. In Fig. 4, line 449, “GLNO underpreforms benchmark model” — should this be “underperforms”?
3. In Sec. 5.2, line 466, “Euqation 16” — should this be corrected to “Equation 16”?
4. In Fig. 4, line 449, the phrase “shape-net car is displaced in prediction Error” — should “displaced” be “displayed”?
5. In line 427, “Despite no strictly definition...” — should “strictly” be replaced by “strict”?
6. In Sec. 5, line 313, Eq. (28) is intended to define the NLL loss, but the left-hand side of the formula is labeled as L₂. Is this a typographical or conceptual mistake?
7. In Table 1, under the Pendulum (c=0) column, it seems the best-performing result is not bolded—was this an oversight?
8. In line 354, the text states “As shown in Table 1, GLNO achieves a relative L₂ error of 0.9373, demonstrating superior capability...”. However, in Table 1, FNO (0.8447) and LNO (0.9512) both report lower errors than GLNO. Could the authors clarify this inconsistency?
9. In lines 207–209, the text mentions “where  \alpha_{ik} can be computed via...”, but  \alpha_{ik} does not appear in Eq. (17). Could the authors clarify where this term comes from?
10. Could the authors provide a detailed derivation for the final equality in Eq. (13)? The multiplication should theoretically yield a double summation, but the equation is written as two separate fractional summations, which seems valid only under specific assumptions.
11. In Eq. (20), why does the exponential term appear as e^{-\simga{i}t} with a positive sign? 
12. In Eqs. (14) and (23), K_{\phi} is evaluated at -s_i and +s_i, respectively. Could the authors explain the reason for this sign inconsistency?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose the Geometric Laplace Neural Operator (GLNO) and its practical realization, GLNONet, for extending Laplacian operator learning to arbitrary Riemannian manifolds. The authors demonstrate improvements over prior works for experiments on uniform grids, irregular grids, and arbitrary manifolds.

### Strengths
The paper contains a detailed mathematical formulation of the proposed method, and the authors performed experiments across a diverse set of problems, which I appreciate. I also appreciate that the authors provided timing and parametric complexity metrics to support the downsides of transformer and graph-based approaches that they argue in Section 2.

### Weaknesses
The authors write that they want to address limitations of existing spectral operators (page 3). However, there have recently been many neural operator architectures that are not explicitly spectral, such as attention-based and convolution-based neural operators. The authors should clarify the benefits of spectral operators compared to these other types that have been explored in the literature.

I would also recommend that the authors add some more challenging experimental problems, particularly some from an existing operator learning benchmark. This would alleviate the burden from the authors of having to perform and tune their own baseline models. This would also enable comparisons with different methods. Another issue I see right now is that the authors are comparing against only a handful of other neural operator methods, such as FNO and GNOT. How did the authors select this subset of methods to compare with? For instance, there are also other methods that operate on unstructured grids (e.g., GINO from “Geometry-Informed Neural Operator for Large-Scale 3D PDEs,” 2023). I would recommend the authors compare with a greater number of baselines.

I would also recommend the authors include some more mathematical background in the appendix for interested readers. Given the nature of this conference as a machine learning venue, not all readers may have the necessary theoretical background to parse through the paper in its current form.

**Minor notes:**
Capitalize “Fourier” — line 107.

### Questions
1. Am I correct in interpreting that GLNO performs worse than FNO in the c = 0 pendulum task? Any ideas why all of the models are doing quite poorly in relative error there (please correct me if I am misunderstanding the table)?
2. How were the baseline models tuned for these experiments?

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
4

### Summary
This paper presented Geometric Laplace neural operator by extending the previously proposed Laplace neural operator. The primary advantage resides in the fact that it can better handle non-periodic inputs, transient response, and irregular domain (non-Euclidean). Notably, the authors illustrate its application on both cannonical and real-world examples.

### Strengths
The paper is well written and it also illustrates the algorithm on real world dataset.

### Weaknesses
Despite the strength, the contribution is somewhat incremental (using generalized laplace transform). It would strengthen the paper if theoretical analysis can also be included, which right now is missing.

The bechamarking is not exhaustive. For example, I was expecting comprison with respect to Geo-FNO, GNO, Sp2GNO, and GINO at the very least as all of these handle non-Euclidean domain. The fact that graph based approaches are slower is well taken, but it does has its advantages and hence, comparing with the same is important.

Also, previous work wavelet neural operator and MWT, which actually addresses that same problem (At least the first two problems associated with non-periodic input and transient response), is absent. Comparison with WNO and MWT will also help, atl east for those on regular grid.

### Questions
The suggestions are already provided in weakness section.

### Soundness
3

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
5

### Summary
The paper introduces the Geometric Laplace Neural Operator (GLNO), a neural operator framework that models both periodic and aperiodic (decaying/growing) dynamics via a generalized Laplace basis with learnable exponential components. It extends the pole–residue formulation to Riemannian manifolds by operating in the Laplace–Beltrami eigen-basis and implements a grid-invariant architecture (GLNONet) that works across irregular meshes without retraining or resampling. Across ODE/PDE benchmarks and real-world geometric tasks, GLNO shows performance.

### Strengths
1. The paper addresses an important problem related to Fourier transform-based neural operators, and the authors propose a very principled approach to tackle it.

2. The technique's ability to generalize to different geometries is a strong point.

3. The authors go beyond simply solving partial differential equations (PDEs) and demonstrate the performance on real-world data.

### Weaknesses
### Write Up
I believe the write-up could be better organized. Some sections need consolidation, while others require more elaboration. For instance, Section 4.1 can be omitted, allowing the authors to directly introduce the generalized geometric Laplace basis. Additionally, the authors should expand upon the sections between lines 225 and 252, as I currently do not understand the message being conveyed in those line.

### Baselines
Two important baselines were overlooked. The following works are significant and should be compared, as they also address the same problem as the authors of this paper:
1. Fourier Neural Operator with Learned Deformations for PDEs on General Geometries
2. Geometry-Informed Neural Operator for Large-Scale 3D PDEs

### Technique
1. The time complexity of the proposed approach is not discussed. I believe the proposed decomposition requires more computation compared to standard FNO-based neural operators, and this should be addressed with appropriate discussion and experimental analysis.

2. Regarding the new basis introduced:
   a) The authors do not analyze the properties of the basis (Equation 16); for example, are they orthonormal?
   b) If they are not orthonormal, are they complete?
   c) Will the operation shown in Equation (22) still be a convolution between the input and the learned kernel function?

All of these points require thorough analysis in the paper.

### Visualization and Experiment Details
1. There needs to be more visualization of the results.
2. Please list the different geometries considered for the Poisson equation

Minor Typo: Capitalize "Fourier" in line 107

However, once these points are addressed, I believe the work is suitable for publication. I also believe these changes can be made during the rebuttal period.

### Questions
1. How are the eigenfunctions of LBO operators calculated for an arbitrary mesh or point cloud? How do you handle situations when the model receives input with a denser or different mesh? Since, as a neural operator, the model should be able to accept inputs at different discretizations, an experimental demonstration of this is necessary, i.e.,
 a. How does the model perform when the input at test time is on a different mesh? The test mesh might have fewer or more faces (or vertices).

 b. How does the model handle a dynamic mesh? For example, in the case of Lagrangian fluid simulation (for example [1]) or the analysis of a deformable object, where the mesh changes. How does the proposed model handle these cases? If it can not, please discuss this as a weakness.


[1] Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs

### Soundness
3

### Presentation
2

### Contribution
3
