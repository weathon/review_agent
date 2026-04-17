# Learning residue level protein dynamics with multiscale Gaussians

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
Many methods have been developed to predict static protein structures, however understanding the dynamics of protein structure is essential for elucidating biological function. While molecular dynamics (MD) simulations remain the in silico gold standard, its high computational cost limits scalability. We present DynaProt, a lightweight, SE(3)-invariant framework that predicts rich descriptors of protein dynamics directly from static structures. By casting the problem through the lens of multivariate Gaussians, DynaProt estimates dynamics at two complementary scales: (1) per-residue marginal anisotropy as 
 covariance matrices capturing local flexibility, and (2) joint scalar covariances encoding pairwise dynamic coupling across residues. From these dynamics outputs, DynaProt achieves high accuracy in predicting residue-level flexibility (RMSF) and, remarkably, enables reasonable reconstruction of the full covariance matrix for fast ensemble generation. Notably, it does so using orders of magnitude fewer parameters than prior methods. Our results highlight the potential of direct protein dynamics prediction as a scalable alternative to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a pipeline for training a model to predict protein dynamics.  The authors model protein dynamics as sampling conformations from a normal distribution with a low-energy mean and a non-identity covariance matrix.  The overall covariance is represented as a combination of a local residue-wise 3×3 covariance matrix and a global residue-wise N×N covariance matrix, matching the dynamics learned from MD simulations.

### Strengths
This paper addresses protein dynamics prediction not implicitly through ensemble generation, but directly by modeling the protein’s residue C$\alpha$ covariance matrix.  This approach enables direct analysis of protein dynamics.  
Modeling the full covariance matrix using a heuristic that combines the cross-residue covariance matrix with the intra-residue position matrix provides an efficient and meaningful representation of dynamics.  
The experiments presented are diverse and interesting, demonstrating the versatility of the proposed approach.

### Weaknesses
1. Throughout the paper, the examples presented focus on local dynamics, where the protein does not deviate significantly from the mean. Proteins that undergo large conformational changes may not be well represented by modeling their dynamics as a normal distribution with a single covariance matrix. For proteins that open and close or contain highly flexible regions, the full dynamic range might collapse into a single blurred representation under this parameterization. Nevertheless, this work represents an important step toward explicit modeling of protein dynamics.  

2. The authors assume that the provided protein structure is in a low-energy state, but there is no discussion of how the model behaves when the input structure is not in such a state.

### Questions
1. When sampling from the normal distribution of protein structures predicted by the model, how often do unphysical protein structures appear?  What prevents this approach from producing impossible torsion angles or bond lengths?  

2. How does the model handle proteins with pronounced conformational changes, such as large domain rotations, extensive motion cascades, or highly distinct alternate conformations like those observed in 4OLE?

3. How does the model behave when the provided protein is not in its lowest-energy state?  How would the model perform as part of a dynamical system, where a protein conformation is iteratively updated along the principal directions of each residue? Would the structure return toward the mean state, or would it break down entirely?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DYNAPROT, a novel, lightweight, SE(3)-invariant deep learning framework for directly predicting protein dynamics from static protein structures. Unlike molecular dynamics (MD) simulations, which demand substantial computational resources, or generative models requiring extensive pre-training and costly sampling procedures, DYNAPROT is an efficient alternative.

### Strengths
Compared to models such as DYNAPROT (with 2.86 million parameters in total) and AlphaFlow (95 million parameters), the parameter count is reduced by several orders of magnitude. Its inference and sampling speeds far exceed those of other methods, rendering it exceptionally practical. Moreover, the model performs well and is trained solely on molecular dynamics data from approximately 1,000 proteins.

### Weaknesses
The model was trained on the ATLAS dataset comprising 100 ns simulations. For numerous biologically significant processes—such as large-scale domain motions and allosteric transitions—100 ns is wholly inadequate, as these typically unfold over timescales ranging from microseconds to milliseconds or longer. Consequently, the model may have predominantly learned local, transient thermal fluctuations while exhibiting limited capacity to capture large-scale conformational changes. Although the paper's zero-shot testing results for BPTI appear favourable, this represents a single protein case study. One cannot conclude from this solitary example that the model generalises well to all long-timescale dynamics.

### Questions
The model was trained on 100ns of data. Beyond the single example of BPTI, have the authors tested the model on additional proteins exhibiting known long-timescale (e.g. > 1us) dynamic characteristics? Is there a possibility that the model might overfit to short-timescale fluctuations, thereby underestimating genuine large-scale motions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents DYNAPROT, an innovative and lightweight SE(3)-invariant model that predicts protein dynamical features—marginal anisotropy and residue-level couplings—directly from static structures. It achieves accuracy comparable to leading generative models without large-scale PDB pretraining, while being more efficient. By constructing covariance matrices and introducing scalar covariances, the model can capture some key features, which are useful for fast ensemble generation.

### Strengths
1. The paper is well organized and clearly stated.
2. The authors propose a protein dynamics framework based on a Gaussian hierarchical representation, which incorporates per-residue covariance tensors and the global coupling matrix into a Gaussian hierarchy.
3. Compared with a single scalar RMSF, the covariance matrix is richer: it captures both the shape and direction of fluctuations.

### Weaknesses
1. The core issue is model misspecification. A Gaussian hierarchy simplifies learning and sampling, and it can not reflect the thermodynamics of MD data. Molecular energy landscapes are multi-well with distinct basins. Gaussian models are locally valid only near a single basin (approximately harmonic vibrations) and cannot capture complex, anharmonic shapes or multimodality across multiple minima. This defect was masked in the atlas dataset. Proteins in Atlas are relatively large, so the existing MD trajectories predominantly capture local fluctuations rather than informative transitions. Could the authors validate the method on simpler systems—e.g., a 2D double-well potential and fast-folding proteins—to demonstrate effectiveness of model?
2. Although Eq. 5 and Proposition 3.1 guarantee positive semidefiniteness, the construction in Eq. 5 lacks clear physical motivation. The marginal covariance $\Sigma_{marginal}$ rightly emphasizes per-residue anisotropy, which is a strength. However, the coupling matrix $C$ enforces isotropic interactions—i.e., identical influence along all three spatial directions—between residues. This design oversimplifies residue–residue coupling and risks weakening key directional effects that are critical in complex biomolecular systems. Could the authors elaborate on the motivation and theoretical basis for this design choice?
3. When constructing $C$, the use of **mean pooling** as a scalar projection to compute each $C_{ij}$ appears casual and overly simplistic for modeling residue–residue coupling. A more completed ablation is needed to justify this design choice.
4. What is the $\mu$ of the BPTI results in Appendix A.3.1? Are the two main conformational changes caused by $\mu$ or by $\Sigma$?

### Questions
The same as weaknesses. In multi-conformation generation tasks, more attention is paid to the changes in system transitions, (rather than the local fluctuation). If the author is willing to explain this point, I am willing to increase the score

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
3

### Summary
DynaProt is a Gaussian-based framework for ultra-fast protein conformation ensemble generation. Unlike traditional generative models that learn protein structure distributions via sampling, DynaProt learns a hierarchical Gaussian parameterization composed of residue-level anisotropic Gaussians and a protein-level isotropic residue–residue covariance matrix that captures coupling between residues.
This formulation enables DynaProt to analytically compute dynamics-related quantities such as RMSF without sampling; reconstruct the full 3N×3N covariance matrix using simple heuristics to support conformation sampling, generate large conformational ensembles extremely efficiently by sampling from reconstructed Gaussian distribution.
Compared with classical normal mode analysis, which also relies on Gaussian approximations, DynaProt learns variances and couplings directly from data rather than through simplified physical potentials, resulting in more accurate ensemble statistics.
Experimental results show that DynaProt achieves competitive performance on some ensemble-level metrics compared to diffusion generative models while being orders of magnitude faster. Although some accuracy is sacrificed due to the inherent Gaussian assumptions, the method offers a compelling balance between speed, accuracy, and physical interpretability, making it a strong and distinctive solution for conformation sampling and protein dynamics analysis.

### Strengths
- The manuscript is well organized, clearly written, and easy to follow, with clean and informative figures that effectively support the main ideas.
- The model addresses a unique and practical perspective in conformational modeling, where not only accuracy but also the ability to efficiently generate large numbers of samples is essential. DynaProt improves upon classical normal mode analysis while being orders of magnitude faster than modern generative models.
- The method produces versatile outputs (e.g., RMSF, residue-level anisotropic Gaussian profiles, residue-residue coupling) and achieves competitive accuracy on some ensemble-level metrics compared to current state-of-the-art models.

### Weaknesses
- The core limitation lies in the Gaussian approximation and reliance on input structures, which restricts expressiveness compared to more flexible generative models - but this is aligned with the paper’s goal of trading some accuracy for major gains in speed and efficiency.
- Given this design philosophy, clearly demonstrating the cases where DynaProt is useful becomes important. Cryptic pocket modeling is a relevant application, but only one case study is provided.
- The current formulation models only coarse-grained residue-level dynamics. It would strengthen the paper if the authors could demonstrate potential extensions to atomic-resolution modeling or protein-ligand complexes.

### Questions
### 1. Relationship between DynaProt-M, DynaProt-J, and DynaProt

In the Experiments “preprocessing” section, DynaProt-M, DynaProt-J, and the IPA backbone appear to be trained as three separate models. However, in Section 4.4, the final DynaProt results are described as being composed from the output of DynaProt-M and  DynaProt-J. Could the authors clarify which is the case? And if separate models are trained, why is it necessary (vs. a single model with two prediction heads)?
    
### 2. Cryptic pocket case study and sampling
In Section 4.2.1, the authors show that predicted marginal variances can indicate ligand-binding dynamics, such as cryptic pocket formation. Have the authors sampled from the full reconstructed Gaussian distribution for this protein, and verified whether both open and closed conformations can actually be generated?
    
Additionally, BioEmu (https://www.science.org/doi/10.1126/science.adv9817) includes several benchmarks with cryptic pocket motions. Evaluating DynaProt on those systems and reporting pocket coverage metrics would further strengthen the results.
    
### 3. Extending to atomic-level or protein-ligand complexes
    
DynaProt demonstrates impressive speed and potential scalability for residue-level modeling. Is it feasible to extend this approach to atomic-level resolution or to explicitly model protein-ligand complexes? Are there technical challenges such as dimensionality of the covariance matrix or parameterizing atom-level couplings?

### Soundness
3

### Presentation
4

### Contribution
3
