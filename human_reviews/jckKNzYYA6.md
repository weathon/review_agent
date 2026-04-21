# AssembleFlow: Rigid Flow Matching with Inertial Frames for Molecular Assembly

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Molecular assembly, where a cluster of rigid molecules aggregated into strongly correlated forms, is fundamental to determining the properties of materials. However, traditional numerical methods for simulating this process are computationally expensive, and existing generative models on material generation overlook the rigidity inherent in molecular structures, leading to unwanted distortions and invalid internal structures in molecules. To address this, we introduce AssembleFlow. AssembleFlow leverages inertial frames to establish reference coordinate systems at the molecular level for tracking the orientation and motion of molecules within the cluster. It further decomposes molecular $\text{SE}(3)$ transformations into translations in $\mathbb{R}^3$ and rotations in $\text{SO}(3)$, enabling explicit enforcement of both translational and rotational rigidity during each generation step within the flow matching framework. This decomposition also empowers distinct probability paths for each transformation group, effectively allowing for the separate learning of their velocity functions: the former, moving in Euclidean space, uses linear interpolation (LERP), while the latter, evolving in spherical space, employs spherical linear interpolation (SLERP) with a closed-form solution. Empirical validation on the benchmarking data COD-Cluster17 shows that AssembleFlow significantly outperforms six competitive deep learning baselines by at least 45\% in assembly matching scores while maintaining 100\% molecular integrity. Also, it matches the assembly performance of a widely used domain-specific simulation tool while reducing computational cost by 25-fold.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AssembleFlow, which models the assembly of molecular crystals by assuming a rigid body and operating in the SE(3) space. It learns vector fields on SO(3) and $\mathbb{R}^3$ separately, using SLERP and LERP, respectively. The reference coordinates systems are constructed with inertial frames.

### Strengths
- The paper is written clearly and is easy to understand.
- The experiment shows good performance.

### Weaknesses
A notable limitation of this work is the claim to be the first to implement rigid generation in SE(3) space for molecular assembly. This claim appears to overlook prior work [1], which closely aligns with the approach presented here, treating molecules as rigid bodies in molecular assembly, separately handling SO(3) and Euclidean space, and applying quaternion representation on SO(3). The differences are minimal, mainly in the methods for constructing rigid frames and in the choice of a normalizing flow over a CNF with flow matching.

Additionally, the paper emphasizes the decomposition of SE(3) into $\mathbb{R}^3$ and SO(3) as a unique strength (e.g., “this decomposition also empowers distinct probability paths…, effectively allowing for separate learning”). However, decomposing SE(3) into SO(3) and $\mathbb{R}^3$ is a widely adopted simplification in SE(3)-based methods ([1,2] among others). To my knowledge, there are no approaches that directly operate on SE(3) without this decomposition, which raises questions about why this common approach is presented as an innovative strength.

Given these points, I am hesitant to support this work for acceptance in its current form.

-----
[1] Rigid Body Flows for Sampling Molecular Crystal Structures, Köhler et al., ICML 2023  
[2] SE(3)-Stochastic Flow Matching for Protein Backbone Generation, Bose et al., ICLR 2024

### Questions
What advantage does the eigendecomposition of the inertial tensor $\hat{I}$ have over the common approach of using centered $X$?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces AssemblyFlow, a flow-matching based generative model to predict the packing of molecules into crystalline form. The method decouples the prediction of crystallization into 2 orthogonal bases, the rotation and the translation of molecular frames. A linear and spherical interpolation scheme are derived for each base, respectively. The score model is parameterized using a SE(3) equivariant network. The model is benchmarked against both generative model and simulation model baselines. Benchmarks and evaluation metrics are developed to evaluate the model performance. AssembleFlow achieves better performance compared with generative model baselines. Compared with simulation baseline PackMol, it achieves comparable performance but is 25 fold faster.

### Strengths
- The paper introduces a rigid frame based flow matching model to generate the structure of organic crystals. The approach is similar to those introduced for frame-based flow matching methods for protein backbone generative, like FrameFlow [1]. Unlike protein, the frame for organic molecules is not well defined. The novelty of this approach is to use the inertial frames of the reference frames for organic molecules. 
- A spherical interpolation scheme is introduced to model the rotation of molecular frames. A quaternion formalism is used to define the interpolation in the SO(3) group. The formalism looks similar to those used for protein frame modeling. It is unclear to me what is the core difference in the author’s approach.
- A set of new metrics, including packing matching (PM), atomic collision, separation, and compactness are introduced for the crystallization packing problem. The set of metrics are novel and meaningful from the domain perspective. It will benefit the community to further develop methods in the field. 
- The signficant improvements of AssembleFlow compared with other generative model baselines show the advantage of modeling the problem as rigid frames. This advantage is one of the core innovations of the method.
- AssembleFlow is able to match the performance of MolPack, a simulation baseline. This is quite significant. The 25 fold speed-up compared with MolPack shows the advantage of generative modeling approach.

[1] Yim, Jason, et al. "Fast protein backbone generation with SE (3) flow matching." arXiv preprint arXiv:2310.05297 (2023).

### Weaknesses
- The decoupling of SE(3) into R^3 and SO(3) groups, i.e., rigid flow matching, has been extensively explored in the protein backbone community. The authors need to review more related papers and articulate their innovations in the methodology.
- As the authors pointed out, numerical stability exists when there is a tie in eigenvalues for the principal axes of inertia. This problem is especially important for symmetric molecules. Can the authors add more details regarding how they address the instability issue? How much will it influence the training stability?
- The real organic crystals are periodic. However, the authors only model them as non-periodic clusters in this work. It is a foundational limitation of both the dataset and the AssembleFlow model.
- In real organic crystals, the rigid assumption may not always apply. The molecular conformation may change. Therefore, the rigid frame modeling approach may be limited in predicting the structure of more flexible molecules.

### Questions
- Can the authors explain the difference between their quaternion interpolation and the other SE(3) interpolation scheme like those developed in FrameFlow? Do you expect a difference in performance if you adopt a different SE(3) interpolation scheme? 
- In Fig. 4, it is very clear that PackMol generated structures have a much lower energy compared with AssembleFlow. This indicates a limitation of the generated structures. Can the authors expand on why AssembleFlow generated structures have a much higher energy?
- The authors mention 2 versions of their SE(3)-equivariant network for inter-molecule modeling. I could not find more results on comparing these 2 versions. Can the authors add more results or point me to the right location?
- How did the authors predict the initial conformation of the individual molecules? Are those predicted by RDKit? Or do you directly take the conformation from the data?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
AssembleFlow introduces a generative model for assembling clusters of rigid molecules. The authors assign inertial frames to each molecule, which reduces the problem to generative modelling on $SE(3)^N$. The authors then solve this with Riemannian flow matching, using linear interpolants for translations and spherical interpolants of quaternions for rotations. The trained model approaches the performance of PackMol but at a much lower computational cost.

### Strengths
The work introduces the task of rigid molecular crystallization in the context of deep learning, with significant potential downstream applications. Applying flow matching to spherical interpolation of quaternions is also novel. The paper is clear, well-written, and well-organized, with helpful figures to explain each concept. I appreciated the extensive background provided throughout the text, including on rotation matrices in the appendix.

### Weaknesses
The novelty in this work is mainly in solving a new application domain, rather than pure machine learning developments. As acknowledged by the authors, flow matching models on $SE(3)^N$ were first developed for assembling rigid residues to design protein backbones, such as FoldFlow [1]. The authors claim that inertial frames are a key innovation over models like FoldFlow, since arbitrary molecules do not have a unique way to assign frames, unlike protein residues. However, if the goal is just to uniquely interconvert between atomic coordinates and frames, this can be achieved by storing a canonical pose for each molecule and applying the Kabsch algorithm [2] to align each molecule in a cluster to the canonical pose, obtaining a rotation matrix for each molecule. This generalizes the proposed inertial frames, which specifically use a canonical pose from applying principal component analysis on the point cloud.

The selected baselines are a somewhat unfair comparison, as none of them use rigid molecules. A simple baseline to accomplish this could just be sampling thousands of random translations and rotations of rigid molecules, and filtering out structures with collisions. This would help emphasize the contribution of this work as a method for *intelligently* assembling rigid frames, not just exploiting molecular rigidity.

Some details of the molecular assembly task are not clear to me. Since a generative model is being used, are multiple samples inferenced before computing metrics? And is the input uncorrelated structure fixed, or is it randomly sampled on every training iteration?

The fact that packing matching of AssembleFlow and PackMol are similar in Table 2, but that energy distributions are quite different in Figure 4, suggests that packing matching is not that sensitive of a metric. A more sensitive metric could be RMSD, which should be possible to calculate using the Hungarian algorithm. If multiple polymorphs are possible for the same molecule, RMSD could be aggregated using the "average minimum RMSD" and "coverage" metrics proposed by Ganea et al. [3] for molecular conformer search.

Minor issues:
- Figure 8 the lower left box content is not centered
- Table 2 arrow is wrong for separation
- Line 176 SO(3) must also have det 1
- Line 285 "SPLER"
- Line 805 "dynamcis"

[1] Bose, A. J., Akhound-Sadegh, T., Huguet, G., Fatras, K., Rector-Brooks, J., Liu, C. H., ... & Tong, A. (2023). Se (3)-stochastic flow matching for protein backbone generation. arXiv preprint arXiv:2310.02391.

[2] https://en.wikipedia.org/wiki/Kabsch_algorithm

[3] Ganea, O., Pattanaik, L., Coley, C., Barzilay, R., Jensen, K., Green, W., & Jaakkola, T. (2021). Geomol: Torsional geometric generation of molecular 3d conformer ensembles. Advances in Neural Information Processing Systems, 34, 13757-13769.

### Questions
How are permutations handled? Are the molecules stored in a particular order? Since each molecule has identical structure, flow matching could learn shorter paths if shown examples that have been permutation-aligned by the Hungarian algorithm - see equivariant flow matching [1,2]

I am curious about the relationship between SLERP of quaternions and geodesic interpolation on SO(3) as discussed by FoldFlow. Are they mathematically equivalent?

[1] Klein, L., Krämer, A., & Noé, F. (2024). Equivariant flow matching. Advances in Neural Information Processing Systems, 36.

[2] Song, Y., Gong, J., Xu, M., Cao, Z., Lan, Y., Ermon, S., ... & Ma, W. Y. (2023). Equivariant flow matching with hybrid probability transport. arXiv preprint arXiv:2312.07168.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present the AssembleFlow framework, which continuously matches weakly correlated structures (like liquid water) to their strongly correlated counterparts, such as crystallized forms like ice. They have carefully designed an underlying theory that leverages advances in flow matching methods to address this problem. AssembleFlow learns SE(3) velocities by decomposing them into translations and rotations. Unlike other methods, they use quaternion representations for rotations, which allows for spherical interpolation. The authors also introduced the idea of reparameterization to directly model SE(3) at time T.

They performed a comprehensive evaluation by comparing results with other models and with PackMol, a standard simulation-based approach well-established in the field. The authors also computed the distribution of DFT energies for both PackMol and AssembleFlow structures. The method is shown to be 25 times faster than PackMol.

### Strengths
Strengths:
Physics-Based, Theoretically Sound Methodology: The approach includes utilizing quaternions for rotations, spherical interpolation, and using an internal reference frame.
Comprehensive Comparison with Other Methods: The authors provide thorough comparisons with existing models.
Use of Valuable DFT Energy Calculations: Employing DFT energies as a measure of the stability of the obtained structures adds credibility to their results.

### Weaknesses
Cons:
Performance Compared to PackMol: While AssembleFlow shows good results, it does not outperform PackMol. The considerable differences in the energy distributions between PackMol and AssembleFlow structures highlight the advantages of PackMol over AssembleFlow.

### Questions
Questions:
Energy Comparison in Figure 4:
Instead of comparing the distributions of DFT energies separately for PackMol and AssembleFlow structures, could you plot the energy differences between PackMol and AssembleFlow for the same structures? Generally, interpreting absolute energy values can be challenging.

Typo in Table 2:
In Table 2, there appears to be a typo. The separation should be indicated with an upward arrow (↑).

### Soundness
3

### Presentation
3

### Contribution
3
