# Scaling the Prior: Size-Consistent Geometric Diffusion for 3D Molecular Generation

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Diffusion models usually operate in fixed-dimensional metric spaces. In contrast, geometric molecular data naturally vary in dimensionality as molecules have different sizes (numbers of atoms). As a simple adaptation, existing diffusion models for geometric molecular generation employ network architectures that can handle variable-sized inputs, such as graph neural networks and transformers. **However, these approaches overlook the fact that the molecular size also determines the spatial scale of the atomic coordinates, which in turn induces inconsistent behaviors in the generative trajectories across different molecular sizes.** The generative process of geometric diffusion for 3D molecular generation can be viewed as first establishing a coarse structural target, followed by progressively refining the precise atomic positions. In particular, larger molecules tend to establish coarse structures earlier than smaller molecules due to their larger spatial scales relative to that of the noise. As a result, the reverse process becomes inconsistent across molecular sizes, with the denoising trajectories relying heavily on molecular sizes rather than on a unified generative pattern. In this work, we are the first to identify and analyze this size-induced inconsistency through a decomposition of the denoising dynamics, which reveals how spatial scale affects the progression of molecular formation, in both 3D structures and atom types. Building on this insight, we propose Scaling the Prior (StP), a simple yet effective approach that normalizes the learning and generative process across molecular sizes by rescaling the prior distribution based on molecular sizes. This adjustment harmonizes the denoising trajectories, enabling the model to learn a unified generative pattern and produce consistently high-quality molecules.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a size-dependent inconsistency in diffusion-based molecular generation, where smaller molecules converge structurally slower than larger ones during both training and sampling. To address this, the authors propose StP, a simple modification that rescales the Gaussian prior variance based on molecular size, harmonizing the denoising trajectories across different molecular scales. This approach improves the consistency and quality of generated molecules from the experiments on QM9 and GEOM-Drugs, and could potentially extend to larger biomolecular systems such as proteins.

### Strengths
1. This paper provides interesting observations on the relation between molecule sizes and proximal structural convergence speed. 
2. Experimental results show that StP works on some baseline diffusion models.

### Weaknesses
1. The paper’s notion of “convergence” is heuristic rather than principled. The proposed $\gamma_t$ and $\beta_t$ only measure correlations between predicted clean samples and final outputs, which depend on model training and not on intrinsic diffusion dynamics. Thus, the claim that smaller molecules “converge slower” lacks a rigorous statistical or theoretical foundation.
2. The use of convex hull–based scaling lacks physical justification and statistical robustness; it is an arbitrary geometric proxy for molecular size rather than a chemically meaningful measure.
3. While I appreicate authors' effort on observations and experiments, the reported gains are marginal on QM9 and GEOM-Drug generation.

### Questions
See Weakness

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
5

### Summary
This paper identifies that using a fixed Gaussian prior in 3D molecular diffusion models causes size-dependent inconsistencies—larger molecules experience weaker relative noise and thus stabilize earlier, yielding seemingly higher quality. To correct this, the authors propose Scaling the Prior (StP), which rescales the prior noise variance according to molecular size rather than normalizing coordinates (which would distort bond lengths). This adjustment aligns denoising dynamics across molecules of different sizes, improving validity and stability on QM9 and GEOM-Drugs datasets.

### Strengths
- The authors tackle the issue of size-dependent inconsistencies from the perspective of diffusion model variance, which is highly innovative.
- The proposed StP method is remarkably simple and elegant, and it can be applied to all types of diffusion models.
- The paper is very well written, and the figures are clear and illustrative.

### Weaknesses
- First, the paper’s conclusion — “at the same noise scale, larger molecules are more stable” — has already been observed in MOLCRAFT [1]. Therefore, the authors should further validate the effectiveness of the StP method on structure-based drug design (SBDD) tasks, such as CrossDocked2020 [2] and GenBench3D [3].
- The baselines used in this paper are relatively outdated. With the emergence of Flow Matching and Mean Flow methods, molecular generation has become both faster and higher in quality. The authors are encouraged to include more flow-based baselines for comparison, such as FlowMol3 [4], SemlaFlow [5], and others.
- In the visualization experiments, the authors did not show per-size visualizations for multiple metrics such as Validity, Uniqueness, and Validity × Uniqueness; nor did they provide speed curves broken down by molecular size.
- In Figure 1, the upper-right element should be an arrow, and in several figures, the axis ticks point outward, whereas they should point inward.

  [1] MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space
  [2] CrossDocked2020: A publicly available dataset for binding pose and affinity prediction
  [3] Benchmarking structure-based three-dimensional molecular generative models using GenBench3D: ligand conformation quality matters
  [4] FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation
  [5] SemlaFlow -- Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching

### Questions
See weaknesses

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
The paper studies the problem of varying molecular size in diffusion generative models. It shows that molecules with different size (number of atoms) can have different performance, where larger molecules usually have better stability than smaller molecules. The paper also proposes a simple method to solve this phenomenon by introducing a way to normalize the Gaussian distribution based on molecular size. The authors evaluate their method on QM9 and GEOM-drugs molecular datasets, with multiple models from the literature (EDM, RADM , and GeoLDM).

### Strengths
* The paper shows a good evaluation on how molecular size can have an effect on the perfromance of the generated molecules, with a trend that molecules with a larger number of atoms can have better performance than molecules with a smaller number of atoms.
* The paper proposes a simple method to deal with this phenomenon, where they can scale the Gaussian prior distribution in the forward process based on molecular size.  The authors empirically show that this normalization can  be applied to different architectures and improve the performance of generated molecules on QM9 and GEOM-drugs datasets.

### Weaknesses
- The paper could benefit from more theoretical analysis and why the scaled prior is effective. Also,  as mentioned by the authors,  one direct approach is to apply size normalization/ normalize the 3D coordiates. So, more discussuion on that is required and  how their method is different from that.

- The scaled prior parameters depend on the averages over the training subset. I think this might have some limitations if test molecules have different size distribution. The paper doesn’t show evaluation on how this might affect the performance, for e.g., molecules that have a different number of atoms from the training data.

### Questions
How does the distribution of the size of the generated molecules vs performance change, after applying the scaling parameter, eg, as in Figure 2?

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
3

### Summary
This paper identifies an inconsistency in molecular generation trajectories of varying sizes when applying traditional diffusion methods to molecular generation. To address this issue, the authors propose a normalization technique called Scaling the Prior (StP), which resolves the inconsistency by rescaling the prior distribution.

### Strengths
1. The issue raised by the authors, the inconsistent convergence rates across molecules of different scales, is a noteworthy problem that deserves attention.  
2. The authors propose a simple yet effective method to address this issue.

### Weaknesses
1. Although the spatial-scale intervention accelerates the convergence of molecular structures, it may compromise the diversity of the generative model. The authors should include comparisons of the Novelty metric across different methods in their experiments.

2. The paper only presents a comparison between molecular size and stability trends observed in the training set, but does not provide a similar analysis for the generated molecules—specifically, how molecular size correlates with stability in the generated samples.

3. The SoTA method compared in the paper, GeoLDM, was published in 2023. Methods introduced recently should also be included in the comparison to ensure the evaluation reflects the current state of the field.

### Questions
1. Why does Table 1 show that the performance improvement on the smaller-scale dataset QM9 is less compared to larger-scale dataset GEOM-Drugs?

2. Molecular size in generation is typically sampled from the size distribution of the training set. Does a training set containing larger molecules inherently offer an advantage over one with smaller molecules? Have the authors conducted any comparative experiments to investigate this?

### Soundness
2

### Presentation
3

### Contribution
2
