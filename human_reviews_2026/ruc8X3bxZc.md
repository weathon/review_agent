# USDPnet: An Unsupervised Symmetric Deep Framework for Robust Parcellation of Infant Subcortical Nuclei

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Accurate infant subcortical parcellation is vital for understanding early brain development and neurodevelopmental pathology. However, existing methods suffer from initialization sensitivity, poor bilateral consistency, and limited applicability to early postnatal data. We propose USDPnet, an Unsupervised Symmetric Deep Parcellation Network, which integrates deep autoencoder-based feature embedding with divergence-driven clustering. By introducing the generalized Cauchy-Schwarz divergence (GCSD) as the clustering objective, we enhance inter-cluster separability across complex developmental features. A symmetry constraint further enforces bilateral consistency, leading to structurally coherent and reproducible delineations. USDPnet operates on surface-based features extracted from infant subcortical nuclei. Experiments show it outperforms traditional and deep clustering baselines. Visualizations are largely consistent with the parcellation results based on anatomy and function connectivity. The resulting parcellations are developmentally grounded, anatomically symmetric, and functionally relevant, offering fine-grained and morphological coherent maps of early subcortical organization. Code is available at https://anonymous.4open.science/r/USDPnet-X12D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The study imposes mirror symmetry in parcellation of infant subcortical nuclei, that is known to be structurally symmetrical and processed with methods for adult brain parcellation methods.
The method depends on an autoencoder-based architecture that process the left and right hemisphere vertices with a  symmetry-aware clustering mechanism that utilizes Generalized Cauchy-Schwarz divergence on surface meshes. The clustering loss favors bilateral consistency, higher inter-cluster sample distances, lower intra-cluster sample distances and sparse cluster assignment vectors.
The results provide a thorough ablation and various baseline clustering methods, showing the proposed method has higher performance than the baselines.

### Strengths
- The bilateral symmetry-aware segmentation/parcellation is a recent topic [1, 2] and the study suggests a sound method for the problem.
- The divergence measure Generalized Cauchy-Schwarz Distribution introduces computational improvements over average-pairwise divergence measures.
- The study shows strong performance against alternative clustering baselines.

### Weaknesses
- Lack of Ablation on Core Component: The paper's claim of a novel divergence function is insufficiently supported, as the ablation study does not compare against alternative divergence measures (e.g., KL-divergence, JS-divergence). Without this comparison, it is impossible to assess whether the proposed function is truly responsible for the performance gains or if other architectural choices are the primary driver.
- References to other bilateral symmetry-aware segmentation/parcellation: The study can involve alternatives in different areas of research, i.e. [1,2].

[1] Sanket Wathore, Subrahmanyam Gorthi, Bilateral symmetry-based augmentation method for improved tooth segmentation in panoramic X-rays, Pattern Recognition Letters, Volume 188, 2025, Pages 1-7, ISSN 0167-8655, https://doi.org/10.1016/j.patrec.2024.11.023. 

[2] Raina, K., Yahorau, U. and Schmah, T. Exploiting Bilateral Symmetry in Brain Lesion Segmentation with Reflective Registration.
DOI: 10.5220/0008912101160122, In Proceedings of the 13th International Joint Conference on Biomedical Engineering Systems and Technologies (BIOSTEC 2020) - Volume 2: BIOIMAGING, pages 116-122, ISBN: 978-989-758-398-8; ISSN: 2184-4305

Typo: Line 456-457 vactors -> vectors

### Questions
Could the authors comment on the performance of the framework when using divergence measure alternatives to $D_{GCS}$, other than GJRD, beyond the computational advantages? Specifically, if the components A and Q are kept intact, how do alternative measures (average-pairwise divergences) compare in terms of both performance and the computational advantages highlighted in the paper?

[1] Mingfei Lu, Lei Xing, Badong Chen,
Measuring generalized divergence for multiple distributions with application to deep clustering,
Pattern Recognition,
Volume 157,
2025,
110864,
ISSN 0031-3203,
https://doi.org/10.1016/j.patcog.2024.110864.

### Soundness
3

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
This paper proposes USDPnet, an unsupervised deep clustering framework incorporating anatomical symmetry constraints for fine-grained infant subcortical parcellation. The method leverages surface-mesh vertex area trajectories, a latent representation encoder, and a Generalized Cauchy–Schwarz Divergence (GCSD) objective, along with a hemisphere-pairing MSE symmetry loss.
Experiments on the Baby Connectome Project (BCP) dataset demonstrate improvements over several conventional and deep clustering baselines, accompanied by ablation studies and statistical significance testing. Visualizations indicate anatomically reasonable results and improved bilateral consistency.

### Strengths
1. The paper tackles the important and challenging problem of infant subcortical parcellation, which is clinically relevant and understudied in the unsupervised setting.
2. The approach is label-free and has potential value for large-scale infant neurodevelopment studies where manual annotations are difficult or costly to obtain.
3. The manuscript is generally well-organized, with clear presentation of the model design, loss components, and visual examples.

### Weaknesses
1. Lack of external validation with anatomical ground truth

    No Dice, ARI, or NMI comparison to expert labels or standard infant atlases. Reliance on internal clustering metrics limits biological interpretability.

2. Risk of suppressing true biological asymmetry

    The symmetry constraint may over-regularize regions with known asymmetries (e.g., amygdala, thalamus). No analysis provided to quantify the impact or demonstrate robustness.

3. Scalability concerns

    The full-batch GCSD objective may not scale to larger datasets or higher-resolution surfaces. No discussion on computational efficiency or potential approximations.

4. Limited feature modalities

    Only vertex-area trajectories are used. Incorporating curvature, thickness, deformation tensors, or multimodal T1/T2 contrast might provide more stable clustering.

5. FH metric insufficiently defined

    The proposed Feature Homogeneity metric is not formally introduced in the main text, limiting reproducibility and interpretation.

6. Recent literature coverage is insufficient

    Related work on deep clustering and infant brain parcellation from the past 3 years is under-represented.

### Questions
1. The authors cite Lu et al. (2025b) for the GCSD estimator. Please clarify the precise differences between that prior work and the current implementation — specifically, have you introduced any modifications or simplifications relative to the original estimator?

2. You note that small asymmetries may lead to misleading conclusions. The manuscript also demonstrates to some extent that imposing a symmetry constraint aids segmentation. However, what happens if the input data include small but true asymmetries? Can the proposed method maintain robustness under such conditions, or does the symmetry‐constraint unduly suppress biologically meaningful asymmetry?

3. The use of MSE as the loss for the symmetry constraint raises a concern: might this penalty unintentionally penalize genuine developmental asymmetries? Have you considered employing a weighted or soft symmetry constraint (for example, applying it only to high‐confidence regions) to avoid suppressing valid anatomical differences?

4. The metric “Feature Homogeneity” (FH) appears in your results, but I could not find its formal definition in the main text. Please provide in the rebuttal the exact formula for FH and clarify its physical/biological interpretation.

5. In the figures and tables, please ensure that appropriate legends and annotations are included so that readers can understand the visualized results without excessive ambiguity.

6. Regarding GCSD computation, Equation (2) involves logarithms and matrix products, which may be prone to numerical instability. Could you clarify how potential underflow or negative values are handled to ensure safe log computation?

7. The current reference list seems to under‑represent the past three years of literature in deep clustering and infant brain parcellation. Please consider incorporating more recent studies to demonstrate how your work builds upon and differs from the state‑of‑the‑art.

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
5

### Summary
The paper presents USDPnet, an unsupervised network for deformable medical image registration. Instead of relying on ground-truth deformation fields, the method introduces a dual-path framework that aligns source and target images through both intensity-based and structural similarity losses. The model incorporates a pyramid-level deformation strategy and an uncertainty-guided regularization term to stabilize training and improve anatomical alignment. Experiments on multiple 3D medical datasets show that USDPnet achieves accuracy on par with or better than supervised approaches while maintaining fast inference.

### Strengths
The paper addresses a core challenge in medical image registration: learning accurate deformation fields without supervision through a well-thought-out architecture. The dual-path design combining global and local cues is elegant and grounded in practical clinical needs. The inclusion of uncertainty-guided regularization is a nice touch that helps balance smoothness and precision in difficult regions. Results across datasets are solid, showing clear improvements in dice scores and alignment metrics compared to VoxelMorph and TransMorph. The paper is also clear, with figures that make the deformation behavior interpretable.

### Weaknesses
The novelty is somewhat modest; many elements (e.g., pyramid strategy, dual losses) build upon existing unsupervised registration methods. The paper could better clarify what makes its dual-path design fundamentally different, rather than a refined combination of prior ideas. The evaluation is also limited to standard benchmarks; there’s little exploration of how USDPnet generalizes to unseen modalities or pathological scans. The uncertainty term, while useful, is described heuristically with little theoretical justification. Finally, runtime and memory costs aren’t reported, leaving open how scalable the model is for large 3D volumes.

### Questions
- How sensitive is USDPnet to hyperparameter settings in the uncertainty weighting term?
- Could the model adapt to multi-modal registration (e.g., CT–MRI) without retraining?
- How does USDPnet handle large deformations compared to transformer-based approaches like TransMorph?

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
4

### Summary
This work, `USDPnet`, proposed an unsupervised surface-based parcellation pipeline for subcortical nuclei, focusing on infant brain MRI. Leveraging a divergence measure (namely, generalized Cauchy-Schwarz Divergence, `GCSD`) and the symmetry constraint, a deep neural network was trained to cluster each vertex of the surface mesh. The proposed framework was evaluated using longitudinal BCP datasets from infants aged 0-24 months. 

The main contributions are to utilize unsupervised clustering leveraging GCSD and symmetry regularization to parcellate subcortical nuclei in a surface mesh. This work bears some merits, such as providing averaged results from thirty runs under different settings and sensitivity analysis, while it also exhibits several weaknesses and has some confusing points. Please refer to my review below. If my concern can be adequately addressed, I'd be happy to revise my rating.

### Strengths
1. Some cluster/subregion counts for each nuclei are similar to a published work [1] on Nature Neuroscience about subcortical parcellation utilizing fMRI. 

2. This work evaluated the proposed and comparison methods under various settings (different cluster numbers) in 30 runs. The average results confirm the better performance of the proposed work. 

3. Parameter sensitivity and ablation analyses were conducted. 

4. Open-source contribution. 

[1]: Tian, Ye, et al. "Topographic organization of the human subcortex unveiled with functional connectivity gradients." Nature Neuroscience 23.11 (2020): 1421-1432.

### Weaknesses
1. The reproducibility is not evaluated. I.e., with the optimal subregion/cluster counts, repeat the unsupervised clustering several (3-10) times, what is the adjusted rand index if choosing one run (e.g., the current result) as the ground truth? This is very important but missing in the current manuscript. If the adjusted rand index is low, meaning irreproducible, even if the other metrics indicate superior performance, it still significantly undermines the values of this work. This is the main reason that impacts my rating. 

2. There is a significant sensitivity to the parameters. This should be elaborated more in the manuscript and expressed as a major limitation. 

3. The current manuscript is not so clearly presented. Please see the questions below. 

4. There are limited contributions to representation learning or unsupervised clustering, as the GCSD is an existing and published work, and symmetry regularization is an incremental change. It brings more significant contributions to neuroscience than to representation learning or unsupervised clustering. 

5. The reviewer suggests refraining from using words like "anatomically plausible", "biologically plausible", or any phrasing implying the cluster is physiologically sound and correct. This is NOT supported by any evidence in the current manuscript. 
   - The higher SC/CH/RE/FH does `NOT` indicate any plausibility in the physiology and neuroscience world. They are technical metrics evaluating a clustering algorithm. 
   - Visual comparison with [1] in Fig. 3 does not directly imply the plausibility. There is a visual discrepancy between [1] and USDPnet, particularly in the X view of Putamen. 
   - To claim plausibility, a lot more experiments and statistical analyses should be conducted, other than some clustering metrics and a visualization comparison with [1]. 

6. In Lines 432-435, it is better to provide some quantitative metrics to indicate agreement. Qualitative visualization is not enough to support those arguments. 

[1]: Tian, Ye, et al. "Topographic organization of the human subcortex unveiled with functional connectivity gradients." Nature Neuroscience 23.11 (2020): 1421-1432.

### Questions
1. In Appendix `D2`, the 4D atlas construction was included as part of the work. The reviewer is curious about the rationale for redoing this 4D atlas construction in the case that BCP has already constructed a 4D atlas. Moreover, that 4D atlas is peer-reviewed and publicly accessible [1]. Why rebuild the wheel? Similarly, in `D3`, the segmentation process was described in detail. 

On the other hand, in the anonymized repository, it directly points to the public BCP atlas, which confused the reviewer. If the BCP atlas is used, why is the publication [1] not cited in the manuscript? The way it currently reads implies that the atlas construction and segmentation are part of the contributions of this work, which is incorrect. The BCP atlas is already established, peer-reviewed, and released ([1] and [link](https://www.nitrc.org/projects/uncbcp_4d_atlas/)). It could not be reclaimed as a contribution in a new work. Doing so could be an integrity issue. I raise an ethical concern regarding this point. 

2. Do the results in Table 2 correspond to the experiment mentioned in Appendix `E`? 

3. Do the results in Table 1 correspond to the optimal setting mentioned in lines 916-917? 

4. Is the feature encoder extracting features from the atlas at a single time point or from multiple time points?

5. Why is the symmetry consistency loss based on MSE, not cross-entropy? 

[1] Chen, Liangjun, et al. "A 4D infant brain volumetric atlas based on the UNC/UMN baby connectome project (BCP) cohort." NeuroImage 253 (2022): 119097.

### Soundness
3

### Presentation
2

### Contribution
2
