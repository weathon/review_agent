# Unified Brain Surface and Volume Registration

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
Accurate registration of brain MRI scans is fundamental for cross-subject analysis in neuroscientific studies. This involves aligning both the cortical surface of the brain and the interior volume. Traditional methods treat volumetric and surface-based registration separately, which often leads to inconsistencies that limit downstream analyses. We propose a deep learning framework, UCS, that registers 3D brain MRI images by jointly aligning both cortical and subcortical regions, through a unified volume-and-surface-based representation. Our approach leverages an intermediate spherical coordinate space to bridge anatomical surface topology with volumetric anatomy, enabling consistent and anatomically accurate alignment. By integrating spherical registration into the learning, our method ensures geometric coherence between volume and surface domains. In a series of experiments on both in-domain and out-of-domain datasets, our method consistently outperforms both classical and machine learning-based registration methods--improving the Dice score by up to 7 points while maintaining regular deformation fields. Additionally, it is orders of magnitude faster than the standard method for this task, and is simpler to use because it requires no additional inputs beyond an MRI scan. Its superior accuracy, fast inference, and ease of use sets a new standard for joint cortical and subcortical registration.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a unified deep learning framework for brain MRI registration that jointly aligns cortical surfaces and volumetric subcortical structures using spherical + volumetric CNNs coupled by a consistency loss. The work aims to overcome limitations of existing methods (e.g., CVS) that treat the two domains sequentially.

### Strengths
Clear contribution: A unified model bridging spherical and volumetric registration is well-motivated.

Technical novelty: The cortical consistency loss connecting domains is elegant and addresses a real limitation in current pipelines.

Strong results: Substantial gains in cortical Dice (up to +7 points) on multiple datasets, with very fast inference vs. CVS.

Generalization: Tested on multiple held-out datasets including IXI and Mindboggle.

Solid ablation studies isolating effects of proposed components.

### Weaknesses
Dependence on spherical preprocessing at training: Requires surface extraction and inflation (FreeSurfer), which can be slow and may fail on challenging scans; more discussion on robustness would help.

Subcortical performance not universally better: Slight drop vs. uniGradICON/SynthMorph on IXI subcortical structures.

Limited modality / population scope: All training on adult T1-weighted datasets (OASIS/ADNI). No fetal/pediatric/clinical robustness experiments.

Ablations could explore more consistency-loss formulations (e.g., geodesic metrics on the sphere).

Clarity: Some math sections are dense; figure captions could better explain preprocessing needed.


No code availabel.

The paper is a bit dense I think more figure would help.

### Questions
Surface preprocessing dependency
Your method requires cortical meshes and spherical inflation during training. How sensitive is performance to inaccuracies or failures in this preprocessing? Could UCS be trained without spherical meshes (e.g., using synthetic or implicit surfaces)?

Inference on challenging scans
Since no surface information is used at inference, how well does the model perform on:

low-resolution or motion-corrupted clinical MRIs?

subjects with pathology altering cortical shape (e.g., tumors, resections)?
Any preliminary results or failure analyses?

Consistency loss mechanics
Could you elaborate on the interpolation needed to map spherical displacement back to 3D mesh vertices? How does sampling density on the sphere affect gradient stability?

Subcortical performance variability
Your method slightly underperforms baselines on IXI subcortical alignment.
Why might IXI behave differently? (acquisition style? age distribution?)
Would modest segmentation supervision balance this out?

Jacobian regularity vs. Dice trade-off
Ablations show increasing κ improves Dice but degrades regularity.
Do you have guidance on selecting κ for different applications (e.g., longitudinal studies)? Could adaptive balancing be learned?

Topology enforcement
While spherical alignment preserves cortical topology, volumetric deformation can still introduce folding. Would incorporating diffeomorphic constraints on the spherical path reduce the remaining ~0.1–0.2% folds?

Sphere parameterization distortion
Stereographic projection introduces sampling and area distortion.
How does this influence fine-scale folding alignment? Any alternatives considered (e.g., icosahedral CNNs)?

Training pairs
Registration direction is arbitrary. Did you augment by swapping moving/fixed pairs? Any asymmetry observed in deformation field quality?

Runtime comparison fairness
Since CVS requires spherical meshes at inference, did timing exclude preprocessing?
Could UCS runtime remain competitive if including preprocessing of spheres during training on real pipelines?

Scalability to groupwise registration
Could your framework support joint template estimation or atlas building?
What changes would be required to extend UCS to multi-subject alignment?

### Soundness
3

### Presentation
3

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
The paper proposes a deep learning framework called UCS that aligns both cortical and subcortical structures in in-vivo neuroimaging by using volume based dense registration combined with surface-based registration of the cortical surface. The paper posits the limitations of volumetric registration methods for registering the cortex despite its success in registering the subcortical structures and global anatomy. The problem with incorporating intensity and surface based registration is that the 2D surface registration has a gradient with a Lebesgue measure of 0 on the ambient 3D space, and to avoid this problem the paper proposes training a 2D and 3D registration network for the surface and volume respectively, and add a consistency term between the losses. Results are shown on four clinical MRI datasets (OASIS, ADNI, IXI, Mindboggle).

### Strengths
1. The mathematical formulation is very well defined, easy to follow, and technically sound. Usual papers in this topic are rather sloppy with overloaded notation or hand wavy definitions, but this paper is much clearer in that sense.
2. Most of the formulation is consistent with a typical deep learning registration framework, and the coupling term is the major technical novelty in the paper, which is easy to undeerstand.
3. Results are shown on a few popular clinical MRI datasets

### Weaknesses
1. The premise of the paper "While effective for aligning subcortical structures and global anatomy, volumetric deformable registration often fails in the cortex. The cortex is a thin, highly curved surface with significant inter-subject variability in folding patterns that is difficult to align in Euclidean space" is not substantiated in any way except for the experiments in the paper. Most methods trained on SynthSeg or Freesurfer labels (e.g. OASIS dataset) have achieved overall dice scores in the order of 0.88, almost to the point of overfitting on the labelmaps. Two of the labelmaps in freesurfer / synthseg labels are the cerebral cortex for each hemisphere. It would greatly strengthen the premise of the paper to show the performance of pre-trained optimization and deep learning methods on these datasets or labelmaps on only the cortical label registration.
2. Exaggerated narrative about slowness of optimization methods - "Moreover, because of its reliance on classical optimization techniques, CVS is computationally expensive, requiring several hours per subject pair, making it impractical for large-scale datasets." and other statements imply classical methods being very slow which has been shown to be false with newer implementations - particularly in the Oncoreg Learn2Reg challenge (where Syndeeds and ConvexAdam perform very well with no or little learning), and FireANTs which shows real-time registration on various clinical datasets.  
3. Equation 2 is not a coupled optimization - coupling is only achieved with Equation 3. This is problematic because the objective (integral) in Equation 3 is only defined on the boundary of the surface $dS$ , which has a Lebesgue measure of 0 on the 3D ambient space. This suffers from the same problem as joint optimization of the volumetric and surface registration using a single volumetric displacement grid. The paper does not justify why Equation 3 does not suffer from the same problem that direct optimization on the surface matching objective using a volumetric displacement does. This is a major issue in the paper - since the whole idea behind not performing direct optimization of the cortical surface was the measure 0 gradient of the surface loss. Since this is the only proposed technical novelty in the paper (coupling the surface and volume registration), proper empirical (showing the gradient propagation with and without the coupling term) or theoretical (proofs) justification is needed.
4. "All CVS experiments are done on an Intel(R) Xeon(R) Gold 5218 CPU. (There is no GPU-based implementation of CVS.)" - Use FireANTs or ConvexAdam for GPUs. These baselines have existed for more than a year now, and are very efficient for the scale of clinical datasets. 
5. Baselines are not adequate. Voxelmorph with segmentation loss should be compared with UCS. Synthmorph and unigradICON are general purpose registration tools like ANTs, but since the method assumes access to a training dataset, at least a few baselines should be trained with the same data. 
6. Line 424: "Including our proposed sphere loss (Base+Dice(all)+Sphere) produces a clear increase in Dice (cortical) while preserving a high Dice(subcortical) score." - this is not the right ablation to run. The correct ablation here is to add the cortical surface loss directly in the training / inference objective instead of the roundabout way (i.e. consistency between network outputs).

Minor issues: 
1. The geometric descriptors used in the surface registration are not mentioned. This hinders reproducibility of the paper.
2. to my knowledge, loss masking in unigradicon does not explicitly align the structures themselves but only registers within the provided masks. im not sure why this baseline is used in that case
3. Line 469 - "Unlike other methods that require additional inputs (cortical meshes, segmentations) at inference, our method requires only structural MRI images."  - This is a partially false statement - since all methods considered in the paper (except unigradICON with masking) do not require additional labels at inference. This is only true for classical iterative methods that do not learn any label-aware features.
4. Another potential weakness in the evaluation setup is that the considered datasets are possibly overfit to in the long run. There are other high resolution, high quality datasets (e.g. Ultracortex 9.4T) that have higher than 1mm resolution (i.e. 0.6mm)  and manual labellings of the cortical surface - providing a challenging evaluation setup for the exact problem tackled in the paper

### Questions
1. 2D registration has to be performed on a sphere where the coordinate $\phi$ wraps around from 0 to $2\pi$. how is this circular nature considered in the spherical registration? appropriate citations or derivations in the supplementary material can be useful 
2. What is the performance of the method without the segmentation loss? This might be used for hippocampus or MTL registration for example where only the registration of a single surface might be of interest without registering other structures. 
3. In many other clinical or research scenarios pertinent to neuroimaging studies, the surface mesh is typically available (using a segmentation algorithm or manually) and accurate registration of the surface and volume might be desired. However, collecting training data across many subjects might be infeasible. In such a scenario, can UCS be used in an online optimization fashion? 

I would like the authors to address some of the weaknesses of the paper followed up by the questions in this section.

### Soundness
3

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
5

### Summary
This paper presents UCS, a deep learning framework for brain MRI registration that combines volumetric and surface-based approaches. UCS employs a coupled architecture with spherical coordinate mappings and a consistency loss to enforce geometric coherence, using surface mesh registration results as additional supervision for volumetric registration. The method achieves state-of-the-art performance across multiple datasets.

### Strengths
1. UCS couples volumetric and spherical registration networks, promoting anatomical consistency across domains.

2. UCS outperforms classical (CVS) and modern deep learning methods (SynthMorph, uniGradICON) in both cortical and subcortical Dice scores.

### Weaknesses
1. The authors claim that their approach provides a unified solution for the registration of cortical and subcortical structures. And the key idea is to use correspondence between surface areas as additional supervision signal. This idea is not novel; a similar approach was presented in https://doi.org/10.1016/j.media.2019.101540

2. From the title, I had the expectation that the proposed approach would have some special designs for the registration of subcortical structures. However, such designs were not found.

### Questions
I don't have questions to ask because the paper is clearly written and the approach is straightforward to understand.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes UCS, a unified learning framework that trains a 3D diffeomorphic registration network and a 2D spherical registration network jointly, with a cortical-consistency loss that ties the cortical ribbon deformation in the volume to the sphere-space deformation.
The writing is clear and the experiments are sufficient. But still, some weakness exists.

I am inclined to give a score of around 5, but since that option isn't available, I will select 4 for now.

### Strengths
- This paper proposes a real and valuable challenge in MRI registration: current pipelines split surface vs volume.
- Key technical idea (soft consistency energy on cortical surface) is simple, plausible, and seems to explain gains; ablation supports this. 
- Sufficient comparison to the right classical joint method (CVS) and to current learning baselines (SynthMorph variants, uniGradICON).
- It is great to test on two additional held-out datasets.

### Weaknesses
- The author claims, "Unlike other methods that require additional inputs (cortical meshes, segmentations) at inference, our method requires only structural MRI images." However, they also mention, "For each image, we use FreeSurfer to generate anatomical segmentations" in the experiment setup. Therefore, it is not truly "only structural MRI images" as input. If you plan to integrate this with your model as a complete pipeline, do you include the time spent on segmentation when comparing to CVS?
- Why this task cannot be done by “3D reg + surface reg, then post-hoc propagation” process? Compared to some baseline methods, they did missing one modality in registration. It is quite nature to think more modality for cross-validation is better. The reviewer would like to see a discussion or clarification on this part

### Questions
- Discuss licensing/availability of FreeSurfer usage. Many ICLR readers are not neuroimaging specialists.

### Soundness
3

### Presentation
3

### Contribution
3
