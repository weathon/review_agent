# Rethinking Intracranial Aneurysm Vessel Segmentation: A Perspective from Computational Fluid Dynamics Applications

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
The precise segmentation of intracranial aneurysms and their parent vessels (IA-Vessel) is a critical step for hemodynamic analyses, which mainly depends on computational fluid dynamics (CFD). However, current segmentation methods predominantly focus on image-based evaluation metrics, often neglecting their practical effectiveness in subsequent CFD applications. To address this deficiency, we present the **I**ntracranial **A**neurysm **V**essel **S**egmentation (IAVS) dataset, the first comprehensive, multi-center collection comprising 641 3D MRA images with 587 annotations of aneurysms and IA-Vessels. In addition to image-mask pairs, IAVS dataset includes detailed hemodynamic analysis outcomes, addressing the limitations of existing datasets that neglect topological integrity and CFD applicability.
To facilitate the development and evaluation of clinically relevant techniques, we construct two evaluation benchmarks including global localization of aneurysms (Stage I) and fine-grained segmentation of IA-Vessel (Stage II) and develop a simple and effective two-stage framework, which can be used as a out-of-the-box method and strong baseline. For comprehensive evaluation of applicability of segmentation results, we establish a standardized CFD applicability evaluation system that enables the automated and consistent conversion of segmentation masks into CFD models, offering an applicability-focused assessment of segmentation outcomes.
The data, code, and model will be made publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces IAVS, a new multi-center dataset consisting of 641 3D MRA images with annotations for aneurysms and their parent vessels, along with computational fluid dynamics (CFD) analysis results. It further proposes a two-stage framework:
1.	A detection network using heatmaps and dynamic candidate selection for global aneurysm localization.
2.	A topology-aware segmentation network with clDice supervision for fine-grained IA-vessel segmentation.
To bridge the gap between image segmentation and hemodynamic analysis, the authors develop an automated CFD applicability evaluation system and propose a new metric—CFD-Applicability Score (CFD-AS)—to assess whether segmentation results can be successfully used for CFD simulations.

### Strengths
•  Clear motivation: The paper identifies an under-addressed but practically relevant gap between conventional segmentation evaluation and CFD usability in intracranial aneurysm analysis.
•  Comprehensive dataset: IAVS is larger and more structured than prior public datasets (e.g., ADAM, Royal), with annotations extending beyond masks to centerlines, STL models, and CFD convergence results.

### Weaknesses
1）Limited methodological novelty (major)
•  The proposed pipeline mainly combines existing components: focal loss for detection, nnUNet backbone for segmentation, and clDice for topology preservation.
•  The CFD applicability metric essentially checks meshability and simulation convergence, which is practical but not conceptually innovative.

2）Evaluation focused on internal dataset
•  Most experiments rely on the newly built IAVS dataset.
•  Although some external tests (e.g., GLIA-Net) are mentioned, there is no convincing demonstration of generalization across domains or acquisition protocols, which is critical for medical applications.

3）Benchmarking and comparison are insufficient
•  While several baselines are included (nnUNet, SwinUNETR, nnDetection), stronger recent methods (transformers, foundation models, SAM-like methods) are missing.

4）CFD-AS metric validity is weakly justified
•  The proposed CFD-Applicability Score is a binary feasibility check (topology, mesh generation, flow convergence). This is useful, but it lacks theoretical grounding or validation regarding hemodynamic accuracy, which should be the ultimate target.

5）Lack of deeper learning or modeling insights
•  The paper does not propose new learning algorithms, loss formulations, or training paradigms tied to CFD properties.
•  It could have explored physics-informed learning or differentiable CFD integration but remains purely post-hoc in its evaluation.

### Questions
1. It will be better to strengthen baseline comparisons by including more recent and competitive segmentation and detection models, such as transformer-based approaches or foundation models. 
2. It will be necessary to include external validation on independent datasets to better evaluate the generalization ability of the proposed framework beyond the IAVS dataset. Demonstrating performance on different imaging sources or clinical settings would significantly strengthen the paper.
3. It will be better to correlate the CFD-Applicability Score (CFD-AS) with actual CFD parameter deviations (e.g., wall shear stress, flow velocity) rather than only reporting binary feasibility. This would provide stronger evidence that the proposed metric reflects real hemodynamic reliability.
4. It will be better to incorporate CFD considerations into the training process, for example by introducing physics-informed constraints or differentiable surrogate models. Such an approach could enhance both the novelty and the clinical relevance of the work.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the IAVS dataset and a two-stage IA-Vessel segmentation framework, as well as a CFD applicability evaluation system and the CFD-AS metric, which aim to bridge the gap between image segmentation and subsequent CFD simulation in terms of practical usability. In its current version, the paper shows innovation and application-oriented design in dataset construction, evaluation design, and experimental results, but there remain several aspects needing improvement, particularly in methodological details, completeness of experimental settings, reproducibility, and systematic analysis of clinical applicability.

### Strengths
1.	The proposed IAVS dataset covers multimodal annotations required for the complete workflow from imaging to CFD (3D MRA images, IA/IA-Vessel masks, STL models, centerlines, meshes, CFD analysis results, etc.), which helps promote standardization and reproducibility of end-to-end research.
2.	By establishing a CFD usability evaluation system and the CFD-AS metric, the study quantifies the “CFD usability” of segmentation results, bridging the gap between common segmentation metrics (e.g., Dice) and real-world applicability.
3.	Stage I’s global localization aids in detecting small aneurysms and surrounding vessels, while Stage II’s topology-aware segmentation (integrating clDice) seeks to preserve vascular connectivity and reduce topological errors that negatively affect meshing and subsequent CFD analysis — a direct optimization for vascular anatomical complexity.
4.	The integration of the segmentation framework with CFD usability evaluation demonstrates that the proposed end-to-end method outperforms conventional segmentation baselines in terms of CFD applicability.

### Weaknesses
1.	Lacks systematic analysis of the coupling between Stage I and Stage II and of the feasibility of end-to-end training; there is no quantitative assessment of error propagation or robustness between the two stages.
2.	Although topological constraint losses (e.g., clDice) show improvements in vascular topology, the paper lacks detailed robustness analysis and quantitative results under different topological abnormalities (adhesion, distal branch misalignment, branch disconnection, etc.). Overemphasis on topological connectivity might sacrifice local geometric accuracy; systematic evaluation and recommended weight ranges should be provided.
3.	The engineering assumptions involved in converting segmentation masks to CFD models (meshing, boundary conditions, material parameters, etc.) should be listed and subjected to sensitivity analysis; otherwise, the stability and generalizability of the results are difficult to assess.
4.	Details on the division and statistical analysis of Set A and Set B are insufficient, lacking disclosure of sample sizes, confidence intervals, and significance tests, which limits the credibility of the conclusions.
5.	The paper does not sufficiently demonstrate empirical evidence linking CFD metrics to clinical outcomes (e.g., rupture risk, treatment decisions), which weakens its direct persuasiveness for clinical decision making

### Questions
1.	How does the global localization error in Stage I affect the segmentation quality and final CFD usability in Stage II? Could end-to-end joint training or co-optimization alleviate error propagation?
2.	While introducing topological constraints (e.g., clDice) benefits vascular connectivity, can it cause local geometric distortions in certain anatomical variations? Is there a lack of systematic sensitivity analysis for the weighting settings?
3.	Is the detailed conversion process from segmentation to CFD model (mesh generation, boundary conditions, material properties, convergence criteria, etc.) fully disclosed? Are patient-specific parameters reproducible?
4.	Have differences in imaging parameters, resolution, and noise among multi-center data been adequately controlled? Is there sufficient evidence of multi-center generalization capability?
5.	Does the study include direct validation of the CFD results’ clinical relevance (e.g., correlation with rupture risk or treatment decisions)? Is there an evaluation of time cost and workflow stability?
6.	Are the sample sizes of Set A and Set B sufficient? Are statistical indicators such as confidence intervals, significance levels, and effect sizes fully reported?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to rethink the intracranial aneurysm vessel segmentation task from the perspective of computational fluid dynamics. This study has a dataset (new or enhanced), benchmark (new two metrics?), and a new method(?). As a reviewer, I believe the main issue with this paper is its excessive scope and the lack of a clear, focused contribution.

### Strengths
- This paper compiles and curates a large-scale 3D MRA dataset, combining existing datasets with a new in-house collection, resulting in a total of 641 volumes and IAs, which is quite impressive.
- The author's promise to open-source data, code, and models is great.

### Weaknesses
- The overall writing appears somewhat disorganized to the reviewer. If the author intends to claim a contribution to data construction, more detailed information is expected, such as how the data was constructed and which aneurysm-related tasks it supports. From the reviewer's perspective, proposing a new model is unnecessary, particularly since you are introducing new metrics simultaneously.
- In its current writing, the reviewer is unclear about how the existing dataset has been enhanced and why these enhancements are significant. Specifically, is CFD-based optimization necessary to improve the dataset, or can it simply be applied as a subsequent step after existing segmentation methods to support all aneurysm-related applications? The author should justify this.
- The loss function for stage 1 is strange. Why is the count classification loss needed?
- For the loss function of stage 2, please add explanations for each parameter....
- Figure 4 is difficult to understand. How does the author do "check vessel geometry"? Not using CFD?

### Questions
See questions in weakness.

Suggestions
- Rethinking the entire story of this paper is suggested. Or it is a huge waste of your dataset!

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This article proposes to tackle the problem of vessel segmentation associated with haemodynamic analysis using Computational Fluid Dynamics (CFD). In particular, this article proposes to study intracranial aneurysms detection. The reason why CFD is involved in this study is to help clinician assess the rupture risk of aneurisms.

Authors propose a novel multi-center dataset that includes vessel masks, centreline, meshing and CFD results; two evaluation benchmarks, one for the localisation of aneurisms and the second for the accurate segmentation of blood vessels. Further, the authors make the point that good voxel-wise segmentation results do not guarantee something that will mesh correctly or results in accurate CFD simulations. Consequently they also propose a CFD suitability score system.

The paper describes their proposal and evaluation for vessel and aneurism detection and segmentation, the preprocessing steps leading to meshing and CFD simulation in the vessel region around detected aneurisms.

### Strengths
The paper describes a novel attempt at setting up a complete system of intracranial aneurism detection and evaluation based on 3D MR angiography. The paper is very interesting, well presented and although not perfect, quite sound in its approach. Given the amount of work necessary to collect, annotate and perform CFD simulation on more than 600 MRA volumes, the proposed dataset is quite unique and valuable.

The evaluation results are clearly the current state of the art among academic published papers although there exist proprietary datasets and methods in the same broad research domain. The availability of code and models will certainly advance the state of the art in this domain.

### Weaknesses
Not very many articles tackles the joint segmentation and CFD aspects of vessel segmentation, yet this has been an ongoing research area for a quite time. The article focuses on relatively recent, deep-learning based methods and does not attempt to review previous classical efforts. Previous to deep-learning, vessel segmentation methods used vesselness methods [1], many of which were reviewed and evaluated in [2]. Relatively recent projects have contributed to very similar objectives [3].

Overall the paper reads well but some important details are obscured in supplementary material, for example the CFD readiness evaluation is only partly automated. Difficult cases still need to be corrected by hand. This is not clear in the main article. 

It is not absolutely clear what CFD results present in the proposed dataset consist off, and what they are used for. In the paper they are not evaluated, only the amenability of result for CFD computation is. It would be extremely useful to associate the CFD results with some clinical evaluation, e.g is an aneurism dangerous? does it need to be operated on? What are the criteria, etc. Since this is the claimed end result of the proposed pipeline, this would seem essential.

The vessels segmentation method uses the clDice method which does not offer guarantee of connectivity contrary to what the authors imply. Indeed the skeleton the clDice is the medial axis, computed through the Lantuejoul formula, which does not yield connected voxels. Instead they should use a truly connected skeleton computable by deep-learning layers [4] or the skeleton recall loss [5] which is connected and very fast. Their results would likely improve.



[1] A.F. Frangi, W.J. Niessen, L.V. Koen, and M.A. Viergever. Multiscale vessel enhancement filtering.
Lecture Notes in Computer Science, 1496:130ff., 1998.
[2] Jonas Lamy, Odyssee Merveille, Bertrand Kerautret, and Nicolas Passat. A benchmark framework for
multiregion analysis of vesselness filters. IEEE Transactions on Medical Imaging, 41(12):3649–3662, 2022.
[3] https://explore.openaire.eu/search/project?projectId=anr_________::8e389f59b7c7aa1d24104847c23b5b09
[4] Mario Viti, Hugues Talbot, Bassam Abdallah, Etienne Perot, and Nicolas Gogin. Coronary artery
centerline tracking with the morphological skeleton loss. In 2022 IEEE International Conference on
Image Processing (ICIP), pages 2741–2745. IEEE, 2022.
[5] Yannick Kirchhoff, Maximilian R Rokuss, Saikat Roy, Balint Kovacs, Constantin Ulrich, Tassilo Wald,
Maximilian Zenk, Philipp Vollmuth, Jens Kleesiek, Fabian Isensee, et al. Skeleton recall loss for connec-
tivity conserving and resource efficient segmentation of thin tubular structures. In European Conference
on Computer Vision, pages 218–234. Springer, 2024

### Questions
- Why not perform CFD through the entire segmented vessel network?
- Blood is not a Newtonian fluid, why choose a Newtonian CFD solver?

### Soundness
3

### Presentation
3

### Contribution
3
