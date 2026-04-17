# PhysHandi: Physics-Based Reconstruction of Hand-Deformable Object Interactions

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
While existing methods for reconstructing hand–object interactions have made impressive progress, they either focus on rigid or part-wise rigid objects—limiting their ability to model real-world objects (e.g., cloth, stuffed animals) that exhibit highly non-rigid deformations—or model deformable objects without full 3D hand reconstruction. To bridge this gap, we present PhysHandi (Physics-based Reconstruction of Hand and Deformable Object Interactions), a framework that enables full 3D reconstruction of both interacting hands and non-rigid objects. Our key idea is to physically simulate object deformations driven by forces induced from densely reconstructed 3D hand motions, ensuring that the reconstructed object dynamics are both physically plausible and coherent with the interacting hand movements. Furthermore, we demonstrate that such simulation of object deformations can, in turn, refine and improve hand reconstruction via inverse
physics. In experiments, PhysHandi outperforms the state-of-the-art baseline across reconstruction, future prediction, and generalization to unseen interactions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper, PHYSHANDI: Physics-Based Reconstruction of Hand–Deformable Object Interactions, proposes a framework for dense 3D reconstruction of interacting hands and deformable objects. The key idea is to couple a parametric hand model (MANO) with a spring–mass simulation of deformable objects, where interaction forces are induced by hand mesh motion. The pipeline consists of three stages: hand reconstruction, object reconstruction, and hand refinement through inverse physics. Experiments on the PhysTwin dataset and a newly collected DENSEHDI dataset show improvements over the prior baseline PhysTwin in reconstruction, future prediction, and generalization

### Strengths
1. Inverse physics for hand refinement: The idea of improving hand reconstruction by leveraging reconstructed deformable object dynamics is interesting.

2. Dataset contribution: The authors additionally collect a new dataset (DENSEHDI) featuring denser hand–object contacts, which could benefit the community.

### Weaknesses
1. Limited generalization beyond lab conditions:
All experiments are conducted in highly controlled RGB-D capture setups (three synchronized RealSense cameras). There is no evidence that the method generalizes to in-the-wild scenarios (e.g., monocular RGB videos, unconstrained lighting, cluttered backgrounds). Given the reliance on multi-view depth, the method is unlikely to scale to real-world applications.

2. Strong assumption on contact regions:
As discussed in Section 5, the framework assumes a fixed hand–object contact topology within a sequence. This is a very strong and unrealistic assumption: in real interactions, contacts appear and disappear dynamically (e.g., fingers releasing or sliding over cloth). This limitation significantly reduces the applicability of the method.

3. Lack of extensive visualizations:
While a few qualitative figures are shown in the main paper (e.g., Fig. 3), the visual results are limited in scope. For a reconstruction method, more extensive qualitative evidence (especially videos in supplementary material) is crucial for evaluating realism and stability. Without this, it is hard to be fully convinced of the claimed improvements.

### Questions
Please refer to Weaknesses.

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
4

### Summary
This paper introduces PHYSHANDI, a physics-based framework for reconstructing hand-deformable object interactions from sparse-view RGB-D videos. PHYSHANDI reconstructs deformable objects with the interaction forces, which are simulated based on reconstructed hand motions. The method further refines hand reconstruction using inverse physics from object deformations.

### Strengths
1. The paper is well-written and well-organized.

2. The topic of hand-deformable object reconstruction is interesting.

### Weaknesses
1. More HOI methods using Spring-Mass models, such as CPF [1], have not been discussed.

2. The proposed framework relies on the initial MANO-based hand reconstruction to drive the subsequent deformable object simulation. If the hand mesh contains significant errors or inaccuracies, to what extent can PHYSHANDI still reconstruct a meaningful deformable object? Have the authors evaluated the robustness of the object reconstruction under degraded or noisy hand estimates?

3. Is the proposed method sufficiently innovative compared to the baseline method? For example, is there any substantial difference between Section 3.1 and PhysTwin?

4. The paper only compares one method. Please provide comparisons with more methods on more datasets. For the qualitative comparison, please provide results from multiple methods on the same test samples.

[1] CPF: Learning a Contact Potential Field to Model the Hand-Object Interaction. ICCV 2021.

### Questions
Same as the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PHYSHANDI, a physics-based framework that jointly reconstructs a full 3D hand and deformable objects from sparse RGB-D sequences. The method treats the dense MANO hand mesh as boundary conditions that drive a spring-mass object via virtual springs, and then performs inverse-physics refinement so that the learned object model provides supervision and correction for the hand. Experiments on PhysTwin scenarios and the new DENSEHDI dataset report improvements over PhysTwin on reconstruction/resimulation, future prediction, and generalization.

### Strengths
1. The work specifically targets deformable-object interaction within hand–object reconstruction, which remains underexplored, and it fills a tangible gap.
2. The supp. discusses the choice of $\delta$, a key hyperparameter; surfacing part of this analysis in the main paper would further clarify the design.
3. The paper releases a dataset that benefits the community.

### Weaknesses
1. Relative to PhysTwin, the contribution appears incremental. The main change introduces MANO as a topological prior for the hand to improve physical modeling near contact. This design understandably differs from PhysTwin, which targets a robot-arm-plus-gripper setting and therefore does not emphasize detailed hand modeling.
2. MANO has limited expressiveness for elastic skin and soft-tissue effects. While it encodes hand topology, optimizing $\Theta$ alone does not capture elastic deformation.
3. The method depends on several potentially brittle components: accurate depth (which consumer sensors like D455 often noise), and CoTracker for initialization. The accuracy of CoTracker likely has a first-order effect on downstream object reconstruction quality.

### Questions
1. Beyond introducing MANO as a hand prior, does the approach support broader application scenarios than PhysTwin, or does it offer a deeper theoretical contribution that the paper can articulate more explicitly?
2. When modeling hand deformation, do the authors consider alternatives (e.g., non-rigid extensions, blendshape or learned corrective fields) that may better account for elastic effects than optimizing $\Theta$ alone?
3. Can the authors include ablations on input quality (e.g., depth noise, missing data) and on dependencies (e.g., CoTracker accuracy, initialization errors) to quantify robustness?

### Soundness
2

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
The paper introduces PHYSHANDI, a physics-based framework for reconstructing and simulating 3D hand-deformable object interactions from sparse-view RGB-D videos. The technical contributions include: (1) Dense hand modeling using the MANO parametric hand model. (2) Deformable object simulation via a spring-mass system, where object deformations are driven by forces from reconstructed hand motions. (3) A three-stage optimization pipeline. In experimetns, the method outperforms the state-of-the-art baseline (PhysTwin) in reconstruction accuracy, future prediction, and generalization to unseen interactions, particularly in scenarios with dense hand-object contacts.

### Strengths
1.	The novel dataset DENSEHDI focuses on dense hand-object contacts, addressing a gap in existing benchmarks. 

2.	This paper achieves the dense 3D reconstruction of both hands and deformable objects simultaneously, ensuring physical plausibility and coherence between hands and object dynamics. 

3.	The inverse physics design help improve accuracy in sparse-view settings, especially the single-view cases.

### Weaknesses
1.	Physics-based simulation and optimization may require significant computational resources. Computational cost should be clarified. 

2.	This paper lacks experiments directly evaluating the accuracy of hand reconstructions. The title of the paper gives hand and object the equal position, but only evaluate objects. The claim in the Abstract “such simulation of object deformations can, in turn, refine and improve hand reconstruction via inverse physics” is not fully supported. 

3.	The quantitative results do not significantly outperform previous PhysTwin method in Table. 1, especially for  metrics such as SSIM. Why?

4. It seems that the model assumes fixed hand-object contact topology within a sequence, limiting its applicability to interactions with dynamic contact changes (e.g., sliding or rolling). 

5. Previous paper "InteractionFusion: Real-time Reconstruction of Hand Poses and Deformable Objects in Hand-object Interactions, ACM SIGGRAPH 2019." should be cited. This is "another kind" of hand-deformable object interactions". In comparison with this paper, this manuscript is more like a dynamic object simulation or cloth simulation paper with a hand-refinement module, which targets refining the object dynamics at the end.

### Questions
The main flaw is lack of experiments on hand reconstruction quality.

### Soundness
2

### Presentation
2

### Contribution
2
