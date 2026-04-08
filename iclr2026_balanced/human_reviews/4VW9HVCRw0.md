## Human Reviewer 1

### Summary
The paper reframes HOI generation from grasp-centric scenarios to free-form interactions and introduces WildO2, an in-the-wild 3D HOI dataset constructed via an O2HOI pairing-and-reconstruction pipeline. It proposes TOUCH, a three-stage framework: (i) text- and geometry-conditioned CVAEs that predict hand/object contact maps; (ii) a multi-stage conditional diffusion model that injects global cues with coarse SSC text early and local geometry with fine DSC text late; and (iii) a lightweight physics-constrained refiner with cycle-consistent contact to correct global pose and sharpen local contacts. The system aims to produce controllable, diverse, and physically plausible interactions beyond grasping.

### Strengths
1. The paper contributes a relatively comprehensive dataset to the community and provides useful dataset statistics and analyses in the supp.
2. The method’s explicit prediction of contact regions, coupled with a coarse-to-fine conditioning schedule, is conceptually sound and well aligned with the goal of improving both global plausibility and local contact fidelity.

### Weaknesses
1. The approach (and the dataset pipeline) relies on one-image-to-3D reconstructions, especially during TTA. Inaccurate object reconstruction can propagate to and bias the estimated hand pose. The paper should analyze or mitigate this dependency—for example, via robustness studies under controlled reconstruction noise, uncertainty-aware weighting, or comparisons with stronger/alternative reconstruction backbones.
2. While modeling dorsal-side contact is interesting, the paper does not clearly articulate advantages over prior grasp-generation methods such as SemGrasp, which also specifies finger contacts and applies TTA for post-processing. A direct comparison—quantitative and qualitative—under matched prompts and settings would better substantiate the claimed benefits.

### Questions
The following questions are based on the weaknesses discussed above; please refer to that section.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces Free-Form HOI Generation, emphasizing controllable and semantically rich interaction synthesis beyond grasping. In addition, the paper proposes TOUCH, a three-stage framework for text-guided, controllable generation of free-form hand-object interactions (HOI). The multi-level diffusion framework is conditioned on fine-grained text and contact maps, integrating global and local semantic cues for physically plausible synthesis. Finally, the paper introduces WildO2, a large in-the-wild 3D HOI dataset (4.4k interactions, 92 intents, 403 objects) from internet videos via an automated O2HOI reconstruction pipeline. Experiments show the advantage of TOUCH over baselines (ContactGen, Text2HOI) in contact accuracy, plausibility, and semantic alignment.

### Strengths
1. The paper addresses free-form HOI generation with fine-grained textual control. The three-stage design effectively combines semantics, geometry, and physics.

2. Comprehensive dataset: WildO2 offers unprecedented diversity, with detailed contact annotations and high-quality reconstructions from in-the-wild videos.

3. Strong quantitative and qualitative performance: improvements over state-of-the-art HOI generation baselines across multiple metrics.

4. Clear ablations and insightful analyses: The impact of contact maps, coarse/fine text, and physical consistency is systematically evaluated.

### Weaknesses
1. Static generation limitation: TOUCH focuses on single-frame poses; temporal dynamics (motion continuity, causality) are left for future work.

2. Dataset scale and noise: Although diverse, WildO2 (4.4k samples) remains smaller, and in-the-wild reconstruction errors (≈45% failure rate) suggest potential biases.

3. Comparisons could be expanded: While ContactGen and Text2HOI are solid baselines, comparisons with other text-conditioned 3D diffusion or affordance models (e.g., DiffH2O, Nl2Contact) would strengthen positioning.

4. Ablations on language encoder: The Qwen-7B module shows gains, but results for alternative encoders (e.g., CLIP, BERT) are only briefly summarized. An analysis of semantic faithfulness could be better.

5. Limited discussion on cross-domain generalization: It is unclear how the model generalizes to unseen object categories or out-of-distribution verbs beyond the 92 labeled intents.

### Questions
1. How does TOUCH handle ambiguous or conflicting textual intents (e.g., “loosely hold” vs. “grasp tightly”)?

2. What about the generalization to unseen object categories or verbs in WildO2?

3. Could the refinement module be extended to temporal HOI (e.g., multi-frame optimization)?

4. For dataset details, will WildO2 include the intermediate 2D-3D alignment pipeline and failure cases?

5. How does the model behave when the text omits contact information (e.g., only “push the cup”)?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper addresses a critical data bottleneck in the hand–object interaction (HOI) domain — the lack of high-quality 3D datasets capturing free-form, non-grasping interactions. While existing datasets focus almost exclusively on structured grasping scenarios collected in laboratory settings, this work proposes WildO2, an in-the-wild 3D HOI dataset that covers diverse everyday manipulations such as pushing, poking, turning, and rotating. WildO2 is automatically constructed from internet videos using an object-only to interaction (O2HOI) frame pairing pipeline, followed by multi-stage 3D reconstruction, contact optimization, and text-based semantic annotation via vision–language models.

Building on this dataset, the authors introduce TOUCH, a three-stage text-guided framework for controllable HOI generation. TOUCH integrates (1) explicit contact map prediction, (2) a multi-level conditioned diffusion model that fuses coarse-to-fine text and geometric cues, and (3) a physical refinement module ensuring realistic contact and alignment. Experiments show that TOUCH generates diverse, semantically aligned, and physically plausible free-form interactions, outperforming prior baselines (e.g., ContactGen, Text2HOI) in contact accuracy, plausibility, and diversity metrics.

### Strengths
The paper’s WildO2 dataset is a major technical contribution, featuring an well-designed and automated data generation pipeline. This pipeline successfully integrate multi-stage object–hand reconstruction, camera alignment, and physical contact refinement, resulting in high-quality 3D annotations and realistic HOI samples. The inclusion of Descriptive Synthetic Captions (DSCs), generated and verified through vision–language models, is particularly valuable for enabling text-guided interaction synthesis tasks. 

The dataset specifically targets free-form, non-grasping hand–object interactions—a type of everyday manipulation that is pervasive in the real world but consistently overlooked in prior HOI datasets, which mostly emphasize stable grasping or object holding.  the work fills a clear research gap and opens new possibilities for studying intent-driven, semantically controllable HOI generation in both computer vision and embodied AI 

the proposed TOUCH framework (contact → pose → refinement) follows a fairly typical architecture within current interaction synthesis pipelines, it is well-implemented and well-validated through both quantitative and qualitative experiments. Its role here effectively complements the dataset.

### Weaknesses
The WildO2 dataset primarily focuses on rigid objects, while articulated or deformable objects (e.g., clothes, plastic bags, napkins) are absent. These categories are often the most likely to trigger free-form and dynamic hand–object interactions in everyday activities. Although using rigid objects is acceptable for building an initial benchmark, this omission limits the dataset’s ability to fully capture the spectrum of natural, unconstrained human–object interactions.

Despite the paper’s aim to model free-form interactions, the proposed TOUCH framework largely inherits design principles from grasp-based synthesis—treating contact as a quasi-static grasping state. While this formulation is reasonable for static contact modeling, free-form interactions are inherently motion-centric, and thus would benefit from a dynamic or sequence-level synthesis perspective rather than purely static pose generation.

The use of the term data generation sec 3.2 may be somewhat misleading, as the proposed pipeline mainly performs 3D reconstruction and alignment rather than generative modeling. Although the inclusion of LLM-generated Descriptive Synthetic Captions (DSCs) introduces a generative component, the overall process is better described as a data reconstruction or annotation pipeline to avoid misleading .

### Questions
What is the average processing time per frame in the data generation pipeline for WildO2? It would be helpful to know the computational cost and scalability of the proposed reconstruction and alignment procedure.

 In the WildO2 dataset, how were the object categories and action types selected? Do the defined free-form action labels correspond to common patterns of real-world human activity, or were they primarily derived from the source video dataset?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper tackles free-form HOI generation, pushing the field beyond the standard grasp-centric paradigm. The authors introduce two main contributions: WildO2, a novel 3D HOI dataset reconstructed from in-the-wild videos, and TOUCH, a three-stage diffusion framework designed to synthesize these diverse interactions from fine-grained text prompts. The work is solid, and the results are impressive.

### Strengths
1.	The formulation of the "Free-Form HOI" task is a forward-looking contribution. It moves the community beyond the well-trodden "grasping" paradigm toward more realistic and diverse interactions.
2.	The proposed data pipeline for reconstructing 3D HOIs from monocular videos is effective. Creating an in-the-wild 3D dataset like WildO2 is a valuable asset for the community.
3.	The TOUCH framework is technically sound. Its three-stage approach is a logical decomposition of the problem, and the coarse-to-fine conditioning is an effective strategy for fine-grained text control.

### Weaknesses
1.	The dataset pipeline's quality is naturally capped by its upstream components (e.g., image-to-3D models, SAM2). Existing 3D generation methods are less used for in-the-wild, low-resolution generation, and a discussion on how to improve the accuracy of the dataset synthesis method in the future would be beneficial.
2.	WildO2 excels in interaction diversity. However, its absolute scale is understandably smaller than that of massive lab datasets (e.g., Gigahands). 
3.	The method's generalizability needs more validation. Adding experiments on other datasets/domains would strengthen the paper's claims, e.g., qualitative results on OakInk and tests on open-set object or CAD models from sources like Objaverse.
4.	The 4/4 layer split for coarse-to-fine conditioning appears empirical. An ablation study is needed to justify this specific architectural choice against other alternatives (e.g., 2/6).

### Questions
1.	The reconstruction failure analysis for the dataset is great. Could you also show some typical generation failures of the TOUCH model itself w.r.t. certain objects or text prompts?
2.	What is the inference speed of the full pipeline? How much overhead does the TTA add, and is it critical for performance?
3.	There is an incomplete citation for Ye et al. (L698) and a missing space in 'WildO2that' (L99).

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
5