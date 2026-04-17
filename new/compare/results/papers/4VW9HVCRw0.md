# Touch: Text-Guided Controllable Genera- Tion Of Free-Form Hand-Object Interactions

Guangyi Han1,†, Wei Zhai1,†, Yuhang Yang1, Yang Cao1**, Zheng-Jun Zha**1,∗
1 MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China
{hanguangyi@mail., wzhai056@, yyuhang@mail., forrest@, zhazj@}ustc.edu.cn

## Abstract

Hand-object interaction (HOI) is fundamental for humans to express intent. Existing HOI generation research is predominantly confined to fixed grasping patterns, where control is tied to physical priors such as force closure or generic intent instructions, even when expressed through elaborate language. Such an overly general conditioning imposes a strong inductive bias for stable grasps, thus failing to capture the rich diversity of daily HOI. To address these limitations, we introduce the new task of **Free-Form HOI generation**, which aims to generate controllable, diverse, and physically plausible HOI conditioned on fine-grained intent, including non-grasping actions like pushing, poking, and rotating. To support this task, we construct **WildO2**, the first large-scale, in-the-wild 3D HOI dataset, which includes non-grasping motions derived from internet videos; it contains 4.4k unique interactions across 92 intents and 610 object categories, each with detailed semantic annotations. Building on this rich dataset, we propose **TOUCH**, a three-stage framework centered on a multi-level diffusion model that facilitates fine-grained semantic control to generate versatile hand poses beyond grasping priors. This process leverages explicit contact modeling for conditioning and is subsequently refined with contact consistency and physical constraints to ensure realism. Comprehensive experiments demonstrate our method's ability to generate controllable, diverse, and physically plausible hand interactions representative of daily activities. Project page is https://guangyid.github.io/hoi123touch/.

## 1 Introduction

Hand-Object Interaction (HOI) is fundamental to expressing intent and executing tasks in human daily life, and the ability to generate controllable interactions is crucial for AR/VR, robotics, and embodied AI (Zheng et al., 2025). While existing HOI generation research has progressed from ensuring physical plausibility (Fang et al., 2020) to incorporating semantic controllability (Li et al., 2024b; Yang et al., 2023), its scope remains predominantly confined to a grasp-centric paradigm. The control signals in these methods, whether simple physical constraints like force closure or coarse high-level instructions (e.g., verb-noun pairs), are often overly general. This simplified conditioning imposes a strong inductive bias that primarily favors the generation of stable grasps (Taheri et al., 2020), sacrificing interaction diversity. Furthermore, even with more sophisticated control such as detailed natural language (via LLMs) (Zhang et al., 2025a;b), the underlying model designs and inherent inductive biases are still fundamentally geared towards generating only grasping interactions, driven by historical focus and prevailing representations. Consequently, these approaches lack the fine-grained control and inherent capability to capture the diverse non-grasping interactions found in the real world, including varied hand poses, contact details, and nuanced semantic intent. To bridge the gap between the limited scope of current methods and the complexity of real-world interactions, we introduce the task of **Free-Form HOI Generation**. The goal is to break graspcentric limitations and shift towards generating diverse interactions, including the vast array of nongrasping manipulations. This task emphasizes expressiveness and controllability in the generation 1

![1_image_0.png](1_image_0.png)

Figure 1: **Overview.** We extend HOI generation beyond laboratory "grasp" settings (left) toward broader daily HOI modalities (right), enabling the modeling of more human-like interactions. Our dataset WildO2, built from Internet videos, covers more contacts, more objects, and more actions, and is enriched with descriptive synthetic captions (DSCs) to support fine-grained semantic controllable HOI generation with our method, TOUCH. process, aiming to synthesize interactions that are not only physically plausible but also semantically rich and truly adaptable to complex human intentions. The core challenge of this task lies in two aspects: what to generate and how to generate it. The former pertains to spatial plausibility: the model must break free from restrictive grasping priors (e.g., palm position and orientation, contact region assumption (Ye et al., 2024; Jiang et al., 2021))
to explore a vast yet physically valid interaction space. To address this, we propose that contact relationships serve as a powerful cue to constrain this high-dimensional space, offering a more nuanced understanding of physically valid interactions. The latter pertains to semantic controllability: the model must accurately map fine-grained textual instructions to specific hand configurations and contact regions. The prior knowledge within Large Language Models (LLMs) offers a promising pathway for this guidance (Tang et al., 2023). A major obstacle to learning this complex mapping is the lack of 3D training data for diverse daily interactions, as existing datasets (Zhan et al., 2024; Fu et al., 2025) are mostly limited to lab-based grasping and object instances. Hardware and capture challenges make large-scale real-world 3D data collection difficult. In contrast, abundant 2D HOI videos online provide rich and realistic daily interaction behaviors. To tackle the proposed task and challenges, we present TOUCH, a three-stage framework for controllable free-form HOI generation. First, we explicitly model the contact on the surfaces of the hand and the object separately by jointly encoding spatial point-cloud relations and semantic information, providing strong spatial priors to mitigate uncertainty from the high degrees of freedom in interaction position and pose. We further incorporate part-level hand modeling for more precise action control. Second, we employ a multi-level diffusion model with attention-based fusion of semantics and geometry: coarse-grained intent and global object geometry guide the early diffusion stages, while fine-grained text and local contact features refine detailed motions in deeper stages, enabling fine-grained semantic controllability. Finally, we introduce self-supervised contact consistency and physical plausibility constraints to optimize the generated interactions, ensuring realism and physical feasibility. Compared to prior methods restricted to grasp generation, TOUCH naturally generalizes to diverse free-form HOI such as pushing, pressing, and rotating. Additionally, based on 3D object reconstruction (Xu et al., 2024), we introduce an automated pipeline to build the dataset WildO2 that jointly recovers and optimizes high-quality 3D hand-object interaction samples from internet videos annotated with interaction intent. By leveraging visionlanguage models (Bai et al., 2023b), we generate fine-grained semantic annotations, resulting in the 3D daily HOI dataset covering diverse interaction intents. Our main contributions are: (1) We propose to extend the HOI from constrained grasping to a broader, more realistic, and more diverse set of daily interactions. (2) We propose TOUCH, a new framework that can generate natural, physically reasonable, and diverse free-form HOI under finegrained text guidance. (3) We build an automated pipeline and construct WildO2, an in-the-wild 3D dataset for daily HOI, providing a critical resource that enables future research in this domain. Extensive experiments demonstrate the superiority of TOUCH.

## 2 Related Work 2.1 Hand-Object Interaction Datasets.

Existing 3D hand-object interaction (HOI) datasets are predominantly collected in controlled laboratory settings, relying either on physics-based simulation synthesis (Hasson et al., 2019) or motion capture systems to record real interactions (Hampali et al., 2020; Liu et al., 2022; Yang et al., 2022; Brahmbhatt et al., 2020). Although these datasets provide valuable support for modeling 3D HOI, they suffer from limited diversity due to constrained camera setups, a small number of participants, and a restricted set of object instances. In contrast, large-scale in-the-wild video datasets (Damen et al., 2020; Grauman et al., 2022; Shan et al., 2020) contain abundant HOI clips, but lack highquality 3D annotations. Some studies have attempted to annotate subsets of these videos in 3D using object template-based optimization methods (Cao et al., 2021; Patel et al., 2022); however, due to the high diversity of open-set objects, scaling such approaches remains challenging.

## 2.2 Template-Free Hoi Reconstruction.

The core bottleneck in reconstructing HOIs in the wild has long been the recovery of diverse object geometries. While existing template-free approaches (Fan et al., 2024; Ye et al., 2022) avoid predefined object model constraints, they are typically trained on limited datasets and exhibit poor generalization to novel objects. In recent years, multi-view diffusion models (Liu et al., 2023a) and large-scale reconstruction models (LRMs) (Hong et al., 2024) have enabled high-quality 3D mesh reconstruction directly from single images (Xu et al., 2024; Liu et al., 2024b) or text prompts (Poole et al., 2022), demonstrating strong generalization capabilities. Motivated by these advances, several HOI studies have explored image-to-3D reconstruction pipelines to handle open-set objects in the wild. However, due to severe hand occlusion, these methods often rely on image inpainting to complete occluded regions (Tian et al., 2025; Liu et al., 2024a; Wen et al., 2025; Liu et al., 2024c), or employ text-to-3D generation to align with coarse reconstruction results (Wu et al., 2024; Chen et al., 2025). Nonetheless, most of these pipelines depend on heuristic completion or registration strategies, resulting in limited geometric consistency with the input, and have yet to be validated at scale in an automated manner.

## 2.3 Data-Driven Controllable Hoi Generation.

In the evolution of HOI generation, interaction guidance has progressively advanced: from coarse control based on grasp type (Feix et al., 2015), to object-conditioned generation (Karunratanakul et al., 2020; Jiang et al., 2021), and further to task/action-level intent constraints (Christen et al., 2024; Yang et al., 2024b;a; Yu et al., 2025). To enhance physical plausibility, contact penetration loss and hand anatomical constraints have been widely adopted (Wei et al., 2024). Additionally, explicit modeling of hand part segmentation and contact relationships with objects has been shown to improve physical realism and detailed expression of interactions (Liu et al., 2023b; Zhang et al., 2024; Li et al., 2024a). Building on these efforts, we propose a multi-level controllable generation framework trained on our newly constructed daily HOI dataset, enabling finer-grained semantic intent control and the flexible generation of free-form HOIs that align with complex human intentions.

## 3 Dataset 3.1 Data Collection And Processing

Our goal is to construct a diverse dataset of 3D hand-object interactions from in-the-wild videos. A primary challenge in this process is the severe occlusion of the object by the hand, which compromises the quality of 3D object reconstruction. To address this, we introduce a semi-automated

![3_image_0.png](3_image_0.png)

data reconstruction and annotation pipeline, centered around a novel Object-only to Hand-Object Interaction (O2HOI) frame pairing strategy. We begin by filtering the Something-Something V2 dataset (Goyal et al., 2017), which is rich in goal-directed human actions, to obtain 8k single-hand, single-object interaction clips. For each clip, we automatically extract an O2HOI pair (details in Appendix): an object-only frame Iref , where the object is unoccluded, and a corresponding interaction frame Ihoi. To obtain a complete object mask in the interaction frame, we segment the object in Iref using SAM2 (Ravi et al., 2024) and then transfer this mask to Ihoi via a robust dense matching model (Edstedt et al., 2024), yielding Minpaint. This mask transfer strategy offers a distinct advantage over common alternatives: it avoids the geometric inconsistencies of diffusion-based inpainting (Liu et al., 2024a) while being significantly more scalable than manual completion (Wen et al., 2025). Consequently, our approach facilitates the automated, large-scale generation of high-fidelity 3D assets for reconstruction.

## 3.2 Data Reconstruction Pipeline

Based on the O2HOI pairs, we build a three-stage generation pipeline to recover 3D HOI.

Stage 1: Initialization. For each pair, we reconstruct a textured object mesh VO
recon from the object-only frame Iref using an image-to-3D model (Xu et al., 2024). Concurrently, we estimate initial MANO (Romero et al., 2017) hand parameters Hinit from the interaction frame Ihoi using a state-of-the-art hand reconstruction method (Pavlakos et al., 2024). Stage 2: Camera Alignment. A challenge arises from coordinate system misalignment: the object mesh VO
recon is created in a canonical space of the object-only frame Iref , while the hand exists in the camera space of the interaction frame Ihoi. To unify them, we align VO
recon to an object-centric global coordinate system relative to the interaction frame by optimizing the camera projection matrix K and extrinsics (R, t). This is achieved by minimizing a camera alignment loss, Lcam, via differentiable rendering. The optimization proceeds in two phases: we initially use mask IoU, Sinkhorn (Cuturi, 2013) loss, and an edge penalty term (to prevent the object from moving out of view). Once the IoU surpasses a threshold, we introduce scale-invariant depth (Eigen et al., 2014) and RGB reconstruction losses for fine-tuning. The overall objective is formulated as:

$$\operatorname*{min}_{\mathbf{K},\mathbf{R},\mathbf{t}};L_{\mathrm{cam}}=L_{\mathrm{mask}}+L_{\mathrm{sinhhorn}}+L_{\mathrm{edge}}+\lambda_{\mathrm{fine}}(L_{\mathrm{depth}}+L_{\mathrm{rgb}}).$$

Stage 3: Hand-Object Refinement. With the aligned camera and object, we refine the initial hand parameters Hinit to achieve physically plausible contact. Specifically, we cast rays from the camera center through pixels within the interaction mask Minpaint. The intersection points of these rays with the 3D hand and object geometries define a potential 3D contact zone. We then optimize H
using a refinement objective Lalign, which combines 2D evidence with 3D physical constraints: hand

$$(1)$$

![4_image_0.png](4_image_0.png)

mask IoU (L
H
mask), 2D joint reprojection error (Lj2d), an ICP loss on the 3D contact zone (Licp),
and physical constraints for contact, penetration, and anatomy based on (Yang et al., 2021).

min H
;Lalign = L
H
mask + Lj2d + Licp + Lphy, Lphy = Lcontact + Lpene + Lanatomy + L*self*. (2)
This pipeline yields 4,414 high-quality 3D hand-object interaction samples after a final stage of manual inspection and refinement, which constitute the ground truth of our dataset.

## 3.3 Data Annotation And Statistics

We enrich our dataset with a multi-level annotation system, generating over 44k annotations. A statistical overview is provided in Fig. 3, with further details in the Appendix.

3D Geometry and Transformation. Each sample includes the final hand-object meshes (VˆH, VˆO)
and the corresponding camera parameters derived from our generation pipeline. **Contact Maps.** We compute dense contact maps between the hand and object surfaces. To handle varying object scales, our method robustly identifies contact regions by combining relative and absolute distance thresholds with bidirectional nearest-neighbor filtering. **Multi-Level Language Descriptions.** We provide two levels of textual descriptions. We inherit the template-based Short Synthetic Captions (SSCs) from Something-Something V2 (e.g., "picking [Something] up"). Additionally, we use a Vision-Language Model (VLM) (Bai et al., 2023b) to generate more detailed Descriptive Synthetic Captions (DSCs), which are manually verified for quality and relevance. **Fine-Grained Hand Part** Segmentation. We segment the hand mesh into 17 parts, including finger pads, nails, knuckles, palmar, and the dorsal region. This partitioning scheme goes beyond the coarse divisions commonly used in grasp generation tasks (Hasson et al., 2019; Liu et al., 2023b)—which often focus only on contact on the inner hand—by also accounting for contact on the dorsal side. This fine-grained segmentation supports detailed local interaction analysis and facilitates alignment with the semantic descriptions in the DSCs.

## 4 Method

This work aims to generate natural and physically plausible hand-object interaction (HOI) poses, parameterized by H, along with corresponding contact maps CH and CO, conditioned on a multi-level textual prompt T and an object mesh VO. To tackle this problem, we propose a three-stage framework, as illustrated in Fig. 4. Specifically, the Contact Map Prediction module (Sec. 4.1) infers the potential contact regions on the hand and object surfaces based on the text and object geometry. The Multi-Level Conditioned Diffusion module (Sec. 4.2) synthesizes a coarse hand pose by integrating coarse-to-fine textual and geometric features within a diffusion framework, ensuring alignment with multi-level constraints. Finally, the Physical Constraints Refinement module (Sec. 4.3) further optimizes the coarse pose to enhance contact realism and prevent penetrations.

![5_image_0.png](5_image_0.png)

## 4.1 Contact Map Prediction

To generate diverse interactions beyond simple grasping, we design two independent yet similar CVAEs (Sohn et al., 2015) to generate binary contact maps for the object and the hand, respectively.

For the object branch, we sample a point cloud PO ∈ R
NO×3(NO = 3000) from its mesh VO,
normalize it, and record the scale factor sO. We use PointNet (Qi et al., 2016) to extract its geometric features, which are concatenated with sO to form the object condition FO. For the hand branch, we generate a canonical point cloud P0H ∈ R
NH×3(NH = 778) from MANO's zero pose and shape parameters H0. This point cloud, combined with a hand-part mask initialized from the fine-grained text TDSC , is processed by PointNet to obtain the hand condition FH. This design integrates the topological structure of the point clouds with text-guided emphasis on interaction-relevant hand regions. Both CVAEs are trained conditioned on their respective geometric features (FO, FH) and a shared text feature FDSC = ftext(TDSC ), which is extracted using the Qwen-7B (Bai et al., 2023a)
processed through a lightweight adapter. The optimization objective is a composite loss function:

$$d i c e+\beta L_{K L},$$

Lcontact = Lfocal + L*dice* + βLKL, (3)
where L*focal* and L*dice* supervise the contact prediction, and LKL structures the latent space. During inference, under the conditional features (FO, FH, FDSC ), the model samples from a Gaussian prior z ∼ N (0, I) and decodes it to produce the predicted binary contact maps Cˆ O ∈ {0, 1}
NO×1 and Cˆ H ∈ {0, 1}
NH×1.

## 4.2 Multi-Level Conditioned Diffusion

The core of our method is a Transformer-based Denoising Diffusion Probabilistic Model (DDPM)
(Ho et al., 2020) that synthesizes hand pose parameters Hˆ conditioned on the object point cloud PO, multi-level text T, and predicted contact maps Cˆ . Instead of predicting noise, our model fθ is trained to directly predict the denoised data xˆ0 = fθ(xt*, t,* y), optimized with an L2 loss on the pose parameters: Ldiff = Et,ϵ-∥xˆ0 − x0∥
2.

Condition Generation: Transformer Inputs. To achieve precise control, our model extracts multilevel conditional features from both geometric and textual modalities. On the geometric side, we use PointNet to extract global features F
O
glb, F
H
glb and point-wise local features from the object point cloud PO, the initial hand point cloud P0H, and the predicted contact maps Cˆ from the previous stage. To focus on interaction regions, we leverage Cˆ to adaptively select features of N O
loc = 128 object points and N H
loc = 64 hand points near contact areas, yielding F˜ O
loc and F˜ H
loc. On the textual side, we utilize ftext to extract both coarse-grained F
SSC
qwen = f*text*(TSSC) and fine-grained F
DSC
qwen text features.

Conditional Injection: Coarse-to-Fine Control. We inject these features into the Ninj = 8 blocks of our Transformer model in a hierarchical, coarse-to-fine fashion. This design ensures that global context, defined by SSCs and global geometry, shapes the overall pose in early denoising stages, while local details, defined by DSCs and contact-point features, are refined in later stages. Specifically, for the i-th Transformer block: Early Stages (i < 4): Global context is injected, with no local features.

y i glb = concat(F
O
glb, F
H
glb, sO, F
SSC
qwen , t), y i loc = ∅. (4)
Later Stages (4 ≤ *i < N*inj): Local details are injected, switching to fine-grained conditions.

y i glb = concat(F
H
glb, sO, F
DSC
qwen , t), y i loc = concat(F˜ O
loc, F˜ H
loc). (5)
To prevent over-reliance on any single condition and enhance robustness, we randomly drop each component of the global condition with a 10% probability during training. Within each block, the global condition y iglb modulates the main features via FiLM (Perez et al., 2018), while the local condition y i loc is integrated through cross-attention to provide fine-grained spatial cues. This dual mechanism effectively decouples global contextual guidance from local geometric refinement. Finally, the updated latent goes through self-attention and a Feed-Forward Network (FFN). Training Loss. To improve training stability and spatial alignment, we introduce two auxiliary losses alongside the primary diffusion loss Ldiff. A global pose loss directly supervises the hand's global rotation rrot and translation T to prevent overall pose drift, an issue exacerbated when directly regressing H, which comprises parameters with disparate numerical ranges (e.g., shape β, pose Θ,
rrot, T). A distance map loss ensures precise contact by supervising the distance map dmap ∈
R 
21×NO from the 21 hand joints to the object surface. The final objective is a weighted sum:
Ltotal = Ldiff + λglobal|ˆrrot − r gt rot| + |Tˆ − T
gt|
+ λdmap|dˆmap − d gtmap|. (6)

$$\mathbf{l}_{\mathrm{map}}^{\sharp}|.$$

$$\mathbf{T}-\mathbf{T}^{2}$$
$|\mathbf{u}\rangle$

## 4.3 Physical Constraints Refinement

To address the common issue of global pose drift in free-form HOI generation, where the hand often fails to make contact with the object, we introduce an efficient physical refinement module. This module is powered by a refiner network, frefiner, which inherits the Transformer architecture of our diffusion model. The process begins with a single forward pass to rapidly correct the global positioning of the initial pose Hˆdiff, establishing primary physical contact. Subsequently, this corrected pose undergoes Ntta iterations of test-time optimization (TTA) to fine-tune local contact details, such as finger placements.

The entire optimization is guided by our self-supervised cycle-consistency loss (Lcyc), which enforces bidirectional mapping consistency between hand and object contact surfaces. The core idea is that a hand contact point, after being mapped to the nearest object point via Φ (hand-to-object), should map back to its original location via the reverse mapping Ψ (object-to-hand), and vice versa.

This loss acts as a powerful regularizer, effectively reducing the ambiguity inherent in the mappings.

We combine this with Lphy (see Eq. 2). The total refinement loss is defined as:
Lrefiner = Lphy + λcyc(EPh∈PCH
||Ψ(Φ(Ph)) − Ph||1 + EPo∈PCO
||Φ(Ψ(Po)) − Po||1). (7)

## 5 Experiments 5.1 Experimental Settings

Our experiments are conducted on the WildO2 dataset. For each hand part contact category, we perform a random 4:1 split, yielding approximately 3.7k training and 677 test samples. To address the long-tailed distribution of hand part labels, we aggregate 10 less frequent hand part categories and then apply resampling using unique 7-bit labels to balance the data. The model is trained for 1000 epochs using the Adam optimizer with a learning rate of 1e-4 and a batch size of 128. The diffusion model's parameters are frozen during the training of the refiner module. We evaluate our method from four perspectives: (1) Contact Accuracy, assessed by IoU and F1-score against ground-truth contacts parts. (2) Physical Plausibility, measured by Mean Per-Vertex Position Error
(MPVPE), Penetration Depth (PD), and Penetration Volume (PV). Note that unlike works focusing on grasping (Jiang et al., 2021), we do not employ physics engine-based stability simulation metrics, as our scope of interactions is broader than force-closure grasps. (3) Diversity, quantified by entropy and cluster size. (4) Semantic Consistency, evaluated using a point cloud-based FID (P-FID) (Nichol et al., 2022), VLM assisted evaluation, and a perceptual score (PS) from 10 users.

![7_image_0.png](7_image_0.png) 

## 5.2 Comparisons

As existing methods have not explored fine-grained controlled HOI generation, we select two representative types of baselines: (1) *ContactGen* (Liu et al., 2023b): an object-conditioned multi-layer CVAE using coarse hand part labels. (2) *Text2HOI* (Cha et al., 2024): a transformer-based conditional diffusion model guided by coarse text conditions. We remove its temporal axis and adapt it for our setting. Compared to typical grasping datasets, hand poses in WildO2 exhibit higher degrees of freedom. Both baseline methods exhibit noticeable overall hand drift. To ensure fair comparison, we also augment them with an optimization-based post-processing module to correct hand poses. Experimental results in Tab. 1 show that our method outperforms baselines across most metrics. Visual results in Fig. 5 further demonstrate that our method generates more realistic HOI poses that better align with input text descriptions.

| Method     | Contact Acc.   | Physical Plausibility   | Diversity   | Semantic Consistency   |       |      |        |       |     |     |
|------------|----------------|-------------------------|-------------|------------------------|-------|------|--------|-------|-----|-----|
| P-IoU↑     | P-F1↑          | MPVPE↓                  | PD↓         | PV↓                    | Ent.↑ | CS↑  | P-FID↓ | VLM↑  | PS↑ |     |
| ContactGen | 0.620          | 0.730                   | 5.46        | 1.296                  | 7.37  | 2.85 | 4.93   | 6.08  | 4.8 | 6.3 |
| Text2HOI   | 0.711          | 0.795                   | 4.69        | 1.239                  | 4.93  | 2.85 | 5.20   | 15.72 | 6.5 | 7.5 |
| Ours       | 0.776          | 0.844                   | 2.97        | 0.932                  | 2.67  | 2.93 | 5.40   | 4.13  | 7.1 | 8.8 |

## 5.3 Ablation Study

To evaluate the effectiveness of our contact-guided generation of spatial relations in HOI and the coarse-to-fine text control design, we conduct ablation studies as shown in Tab. 2, ✗ means without.

For clarity, TTA is disabled. We argue for the primacy of contact metrics, as penetration metrics, PD and PV are meaningful only after hand-object contact is established; otherwise, they can be misleading. This is starkly exemplified by the "✗ refiner" variant, which scores poorly on contact yet achieves deceptively low PD/PV values simply because the generated hand drifts away from the

Metrics P-IoU↑ P-F1↑ MPVPE↓ PD↓ PV↓ P-FID↓

Ours(✗ **TTA)** 0.728 0.805 3.00 1.093 4.82 4.84

✗

 hoc. 0.492 0.611 4.93 1.330 5.50 5.41

✗

 refiner 0.513 0.621 5.05 0.723 2.98 5.84

✗

 Lcyc. 0.702 0.787 3.00 1.100 5.29 5.79

✗

 mul. 0.525 0.631 5.00 1.464 6.52 6.84

✗

 TDSC 0.698 0.784 3.02 1.119 5.28 6.09

✗

 TSSC 0.687 0.778 2.92 1.119 5.17 5.52

CLIP 0.713 0.798 2.87 1.136 4.85 4.84

BERT 0.705 0.790 2.91 1.182 4.99 6.08

MPNet 0.704 0.788 2.87 1.114 5.06 6.02

![8_image_0.png](8_image_0.png)

Table 2: Analyzes the contributions of various components, including the absence of MO and MH (hoc.) for guiding spatial relationship generation, the multi-level network structure (mul.), and the multi-level text (✗
TDSC , ✗ TSSC ).

Figure 6: Contact guidance visualization.

![8_image_1.png](8_image_1.png)

object, thus avoiding interaction entirely. This distinguishes our complex task from traditional grasping, where contact is facilitated by priors. The consistent degradation in contact performance upon removing any module confirms their synergistic importance. Fig. 6 offers a qualitative visualization of the step-by-step improvements afforded by our contact guidance.

## 5.4 Discussion 5.4.1 Ablation On Text Encoders.

We replace the text encoder with other common token- or sentence-level encoders (e.g., CLIP (Radford et al., 2021), BERT (Devlin et al., 2019), MPNet (Song et al., 2020)) to analyze their impact on generation quality. Results indicate that Qwen-7B offers better performance in capturing finegrained semantic details, detailed in Tab. 2.

## 5.4.2 Out-Of-Domain Generation.

To evaluate generalization, we sample novel object CAD meshes from the large-scale 3D dataset Objaverse (Deitke et al., 2023), utilizing LLMs to generate captions that emulate our DSC format. As visualized in Fig. 7, our approach successfully produces plausible interaction poses for these out-of-domain models, demonstrating strong generalization capability.

## 5.4.3 Semantic Controllability

Controllable Semantic Generation. Our model demonstrates strong controllability and high semantic faithfulness in interpreting fine-grained user intents. As shown in Fig. 8, by varying textual control signals (such as contact regions and interaction semantics (e.g., Push/Lift) for an object), the model successfully produces diverse and physically plausible hand poses, validating its ability to execute multi-faceted semantic directives. Semantic Nuances of Force Expression. Although the model does not explicitly model physical forces, it learns to associate force-related terms like "firmly" and "gently" with contact geometry. It generates larger, denser contacts for "firm" prompts and more marginal, sparser contacts for "gentle" ones, as shown in Fig. 9. Quantitative analysis on WildO2 confirms this finding, revealing a 22-25% larger average contact area for "firm/tight" interactions.

![9_image_0.png](9_image_0.png)

## 6 Conclusion

In this paper, we addressed the limitations of grasp-centric approaches by introducing the Freeform HOI Generation task. Our work expands the synthesis paradigm beyond simple grasping to a broader, more semantically expressive spectrum of interactions. To support this, we built an automated pipeline to construct WildO2, an in-the-wild 3D dataset for daily HOIs, providing a critical resource to enable future research in this domain. Limitations and Future Directions. Our framework currently focuses on static HOI snapshots, which inherently limits its ability to capture the temporal dynamics of an interaction process. While our pipeline offers rapid expansion, the current dataset scale also presents an area for future growth. In the future, we plan to extend our work to dynamic sequences by leveraging large-scale video datasets and incorporating 6-DoF object pose estimation, thus modeling the entire humanenvironment interaction process.

## 7 Acknowledgments

This work is supported by the National Natural Science Foundation of China (NSFC) under Grants 62225207, 62436008, 62306295, 62576328, and 625B2175. The AI-driven experiments, simulations and model training were performed on the robotic AI-Scientist platform of Chinese Academy of Sciences.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023a.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv preprint arXiv:2308.12966*, 2023b.

Samarth Brahmbhatt, Chengcheng Tang, Christopher D Twigg, Charles C Kemp, and James Hays.

Contactpose: A dataset of grasps with object contact and hand pose. In European Conference on Computer Vision, pp. 361–378. Springer, 2020.

Zhe Cao, Ilija Radosavovic, Angjoo Kanazawa, and Jitendra Malik. Reconstructing hand-object interactions in the wild. ICCV, 2021.

Junuk Cha, Jihyeon Kim, Jae Shin Yoon, and Seungryul Baek. Text2hoi: Text-guided 3d motion generation for hand-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1577–1585, 2024.

Hongyi Chen, Yunchao Yao, Yufei Ye, Zhixuan Xu, Homanga Bharadhwaj, Jiashun Wang, Shubham Tulsiani, Zackory Erickson, and Jeffrey Ichnowski. Web2grasp: Learning functional grasps from web images of hand-object interactions, 2025. URL https://arxiv.org/abs/2505. 05517.

Sammy Christen, Shreyas Hampali, Fadime Sener, Edoardo Remelli, Tomas Hodan, Eric Sauser, Shugao Ma, and Bugra Tekin. Diffh2o: Diffusion-based synthesis of hand-object interactions from textual descriptions. In *SIGGRAPH Asia 2024 Conference Papers*, pp. 1–11, 2024.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in neural* information processing systems, 26, 2013.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. The epic-kitchens dataset: Collection, challenges and baselines. *IEEE Transactions on Pattern* Analysis and Machine Intelligence (TPAMI), 2020.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects. *arXiv preprint arXiv:2307.05663*, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Johan Edstedt, Qiyu Sun, Georg Bokman, M ¨ arten Wadenb ˚ ack, and Michael Felsberg. RoMa: Robust ¨
Dense Feature Matching. *IEEE Conference on Computer Vision and Pattern Recognition*, 2024.

David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. *Advances in neural information processing systems*, 27, 2014.

Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu Chen, Muhammed Kocabas, Michael J Black, and Otmar Hilliges. Hold: Category-agnostic 3d reconstruction of interacting hands and objects from video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 494–504, 2024.

Hao-Shu Fang, Chenxi Wang, Minghao Gou, and Cewu Lu. Graspnet-1billion: A large-scale benchmark for general object grasping. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11444–11453, 2020.

Thomas Feix, Javier Romero, Heinz-Bodo Schmiedmayer, Aaron M Dollar, and Danica Kragic.

The grasp taxonomy of human grasp types. *IEEE Transactions on human-machine systems*, 46 (1):66–77, 2015.

Rao Fu, Dingxi Zhang, Alex Jiang, Wanjia Fu, Austin Funk, Daniel Ritchie, and Srinath Sridhar.

Gigahands: A massive annotated dataset of bimanual hand activities. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17461–17474, 2025.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al.

The "something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pp. 5842–5850, 2017.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18995–19012, 2022.

Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In CVPR, 2020.

Yana Hasson, Gul Varol, Dimitris Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and ¨
Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In *CVPR*,
2019.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in* neural information processing systems, 33:6840–6851, 2020.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d, 2024. URL https://arxiv.org/abs/2311.04400.

Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang. Hand-object contact consistency reasoning for human grasps generation. In Proceedings of the International Conference on Computer Vision, 2021.

Korrawe Karunratanakul, Jinlong Yang, Yan Zhang, Michael Black, Krikamol Muandet, and Siyu Tang. Grasping field: Learning implicit representations for human grasps, 2020. URL https: //arxiv.org/abs/2008.04451.

Hongxiang Li, Yaowei Li, Yuhang Yang, Junjie Cao, Zhihong Zhu, Xuxin Cheng, and Long Chen.

Dispose: Disentangling pose guidance for controllable human image animation. *arXiv preprint* arXiv:2412.09349, 2024a.

Kailin Li, Jingbo Wang, Lixin Yang, Cewu Lu, and Bo Dai. Semgrasp: Semantic grasp generation via language aligned discretization. In *European Conference on Computer Vision*, pp. 109–127.

Springer, 2024b.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick.

Zero-1-to-3: Zero-shot one image to 3d object. In *Proceedings of the IEEE/CVF international* conference on computer vision, pp. 9298–9309, 2023a.

Shaowei Liu, Yang Zhou, Jimei Yang, Saurabh Gupta, and Shenlong Wang. Contactgen: Generative contact modeling for grasp generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20609–20620, 2023b.

Yumeng Liu, Xiaoxiao Long, Zemin Yang, Yuan Liu, Marc Habermann, Christian Theobalt, Yuexin Ma, and Wenping Wang. Easyhoi: Unleashing the power of large models for reconstructing hand-object interactions in the wild. *arXiv preprint arXiv:2411.14280*, 2024a.

Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21013–21022, June 2022.

Zhiheng Liu, Ka Leong Cheng, Qiuyu Wang, Shuzhe Wang, Hao Ouyang, Bin Tan, Kai Zhu, Yujun Shen, Qifeng Chen, and Ping Luo. Depthlab: From partial to complete. arXiv preprint arXiv:2412.18153, 2024b.