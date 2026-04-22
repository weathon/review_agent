# MAPLE: Context-aware Multimodal Augmentation for Long-tail 3D Object Detection

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
3D object detection is essential for autonomous driving but remains limited by the long-tail distribution of real-world data. 
Instance-level augmentation methods, whether copy-paste or asset rendering, are typically restricted to LiDAR and offer only modest variation with limited scene context. 
We introduce MAPLE, a training-free pipeline for multimodal augmentation that generates synchronized RGB-LiDAR pairs. 
Objects are inserted through context-aware inpainting in the image domain, and paired pseudo-LiDAR is reconstructed via depth estimation. 
To ensure cross-modal plausibility, MAPLE incorporates semantic and geometric verification modules that filter inconsistent generations. 
We further propose a success-rate evaluation that quantifies error reduction across verification stages, providing a principled measure of pipeline reliability. 
On the nuScenes benchmark, MAPLE consistently improves both detection and segmentation in multimodal and LiDAR-only settings. 
Code will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a training-free, context-aware multimodal augmentation pipeline for long-tail 3D detection that inserts rare objects via diffusion-based image inpainting guided by VLM descriptions, reconstructs paired pseudo-LiDAR with depth estimation, and enforces semantic geometric verification alongside a new success-rate reliability evaluation.

### Strengths
* The paper frames the long-tail data challenge in 3D perception and the limits of current instance-level, LiDAR-only augmentation.
* The combination of VLM-generated subclass prompts and diffusion inpainting yields objects that blend with scene geometry and occlusions addressing a weakness of 2D/3D placement.

### Weaknesses
* Object subclass and physical size are sourced from a VLM, so misclassification or hallucinated dimensions can misguide both inpainting and scaling.
* All downstream results are on nuScenes, external datasets or real-vehicle studies are absent, limiting claims about generalization.

### Questions
Additional experiments on external datasets would help validate the effectiveness of the method.

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
The paper introduces MAPLE, a training-free, context-aware, multi-modal instance-augmentation pipeline for long-tail 3D perception. Objects are first inpainted in RGB images using VLM-guided textual subclass prompts. Then, pseudo-LiDAR instances are reconstructed from monocular depth and rescaled to match size priors. Finally, semantic (prompt/region consistency) and geometric (spatial filtering + size-prior) verification remove implausible generations. The authors additionally propose a success-rate evaluation to quantify how verification reduces the end-to-end failure rate. On nuScenes, MAPLE improves both multimodal (BEVFusion) detection and 3D semantic segmentation, and also yields competitive LiDAR-only gains.

### Strengths
1. Training-free, modular pipeline composing strong 2D foundation models; no retraining needed.
2. Verification matters: equations + empirical reductions from P(Fnaive) to P(Fverified) (e.g., bicycles 39.7%→2.4%).
3. Zero-shot rare-class recovery (bicycle 8.021 AP with no real bicycles).
4.Cost disclosure and module runtime (useful for practitioners).

### Weaknesses
1. Limited scope of evaluation. Only nuScenes is used; no results on Waymo or KITTI. Gains on detection are modest (e.g., +1.10 mAP for BEVFusion; CenterPoint roughly on par), which raises concerns about generality/impact for detection-centric settings.
2. Depth dependence & priors. Pseudo-LiDAR relies on monocular depth plus single-axis size scaling (fixing sz, leaving sx, sy flexible). This can bias aspect ratios and headings; ablations on different depth estimators or λ-range in size-prior checks are absent in the main paper.
3. “First” claim should be tempered. The paper positions MAPLE as the first training-free multimodal augmentation with verification. While the combination is compelling, prior works already explore training-free LiDAR-only augmentation and VLM/diffusion-assisted pipelines; please clarify boundaries of novelty and cite broader contemporaries (e.g., data synthesis pipelines for multi-sensor fusion).
4. The strongest numbers are in segmentation; for detection, motorcycle FID is worse than PGT-Aug (5.10 vs 3.70). A deeper analysis of failure modes (thin structures, spokes, handle bars) would help interpret when MAPLE is preferable.
5. Typos and artifacts. e.g.,:
1) Section title: “ZERO-SHOPT” (should be zero-shot). 
2) Page 5: “nstead of” (missing “I”). 
3) Section E.2 prompt: “TARGET_LABEL â´L´L {bicycle, motorcycle}” (encoding/∈ issue), etc.

### Questions
1. Can the authors report results on Waymo (at least a subset) or KITTI-360 to demonstrate cross-dataset robustness?
2. Please quantify the temporal realism of multi-sweep simulation (e.g., per-sweep Chamfer to real sweeps, object-track smoothness, or optical-flow/scene-flow consistency).
3. How sensitive are results to the depth estimator choice and the λmin/λmax bounds in the size-prior check? Provide ablations.
4. Could the authors release verification prompts and a deterministic seed/config bundle to ensure reproducibility given fast-evolving foundation models?
5. For the zero-shot experiment, what is the precision/recall curve and the number of true positives? The table shows AP=8.021—please add qualitative error analysis.

### Soundness
3

### Presentation
2

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
This paper presents a pipeline for augmenting camera-lidar data for autonomous driving scenes, called MAPLE. The pipeline combines a myriad of off-the-shelf foundation models for: VLMs for description generation, image inpainting, mono-to-depth prediction, image-to-video generation, SAM for image segmentation mask, etc.
To insert a new object into a target scene, first a VLM is prompted to suggest a type of object with certain properties, which is then inpainted into the target's scene image. Depth prediction of the image is used to suggests a 3d structure, which can be converted into virtual lidar scans. Image-to-video generation is used for short term object dynamics (e.g. moving limbs), since lidar-based object detection typically requires combining multiple lidar scans.
An final verification step on the geometry and semantics helps identify if the generated result is realistically embedded in the original scene.
In experiments on nuScenes, MAPLE is used to augment infrequent classes: construction vehicles, motor cycles, bicycles. Results show that MAPLE augmentation improves segmentation performance over other data augmentation techniques, and detection results are on par with prior work. Analysis further shows how the spatial distribution of MAPLE's inserted objects is distinct from the grid-like patterns of prior augmentation works.

### Strengths
* Data augmentation is a key task to produce robust perception for automated driving; Addressing multi-modal augmentation for synchronized multi-sensor setups is a relevant task.
* The new MAPLE data augmentation technique does not require training, as it relies on powerful foundation models pre-trained for specific tasks which are readily available nowadays. It seems to be a sensible modern approach to data augmentation.
* MAPLE generates both camera and lidar data, including segmentation masks, and also considers real-world details such as combining multiple lidar sweeps, and the motion between those sweeps.
* Based on presented qualitative results, generated samples appear varied, and reasonably realistic. MAPLE includes a geometric verification step to check for realistic results; Analysis of the real and generated data distribution show good overlap with real data (similar to Text3DAug baseline).
* Data augmentation for less common classes with MAPLE brings strong performance improvements on segmentation tasks, compared to baseline augmentation techniques.

### Weaknesses
* This pipeline addresses a data-augementation challenge which is relevant for researchers in the autonomous driving application domain. However, the method itself does not provide particularly novel insights on (applied) machine learning itself. I therefore consider this work better suited for a conference related to computer vision or autonomous driving, rather than ICLR.

* If I understand correctly, experiments are performed using OpenAI's GPT-4 using a remote API, and not using open source model parameters. Although the abstract indicates code will be released to support reproducibility, using a commercial closed-source model will not guarantee others could reproduce the results in the future (even with the promised code release). It would have been better to also include experiments using open source VLMs to ensure future reproducibility.

* The paper claims to address long-tail 3D object detection (paper title), but the the prompt in line 174 specifically states "{TARGET_LABEL} commonly found in urban environments", and placement of generated objects is context-aware. As the examples in Figure 4 show, the samples appear to improve intra-class variation and class-imbalance in the training data of typical objects in typical settings, but not really rare objects and/or in rare situations that constitute the really difficult cases (cars on sidewalks, near crashes, furniture on the road, etc.). This doesn't mean the method isn't useful, but claiming it addresses the long-tail seems to be an overclaim.

* Contribution ③, which is about demonstrating the MAPLE method on nuScenes, is not a separate contribution from Contribution ①, the MAPLE method itself. Performing experiments to demonstrate a new method is an expected requirement for a methodological contribution, and thus not a separate contribution by itself.

* The method inserts generated pointclouds into the lidar pointcloud, but the paper does not describe if this process also produces "shadows" behind the generated object, which would hide regions in the scene that have become occluded. If ommitted, it would severely limit the realism of the lidar simulation.

* Experiments, section 5.1: Implementation details are sparse, and more details could be given about how the augmented dataset compares to the original unaugmented one: Overall size, resulting class distribution (vs original), but also practical details as how long it took to generate the augmented data (inc used hardware), etc. As it stands, it is difficult to judge how much effort/time/resources it requires to use MAPLE in practice, and what it took to produce the presented experimental results. It would also have been useful to explore the relation of the amount of MAPLE data augmentation vs performance: at what point does performance plateau, or could we have kept pushing the performance by just adding more of your augmentations?

* Related, for a data augmentation paper, I believe it is important to show that the strategy works on multiple datasets, not only on nuScenes, as then there is a risk that design choices have been optimized to that particular setting.

* Experimental evaluation shows overall mAP/mIoU over all classes, and detailed results for the augmented classes. Unsurprisingly, the performance on these selected classes improves slightly by data augmentation, but it is not clear how performance of the other more common classes (e.g. vehicles) is impacted by the augmentation, if at all.

* Experiments show that in terms of object detection, MAPLE performs on-par (sometimes better, sometimes worse) with prior works, Text3DAug and PGT-Aug. Line 417 explains how these mehods insert objects without scene context, but it isn't clear what other differences the methods have, and thus what can be concluded from the observed (lack of) performance difference.

Minor:
* Figures are referenced out of order, e.g. the first reference is Figure 3 on line 043.
* Line 269, "consistent sequence. nstead" -> typo
* Line 369, "FID" is used here for the first time without any explanation for those unfamiliar with what it means or measures.
* line 188: "(sx, sy, sz) follow priors from Tc.", unclear, how are LLM queries turned into prior distributions?

### Questions
* line 207: "Because inpainting attempts are not always successful, we generate multiple candidates and retain only those that satisfy both conditions." -> Unclear: for a single inpainting task, retain all that satisfy (so multiple variants of same scene?), or retain one (wich one)?
* For your experiments, how does the augmented dataset compare to the original one, and how much resources did it require? How does performance scale with the amount of augmented data?
* Beyond the placement distribution, how does MAPLE conceptually compare to Text3DAug and PGT-Aug? Why do the methods perform similarly for object detection, even though MAPLE peforms much better on segmentation?

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
4

### Summary
The paper proposes a training-free pipeline for multimodal instance augmentation to address long-tail 3D object detection. It generates synchronized RGB–LiDAR pairs by inserting objects via context-aware image inpainting and reconstructing pseudo-LiDAR through depth estimation.

### Strengths
-- MAPLE inserts objects via diffusion-based inpainting conditioned on VLM-generated descriptions, enabling diverse intra-class variation  not possible with copy-paste or mesh rendering.

-- Semantic verification uses VLM-based QA to filter mislabeled or incomplete inpaintings, reducing naive failure rates 

--MAPLE improves mAP for construction vehicles in multimodal detection and mIoU in LiDAR-only segmentation. FID scores confirm pseudo-LiDAR fidelity, reflecting better structural similarity to real data.

### Weaknesses
-- The “1,000 human-annotated images” for αsem estimation lack annotation protocol details (e.g., the distribution of annotators).

-- The paper claims “less than $50 for semantic verification” (p. 23) but does not validate this with actual API billing, which is important for scaling data.

-- The method does not compare with some multi-modal data augmentation methods, e.g., [1][2], make some comparisons to such VLM-free based methods is helpful.

-- The overall pipeline seems complicated, make it diffcult for real-world applications.

[1] Exploring data augmentation for multi-modality 3d object detection. 2020

[2] 3d data augmentation for driving scenes on camera. 2024

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
