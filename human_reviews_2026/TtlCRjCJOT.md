# Render-FM: Feedforward Model for Real-time Photorealistic Volumetric Rendering

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Current neural volumetric rendering methods like NeRF and 3D Gaussian Splatting (3DGS) achieve photorealistic quality but require prohibitive per-scan optimization (30+ minutes for 3DGS, 10+ hours for NeRF), limiting clinical applicability. We propose Render-FM, a feedforward model that directly regresses 6D Gaussian Splatting parameters from CT volumes in a single 2.8-second forward pass—a 500× speedup. Our key innovation, Anatomy-Guided Priming (AGP), leverages segmentation masks and transfer functions to provide anatomically-informed initialization. Trained on 991 diverse CT scans, Render-FM employs a 3D U-Net architecture to predict per-voxel 6DGS parameters, enabling immediate real-time rendering (328+ FPS). Experiments demonstrate that Render-FM achieves superior quality  compared to optimized baselines (27.30 vs 26.63 dB PSNR), with optional 89-second fine-tuning reaching 31.67 dB PSNR. Unlike per-scan methods, Render-FM generalizes to unseen anatomies, novel transfer functions, and compositional organ visualization without retraining. This advancement transforms clinical volumetric visualization, reducing preparation time from hours to seconds while maintaining or exceeding state-of-the-art quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors proposes Render-FM, a feed-forward model that maps a multi-channel CT volume directly to 6D Gaussian Splatting parameters in a single forward pass. Optimizations were majorly conducted related to acceleration. 3D nnUNet was deployed, and model was trained on 991 CT scans. Quantitative measurements are relatively sufficient. There are minor weaknesses can be seen below.

### Strengths
1. Easy read
2. Very clear details statements on the key points and the experimentations.

### Weaknesses
1. Since it is a clinical application, it would be great if some task specific metrics could be evaluated. Like but not limited to radiologists’ qualitative evaluation, edge-related measurements, etc.
2. The scope of the study experimentations might be limited. I think the study only focusing on medical imaging, and one specific medical imaging modality (CT). Like you mentioned in the conclusions: “We presented Render-FM, a feedforward model for real-time, high-fidelity volumetric rendering of CT scans using 6D Gaussian Splatting.” 
3. Only one baseline was compared, might be insufficient.

### Questions
1. “We introduce Render-FM, a foundation model for volumetric rendering through direct feedforward prediction of 6D Gaussian Splatting (6DGS) parameters from CT volumes. Unlike” I read through the paper, the authors seem like proposing UNet-like architecture to learn the mapping. Where can I see if it is related with foundation model? I might not be convinced by the authors if the argument would be that the “foundation” came from 991 CT images.
2. While AGP vs non-AGP has been explored for 6DGS, are there any reasons the authors didn’t do ablations regarding components inside Render-FM?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Render-FM, a foundational model that generates 3D renderings directly from CT volumetric scans. The approach incorporates an anatomy-guided initialization module that leverages Gaussian primitives to provide an effective structural prior for rendering. The paper also demonstrates a considerable acceleration in rendering speed through a feed-forward model design, highlighting the potential of Render-FM for real-time clinical applications.

### Strengths
- The paper addresses an important and clinically relevant problem by enabling volumetric rendering of CT scans. Such 3D representations could have significant practical value in diagnostic workflows and treatment planning.


- The paper is well-structured and clearly written, with the problem statement clearly defined, appropriate use of figures to illustrate the methodology and results, and rigorous experimental validation supporting the stated claims.

### Weaknesses
One general concern is the dependence on segmentation masks for detecting the anatomy and limited discussion of real-world deployment scenarios raise concerns about the model’s generalizability and practical integration into clinical workflows.

### Questions
- Although 6DGS with the AGP (anatomy guided priming) module was implemented, could the AGP module also improve the results of other 3DGS-based approaches? An ablation experiment can provide some insights about this question.

- The paper states that the reduction in optimization time enables real-time clinical applications. Could the authors elaborate on the specific types of clinical scenarios or use cases where such real-time rendering would be beneficial? Is there already studies showing that a lack of such visualization is causing workflow constraints ? Or the use of such rendering have helped in clinical decision making? These findings can further strengthen this work focusing more on a clinical aspect.

- The AGP model relies on segmentation masks, and TotalSegmentator[1] was used when these masks were unavailable. In a real-world clinical setting, where segmentation masks may not always be present, how would the model handle new, unsegmented samples?  What would happen if the segmentation from TotalSegmentator was not perfect? What drawbacks would such a scenario create? Can the confidence measures of segmentation be incorporated for the Gaussian initialization? The method performs well on the baseline dataset, but such an analysis would help in assessing the real-world clinical application.

- Table 1., shows Render-FM with FT on dataset that is unseen during training. I did not not understand what FT means in this scenario? What is the difference between OOD seen vs OOD unseen ? 

- Generalization beyond TotalSegmentator dataset. Since the claim is that Render-FM is a foundational model, the question arises about it's generalization. Could the model be tested against baselines in a zero-shot (or truely unseen fashion) on dataset/s that do not overlap with the TotalSegmentator datasets like AMOS[2] or CHAOS[3] ?

References:
1. Jakob et al., TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images

2. Ji et al., AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation

3. CHAOS challenge., https://chaos.grand-challenge.org/

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
3

### Summary
- Given a CT volume of a torso, this paper aims to render a mesh view of segmented organs with more realistic/physics-based appearances. 
- To do so, it takes the CT volume, the segmentations, and a voxel-wise transfer function and uses them to fit a Gaussian splatting-based model on images generated by a physics-based renderer. 
- However, doing so directly would be slow, as the physics-based renderer apparently requires 18 seconds to render a single view. 
- To that end, this paper proposes to use the same set of inputs and use a UNet to predict the parameters of the Gaussian splatting model directly.
- This paper trains the UNet above on a subset of the TotalSegmentator dataset and presents experiments comparing it to a previous Gaussian Splatting-based method.

### Strengths
- The application of Large Gaussian Models to large medical volume visualization is fun and interesting.
- The quality of illustrations is quite impressive.
- Once you figure out what the paper is actually trying to do, the methods section is well presented.

### Weaknesses
### Motivation, clarity, and scope of claims:

#### 1. Speed and realism:
IMO this paper's front matter is very confusingly presented as it assumes deep familiarity with direct volume rendering and makes claims that are not reflective of typical practice. Typically, widely used medical image viewers such as 3D Slicer or ITK-SNAP have volume renderers that fit a volume within a second or two and can be immediately interacted with. As examples, here are a few examples of volume renders achieved in seconds using Slicer: [1](https://discourse.slicer.org/t/screen-space-ambient-occlusion-for-volume-rendering/32323/27?u=lassoan), [2](https://www.youtube.com/watch?v=l8wlaCfYWG4), [3](https://www.youtube.com/watch?v=KadGfXmOs5Y)

On the other hand, this paper starts off by claiming that all previous methods require an hour+ to obtain a single mesh that can be interacted with. Furthermore, various key concepts such as transfer functions are never defined in the text and are left up to the reader who may be unfamiliar with this niche of volumetric visualization. 

After multiple reads and skims of the papers that are cited within, this paper's claims are true if a few assumptions are made: (1) one *has to* use a Gaussian Splatting based method; (2) one *must* use an expensive physically-based rendering algorithm to generate ground truth views for the GS algorithm to achieve slightly better realism to textures and second order effects. 

However, in practice, volumetric rendering algorithms that are already widely used achieve very fast rendering times and are reasonably realistic. This paper aims to rapidly get the last-mile of realism using Gaussian Splatting and a training set of physically-based renders, which is fine, but it is not at all clear from the presentation.

#### 2. Technical contributions:

Reductively speaking, the paper is a combination of Large Gaussian Models and the 6DGS volume rendering method (ICLR25). While combination papers are absolutely fine and appreciated, the paper does not detail what is particularly challenging about this application and what new insights readers can take away from the execution.

### Experiments

#### 1. Only a single baseline:
- As detailed above, medical volume visualization existed before this paper, 6DGS, GS, and NeRF, yet none of these approaches are benchmarked against in the paper. There is a sole baseline in the experiments (6DGS), which is more of an ablation, as 6DGS is part of the proposed method.
- The paper claims that physically-based rendering takes 18 seconds to render a single view. Is this on a CPU or a GPU? This seems extraordinarily high for a GPU implementation, and I see that there are GPU implementations available.

#### 2. Foundation model claims:
The model is trained on a subset of the TotalSegmentator CT dataset (991 volumes) and the experiments only include evaluations of render quality on a held-out subset of TS and a subset of the highly-related and very similar CT-ORG dataset.

This IMO is insufficient to make claims of being a foundation model for medical volume rendering. One baseline, one anatomical application, two ablations, and two datasets are not enough to make such a claim -- the proposed network should show evidence of generalization to new imaging contexts such as new modalities (e.g., on TotalSegmentator-MRI), new anatomical regions (e.g., vessels in the heart and brain), etc. I believe the proposed method requires retraining with a substantially larger and broader imaging dataset.

### Minor comments:
- The emphasis on being “nnUNet-inspired” is odd; the core contribution does not involve automatic configuration of training hyperparameters, which is central to nnUNet (which is otherwise just a plain UNet).
- The opening paragraph of the related work section should be in the Introduction, as it makes it clearer.
- Why is TotalSegmentator resampled to 1.5mm isotropic here? If realistic renders are required, the user would want renders at native/high resolution.

### Questions
This paper is not directly in my area of expertise so please correct me if I misunderstood something and I would be happy to revisit my rating. Some major points to discuss:
- Please elaborate on the choice/lack of baselines and contextualize this submission, given that current volume viewers all produce fast interactive renderings.
- Please clarify the technically challenging aspects of combining large Gaussian models with 6DGS.
- Please further justify the foundation model characterization given the scope of the presented experiments.

### Soundness
2

### Presentation
1

### Contribution
2
