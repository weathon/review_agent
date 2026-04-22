# LiveMoments: Reselected Key Photo Restoration in Live Photos via Reference-guided Diffusion

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Live Photo captures both a high-quality key photo and a short video clip to preserve the precious dynamics around the captured moment. 
While users may choose alternative frames as the key photo to capture better expressions or timing, these frames often exhibit noticeable quality degradation, as the photo capture ISP pipeline delivers significantly higher image quality than the video pipeline. This quality gap highlights the need for dedicated restoration techniques to enhance the reselected key photo. To this end, we propose LiveMoments, a reference-guided image restoration framework tailored for the reselected key photo in Live Photos. Our method employs a two-branch neural network: a reference branch that extracts structural and textural information from the original high-quality key photo, and a main branch that restores the reselected frame using the guidance provided by the reference branch. Furthermore, we introduce a unified Motion Alignment module that incorporates motion guidance for spatial alignment at both the latent and image levels. Experiments on real and synthetic Live Photos demonstrate that LiveMoments significantly improves perceptual quality and fidelity over existing solutions, especially in scenes with fast motion or complex structures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents LiveMoments, a method for selecting and restoring a new low-quality (LQ) key photo from a short clip surrounding some key high-quality (HQ) photo. To this end, the authors build a model based on latent flow models and learnable networks for the HQ key image, the LQ candidate, and the motion between the two frames modeled as optical flow. The authors also propose to perform image space motion alignment based on image patches. The authors train the model using open source high-quality data and introduce three benchmarks for evaluation, a synthetic one and two real-world Live Photo datasets.

### Strengths
* The paper introduces a novel task, Reselected Key Photo Restoration in Live Photos, and builds a model specifically designed for solving this task.
* The paper introduces novel benchmarks for this task.
* LiveMoments substantially outperforms competing methods from several related domains on the evaluated metrics.
* The paper is written clearly, although some parts are not clear, as will be detailed next.

### Weaknesses
* Although I appreciate the effort invested in this paper, I am not sure that, in terms of the scope and potential impact, it fits ICLR. The paper addresses a task that, in my opinion, is very nuanced, and it is challenging to see how to generalize the method to other tasks, setups, and types of image degradations. Could the authors please clarify the possible future directions for extending the proposed method and its other domains of applicability?
* To me, section 3.4.2 is not clear enough. For instance,
  - Is the space motion alignment done for each pixel? If so, then how is consistency preserved across pixels? 
  - Is it applied to the output of the model or to the input?
  - How is the patch size selected? Perhaps an ablation about that can clarify it.
* Code is missing from the submission to self-evaluate the method quality and better understand the implementation.
* Minor:
  - Some information is missing about the vivoLive144 and iPhoneLive90 datasets. For instance, what types of scenes were used to construct these datasets?
  - Missing explanation for what is $f(x)$ in line 194 and $Q$ in line 232.
  - Why wasn't FID evaluated on the synthetic dataset?

### Questions
.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces LiveMoments for reselected key photo restoration in Live Photos. It adopts a dual branch diffusion architecture with a ReferenceNet and a RestorationNet, and adds a unified Motion Alignment module that injects flow guided priors at latent and image levels. The authors build three benchmarks and propose relative no reference metrics tailored to the task. Experiments on synthetic and real Live Photo datasets demonstrate consistent gains over RefISR, RefVSR, and diffusion based SISR baselines.

### Strengths
- Clearly defined and practical new task specific to Live Photos with well motivated formulation  
- Elegant dual branch diffusion design with cross attention that effectively transfers fine details from the original key photo  
- Unified motion alignment at latent and image levels addresses large temporal offsets in real captures  
- Benchmarks and task specific metrics are thoughtfully constructed and results are strong and consistent

### Weaknesses
- Efficiency comparison is limited  runtime and memory are reported for the proposed method only  a fair cross method efficiency table would strengthen the claim

### Questions
1. How sensitive is the method to errors in optical flow  can the authors include a noise injection or alternative flow estimator study and show qualitative failure cases where misalignment occurs  
2. Can the authors provide a runtime and memory comparison to representative RefISR and diffusion restoration baselines under the same hardware and resolution

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
I think the paper introduces a practical task: restoring a reselected low-quality Live Photo frame using the original high-quality (HQ) key photo as a reference. The method, LiveMoments, uses a dual-branch diffusion transformer (ReferenceNet + RestorationNet) with cross-attention fusion and a unified motion-alignment module: (i) latent-level motion embeddings from RAFT flow injected as attention bias; (ii) image-level Patch Correspondence Retrieval (PCR) for tile-wise inference at 4K. Datasets include SynLive260 (synthetic) and real vivoLive144 / iPhoneLive90, plus a relative no-reference metric that normalizes to the HQ reference. Results show consistent perceptual gains on real data.

### Strengths
(1) I think the task setting is new and useful: both the degraded target and the reference come from the same Live Photo (temporal offset), ensuring content coherence while reflecting real ISP quality gaps.   

(2) Method-wise, a dual-branch SD3-based design with reference KV concatenation for cross-attention, plus flow-guided attention bias and PCR for high-res tiling, is a targeted solution to alignment at 4K.  

(3) The relative no-reference evaluation tied to an HQ reference matches the task goal better than generic NR metrics. (Details are described in the experiments.)

### Weaknesses
(1) Reference dependence and mismatch boundary. I think the approach relies on scene-level consistency between the HQ key photo and the reselected frame; large non-rigid motion or occlusions may cause the model to transfer incorrect textures. A failure-case analysis (large pose/occlusion) or a gating mechanism would help.    

(2) Flow robustness under strong degradations. The latent alignment hinges on RAFT performance on degraded inputs; the paper simulates matched degradation for flow, but I’d like sensitivity tests to flow errors/alternatives.  

(3) More metrics are needed in this paper. The relative NR metric is task-aligned but may be reference-pipeline specific; broader validation (more phone ISPs, user studies with correlation stats) would strengthen it.

### Questions
- How sensitive is performance to RAFT errors under heavy compression/blur? Any comparisons to alternative matching/flow or ablations with noise injected into flow?  

- Do the relative NR metrics correlate with human MOS across different devices/ISPs (Spearman/Pearson)? Any user study?  

- What are branch-wise params/VRAM? Could KV caching or half-precision in the reference branch speed up 4K inference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the task of Reselected Key Photo Restoration for Live Photos, 
where a user-selected frame from the short video is restored using the original high-quality key photo as reference. 
The paper formulates this as a reference-guided diffusion problem and proposes a dual-branch architecture 
combining a RestorationNet for the degraded frame and a ReferenceNet for the original photo, fused via cross-attention. 
A unified Motion Alignment module enables alignment both in the latent space through motion-guided attention 
and in the image space via a Patch Correspondence Retrieval (PCR) strategy.
Experiments demonstrate significant quantitative and visual gains over baselines.

### Strengths
1. Novel problem definition.
The paper clearly motivates the reselected key photo restoration task, which occupies a unique middle ground between RefSR and RefVSR. 
It targets a realistic and underexplored use case in mobile photography, differentiating itself from existing triple-camera or multi-view SR tasks.

2. Strong methodological design.
The proposed dual-branch diffusion architecture (Fig. 2) is technically sound. 
The ReferenceNet and RestorationNet fusion through cross-attention is a natural yet effective extension of SD3, 
while the Motion Alignment module appropriately handles temporal and spatial discrepancies 
through latent-space bias injection and patch-wise retrieval.

3. Comprehensive evaluation.
LiveMoments achieves superior quantitative results across all datasets and metrics (Tables 1-2) 
and includes clear ablation studies (Tables 3-4) isolating the effect of each module. 
The analysis of dual-branch fusion and motion-guided attention is particularly systematic.

### Weaknesses
1. Limited exploration of motion estimation reliability.
The approach depends on RAFT-based optical flow estimated from a synthetically degraded reference. 
While this improves alignment, the paper lacks quantitative analysis of flow robustness under large motion or occlusion. 
For instance, failure cases or confidence-weighted alternatives could clarify robustness limits.

2. Computational efficiency and scalability.
Appendix B reports 40 seconds for a 4K Live Photo inference, which, though acceptable for offline processing, may constrain deployment. 
Reporting model size and comparing computational cost against strong baselines (e.g., CoSeR, SUPIR) would make efficiency claims more transparent.

3. Ablation studies on training strategy.
While Table 3 examines architectural components, 
there is no analysis of the synthetic degradation's impact on real transfer. 
The reliance on Real-ESRGAN for training may bias results toward its degradation prior. 
An additional sensitivity or adaptation study could strengthen the generalization argument.

4. Positioning relative to RefVSR extensions.
Although the paper distinguishes LiveMoments from multi-camera RefVSR datasets, 
a more explicit discussion on whether the method generalizes to multi-frame or multi-reference inputs would help clarify scalability 
to broader camera setups.

### Questions
1. Flow failure handling.
In cases where RAFT yields low-confidence or divergent motion vectors (e.g., due to occlusion), 
does the model include any fallback or gating mechanism?

### Soundness
3

### Presentation
3

### Contribution
3
