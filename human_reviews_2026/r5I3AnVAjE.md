# From Diffusion to Rectified Flow: Rethinking Text-Based Segmentation

- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
Text-based image segmentation aims to delineate object boundaries within an image from text prompts, offering higher flexibility and broader application scope compared to traditional fixed-category segmentation tasks. 
Recent studies have shown that diffusion models (e.g., Stable Diffusion) can provide rich multimodal semantic features, leading to studies of using diffusion models as feature extractors for segmentation tasks. Such methods, however, inherit the generative natures of diffusion models that are harmful to discriminative segmentation tasks. In response, we propose RLFSeg, a novel framework that leverages Rectified Flow to learn direct mapping from the image to the segmentation mask within the latent space. The model is thus freed from the noise-denoise process and the need to optimize the time step of diffusion models, resulting in substantially better performance than previous diffusion-based methods, especially on zero-shot scenarios. By introducing label refinement and an Adaptive One-Step Sampling strategy, the model achieves higher accuracy even on a single inference step. The framework redirects a pretrained generative model to the discriminative segmentation task with zero modification to model structure, thus reveals promising application potential and significant research value.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposed RLFSeg, a text-based image segmentation framework that replaces the stochastic denoise process in diffusion with a deterministic rectified flow learned in the latent space of a pretrained Stable Diffusion model. The method reframes the standard noise-to-image rectified flow formulation to image-to-mask formulation by learning a straight-line displacement from the image latent to mask latent., conditioned on the CLIP text embedding. In addition to this reformulation, two components are added: A SAM-driven label refinement to mitigate coarse masks during training, Adaptive One-step Sampling at inference to rescale the predicted latent step when it unders-shoots the mask. Without architectural changed, the approach achieves strong results on benchmarks and shows zero-shot generalization to multiple benchmarks.

### Strengths
- The method keeps SD architecture intact, LoRA-only tuning makes the approach parameter-efficient and easy to reproduce.
- Single-step mapping is fast and avoids denoising schedules. Qualitative results show sharper boundaries than the compared diffusion baselines.
- Refinement with dynamic selection addresses polygon-style annotation artifacts and improves mIoU in ablations.
- On PhraseCut, higher mIoU than LD-ZNet; zero-shot to RefCOCO/+/G-Ref is consistently stronger than the listed diffusion/older baselines.

### Weaknesses
- At core, the method fine-tunes SD with a flow-matching loss to map image latents to mask latents; flow matching for repurposing is not new, and AOS is an inference-time heuristic. The contribution is mainly an application/engineering of known tools (RF, LoRA, SAM) rather than a conceptual advance.
- No comparisons to strong discriminative referring segmentation models (LAVT, ReferFormer, CRIS) or foundation pipelines (SEEM, SAM+grounding). Also omits recent diffusion-attention methods (e.g., ConceptAttention) that directly tackle text-to-mask without fine-tuning. This makes it impossible to assess overall SOTA relevance
- The SD VAE is trained on natural images, not binary masks. Possible loss of thin structures/edges is not examined.
- AP on PhraseCut is below LD-ZNet despite higher mIoU, hinting at calibration/thresholding issues although the method outperforms all other methods by a large margin on other benchmarks.

### Questions
- Can you add (or report published) numbers for LAVT/ReferFormer/CRIS (fine-tuned on PhraseCut) and SEEM/SAM+Grounding zero-shot pipelines on the same splits? This is essential to establish significance beyond diffusion-only comparisons.
- Please discuss/compare against ConceptAttention (and similar diffusion-attention approaches). Why is rectified-flow + LoRA preferable to training-free attention-based masking?
- Any evidence that the SD VAE harms binary mask fidelity (thin structures)? Did you try a mask-aware VAE or image-space RF?

### Soundness
3

### Presentation
3

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
This paper addresses the problem of using diffusion models to perform text-based segmentation.  It proposes several changes to existing approaches.  These include using rectified flow to learn a direct mapping from image to segmentation.  They also propose using Segment Anything to refine the ground truth segmentations during training, and a method to adjust the velocity and more effectively estimate the distance from image to solution.

### Strengths
All the proposed enhancements seem well motivated and intuitive.  Experimental results look very good.  In some cases the quantitative improvements are quite substantial.  The experiments seem thorough.  The approach is very clearly presented.

### Weaknesses
The paper builds pretty closely on other recent results, so the degree of innovation is somewhat moderate, but it still seems like a good step forward.

The paper is generally quite clear, but could benefit from better proof-reading.  For example, on line 43 “diversify” should be “diversity” and on line 226 “which trained” should be “which is trained”.

### Questions
The authors claim that the method improves over others on the PhraseCut dataset.  But AP is a bit worse than other methods, while mIoU is somewhat better.  Is there any reason to prioritize mIoU?  Do the authors have any idea why it improves on mIoU but not AP?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper attempts to employ rectified flow for semantic segmentation and propose a framework with three components, namely Rectified Latent Flow (RLF), Refinement and Dynamic Selection (RDS), and Adaptive One-Step Sampling (AOS). RLF reconciles the generative nature of diffusion with the deterministic demands of segmentation. RDS iteratively improves mask quality and mitigate annotation noise. AOS improves boundary accuracy and overall mask precision in a single step.

### Strengths
Adopting flow models for segmentation is interesting.

### Weaknesses
1. Overclaiming and novelty issues. This paper emphasizes that the diversity of noise conflicts with the deterministic nature of segmentation tasks, and claim a novel framework that leverages Rectified Flow to learn direct mapping from the image to the segmentation mask. However, similar idea was already introduced in SemFlow [1], which establishes mapping between images and masks via rectified flow.

2. The motivation of RDS is unclear. Refining annotations with additional model, SAM, modifies the training target of RLFSeg, which makes the comparisons with other baselines unfair. 

3. Missing SOTAs. It is encouraged to compare against relevant methods such as ADPP [2] and VPD [3].

4. The task definition is unclear. What is the definiton of `text-based segmentation`? It appears similar to referring segmentation,, but the author mentions `semantic segmentation` in L462, L475.


5. The writing is poor and hard to follow. In L70, `RLF directly predicts the transformation from the latent representation of the input image to that of the segmentation mask in a single step`. However, intuitively, single-step inference generally produces worse performance, and I do not see corresponding ablation studies. There are also typos, e.g., it should be `single step.` rather than `single step` in L90.

[1] SemFlow: Binding Semantic Segmentation and Image Synthesis via Rectified Flow, NeurIPS 2024.

[2] Aligning Generative Denoising with Discriminative Objectives Unleashes Diffusion for Visual Perception, ICLR 2025.

[3] Unleashing Text-to-Image Diffusion Models for Visual Perception, ICCV 2023.

### Questions
1. In L40, the authors claims that diffusion models as feature extractors are sub-optimal. I would like to see experimental evidence, including comparisons with VPD.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors propose a framework for text-based image segmentation that adapts Latent Diffusion Models (LDMs) using Rectified Flow (RF). They learn direct, deterministic mappings from image latents to mask latents without using stochastic noise to calculate intermediate time-step latents. They also introduce a Segment Anything (SAM) based mask refinement process, dynamic mask selection during training, and an adaptive one-step sampling for single-step inference. 
Experiments on segmentation datasets show improved performance. Ablations also verify their selected design choices.

### Strengths
1. Interesting annotation refinement strategy.
2. One-step sampling makes sense give the deterministic nature of segmentation.
3. Valid analysis on how noise addition conflicts the nature of segmentation tasks.

### Weaknesses
**1. Inadequate related work discussion for mapping image to mask**: 
This has been explored in numerous prior work (see paper below). These works are not discussed at all, please discuss these in related work section.
  - SemFlow (NIPS '24): https://arxiv.org/pdf/2405.20282 
  - Section 3.2 in Method seems identical to this earlier paper (i.e. even uses same pretrained weights). Please distinguish how the proposed method differs.  

**2. Inadequate related work discussion for adaptive one-step sampling**: Several prior works explore these ideas, please discuss them. 
  - https://arxiv.org/pdf/2509.00036v1
  - https://openreview.net/pdf?id=YJ1My9ttEN (ICLR 2025)

**3. SAM labels dependency:** The RDS component appears to depend heavily on ability of SAM to generate good masks. 
  - This becomes a weakness of the method, especially for domains where SAM may under perform. Exploration of such domains maybe interesting.
  -    The authors discuss negatives of human annotated masks (as polygon). However, are GT masks in the evaluation datasets collected in the same way (human annotated polygon masks)? 

**4. Overfit to dataset annotation style**
 - Is this arising due to the noise-free training strategy of the proposed framework? 
  - Would re-introducing noise fix this issue, instead of using the SAM based setup? 

**5. Compute Budget**
  - What is the additional compute costs for training proposed model? 
  - Is their a change in inference costs? How does this compare to prior work? 

**6. Baseline details missing**
  - Implementation details of the reported baselines are missing. 
  - Are the baselines pretrained on the same data as SD1.5? 
  - How does proposed method compare to similar prior work such as SemFlow?


* SemFlow (NIPS 24): https://arxiv.org/pdf/2405.20282

### Questions
See weaknesses. 

Compare against the SemFlow method that is very similar to author's approach?

### Soundness
2

### Presentation
3

### Contribution
2
