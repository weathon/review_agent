# CroCoDiLight: Repurposing Cross-View Completion Encoders for Relighting

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Cross-view completion (CroCo) has proven effective as pre-training for geometric downstream tasks such as stereo depth, optical flow, and point cloud prediction. In this paper we show that it also learns photometric understanding due to training pairs with differing illumination. We propose a method to disentangle CroCo latent representations into a single latent vector representing illumination and patch-wise latent vectors representing intrinsic properties of the scene. To do so, we use self-supervised cross-lighting and intrinsic consistency losses on a dataset two orders of magnitude smaller than that used to train CroCo. This comprises pixel-wise aligned, paired images under different illumination. We further show that the lighting latent can be used and manipulated for tasks such as interpolation between lighting conditions, shadow removal, and albedo estimation. This clearly demonstrates the feasibility of using cross-view completion as pre-training for photometric downstream tasks where training data is more limited.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CroCoDiLight, which repurposes the pre-trained CroCo encoder for photometric tasks. The authors hypothesize that CroCo implicitly learns lighting information through cross-view completion training on image pairs with varying illumination. The method is demonstrated on tasks including lighting stabilization in timelapse, temporal upsampling, shadow removal, and intrinsic decomposition, trained on datasets two orders of magnitude smaller than CroCo's original training data.

### Strengths
- **S1.** Novel insight that cross-view completion models implicitly learn photometric understanding. The hypothesis that CroCo must estimate and manipulate lighting to complete masked patches across views with varying illumination is interesting and well-motivated.

- **S2.** Efficient learning paradigm requiring datasets two orders of magnitude smaller than original CroCo training. This demonstrates that photometric knowledge is already embedded in the pre-trained encoder and only requires extraction rather than learning from scratch.

- **S3.** Demonstrates feasibility of repurposing cross-view completion foundation models for photometric tasks, opening a new direction for leveraging geometric pre-training for appearance-related downstream applications.

### Weaknesses
- **W1.** My main concern is the paper's positioning and the scope of investigation. The paper is framed as an application showcase (e.g., lighting stabilization in timelapse, temporal upsampling, shadow removal, intrinsic decomposition, etc.), but its core contribution is the insight into repurposing foundation models. It would be much stronger if repositioned as a systematic investigation (similar to Probe3D) into which and how pre-trained vision foundation models capture photometric properties, and why. The current study is confined to CroCo, missing a crucial comparative analysis against other foundation models. An investigation should include:
    - Other two-view encoders (e.g., DUST3R, MAST3R) and matchers (e.g., RoMa, GIM).
    - Single-view models known for strong correspondence (e.g., DINOv2, DINOv3).
    - Multi-view models where two-view is a special case (e.g., VGGT, Pi-3).
  - Such a comparison would provide more generalizable insights into how different pre-training objectives (cross-view, contrastive, etc.) contribute to learning photometric understanding.

- **W2.** Mixed results on quantitative evaluations and missing some evaluations.
    - We only have quantitative results for shadow removal (Table 1) and intrinsic decomposition (Table 2), while the other applications (lighting stabilization in timelapse, temporal upsampling) lack any quantitative benchmarks.
    - The intrinsic decomposition results are state-of-the-art (Table 2), while shadow removal is not (Table 1). This is acceptable if the paper is repositioned as an investigation (per W1), where the goal is demonstrating feasibility rather than beating every SOTA. However, under the current paper's narrative, it is difficult to justify the advantage of CroCoDiLight over other specialized methods.
    - That is, if we shift the paper's focus from application showcase to systematic investigation, we don't need to provide quantitative results for all applications, and we don't necessarily need to beat every SOTA.

Overall, the paper's insight (cross-view completion models implicitly learn photometric understanding) is valuable, but the paper's positioning and scope are not strong enough to provide a comprehensive investigation of this insight. I welcome the authors' response to address these concerns.

### Questions
- **Q1.** (Related to W1, authors may answer together) Have you experimented with other two-view foundation models like DUST3R or MAST3R (which build on CroCo)? What about single-view and multi-view models?

- **Q2.** (Related to W2, authors may answer together) Could you provide quantitative metrics for lighting stabilization and temporal upsampling tasks? For example, comparing temporal coherence metrics or perceptual quality against baseline interpolation methods? Or is that something beyond the scope of this paper?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce a method to disentangle CroCo latent representations into two components: a global latent vector capturing illumination, and patch-wise latent vectors representing the intrinsic properties of the scene. The model is trained in a self-supervised manner using pixel-wise aligned image pairs taken under different lighting conditions, guided by per patch cross-lighting and intrinsic consistency losses. They demonstrate that the disentangled latent space can be effectively leveraged for novel tasks such as interpolating between lighting conditions, shadow removal, and albedo estimation.

### Strengths
I found the approach of using an encoder-decoder architecture inspired by the Croco architecture to disentangle illumination from intrinsic scene representation both interesting and original. The design of the self-supervised training framework, particularly the proposed losses, appears well thought out and conceptually sound.

The paper demonstrates, through a range of tasks—including lighting interpolation, shadow removal, albedo estimation, and intrinsic image decomposition—that the proposed disentanglement approach is effective. The latent representations prove useful for handling these diverse downstream tasks. While the model does not outperform state-of-the-art methods specifically tailored for each task, the results are nonetheless promising, and the visual examples are rather convincing.

### Weaknesses
It is probable that leveraging the pretrained CrocoV2 model is beneficial, because the model was trained on a large set of image pairs captured under varying lighting conditions. Additionally, photometric augmentations applied during training likely enhanced the model to be robust to lighting changes. Still the experiments in the paper do not completely prove this.  How would the model work if instead of the Croco encoder the  MAE, DINOv2 or DINOv3 encoder is used and disentangled? Would the model perform less well on the downstream tasks? 

Also, the ablation in Table 6 raises a concern: the architecture of the two models compared are no longer identical so while supposedly it is true, it is not directly shown that the gain comes from the pre-trained model and not from the architecture choice.  It would have been insightful to evaluate also a model that retains the CroCo encoder architecture but is initialized from scratch. While indeed the training dataset is smaller, the learning task seems simpler than the hidden content reconstruction, making such an experiment worthwhile. Note also that the performance gain with the CroCo pretraining is significant only for the Intrinsic Image Decomposition task, much less for the Shadow Removal.

### Questions
I like the illustration and the narrative in Figure 1 as it effectively conveys how the CroCo model implicitly learns to extract content information from the second image and appearance information from the first, guided by the training data. This raises an interesting question: could the two models be integrated to jointly learn both the reconstruction and the disentangled latent representations assuming relevant training set (e.g. image triplets)? Such a unified approach—where mask content reconstruction and disentangled latent representations are jointly learned—could potentially enhance consistency and improve performance not only on the downstream tasks explored in this paper (e.g., shadow removal, albedo estimation, lighting interpolation), but also on geometric tasks such as 3D reconstruction.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents CroCoDiLight, which leverages supposedly inherent lighting disentanglement capability within CroCo to modify & repurpose CroCo for de-lighting / relighting tasks. To achieve this, the authors introduce two networks that explicitly separate CroCo's patch embeddings into a lighting vector and lighting-invariant latents, then recombine them, demonstrating that photometric understanding is already embedded in CroCo's representations and can be efficiently extracted for explicit control and various relighting tasks such as interpolation between lighting conditions, shadow removal, and albedo estimation.

### Strengths
- The paper is well-written and easy to follow. Good writing.
- The paper starts with a strong observation & hypothesis, recognizing the (possible) inherent capability within the original CroCo paper that its encoder implicitly encodes illumination information, which enables delighting & relighting during its novel view reconstruction task, and extending it to the hypothesis that this capability can be explicitly harnessed to achieve photometric tasks such as relighting/shadow removal/delighting. I believe the work is well-motivated and tackles an interesting question about the nature of CroCo and its representation.
- The authors offer a simple and intuitive solution to the problem (though this might point to the lack of novelty, as I would mention in the weakness section) by including a latent vector that disentangles lighting from geometry during the training phase. The method is simple and straightforward, effectively achieving its goal of lighting disentanglement as shown in the results.

### Weaknesses
- The original CroCo paper was a representation learning paper, focused on pre-training the model to be generally more suitable for various 3D / NVS downstream tasks from a simple two-view reconstruction loss. However, it seems that this work is more focused on training a model towards each specific downstream task (relighting / shadow removal / intrinsic image decomposition), which makes this work more closely aligned with existing relighting methods, of which there are already many. However, in this view, the core method of this paper (adding a separate latent vector for style and teaching model to change 'style (lighting)' of the image) very closely resembles previous GAN methods that achieve similar goals in generative scene and seems to lack novelty. Is there a more general implication for this method that may be relevant to representation learning / other downstream tasks, as was the original CroCo?
- The method requires datasets with identical geometry under different lighting, significantly limiting available training data. While synthetic datasets like HyperSim could be used, they introduce domain gap issues. How the authors address this fundamental limitation remains unclear.
- Lighting manipulation requires "walking" through latent space, making it difficult to achieve specific desired lighting conditions. The paper lacks a demonstration of how users can specify target lighting or achieve reproducible, controllable results without another scene that has desired lighting - can this point be further elaborated?
- What does this method have in advantage in comparison to Diffusion-based relighting methods, especially IC-Light (ICLR 2025), whose lighting can be controlled with text prompt as well as can be applied to various domains beyond scene imagery (i.e. including human faces, etc.)? Please elaborate.

### Questions
Please see Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores whether CroCo encoders, originally trained for cross-view completion with geometric objectives, also implicitly learn photometric representations due to illumination variations in training pairs. The authors propose CroCoDiLight to make this knowledge explicit through a delighting transformer that disentangles CroCo patch embeddings into a single lighting latent and intrinsic patch latents. Also a relighting transformer R that recombines them, and a single-view decoder D for high quality RGB reconstruction. Training uses only 57k pixel-aligned image pairs with different illumination, two orders of magnitude less than CroCo's training data as claimed in the paper. The method demonstrates applications in lighting interpolation, timelapse stabilization, shadow removal, and albedo estimation. Results show state-of-the-art on IIW among methods not trained on IIW, though shadow removal and construction metrics are good but not convincing.

### Strengths
The hypothesis that CroCo learns implicit photometric understanding is intuitive and the paper validates it convincingly. The delight-relight framing is elegant.
Needing only 57k pairs versus CroCo's 5.3 million is a strong practical advantage, especially given that aligned multi-illumination data is scarce. The paper shows strong albedo results- achieving 14.3% WHDR on IIW without training on IIW is impressive and suggests the intrinsic latents capture meaningful scene properties.
The paper covers multiple downstream tasks and provides extensive qualitative results. The failure case analysis in Appendix F is honest and valuable. The timelapse stabilization and lighting interpolation demos are compelling and showcase practical utility though there are some limitations such as shadow motion not being entirely smooth and the method struggling with sharp shadow boundaries that move rapidly across scenes.

### Weaknesses
Sharp shadow handling (shading effects, cast shadows, etc) is inadequately addressed: This is my biggest concern. The method uses a single global lighting latent, which fundamentally cannot capture the geometric information needed for sharp shadow boundaries. Sharp shadows arise from point lights and hard occluders, they encode precise light direction, occluder position, and surface geometry. A single image-space vector cannot represent this information, especially when shadow boundaries need to move correctly across multiple frames. The evidence is scattered throughout:

Fig. 17 shows direct shadows being replaced with ambient occlusion
Section 5.1 notes shadow motion during interpolation is "not entirely smooth" and Section F.1 admits tiles fully in shadow fail.
Most successful examples (Figs. 3-4, 8-10) show soft shadows, diffuse lighting, or outdoor scenes with gradual illumination changes
The timelapse examples work well for slow sun movement creating soft shadow transitions, but would likely fail for a person walking past a lamp creating sharp moving shadows

Tiling artifacts is unresolved: Section 3.5 and Fig. 16 show color inconsistencies from the sliding window approach. Paper mention potential fixes (Poisson blending, global reference tile) but don't implement them. Why present solutions but not evaluate them? The lighting latent being "optimally used" at 448×448 is a fundamental design limitation. This significantly undermines the high-resolution claims. Shadow removal metrics don't really match qualitative results.

Fig. 17 shows cases where your method is "working better" by removing additional shadows, but this also suggests the model isn't learning what the benchmark defines as shadow removal in my opinion.

Limited architectural justification- Why a single lighting latent and why D=1024? 

Paper didn't provide ablations on:

Multiple lighting latents per image/tile (which would help with local lighting)
Lighting latent dimensionality (is 1024 dimensions necessary? wasteful?)
Spatial lighting maps vs. single vector

The Table 3 ablation uses a much simpler baseline (just linear embedding + DPT head), making it unclear whether gains come from CroCo features or better architecture. A fairer comparison would use the same I/R architecture without CroCo pretraining.
Image-space lighting is a fundamental limitation: Section 5 and Appendix C mention the lighting latent works in "image space" not "world space" but don't explore the implications. This means:

The method can't handle viewpoint changes
It can't reason about 3D light positions or directions
It's brittle to even small camera motion
Shadows will appear in wrong positions if the camera moves slightly

### Questions
Can you quantify performance degradation as shadow sharpness increases? Even a simple analysis binning test images by edge gradient magnitude or manually annotating hard vs. soft shadows would help establish the method's scope.
Why not implement the color correction solutions you mention (Poisson blending, global reference) and show results? This seems critical for addressing both the metrics gap and the tiling artifacts. 
A dimension ablation (dimensionality of S0) would help understand what information is being compressed.
For the world-space vs. image-space issue- did you try encoding light direction or position explicitly? Even rough geometric cues might help.
How does the method handle colored lighting vs colored surfaces? The disentanglement seems like it would be ambiguous
Lack of extensive ablation studies-
Multiple lighting latents per image/tile (which would help with local lighting)
Lighting latent dimensionality (is 1024 dimensions necessary? wasteful?)
Spatial lighting maps vs. single vector

### Soundness
2

### Presentation
3

### Contribution
2
