# GenFusion: Feed-forward Human Performance Capture via Progressive Canonical Space Updates

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
We present a feed-forward human performance capture method that renders novel views of a performer from a monocular RGB stream. A key challenge in this setting is the lack of sufficient observations, especially for unseen regions. Assuming the subject moves continuously over time, we take advantage of the fact that more body parts become observable by maintaining a canonical space that is progressively updated with each incoming frame. This canonical space accumulates appearance information over time and serves as a context bank when direct observations are missing in the current live frame. To effectively utilize this context while respecting the deformation of the live state, we formulate the rendering process as probabilistic regression. This resolves conflicts between past and current observations, producing sharper reconstructions than deterministic regression approaches. Furthermore, it enables plausible synthesis even in regions with no prior observations. Experiments on both in-domain (4D-Dress) and out-of-distribution (MVHumanNet) datasets demonstrate the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a feed-forward human performance capture method that reconstructs and synthesizes novel views of human subjects from a monocular RGB video stream. The key idea is to maintain a progressively updated canonical space, where visual features from consecutive frames are accumulated and fused using visibility-based weighting. This canonical space serves as a temporal memory that compensates for missing observations in the current live frame. To render novel views consistent with both past observations and current deformation, the authors employ a probabilistic regression formulation based on diffusion models. The probabilistic rendering alleviates blurriness caused by misalignment between frames and enables plausible synthesis even for previously unobserved regions.
The method is evaluated on several datasets and compared against deterministic and probabilistic baselines such as NHP, SHERF, Champ, and GauHuman, showing improved perceptual quality and temporal consistency.

### Strengths
The paper elegantly combines canonical-space-based temporal accumulation with a diffusion-based rendering framework. This hybrid approach effectively mitigates the limitations of deterministic regression. Unlike other optimization-based methods , this approach runs efficiently and generalizes to unseen subjects without per-frame or per-scene training.


The probabilistic rendering formulation allows plausible completion of unseen regions and reduces the dependency on perfectly aligned geometry

### Weaknesses
The method relies on accurate SMPL-X alignment to build temporal correspondences, but the paper does not quantify how fitting errors or template inaccuracies affect the reconstruction quality. Including such results will make this paper stronger.

Although the probabilistic rendering can hallucinate plausible details, the canonical space itself remains template-driven, which may limit fidelity for highly non-rigid clothing dynamics, which can be seen from the demo video.

### Questions
What is the inference time of the proposed method?

How long an sequence can it support?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a method to synthesize novel view of human images from monocular videos. At the core of the method is a “progressive feature fusion” module, which fuses features in a canonical space utilizing a parametric SMPL model . The authors conducted experiments on 4D-Dress and MVHumanNet to verify the proposed method.

### Strengths
The paper is easy to follow.

Synthesizing novel view images from monocular videos is an interesting task with practical applications.

The method is technically sound, leveraging a parametric model to fuse features in a canonical space.

### Weaknesses
**Insufficient experiments and generalization issues.** 

The paper claimed that the method can reconstruct humans from a monocular RGB stream, but the model was trained and evaluated on multiview video datasets. The experimental results on in-the-wild videos are required to illustrate the generalization capability. 

**Insufficient evaluations and comparisons.**  

The baseline methods (e.g., NHP, SHERF) are old. The comparisons and discussions with SOTA generalizable human generations are not included, such as AniGS, LHM, Human4DiT, and Vid2Avatar-Pro: Authentic Avatar from Videos in the Wild via Universal Prior [Guo et al. CVPR 25]. 

**Setup.** 

What are the advantages of this method which requires video as input over one shot method which only requires one image such as LHM, Human4DT?

### Questions
How to handle loose clothing such as dress as the method requires SMPL/SMPLX model and clothing warping is not accurate?

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
This paper presents a feed-forward method for human Performance Capture from a monocular input video stream. The core is a conditional diffusion model, serving as a renderer, conditioned by a canonical feature context fused from previously seen frames and image features at the current time step. The SMPL-X model is adopted to bridge the canonical context with live frames. The experiments show some good results.

### Strengths
1. This paper presents an effective combination of existing ideas. As is the usual practice, progressively fusing information from previous frames to a canonical SMPLX model, which can then be naturally warped to live frames and provide some missing features that may be occluded at the current live frame. 
Rather than directly render the feature context into an image with pixel-level loss, the authors use it to condition a diffusion model to 'generate' the image with a given camera view. Overall, it's a good combination. 

2. The proposed method is technically sound, with clear justifications for each component. I believe it can be reproduced with the given supp details. 

3. The experimental validations are reasonable and robust, and evaluate the method across various dimensions, especially Cross-dataset and In-the-wild generalization ability. 

4. The method consistently outperforms baselines, and visual results are compelling.

5. This paper, which represents the integration of generative AI with 3D reconstruction / Performance Capture, is an interesting and meaningful direction.

### Weaknesses
1. The paper does not explicitly discuss performance over very long sequences (e.g., minutes of video). Does the proposed method suffer from the long sequence forgetting issue? There should be an experiment. 

2. Such diffusion-based generative renderers fail to capture human performance accurately. The most evident issue is color bias, as demonstrated in the paper, which overall results in much lower PSNR, along with the synthesis of spurious or unrealistic details.

3. The dependence on the SMPL-X model, especially the overly simplistic fusion strategy based on a moving feature average of SMPL-X vertices, introduces sparsity that directly limits model performance. Furthermore, such an approach fails to account for dynamic garments and, in particular, topological variations.

4. The failure cases should be included in the discussion.

### Questions
Mainly listed in weakness. Besides, could you provide quantitative results for the inference latency of your full method? Is it possible to achieve real-time performance capture, e.g., with fewer diffusion steps?

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
This paper introduces a unified framework for aligning human motion sequences with multiple modalities (text, video, audio) within a shared embedding space, alongside a novel generative pipeline for motion synthesis conditioned on arbitrary inputs. The key contributions are:
a) MuTMoT: a multi-scale temporal motion Transformer that hierarchically encodes and decodes 3D motion sequences;
b) REALM: a retrieval-augmented latent diffusion model that utilizes learnable frame tokens and cross-modal conditioning to generate high-quality motion.
The model is evaluated across several tasks, including text-to-motion generation, motion retrieval, and zero-shot action recognition.

### Strengths
1.The paper proposes a modular and extensible architecture that combines multi-modal alignment, contrastive learning, and latent diffusion. The use of learnable frame-level tokens and time-aware modulation is a particularly notable design choice.
2.The model achieves strong quantitative performance on text-to-motion benchmarks (e.g., HumanML3D), outperforming existing baselines in standard metrics.
3.The supplementary ablation studies are reasonably thorough, covering most core components. 
4.The architecture appears broadly generalizable to other multi-modal backbones.

### Weaknesses
1.Despite claiming robust multi-modal alignment and generative capabilities, the method relies entirely on frozen, pretrained LanguageBind encoders for all non-motion modalities (text, audio, image, video). As a result, the framework lacks novel contributions toward modality-specific understanding. Moreover, only text-conditioned generation is quantitatively evaluated, while other modalities (audio, video, image) are not assessed in the main paper.
2.The supplementary generation videos exhibit noticeable artifacts, such as foot sliding and physically implausible transitions (e.g., in stands_up_from_a_laying.mp4, the subject appears to float unnaturally), which undermines the claimed motion quality.
3.Although the model achieves strong retrieval performance, a significant portion of the improvement appears to stem from GPT-4o-based text paraphrasing augmentation. As shown in Sec. B.3, Table 3, removing this augmentation causes R@1 to drop from 69.56 to 62.74. This raises concerns that the architectural contributions alone may not fully account for the observed gains. Clarifying the role of this augmentation and evaluating performance without it would strengthen the claims.
4.While training and inference are briefly described in the supplement, the paper lacks a clear, step-by-step explanation or diagram of the overall motion generation pipeline, which affects both clarity and reproducibility.

Limitations
1. The training process is resource-intensive, requiring 8× RTX A5000 GPUs and ~5 days for REALM to converge, which may limit reproducibility and accessibility.
2. During contrastive training, all non-corresponding modality-motion pairs appear to be treated as negative samples, without consideration for potential semantic similarity. This could penalize semantically related but unmatched pairs (false negatives), potentially degrading the embedding granularity and generalization ability.

### Questions
1.Can you provide quantitative or user study results for motion generation from non-text modalities (e.g., video-to-motion, audio-to-motion)?
2.Could you clarify the impact of frame-wise conditioning versus simpler global token conditioning through ablation?
3.How are positive and negative samples selected?
4.How sensitive is the model to the quality or relevance of the retrieved reference motions? And how are the candidate motion embeddings collected?

### Soundness
3

### Presentation
3

### Contribution
2
