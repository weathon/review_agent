# Bootstrap Your Own Noise: Denoising Adaptive Noise in Diffusion Models for SSL

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We introduce Bootstrap Your Own Noise (BYON), a self-supervised pretraining framework that unifies denoising diffusion with uncertainty-guided contrastive learning to enhance both local and global feature representations. BYON forms a self-reinforcing loop: contrastive learning improves reconstruction quality, and in turn, improved reconstructions refine feature alignment. A Semantic Uncertainty Estimation (SUE) module adaptively reweights contrastive updates based on reconstruction quality, while an Image-specific Adaptive Noise (IAN) adaptively modulates the noise intensity at the image level based on token saliency, perturbing more informative images more strongly.
BYON consistently boosts performance on image classification, semantic segmentation, object detection, instance segmentation, and fine-grained visual classification (FGVC) tasks. To ensure reproducibility, the code is available in the Supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper, "Bootstrap Your Own Noise" (BYON), introduces a new self-supervised learning (SSL) framework. The core concept is what the authors call a "self-reinforcing loop" that unifies three major paradigms: contrastive learning (like SimCLR), diffusion-based image reconstruction (denoising) and MIM-based image reconstruction (demasking).
To make this work, they introduce two novel modules: a "Semantic Uncertainty Estimation" (SUE) module that adaptively re-weights the contrastive loss (paying more attention to good reconstructions) and an "Image-specific Adaptive Noise" (IAN) module, which cleverly adds more noise to more salient or "informative" parts of the image. They show that the resulting pre-trained model produces very strong features that transfer well to a whole range of downstream tasks like classification, detection, and segmentation.

### Strengths
1.	The authors have managed to creatively combine three powerful ideas (denoising/demasking, contrastive, and uncertainty) into a single, cohesive framework. The IAN module, in particular, which adapts the noise schedule based on image content, strikes me as a very smart idea that moves beyond the typical "one-size-fits-all" noise of standard diffusion models.

2.	The evaluation looks impressive.

### Weaknesses
1.	This framework has a lot of moving parts: a diffusion model, a contrastive learning branch, an uncertainty module (SUE), and an adaptive noise module (IAN). This has to be an absolute monster to train and tune. The paper indicates that ‘BYON incurs higher cost than DiffMAE/MaskDiT due to the added contrastive/uncertainty pathways’, which I suspect is a major weakness compared to simpler, more scalable methods.

2.	Second, because there are so many new components, it's hard to tell what's really driving the performance. From Table 1, we can see that De-masking + SUE already achieves acc of 83.02%, the introduction of De-noising even make acc a little bit worse. Even with all components, the performance gain seems to be marginal, compared to simply using De-masking.

3.	All the major components including denoising, demasking, contrastive learning and uncertainty estimation are existing and well-explored techniques. The simple combination of these limits the originality of this work.

### Questions
See the weaknesses.

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
1

### Summary
The authors proposed self-supervised, uncertainty-guided contrastive learning method with diffusion model to improve feature representations. They suggested 3 main platforms SUE, IAN, and bootstrapping representations from the diffusion model. The suggested model developed a self-reinforcing loop: better contrastive alignment enhances reconstructions, which in turn refines global feature learning. This method outperforms existing masked image modeling and diffusion-based approaches across classification, segmentation, detection, and fine-grained recognition tasks(ImageNet top-1 accuracy by up to 0.6% and segmentation by 1.8%).

### Strengths
- The overall organization and style of the manuscript is high quality. It's easy to follow and try to explain the suggested concept clearly.

-The idea to use image-specific adaptive noise (IAN) with self supervised learning is promising in contrastive learning; IAN makes diffusion model be hard to discriminate source encoder's feature and reference feature.

### Weaknesses
- Application set is limited on vision benchmarks such as image classification, segmentation, and detection. can it be further utilized for multimodal learning or cross-modal  cases? This method relies on saliency uncertainty and it is specialized for vision tasks so it may limit  the broadness of applications.

- algorithmic advancement is limited; such contrastive learning may be biased to training set and believed to be unstable under limited configurations (i.e. limited datasets, small model.). The idea brings contrastive learning with bootstrapping generation from diffusion models and it looks working. However, the computational cost of retraining the representers (source and reference encoders) appears to outweigh the resulting reduction in error.

- lacks of details on self-reinforcing feedback loop. Self feedback loop is a main workhorse about this paper, but I didn't catch detailed mathematical description for that. For example, the type of used semantic guidance and category of uncertainty measure were not explained.

### Questions
-The section of self-reinforcing feedback loop limits the details. I would like to know the learning objective for self-reinforcing feedback loop and the network type to be used.  Does it use eq.(3) for their training objective?

-IAN adapts diffusion noise per image using token saliency scores. But the interaction between IAN’s noise schedule and the diffusion timesteps is not fully specified in terms of \lambda, \delta, or \eta. Is it chose in a manner of greedy search?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- The paper is a self-supervised learning method that aims to improve previous MIM + diffusion pretraining approaches.
- The authors’ motivation is that earlier MIM + diffusion methods understand local features well but lack global feature understanding.
- To address this, they introduce a contrastive loss along with additional components called the SUE module and IAN.
- Using the proposed SSL framework, which enhances both local and global feature understanding, they achieve improved top-1 accuracy on ImageNet classification and significant gains on downstream tasks such as semantic segmentation (ADE20K), object detection (COCO), and instance segmentation (COCO).

### Strengths
The proposed SSL method shows performance improvements in image classification, segmentation, and detection, which demonstrates its effectiveness. (There is a slight improvement in image classification, but a large improvement in detection and segmentation.)

### Weaknesses
- The authors could have provided more detail in both the architecture figures and the loss formulations.

1. Figure detail: For example, Figure 2 could have been illustrated more clearly, with notations and loss details included. As currently presented, it is difficult to interpret intuitively.
2. Loss detail: The definitions of L_demask, L_denoise are missing.

- It would be helpful to include qualitative comparisons, such as attention maps or other visual analyses, across MIM, diffusion, and MIM + diffusion. These visualizations could provide intuitive insights into whether different pretraining methods capture complementary aspects of the representation.

### Questions
- If both MIM and diffusion pretraining aim to learn local features, what is the advantage of combining them? Do they learn local features even better when used together compared to using only one pretraining method?
- In the proposed BYON framework, global feature learning comes from applying contrastive learning on the ViT CLS tokens. However, this same strategy can also be applied to pure MIM pretraining, since MIM methods also use transformer architectures with CLS tokens. The authors should provide quantitative results showing that applying contrastive learning on CLS tokens only with MIM pretraining is inferior to using MIM + diffusion to justify their claim.
- Looking at Table 1, which ablates the proposed components, de-masking (MIM) alone already achieves strong accuracy (82.89), better than de-noising (diffusion) at 80.14. Using both de-noising and de-masking yields 82.86, which suggests that the performance gain mainly comes from de-masking rather than de-noising. This raises the question of whether combining MIM and diffusion is truly necessary for an effective pretraining objective.
- Since the ablation in Table 1 seems to focus solely on image classification, it’s unclear whether the same trend holds for downstream tasks like detection or segmentation. Could you include quantitative downstream results for the ablated components to support the claim that combining MIM + diffusion yields broader benefits?

I would be willing to increase my rating if the authors provide further results or clarifications that address these points.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes **Bootstrap Your Own Noise (BYON)**, a self-supervised learning framework that combines diffusion-based denoising with contrastive representation learning.
To address the limitations of uniform noise and instability in previous diffusion SSL methods, the authors introduce two components:

1. **Semantic Uncertainty Estimation (SUE)**, which adaptively weights contrastive objectives based on reconstruction reliability.
2. **Image-specific Adaptive Noise (IAN)**, which adjusts the corruption strength according to token saliency, encouraging more informative learning from complex regions.

By coupling local denoising and global alignment objectives, BYON enables mutually reinforcing feature learning.
Extensive experiments on ImageNet, ADE20K, and COCO demonstrate consistent improvements over prior diffusion- and masking-based pretraining approaches.

### Strengths
* **Novel integration:** The paper successfully bridges diffusion-based denoising and contrastive learning within a unified self-supervised framework, which represents an important research direction in representation learning. While it remains an open question whether diffusion models can learn representations comparable to or distinct from those obtained through contrastive learning, this work demonstrates that integrating the two paradigms—reconstruction-based methods (e.g., MAE, diffusion) and contrastive learning—can yield complementary benefits and improved performance. This integration is a meaningful and timely contribution.
* **Balanced local–global learning:** The method couples local reconstruction (denoising) with global alignment (contrastive learning), addressing a key gap between generative and discriminative SSL approaches.
* **Comprehensive experiments:** BYON is thoroughly evaluated across classification, segmentation, and detection benchmarks, showing consistent and notable improvements over strong baselines such as MAE, DiffMAE, and MaskDiT.
* **Clarity and reproducibility:** The paper is well-organized, provides detailed ablation studies, and reimplements baselines under consistent training settings, supporting the credibility of the reported gains.

### Weaknesses
* **Analysis on self-supervision components:**
  If I understand correctly, the core contribution of this paper lies in integrating *diffusion- and reconstruction-based learning (MAE, denoising)* with *contrastive learning* to jointly learn global and local representations (as shown in Eq. 12).
  However, the ablation study mainly focuses on *de-noising* and *de-masking*, without isolating the effect of *contrastive learning*.
  It would be valuable to analyze how much the contrastive objective itself contributes to the final representation quality.

* **Positioning vs. prior diffusion SSL:**
  The paper could better clarify how BYON conceptually differs from prior diffusion-based SSL methods (e.g., DiffMAE, MaskDiT) beyond empirical performance gains, particularly regarding the training objectives and the resulting representation properties.

### Questions
* **Component-wise analysis:**
  Could the authors provide quantitative and/or qualitative comparisons between (1) contrastive-only, and (2) denoising / demasking-only settings?
  Such analysis would highlight how each supervision signal contributes to the learned representation, offering deeper insight into the proposed integration of reconstruction and contrastive paradigms.

* **Qualitative feature visualization:**
  While downstream metrics (e.g., classification, segmentation) demonstrate overall representation quality, it would also be interesting to visualize the feature space directly.
  The self-attention maps in Figure 5 are promising, showing meaningful inter-object similarity.
  Could the authors further visualize feature embeddings, for instance, via PCA or similarity maps as done in DINO[1,2], to illustrate the qualitative differences among features learned with (a) denoising/demasking only, (b) contrastive loss only, and (c) the combined BYON setup?
  Such analysis would strengthen the claim that BYON effectively unifies local and global representation learning.


[1] Emerging Properties in Self-Supervised Vision Transformers

[2] DINOv2: Learning Robust Visual Features without Supervision

### Soundness
3

### Presentation
3

### Contribution
3
