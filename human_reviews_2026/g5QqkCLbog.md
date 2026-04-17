# SketchingReality: From Freehand Scene Sketches to Photorealistic Images

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Recent years have witnessed remarkable progress in generative AI, with natural language emerging as the most common conditioning input. As underlying models grow more powerful, researchers are exploring increasingly diverse conditioning signals -- such as depth maps, edge maps, camera parameters, and reference images -- to give users finer control over generation. Among different modalities, sketches constitute a natural and long-standing form of human communication, enabling rapid expression of visual concepts. Yet algorithms that effectively handle true freehand sketches -- with their inherent abstraction and distortions -- remain largely unexplored.
In this work, we distinguish between edge maps, often regarded as “sketches” in the literature, and genuine freehand sketches. We pursue the challenging goal of balancing photorealism with sketch adherence when generating images from freehand input. A key obstacle is the absence of ground-truth, pixel-aligned images: by their nature, freehand sketches do not have a single correct alignment. To address this, we propose a modulation-based approach that prioritizes semantic interpretation of the sketch over strict adherence to individual edge positions. We further introduce a novel loss that enables training on freehand sketches without requiring ground-truth pixel-aligned images.  We show that our method outperforms existing approaches in both semantic alignment with freehand sketch inputs and in the realism and overall quality of the generated images.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
### Summary

This paper introduces a method for scene sketch-to-photo synthesis. The paper argues that the previous study mainly focuses on the conversion from edge maps to photos. Thus, the proposed method is developed to convert actual freehand sketches to photorealistic images, leveraging semantic sketch features. Compared to simple ``sketch-to-photo'' methods like ControlNet and T2I Adapter, the proposed method yields more plausible results.

### Strengths
The proposed method yields better results compred to *standard* image-to-image conversion methods like ControlNet and T2I Adapter.

### Weaknesses
### Major comments regarding the novelty

Although the paper claims the majority of prior methods just convert edge maps to photos (this may be true to some extent; for example, ControlNet is often trained on edge maps), the conversion from abstract freehand scene sketches to photos is actively studied, as in but not limited to [a-c].

Compared to those (bunch of) existing attempts, the novelty of the submitted paper is unclear. The novelty and merit (over methods inputting freehand sketches) are not adequately discussed in the related work sections, nor are they compared in the experiments.

- [a] Xiang, Xiaoyu, et al. "Adversarial open domain adaptation for sketch-to-photo synthesis." WACV 2022.
- [b] Wang, Jiayun, et al. "Unsupervised scene sketch to photo synthesis." ECCV 2022.
- [c] Koley, Subhadeep, et al. "Picture that sketch: Photorealistic image generation from abstract sketches." CVPR 2023.

### Questions
I would ask the authors to clarify the novelty and merit compared to existing methods for converting free-hand (or abstract) sketches to photos.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The present work seeks to improve sketch-guided image generation algorithms, e.g.,ControlNet and Adapter-based models. Specifically, the authors leverage an existing sketch-focused semantic embedding model to generate features that help guide the training of a new model that includes a novel modulation network trained on a loss that estimates likelihoods of different sketch pixels belonging to different semantic categories. Notably, the authors train on a mix of automatically generated sketches from images, and freehand sketches, along with language supervision leading to the ability to represent sketches and their relationship to images at multiple levels of abstraction.
I generally found the results quite compelling, although the improvement in metrics did not seem to always seem to qualitatively lead to images that looked much better. I found that while individual sections were well written, the organization of sections was somewhat disorganized requiring much back and forth between sections of the paper to understand the key methods and results. I elaborate more in the sections below.

### Strengths
* Both theoretically and empirically, I found the attention supervision loss to be quite elegant and effective. I find this to be among the paper’s stronger contributions and a key step towards translating freehand sketches to semantically coherent images.
* As with any good vision paper, I found the ablation experiments and comparisons to existing methods quite thorough and generally compelling.
* While I do have issues with the general clarity of the methods overall, I found some of the overview sections (e.g., preliminaries on LDM) well executed for the amount of space taken up by the section.

### Weaknesses
* The paper is at times hard to read because of its structure. For example, the modulation network is introduced before the reader knows what the sketch features that it uses are. I recommend laying out the necessary details of all the ‘ingredients’ of a module before diving into details about the module.
* I think there needs to be more details of the user study in the main text. I also find 23 participants to be quite a small pool.
* While I do find the interleaved discussion of the results in the Ablation section well written, the conclusion is quite limited both in terms of length and overall takeaways.
* While the modulation network and the attention supervision approaches are key contributions, the gains (here measured via improvements in FID, CLIP, and LPIPS) are fairly incremental with the image quality not appearing to be vastly superior. I think showing that users vastly prefer using this model relative to others would be compelling.

### Questions
* Perhaps because I’m not deeply familiar with Ham et al., 2023, but while I understand why low variance is penalized in the scale and shift maps (S, B), could the authors motivate the L1 regularization of these feature maps too? Is this just to smoothen the training process or is there a deeper theoretical significance?
* Could more be said about how the last 3 layers of the sketch encoder (from Bourouis et al.) was fine-tuned? 
* I think currently the ‘Attention Supervision’ part of the training pipeline is the least clear, particularly Eq 5. I would recommend a greater explanation of the Sun et al. formulation either in that section or in the Appendix.
* I would like to better understand why training on a mix of synthetic and freehand sketches led to more vivid color palettes as the authors note.

### Soundness
3

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
4

### Summary
The paper tries to tackle generating photorealistic images from freehand scene sketches and preserving structure and semantics. It augments a pretrained text-to-image model Stable Diffusion 2.1 with a lightweight modulation network that scales and shifts the predicted noise using sketch-conditioned features. A cross-attention supervision loss enforces text–spatial alignment. Training mixes human and synthetic sketches and emphasizes early high-noise timesteps to balance cost and global layout learning. Experiments show consistent gains over baselines on FID, CLIP similarity, and LPIPS.

### Strengths
The method handles abstract and deformable sketches for the scene-level sketch-to-photo generation. Attention supervision explicitly ties language tokens to spatial regions and the modulation head is lightweight and only active in early timesteps, which keeps computation modest. It plugs into a standard SD2.1 pipeline without re-architecting. This modularity makes the approach easy to reproduce.

### Weaknesses
The paper does not offer a mechanistic explanation or theoritical analysis for why the noise-modulation head works. Evidence is largely empirical like metric tables and ablations without probing the internal reasons. Moreover, it does not clearly position the method against mainstream fine-tuning techniques such as LoRA comparison and the core distinction from other fine-tuning techniques remains under-analyzed. 

The paper does not clearly articulate the mechanistic between the noise-modulation head and the target attention maps like why cross-attention alignment must be enforced via scale/shift on the predicted noise. The necessity of using target attention maps through this specific noise-space finetuning pathway remains not solid and reasonable.

The backbone choice is not state-of-the-art as the method is demonstrated on Stable Diffusion 2.1 only. There are no experiments on newer text-to-image backbones (like SDXL or FLUX). Without such results, generality are weakened and the figures presented seem relative lower quality in current phase, hurting the claim of the method effectiveness.

### Questions
Could the author explain more about why must cross-attention alignment be enforced via scale/shift on predicted noise finetuning?

### Soundness
2

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
4

### Summary
This work focuses on the task of scene-level sketch-based photorealistic image synthesis. It proposes a new modulation network and a loss function that emphasize the semantic structure of sketch inputs thereby improving generation quality. The authors have conducted experiments on the FS-COCO dataset and compared their method against existing conditional image generation methods, such as ControlNet, T2I Adapter, and FreeControl. The experimental results demonstrate that the proposed method outperforms these baseline methods across various metrics, including FID, CLIP and LPIPS.

### Strengths
1. The proposed method achieves impressive visual quality in its generated results, which is further supported by strong quantitative performance.
2. This work addresses a key challenge in sketch-based image generation, the inherent abstraction and ambiguity of sketches, which often leads to distortion in the results of existing methods. The proposed modulation network effectively addresses this issue by emphasizing the semantic structure of sketches. Consequently, the generated images achieve a good balance between visual quality and consistency with the input sketch
3. The paper is well-written and easy to follow.

### Weaknesses
1. The authors state that because a sketch can be abstract and ambiguous, their method focuses on extracting its semantic and structure information. In that case, it raise the question of whether an alternative approach, such as performing sketch captioning first and then feeding the resulting text into a standard T2I model (or baseline methods used in this paper), could be viable. A discussion or comparison against such a two-stage pipeline would be a valuable addition.
2. All experiments are conducted on the FS-COCO dataset, which contains relatively high-quality freehand sketches. To better assess the model’s robustness, it would be beneficial to evaluate its performance on datasets with more abstract or simplistic sketches, e.g., QuickDraw? This would provide greater insight into the model's generalization capabilities with respect to sketch abstractness.
3. The paper focuses on photorealistic image synthesis. It would be interesting to see if the proposed method can generalize to other artistic styles, such as cartoons. Including a few examples or a brief discussion on the model's adaptability to different styles would broaden the perceived applicability of the work.

### Questions
(1) The authors mentioned that the sketch encoder requires finetuning. Could the authors provide further details on this process? 

(2) What are the corresponding text prompts used to generate the examples shown in Fig. 5?

### Soundness
3

### Presentation
3

### Contribution
3
