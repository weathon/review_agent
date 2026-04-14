# Locality Alignment Improves Vision-Language Models

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 5

## Abstract
Vision language models (VLMs) have seen growing adoption in recent years, but many still struggle with basic spatial reasoning errors. We hypothesize that this is due to VLMs adopting pre-trained vision backbones, specifically vision transformers (ViTs) trained with image-level supervision and minimal inductive biases. Such models may fail to encode the class contents at each position in the image, and our goal is to resolve this with a vision backbone that effectively captures both local and global image semantics. Our main insight is that we do not require new supervision to learn this capability – pre-trained models contain significant knowledge of local semantics that we can extract and use for scalable self-supervision. We propose a new efficient post-training stage for ViTs called locality alignment and a novel fine-tuning procedure called MaskEmbed that uses a masked reconstruction loss to learn semantic contributions for each image patch. We first evaluate locality alignment with a vision-only benchmark, finding that it improves a model’s performance at patch-level semantic segmentation, especially for strong backbones trained with image-caption pairs (e.g., CLIP and SigLIP). We then train a series of VLMs with and without locality alignment, and show that locality-aligned backbones improve performance across a range of benchmarks, particularly ones that involve spatial understanding (e.g., RefCOCO, OCID-Ref, TallyQA, VSR, AI2D). Overall, we demonstrate that we can efficiently learn local semantic extraction via a locality alignment stage, and that this procedure benefits VLM training recipes that use off-the-shelf vision backbones.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel, computationally efficient mask-based self-supervised loss function designed to enhance the local feature representations of a pretrained Vision Transformer (ViT) initially trained on a global, image-level task. This approach aims to improve the ViT's utility for Vision-Language Model (VLM) training. By applying their method across various backbones, the authors demonstrate its generalizability and report gains in patch-level semantic segmentation and across multiple vision-language tasks.

### Strengths
Simplicity and novelty of the approach, with a clear explanation of the intuition behind the formulation of patch embeddings (g_i) and the task.

Demonstration of general applicability by using a variety of backbones trained on different tasks.

Comprehensive evaluation across diverse VLM benchmarks.

### Weaknesses
Ablation studies could be more comprehensive. Some, like the effects of data augmentation and training data scale, feel unnecessary or self-evident. Exploring a broader range of datasets, such as CC3M (diverse) versus IN1k or SAM images (multi-object), could have offered more insightful findings on generalizability.

Limited ablation of the loss function. For example, testing reconstruction of only unmasked tokens rather than the entire embedding sequence could provide valuable insights into the role of different token types in the loss.

Comparison with alternative masking strategies is missing. While the rationale for masking in the current way is sound, comparing with an approach like dBOT—where the student rather than the teacher is masked—could have strengthened their case, as dBOT follows a nearly identical pipeline and has shown strong spatial feature learning.

dBOT: Exploring Target Representations for Masked Autoencoders

### Questions
In the current setup, the entire embedding sequence is reconstructed during training. Have you considered ablations that reconstruct only unmasked tokens?

While I understand your reasoning for masking the teacher, have you explored or considered a comparison with approaches where the student is masked, as in the dBOT framework? 

(This one is due to my lack of knowledge about single stage VLMs), Can you give more context on the significance of the VLM benchmark improvements? Some of the relative improvements on Figure 5 are so small that they seem like noise. (but again, it just might be duo to my lack of knowledge)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a  MaskEmbed fine-tuning procedure that uses a masked reconstruction loss to improve the path-level semantics of the visual encoder. Experiment results show that the proposed method can improve performance across a range of VLM benchmarks.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed MaskEmbed training diagram is effective in learning local semantics.

### Weaknesses
1. The authors need to include comparisons and discussions with more methods, such as dBOT[1] and UMG-CLIP[2].

a) dBOT[1] employs a distillation strategy similar to the method presented in the paper, so it is necessary to discuss the differences with this method and provide performance comparisons.

b) UMG-CLIP[2] directly incorporates fine-grained annotations to enhance CLIP's Locality Alignment. I am curious whether the proposed method has advantages over this method in some fundamental visual perception benchmarks.  Besides, is it possible to use a similar visualization approach as in UMG-CLIP to further illustrate the locality of the features in MaskEmbed?

2. The experimental comparisons are not sufficient.

a) As previously mentioned, it is necessary to supplement more results on visual perception benchmarks.

b) The paper employs a one-stage strategy for training VLMs. As far as I know, the majority of current VLM methods employ a multi-stage training approach, and the mentioned Llava-1.5 does as well, which gives it better performance than the one-stage baseline in this paper. I hope the authors can train the VLM model according to the Llava-1.5 setup and conduct performance comparisons on more benchmarks used in Llava-1.5 (r.f. Benchmarks in Table 3 and Table 4 of Llava-1.5).

c) To my knowledge, the latest method in the Llava series, Llava-OneVision[3], fine-tunes the vision encoder part simultaneously during VLM training. Many other recent methods[4][5] also adopt this setting, making the claim in lines 52-54 somewhat Inadequate. I wonder whether simultaneously fine-tuning the vision encoder could lead to VLMs gaining local semantic understanding, potentially diminishing the advantages of the proposed method.

[1] Exploring target representations for masked autoencoders.

[2] UMG-CLIP: A Unified Multi-Granularity Vision Generalist for Open-World Understanding.

[3] LLaVA-OneVision: Easy Visual Task Transfer.

[4] Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution.

[5] MiniCPM-V: A GPT-4V Level MLLM on Your Phone.

### Questions
Please refer to the 'Weaknesses' part.

### Soundness
3

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
This paper proposes improving the performance of Vision-Language Models (VLMs) on region-level visual tasks by enhancing their **locality alignment**. The authors argue that the poor performance of many current VLMs on spatial reasoning tasks can be attributed to the weak locality alignment of vision models, which is caused by image-level supervision and minimal inductive biases. To address this, the paper introduces a post-training approach called **MaskEmbed**, which aims to improve the locality of features. Specifically, MaskEmbed applies the same mask to both the input image of the teacher model and the output features of the encoder model. The masked features are then aligned with the teacher model’s output features through a decoder, thereby enhancing the encoder’s locality alignment. Both visualizations and experimental results demonstrate that MaskEmbed effectively improves locality alignment and boosts performance on downstream tasks.

### Strengths
The proposed method is relatively simple and provides a notable performance improvement.

### Weaknesses
1. **Issues with the Main Claim**:  
   The paper’s primary claim is confusing. It begins by hypothesizing that VLMs perform poorly on region-level tasks due to image-level supervision and minimal inductive biases. However, pre-training methods like DINO, despite using image-level supervision, exhibit strong locality, to the point where their features can even be directly used for semantic segmentation maps. This suggests that the initial assumption may be flawed. Additionally, regarding the claim about minimal inductive biases, is the paper referring to the lack of inductive biases in ViT architectures? If so, would using a convolution-based structure like ConvNext or a hierarchical structure like Swin Transformer resolve this issue? The paper fails to sufficiently justify this claim, making the motivation behind **MaskEmbed** unclear. Consequently, **MaskEmbed** may not be effective in the aforementioned scenarios.

2. **Methodological Concerns**:  
   There are also some issues with the methodology. The goal of **MaskEmbed** is to make the tokens output by the encoder—when the entire image is input—align with the tokens output by the teacher model when only partially visible patches are input. This approach could weaken the encoder's ability to model global interactions, potentially limiting its capacity to capture long-range dependencies. Such a strategy could harm performance on tasks requiring global semantic understanding. Furthermore, this method may also negatively impact region-level tasks if the masking approach crosses a certain threshold, as it constrains the model’s capacity for representation learning. These factors suggest that **MaskEmbed** may be highly sensitive to hyperparameters, which significantly limits its overall contribution.

**Response to the rebuttal**

Thank you to the authors for their comprehensive and detailed responses—most of my concerns have been resolved.

However, regarding the use of models with stronger locality, such as DINO and MAE, as vision encoders, the authors avoided addressing this issue by pointing out that most VLMs use CLIP/SigCLIP as their vision encoder. Nevertheless, many recent MLLM studies often incorporate mixed vision encoders, as demonstrated in [1,2]. Therefore, it would still be valuable to verify the effectiveness of the proposed method when applied to models with stronger locality.

In addition, although this work focuses on the VLM domain, its techniques primarily target vision encoders. To better highlight its advantages, I suggest integrating the proposed method into some vision pre-trained models, such as MAE or DINO, and evaluating whether the optimized models exhibit advantages in dense prediction tasks. This would better showcase the contributions of this work.

Despite some minor concerns, I find this to be an interesting study overall, and I have raised my score accordingly.

References:

[1] Jiang D, Liu Y, Liu S, et al. From clip to dino: Visual encoders shout in multi-modal large language models. arXiv preprint arXiv:2310.08825, 2023.

[2] Shi M, Liu F, Wang S, et al. Eagle: Exploring the design space for multimodal LLMs with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024.

### Questions
1. How were the scales in **Figure 5** determined? The performance gains appear to be quite small, yet they are magnified in the figure, making it difficult to assess the actual level of improvement provided by this method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a method called Locality Alignment, which aims to improve the spatial reasoning capabilities of Vision Language Models (VLMs) by enhancing the local semantic understanding of pre-trained Vision Transformers (ViTs). Specifically,  the authors propose MaskEmbed, which uses a masked reconstruction loss to learn the semantic contributions of each image patch.

### Strengths
1. The paper is overall well-written.
2.  Locality alignment is efficient, requiring minimal additional computation compared to pre-training, making it a cost-effective solution.
3. The authors provide theoretical analysis and practical experiments to support their claims.

### Weaknesses
1. I suggest the authors to conduct a thorough analysis of MaskEmbed's sensitivity to hyperparameters. This includes varying mask sizes, patch sampling strategies, and the influence of different reconstruction targets. By understanding these sensitivities, the paper can provide guidelines for applying MaskEmbed effectively across various scenarios. Besides,  including additional evaluations that specifically test the impact of MaskEmbed on global semantic understanding tasks may help to validate whether the method indeed compromises the model's ability to capture long-range dependencies.
2. The methodology of MaskEmbed involves fine-tuning the encoder to align its output tokens with those of a teacher model when only partial patches of an image are visible. This approach, while innovative, raises concerns about its impact on the encoder's ability to model global interactions and capture long-range dependencies, which are crucial for tasks requiring global semantic understanding.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
