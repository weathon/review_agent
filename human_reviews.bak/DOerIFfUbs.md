# Enhancing Vision-Language Model with Unmasked Token Alignment at Scale

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 3

## Abstract
Contrastive pre-training on image-text pairs, exemplified by CLIP, becomes a standard technique for learning multi-modal visual-language representations. Although CLIP has demonstrated remarkable performance, training it from scratch on noisy web-scale datasets is computationally demanding. On the other hand, mask-then-predict pre-training approaches, like Masked Image Modeling (MIM), offer efficient self-supervised learning for single-modal representations. This paper introduces Unmasked Token Alignment (UTA), a method that leverages existing CLIP models to further enhance its vision-language representations. UTA trains a Vision Transformer (ViT) by aligning unmasked visual tokens to the corresponding image tokens from a frozen CLIP vision encoder, which automatically aligns the ViT model with the CLIP text encoder. The pre-trained ViT can be directly applied for zero-shot evaluation even without training on image-text pairs. Compared to MIM approaches, UTA does not suffer from training-finetuning inconsistency and is much more training-efficient by avoiding using the extra $\mathrm{[MASK]}$ tokens. Extensive experimental results demonstrate that UTA can enhance CLIP models and outperform existing MIM methods on various uni- and multi-modal benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper claims that crontrastive pretraining from scratch is computationally demanding and masked image modeling will introduce training-finetuning inconsistency problem. Thus, they use the CLIP model as the teacher, to train a vision transformer. When performing distillation, they do not align the masked token as before, but the unmasked token instead. In zero-shot and finetuning experiments, the proposed method gains a large improvement.

### Strengths
- The method is described clearly.

- The workload of the experiment is heavy.

### Weaknesses
- The biggest weakness is the motivation. Prior works used the masked tokens as the prediction tokens to learn their representation. However, in this paper, they predict the unmasked token instead. Thus, I wonder what information can masked token provide if the model aims to predict the unmasked token. In the former method, they provide the context information with the unmasked tokens. The learning of these former methods makes the context increasingly effective. But in this paper, since unmasked tokens are represented with no difference from the normal tokens, what novel information can the model learn by recovering such unmasked tokens based on similar normal tokens? Or to say the least, the method proposed by the author is similar to the simple distillation process which drops tokens randomly.
- In section 2.2, the author claims that ‘It causes training-finetuning inconsistency and makes the trained ViT unable to perform zero-shot classification without fine-tuning’. In my opinion, this description is not accurate because the trained ViT could be able to transfer to unseen domains no matter whether it is with or without [MASK] token training. So I hope the author can present citations or experimental results to prove this statement.
- In terms of technical novelty, this paper lacks some careful design. Besides, there seems to be a weak connection between the part of the ablation study and motivation, such as **the positional embedding** analysis. Since ICLR is a top-tier conference, this paper also lacks a solid theoretical foundation or forward-looking research direction and should be revised.

### Questions
1. In section main result, I only see some statistics, instead of any analysis about increase or decrease. Could you not only present the number which I can see from the figure or table but also give some details about the advantages or disadvantages of the method you proposed?
2. In Tab. 3, could you explain why you useLLaVA-Bench which is often used to test the multimodal instruction ability? And why the UTA with G/14 model did not beat that of L/14 model but exceeded in conversation and reasoning by a large margin?
3. From the results of Tab 4 and Tab 5, it can be seen that the proposed UTA just achieves a very minor improvement compared to EVA-02. How to further justify the superiority of your method in such settings?
4. In ‘UTA for pre-training the textencoder’, there is an interesting result, the same method does help to visionencoder, but not the text encoder. Could you give more discussion? I wonder whether your motivation is convincing and why it does not work in text modality?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to distill the knowledge of a pretrained CLIP model and further enhance its vision-language representations. The core idea is to train a VIT with the masked image modeling objective that aligns the unmaked token embeddings with the CLIP embeddings. After performing the masked image modeling, the VIT and CLIP text encoders are further finetuned with an image-text contrastive loss. The experiments are conducted on a wide range of benchmark datasets and show very promising performance.

### Strengths
-It is a good idea to perform unmask token alignment with the pretrained CLIP model due to its efficiency. 

-The experiments are quite extensive including many downstream tasks. The method also achieves the SOTA on most of the datasets.

### Weaknesses
-Motivation of reducing pretraining costs is not convincing. In particular, the abstract claims that "training CLIP from scratch on noisy web-scale datasets is computationally demanding". Although this is true, this paper does not solve this issue at all because it still relies on a pretrained CLIP model at the first place.

-The performance gains seem to come from the contrastive finetuning rather than the proposed unmasked alignment pretraining. In Table 1, comparing with the CLIP teacher i.e., EVA-CLIP, there is always a performance drop for UTA without finetuning. This is concerning because this paper claims the unmasked alignment pretraining as one of the main contributions. 

-One important baseline is missing. Since finetuning is very effective for zero-shot image classification, this paper should also compare with the CLIP teacher i.e., EVA-CLIP,  that is also finetuned on the same dataset. 

-Improving the VIT efficiency by dropping masked tokens has been done in [A].  This paper fails to cite this important reference paper and claims it to be something new. 

[A] Li et al., Scaling Language-Image Pre-training via Masking. CVPR 2023

-The improvement in Table 4 & 5 (detection and segmentation) is marginal (< 1%). This is also concerning. The method seems to be limited to image-level prediction tasks.

### Questions
The authors are highly encouraged to address my questions mentioned in the weakness. In addition, I have the following questions. 

-Is the CLIP teacher-model always the giant EVA-CLIP? 

-It would be good to provide the CLIP teach results in Table 3, 4, & 5.

-This paper says that it is following previous works to perform the second-stage contrastive finetuning without proving the reference. Please provide the reference. 

-Is finetuning helpful in Table 4 & 5?

-Is masking strategy applied in the finetuning stage?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper propose a Unmasked Token Alignment (UTA) strategy to improve the performance of pre-trained ViT. It achieves higher performance than CLIP.

### Strengths
-The proposed method is a universal strategy to improve the learned representation. The learned features can be well-used on various downstream tasks.

-Compared with MAE, it shows significantly higher performance on ImageNet. It is interesting to discuss which pre-trained strategy is better.

### Weaknesses
-Will the new module bring extra training cost?

-It may be unfair to directly compare UTA and MAE, as UTA uses an extra tearcher but MAE is only trained with itself.

### Questions
See the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an approach to train a ViT with the target of "aligning" with existing CLIP visual tokens. Specifically, during training, part of the image tokens are masked out and the learning target is to "align" the resulting embedding with the unmasked portion of the CLIP visual tokens. After the pre-training stage, the model can be further fine-tuned on image-text pair data to further enhance its cross-modal capability. The authors conduct extensive experiments on benchmark datasets that show good performance on a number of zero-shot image classification, vision-only and vision-language tasks.

### Strengths
- The approach is very intuitive and technically sound, with good reproducibility.
- The paper is well written, with clear motivation, approach and detailed experimental results.

### Weaknesses
- Contribution is small. UTA is essentially a variant of the popular Feature Distillation (FD) approach. Ablation study in table 6 indeed show that the performance of UTA is only slightly better than FD.
- Zero-shot performance on ImageNet zero-shot is a bit unfair when comparing UTA against open-CLIP / EVA-CLIP, as the former use ImageNet-21k for training (although without labels). A fairer experiments in my opinion would be pre-training UTA on a random set of unlabeled web images.

### Questions
see above

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent
