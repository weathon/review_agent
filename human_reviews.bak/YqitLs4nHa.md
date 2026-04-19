# Unleashing the Power of Visual Prompting At the Pixel Level

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6

## Abstract
This paper presents a simple and effective visual prompting method for adapting pre-trained models to downstream recognition tasks. Our approach is underpinned by two key designs. First, rather than directly adding together the prompt and the image, we treat the prompt as an extra and independent learnable entity. We show that the strategy of reconciling the prompt and the image matters, and find that
warping the prompt around a properly shrinked image empirically works the best. Second, we re-introduce two “old tricks” commonly used in building transferable adversarial examples, i.e., input diversity and gradient normalization, into the realm of visual prompting. These techniques improve optimization and enable the prompt to generalize better. We provide extensive experimental results to demonstrate the effectiveness of our method. Using a CLIP model, our prompting method registers a new record of 82.5% average accuracy across 12 popular classification datasets, substantially surpassing the prior art by +5.2%. It is worth noting that such performance not only surpasses linear probing by +2.2%, but, in certain datasets, is on par with the results from fully fine-tuning. Additionally, our prompting method shows competitive performance across different data scales and against distribution shifts.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper mainly targets visual prompting, which is a finetuning technique that does not require changing original model architecture. The authors introduce several tricks including adjusting input resolution, data augmentation and gradient normalization.

### Strengths
1. The proposed method is simple yet effective.
2. The authors have provided extensive experiment results.

### Weaknesses
1. The novelty of this paper is limited. It is more like trials of existing training tricks of neural networks such as augmentation and normalization in VP-style methods. It is not surprising that such techniques can lead to improvement.
2. The authors mentioned VPT for several times in this paper. However, I cannot see the relation between EVP and VPT. Actually these are two kinds of methods. EVP and VP try to adapt the input space while VPT modifies the latent space and the original model architecture. Besides the design of EVP also has nothing to do with VPT.
3. I wonder why some of the performance of VP is Tab.1 and Tab.2 are inconsistent with that in the original paper. For example, VP-instagram has 22.9 accuracy on Flowers according to the original paper, while it is 4.8 in this paper. If the authors have re-implemented VP with different settings, then the difference should be highlighted in the paper.
4. It is worth noting that the proposed EVP is worse than VP on non-CLIP models without preprocessing classes. However, it is not clear how VP can be improved with such preprocessing, which is the same as the FLM proposed in [1].
5. It would be better to compare the proposed method with other VP variants, such as VP-RLM and BlackVIP [2].


[1] Chen A, Yao Y, Chen P Y, et al. Understanding and improving visual prompting: A label-mapping perspective[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 19133-19143.

[2] Oh C, Hwang H, Lee H, et al. BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 24224-24235.

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a set of useful techniques to improve the effectiveness of visual prompting in downstream tasks. Major techniques include: shrinking the input image size and then performing data augmentation; adding positional embedding to visual prompts at the beginning of the token sequence; gradient normalization using the whole image’s gradient.

### Strengths
The paper is easy to follow and the techniques proposed are effective in improving the performance. Authors put up many experimental results in the paper which might be helpful to the community.

### Weaknesses
The paper is not interesting since many techniques are engineering trials and seem straightforward. For example, resizing the image can only be taken as a data augmentation operation. The practice of applying enough augmentations is beneficial to performance is a widely proven trick. 

The underlying reason for gradient normalization is not elaborated. The authors borrowed ideas from the adversarial literature but that does not ensure the effectiveness of the gradient normalization strategy in visual prompting so a more precise formal reasoning is needed.

Results are not promising. Even though the proposed set of techniques are able to enhance the performance, it is hard to catch linear probing in certain datasets, not to mention fine-tuning. Since linear probing is such an easy method that any machine can apply, the advantage of visual prompting is not clearly demonstrated.

### Questions
No questions

### Soundness
1 poor

### Presentation
2 fair

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
Overall, this paper presents a new and simple visual prompt tuning method for leveraging a pretrained foundation model. The proposed method resizes raw image, apply data augmentation, and then add noises as visual prompts on the border of images. Besides, this work also leverage L2-based gradient normalization on the prompt. As a result, the proposed method can keep all the contents of raw input images and this paper shows data augmentation and L2-norm based gradient normalization on the prompts give the best performance improvements. Comparing the proposed method with previous works VP and VPT, the proposed method shows better performance on CLIP and non-CLIP pretrained model on 12 datasets. Ablation studies are also presented.

### Strengths
+ The proposed method is very simple to operate. Meanwhile, it also shows better performance than previous visual prompt tuning method.

+ The experiment is solid, in that CLIP and non-CLIP models are tested on 12 datasets. Besides, the proposed method is also analyzed on different perspectives, for example robustness and impact on data scales.

+ Overall writing is easy to follow, even though some details need to improve.

### Weaknesses
- The novelty sounds not strong. The proposed method is very similar to previous method VP, that noises are added on the border of images. 

- In Table 2, it needs to explain. EVP* more. What is the pre-processing used? Also, I cannot find enough details in sec 4.2.

- The motivation to introduce gradient normalization into visual prompt tuning is not very strong. The goal of adversarial attacks is to fool a classifier, which is different to visual prompt tuning.

### Questions
How to select the size of resized image patch in your proposed method? 164x164 performs the best.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
