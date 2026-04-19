# Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6

## Abstract
We propose to improve transformers of a specific modality with irrelevant data from other modalities, e.g., improve an ImageNet model with audio or point cloud datasets. We would like to highlight that the data samples of the target modality are irrelevant to the other modalities, which distinguishes our method from other works utilizing paired (e.g., CLIP) or interleaved data of different modalities. We propose a methodology named Multimodal Pathway: given a target modality and a transformer designed for it, we use an auxiliary transformer trained with data of another modality and construct pathways to connect components of the two models so that data of the target modality can be processed by both models. In this way, we utilize the universal sequence-to-sequence modeling abilities of transformers obtained from two modalities. As a concrete implementation, we use a modality-specific tokenizer and task-specific head as usual but utilize the transformer blocks of the auxiliary model via a proposed method named Cross-Modal Re-parameterization, which exploits the auxiliary weights without any inference costs. We observe significant and consistent performance improvements with irrelevant data of image, point cloud, video, and audio. For example, on ImageNet-1K, a point-cloud-trained auxiliary transformer can improve an MAE-pretrained ViT by 0.6\% and a ViT trained from scratch by 5.4\%. The code and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work tackles the problem of how to improve the transformer of one modality from the models of other modality with irrelevant data. It proposes a novel method named Multimodal Pathway equipped with cross-modal re-parameterization. It performs experiments with four modalities – images, videos, point clouds and audio with the datasets.

### Strengths
1. This work addresses an interesting and important problem – how to transfer knowledge from one modality to another modality.

2. The proposed approach is sound and reasonable.

### Weaknesses
1. The problem and motivation that this work focuses are not so novel. 
- Many recent models do not require paired multimodal data for pre-training and fine-tuning. 
- Knowledge transfer from one modality to another is actively studied in many directions. 
- Unfortunately, the related work of this paper is largely cursory. 
- Many previous works have shown that the four modalities that this work considers – images, videos, point clouds and audio – are related enough to learn from one modality and help for another modality with no pairing. 

2. The effectiveness of the proposed approach is not sufficiently demonstrated in experiments. 
- The reported performance improvements are somewhat marginal over MAE as shown in Table 1 (i.e., mostly less than 1.0% in accuracy and at best <2.0%). 
- Overall, the proposed cross-modal re-parameterization seems reasonable, more thorough experimental supports may be required. 
- More supports include more baselines, other novel modalities, more performance gaps, etc, which can make this submission much stronger.

### Questions
Please refer to the weakness.

### Soundness
3 good

### Presentation
3 good

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
This work proposes Multimodal Pathway Transformer (M2PT), a model to improve target modality from other modalities with non-paired data. M2PT consists of modality-specific tokenizers to transform raw inputs into features, and multiple linear layers inside each transformer block. Given the non-paired data of different modalities,  M2PT processes these data simultaneously and shows that the auxiliary data can improve the model's performance on target modalities.

### Strengths
1. This method is simple and the paper is easy to understand.
2. Lots of experiments are conducted to show that auxiliary modality can improve the model's performance on target modality.

### Weaknesses
1. The author claims that incorporating the auxiliary modality would improve the model's performance on the target modality, even if there is no any relevance between the data. However, the reasoning behind this enhancement in performance remains unexplained. Furthermore, it cannot be confirmed that non-paired data is entirely unrelated, thus raising doubts about the validity of all related statements.

2. In M2PT, numerous additional parameters were introduced, which could potentially account for the observed model improvement, rather than the inclusion of an auxiliary modality. The author didn't conduct any ablation experiments to investigate this.

### Questions
see weakness above

### Soundness
2 fair

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a cross-modal re-parameterization method to investigate the usage of irrelevant data to improve the overall performance of models.

### Strengths
This work present a wide study on multiple modalities of data, as well as tasks, which I think it's a contribution to the community. Moreover, the cross-modal reparameterization seems simple and straightforward yet effective, compared to largely pretrained MAE. Overall, the paper is well-written and easy to follow.

### Weaknesses
It would be great or to visualize the intermediate representation/weights with or without the re-parameterization method to see how it shifts. Also, does the re-parameterization also help the performance of the irrelevant dataset?

### Questions
1. Can the authors give a rationale or even guess why including irrelevant data works?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
