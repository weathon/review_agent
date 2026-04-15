# Vision-Language Subspace Prompting

- Decision: Reject
- Scores: 5, 3, 5, 6

## Abstract
Prompting vision-language models like CLIP to adapt to downstream tasks is currently topical. A seminal technique to this end is context optimization, which replaces a subset of textual tokens with trainable parameters (a.k.a soft prompts). However, current pipelines use a single vector embedding induced by soft prompts as the classifier weight for visual recognition. This can lead to problems where the learned soft prompts overfit to base classes’ training data, resulting in poor performance when applied to novel classes. Several approaches were proposed to address this issue by regularizing the learned soft prompts to align them with handcrafted text/hard prompts. However, excessive regularization of the soft prompts can hurt the model’s performance on the base classes it is trained on. Maintaining the right balance to ensure strong base- and novel-class performance is crucial but non-trivial. In this paper, we introduce a novel subspace-based prompt learning method, named SuPr, which can effectively model subspaces spanning the embeddings
of both the learnable soft and the textual/hard prompts. Our subspace-based alignment between hand-crafted and learnable prompts balances these effects to achieve excellent fitting of base classes as well as generalization to novel classes. With the advantages of subspace modelling, our SuPr shows its effectiveness on generalization from base to new, domain generalization, cross-dataset transfer and few-shot learning, leading to new state-of-the-art results in all settings.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work focused on how to conduct prompt tuning on vision-language models (i.e., CLIP), and proposed a subspace-based prompt learning method that divided soft prompts with orthonormal subgroups, regularized by hard prompts. Experiments on base-to-new classes, domain generalization, and cross-dataset transfer settings show the effectiveness of the method.

### Strengths
+ The proposed method achieved competitive performance on base-to-new classes, domain generalization, and cross-dataset transfer settings.

+ The method is simple but effective, although some insights behind the method are not clear now.

### Weaknesses
- Analysis about "Is subspace modeling useful" in Section 4.4. The conclusion is obtained based on the comparisons between SuPr w/o reg, CoOp, and CoOp-Ensemble. It is not clear what are the detailed differences among the three methods, which is essential to understand whether the comparisons can lead to the conclusions, as the performance gain may come from other components.

- SVD for subspace modeling. It is a bit hard for me to understand the role of SVD in subspace modeling. According to Sec. 3.2, it seems that SVD is to guarantee that the matrix $U_c$ is an orthonormal matrix. If so, is it possible to only restrict $U_c$ to be orthonormal without the SVD operation? Also, it is interesting to know the ablation where $U_c$ is no longer an orthonormal matrix. In this potential ablation study, can we say the subspace are no longer disentangled/independent?

- Main technical contribution. It seems that the main messages of this work are (1) dividing soft prompts into subgroups, (2) regularizing soft prompts with hard prompts. There lack insights why the subgroup manner works beyond the technical tricks.

- Analysis on subspace. Does the subspace have any semantic information, or what does each subspace represent? That would contribute to explainability.

### Questions
Please see weaknesses for detailed comments.

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
The paper proposes SuPr, a novel sub-space prompt learning method to improve the generalization ability of large pre-trained vision language models, especially CLIP. Specially, authors learned several partitions of soft prompts and project them into subspaces while using hard prompts to regularize them. The experiment results show the effectiveness of their method.

### Strengths
1.	Improving the generalization ability of pre-trained models is a interesting topic.
2.	Using subspace to enrich the semantic meaning of soft prompts is a interesting direction.

### Weaknesses
1.	Results are not consistent. For some dataset, it can achieve slightly better results than SOTA methods, but the results are not good in EuroSAT dataset. The author should explain reasons or assumptions at least.
2.	The experiments are not enough. For example, there is no numerical ablation study for each component. 
3.	Overall, the paper is written in a rush way which results in many confusing explanations.

### Questions
See the weakness part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
this paper addresses the prompt learning of vision-language models to achieve better base- and novel-cllass performance with subspace  modelling.  The papers proposes the subspace modelling of soft prompts, as well as its regualization with hard prompts and ensembling methods. Experiments verified the effectiveness of the proposed method.

### Strengths
1. the overall method and experiments are reasonable and convincing. This is a good practice for VLMs soft prompting. 
2. the paper is well written and easy to follow. 
3. the paper marks the first integration of subspace modelling with VLMs.

### Weaknesses
the improvement of this paper is not significant according to the Tables (<1% in Table 1, 2,3).

### Questions
1. this is a good practice of  integration of subspace modelling with VLMs. How about the novelty of the method in the subspace modellling domain?
3. Why LASP is not compared in Table 3 and Table 4?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new subspace-based prompt learning method to search a balance between hand-crafted and learnable prompt. The learn model can achieve high performance on the base classes and it can also generalize to new classes.

### Strengths
-The paper is well-written and easy to follow.

-It is interesting to see that the proposed method work well on many datasets.

### Weaknesses
-The proposed method fix the parameters of text encoder and image encoder. Will it achieve better performance when making all these parameters learnable.

-Will the proposed training strategy introduce extra training cost?

### Questions
See the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
