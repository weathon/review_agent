# On the Transfer of Object-Centric Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 5, 3, 6

## Abstract
The goal of object-centric representation learning is to decompose visual scenes into a structured representation that isolates the entities into individual vectors. Recent successes have shown that object-centric representation learning can be scaled to real-world scenes by utilizing features from pre-trained foundation models like DINO. However, so far, these object-centric methods have mostly been applied in-distribution, with models trained and evaluated on the same dataset. This is in contrast to the underlying foundation models, which have been shown to be applicable to a wide range of data and tasks. Thus, in this work, we answer the question of whether current real-world capable object-centric methods exhibit similar levels of transferability by introducing a benchmark comprising seven different synthetic and real-world datasets. We analyze the factors influencing performance under transfer and find that training on diverse real-world images improves generalization to unseen scenarios. Furthermore, inspired by the success of task-specific fine-tuning in foundation models, we introduce a novel fine-tuning strategy to adapt pre-trained vision encoders for the task of object discovery. We find that the proposed approach results in state-of-the-art performance for unsupervised object discovery, exhibiting strong zero-shot transfer to unseen datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a benchmark to evaluate the zero-shot generalizability of object-centric learning methods. The benchmark comprises seven synthetic and real-world multi-object datasets, allowing comprehensive testing across different scenarios. Additionally, the paper demonstrates that training on these diverse real-world images significantly enhances the transferability of existing methods to unseen contexts. Moreover, the authors propose an object-centric fine-tuning strategy, building on DINOSAUR’s pre-trained vision encoder, which achieves state-of-the-art performance in unsupervised object discovery with robust zero-shot transfer abilities.

### Strengths
- The paper addresses an important topic of exploring zero-shot object-centric learning OOD settings.
- The fine-tuning approach of DINOSAUR improves object-discovery performance and closes the gap between zero-shot and supervised learning.
- Extensive experiments are conducted across various datasets, providing insights into the zero-shot generalizability and scalability of object-centric models.

### Weaknesses
My major concern on this paper is regarding the evaluation design:
- The evaluation relies on pre-trained DINO features, which may already possess OOD capabilities due to large-scale pre-training. Since object-centric learning methods using pre-trained encoders only need to focus on feature grouping, these methods can easily transfer learned grouping techniques to new datasets supported by the powerful pre-trained features. As a result, it is difficult to ascertain whether the observed zero-shot generalization stems from the object-centric method itself or DINO’s inherent robustness.
- The paper uses object segmentation as the primary evaluation metric, which might not fully capture the quality of the learned representations. The inclusion of additional downstream tasks, such as object classification or property prediction, would provide a more comprehensive evaluation of the generalizability.

Furthermore, the method proposed in this paper offers limited novelty: the fine-tuning process relies on well-known techniques and does not introduce significant new insights beyond improved hyperparameter tuning.

Lastly, the claim that “training on diverse real-world images improves transferability to unseen scenarios” adds little to the current knowledge, as it is already widely accepted that increased data quantity and diversity generally lead to better model performance.

### Questions
See the weaknesses section. Besides, I strongly encourage the authors to conduct experiments without the DINO encoder. Since DINO is pre-trained on ImageNet, which is significantly larger than those used in this paper’s experiments, it's hard to accurately evaluate the model's transferability when using DINO.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies the zero-shot performance of object-centric representation learning. The authors first combine 7 existing datasets to establish a comprehensive testbed, enabling them to evaluate the transferability of three recently proposed object-centric learning models. Based on the experimental findings, they propose finetuning the pre-trained encoders on the COCO dataset to enhance zero-shot performance on their testbed.

### Strengths
- The paper is well-motivated, beginning with a clear question and proposing a testbed to evaluate the performance of current methods. Based on the evaluation results, the authors propose method to further enhance the model.

- The proposed finetuning method is straightforward yet effectively addresses model collapse problem

### Weaknesses
1. I would consider this as an analysis paper. However, many of the conclusions appear somewhat obvious. For instance, SAM already demonstrates strong zero-shot transfer capabilities in object-centric models, even extending well to domains like medical imaging. Additionally, it’s widely understood that training on complex natural datasets is crucial for zero-shot transfer performance, and using real data is key to enhancing this ability.
This paper would be more interesting with additional and more detailed experiments and insights provided by the testbed. For example, is it possible to identify some common failure cases and explore possible reasons behind these failures? I guess that could offer more insights.
2. The paper proposes to finetune on an additional dataset (COCO) to enhance zero-shot performance. However, it doesn't have any specific design for zero-shot task. Additionally, task-specific finetuning on a general pretrained model typically leads to performance improvements.

### Questions
Please see the weaknesses section. 

A small question: Would it be possible to provide a more detailed comparison of the similarities between COCO and the 7 datasets in testbed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper discusses the transferability of self-supervised models, such as DINO, and the factors about the transferability performance. Then, the authors further explore the finetuning strategy to adapt the model to conduct object discovery task.

### Strengths
+ The topic about object discovery task is interesting.

### Weaknesses
The writing is confusing and difficult to understand. Not only the expression of writing is confusing, but also many important clarifications about technique can not be found in the paper. We list these issues as follows.
+ The evaluation of zero-shot transfer. In the main body of paper, the authors only list the datasets and metrics used for evaluation, but don't mention why and how.
+ For object centric finetuning, the authors don't mention how to conduct slot attention and top-k mlp decoder (the structure details) and why use these module. What is the loss function? Where do the output images come from in Figure 3? Are the output masked images just the output of DINOv2?
In summary, I am inclined to reject this submission in the current presentation.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the zero-shot transfer capabilities of object-centric representation learning models and proposes improvements to enhance their generalization. The key contributions are:
- A benchmark comprising 7 diverse datasets to evaluate zero-shot transfer of object-centric models.
- A novel finetuning approach that adapts pre-trained vision encoders for object discovery.
- The method proposed in this paper achieves state-of-the-art results for object discovery tasks on both in-distribution and out-of-distribution scenarios.

### Strengths
- Technical Innovation:
	- First systematic study of zero-shot transfer in object-centric learning
	- Novel finetuning strategy that successfully adapts pre-trained encoders
- Adequate experimentation and analysis：
	- Comprehensive empirical evaluation across multiple datasets and metrics
	- Thorough ablation studies validating each component
- Clear presentation:
	- Well-structured and clearly written
	- Comprehensive appendix with implementation details

### Weaknesses
- The proposed method lacks novel and essential insights.  The fine-tuning strategy, high-resolution adaptation, and top-k decoding are engineering improvements that come easily to mind. 

- The paper shows that current models don't scale well with data size (Fig 2b), especially for real-world data, but doesn't propose any analyses or solutions for this limitation.

- These datasets, while diverse, are still relatively small-scale for pre-training compared to modern vision datasets. The ScanNet and YCB datasets used in [1] provide images without background. Only COCO, PASCAL, and EntitySeg are more consistent with the natural image distribution. This may be one reason why no evident scaling law has been observed.

[1] Promising or Elusive? Unsupervised Object Segmentation from Realworld Single Images.

### Questions
1. I'd like to discuss the "blockwise exponentially decaying learning rates". According to the ablation studies (Tab. 1),  blockwise learning rates don't bring significant improvement. The authors propose it because "the encoder would initially drift away from its pre-trained initialization, likely induced by the noisy gradients from the randomly initialized slot attention module". Since this phenomenon may be caused by the randomness of slot initialization, why not consider changing the initialization method of slots? Why not initialize slots with learnable queries like BO-QSA [1], OSRT [2], and SPOT [3], which have been proven to be effective? Instead, this paper introduces "blockwise exponentially decaying learning rates".  If initializing slots with learnable queries does not solve this problem, the problem may not be caused by random slot initialization. 

2. I'm interested in why these models don't scale well with dataset size, especially on real-world datasets such as COCO. What might help to improve scaling behavior? If the authors find out in their experiments what factors help enhance scaling behavior, I believe it is important to highlight them to provide more insight into the progress in this field, which can also increase the impact of this work.

[1] Improving Object-centric Learning With Query Optimization.
[2] OSRT: Object Scene Representation Transformer.
[3] SPOT: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers.

### Soundness
4

### Presentation
4

### Contribution
3
