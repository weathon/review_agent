# Fine-tuning CLIP’s Last Visual Projector: A Few-Shot Cornucopia

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 5, 3, 5

## Abstract
We consider the problem of adapting a contrastively pretrained vision-language model like CLIP (Radford et al., 2021) for few-shot classification. The existing literature addresses this problem by learning a linear classifier of the frozen visual features, optimizing word embeddings, or learning external feature adapters. This paper introduces an alternative way for CLIP adaptation without adding “external” parameters to optimize. We find that simply fine-tuning the last projection matrix of the vision encoder leads to strong performance compared to the existing baselines. Furthermore, we show that regularizing training with the distance between the fine-tuned and pretrained matrices adds reliability for adapting CLIP through this layer. Perhaps surprisingly, this approach, coined ProLIP, yields performances on par or better than state of the art on 11 few-shot classification benchmarks, few-shot domain generalization, cross-dataset transfer and test-time adaptation. Code will be made available online.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes only to fine-tune the last projection layer of the vision encoder and finds this simple paradigm can yield on-par or better few-shot classification performance.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed method seems to be simple and effective.

### Weaknesses
1. I hope to see more in-depth analysis and discussion to indicate the reasons why ProLIP is effective, rather than descriptions like "Perhaps surprisingly." The current version of the paper makes me feel that the proposed method is trivial.

2. The paper seems to show significant performance improvements only in specific datasets (e.g., EuroSAT) or 16-shot scenarios, while the performance improvement is quite limited in other contexts.

3. The paper conducted a grid search for training parameters, which may be the reason for its superiority over other methods. I am curious whether using a similar parameter search for the baseline methods would diminish ProLIP's advantages.

4. The paper lacks performance comparisons with more advanced methods, such as PromptSRC[1] and CoPrompt[2].

[1] Self-regulating Prompts: Foundational Model Adaptation without Forgetting. ICCV, 2023.

[2] Consistency-guided Prompt Learning for Vision-Language Models. ICLR, 2024.

### Questions
Please refer to the 'Weaknesses' part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ProLIP, a method for few-shot adaptation of CLIP. The core idea of ProLIP is to fine-tune only the final visual projection matrix of the vision encoder during training. Additionally, it incorporates L2 regularization to maintain the matrix's proximity to the pretrained version. ProLIP is simple and efficient, demonstrating strong performance relative to other few-shot methods.

### Strengths
1.	The approach is simple and efficient. ProLIP does not introduce new training parameters; instead, it fine-tunes a pretrained projection matrix, making it more efficient than prompt learning.
2.	The performance is strong. ProLIP outperforms methods like LP++ and adapter-based approaches. 
3.	The presentation is clear and easy to understand.
4.	The experiments are comprehensive. In addition to comparisons with other state-of-the-art methods, the authors investigate the sensitivity to hyperparameter choices and the parametric λ.

### Weaknesses
There are no specific drawbacks noticed.

### Questions
How does ProLIP compare to LP++ with the proposed regulation. I believe the primary distinction between ProLIP and LP++ lies in ProLIP's imposition of regularization on the projection layer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces ProLIP, a method for few-shot adaptation of the CLIP vision-language model. ProLIP fine-tunes only the final projection layer of CLIP’s visual encoder, eliminating the need for additional parameters or complex architecture changes. This approach aligns with CLIP's training by using text embeddings as classification weights, while a regularization term ensures that the updated weights stay close to their pretrained values, reducing overfitting. Experiments show that ProLIP achieves competitive or superior performance to existing few-shot methods across tasks such as cross-dataset transfer and domain generalization.

### Strengths
1: ProLIP requires only fine-tuning of the last projection layer in CLIP’s visual encoder, eliminating the need for additional parameters, such as external adapters or prompt-tuning layers. This reduces memory and computational costs, making it feasible for resource-limited scenarios.

2: The proposed method is simple, i.e., only fine-tuning the visual projector in the CLIP model.

3: ProLIP includes a norm regularization that constrains the projection matrix’s weights, keeping them close to their pre-trained values. This regularization not only prevents overfitting in few-shot scenarios but also enhances the model’s stability and generalization.

### Weaknesses
1: The overall work sounds a little bit trivial. This paper mainly claims that only fine-tuning the final projector layer (part of model weights) can benefit the few-shot classification task. The modification is a kind of simple baseline rather than a novel method. There are also fewer provided insights into the design and the scope of downstream tasks is only limited to the few-shot task.  

2: Misaligned results. The results of the proposed ProCLIP are not consistent between Table 1 and Table 2.

### Questions
N/A.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces ProLIP,  which involves fine-tuning only the last visual projection matrix of the CLIP model, for post-training. This fine-tuning uses a masked reconstruction loss to learn semantic contributions for each image patch, enhancing the model's ability to capture local semantics without the need for additional annotated data.

### Strengths
1. The paper evaluates ProLIP on various vision-centric and vision-language benchmarks. Besides, the method is shown to be effective across different datasets in few-shot classification, domain generalization, cross-dataset transfer, and test-time adaptation.
2. The method is applicable to various models trained with image-level supervision, including CLIP and SigLIP, and can be used for different vision-language tasks.

### Weaknesses
1. The main concern is the novelty, the approach presented in this article is more of a trivial trick than a novel academic contribution. To the best of my knowledge, fine-tuning the mlp of the last layer of a large model is a very common trivial trick for fine-tuning either large language models and mllm models.
2. The experimental performance still falls short of the state-of-the-art approachs [1,2,3,4], suggesting that such a simple strategy is not the best one for fine-tuning.
3. The performance of ProLIP is sensitive to the choice of regularization strength ($\lambda$), which may require tuning or adaptive strategies for different datasets or tasks.
4. The paper assumes that pre-trained models contain sufficient knowledge for local semantics, which may not always be the case and could limit the method's effectiveness in certain scenarios.

[1] Khattak, Muhammad Uzair, et al. "Maple: Multi-modal prompt learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[2] Khattak, Muhammad Uzair, et al. "Self-regulating prompts: Foundational model adaptation without forgetting." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
[3] Wang, Yaoming, et al. "VioLET: Vision-Language Efficient Tuning with Collaborative Multi-modal Gradients." Proceedings of the 31st ACM International Conference on Multimedia. 2023.
[4] Roy, Shuvendu, and Ali Etemad. "Consistency-guided prompt learning for vision-language models." arXiv preprint arXiv:2306.01195 (2023).

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes ProLIP, a parameter-efficient method for adapting CLIP to few-shot classification. ProLIP proposes to finetune only the last projection layer of the vision encoder without adding other parameters or requiring prompt engineering. It also employs regularization of the weight matrix to preserve pretrained knowledge of the original CLIP. It achieves strong performance on few-shot classification and test-time adaptation.

### Strengths
* ProLIP provides a very simple and efficient adaptation compared to the previous methods, by finetuning only the last projection layer of the vision encoder. 

* ProLIP shows comparable or superior performance on various tasks from few-shot classification, domain generalization and test-time adaptation.

### Weaknesses
There are several aspects where the paper could be improved:

* A more in-depth analysis of how ProLIP achieves competitive results compared to more parameter-intensive methods. This would help providing insights into why this simple approach is so effective.

* Experiments with larger backbones, like ViT-Large, could be beneficial. Given that the method finetunes only the last projection layer, model scalability may be one of ProLIP’s strengths and could add depth to this.

* Additional evaluations on diverse out-of-domain datasets. Since ProLIP fine-tunes only a very small portion of the model, it would be interesting to see the range of new data domains and tasks that ProLIP can adapt (or where it fails) to in few-shot settings.

* Ablation on different shot numbers. It would be beneficial to have additional analysis of ProLIP’s performance across different few-shot settings (e.g., from 1-shot to N-shots), to show how well ProLIP leverages an increasing number of shots.

* ProLIP could potentially struggle on tasks with longer texts as the text encoder is kept completely frozen, compared to the prompt tuning methods.

### Questions
* Could ProLIP be combined with prompt tuning methods, and if so, would this joint approach yield even greater performance improvements? Exploring this combination might reveal synergistic benefits of ProLIP’s efficient layer tuning with prompt-based adaptations.

* How does ProLIP perform on datasets not represented in CLIP’s pretraining (e.g., medical or aerial imagery)? Expanding on the cross-dataset experiments could further demonstrate ProLIP’s adaptability to diverse data distributions.

* Comparison of training time and computation against prompt tuning and adapter-based methods.

### Soundness
3

### Presentation
3

### Contribution
2
