# Fine-tuning can cripple foundation models; preserving features may be the solution

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 5, 3

## Abstract
Pre-trained foundation models, due to their enormous capacity and their training using vast amounts of data can store knowledge about many real-world concepts. To further improve performance on downstream tasks, these models can be fine-tuned on task specific datasets. While various fine-tuning methods have been devised and have been shown to be highly effective, we observe that a fine-tuned model's ability to recognize concepts on tasks different from the downstream one is reduced significantly compared to its pre-trained counterpart. This is clearly undesirable as a huge amount of time and money went into learning those very concepts in the first place. We call this undesirable phenomenon "concept forgetting'' and via experiments show that most end-to-end fine-tuning approaches suffer heavily from this side effect. To this end, we also propose a rather simple fix to this problem by designing a method called LDIFS (short for $\ell_2$ distance in feature space) that simply preserves the features of the original foundation model during fine-tuning. We show that LDIFS significantly reduces concept forgetting without having noticeable impact on the downstream task performance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose an extensive study to test the ability of several fine-tuning algorithms to preserve the initial knowledge of the fine-tuned models. Based on their results, they propose to add a distillation-based regularization term, aiming to prevent the so-called catastrophic forgetting. The experimental results over their approach shows the algorithm is capable of preserving the model's knowledge, with almost no accuracy drop in the fine-tuned dataset.

### Strengths
**originality**: this is the first time a have seem such a exhaustive analysis regarding the forgetting problem of fine-tuned models.

**quality**:  the authors provide an exhaustive and comprehensive analysis over multiple fine-tuning models, showing both their advantages and limitations.

**clarity**: the paper is easy to read and to follow. Furthermore, their regularization term is very easy to implement.

**significance**: the concept of catastrophic forgetting is of special interest for modern machine learning models that requires online training.

### Weaknesses
**originality**: as the authors already mentioned, their idea is extremely similar to some distillation techniques, specially with the concept of self-distillation [1]. Besides that, there were several studies focused on the idea of preserving the model's prior knowledge. This field is often referred as Lifelong Learning [2] (a subsection of the incremental learning theory). [2] also provides a knowledge distillation approach t o solve this problem, although I must say it is not the same to the one provided by the authors. I suggest the authors to introduce in their paper some of these approaches, stating the advantages of their solution.

**significance**: due to the originality issue, it is not clear that this paper provides enough innovations to be accepted in a venue of this kind.


[1] Zhang, L., Bao, C., & Ma, K. (2021). Self-distillation: Towards efficient and compact neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8), 4388-4403.

[2] Hou, S., Pan, X., Loy, C. C., Wang, Z., & Lin, D. (2018). Lifelong learning via progressive distillation and retrospection. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 437-452).

### Questions
- What are the advantages of your algorithm compared with other lifelong learning approaches (like the one mentioned in the weaknesses section)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper analyzes the concept of forgetting phenomenon when fine-tuning pre-trained models.
To solve such a forgetting problem, they propose a method called LDIFS which adds l_2 distance to the feature space.
The method can significantly reduce the concept forging on several downstream datasets.

### Strengths
1. The writing and presentation is fluent.
2. The analysis of forgetting on various datasets is comprehensive.
3. The method is easy to understand and reasonable.

### Weaknesses
1. The baselines to be compared are limited. The experiments mainly compare with baselines with additional regularization to prevent forgetting. But works like Wise-FT mentioned in the related work also show good and even better performance on preventing forgetting which is not compared in this paper.

2. It is not direct to tell how good the proposed method is from Table 1. 
It will be more clear if the results in Table 1 can be visualized in some other direct ways, e.g., take the car's value as the x-axis and others as the y-axis, then plot the performance of all methods on a picture.

### Questions
1. I think it will be better to add some experiments that compare the methods like WISE-FT or analyze whether the proposed method can combined with the WISE-FT to achieve a better performance.
2. As we know, the hyperparameters of the regularization weights \lambda may affect the final performance of the fine-tuned model. But I do not find any discussion about that. It will be better to add such kind of analysis about the hyperparameters.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a regularization method to reduce the "concept forgetting" when finetuning a pre-trained foundational model with limited data. It first shows that finetuning CLIP on a downstream task degrades the model's recognition of concepts outside the target downstream task. To mitigate this issue, it proposes a simple regularization loss, which encourages to preserve the original CLIP during finetuning. Specifically, it empirically shows that minimizing the l-2 distance to the original model in the feature space (LDIFS) is consistently better than minimizing the l-2 distance in the parameter space, resulting in less "concept forgetting". Experiments are done for 6 different finetuning methods on 9 downstream classification tasks.

### Strengths
- This paper studies an interesting problem, i.e., "concept forgetting" of a pre-trained foundational model when it is finetuned to relatively small data of a target downstream task.
- The idea to minimize the distance to the original model in the feature space is simple but works well. In the experiments, it consistently shows better performance than the alternative to minimize the distance in the parameter space.
- In particular, the paper provides extensive experiments on all combinations of 6 different finetuning methods and 9 classification datasets.
- The paper is well written. It was easy to follow.

### Weaknesses
1. Although the paper reported results of many experiments, I think the impact could be somewhat limited because all downstream tasks are classification with relatively small-sized training data. I am not sure if the observation in this paper (that l2-distance in features space is better metric for regularization) will generalize to other practical settings when there are medium-sized training data that covers many concepts. For example, there are observations that CLIP finetuned on existing detection dataset achieves good performance on open vocabulary detection [1], which implies that the finetuned CLIP can recognize concepts unseen in the finetuning dataset instead of forgetting them, possibly by interpolating the seen concepts. Also, when downstream task is not a classification task but is less similar to pre-training contrastive loss, such as detection, the stability-plasticity trade-off could be worse and the effect of regularization methods could be different. I think it would be interesting to do similar experiments in the paper to such other downstream tasks.

2. I think "catastrophic forgetting" and "concept forgetting" address the same problem, as also mentioned in the paper. Since catastrophic forgetting have been studied in many continual learning approaches, it would be nice to discuss them in more detail in the related work chapter and also add comparison with some representative methods [2] in the experiments as strong baselines when they are applicable. Especially, the idea to distill knowledge from the original model to the finetune model as a regularization looks similar, which has been studied for continual learning.

[1] Simple Open-Vocabulary Object Detection with Vision Transformers, ECCV 2022  
[2] A continual learning survey: Defying forgetting in classification tasks, TPAMI 2020

### Questions
- Additional discussion and comparison with continual learning methods would be necessary
- Additional evaluation on other downstream tasks beyond classification would be nice
- Tables and figures are very small. Some of them could be merged (fig 2 and 5) or moved to the appendix.
- Finetuning with reduced learning rate for the backbone [1] could be another baseline

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper discussed a phenomenon called “concept forgetting,” where fine-tuning a large-scale pre-trained vision language model (CLIP) on a downstream task will significantly reduce the performance on tasks distinct from the particular task. The authors showed extensive empirical evidence for concept forgetting, described a solution that distills features from the pre-trained model during fine-tuning, and presented promising results on continual learning.

### Strengths
* The paper addressed an important and trendy topic on adapting foundation models.

* The paper is well organized. Key concepts are clearly introduced and technical details are easy to follow.
 
* Results on continual learning are encouraging.

### Weaknesses
While the phenomenon of “concept forgetting” is interesting, as the authors admitted, it is closely related to catastrophic forgetting that was described in many prior works. Yes, I agree there are some subtle differences as described in Sec 3.1, yet it is not surprising to expect “foundation models” to have a similar behavior. Indeed, this forgetting phenomenon of CLIP has been discussed in some recent papers [a, b].

The proposed solution follows the knowledge distillation framework, which has been previously considered by several prior methods to alleviate catastrophic forgetting in the context of continual learning (e.g., iCaRL and [c]). The proposed solution is conceptually similar to those prior methods. The innovation seems quite incremental. 

[a] Lee et al., Do Pre-trained Models Benefit Equally in Continual Learning? WACV 2023

[b] Ding et al., Don't Stop Learning: Towards Continual Learning for the CLIP Model, aXiv 2022

[c] Li and Hoiem, Learning without Forgetting, TPAMI 2017

### Questions
It will be great if the authors can highlight the conceptual innovation of the paper w.r.t. those missing prior works (e.g., [a-c]).

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
