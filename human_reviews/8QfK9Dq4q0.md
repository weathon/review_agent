# Class Incremental Learning via Likelihood Ratio Based Task Prediction

- Decision: Accept (poster)
- Scores: 5, 8, 5, 6

## Abstract
Class incremental learning (CIL) is a challenging setting of continual learning, which learns a series of tasks sequentially. Each task consists of a set of unique classes. The key feature of CIL is that no task identifier (or task-id) is provided at test time. Predicting the task-id for each test sample is a challenging problem. An emerging theory-guided approach (called TIL+OOD) is to train a task-specific model for each task in a shared network for all tasks based on a task-incremental learning (TIL) method to deal with catastrophic forgetting. The model for each task is an out-of-distribution (OOD) detector rather than a conventional classifier. The OOD detector can perform both within-task (in-distribution (IND)) class prediction and OOD detection. The OOD detection capability is the key to task-id prediction during inference. However, this paper argues that using a traditional OOD detector for task-id prediction is sub-optimal because additional information (e.g., the replay data and the learned tasks) available in CIL can be exploited to design a better and principled method for task-id prediction. We call the new method TPL (Task-id Prediction based on Likelihood Ratio). TPL markedly outperforms strong CIL baselines and has negligible catastrophic forgetting. The code of TPL is publicly available at https://github.com/linhaowei1/TPL.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the author proposes the use of a Likelihood Ratio to identify the Task-id for Class Incremental Learning (CIL). Traditionally, out-of-distribution (OOD) detectors were used for task identification, but this paper introduces a new method called TPLR (Task-id Prediction based on Likelihood Ratio) that leverages additional information like replay data and learned tasks for more effective and principled task-id prediction. TPLR outperforms traditional CIL approaches.

### Strengths
The motivation and method sound solid to me. I agree with the author that task-id prediction is pivotal for CIL under specific circumstances.

- Motivation is clear and straightforward:  The author argues that using a traditional OOD detector is not optimal for task predictions and here they leverage the information in the CIL process for task-id prediction.

- The proof and methods in section 3 and 4 look good to me.

### Weaknesses
1. The writing requires improvement. The author frequently used abbreviations and jargon, especially in the introduction, which occasionally left me puzzled. It would be beneficial if these terms were interpreted more straightforwardly. 

2. The related works are also unclear: 
- Although the author clarifies their focus on Class incremental learning, which doesn't provide the task-id during inference, it remains ambiguous whether they are using a memory buffer (rehearsal-based) or are memory-free (online CIL). I suggest the author address this in the introduction and related works.
- Some recent benchmarks are missing: The author left memory-free (non-replay-based approaches) CIL in related works. The author also left balanced CIL works, e.g., SS-IL, TKIL.

3. Experimental settings:
- Table 1 is impressive, but the comparisons seem biased. The author claims they compared with 17 baselines, including 11 replay-based and 6 non-replay-based. From my understanding, the author requires a memory buffer, as indicated in the "Overview of the Proposed TPLR", equation 2.
-  It would be more equitable if the author juxtaposed their method with replay-based CIL. Specifically, the author should draw a clear comparison with methods using task-id prediction, highlighting the advantages of their technique. 
- One import baseline is missing: AFC[3]
  

4. The inference setting remains unclear. Does the author predict both the task-id and class-id simultaneously? Is there any fine-tuning step involved? Typically, some fine-tuning follows the task-id prediction. e.g., iTAML. If the author's method circumvents this, it could be seen as a distinct advantage. Therefore, I recommend the author incorporate a discussion about the computational load when integrating likelihood ratio predictions, elucidating the benefits and drawbacks of this model.

5. Lacks Visualizations: Could the author add a real visualization of data distribution, like the "Feature-based likelihood ratio score" in Figure 1. It will be strong evidence the TPLR works well.


[1] Ss-il: Separated softmax for incremental learning. In Proceedings of the IEEE/CVF International conference on computer vision 

[2] TKIL: Tangent Kernel Optimization for Class Balanced Incremental Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision 

[3] Class-incremental learning by knowledge distillation with adaptive feature consolidation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition

### Questions
Please refer the weakness;

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors address the challenge of task identification (task-id prediction) in Class Incremental Learning (CIL). They propose a novel method named TPLR (Task-id Prediction based on Likelihood Ratio), which enhances task-id prediction by utilizing replay data to estimate the distribution of non-target tasks. This approach allows for a more principled solution compared to traditional Out-of-Distribution (OOD) detection methods that cannot estimate the vast universe of non-target classes due to lack of data.

TPLR calculates the likelihood ratio between the data distribution of the current task and that of its complement, providing a robust mechanism for task-id prediction. The method is integrated into the Hard Attention to the Task (HAT) structure, which employs learned masks to prevent catastrophic forgetting, adapting the architecture to facilitate both task-id prediction and within-task classification.

The authors demonstrate through extensive experimentation that TPLR substantially outperforms existing baselines in CIL settings. This performance is consistent across different configurations, including scenarios with and without pre-trained feature extractors. The paper's contributions offer significant advancements for task-id prediction in CIL, proposing a method that leverages available data more effectively than prior approaches.

### Strengths
Originality:

- TPLR's innovation lies in its unique application of likelihood ratios for task-id prediction, an approach that distinctively diverges from traditional OOD detection methods.
- The paper creatively leverages replay data to estimate the data distribution for non-target tasks, which is a novel use of available information in the CIL framework.
- Integration of TPLR with the HAT method showcases an inventive combination of techniques to overcome catastrophic forgetting while facilitating task-id prediction.

Quality:

-The methodological execution of TPLR is of high quality. It is underpinned by a strong theoretical framework that is well-articulated and logically sound.
- Extensive experiments validate the robustness and reliability of TPLR, demonstrating its superiority over state-of-the-art baselines.

Clarity:

The paper writing quality is satisfactory.

Significance:

TPLR's ability to outperform existing baselines marks a significant advancement in the domain of CIL, potentially influencing future research directions and applications.
The paper's approach to using replay data for improving task-id prediction could have broader implications for continual learning paradigms beyond CIL.

### Weaknesses
The key weakness of this work I would argue is its overly complex presentation. I find that the organization of the paper can easily distract and confuse the reader, often finding myself fishing for key details of the main method.

### Questions
- While the writing quality is satisfactory, I would argue for a friendlier approach to outlining the proposed method. First, outline the key ingredients. Then explain how they interact. Finally cross-reference these with the existing figure. 
- The existing figure is a bit too 'noisy' in terms of the information it is showing and the order it is showing it in. Consider reorganizing it so it can be read from left to right, top to bottom and with more emphasis on the key ideas and less detail that can distract from that.

### Soundness
4 excellent

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel method for class incremental learning (CIL) by directly predicting the task identifiers to perform the task-wise prediction. Using the energy based models, the given model computes the scores for each task based on the Mahalanobis distance and KNN distance, and estimate the task label. Furthermore, the proposed model actively utilizes the pre-trained model by just training the adapter module to efficiently train the parameters. In the experiment, the given algorithm outperforms the baselines in both CIFAR and Tiny-ImageNet dataset. In addition, the authors show the effectiveness of using each component in the ablation study.

### Strengths
1. By directly estimating the task identifier, the proposed algorithm outperforms other baselines in the benchmark dataset.

2. Since the proposed model utilize the task-wise classifier, it can be robust to the class imbalance problem which can occur when the difference between the size  of replay buffer and training data are large.

### Weaknesses
1. I wonder the proposed methods can achieve high task-prediction accuracy. Different from the ideal situation, the accuracy may be lower than we expected. if the semantics across different classes are similar, the task-prediction accuracy can be low, and the overall performance also can decrease. 

2. Can this method outperform other baselines when it does not use the pre-trained model in ImageNet-1K? Furthermore, if the dataset used for pre-training are randomly selected (i.e. Randomly extract 500 classes from ImageNet-1K), can this method outperform other baselines? Since ImageNet-1K or other large datasets contain similar classes, the task-prediction is much harder than CIFAR or Tiny-ImageNet

### Questions
Already mentioned in the Weakness section

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose to use out-of-distribution ideas to solve the gap between task incremental and class incremental learning, indirectly predicting the task label and using it to refine the class prediction. They pose that using a low forgetting method such as HAT and pairing it with a good task-prediction from the ood-inspired setting, allows better estimates of both the intra- and inter-task probabilities, which leads to better performance in CIL scenarios.

### Strengths
The proposed method is simple and well explained, backed up with justification of why they choose the likelihood ratio strategy. The idea of using ood ideas to overcome the task-ID limitation of TIL is interesting and aligns with the continual learning community directions. The experimental results are compared with a large array of existing methods and state-of-the-art approaches.

### Weaknesses
The proposed method is for the most part an extension of existing previous work, which requires a replay buffer, pretrained models and the need for a forward pass for each task learned. Therefore, the advantage of not needing the task label at inference is not well contrasted with the limitations (mostly mentioned at the end of the appendix only). I would expect further discussion and justification about how these benefits and limitations balance in the main part of the manuscript.

### Questions
It is mentioned that HAT prevents CF, but it actually only mitigates it. It is discussed later in the appendix that a very large sigmoid is used in order to force an almost binary mask to promote more of that CF mitigation. However, how relevant is that the masks are binary and that the sigmoid is close to a step function? Would then a method that guarantees no forgetting such as PNN [Rusu et al. 2016], PackNet [Mallya et al. 2018] or Ternary Masks [Masana et al. 2020] be more suitable for the proposed strategy? How do you deal with HAT running out of capacity when the sequence gets longer?

In Table 1, which of these results are using the task label at inference time? For example, HAT needs the task label. So are the results of HAT comparable here with the other methods? Or is HAT having a forward pass with each task label and then using some heuristic to pick the class?

For the experiments on running time, in Table 9 of the appendix it is only shown the running times for the 4 methods that have the same base strategy. How do those compare with all the other methods, because I would assume that for large sequences of tasks, it might become quite a limiting factor to have to forward each sample/batch T times. I would argue that is a relevant discussion to have in the main manuscript.

In the introduction it is mentioned "This means the universal set [...] includes all possible classes in the world [...], which is at least very large if not infinite in size...". Is there some paper or relevant source to back this? One of the papers that comes to mind is [Biederman, 1987], which states that there are between 10k to 30k visual object categories that we can recognize in images. And that would hint towards learning an estimate of the distribution for objects in images would not be such unfeasible (specially now with foundational models).

In conclusion, I find the idea interesting and relevant. However, the small extension from existing related work, and the lack of a better discussion of the limitations and motivation/relevance for the community could be improved.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
