# Teacher Ascent: Robust and Efficient Machine Unlearning via Knowledge Distillation and Continual Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Removing specific knowledge from a trained machine learning model is an open problem of increasing importance. Growing dataset sizes increase the likelihood of introducing biased, inaccurate, or private data. Moreover, increasing the number of parameters makes retraining models more costly. 
While powerful Machine Unlearning methods have emerged as effective alternatives to retraining, their practical application is often hindered by narrow functional ranges for hyperparameters, which typically require access to a retrained model for effective tuning. Established unlearning methods like SCRUB+R and SSD require precise specification of their hyperparameters to achieve unlearning whilst preventing catastrophic forgetting. We address this challenge by proposing Teacher Ascent (TA), a novel unlearning method that is based on knowledge distillation and continual learning. 
Inspired by Elastic Weight Consolidation (EWC), TA forgets target data while protecting parameters essential for generalization by using the Fisher Information Matrix. We conduct experiments on MNIST, CIFAR, and Pins Face Recognition across various unlearning scenarios: forgetting entire classes, subclasses, and mislabeled samples. Our results demonstrate that Teacher Ascent both mimics the functional behavior of a retrained model across unlearning tasks while being 6-19 times more efficient than retraining. More importantly, TA mitigates catastrophic forgetting and demonstrates robustness across a wide range of hyperparameters.
By overcoming the critical stability and tuning challenges of previous approaches, Teacher Ascent represents a significant step towards making machine unlearning a viable and practical tool for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Teacher Ascent(TA), a novel and efficient fine-tuning method for Machine Unlearning designed to overcome the critical hyperparameter sensitivity and stability issues, such as catastrophic forgetting, inherent in prior state-of-the-art methods like SCRUB+R and SSD. Teacher Ascent is built on principles from knowledge distillation and continual learning, aiming to remove the influence of a specified forget set ($D_f$) while preserving performance on the retain set ($D_r$), thereby producing a model functionally equivalent to one trained only on $D_r$. The method achieves this stability through two key mechanisms: maximizing bounded Kullback-Leibler divergence terms during data removal to circumvent catastrophic forgetting, and employing a regularization term inspired by Elastic Weight Consolidation (EWC) that uses the Fisher Information Matrix (FIM) to protect parameters essential for generalization. Across diverse unlearning scenarios involving entire classes, subclasses, and mislabeled samples on datasets like MNIST, CIFAR, and Pins Face Recognition, the results demonstrate that Teacher Ascent consistently mimics the functional behavior of a retrained model.

### Strengths
- TA effectively mitigates catastrophic forgetting and demonstrates robustness across a wide range of hyperparameters, solving a critical reliability gap present in state-of-the-art methods. Specifically, TA achieves this stability by employing bounded KL divergence terms during the removal of the forget set, and incorporating a regularization term (inspired by EWC) that protects parameters essential for generalization using the Fisher Information Matrix (FIM) ratio of the retain and forget sets.

- TA is significantly more efficient than the full retraining from scratch on retain sets. Across various unlearning tasks, TA is shown to be 6 to 19 times more efficient than retraining while still matching the performance of the retrained model.

### Weaknesses
- The comparison currently centers on SSD and SCRUB+R, but several contemporary KD-based unlearning methods use the original model as a teacher and thus provide a natural point of reference for TA (e.g., DELETE[1]: decoupled/masked distillation). Even if some assumptions differ, including at least one such baseline—or clearly stating exclusion criteria (data access, objective, or compute)—would make the empirical section more complete.

- TA targets a full-access scenario where both $D_r$ and $D_f$ are available at unlearning time, and the method explicitly relies on $D_r$. Considering the realistic scenario in which an unlearning request comes in from the outside, such full-access assumption is not suitable for unlearning. The paper reveals that only mini-batch is sampled from $D_r$ in the Repair stage, fixed to k=1 in all experiments, and only speculates that "a large dataset may require k>1". There is no experiment/ablation to systematically reduce (ratio and k change) the $D_r$ usage or evaluate the 'only partially used' variation.

- In Table 3, please clarify the boldface rule for Time. If SSD (or any other method) is faster but fails the unlearning criteria, consider (a) bolding the fastest among methods that meet success criteria, with those criteria stated in the caption, and (b) visually flagging failed settings (e.g., gray or a footnote). This prevents misinterpretation of boldface as the absolute column minimum.

- Line 86, Minor typos: “Elastic Weight Consollidation” → “Consolidation”

[1] Zhou, Y., Zheng, D., Mo, Q., Lu, R., Lin, K. Y., & Zheng, W. S. (2025). Decoupled distillation to erase: A general unlearning method for any class-centric tasks. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 20350-20359).

### Questions
- TA requires computing diag FIM on datasets at $\theta_0$. Therefore, the cost increases linearly with the number of parameters and the size of the data. I think it's hard to use it practically in a large model.

- Recent evidence suggests that MIA scores tend to become less informative as model capacity grows. Experiments on the large model are not in this paper, but what do you think about this?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Teacher Ascent (TA), a novel fine-tuning–based machine unlearning method that removes the influence of specific data subsets without full retraining. TA integrates knowledge distillation with Elastic Weight Consolidation (EWC), using a Fisher Information–based regularizer to preserve parameters essential for generalization while bounding the KL-divergence to prevent catastrophic forgetting. Compared to prior methods like SCRUB+R and SSD, TA exhibits strong robustness to hyperparameter variations and avoids instability caused by unbounded divergence terms. Experiments on MNIST, CIFAR-10/100, and Pins Face Recognition show that TA achieves performance comparable to a retrained model while being more computationally efficient.

### Strengths
The paper is well-motivated, clearly identifying the limitations of the previous state-of-the-art method SCRUB+R, particularly its instability and sensitivity to hyperparameters, and introducing a simple yet effective solution through the proposed Teacher Ascent (TA) framework.

TA demonstrates strong robustness, stability, and efficiency in unlearning tasks. As shown in Figure 1, TA maintains performance comparable to retraining while avoiding the catastrophic forgetting observed in SCRUB+R. Figure 2 illustrates TA’s low accuracy variance across a wide range of hyperparameters, highlighting its robustness and ease of tuning. Table 3 provides empirical evidence of TA’s significant computational efficiency, achieving much faster running times than SCRUB+R while maintaining similar unlearning quality.

### Weaknesses
Novelty: While the paper presents a well-engineered improvement over SCRUB+R, the novelty and insight appear somewhat incremental. TA essentially extends SCRUB+R by adding two additional components: (1) an entropy maximization term (Eq. 5) to enhance forgetting stability, and (2) an EWC-style FIM regularization with dampening factor (proposed by SSD) to preserve critical parameters. Although these modifications yield practical improvements, they mainly combine existing techniques rather than introducing a fundamentally new theoretical perspective or algorithmic paradigm. 

Contribution: The paper does not provide deeper analysis or theoretical justification explaining why these components effectively mitigate SCRUB+R’s drawbacks. As a result, the insight and interpretability of the proposed approach remain limited. Incorporating theoretical analysis could strengthen the contribution and clarify the underlying mechanism behind TA.

Experiment: The paper compares TA against SCRUB+R (2023) and SSD (2024) only, which does not convincingly demonstrate the effectiveness of TA. Given that the venue will be held in 2026, the author needs to compare it against broader and more recent unlearning baselines.

### Questions
1. Can the authors explain why the proposed terms can improve the robustness of hyperparameter selection?
2. Why does TA compute faster than SCRUB+R, given that they are similar? Is that because the k was set to 1 in TA?
3. Since SCRUB+R is sensitive to hyperparameters, what is the strategy of its hyperparameter selection? Since they are searched by TPE, does that affect and lower the performance of it?

### Soundness
2

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
2

### Summary
This paper proposes Teacher Ascent (TA), a robust and efficient machine unlearning method. Its main strength is a novel design that addresses the hyperparameter sensitivity of prior work, demonstrated through a more realistic evaluation protocol. However, the experimental validation is limited to small-scale datasets, which raises concerns about its real-world applicability.

### Strengths
Novel and Robust Method: TA's design, featuring bounded KL-divergence and a discriminative EWC regularizer, is a novel and effective solution to the instability problems of prior SOTA methods.
Practical Evaluation Protocol: The proposed protocol of using a proxy task for hyperparameter tuning is innovative and better reflects real-world constraints.
Strong Performance: On the tested benchmarks, TA achieves performance comparable to the gold-standard retraining but is significantly more efficient and robust.

### Weaknesses
Formatting Errors: There are multiple formatting issues: missing commas after "e.g.", missing punctuation for equations, unnumbered equations, and the caption for Table 2 is incorrectly placed below the table.
Limited Experimental Scale: All experiments are conducted on small-scale datasets (CIFAR-10/100). This is a major concern, as it's unclear if the method's effectiveness and efficiency gains will generalize to the large-scale models and larger forget sets common in real-world applications. Experiments on a larger scale are needed.

### Questions
My primary concern, which will heavily influence my final recommendation, is the limited scale of the experimental validation. The paper makes strong claims about real-world applicability and robustness, yet the empirical evidence is exclusively based on small-scale datasets like CIF-AR-10/100, where forget sets are only a few hundred images. This raises critical questions about the method's generalizability.
To address this limitation, I would like the authors to respond to the following points during the rebuttal phase. A convincing response here could significantly change my opinion of the paper.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Teacher Ascent (TA), a machine unlearning method designed to remove specific knowledge from a trained model without retraining it from scratch. This work improves upon SCRUB+R, which applies knowledge distillation to align the student’s output distribution with the original model on the retain dataset while encouraging a uniform output on the forget dataset. In addition, it introduces a continual learning inspired regularization based on Elastic Weight Consolidation (EWC) to selectively preserve parameters important to the retain dataset, preventing catastrophic forgetting.

### Strengths
The methodology section is clearly written and well-structured. The logical flow of the motivation and related work is coherent and easy to follow, making the paper accessible even to readers who are new to the field.

### Weaknesses
However, the experimental section lacks sufficient clarity. For example, the authors do not adequately explain each line in Figure 1, making it difficult for readers to interpret the results. Moreover, the results reported in Table 3 are somewhat confusing. It is unclear whether higher or lower values are desirable for each metric, and the rationale for highlighting certain results in bold as the best outcomes is not well explained.

### Questions
The results presented in Table 3 appear unusual. For instance, the Forget Accuracy is generally expected to decrease; however, in the corrupted setting of CIFAR-10, the Teacher Ascent (TA) method achieves a Forget Accuracy of 1.00, higher than SCRUB+R and SSD, and it is labeled as the best result. Could the authors clarify this outcome and explain how such a result aligns with the expected behavior of unlearning methods?
In addition, in the whole-class setting of CIFAR-10, the Retain Accuracy values for all methods are reported as 1.00, yet only TA and SCRUB+R are highlighted in bold. The authors are encouraged to clarify the criteria used for emphasizing these results.

### Soundness
1

### Presentation
2

### Contribution
2
