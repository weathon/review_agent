# CAFÉ: Coverage-Aware Self-Distillation to Mitigate Forgetting in Deep Networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Deep neural networks rarely exhibit global overfitting in the classical sense, yet they often suffer from a less visible problem - forgetting of previously learned patterns. This phenomenon, which was termed local overfitting, degrades performance in specific regions of the input space even as overall accuracy improves. To address this problem, we propose CAFÉ (Coverage-Aware Forgetting Elimination) - an online, validation-aware, single model method, which mitigates forgetting during training while exploiting self-distillation. CAFÉ identifies and prioritizes checkpoints that uniquely recover forgotten validation samples, dynamically weighting their contributions to form evolving soft labels for each epoch of training. Our experiments show that CAFÉ consistently outperforms both standard training and recent self-distillation SOTA methods under clean and noisy labels, across CIFAR-100 and TinyImageNet, with and without data augmentation. Beyond raw accuracy gains, our results provide quantitative evidence of the substantial impact of forgetting on deep learning performance, and demonstrate that targeted mitigation yields measurable robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CAFÉ (Coverage-Aware Forgetting
Elimination), a method aimed at reducing local forgetting in deep network training. The approach constructs a teacher distribution by weighting past checkpoints according to their *marginal validation coverage*, thus emphasizing models that uniquely contribute to correctly predicted validation samples. This self-distillation mechanism helps preserve useful past knowledge without ensemble inference or additional model parameters. Experiments on CIFAR-100, CIFAR-100N, and TinyImageNet show consistent gains over strong baselines such as SAT and early stopping, and an efficient variant (Light CAFÉ) maintains similar performance with much lower storage cost.

### Strengths
1. The proposed CAFÉ framework is simple, effective, and model-agnostic, requiring no architectural changes or additional inference time cost.
2. The paper addresses an important problem (i.e., local forgetting during standard supervised training) with clear motivation and empirical evidence.

### Weaknesses
1. The method requires storing multiple checkpoints during training (even if Light CAFÉ mitigates this), which could be burdensome for large-scale models.
2. The experiments are limited to CNN architectures (ResNet, DenseNet). It remains unclear whether the observed local forgetting phenomenon also occurs in Transformer-based models such as ViT or Swin Transformers, which are now dominant in vision tasks. If such forgetting exists, would the proposed CAFÉ still be effective, or would it require adaptation to the Transformer training dynamics?
3. The analysis of “local forgetting” is primarily demonstrated on CIFAR-100 (and its noisy variant). It is unclear whether the same phenomenon appears on large-scale datasets such as ImageNet, where the data distribution is more diverse and the training is typically longer and more stable. Is local forgetting a general property of deep network training, or is it amplified by small datasets and limited data diversity? Moreover, would CAFÉ still provide benefits when the model already achieves high coverage on large datasets?
4. The paper lacks a direct comparison to other forgetting mitigation techniques in continual learning outside self-distillation (e.g., rehearsal-based or regularization-based methods).

### Questions
Please see the Weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes CAFÉ,  a self-distillation method that mitigates local forgetting in deep neural networks. CAFÉ weights previous training checkpoints based on their marginal validation coverage, a metric that measures how many validation examples a checkpoint gets right that others miss. With respect to other methods that solve the problem by using an ensemble of previous checkpoint, at the end Cafè is able to produce a single model. This coverage-weighted knowledge is distilled into the current model, forming adaptive soft targets that preserve useful information over time. Experiments are conducted on CIFAR-100 and TinyImageNet where the proposed method is compared against self-distillation and ensemble methods, under both clean and noisy labels.

### Strengths
- The idea of using previous checkpoints during training is interesting, as it allows the method, unlike other approaches that rely on checkpoint ensembles, to produce a single model at the end of training.
- The paper also proposes a lightweight version that reduces storage requirements by limiting the number of checkpoints, without compromising performance.
- As shown in Fig. 2, the method effectively mitigates the problem of local forgetting during training, whereas other methods address it only post hoc

### Weaknesses
- The results are somewhat incremental with respect to KF in Tables 1 and 2, and with respect to PS-KD in Table 3.
- Style of the paper: The paper lacks clarity in the presentation of results. Table 3 shows several rows missing. Figure 4 was obtained by overlaying results on top of an image taken from another paper (as clearly stated in lines 378–381). The last row of Tables 1 and 2 is misleading, as it initially appears to show improvement over the state of the art, while it actually compares against ERM + early stopping. Line 79 includes the phrase “see review below” without directly referencing the relevant section.
- Experiments using more recent vision backbones (e.g., ViT) are missing. see questions.
- Figure 5b is unclear and difficult to interpret. Please clarify the main message in the caption, even if it is already discussed in the main text.
- The title can be misleading as usually forgetting refers to forgetting of previous knowledge in the Continual Learning setting. I would clarify that.

### Questions
- What is the main difference between CAFE and FK? As far as I understand, FK also uses a similar strategy involving previous checkpoints. The main difference (and advantage) of CAFE is that the proposed method does not require an ensemble during inference.
- Since Table 3 shows that PS-KD achieves results similar to CAFE, it would be useful to highlight the main differences between the two methods.
- In the KF paper, experiments were also conducted using more recent visual backbones such as ViT. Why are these missing from the proposed work?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the problem of local overfitting in deep learning, where previously learned patterns are forgot within training. The proposed method keeps all previous checkpoints and aggregates their predictions to build pseudo labels. Experimental results show the effectiveness of the proposed methods on small image datasets.

### Strengths
+ Addressing local overfitting in deep learning is an interesting research topic.

+ The proposed method is easily understandable.

+ The performance gain is consistent throughout experiments.

### Weaknesses
- No ablation study on the design choices, including hyperparameter tuning. For example, is the schedule for beta in L254 optimal?

- As the proposed method requires to keep all previous checkpoints and runs them to get their outputs, an analysis on the computational cost compared with baseline methods is required.

- In Figure 5, why the marginal coverage of Vanilla peaks in the middle? Following the idea in STEP 2: MARGINAL COVERAGE SWEEP, the best performing model should be chosen at first, which usually appear around the end of training. If not, then it implies that the learning rate schedule is simply suboptimal, and could be better by hyperparameter tuning. In other words, the comparison might not be fair, as the optimization of Vanilla appears to be not properly done.

- Citation format is problematic in some places, e.g., L329.

- Accuracy is not sufficient to catch the degree of forgetting.

### Questions
Please address concerns in Weaknesses above.

### Soundness
3

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
4

### Summary
This paper introduces CAFÉ (Coverage-Aware Forgetting Elimination), an online, validation-aware self-distillation method to mitigate local overfitting and forgetting in deep neural networks. CAFÉ dynamically identifies checkpoints with unique validation coverage and forms soft teacher targets for subsequent training epochs based on their marginal contributions. Extensive experiments demonstrate CAFÉ’s robustness to clean and noisy labels, outperforming standard ERM, prior self-distillation baselines, and specialized ensembles (such as Knowledge Fusion) on CIFAR-100, TinyImageNet, and CIFAR-100N. The work also includes in-depth ablations and theoretical analysis of its claims and complexity.

### Strengths
1. Interesting approach to handle forgetting with self distillation from previous checkpoints based on validation accuracy.
2. Relatively easy to understand approach.
3. Good performance improvement on the CIFAR datasets, especially in high noise cases.
4. Significant improvements shown in Table 3 for clean data.

### Weaknesses
1. Reliance on a validation set that is representative of the test set. In cases where the test set may be significantly different, this approach will not be effective.
2. Very low improvement on TinyImageNet for low (symmetric) or zero noise cases. (Table 2).
3. Very few compared methods in Table 1 and 2. Reduces the confidence in the overall effectiveness of the approach.
4. Could not find any forgetting measure or metric.
5. Since the objective is dealing with the "forgetting of previously learned patterns.", wont incremental learning experiments be a better judge of how good the approach is in dealing with forgetting.

### Questions
1. Discuss the reliance on a validation set when it is not fully similar to the test set
2. Why is the improvement very low on TinyImageNet for low (symmetric) or zero noise cases. (Table 2).
3. Are there no more recent papers that can be compared with in Table 1 and 2?
4. Could not find any forgetting measure or metric.
5. Wont incremental learning experiments be a better judge of how good the approach is in dealing with forgetting?

### Soundness
2

### Presentation
2

### Contribution
2
