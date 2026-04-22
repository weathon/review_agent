# Task-Focused Consolidation with Spaced Recall: Making Neural Networks Learn like College Students

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 4

## Abstract
Deep neural networks often suffer from a critical limitation known as catastrophic forgetting, where performance on past tasks degrades after learning new ones. This paper introduces a novel continual learning approach inspired by human learning strategies like Active Recall, Deliberate Practice, and Spaced Repetition, named Task-Focused Consolidation with Spaced Recall (TFC-SR). TFC-SR enhances the standard experience replay framework with a mechanism we term the Active Recall Probe. It is a periodic, task-aware evaluation of the model’s memory that stabilizes the representations of past knowledge. We test TFC-SR on the Split MNIST and the Split CIFAR-100 benchmarks against leading regularization-based and replay-based baselines. Our results show that TFC-SR performs significantly better than these methods. For instance, on the Split CIFAR-100, it achieves a final accuracy of 13.17% compared to Standard Experience Replay’s 7.40%. We demonstrate that this advantage comes from the stabilizing effect of the probe itself, and not from the difference in replay volume. Additionally, we analyze the trade-off between memory size and performance and show that while TFC-SR performs better in memory-constrained environments, higher replay volume is still more effective when available memory is abundant. We conclude that TFC-SR is a robust and efficient approach, highlighting the importance of integrating active memory retrieval mechanisms into continual learning systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Continual Learning involves training a model on a sequence of tasks, with the main goal of learning new tasks without forgetting previously learned ones. This paper presents an approach inspired by 3 human learning strategies. (1) Active Recall (which retrieves previous information), (2) Deliberate Practice (learn until proficiency is achieved), and (3) Spaced Repetition (add space between retrieving information). The paper evaluates the proposed method on 2 standard benchmarks (Split MNIST and CIFAR100) and compares it with classical methods (EWC, SI, and ER). The paper also presents ablation experiments on the size of the memory buffer.

### Strengths
- The authors motivate the idea of incorporating brain-inspired strategies into the training of Deep Learning methods. One of the challenges of current Deep Learning models is their limited ability to accumulate knowledge, something that happens naturally in humans. Human learning seems a natural source of inspiration for proposing new approaches to tackle this challenge.
- Section 4 (Discussion) provides a clear explanation of the results and their comparison with other methods. Showing the conclusions together with explanations for their decisions.

### Weaknesses
- There is an evident lack of understanding of previous methods in the area of Continual Learning. For example, the paper's "Mixed-Batch" strategy, which it calls novel, is a common practice in the area. 
    - Chrysakis, Aristotelis, and Marie-Francine Moens. "Online continual learning from imbalanced data." International Conference on Machine Learning. PMLR, 2020.
- It is unclear how and when the Adaptive Active Recall is used during training. Algorithm 1 shows only how it is used during inference, so it only affects how accuracy is calculated. 
    - There is no ablation on how this can affect the final results of the experiments.
- The experiments section lacks more complex scenarios. MNIST and CIFAR100 are standard in the area but are mostly outdated due to their simplicity. More complex scenarios are needed, such as Tiny ImageNet, CORE50, or others.
- The only replay-based method is Standard Replay, which has some weird results in Figure 3. Why does it decrease when using 1000 samples? Shouldn't it show an increase in performance?
- More and better baselines are needed. Multiple methods use memory to tackle the challenge of Catastrophic Forgetting. As mentioned in the paper, iCarl is one of them, but others, such as DER, can also be used for comparison.
    - Buzzega, Pietro, et al. "Dark experience for general continual learning: a strong, simple baseline." Advances in neural information processing systems 33 (2020): 15920-15930.
    - Depending on how the authors describe the method, comparing it with other memory-based methods is not adequate, because the proposed method can complement any such method. Similar to this idea are storage policy methods, for example:
        - Hurtado, Julio, et al. "Memory population in continual learning via outlier elimination." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Questions
- Did you use previous implementations or frameworks to run experiments of the previous methods, or did you implement them from scratch? 
    - This question concerns results from previous methods, which are lower than those reported in other papers.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a Continual Learning approach inspired by how a college student learns. In a nutshell, the method emphasizes the repetition of past concepts and the careful selection of intervals between these repetitions. In practice, it builds upon Experience Replay (a well-established baseline in the field) by introducing a scheduling mechanism that determines when examples from the memory buffer are replayed. The authors evaluate their method on Split MNIST and Split CIFAR-100, reporting mixed results.

### Strengths
Standard rehearsal approaches typically interleave batches of examples from the current task with batches retrieved from the memory buffer. In contrast, this work proposes replacing this regular and constant scheduling with an adaptive mechanism, where rehearsal is triggered only when accuracy on past tasks decreases. This is a sound and promising intuition: such an adaptive strategy could reduce unnecessary computation (e.g., memory retrieval and additional forward–backward passes) while mitigating the risk of overfitting to the limited set of samples stored in the buffer, a well-known issue in rehearsal-based methods.

### Weaknesses
The paper has countless problems in its current form. The impression is that the paper is an exercise for a student to practice submitting to top conferences. In general, the writing is good, but it seems that an LLM did much of the work, as also corroborated by the statement placed by the authors on the last page. These are some of the major points I would like to point out:
- The motivation is unclear. The authors make a proposal, which has some intuitive explanation inspired by how a student learns, but it is not clear how this proposal can improve upon well-established approaches.
- The experimental section is really poor. The authors perform experiments on Split MNIST and CIFAR-100, with not very good results. Many competitors are missing (e.g., DER++, iCaRL, and so on), and the results on CIFAR-100 (around 10–15% accuracy) are not consistent with previous papers or with common sense.
- The experimental comparison with Experience Replay should be clearer and more thorough. Since the major contribution of this paper is the scheduling approach, it must be shown clearly that this modification leads to substantial improvements across multiple settings. Actually, this is hard to conclude when looking at the tables.

### Questions
No questions.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a continual learning method, Task-Focused Consolidation with Spaced Recall (TFC-SR), inspired by human learning strategies such as Active Recall and Spaced Repetition. The proposed method extends the standard experience replay (ER) framework with a novel mechanism called the "Active Recall Probe." This probe consists of a periodic, no-gradient forward pass on a batch of replay samples to evaluate the model's memory of past tasks. Based on whether the performance on this probe exceeds a predefined "mastery threshold," an adaptive schedule increases or decreases the interval until the next probe. The method is evaluated on Split MNIST and Split CIFAR-100. On Split MNIST, TFC-SR's performance is comparable to standard ER. On Split CIFAR-100 with a buffer of 1000 samples, TFC-SR significantly outperforms ER (13.17% vs. 7.40%). However, an ablation study reveals that this advantage is specific to memory-constrained settings and disappears when a larger buffer is used, where standard ER becomes superior.

### Strengths
1. Intuitive and Simple Concept: The motivation drawn from human cognitive science—specifically Active Recall and Spaced Repetition—provides a compelling and intuitive narrative for the method.
2. Strong Performance in a Specific Regime: The paper demonstrates a significant performance advantage for TFC-SR in a memory-constrained setting on a challenging benchmark. 
3. Insightful Ablation and Discussion: The ablation study on buffer capacity (Section 3.4) is a highlight of the paper.

### Weaknesses
1. Major Disconnect Between Proposed Mechanism and Likely Cause: The paper's central narrative is built around the cognitive science concepts of "Active Recall" and "Spaced Repetition." However, the authors' own analysis in Section 4 strongly suggests that the observed performance gain on Split CIFAR-100 is an architectural artifact related to Batch Normalization.
2. Inconsistent and Underwhelming Empirical Results: The experimental results are mixed and do not support a general claim of the method's superiority.
3. Limited Algorithmic Novelty: The core algorithm is a very incremental1 modification of experience replay.

### Questions
1. Your discussion section compellingly argues that the performance gain on Split CIFAR-100 is likely due to the stabilization of Batch Normalization statistics by the probe's forward passes. To definitively test this hypothesis, have you considered replacing the BN layers in the ResNet-18 model with an alternative normalization scheme that does not use running statistics, such as Group Normalization or Layer Normalization?
2. The results demonstrate that TFC-SR's advantage is highly sensitive to the replay buffer size, with standard ER being superior on Split MNIST and also on Split CIFAR-100 with a large buffer

### Soundness
3

### Presentation
2

### Contribution
2
