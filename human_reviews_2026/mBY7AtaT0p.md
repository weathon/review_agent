# HAM: Hierachical Adapters Merging for Scalable Continual Learning

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Continual Learning allows models to acquire knowledge incrementally, but is challenged by catastrophic forgetting, a phenomenon in which the learning new tasks disrupts previously acquired knowledge.
Although large pre-trained models can partially mitigate forgetting by leveraging their existing knowledge and over-parameterization, they often struggle when confronted with novel data distributions.
Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA, enable efficient adaptation to new data.
However, they still face challenges in scaling to dynamic learning scenarios and long sequences of tasks, as maintaining one adapter per task introduces complexity and increases the potential for interference.
In this paper, we introduce Hierarchical Adapters Merging (HAM), a novel framework that dynamically combines adapters from different tasks during training.
For each experience, HAM trains a low-rank adapter along with an importance scalar, then dynamically groups tasks based on adapter similarity.
Within each group, adapters are pruned, scaled and merged, facilitating transfer learning between related tasks.
Extensive experiments on three vision benchmarks demonstrate that HAM surpasses state-of-the-art methods, achieving up to 4\% accuracy improvement over the best baseline and nearly doubling efficiency in both training and inference, with particularly strong advantages as the number of tasks increases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces the Hierarchical Adapters Merging (HAM) framework, whose main goal is to mitigate forgetting in ViT models. The proposed method is sound, leveraging the idea of first training task-specific experts (adapters) for each task. Then, it groups the adapters via weight similarity to reduce computational cost. Finally, it merges all grouped adapters into a single module to alleviate the need for task identifiers in class-incremental learning settings. It shows promising results on several datasets compared to previous works.

### Strengths
1. Tackle the crucial problem of how to mitigate the forgetting when fine-tuning the transformer-based models.
2. The proposed method is sound and logical. Which uses the concept of MoE and weight similarity to merge the weight, to reduce the computational cost while keeping the critical knowledge for each task.
3. The method shows significant performance gain compared to previous works across 3 datasets.

### Weaknesses
1. The testing dataset is limited. In the era of foundation models, testing only on CIFAR/CUB/Tiny-ImageNet seems underwhelming. Please at least add a larger-scale dataset such as ImageNet. Also, it would be much better if the authors could test it on large foundation models such as CLIP and other more complicated tasks, which would show that the proposed method can actually work in real-world use cases.

2. This paper utilizes parameter isolation concept with subsequent weight merging. Since both MoE-like methods [1, 2, 3] and weight merging [4, 5, 6] are established concepts for continual learning, please include citations to these works and provide a concise discussion.

[1] Aljundi, Rahaf, Punarjay Chakravarty, and Tinne Tuytelaars. "Expert gate: Lifelong learning with a network of experts." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[2] Chen, Hung-Jen, et al. "Mitigating forgetting in online continual learning via instance-aware parameterization." Advances in Neural Information Processing Systems 33 (2020): 17466-17477.

[3] Yu, Jiazuo, et al. "Boosting continual learning of vision-language models via mixture-of-experts adapters." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[4] Mallya, Arun, and Svetlana Lazebnik. "Packnet: Adding multiple tasks to a single network by iterative pruning." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.

[5] Serra, Joan, et al. "Overcoming catastrophic forgetting with hard attention to the task." International conference on machine learning. PMLR, 2018.

[6] Hung, Ching-Yi, et al. "Compacting, picking and growing for unforgetting continual learning." Advances in neural information processing systems 32 (2019).

### Questions
1. The way of grouping adapter weight by just computing cosine similarity is surprisingly simple. Have the authors tried the representation-level similarity (e.g. CKA[7]) instead. It will be better if the authors can provide more qualitative result or analysis on does the grouping works as intended.
2. The concept of merging all trained weights into a single module to alleviate the need for task identifiers is sound. But what if I want to continue training after the merging happens? Will the performance degrade severely since the previously learned knowledge all merges into a single module, which will be brittle? The author should conduct such experiments. This is crucial because if we can’t keep training the model after it is merged, it will defeat the concept of lifelong learning.

[7] Kornblith, Simon, et al. "Similarity of neural network representations revisited." International conference on machine learning. PMlR, 2019.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HAM (Hierarchical Adapters Merging), a novel continual learning framework that dynamically groups and merges LoRA adapters during training to mitigate catastrophic forgetting and enhance knowledge transfer across tasks. HAM trains task-specific LoRA adapters with importance scalars, clusters similar adapters, prunes less significant weights, and merges them hierarchically. The method is evaluated on three vision benchmarks (CIFAR-100, CUB-200, Tiny-ImageNet) and demonstrates superior performance, especially on long task sequences, achieving up to 4% higher accuracy and nearly double the training/inference efficiency compared to strong baselines.

### Strengths
+ Provides a scalable solution for real-world continual learning, especially in long-task scenarios.
+ Propose an interesting dynamically grouping method.

### Weaknesses
+ The method is only evaluated in vision tasks; validation in NLP or multimodal settings is missing.
+ The method is not evaluated on commonly used dataset like imagenet-r in pre-trained model based CL. The used CIFAR, CUB, etc. are included in the pre-trained data.
+ During the inference time, existing lora-based cl method can be merged into pre-trained weights to achieve the same inference time as the pre-trained model. However, the proposed method may be not able to achieve this.

### Questions
See the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes novel methods based on PEFT for continual learning. While there exist a lot of branches of methods for continual learnimg, most of them suffers from lack of plasticity if the network remains same size. Due to this, network expansion should be the most promising way of solving continual learning. However, they lack scalability since network should grow linearly with the number of tasks. In that sense, PEFT naturally gained attention to make network expansion scalable. This paper try to propose new PEFT based extension method which is more efficient than pre-existing algorithms.

### Strengths
- Paper is well written and easy to follow. The authors well explained the motivation of each component and the existing related works. 
- The authors analyse and disassemble problem well and propose multiple components well aligned with these sub-problems.

### Weaknesses
- Although the group number is fixed, if they simply concat the pruned matrices, it is equivalent to concatenating pruned matrices over all tasks. So if this is right, pruning the matrix would be the only component in this method which lacks novelty. Also the authors should compare with this simple approach. 

- Backbone is pretrained on ImageNet while the benchmarks seems to be the subset of ImageNet. This can not be fair comparison then.

- Lack of experiments on large scale dataset. Tiny-Imagenet seems to be too small. A lot of existing literatures conducted experiments on at least ImageNet or ImageNet-R scale.

- Lack of comparision with enough baselines. Most of them looks outdated and also the performances are even marginally improved to those.

- Lack of demonstrations on experiments section. There are no demonstrations about how the authors split the task, select the baselines, and which refers to which paper.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Hierarchical Adapters Merging (HAM), a continual learning framework that enhances Low-Rank Adaptation (LoRA) by dynamically grouping and merging task adapters to prevent forgetting and improve knowledge transfer. By assigning importance weights and hierarchically combining related adapters, HAM achieves up to 4% higher accuracy, 9% less forgetting, and twice the efficiency of prior methods, offering a scalable solution for lifelong learning across many tasks.

### Strengths
1. The paper is well-structured and clearly written.

2. The study addresses an important problem in continual learning using low-rank adaptation.

### Weaknesses
1. The primary concern lies in the novelty of the proposed approach. The idea of Hierarchical Adapters Merging for LoRA appears similar to prior work, such as *TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree*. The authors should clarify the distinctions and emphasize the novel contributions of their method.

2. The paper overlooks a significant body of research on continual learning (or continual fine-tuning) in large language models, including *O-LoRA*, *LoRA-MoE*, and *TreeLoRA*. A discussion comparing this work with these studies would strengthen the paper’s context and positioning.

3. The experimental evaluation is primarily based on synthetic image datasets. It is recommended to include additional experiments on large language models (LLMs) to demonstrate the broader applicability and effectiveness of the proposed method.

References:

[O-LoRA] *Orthogonal Subspace Learning for Language Model Continual Learning*

*LoRA-MoE: Alleviating World Knowledge Forgetting in Large Language Models via MoE-Style Plugin*

*TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree*

### Questions
See Weaknesses above

### Soundness
2

### Presentation
3

### Contribution
2
