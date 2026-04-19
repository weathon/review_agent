# Visual Prompting Upgrades Neural Network Sparsification: A Data-Model Perspective

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
The rapid development of large-scale deep learning models questions the affordability of hardware platforms, which necessitates the pruning to reduce their computational and memory footprints. Sparse neural networks as the product, have demonstrated numerous favorable benefits like low complexity, undamaged generalization, $\textit{etc}$. Most of the prominent pruning strategies are invented from a $\textit{model-centric}$ perspective, focusing on searching and preserving crucial weights by analyzing network topologies. However, the role of data and its interplay with model-centric pruning has remained relatively unexplored. In this research, we introduce a novel $\textit{data-model co-design}$ perspective: to promote superior weight sparsity by learning important model topology and adequate input data in a synergetic manner. Specifically, customized $\textbf{V}$isual $\textbf{P}$rompts are mounted to upgrade neural $\textbf{N}$etwork $\textbf{s}$parsification in our proposed $\textbf{\texttt{VPNs}}$ framework. As a pioneering effort, this paper conducts systematic investigations about the impact of different visual prompts on model pruning and suggests an effective joint optimization approach. Extensive experiments with $3$ network architectures and $8$ datasets evidence the substantial performance improvements from $\textbf{\texttt{VPNs}}$ over existing start-of-the-art pruning algorithms. Furthermore, we find that subnetworks discovered by $\textbf{\texttt{VPNs}}$ from pre-trained models enjoy better transferability across diverse downstream scenarios. These insights shed light on new promising possibilities of data-model co-designs for vision model sparsification. Codes are in the supplement.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
To upgrade vision model sparsification, the paper proposed a data-model co-design sparsification paradigm, where integrating input image with the learnable perturbation, and a network tuning strategy is designed to optimize this issue.

The algorithm has demonstrated excellent performance on CIFAR-10 and CIFAR-100 datasets.

### Strengths
1. The manuscript exhibits a commendable level of writing proficiency, featuring well-crafted graphics that enhance the overall presentation and a compelling narrative.

2. The algorithm has showcased remarkable efficacy when applied to the CIFAR-10 and CIFAR-100 datasets, achieving good performance.

### Weaknesses
1. **Inadequate experiments.** This is a primary concern for the reviewer. The paper only presents experiments on CIFAR-10 and CIFAR-100, which, in the era of big data, are considered insufficient. These experiments do not adequately demonstrate the performance of the proposed method. Conducting experiments on larger datasets, such as ImageNet-1k, is essential.

Additionally, it's worth noting that the method utilizes ImageNet-1K pre-trained weights, which were trained on a resolution of 224. However, the method is tested on CIFAR data with a resolution of 32. It is evident that padding the data to a resolution of 224 can significantly boost performance. From this perspective, experiments specifically conducted on ImageNet with a resolution of 224, and direct performance comparisons with fine-tuning on this resolution, are crucial.

Figure 10 further substantiates this conclusion, showing that the optimal performance is achieved at a resolution of 224, with diminishing performance as the resolution decreases. Therefore, padding CIFAR data with a resolution of 32 to 224 doesn't necessarily demonstrate the superiority of the method. The gains observed in this case could be attributed to the ImageNet-1K pre-trained weights at a resolution of 224.

2. The reviewer also suggests providing performance comparisons with smaller models, as performance metrics on sparser models, such as using MobileNet, would be more indicative and informative.


3. Furthermore, the method exhibits significant limitations, as it necessitates the use of pre-trained weights from a larger dataset. The current version seems to require transforming the ImageNet model to CIFAR. It would be insightful to explore the performance without pre-trained weights. Additionally, conducting comparisons on a larger pre-trained dataset, such as ImageNet, appears necessary for the current version.

### Questions
In each figure, the authors have plotted curves labeled as "our best," which may not be entirely necessary as this information can be inferred from the VPNs curve. 

Moreover, this plotting style has the potential to cause confusion; upon initial review, it might be perplexing why the curve representing "ours" appears as a straight line.

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a joint model pruning and visual prompting learning method. By combining these two methods, it could recover the performance loss by pruning. This method is validated on several pruning methods and datasets to demonstate its universality.

### Strengths
The idea is easy to follow and effective.It is interesting to see that only a small number of learned parameters could improve the performance.

### Weaknesses
1. I think the authors should focus more on the structured sparse case, unstructured pruning is well known that will not contribute to any acceleration in practice.
2. For structured pruning, the speedup ratio should use latency, not theoritical FLOPs. And more recent methods should be compared.
3. The visual prompting method essentially uses lower resolution for the input images. It is necessary to compare a baseline that using a lower resolution image as input, and then prune less parameters to maintain the same FLOPs as the proposed VPN. 
4. Following the previous point, I also wonder whether this method will deteriorate some applications that are senstive to resolution, such as object detection or etc.

### Questions
See above.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to use visual prompts to improve performance of the pruned model by applying visual prompts earlier in the process, aka, before the model fine-tuning. This effort was motivated by the experiments of applying post-pruning prompt to the sparse models with and without fine-tuning. As post-pruning prompts showed only marginal gains to subnets that went through fine-tuning, authors proposed to apply the visual prompts earlier in the process. The proposed scheme was compared with eight pruning baselines on eight classification tasks. Numerical comparisons show the potential of the proposed scheme.

### Strengths
- The idea of applying visual prompts to identify a subnet which further leads to a better pruning results is interesting.
- The idea of using visual prompts to control a pretrained vision model is an interesting direction to pursue.

### Weaknesses
- The paper is not easy to read. There's a particular emphasis on "data model co-design", but it takes quite a while to understand what this refers to concretely.
- Why is the visual prompts essential? How about learning additional parameters without using the visual prompt?
- How would visual prompts be different from data augmentation?

### Questions
Please see my questions in the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new post-pruning method by introducing visual prompts into the pruning pipeline. The authors introduce visual prompts into trained vision models. During pruning, not only weight masks but also visual prompts are optimized. The whole pipeline includes two steps: (1) fixing pretrained weights of vision models, tuning masks and visual prompts; (2) fixing the mask, fine-tuning both weights and visual prompts. The authors adopt the proposed method on several downstream classification tasks and compare the performances of pruned models with other pruning methods. The experiments show that the proposed method can achieve better performance than other methods with the same sparsity.

### Strengths
1. The writing and presentation of this paper are quite good. The logic of the article is very clear, and the choice of words in the writing is also very precise.
2. The idea of introducing tunable visual prompts into the pruning pipeline is intriguing. The experiments validate the effectiveness of the proposed strategy.
3. The authors compare the proposed method with other pruning methods and additionally apply the proposed visual prompt pruning strategy to these methods, demonstrating the transferability of their approach.

### Weaknesses
1. Introducing visual prompts into vision models seems to boost their performance. However, comparing models with visual prompts (the proposed method) to those without (other baseline methods) might not be entirely fair. What if we apply both the proposed and baseline methods to a model that has already been fine-tuned with visual prompts?
2. I have some doubts regarding the generality and performance of this paper.
(1) Why must we conduct experiments on downstream tasks of ImageNet? Why not directly on ImageNet itself, since most pruning work actually focuses more on performance on ImageNet?
(2) The method proposed in this paper seems to be limited to scenarios where visual prompts can be applied, with their main application currently being in classification tasks. How can the proposed approach be used for other tasks, such as detection, segmentation, etc.?

### Questions
Please refer to the weaknesses. I hope the authors can provide more experiments to demonstrate the effectiveness of the method.

The provided additional experiments have addressed my concerns about the performance of VPN in ImageNet and object detection tasks. However, after reading other reviewers' comments, I consider decreasing my score to 5. The reasons are as follows:
(1) The author describe the advantagement of unstructured sparsity on nonGPU hardware. However, they do not report the latency of unstructured sparsity on CPU. I think conducting experiments on some lightweight networks and reporting latency tested on CPUs can make the results more convincing.
(2) In regards to structured pruning, the experiments carried out on the CIFAR-100 dataset are insufficient.

Overall, I believe this paper needs more meticulous refinement in its experiments. If it can validate its approach on large-scale datasets for both unstructured and structured pruning settings, it would then be a very solid paper.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
