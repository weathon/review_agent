# TP-Spikformer: Token Pruned Spiking Transformer

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Spiking neural networks (SNNs) offer an energy-efficient alternative to traditional neural networks due to their event-driven computing paradigm. However, recent advancements in spiking transformers have focused on improving accuracy with large-scale architectures, which require significant computational resources and limit deployment on resource-constrained devices. In this paper, we propose a simple yet effective token pruning method for spiking transformers, termed TP-Spikformer, that reduces storage and computational overhead while maintaining competitive performance. Specifically, we first introduce a heuristic spatiotemporal information-retaining criterion that comprehensively evaluates tokens' importance, assigning higher scores to informative tokens for retention and lower scores to uninformative ones for pruning. Based on this criterion, we propose an information-retaining token pruning framework that employs a block-level early stopping strategy for uninformative tokens, instead of removing them outright. This also helps preserve more information during token pruning. We demonstrate the effectiveness, efficiency and scalability of TP-Spikformer through extensive experiments across diverse architectures, including Spikformer, QKFormer and Spike-driven Transformer V1 and V3, and a range of tasks such as image classification, object detection, semantic segmentation and event-based object tracking. Particularly, TP-Spikformer performs well in a training-free manner. These results reveal its potential as an efficient and practical solution for deploying SNNs in real-world applications with limited computational resources.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents TP-Spikformer, a method to speed up Spiking Transformers by pruning unimportant tokens. Its main ideas are: 1) a new scoring rule called IRToP, which finds important tokens by checking if they look different from their neighbors (space) and if they change over time (time); and 2) a pruning architecture called IR-Arc, which makes unimportant tokens "skip" the heavy computation in a block instead of deleting them. This keeps the model's structure intact. Experiments show the method works well on many models and tasks, and impressively, it boosts speed with almost no accuracy loss even without any retraining.

### Strengths
1. The method is very versatile. It works on simple Spiking Transformers, complex ones with convolutions (like QKFormer), and on many different tasks like classification and detection. This shows it's a useful, general tool.
2. The biggest plus is its "training-free" ability. This is very practical for real-world use where retraining is too expensive.
3. The paper backs up its claims with strong experiments. The ablation studies clearly show why each part of the design is necessary, and the results on big datasets like ImageNet prove that the speedup is real.

### Weaknesses
1. The paper doesn't have a good, automatic way to decide how many tokens to prune in each layer. For some models, they just set the numbers by hand. For others, they run a slow and expensive "grid search" beforehand. The reported speedup doesn't include this search time, which makes the method seem faster than it is to set up.
2. The method prunes at the whole "block" level. A token either skips both the Attention and MLP parts, or does both. It might be better to prune them separately (e.g., prune more tokens for Attention and fewer for MLP) to get a better speed-vs-accuracy balance.
3. The paper only considers skipping tokens. Another popular technique is "token merging," where several similar tokens are combined into one. This might preserve more information than just letting old tokens pass through unchanged. The paper should have compared its method to token merging.
4. All experiments were on "directly trained" SNNs. It's unclear if the method would work on ANN-to-SNN conversion.

### Questions
1. How long does the grid search for finding the pruning ratios actually take? Have you thought about a simpler or automatic way to find these numbers?
2. Could you explain why you chose to skip tokens instead of merging them?
3. Do you think your method would work for SNNs that are converted from ANNs? For example, you can apply your method to "Masked Spiking Transformer".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposed an information retaining token pruning framework for spiking transformers.

### Strengths
1. This paper proposed an information retaining token pruning framework for spiking transformers. 

2. The writing in this paper is good.

3. The experiments are quite extensive, including classification, segmentation, detection, and tracking tasks.

### Weaknesses
1. My main concern lies in the motivation. Currently, directly trained spiking Transformers are relatively small- or medium-scale models. Although pruning slightly reduces performance while improving throughput（eg, TP-Spikformer with SDT-V1-8-768 on imagenet: -1.53% acc, thr 29%）， this trade-off does not constitute a strong motivation.

2、There is a lack of discussion on overall model training costs, such as training time and memory consumption.

3、There is a lack of spike-driven characteristics and energy consumption discussion on the heuristic spatiotemporal information-retaining criterion.

### Questions
see above

### Soundness
3

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
5

### Summary
This paper introduces TP-Spikformer, a simple and effective token pruning method designed to reduce the high computational and storage costs of large-scale spiking transformers, making them more suitable for resource-constrained devices. The core of the method consists of two main contributions. First, it proposes a heuristic Information-Retaining Token Pruning (IRToP) criterion, which scores and identifies important tokens by evaluating both their spatial distinctiveness from neighbors and their temporal variation across time steps . Second, it introduces the Information-Retaining Architecture (IR-Arc), which, instead of permanently dropping unimportant tokens, applies a block-level early stopping strategy—uninformative tokens bypass computation within a block and are then reassembled with the processed informative tokens. A significant advantage of this approach is its versatility and efficiency, as it requires no retraining or architectural modifications, achieving competitive performance even with zero fine-tuning. The authors demonstrate the method's effectiveness, efficiency, and scalability across diverse architectures (like Spikformer, QKFormer, and SDT) and a range of visual tasks, including classification, detection, segmentation, and tracking.

### Strengths
This paper presents a robust and highly impactful contribution, demonstrating exceptional strengths across originality, quality, clarity, and significance. The work's originality is outstanding, introducing a novel, training-free token pruning framework for spiking transformers, which stands in stark contrast to prior methods that require costly retraining and architectural modifications. The core ideas are creative and well-motivated: the bio-inspired IRToP criterion offers a new heuristic for identifying important tokens based on both spatial saliency and temporal dynamics, while the IR-Arc architecture cleverly uses a block-level early stopping and reassembly strategy instead of direct token removal, making it compatible with modern hierarchical SNNs. The research quality is superb, validated through extensive experiments across a diverse set of architectures (Spikformer, SDT-V1/V3, QKFormer) and a wide range of tasks, including classification, detection, segmentation, and tracking. The results consistently show significant reductions in computational overhead with minimal performance degradation, and thorough ablation studies rigorously justify the design choices. The paper is presented with excellent clarity, logically progressing from a well-defined problem to an elegant solution, with effective visualizations and a clear algorithm to aid understanding. Finally, the work is of high significance as it provides a simple, practical, and broadly applicable tool that addresses a critical bottleneck in deploying large-scale SNNs. By enabling efficient compression without retraining, TP-Spikformer dramatically lowers the barrier for applying spiking transformers in resource-constrained, real-world scenarios, marking a key step forward for the field of energy-efficient neuromorphic computing.

### Weaknesses
1. The paper would be strengthened by providing further experimental results, such as from an entropy or visualization perspective, to more rigorously justify the effectiveness of the Spatial and Temporal token scorers in measuring token importance.
2. The authors state that TP-Spikformer performs well even without training, but its accuracy drops significantly when using QKFormer and SDT-V3.

### Questions
1. Can the proposed method be applied to train a spiking transformer from scratch?
2. How could the proposed method be generalized to non-grid-like modalities?
3. How well does the proposed method preserve key information during pruning?What are the key differences between token pruning in spiking transformers and conventional ANN-based transformers?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a token pruning method for spiking transformers. First, a heuristic spatio-temporal information-retaining criterion is proposed to evaluate token importance. Second, based on the importance scores, a block-level early stopping strategy is proposed for uninformative tokens. Experiments are conducted on Spikformer, QKFormer and Spike-driven Transformer V1 and V3.

### Strengths
1. The narrative is very convincing that token-level sparsification is a very effective  way for transformer-style models to boost computational  efficiency.

2. This paper's experiments are very solid and extensive. The token sparsification method is verified across different tasks and multiple datasets, which makes the effectiveness very convincing.

3. This work demonstrates the most advanced token  pruning results for spiking ViT in comparison with previous spiking token  pruning methods.

### Weaknesses
The substantive contributions of this paper significantly overlap with previous work on ANN transformer sparsification, demonstrating limited novelty. It essentially replicates the success of existing ANN transformer sparsification approaches, with even the narrative framework bearing striking resemblances.

1. The spatial token scorer is highly similar to the ANN token-pruning one [1]. Thus, the novelty of this paper is significantly challenged.

2. The ablation study in Table 6 is conducted on ImageNet, a static dataset with no temporal disparity, thus I cannot confirm the effectiveness of the temporal pruning scorer. The visualization in Figure 5 does not entirely prove the effectiveness of the temporal pruning scorer. Instead, it exhibits limitations in spatial perception of the spatial scorer. 

3. It is very counterintuitive that the temporal scorer is used to prune spatial token rather than prune temporal token. Moreover, temporal pruning is applied regardless of whether the dataset is dynamic or static.

4. The retaining is already existent. This is also called token emerger technique. Or, retaining can be seen as a naive token emerger that can restore non-informative tokens for future use. I can easily give an example of such retaining [2].

In short, this paper shows severe duplication of ANN token-pruning domain. In essense, the success of this paper's method is inherited from the ANN  token-pruning  works. Such that this paper's contribution is limited. Moreover, the spatial token pruning may be even better if the most advanced ANN token pruning is applied.

This is more like a very good technique report concerning the verification of ANN methods applied to SNNs.

[1] Similarity-Aware Token Pruning: Your VLM but Faster

[2] HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers

### Questions
What's the main contributions from the methodology perspective?

Why not cite the ANN transformer sparsification papers which this paper's main techniques originate from? Similarity-based pruning has already existed, even the token-level sparsification has already been researched for years. But no citation is found in this paper.

### Soundness
3

### Presentation
3

### Contribution
1
