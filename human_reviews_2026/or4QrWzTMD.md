# UWYN (Use Only What You Need): Efficient Inferencing in 2D and 3D Vision Transformers

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 2, 8

## Abstract
Efficient computation for Vision Transformers (ViTs) is critical for latency-sensitive applications. However, early-exit schemes rely on auxiliary controllers that introduce non-trivial overhead. We propose UWYN, an end-to-end framework for image classification and shape classification tasks that embeds exit decisions directly within the transformer by reusing the classification head at each residual block. UWYN first partitions inputs via a lightweight feature-threshold into “simple” and “complex” samples: simple samples are routed to a shallow ResNet branch, while complex samples traverse the ViT and terminate as soon as their per-block confidence exceeds a preset confidence level. During the ViT pass,UWYN also dynamically prunes redundant patch embeddings and attention heads to further cut computation. We implement and evaluate this strategy on both 2D (ImageNet, CIFAR-10, CIFAR-100, SVHN, BloodMNIST) and 3D (ModelNet-40, Scan Object NN) benchmarks. UWYN reduces Multiply-Accumulate operations (MACs) by over 75% compared to SOTA models, e.g., LGViT [ACM MM ’23] achieving 83.29% accuracy on CIFAR-100 and 84.39% on ImageNet. We also show faster inference with minimal accuracy loss.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents the UWYN (use only what you need) method for efficient inference in ViTs. This algorithm increases efficiency by, among other things, utilizing an early-exit policy for classification, routing easy-to-classify samples prior to sending them through the full model, and pruning attention heads and patches which contain redundant information. The early exit policy has little parameter/compute overhead as it utilizes the existing classification head at each residual block. In both 2D and 3D datasets, this method exhibits minimal accuracy loss with significant increases in inference speed.

### Strengths
The strengths of this paper include the introduction of a method that requires very little training to significantly increase the efficiency of ViTs through routing, early exit and patch/head pruning. The method takes advantage of the redundancy that exists in these models to compress and extract information more efficiently. Finally, the method is comprehensibly tested through several benchmarks showing significant improvement for model efficiency without substantially decreased performance. It seems that the combination of all these elements is a novel contribution to this area of research. I also think the routing pre-ViT is a clever idea.

### Weaknesses
I think there are a lot of strengths in the paper. That being said, I was very confused how the head/patch pruning model was trained. Specifically, how do you handle the nondifferentiality of subselecting a subset of heads/tokens to use? Additionally, the overall architecture of the model seemed somewhat vague. For example, in the ViT, how were the FFNs specified?

### Questions
Please describe how the head/patch pruner is trained.

Please give a more detailed description of the architecture for the models used in the evaluations in section 4.

Out of curiosity, what would happen to the performance if you had a strict budget instead of a threshold for heads/pruned patches (e.g. only utilize the top k heads/patches per layer)? I suppose this would be more difficult to train. 

How was the pre-ViT router developed? do you think would it be better to train this for each individual model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a method that incorporates the both a selective inference strategy with a parallel, hybrid architecture with ViT/CNN, and a block-wise early-stop and attention pruning strategy for ViT. The architecture can be used for 2D/3D classification, and achieves better results than its selected baselines.

### Strengths
- The hybrid dual branch + head pruning method is quite intuitive and easy to understand
- The experiment results are competitive against its baselines

### Weaknesses
- The novelty of the method is very limited, as both main pruning techniques involved in the framework (hybrid architecture/dual branch, early stopping/head pruning) are not original
- Vast majority of the baseline methods compared in the experiments are from 2022-23 era. This is a fast-moving research area and newer baselines are lacking.
- Details are lacking for how UWYN implements the object detection task.
- Minor issues  - line 336 PartialFormer (?)

### Questions
- How dod you come up with the #param number for UWYN? ResNet50 alone has 23.9M params yet the paper reports 22.86m
- In reality, I assume you'll have to load both branches in the memory for inference. Is there an evaluation on how much memory overhead this dual-branch design would bring?
- Can UWYN handle dense prediction tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the challenge of dynamically allocating computational resources in efficient ViT based on input complexity. Specifically, simple inputs are processed by a shallow ResNet for efficiency, while complex inputs are routed to a ViT equipped with an early-exit mechanism for enhanced feature extraction. In the ViT branch, a classifier is appended to each block to determine whether the confidence of the predicted class exceeds a predefined threshold, triggering an early stop if satisfied. Experimental results demonstrate the effectiveness of the proposed framework.

### Strengths
1. The idea of dual-branch architectural design for inputs of different complexity is interesting. And the input complexity analysis via Sobel convolution is efficient.

2. The method is introduced in detail, enhancing its reproducibility.

3. The efficiency comparisons are conducted on edge devices.

### Weaknesses
1. Messy format. Please use "[]" for in-line references rather than "()" to distinguish different usages. There are too many brackets in the manucsript.

2. Some state-of-the-art efficient ViT methods are missing, such as [1,2,3].

[1] Vasu, Pavan Kumar Anasosalu, et al. "Fastvit: A fast hybrid vision transformer using structural reparameterization." ICCV, 2023.

[2] Hatamizadeh, Ali, et al. "Fastervit: Fast vision transformers with hierarchical attention." ICLR, 2024.

[3] Wang, Ao, et al. "Repvit: Revisiting mobile cnn from vit perspective." CVPR, 2024.

### Questions
Refer to the weaknesses

### Soundness
4

### Presentation
4

### Contribution
4
