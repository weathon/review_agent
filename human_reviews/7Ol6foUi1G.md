# Data-independent Module-aware Pruning for Hierarchical Vision Transformers

- Decision: Accept (poster)
- Scores: 8, 6, 6

## Abstract
Hierarchical vision transformers (ViTs) have two advantages over conventional ViTs. First, hierarchical ViTs achieve linear computational complexity with respect to image size by local self-attention. Second, hierarchical ViTs create hierarchical feature maps by merging image patches in deeper layers for dense prediction. However, existing pruning methods ignore the unique properties of hierarchical ViTs and use the magnitude value as the weight importance. This approach leads to two main drawbacks. First, the "local" attention weights are compared at a "global" level, which may cause some "locally" important weights to be pruned due to their relatively small magnitude "globally". The second issue with magnitude pruning is that it fails to consider the distinct weight distributions of the network, which are essential for extracting coarse to fine-grained features at various hierarchical levels. 

To solve the aforementioned issues, we have developed a Data-independent Module-Aware Pruning method (DIMAP) to compress hierarchical ViTs. To ensure that "local" attention weights at different hierarchical levels are compared fairly in terms of their contribution, we treat them as a **module** and examine their contribution by analyzing their information distortion. Furthermore, we introduce a novel weight metric that is solely based on weights and does not require input images, thereby eliminating the **dependence** on the patch merging process. Our method validates its usefulness and strengths on Swin Transformers of different sizes on ImageNet-1k classification. Notably, the top-5 accuracy drop is only 0.07% when we remove 52.5% FLOPs and 52.7% parameters of Swin-B. When we reduce 33.2% FLOPs and 33.2% parameters of Swin-S, we can even achieve a 0.8% higher relative top-5 accuracy than the original model. Code is available at: [https://github.com/he-y/Data-independent-Module-Aware-Pruning](https://github.com/he-y/Data-independent-Module-Aware-Pruning).

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces the Data-independent Module-Aware Pruning (DIMAP) for hierarchical vision transformers (ViTs). Instead of relying on magnitude values like traditional pruning techniques, DIMAP evaluates "local" attention weights in hierarchical ViTs by their impact on information distortion. By examining weight distribution at different hierarchical levels, DIMAP offers an alternative to magnitude-based methods. Tests on Swin Transformers highlight its efficiency, with a small decrease in top-5 accuracy and notable reductions in FLOPs and parameters.

### Strengths
1. The paper presents DIMAP, a unique pruning method for hierarchical ViTs. Instead of the prevalent focus on weight magnitudes, DIMAP evaluates attention weights based on their effect on information distortion, setting it apart from standard methods.

2. The approach is grounded in solid theory, with mathematical models reflecting a comprehensive grasp of the issue. The choice to use distortion analysis as a measure of weight significance stands out.

3. The paper's organization is logical, detailing the obstacles and answers related to pruning hierarchical ViTs. Its equations, coupled with their explanations, offer transparency and ease of understanding for readers.

4. Given the rising importance of hierarchical ViTs in computer vision, the paper's relevance is significant. Efficient pruning is vital for these transformers in limited-resource scenarios. DIMAP, focusing on maintaining crucial attributes while cutting down computations, meets a current community demand.

### Weaknesses
Complexity: While the module-aware method proves efficient, it could complicate the pruning process, particularly for extensive models or vast datasets. How much computational burden does this additional step bring?

The threshold determination is at the module level. Could this be determined at the model level instead? What differentiates these two levels of thresholding?

### Questions
The computational cost and threshold explained in weakness section.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Traditional pruning methods ignore the unique properties of hierarchical ViTs by utilizing magnitude as the weight importance, which leads to issues in weight pruning at both local and global scales. DIMAP addresses these challenges by treating attention weights at different hierarchical levels as a "module" and examining their significance by analyzing their information distortion. The proposed method demonstrates superior performance on Swin Transformers when compared with other approaches, especially in terms of accuracy and efficiency in the pruning process.

### Strengths
1. The idea is somewhat novel. The unique properties of hierarchical ViTs leads to the proposing of a module-aware approach for pruning hierarchical ViTs.
2. The paper is well-structured, with clear explanations of the challenges with existing pruning methods, the introduction of their novel method, and mathematical derivations supporting their approach.
3. The experiments are good. The top-5 accuracy drop is only 0.07% when we removing 52.5% FLOPs and 52.7% parameters of Swin-B.

### Weaknesses
1. Why it's called "data independent" since the weights are updated with training data?
2. A comparative study with existing pruning methods, including their weaknesses when applied to hierarchical ViTs, could offer more insights into the significant advantages of DIMAP.
3. More information on the implementation specifics, hyperparameters used, and other nuances would be helpful for reproducibility and understanding the intricacies of DIMAP.
4. It's not clear that uniform pruning means pruning parameters uniformly or pruning FLOPs uniformly?
5. While the method has been validated on the Swin Transformer architecture on ImageNet-1k, it would be beneficial to see its performance on other Vision Transformer architectures or datasets to establish its versatility.

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Hierarchical Vision Transformers (ViTs) offer linear computational complexity concerning image size and create hierarchical feature maps. To solve the aforementioned issues, the authors  have developed a Data-independent Module-Aware Pruning method (DIMAP) to compress hierarchical ViTs. To ensure that "local" attention weights at different hierarchical levels are compared fairly in terms of their contribution, they treat them as a module and examine their contribution by analyzing their information distortion. Furthermore, they introduce a novel weight metric that is solely based on weights and does not require input images, thereby eliminating the dependence on the patch merging process. The experiments are fine to me.

### Strengths
The authors introduce a new pruning approach called Data-independent Module-Aware Pruning (DIMAP). DIMAP compares "local" attention weights fairly by treating them as a module and examining their contribution based on information distortion. A new weight metric that doesn't need input images is also introduced, making it data-independent. The method has been validated on Swin Transformers, demonstrating its effectiveness and efficiency.

### Weaknesses
To me the extensive applications will be beneficial to be new contributions of field.
Could you discuss more the new applications of your method.

### Questions
To me the extensive applications will be beneficial to be new contributions of field.
Could you discuss more the new applications of your method.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
