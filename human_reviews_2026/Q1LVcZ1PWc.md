# Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
In recent years, the long-range attention mechanism of vision transformers has driven significant performance breakthroughs across various computer vision tasks. However, these advancements come at the cost of inefficiency and substantial computational expense, especially when dealing with sparse data. While sparse attention mechanisms have been introduced to mitigate these issues by pruning tokens involved in attention, they often lack context-awareness and intelligence, frequently limiting the number of selected tokens uniformly across different inputs. To address these challenges, we propose a novel algorithm: Select and Pack Attention (SPA). SPA dynamically selects informative tokens using a low-cost gating layer and packs these selected tokens into new batches, allowing for a variable number of tokens to be used in GPU batch training and inference. Through extensive experiments on diverse datasets and multiple computer vision tasks, our method demonstrates superior performance and efficiency, including a 0.5-2.7 AP improvement in object detection and a 10.9%-24.9% reduction in computation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper authors analyzed the current issues with sparse attention mechanisms and propose a  SPA mechanism to address these challenges for both efficiency and performance. The SPA focuses attention computations solely on informative tokens using a supervised gatingblock in Vision Transformers. Experiments validated the effectiveness of SPT.

### Strengths
The credible innovations of the paper are as follows:
Integrated into the Swin Transformer’s hierarchical architecture, SPA forms the efficient Select and Pack Transformer, which works as image backbone network for various computer vision tasks and generates multi-scale representations. Employ multi-scale selection labels for explicit supervision using object labels to enhance selection accuracy and ensure effectiveness in complex computer vision tasks. In terms of object detection, the method of this paper compared to the current SOTA , not only improves detection accuracy but also reduces computational costs.

### Weaknesses
In the conclusion section of the paper, the analysis of the paper's shortcomings should be added.

### Questions
1. In Tables 1 to 3, the comparison method is based on results published in 2021. Why not choose the most recent research?
2. In the conclusion section, the analysis of the paper's shortcomings should be added.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To address the issue of numerous target-irrelevant redundant computations in attention calculation for object detection, this paper proposes a more efficient Vision Transformer (VIT). Different from traditional DynamicVIT and SparseVIT, which pad target-irrelevant tokens for a fixed batch size, the proposed SPT (Specific Token Preservation) only retains target-relevant tokens and packages them into the same batch for training. The proposed SPA (Sparse Attention Calculation) performs self-attention computation on the tokens screened and packaged by SnP (Selection and Packaging). Extensive comparisons with baselines on three datasets (COCO2017, BDD-S, and PASCAL VOC 2012) demonstrate that the proposed method achieves the

### Strengths
1. Combines Swin Transformer (SwinT) to propose a more efficient attention computation method, and supervises token selection through multi-scale labels.
2. Validates the proposed method on three sufficient datasets.

### Weaknesses
1. The overall architecture flowchart is slightly rough, especially in illustrating how multi-scale label selection is trained synchronously.
2. The gate mechanism of SnP is not clearly defined.

### Questions
Q1: If the model makes misjudgments in token selection, will more severe and irreversible errors occur? For instance, if an object is classified as background by the gate mechanism, does the model permanently lose the opportunity to learn the feature of it? Besides, Could you design an experiment to compare the proportion of tokens that actually contain objects before and after the screening process?​
Q2: Will there be significant differences between different gate?​
Q3: Will the multi-scale labels of small objects disappear as the image undergoes up sampling?​
Q4: The improvement in computational efficiency is accompanied by limitations in the model's field of view, which may sacrifice the model's expressive ability. Would retaining an appropriate number of background tokens around the image lead to better performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes Select and Pack Attention (SPA) and build a computationally efficient Vision Transformer architecture based on SPA. Specifically, the authors introduce several linear layers to predict the importance scores of visual tokens. Through sampling based on the score, SPA can extract the informative tokens, thereby reducing the sequence length involved in the attention process. Experimental results demonstrate that the proposed method achieves advantages in both accuracy and speed.

### Strengths
1.	The motivation of this work is clear. The authors point out the limitations of previous sparse attention methods, including their uniform sampling strategies and training implementations. They further propose the importance-based sampling scheme and a token packing implementation strategy to effectively address these issues.
2.	The presentation of experimental results is clear and well-organized. In each results table, the authors provide quantitative metrics including model performance, computational cost, and inference speed, which intuitively demonstrate the effectiveness of the proposed method. I also appreciate the experiment corresponding to Fig. 4, which employs GT-based selection to effectively decouple the effects of intermediate-layer GT supervision from the token selection process, thereby validating the effectiveness of the selection strategy itself.

### Weaknesses
1.	The main limitation lies in the restricted applicability, which is confined to the 2 tested tasks: object detection and instance segmentation. For tasks with sparser prediction targets, such as classification, captioning, and VQA, there is no available bounding box or segmentation mask to provide supervision for importance score prediction. Conversely, for tasks with denser prediction targets, such as SAM-style segmentation or panoptic segmentation, where targets may cover the entire image, the proposed token selection strategy would degenerate into uniform sampling across the whole image (as all tokens would have selection labels of “1” under the current framework). This greatly limits the general applicability of the proposed method.
2.	The paper demonstrates that the proposed method achieves superior efficiency within the current overall model architecture. However, this improvement arises from multiple contributing factors, such as the hierarchical design, window attention, and the proposed SPA. A more critical aspect is the efficiency gain brought specifically by SPA itself. For instance, under the same model architecture, how much difference in computational cost and inference speed would there be if Stages 3 and 4 used standard global attention compared to SPA?
3.	Not a major weakness, but the paper has some formatting issues. Many figures and tables are placed one page away from the corresponding text description that references them, which causes inconvenience for readers. It is recommended to adjust the layout.

### Questions
1.	In line 260, the authors mention “we set L to be M^2.” How is this value determined? As my understanding the length of the package containers should depend on the token sequence length corresponding to the scale of each stage, as well as the proportion of the predicted target relative to the entire image. Why is it related to the window size?
2.	Why is the Gumbel-Softmax operation introduced? Once the importance scores are determined, what is the difference between sampling based on these scores and deterministically selecting high-importance tokens (e.g., those above a fixed threshold)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new sparse attention mechanism, "Select and Pack Attention" (SPA), to make vision transformers more efficient, particularly for object detection. The core problem is that standard ViTs waste computation on uninformative background tokens.
SPA tackles this in two steps:
Select: It uses a lightweight linear gating layer to dynamically choose informative tokens. Crucially, this gate is explicitly supervised during training using ground-truth object labels (like bounding boxes), which helps it learn to be more accurate than heuristic-based pruning methods.
Pack: To handle the irregular number of selected tokens per image, it packs all selected tokens from a batch into fixed-size "package containers." This allows for efficient, parallel GPU processing without the massive padding overhead of other sparse methods.
The authors integrate SPA into a hierarchical backbone called the "Select and Pack Transformer" (SPT), which they apply only in the later stages to preserve early-stage features. They show that SPT improves both performance and computational efficiency compared to other state-of-the-art sparse attention models.

### Strengths
The main strength of this paper is the idea of using explicit supervision from downstream labels (bboxes/masks) to train the token selection gate. This is a simple and effective departure from most prior work, which relies on unsupervised heuristics (like token norms) or uniform pruning. This supervised approach is far more likely to preserve small, important objects, which is a common failure point for other sparse methods. The "packing" mechanism is also a smart engineering solution to the practical problem of running dynamic token selection in a batched GPU environment. This combination makes for a practical contribution to the field of efficient object detection.

The experimental validation is reasonably comprehensive. The authors don't just test on COCO; they also use BDD100K and create the BDD-S dataset to specifically test their hypothesis about small objects. The strong results on BDD-S are a convincing piece of evidence. The ablation studies in Tables 6 and 7 are also good, clearly justifying the design choice of applying SPA in later stages and showing the benefit of the selection loss.

The paper is exceptionally well-written and easy to follow. The problem is motivated clearly. Figure 1 provides a good visual intuition for the problem, and Figures 2 and 3 clearly lay out the proposed architecture and the core SPA block.

### Weaknesses
- The selection gate is trained using ground-truth object labels. This works great for fine-tuning on a detection or segmentation dataset, but it's unclear how this backbone could be pre-trained on a classification-only dataset like ImageNet. Does this method forgo pre-training entirely? Or does the gate have to be trained from scratch only during fine-tuning? This reliance on labels seems like a step back compared to other token selection methods. For example, EViT and DynamicViT learn to prune tokens dynamically without needing explicit object-level supervision. EViT, for instance, uses the attention to the class token as a guide, and DynamicViT trains its prediction module differentiably. 

- The method is conceptional similar to EViT but using ground truth labels to learn how to prune.

- Missing Related Work: The related work section on efficient transformers (2.2) is missing some relevant research on sparse attention patterns. This paper would be much stronger if it positioned itself relative to: 

   - Fixed sparse patterns (e.g., "Blockwise Self-Attention for Long Document Understanding, Big Bird: Transformers for Longer Sequences"). 

   - Methods that combine local and global attention patterns (e.g., "Twins: Revisiting the design of spatial attention in vision transformers"). 

   - Axis-decoupled attention pattern (e.g., "Axial attention in multidimensional transformers"). 

   - Learned sparse attention pattern (e.g., "Sparsifiner: learning sparse instance-dependent attention for efficient vision transformers").

- Modest Overall Speedup: 
While the backbone computation reduction of 10-25% is good, the actual wall-clock speedup (FPS) reported in the tables is modest. For example, in Table 1, SPT-T (54 FPS) is only slightly faster than the dense Swin-T (50 FPS). The authors themselves note that the early (dense) stages dominate the computation, which limits the total possible gains

- Missing Performance vs. Pruning Trade-off Analysis: The experimental evaluation is the lack of a "performance vs. efficiency" trade-off curve. The authors present results for what seems to be a single, fixed operating point. But they didn't show how performance is affected by the level of pruning. An ablation study is missing: what happens to the accuracy as you change the selection ratio and apply selection on shallow layers? For example, how gracefully does performance degrade if the model is forced to prune 80% of tokens? How much does it improve if it only prunes 50%?

### Questions
See W1 and W5

### Soundness
3

### Presentation
3

### Contribution
2
