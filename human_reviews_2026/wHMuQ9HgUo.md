# Hierarchical Prototype Learning for Semantic Segmentation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Conventional semantic segmentation methods often fail to distinguish fine-grained parts within the same object because of missing links between part-level cues and object-level semantics.
Inspired by how humans recognize objects, which involves first identifying them as a whole and then distinguishing their parts, we propose a hierarchical prototype-based segmentation method called Hierarchical Prototype Segmentation (HiPoSeg). This builds a structured prototype space that captures both abstract object-level representations and detailed part-level features, enabling consistent alignment between levels. HiPoSeg leverages a hierarchical contrastive learning strategy to structure semantic representations across levels, encouraging both intra-level discrimination and cross-level consistency.
Experiments on standard benchmarks such as Cityscapes, ADE20K, Mapillary Vistas 2.0, and PASCAL-Part-108 demonstrate that HiPoSeg produces consistent performance improvement with an average gain of +3.07\%p mIoU without any additional inference cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Traditional semantic segmentation methods often overlook semantic relationships across levels. To address this issue, the article proposes a Hierarchical Prototype Learning (HiPoSeg) method. By introducing high- and low-level prototype contrastive learning and high- and low-level alignment constraints, it effectively combines coarse categories and fine-grained part semantics, improving segmentation accuracy and consistency. Experiments across multiple datasets validate the proposed HiPoSeg method's ability for fine-grained segmentation and semantic consistency.

### Strengths
The paper effectively integrates coarse- and fine-grained semantics via hierarchical contrastive learning, thereby improving segmentation accuracy. 

It enhances semantic consistency across different levels by employing alignment constraints between high- and low-level features.

### Weaknesses
- In Section 3.2, regarding the setting of some hyperparameters (e.g., "σ1=0.25, σ2=1" and "m=0.9"), providing an analysis of how the choice and adjustment of these hyperparameters affect performance would help enhance the reproducibility and understanding of the method. Additionally, the paper introduces a step-by-step activation mechanism at different stages (from feature representation to prototype learning, and then to alignment loss). It is recommended to provide more experimental or theoretical support to clarify why this staged training arrangement effectively facilitates hierarchical learning.

- The explanation in Section 3 could benefit from more step-by-step details, especially regarding prototype construction. Although Figure 2 provides a good overview of the overall structure, adding more detailed annotations and explanations would help readers better understand each step.

- The latest relevant methods are missing from the comparison.

### Questions
- Some table captions are above the tables, while others are below. It is recommended to review and adjust the table layout for greater clarity.

- In the first paragraph of Section 3, "...prototype space in 3.1. Second, in 3.2..." should be revised to "...prototype space in Sec. 3.1. Second, in Sec. 3.2...“.

- There is an unnumbered equation between Equation 1 and Equation 2.

- Table 7 should be a figure, not a table.

### Soundness
3

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
* The paper introduces Hierarchical Prototype Segmentation (HiPoSeg), a new framework for semantic segmentation that models visual recognition as a coarse-to-fine process, reflecting how humans perceive objects.
* HiPoSeg builds a hierarchical prototype space that captures both high-level object semantics and fine-grained part semantics, linking them through a hierarchical contrastive learning objective. To maintain consistency across these levels, the model applies a multi-level alignment constraint, reducing misclassifications between unrelated categories.
* A key strength of HiPoSeg is that it functions as a training-only, plug-and-play module, meaning it adds no extra computational cost during inference while still enhancing semantic consistency and mean Intersection over Union (mIoU).
* Tested on four major benchmarks, Cityscapes, ADE20K, Mapillary Vistas 2.0, and PASCAL-Part-108, HiPoSeg consistently outperforms strong baselines such as HSSN, LogicSeg, and ProtoSeg, achieving notable improvements across all datasets.

### Strengths
* The paper clearly formalizes hierarchical semantics by organizing prototypes in a top-down hierarchy, addressing the structural limitations of flat segmentation.
* The proposed component is used only during training, requiring no additional computation or parameters at inference, which is practically appealing.
* Cross-level semantic consistency: Through an alignment constraint, HiPoSeg maintains strong coherence between high-level and low-level features, minimizing semantic drift and enhancing the precision of fine-grained segmentation.
* Comprehensive validation: The paper provides extensive quantitative and qualitative analyses across multiple benchmark datasets, clearly demonstrating the robustness and generality of the method.
* HiPoSeg achieves higher mIoU than SoTA models (HSSN, LogicSeg, ProtoSeg, ContextSeg), even with a weaker backbone (ResNet-101), underscoring its efficiency and scalability.

### Weaknesses
* While the hierarchical prototype framework is well-motivated, it conceptually overlaps with prior hierarchical reasoning models such as HSSN (CVPR 2022) and LogicSeg (ICCV 2023). The improvement may be seen as incremental rather than fundamentally new.
* The experiments focus primarily on two-level hierarchies. The extension to deeper hierarchies (e.g., three-level settings in ADE20K and Mapillary) is only lightly discussed, leaving generalization uncertain.
* Although the paper presents loss-wise ablations, it lacks an analysis of hyperparameter sensitivity.
* The qualitative comparisons rely heavily on visual inspection of boundary sharpness and small-object recognition, which are somewhat subjective and lack quantitative support.
* Stability under multi-term losses: The hierarchical contrastive loss combines multiple objectives (Lfh, Lfl, Lalign), which may cause gradient interference in large-scale training unless carefully balanced.
* HiPoSeg assumes a fixed and known label tree, making it less applicable to datasets without clear hierarchical semantics or dynamically evolving taxonomies.
* Although mentioned in the conclusion, the paper does not empirically validate HiPoSeg in open-vocabulary or long-tailed scenarios.

### Questions
Can the hierarchical prototype space be learned when the label hierarchy is not predefined?
What is the rationale behind the curriculum schedule for enabling hierarchical components, and how sensitive is the model to these thresholds?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduce HiPoSeg, a approach to semantic segmentation that leverages hierarchical prototype learning to improve segmentation performance, particularly for fine-grained tasks. The key idea is to model both high-level categories and their subcategories through hierarchical prototypes, allowing the model to capture both coarse and fine-grained semantic structures.

### Strengths
* The paper introduces hierarchical prototype learning for semantic segmentation, combining prototype-based learning with a hierarchical structure, which improves the model’s ability to differentiate fine-grained subcategories.
* The problem is clearly defined, and the paper is well-structured,  presenting experimental results in an understandable manner.
* HiPoSeg improves segmentation results, and its scalability and flexibility suggest broad potential for various applications.

### Weaknesses
* This paper uses DeepLabv3, a classic but somewhat outdated baseline. To validate the generalizability of this method, the authors should consider using a wider variety of CNNs and Transformer-based baselines. Also, this can demonstrate the author's claim of "plug-and-play efficiency".

* The paper mentions the novelty of hierarchical prototype learning, but it lacks a discussion with other similar methods, such as other prototype-based semantic segmentation methods or hierarchical network approaches.

* It would be useful to clearly distinguish the fundamental differences between this approach and existing hierarchical methods, for example, does the prototype learning mechanism in HiPoSeg offer better generalization ability? Does it form a clearer semantic structure in the feature space?  Does HiPoSeg more effectively leverage logical constraints or hierarchical consistency?

* It may be better to add visual analysis (e.g., t-SNE or prototype space structure plots) to demonstrate how the prototype space is organized and whether the hierarchical structure is reflected in the learned representations.

* While the paper introduces hierarchical prototype learning for handling fine-grained segmentation tasks, there has not been sufficient exploration of its performance on fine-grained subcategories and its advantages in these tasks.


I would be happy to revise my score if the author addresses these points.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Existing semantic segmentation methods often treat the task as flat classification, ignoring structural relationships between classes and failing to replicate human “whole-to-part” hierarchical recognition, leading to poor fine-grained part segmentation. To address this, the paper proposes HiPoSeg, a hierarchical prototype-based method that constructs a structured prototype space capturing object-level and part-level features, uses hierarchical contrastive learning for intra-level discrimination and cross-level consistency, and only acts during training (no inference overhead). Experiments on Cityscapes, ADE20K, Mapillary Vistas 2.0, and PASCAL-Part-108 show an average +3.07%p mIoU gain over baselines and state-of-the-art methods. Its core contributions include: 1) hierarchical prototype learning for structured class representation; 2) multi-level alignment constraints to suppress semantic leakage; 3) a plug-and-play, training-only design with zero inference cost.

### Strengths
1. Unlike prior hierarchical segmentation methods (e.g., HSSN, LogicSeg) that handle hierarchy as a fixed auxiliary term or fuse signals at the probability level, HiPoSeg explicitly constructs a hierarchical prototype space and uses contrastive learning to structure class representations—this design principle is a meaningful departure from existing work.
2. The experimental design is relatively systematic: it validates performance on 4 diverse benchmarks (urban scenes, daily scenes, part parsing), and ablation experiments (Table 5-6) confirm the necessity of high-level/low-level contrastive losses and alignment constraints, ensuring the method’s core components are evidence-based.
3. The method description is detailed: Section 3 clearly defines prototype construction (Eqs. 1-4), hierarchical contrastive learning (high/low-level losses), and alignment constraints (Eqs. for 𝒱ₐₗᵢ₉ₙ), making the technical implementation reproducible in principle.
4. The “zero inference overhead” design is practically valuable—HiPoSeg acts as an auxiliary training component, avoiding the common trade-off between performance gain and deployment cost in segmentation research, which is useful for real-world applications (e.g., autonomous driving).

### Weaknesses
1. Experimental Completeness: No hyperparameter sensitivity analysis: The paper sets σ₁=0.25, σ₂=1, m=0.9 without justifying why these values are optimal or how performance changes if they vary (e.g., σ₁=0.1 vs. 0.5).
Lack of alternative backbone validation: All experiments use ResNet-101, but no results on mainstream backbones like HRNet-W48 or SegFormer are provided, making it unclear if HiPoSeg’s gains generalize beyond ResNet.
2. Insufficient qualitative challenging cases: The qualitative results (Figure 3/4) only show simple scenes (e.g., clear buses in Cityscapes) and omit scenarios with dense small objects (e.g., crowded pedestrians) or severe occlusion—critical for verifying fine-grained segmentation ability.
3. Motivation and Context:The related work section does not contrast HiPoSeg with 2024 SOTA methods (e.g., Contextrast’s contextual contrastive learning) in depth, only listing them in tables. It fails to explain why HiPoSeg’s hierarchical prototype design outperforms these latest methods, weakening the motivation’s urgency.
4. No failure case analysis: The paper only reports successful segmentation results but does not analyze when HiPoSeg fails (e.g., which classes/regions still have high error rates) or why, limiting insights into the method’s limitations.
5. Theoretical Depth:The paper does not provide quantitative analysis of prototype space (e.g., t-SNE visualizations of high/low-level prototypes or feature similarity metrics between related classes like “truck” and “bus”). It only claims “structured representation” without empirical evidence of how prototypes reduce cross-class confusion.

### Questions
1. Experimental Design:
Could you provide results of HiPoSeg on alternative backbones (e.g., HRNet-W48, SegFormer-B5) to verify its generalization? If performance drops on certain backbones, what is the root cause?
Could you add a hyperparameter sensitivity study (varying σ₁, σ₂, m, and temperature τ/κ) and explain how you selected the optimal values?
Could you supplement qualitative results on challenging scenes (dense small objects, severe occlusion) and analyze error cases (e.g., which low-level classes still have low IoU)?
2. Motivation and Comparison:
How does HiPoSeg’s hierarchical prototype design specifically outperform 2024 SOTA methods like Contextrast (which uses contextual contrastive learning)? Could you add a head-to-head comparison with these methods on the same benchmarks?
Prior work (e.g., ProtoSeg) also uses prototypes—why is combining prototypes with hierarchy more effective than flat prototype learning? Could you provide a controlled experiment comparing HiPoSeg with a flat prototype baseline?
3. Theoretical and Mechanistic Analysis:
Could you provide t-SNE or UMAP visualizations of the prototype space to show how high/low-level prototypes of related classes (e.g., “car”/“truck” in high-level, “horse head”/“horse leg” in low-level) are structured?
For rare low-level classes (e.g., “train” in Cityscapes, which has a baseline IoU of 23.88%), how does HiPoSeg’s prototype memory bank address data sparsity? Could you analyze the prototype update process for these classes?

### Soundness
3

### Presentation
3

### Contribution
3
