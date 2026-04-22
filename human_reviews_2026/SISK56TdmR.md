# TARO: Toward Semantically Rich Open-World Object Detection

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Modern object detectors are largely confined to a "closed-world" assumption, limiting them to a predefined set of classes and posing risks when encountering novel objects in real-world scenarios. While open-set detection methods aim to address this by identifying such instances as *Unknown*, this is often insufficient. Rather than treating all unknowns as a single class, assigning them more descriptive subcategories can enhance decision-making in safety-critical contexts. For example, identifying an object as an *Unknown Animal* (requiring an urgent stop) versus *Unknown Debris* (requiring a safe lane change) is far more useful than just *Unknown* in autonomous driving. To bridge this gap, we introduce TARO, a novel detection framework that not only identifies unknown objects but also classifies them into coarse parent categories within a semantic hierarchy. TARO employs a unique architecture with a sparsemax-based head for modeling objectness, a hierarchy-guided relabeling component that provides auxiliary supervision, and a classification module that learns hierarchical relationships. Experiments show TARO can categorize up to 29.9% of unknowns into meaningful coarse classes, significantly reduce confusion between unknown and known classes, and achieve competitive performance in both unknown recall and known mAP. Code is available at: https://anonymous.4open.science/r/TARO

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes TARO, a novel framework for semantically rich open-world object detection (OWOD). Unlike conventional OWOD methods that label all novel objects as a single “Unknown” class, TARO leverages a semantic hierarchy to assign unknown objects to meaningful coarse-grained parent categories (e.g., “Unknown Vehicle” or “Unknown Animal”). Built upon Deformable DETR, TARO introduces three key components: (1) a sparsemax-based objectness head to model competition and sparsity among object queries; (2) a hierarchy-aware classification activation that enforces parent–child consistency via learnable coupling parameters; and (3) a hierarchy-guided relabeling strategy that uses non-leaf node confidences to provide auxiliary supervision for objectness. Experiments show that TARO achieves high unknown recall (U-R), low absolute open-set error (AOSE), and reports a new metric, hierarchy accuracy (HAcc), reaching up to 29.9% on the OWOD split.

### Strengths
The main strength of this work lies in its redefinition of the OWOD task: instead of treating all unknowns uniformly, TARO enables semantically meaningful categorization of novel objects into coarse parent classes, which is highly valuable in safety-critical applications such as autonomous driving. The proposed components, sparsemax-based objectness modeling, adaptive hierarchical coupling, and hierarchy-guided relabeling, are well-motivated and collectively address key challenges in semantic generalization under open-world settings.

### Weaknesses
- Limited generalization due to semantic coverage in training: The reported HAcc drops dramatically to ~5% on the OW-DETR split (referred to in the manuscript as a more challenging benchmark), compared to 29.9% on the OWOD split. This is because the OW-DETR split groups semantically similar classes (e.g., all animals) into the same task, preventing the model from ever observing certain parent categories (e.g., “Food”) during training. Consequently, TARO cannot meaningfully categorize unknowns from unseen branches of the hierarchy, revealing a strong dependence on the specific task partitioning and limited true open-world generalization.
- Disproportionate complexity versus performance gain: TARO introduces multiple novel components (sparsemax objectness head, hierarchical activation, relabeling mechanism), yet its known-class mAP consistently lags behind strong baselines such as RandBox and ALLOW-DETR across multiple tasks. While it improves U-R and reduces AOSE, this trade-off—sacrificing known detection accuracy for better unknown recall—may not be acceptable in real-world systems where both capabilities are critical. The marginal gains do not sufficiently justify the added architectural complexity.
- Evaluation bias in HAcc: HAcc is computed only over detected unknown objects, ignoring the large number of missed detections (i.e., low U-R). When U-R is low, HAcc is evaluated on a highly selective subset of “easy” unknowns, potentially inflating the perceived hierarchical reasoning ability. Moreover, the stark discrepancy in HAcc between benchmarks (29.9% vs. 5%) undermines the reliability and comparability of this metric, weakening the paper’s main claim.
- Lack of comparison with recent OWOD and foundation-model-based approaches: The paper does not compare against recent methods that leverage large vision-language models (e.g., CLIP, SAM) or other open-vocabulary detectors[1, 2, 3]. Given the rapid progress in this area, it is essential to position TARO relative to both traditional OWOD methods and modern foundation-model-based alternatives. Without such comparisons, the claimed advantages of TARO remain unsubstantiated.

[1] ktcn: enhancing open-world object detection with knowledge transfer and class-awareness neutralization

[2] Recalling Unknowns Without Losing Precision: An Effective Solution to Large Model-Guided Open World Object Detection

[3] Exploring Orthogonality in Open World Object Detection

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TARO, a framework for open-world object detection that introduces a hierarchical, semantically aware treatment of unknown categories. Instead of labeling all novel objects as “Unknown,” TARO maps them to parent classes within a taxonomy. The method integrates a Sparsemax-based objectness head, hierarchy-aware activation, and a relabeling strategy, leading to stable and interpretable improvements on OWOD and OW-DETR benchmarks.

### Strengths
The idea of hierarchical modeling for open-world detection is intuitive and meaningful, addressing a real limitation of prior “flat” OWOD formulations.

The method design is coherent, with each component targeting a specific issue (semantic inconsistency, suppression of unknowns, weak supervision).

Results are strong and comprehensive, showing consistent gains across key metrics (U-R, AOSE, HAcc).

The work is reproducible and does not rely on external large models, which strengthens its engineering value.

### Weaknesses
The hierarchical accuracy gains are somewhat limited; the model seems to underuse the taxonomy signal.

The Sparsemax head lacks gradient or stability analysis—some intuition on optimization behavior would be useful.

Performance varies notably between splits, and the paper could analyze failure cases to better understand the taxonomy imbalance.

Missing comparison with recent vision-language detectors, which could help position TARO among broader open-world approaches.

### Questions
Hierarchy-aware activation (Eq. 2):
Could you clarify why a multiplicative interaction is chosen to propagate activation along the taxonomy? Have you explored additive or normalization-based alternatives? A brief sensitivity analysis of the scaling factor α₍c₎ would help justify this design.

Sparsemax head behavior:
Sparsemax can yield zero gradients for inactive logits. Did you observe any optimization instability or dead queries in early training? Some gradient statistics or qualitative examples would be helpful.

Relabeling strategy robustness:
The relabeling threshold (minimum non-leaf score among matched queries) might be sensitive to image content. Have you considered adaptive or percentile-based thresholds? An ablation comparing strategies could clarify its stability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes TARO, a hierarchical open-world object detection framework that categorizes unknown objects into coarse semantic classes using a sparsemax-based objectness head and hierarchy-aware mechanisms. The idea is interesting and the results are promising, but the taxonomy construction and visualization of hierarchical predictions need clearer explanation and stronger experimental support.

### Strengths
- Research on object detection in open-world scenarios is of great significance, as it addresses the challenges of recognizing unseen or novel categories in real-world environments.
- Investigating hierarchical OOD (out-of-distribution) detection is particularly valuable, as it enhances the interpretability and semantic understanding of the model’s predictions.

### Weaknesses
- The authors seem to have omitted discussion of open-vocabulary object detection in the introduction. This task is somewhat different from open-set detection or closed-set detection, and it is also an important and popular branch.
- In the design principles, the uncertainty (or objectness probability) of unknown objects should ideally increase as the semantic hierarchy becomes coarser, that is, higher-level parent categories should correspond to greater uncertainty. However, this principle is not explicitly reflected in the current design formulation.
- The result presentation is insufficient, the paper only shows qualitative detection outputs. A better way to present the results would be to include bar charts illustrating the predicted categories and objectness confidence at different hierarchical levels, which would make the findings more interpretable and clearly demonstrate the model’s hierarchical reasoning behavior.
- The paper does not specify how the hierarchical taxonomy (T) is constructed, is it based on WordNet, ChatGPT-generated relations, or manually defined structures? The transferability of this taxonomy across different datasets should be discussed. Moreover, since the authors emphasize the *open-world* setting, it is unclear whether the hierarchy needs to be continuously updated as new categories emerge, or if it is designed for specific scenarios and fixed applications. The authors should also discuss how different forms or depths of the taxonomy may affect the model’s performance and stability.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces TARO, a framework for open-world object detection (OWOD) that advances beyond simply flagging unknown instances as “unknown,” instead leveraging a semantic hierarchy to classify unknowns into meaningful coarse categories. TARO integrates a sparsemax-based objectness head, a hierarchy-aware classification module, and a taxonomy-guided relabeling strategy. Results on OWOD and OW-DETR splits show that TARO achieves strong recall of unknown objects, reduces confusion between known and unknown categories, and uniquely categorizes unknowns into higher-level nodes of a semantic hierarchy, supported by an extensive quantitative, qualitative, and ablation analysis.

### Strengths
- **Extension of Open-World Object Detection (OWOD):** The core contribution of this work is its move beyond the simple "known vs. unknown" binary. TARO not only identifies novel objects but also groups them into meaningful coarse-grained categories. This capability is highly practical for real-world settings like autonomous driving and aligns more closely with how humans handle novel items.
- **A Well-Integrated Architecture:** The architecture cleverly synthesizes three key components : a sparsemax-based objectness head to manage query competition ; a hierarchy-aware activation (with a learnable strength parameter) to enforce parent-child consistency ; and a hierarchy-guided relabeling strategy to provide auxiliary supervision.
- **Thorough Empirical Evaluation:** The experimental section is robust, comparing TARO against strong baselines based on both DETR and Faster R-CNN . The authors use a comprehensive suite of metrics (like mAP, U-R, and HAcc) to demonstrate performance . The qualitative results in Figures 3 and 4 also visually demonstrate the model's effectiveness.
- **Clear Ablation Studies:** The ablation study in Table 3 clearly dissects the model. It systematically dismantles the architecture to validate the individual contributions of the sparsemax head, the relabeling strategy, and the learnable strength parameter, providing strong support for the final design choices .
- **Reproducibility:** The authors provide complete implementation details, including hyperparameters, training schedules, and a public code repository, ensuring the work can be reproduced .

### Weaknesses
1. **Limited Novelty:** The paper's main novelty lies in its synthesis of existing ideas, not in a brand new algorithm. While the integration of sparsemax, hierarchical awareness, and relabeling is well-motivated , these individual components are already known in the field. The contribution is more about a careful and effective combination rather than a new algorithmic discovery.
2. **Hierarchy and Dataset Limitations:** The model is constrained by its fixed, hand-defined taxonomy. The hierarchy used is based on COCO/VOC and is only moderately sized. This is a poor fit for real-world applications like autonomous driving, which need to handle much larger or more fluid taxonomies. The paper acknowledges this  but doesn't offer a practical way to scale the method or adapt it to evolving hierarchies.
3. **Incomplete Justification for Sparsemax:** The reasoning for using sparsemax feels incomplete. While the ideas of competition and sparsity make sense , the ablation study (Table 3) doesn't show a massive performance win. The paper only compares it to a standard softmax, which is arguably a weak baseline. Other obvious alternatives, like focal loss or top-k activations, aren't discussed, making it hard to be sure if sparsemax is truly the best choice.
4. **Limited Qualitative Results and Scalability:** The qualitative results are limited and don't address scalability. The figures (Fig. 3 & 4) show only a few selected examples. This makes it hard to judge how robust the categorization really is, especially for ambiguous objects. It's also unclear how the method would perform computationally or accurately if the taxonomy grew to include hundreds of parent nodes.

### Questions
1. **Out-of-Taxonomy Generalization:** Can the approach be extended or modified to address cases where unknown objects fall entirely outside the fixed taxonomy (e.g., a truly new semantic domain)? How would you propose to detect/flag such hard unknowns?
2. **Comparison to Open-Vocabulary Approaches:** Why are recent open-vocabulary/region-word alignment methods omitted from the empirical comparison? Could you add them, and do you expect TARO’s hierarchical approach to outperform such models for coarse-level categorization?
3. **Robustness Analysis:** How sensitive is TARO to (a) the taxonomy structure (e.g., deeper trees or more overlapping parent nodes), (b) thresholding choices in relabeling, and (c) the initial value/range for $\alpha_c$? Could the authors provide additional experiments or analysis?
7. **Scalability:** Given the moderate-sized taxonomy in Figure 5, how does TARO’s computational cost scale with substantially deeper or broader semantic trees? Are there practical bottlenecks in the hierarchy-aware activation or relabeling as taxonomy grows?

### Soundness
3

### Presentation
2

### Contribution
2
