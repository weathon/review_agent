# Learning AND–OR Templates for Compositional Representation in Art and Design

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
This work proposes a compositional AND–OR template for art and design that encodes the part–relation–geometry organization of images in a structured and interpretable form. Within a maximum-entropy log-linear model, we define a unified consistency score as log-likelihood gain against a reference distribution and decompose it into term-level evidence, enabling an evidence-to-prescription mapping for actionable composition guidance. Learning is performed by a penalized EM-style block-pursuit with sparsity and local mutual exclusivity: object templates are learned first and reused as scene terminals to induce scene templates. A semi-supervised structural expansion, which is triggered by matching gain and structural-consistency thresholds, bootstraps new branches from unlabeled, high-quality images. Evaluations on a curated compositional dataset and AVA/AADB themes show strong agreement with expert paradigms, interpretable parse trees, and competitive performance with deep baselines while exhibiting higher alignment with human ratings. The learned templates also act as lightweight structural conditions to steer AIGC generation and layout design. Overall, the framework delivers a transferable structural prior with favorable data/parameter efficiency and a unified pathway for explainable visual assessment and generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces a compositional AND–OR template framework for art and design. Authors extend classical AND–OR templates (AOTs) from object-level vision to scene-level compositional understanding, and they combine interpretability, compositionally, and aesthetic reasoning within a mathematically rigorous framework.

### Strengths
The proposed approach produces good interpretability with parse trees and term-level attribution.

Authors design a maximum-entropy log-linear unified score, and learn via penalized MLE with an EM-type block-pursuit.

### Weaknesses
The method proposed in this paper is too complicated, making it less practical in application.

The source code provided by the author is confusing, and I find it difficult to see its relevance to this paper.

### Questions
Authors should discuss the detail difference between related work and state their contribution.
[1] Improving Compositional Generation with Diffusion Models Using Lift Scores

Is the template-based generation method suitable for complex scenes, such as dense crowds and urban street scenes?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel framework for learning compositional, interpretable representations of images, specifically tailored to capture the structural regularities found in art and design. The core idea is a two-level AND-OR Template (AOT). The method is evaluated on curated datasets and standard aesthetic benchmarks (AVA, AADB), showing competitive performance with deep learning baselines while offering superior interpretability and data efficiency.

### Strengths
- The proposed method in this paper offers a transparent, auditable pathway from image features to a qualitative evaluation. The decomposition of the consistency score into object, relation, and geometry terms directly maps to actionable feedback, which is highly valuable for creative assistance tools. The technical approach is sound and thoroughly described. The two-level learning pipeline is a practical way to manage combinatorial complexity. The inclusion of a semi-supervised structural expansion (SSE) mechanism is a thoughtful addition that enhances the framework's scalability and realism.

### Weaknesses
- While the block-pursuit algorithm with sparsity constraints is designed to combat combinatorial explosion, its scalability to highly complex scenes with dozens of interacting objects is not thoroughly demonstrated.
- The method proposed in this paper focuses exclusively on the structural patterns like relations and geometry. It explicitly does not account for other critical aesthetic factors like lighting, color harmony, texture, and material properties. This limits its comprehensiveness as a full aesthetic assessment tool; it is primarily a compositionalassessment tool.
- The learning process appears sensitive to the initial seeds. The paper does not fully explore how robust the method is to poor initial templates or how to bootstrap without a high-quality, small labeled set.
- Some of the images in the paper have low resolutions, caused their details to become blurred. This could be polished in the revised version of this paper.

### Questions
- The templates are learned for specific themes. Can the framework handle or represent more abstract, non-object-centric compositional principles such as "balance," "rhythm," or "negative space" that may cut across different concrete themes?
- The semi-supervised expansion relies on a dual-threshold rule. Could you discuss the sensitivity of the final model to the choice of these thresholds?

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
2

### Summary
This paper proposes a compositional AND-OR template for art and design that includes the part-relation-geometry of images in a structured and interpretable form. It extent AND-OR templates from the object level to the scene level. The learning process is a penalized EM-style block-pursuit with sparsity and local mutual exclusivity. Authors also propose a semi-supervised structural expansion methods which could bootstrap new branches from on the boat high-quality images. Experiments show that the method proposed in this paper achieves good objective consistency and subjective reliability. Compared to other single image important methods, the method in this paper achieves, better consistency score with compositional AND-OR templates.

### Strengths
1. The AND-OR template for scene proposed in this paper contribute towards more controllable and explainable components for image generation and classification. 
2. Compared to other deep neural network based methods, the master propose in this paper is lightweight with fewer parameters.

### Weaknesses
1. In the objective consistency evaluation, only two examples of the proposed method are presented. The image generated in other ways are not shown for comparison. Yes, we can draw the conclusion that the generated image is mostly consistent to the template, but we can't draw the conclusion about to what extent of the consistency is compared to other way of generation. 
2. In table 1, the accuracy of the proposed method is lower than baseline ViT-B/16 + templates. 
3. In the section 5.1.2 subjective reliability analysis, during the evaluation, the most consistent rate is 43.6%, and the mostly consistent rate is 31.2%, and 12% is inconsistent.  It is unclear how well it is compared to the consistent evolution results based on other methods. And more analysis about the cases labelled as inconsistent are expected.

### Questions
1. The proposed method learn the template based on the training images. Could this message extend to other elements that are not included in the training images?

### Soundness
2

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
3

### Summary
The paper proposes a compositional representation for art and design using AND–OR templates (AOTs) under a maximum-entropy log-linear model. The framework defines a unified consistency score (log-likelihood gain vs. a reference distribution) that decomposes into interpretable object-, relation-, and geometry-level evidence. Learning proceeds via a penalized EM-type block-pursuit algorithm with sparsity and local mutual exclusivity constraints, first learning object templates and then reusing them as scene terminals. A semi-supervised structure expansion (SSE) further grows the model from unlabeled images. The method shows interpretability, human alignment, and competitive performance with deep baselines on aesthetic datasets (AVA/AADB) and scene understanding tasks.

### Strengths
1. The AND-OR graph structure is inherently interpretable.
2. The proposed method achieves comparable performance on the AVA dataset, with smaller training parameters.
3. The proposed SSE is practical and well-justified, bridging labeled and unlabeled data in a structured learning context.

### Weaknesses
1. the expectation module is dependent on YOLO, object that are not part of YOLO detection algorithm are not going to be avilable.

### Questions
1. How does the system handle failures or partial detections from YOLO? Does this dependency undermine the claim of a fully interpretable system?

### Soundness
3

### Presentation
3

### Contribution
2
