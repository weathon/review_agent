# SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
A consistent trend throughout the research of oriented object detection (OOD) has been the pursuit of maintaining comparable performance with fewer and weaker annotations. This is particularly crucial in the remote sensing domain, where the dense object distribution and a wide variety of categories contribute to prohibitively high costs. Based on the supervision level, existing OOD algorithms
can be broadly grouped into fully supervised, semi-supervised, and weakly supervised methods. Within the scope of this work, we further categorize them to include sparsely supervised and partially weakly-supervised methods. To address the challenges of large-scale labeling, we introduce the first Sparse Partial Weakly-Supervised Oriented Object Detection (SPWOOD) framework, designed to efficiently leverage only a few sparse weakly-labeled data and plenty of unlabeled data. Our framework incorporates three key innovations: (1) We design a Sparse-annotation-Orientation-and-Scale-aware Student (SOS-Student) model to separate unlabeled objects from the background in a sparsely-labeled setting, and learn orientation and scale information from orientation-agnostic or scale-agnostic
weak annotations. (2) We construct a novel Multi-level Pseudo-label Filtering (MPF) strategy that leverages the distribution of model predictions, which is informed by the model’s multi-layer predictions. (3) We propose a unique sparse partitioning approach, ensuring equal treatment for each category. Extensive experiments on the DOTA-v1.0 and v1.5 datasets show that SPWOOD framework
achieves a significant performance gain over traditional OOD methods mentioned above, offering a highly cost-effective solution.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes the SPWOOD (Sparse Partial Weakly-Supervised Oriented Object Detection) framework, which is the first to support multi kinds of annotations, such as RBox, HBox and point under a sparse partial weakly-labeled setting, forming a unified training pipeline for oriented object detection (OOD) task.
The SPWOOD framework is built upon the classic Teacher-Student paradigm, aiming to efficiently utilize a minimal amount of sparsely and weakly labeled data alongside a large volume of unlabeled data. The core innovations include three aspects:

1.	SOS-Student (Sparse-annotation-Orientation-and-Scale-aware Student) Model: This model is specifically designed for sparse annotation scenarios, effectively distinguishing between unlabeled objects and background. It also employs self-supervised learning to extract crucial orientation and scale information from weak annotations that initially lack these details.

2.	Multi-level Pseudo-label Filtering (MPF) Mechanism: To overcome the limitations of conventional pseudo-label filtering strategies (e.g., static thresholds or single-distribution modeling), MPF utilizes the Gaussian Mixture Model (GMM) to model the confidence distribution of predictions at every layer of the Feature Pyramid Network (FPN). This allows for dynamic adjustment of filtering thresholds, leading to the generation of higher quality and more stable pseudo-labels.

3.	Overall Sparse Method: An innovative sparse dataset creation technique. It treats all annotations across the entire dataset as a whole, applying a consistent sampling ratio to each category. This ensures that all classes are treated equally and effectively maintains the original dataset's class distribution.

The paper conducts extensive experiments on the DOTA-v1.0 and v1.5 datasets under a sparse-partial annotation setting. The results demonstrate that the SPWOOD framework achieves significant performance improvements compared to traditional semi-supervised and weakly supervised methods and reaches highly competitive performance with current state-of-the-art OOD algorithms.

### Strengths
1.	A Unified Framework: SPWOOD is the first OOD framework to support multi kinds of annotations, such as RBox, HBox and point under a sparse partial weakly-labeled setting, significantly pushing the boundaries of OOD research. It provides a theoretical and practical foundation for remote sensing applications where labeled data is extremely scarce.

2.	Strong Robustness to Sparsity: The SOS-Student model effectively addresses the core challenge in sparse annotation—the misclassification of unlabeled objects as background (Hard Negatives)—by introducing an adaptive loss modulation mechanism. This improves the model's learning accuracy in sparsely annotated scenarios.

3.	High-Quality Pseudo-label Generation: The MPF mechanism accounts for the prediction score discrepancies across different FPN levels by independently modeling the distribution for each layer. This allows for dynamic and targeted filtering of pseudo-labels, which significantly enhances the quality and stability of the guidance signal during the semi-supervised learning phase.

4.	Fair Data Sampling Mechanism: The proposed Overall Sparse Method ensures a fair sampling process by applying a consistent sampling ratio to all classes across the entire dataset. This avoids the disproportionate preservation of rare categories often seen in traditional Single-Image Sparse Methods, ensuring the class distribution of the sparse data remains consistent with the original dataset.

5.	Superior Performance: The experimental results demonstrate that SPWOOD achieves highly competitive performance on the DOTA datasets, validating the effectiveness of the proposed methodology.

### Weaknesses
1. Missing Bounding Box Parameter Definition: In the context of the bounding box loss (Formula 4), the crucial dimensional parameters w and h are not explicitly defined. It is unclear whether these variables refer to the predicted box, anchor box.

2. The sparse processing and the semi processing, it is unclear whether the data is first made sparse and then split, or vice versa. This opacity in the data pre-processing pipeline severely hampers the reader's ability to accurately replicate the structure of the final SPWOOD dataset.

3. Authors should incorporate a dedicated discussion about the inherent constraints of the current research and promising avenues for future work.

### Questions
1. Regarding Parameter Definitions in Formulas (4): In the bounding box loss (Formula 4), please explicitly define the symbols w and h?. Are these variables referring to the dimensions of the predicted box, the anchor box?

2. Could the authors provide a flowchart or a detailed textual description clarifying the precise order of execution between the sparse processing and the semi processing during the SPWOOD dataset generation?

3. Given that the point annotation scenario provides the most minimal supervision, have the authors considered integrating the knowledge or capabilities of large vision foundation models (e.g., SAM) in future work?

4. To enhance the paper's academic completeness and facilitate future research, the authors are requested to incorporate a dedicated discussion about inherent constraints of the SPWOOD framework and promising avenues for future development.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SPWOOD, a Sparse Partial Weakly-supervised Oriented Object Detection framework that reduces annotation costs by jointly leveraging sparse, weakly annotated, and unlabeled data. The method introduces three main components: a Sparse-annotation-Orientation-and-Scale-aware Student (SOS-Student) to learn orientation and scale from weak annotations, a Multi-level Pseudo-label Filtering (MPF) strategy to dynamically select reliable pseudo-labels, and an Overall Sparse Method ensuring balanced category sampling. Overall, the paper is clearly written and easy to follow. However, the related work discussion and experimental scope could be more comprehensive, and the novelty relative to closely related sparse or partial weak supervision literature needs clearer differentiation.

### Strengths
1.The paper investigates a new and practical Sparse Partial Weakly-supervised Oriented Object Detection problem that reduces annotation effort in remote sensing.

2.The proposed SOS-Student effectively integrates orientation and scale learning under sparse and weak supervision. 

3.The writing and structure are clear, logical, and easy to follow, with well-presented figures and tables. 

4.The framework’s generality across multiple annotation types (RBox, HBox, Point) increases its potential practical significance.

### Weaknesses
1.The related work lacks discussion of similar sparse partial supervision paradigms explored in other domains, such as 3D detection (e.g., HINTED), which share comparable annotation reduction strategies and technical challenges.

2.The experimental validation is confined to DOTA datasets, limiting the evidence of generality; evaluation on other datasets or domains would strengthen the claims.

3.The baseline selection is relatively weak, with no comparison to stronger or more recent sparse partial supervision approaches in general object detection.

4.The paper lacks qualitative failure analysis, which would help identify common error patterns and highlight the limitations of SPWOOD under difficult or ambiguous scenes.

5.The claim of being the “first” to propose this setting seems overstated, as the method mainly adapts existing ideas from related supervision paradigms; broader validation, stronger baselines, and deeper technical discussion would make the contribution more convincing.

### Questions
1.How does SPWOOD perform on other datasets beyond DOTA to verify its generalization capability?

2.How do existing sparse partial supervision methods perform or adapt in oriented object detection, and how does SPWOOD differ from them technically?

3.Would including visual qualitative examples or failure cases help demonstrate how SPWOOD handles sparse or ambiguous annotations?

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
This paper introduces SPWOOD, a framework to tackle the novel and challenging task of Sparse Partial Weakly-Supervised Oriented Object Detection. This problem setting is highly relevant, especially for domains like remote sensing where annotation costs are prohibitive.

### Strengths
The primary contributions are: (1) defining this new, complex problem setting; (2) proposing a unified teacher-student framework that leverages sparse, weak (HBox/Point), and unlabeled data simultaneously; (3) introducing several components, including a Sparse-annotation-Orientation-and-Scale-aware Student (SOS-Student) and a Multi-level Pseudo-labels Filtering (MPF) mechanism. The experiments conducted on DOTA datasets demonstrate that the proposed framework achieves state-of-the-art performance, significantly outperforming methods from related sub-fields.

### Weaknesses
Despite the impressive results and the important problem formulation, I have several major concerns regarding the methodology's novelty and the paper's internal consistency.
1. Questionable Novelty of the Core Supervised Model (SOS-Student)
The paper presents the SOS-Student as a key innovation, but its core components for learning orientation and scale appear to be directly adopted from prior work.
•	Orientation Learning (Sec 3.2.2): This is explicitly stated to be a symmetry-aware approach from Yu et al. (2023).
•	Scale Learning (Sec 3.2.3): This adopts the spatial layout learning method from Yu et al. (2025a), using Gaussian overlap and Voronoi watershed losses.
While combining these techniques in a new framework is an engineering contribution, the paper presents them as foundational parts of its proposed model without clearly delineating its own novel algorithmic contribution. The main new component, Sparse Annotation Learning (Sec 3.2.1), is itself insufficiently described.
2. Lack of Crucial Implementation Details and Unverified Assumptions
The paper omits critical details for key methods, making them unreproducible and their effectiveness difficult to assess.
•	Sparse Annotation Learning: The proposed modification to Focal Loss hinges on an "adaptive factor $w$" to down-weight potential unlabeled objects. The paper provides no information on how $w$ is defined or computed. Is it a fixed hyperparameter, a function of the prediction confidence $p_t$, or something else? This is a crucial detail that is entirely missing.
•	Multi-level Pseudo-labels Filtering (MPF): This method is based on the strong assumption that the teacher's prediction scores at each FPN level can be effectively modeled by a Gaussian Mixture Model (GMM). The paper provides no empirical validation (e.g., visualizations of score distributions) to support this claim. If the distributions are heavily skewed or non-Gaussian, the GMM-based filtering may not be robust.
3. Significant Contradiction in the Experimental Design and Claimed Contributions
This is my most serious concern. The authors propose a new "Overall Sparse Method" for dataset creation as a contribution to overcome the bias of the existing "Single Sparse Method" (Lines 100-104, 375-377).
However, the paper's own ablation study in Table 5 clearly shows that their proposed "Overall Sparse Method" is consistently and significantly inferior to the "Single Sparse Method" across all settings. More problematically, the authors then abandon their own proposed method and use the superior-performing "Single Sparse Method" for all main comparison experiments (Table 1, 2), citing "a fair comparison with prior studies" (Line 394).
This creates a major logical flaw: a method introduced as a contribution is empirically proven to be worse and is subsequently discarded in the main experiments. A contribution that fails its own evaluation cannot be claimed as a strength of the paper and undermines the work's scientific rigor.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a novel and promising setting (Sparse Partial Weakly-Supervised ) that makes significant strides in addressing the high annotation costs associated with oriented object detection in remote sensing imagery. Its innovations, methodology, and experimental results demonstrate considerable potential.

### Strengths
1. The paper introduces a novel Sparse Partial Weakly-supervised Oriented Object Detection (SPWOOD) framework designed to significantly reduce annotation costs for oriented object detection in remote sensing. This task setting is more challenging and can further reduce annotation costs.

2. SPWOOD effectively leverages sparsely annotated data, weakly annotated data, and a large volume of unlabeled data, supporting various annotation formats (RBox, HBox, Point) or their combinations. This flexibility is a significant advantage, allowing it to adapt to diverse annotation resources.

3. The author employs a “simplify and divide” approach to transform a seemingly intractable, contradictory learning task into a series of subproblems that can be tackled through technical means.

4. The paper is well-structured.

### Weaknesses
1. While attempting to classify different methods and their cost-effectiveness, it is overly dense. 

2. The paper claims the Overall Sparse Method addresses the bias of the Single Sparse Method. However, Table 6 on page 9 shows that the Single Sparse Method sometimes yields higher annotation counts and AP performance for several categories, which appears contradictory to the paper's claims. While the paper attempts to explain this by stating the Single Sparse Method "tends to retain more annotations for categories with initially scarce labels," the explanation is not deep enough.

3. The description of the Multi-level Pseudo-label Filtering (MPF) implementation details is not sufficiently thorough. For example, specific details on the initialization and update mechanisms for GMM parameters (mean, variance, weights) are missing.

4. While the ablation study demonstrates MPF's superiority over CPF, a more detailed analysis of the individual contributions of different components within SPWOOD (e.g., orientation learning, scale learning, sparse annotation learning within SOS-Student) would be beneficial.

### Questions
1. I would like to know whether the symmetry assumption in directional learning limits the generalisation ability of the method.

2. The paper demonstrates SPWOOD's capability with a combination of annotation types (R:H:P=1:1:1 in Table 1). This is a realistic and strong setting. However, how does the framework's performance vary with different proportions of weak labels? 

3. The ablation study on sparse data creation methods (Table 5 & 6) presents a very interesting, albeit counter-intuitive, finding: the "Single Sparse Method," which introduces sampling bias by over-representing rare categories, outperforms the more distributionally faithful "Overall Sparse Method." The paper attributes this to the raw annotation count. Could this also indicate that the model is simply overfitting to these rare classes, and its performance on common classes might be suffering? Furthermore, what are the practical implications of this finding?

4.  Is the performance gain of SPWOOD significant enough to justify its potential increase in computational cost, particularly for resource-constrained research environments?

### Soundness
3

### Presentation
3

### Contribution
3
