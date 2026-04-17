# A Data-driven Typology of Vision Models from Integrated Representational Metrics

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Large vision models differ widely in architecture and training paradigm, yet we lack principled methods to determine which aspects of their representations are shared across families and which reflect distinctive computational strategies. We leverage a suite of representational similarity metrics, each capturing a different facet—geometry, unit tuning, or linear decodability—and assess family separability using multiple complementary measures. Metrics preserving geometry or tuning (e.g., RSA, Soft Matching) yield strong family discrimination, whereas flexible mappings such as Linear Predictivity show weaker separation. These findings indicate that geometry and tuning carry family-specific signatures, while linearly decodable information is more broadly shared.
To integrate these complementary facets, we adapt Similarity Network Fusion (SNF), a method inspired by multi-omics integration. SNF achieves substantially sharper family separation than any individual metric and produces robust composite signatures. Clustering of the fused similarity matrix recovers both expected and surprising patterns: supervised ResNets and ViTs form distinct clusters, yet all self-supervised models group together across architectural boundaries. Hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders, suggesting convergence between architectural modernization and reconstruction-based training.
This biology-inspired framework provides a principled typology of vision models, showing that emergent computational strategies—shaped jointly by architecture and training objective—define representational structure beyond surface design categories.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a comparison of ImageNet-pretrained computer vision models using multiple representational similarity metrics (RSA, CKA, SoftMatch, etc.) and adapts Similarity Network Fusion (SNF) from multi-omics biology to integrate these metrics. Every representational similarity metric has its own characteristics and invariances, making similarity assessments metric-dependent. To evaluate metric differences, the authors group 35 vision models into four primary "model-families" (supervised CNNs, self-supervised CNNs, supervised Vision Transformers, self-supervised Vision Transformers) plus two CNN/ViT hybrids (ConvNeXt, Swin). They use model-family separability as a criterion to select the best metric, finding that geometry/tuning-preserving metrics discriminate better between families than metrics measuring "linearly accessible information" (metrics learning an affine transformation between the representations). SNF achieves the strongest separation by integrating multiple metrics. The authors then apply hierarchical clustering to the SNF-fused similarity matrix to discover a "data-driven typology," which mostly recovers the pre-defined families. The typology reveals that self-supervised models cluster together across architectural boundaries, and hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders (MAE).

### Strengths
1. The adaptation of Similarity Network Fusion (SNF) to integrate multiple representational similarity metrics is methodologically sound and addresses the limitation that individual metrics capture only partial aspects of representational structure. SNF achieves stronger family separation than any single metric alone, demonstrating the value of integration. This framework could be extended in interesting directions not explored in the paper, such as integrating local and global representational similarity (e.g., using different CKA kernels) or combining layer-wise and network-wide comparisons.

2. Cross-dataset robustness checks: Testing the typology across four datasets and showing consistent patterns (Figures 3, C.1-C.3) strengthens confidence that the findings reflect genuine representational properties rather than dataset-specific artifacts.

### Weaknesses
- Major
    1. Circular reasoning / "test-set tuning" problem: As far as I understand, the most critical point is using pre-defined model-family labels to evaluate which metrics best separate these families (Section 2.3, Figure 2), then showing that the integrated metric SNF is “winning “in this aspect and then using SNF to claim "data-driven" discovery of clusters that largely match the original labels. This is analogous to selecting a model based on test set performance. 
    2. Limited validation of linear accessible information interpretation: The authors interpret weak discrimination by  linear accessible information metrics as evidence that models "converge on similar linearly accessible information" (L320-323). However, these metrics (Linear Predictivity, CCA variants) quantify similarity via correlation between the target representation and its linear reconstruction. Correlation may be insufficiently sensitive—high correlation can occur even when reconstructions are poor in absolute terms, potentially inflating similarity scores. The paper would benefit from: (a) complementary reconstruction quality metrics (R², mean squared error, variance explained) to verify that high correlation reflects genuine similarity rather than metric leniency, and/or (b) downstream task validation (linear probing, transfer learning) to test whether metric-based similarity predicts functional similarity. Without these checks, it's unclear whether weak discrimination reflects genuine convergence in linearly accessible information or limitations of correlation-based evaluation that fails to detect meaningful representational differences
- Minor
    1. Limited model scope undermines relevance: The authors restricted themselves to ImageNet-1k pretrained models to avoid training data domain to be a influencing factor on the representational similarity values. However,  restricting to ImageNet-1k pretraining excludes modern web-scale and multimodal encoders (CLIP, DINOv2, SigLIP, EVA-CLIP) that dominate current practice. The typology characterizes legacy models but may not generalize to the contemporary landscape. Self-supervised model diversity is particularly limited (mostly ResNet-50 variants). The claimed "data-driven typology" framework's value depends on its applicability to current models.
    2.  Inconsistent terminology and poor figure design:
         - Terms self-supervised vs unsupervised (label of the methods) used inconsistently. 
         - It would have been easier to follow the information/ observations if the authors had thought of a method to aggregate the values in Figure 2 and putting the correct table 
         - Figure 4's different orderings across subplots make cross-metric comparison impossible, and Figures 4-5 are difficult to read with labels too small. 
         - Figure 2 should clearly indicate which metrics measure which facets (geometry/tuning/linear-accessibility) or alternatively using a table.
         - Unmotivated metric fusion: Section 2.4 jumps directly to SNF implementation without justifying why metric integration is needed. What problem does integration solve that single metrics don't? The paper would benefit from explicit motivation (e.g., showing that single metrics give contradictory groupings, or that different metrics capture complementary information that's individually incomplete).

### Questions
- Circular reasoning: How do you address the concern that using family labels to select the best metric (Section 2.3), then using that metric to discover families, constitutes circular reasoning?
- Functional validation: Can you provide downstream task performance (linear probing, transfer learning) to validate whether weak linear-accessibility discrimination reflects genuine convergence or just methodological artifacts? *This could be a lot of work*. In case limited time, could the authors simply comment on this concern?
- Pooling choice: Why use average pooling instead of the tokens models were trained with (e.g., CLS for DINO)? Did you test sensitivity to this choice?
- Modern models: Would the authors expect typology change if including web-scale models (CLIP, DINOv2)? Do you expect family boundaries to remain stable?
- Prior work: How does your approach differ from prior model clustering work (Kornblith 2019, Ding 2021, Rosenfeld 2021)? Is SNF integration the “only” novelty?
- Figure 4 patterns: Why do VGG models show less within-family similarity than ResNets in most metrics, despite both being supervised CNNs?

### Soundness
2

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
3

### Summary
The paper takes inspiration from the field of biology to build a taxonomy of vision models based on their similarity with respect to the various metrics, similar to how taxonomy of biological organisms are build based on various genetic, phenotypical and morphological similarities. The authors posit that such a taxonomy of model can reveal similarities in learned representation that can transcend architectural and training objective differences. Thirty-five vision models were categorized into six categories based on training style and architecture, and their similarity computed with several metrics. An approach of combining the metrics that directly mirrors approaches used in biology has been presented.

### Strengths
1.	Biology inspired approach for hierarchically clustering vision models similar to building a taxonomy of biological species.
2.	Hierarchical clustering to produces a typology that transcends architecture labels. For example, the finding that self-supervised CNNs cluster with self-supervised ViTs rather than with their supervised CNN counterparts shows that training objective can take precedence over the architecture in determining the learned representation. 
3.	Extensive experiment covering wide variety of vision models and similarity metrics.

### Weaknesses
1.	No discussion on how to analyze SNF based taxonomy. How can a researcher know which aspect of two models are contributing to similarity without looking at individual metrics and just from SNF? If I need to look at individual metrics, than what is the need of SNF? For a researcher who analyzes the reason for the spurious similarity between MAE trained ViT and fully supervised ConvNext, what kind of lead does SNF provide, beyond just telling that there is some similarity?
2.	SNF’s similarity score is not interpretable to understand how each individual metric is contributing to the overall score. This also relates to the previous point of how a researcher can make use of the score for further analysis.

### Questions
Please refer to the weaknesses section

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
The authors propose the use of existing representation similarity metrics to compare the representations generated by various vision models. Furthermore, they propose Similarity Network Fusion (SNF), which combines existing metrics in a way that improves consistency and discriminative power.

### Strengths
•	The main claim of the paper, that similarity metrics (and specifically SNF) display interesting clustering patterns of architectures and training paradigms, is supported well. The experiments demonstrate higher separation between families with SNF compared to other metrics, and qualitatively SNF seems to be the best metric.
•	The authors clearly justify the gap that the proposed method is filling. Existing methods are inconsistent or fail to consider multiple types of representation structure, and the proposed method addresses these problems directly. 
•	The writing is largely very clear and direct, with little confusion in most of the paper. /
•	With proper justification of its importance, the proposed method may provide a notable improvement over existing methods.

### Weaknesses
•	While the proposed work seems to glean interesting insights into model representations, it is not clear what the broader impact of this work is. Little time is spent justifying the importance of the method or possible applications. Most taxonomies in other fields (biology, linguistics, etc.) are an organizing means to an end to understand something about the underlying domain that the organization structure reveals, such as whether language similarity follows geographic proximity or evolutionary mechanisms and functional traits in biology. In the case of the proposed work, it is not clear what the taxonomy reveals (or would reveal) about the models’ assumptions, limitations, and applicability. Either theoretically or experimentally, showing the taxonomy improves model selection and performance for a task would better demonstrate its impact.
•	Unclear communication in critical parts of the paper, such as the methods section or when discussing the merits of the proposed method, hinder the possible impact of the paper. 
      o	While the work overall is clear and straightforward, the presentation is hurt by some specific lack of clarity and sloppiness, particularly in figures and methods. 
      o	Figure 3 has inconsistent y-axis scales, obscuring the fact that the separability metrics have very different values between datasets. 
      o	Other figures are referenced with somewhat misleading descriptions, such as Figure F1. This figure is used to argue that the proposed method has some of the best performance on the CCC metric, but this is a stronger claim than the figure implies. Several metrics have similar performance, and the proposed method is not the top performing method in any of the given categories. 
      o	In the methods section, the equations detailing the implementation of SNF are not described in detail, making it more difficult to understand the proposed methodology. Additionally, critical parts of experimental design are omitted from the main manuscript, such as how representations are generated for varying types of models.

### Questions
-	Figure F.1 seems to imply that several methods (specifically RSA and SoftMatch) have high cophenetic correlation coefficients across all four datasets. What does this imply about the proposed method? What does this imply about existing methods? Is this metric a good measure of the usefulness of a metric?
-	When discussing separation ability metrics, there is a specific focus on model “families”. What defines a family in this case: training paradigm, architecture, both, or neither? 
-	In the paper, there is a brief mention of possible uses for such metrics. Are there cases when similar metrics have been used to analyze real-world behavior in contrast with metric similarity? Are there cases where similar metrics have been applied in other ways?

### Soundness
3

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
3

### Summary
The authors look at different metrics to quantify representational similarity between computer vision models. In particular Singular Vector Canonical Correlation Analysis (SVCCA), Projection-Weighted Canonical Correlation Analysis (PWCCA), Linear CKA, RSA, Soft Matching, Procrustes Alignment and Linear Predictivity are all computed and then combined with Similarity Network Fusion, a method inspired by multi-omics integration. The study clusters a range of 35 different vision models spanning different architectures (CNN vs ViTs) and objectives (SSL vs. supervised). By using various separation/clustering metrics, the authors compare their SNF metric with simple averaging of the metrics and each metric individually. The SNF approach recovers best the grouping by families and clusters SSL models together putting more emphasis on the objective in this case rather than the architecture.

### Strengths
* The paper studies an important question and applying Similarity Network Fusion to combining multiple representational metrics is novel as far as I can tell.
* The approach is clearly described and the paper is well motivated.
* The study investigates a broad range of different representational similarity metrics.

### Weaknesses
* The overall diversity of models considered for the analysis is quite limited. The analysis misses image/text models which are a widely used category of vision models and would add a fundamental new objective to the analysis. Further, only SSL models which have been trained on ImageNet-1k have been considered. Given the finding that SSL models cluster together over different architectures in the paper, it would be particularly interesting to see whether this also holds over different pretraining datasets. There are multiple publicly available SSL models which have been trained on other datasets (e.g. [1, 2]) which could be used in this study.

* The paper only considers natural image datasets: ImageNet-1k, CIFAR10, CIFAR100, Ecoset for computing similarities. Previous work [3, 4] has shown that similarities between models depend on the dataset which is used to compute the representations. A thorough investigation of representational similarity should go beyond ImageNet and also consider e.g. fine-grained and more structured datasets.

* There is not really a clear ground truth structure of which models should be closer together than others. On the one hand the paper is defining some families and evaluating the clustering based on this definition. But on the other hand, it is also noted that SSL models cluster over architecture families, basically asking the question whether we can even benchmark the right organization of similarity. I think the “desired similarity” should be more clearly described in this paper and the problem of benchmarking it further discussed.

[1] Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." TMLR 2024.

[2] Goyal, Priya, et al. "Self-supervised pretraining of visual features in the wild." arXiv preprint arXiv:2103.01988 (2021).

[3] Cui, Tianyu, et al. "Deconfounded representation similarity for comparison of neural networks." Advances in Neural Information Processing Systems 35 (2022): 19138-19151.

[4] Ciernik, Laure, et al. "Objective drives the consistency of representational similarity across datasets." ICML 2025.

### Questions
* What do the colors in the dendrogram in Figure 5 represent?

* Were the family definitions in Appendix B used to compute the separation ability metrics or how were these exactly computed?

### Soundness
3

### Presentation
3

### Contribution
2
