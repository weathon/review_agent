# What is Important? Internal Interpretability of Models Processing Data with Inherent Structure

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
This paper introduces a methodology for constructing interpretable neural networks that quantify the importance of structured input components directly within their internal mechanisms, thereby eliminating the need for traditional explanation methods that rely on post-hoc saliency map generation. Our approach features a two-stage training procedure. First, component specific representations and importance scores are discovered using appropriately designed convolutional neural networks, which are trained jointly. Second, an architecture with relaxed structural constraints, leveraging the previously acquired knowledge, is fine-tuned to capture spatial dependencies among components and to integrate global context. We systematically evaluate our method on Oxford Pets, Stanford Cars, CUB-200, Imagenette, and ImageNet, measuring interpretability-performance trade-offs with metrics for semanticity, sparsity, reproducibility, and, when required, causality (via insertion/deletion-inspired scores). Our architecture achieves improved semantic alignment with ground-truth segmentation annotations compared to post-hoc saliency maps, which, when available, serve as surrogates for expected saliency maps. At the same time, it maintains low variance in importance scores across runs, demonstrating strong reproducibility. Crucially, our architecture provides interpretability gains without sacrificing accuracy. In fact, both with non-pretrained and pretrained backbones, it frequently achieves higher predictive performance than parameter-matched baselines. Overall, compared to both conventional models and post-hoc interpretability techniques under matched computational budgets, our framework produces models that are accurate, stable, and that deliver causally grounded explanations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a two-stage inherently interpretable image classification framework centered on *learned importance scores*.
 (1) IA (Importance Architecture): the image is divided into non-overlapping patches, each encoded independently into embeddings (E), and an auxiliary ImportanceNet predicts an importance weight (a \in [0,1]) for each patch; the classifier aggregates the embeddings weighted by (a).
 (2) EA (Embedding Architecture): the learned (a) is frozen, and a light-weight ContextNet integrates contextual dependencies over the weighted embeddings.
 (3) PA (Pixel Architecture): the same importance mask (a) is applied at the pixel level before feeding into a standard backbone classifier.

The experiments evaluate four main aspects: semantic alignment (IoU and a custom distance metric (d)), sparsity, reproducibility, and causality tests (insertion/deletion analysis on EA). Several datasets are tested, and accuracy curves and tables are reported.

### Strengths
- The proposed modules make minimal architectural modifications while enforcing importance estimation *within* the inference path (i.e., not a post-hoc explanation).
- The authors visualize patch-level importance heatmaps across different patch and embedding scales (Fig. 3) and show masked images with different thresholds (t) (Figs. 6–9).

### Weaknesses
#### **1. Overstated novelty; limited comparison scope (biased evaluation)**

- The paper’s *Related Work* and experimental comparisons focus almost exclusively on post-hoc saliency methods (Grad-CAM, IG, SHAP, Occlusion, etc.), claiming superior semantic alignment. However, many representative in-model / inherently interpretable approaches (e.g., prototype-based, tree-structured, alignment- or concept-based models) are neither discussed nor compared. This omission systematically inflates the perceived novelty and contribution.
- Structurally, the proposed method is essentially a gating/masking mechanism followed by lightweight context integration—the paper itself states that EA fuses contextual information over the reweighted embeddings $E'$ computed by applying $a$ to $E$ .  Such “mask + context” architectures are already common in prior inherently interpretable works.
   Without fair comparison against strong in-model baselines, the claimed originality and advantage are not convincing.

------

#### **2. Simplicity and overlap with existing gating/masking paradigms**

- The IA’s main modification is a separate ImportanceNet that outputs patch-wise weights $a$, which are then applied as element-wise gates before aggregation.
   EA merely freezes $a$ and adds a shallow CNN for contextual fusion.
- Conceptually, this is equivalent to a soft attention or gating layer without additional theoretical constraint or empirical justification showing superiority over numerous prior *learned mask / attention-based interpretability* models.
   As such, the methodological novelty is limited for an ICLR-level contribution.

------

#### **3. Metric and baseline comparability issues (Sections 4.1 & 4.2)**

##### **3-a. Inconsistency in the (d) metric computation**

- The proposed $d$ metric is defined on patch-level importance values $a$ and patch-averaged semantic references (m) (L1 distance). Yet most post-hoc methods produce pixel-level saliency maps. The paper does not clearly state how these pixel-level maps were converted to patch-level scores (mean? max? normalization?). It also omits whether the same normalization and binarization thresholds were used across methods. Without consistent scaling and thresholding, cross-method comparisons on $d$  are not rigorous.
- Furthermore, the semantic mask (m) itself may be a weak reference: important patches often contain a mix of object and background pixels (e.g., a cat ear patch with more background but higher discriminative value than a belly patch).
   Hence, the (m)-based ground-truth importance may not faithfully represent semantic relevance, compromising the validity of $d$ .

##### **3-b. Incomplete accuracy baselines (Section 4.2)**

The classification accuracy comparison includes only the authors’ three variants (IA, EA, PA). This is insufficient to claim that “our interpretable mechanism preserves or improves predictive performance.” Also, clarify the hyperparameter protocol—e.g., was the PA threshold $t$ tuned only on the validation set and fixed for test reporting?

------

#### **4. Poor writing and presentation quality**

- The manuscript is difficult to read and loosely organized; key experimental details are scattered across sections.
- Most figures are non-vector graphics with tiny fonts, making them hard to interpret. Notably, Figure 5 (“Kernel density estimate plots…”) appears but is never referenced or discussed in the main text, which undermines clarity and professionalism.

### Questions
**Details of $d$ Computation:**
 How are the pixel-level heatmaps from different baselines unified into the patch-level $a$? Was a consistent normalization and thresholding strategy applied across all methods?

**Threshold Selection for PA:**
 In Table 2 and Figures 7–9, do the “best results” correspond to the threshold $t$ selected *only on the validation set* and then reported *once* on the test set?

**Missing Strong Baselines:**
 Why are strong *intrinsically interpretable* methods such as ProtoPNet, ProtoTree, NBDT, and B-cos not included for end-to-end comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for inherently interpretable neural networks that directly encode and quantify the importance of structured input components—such as regions in an image, tokens in a sequence, or nodes in a graph—within the model’s internal computations. The proposed approach integrates explainability through a two-stage methodology. In the first stage, convolutional networks jointly learn component representations and importance scores. In the second, a refined model relaxes structural constraints to capture spatial and contextual dependencies while preserving interpretability anchors.

### Strengths
The two-stage procedure elegantly separates the discovery of component importance from the modeling of global dependencies, offering clear interpretability anchors and facilitating adaptation to other data types (images, sequences, graphs).

Comprehensive experiments on multiple benchmark datasets show consistent improvements and predictive accuracy compared to state-of‑the‑art baselines.

### Weaknesses
Lack of novelty. Your approach analyzes data at the patch level, which seems conceptually similar to ViT‑Shapley[1]. Could you clarify what advantages your method offers compared to ViT‑Shapley, or what specific motivation drives your work beyond that prior approach?

In ViT‑Shapley, the fairness of patch‑level importance assessment is ensured through the use of Shapley value computations. How does your method guarantee comparable—or superior—fairness in evaluating the contribution of each visual patch?

Furthermore, regarding the metrics for interpretability: human studies are often considered an intuitive and widely accepted means of assessing interpretability. However, such evaluations appear absent from your paper. Could you explain the reasoning behind this choice or provide justification for omitting human studies?

[1]: Learning to Estimate Shapley Values with Vision Transformers.

### Questions
See Weaknesses section.

### Soundness
2

### Presentation
2

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
This paper proposes a method for building intrinsically interpretable neural networks that can directly measure the importance of structured input components within the model itself, eliminating the need for post-hoc explanation techniques such as saliency maps. The approach employs a two-stage training process:
1. A specialized CNN jointly learns component-specific representations and importance scores.
2. A refined architecture with relaxed structural constraints is then fine-tuned to capture spatial dependencies and global context.

The authors evaluate the method on multiple datasets, including Oxford Pets, Stanford Cars, CUB-200, Imagenette, and ImageNet, analyzing the interpretability–performance trade-off using metrics such as semanticity, sparsity, reproducibility, and causality.
Results show that the proposed architecture achieves better semantic alignment with ground-truth annotations and higher reproducibility than traditional post-hoc saliency methods. Moreover, it provides interpretability improvements without sacrificing accuracy—and often even exceeds the predictive performance of parameter-matched baselines, both with and without pretrained backbones.

### Strengths
They propose a novel method that directly quantifies the importance of structured input components within the model itself, eliminating the need for post-hoc explanation techniques. The method is extensively evaluated across multiple datasets, and the quantitative results demonstrate its clear superiority over existing approaches.

### Weaknesses
The baseline approaches used for comparison are relatively outdated. There exist more advanced methods beyond Grad-CAM that could provide a stronger and fairer evaluation, such as RISE (Petsiuk et al., 2018) and Shap-CAM (Zheng et al., 2022). Moreover, it is unclear why the authors only compare against Grad-CAM without including its improved variant, Grad-CAM++. Limiting comparisons to older methods weakens the validity of the claimed superiority of the proposed approach.
1. Vitali Petsiuk, Abir Das, and Kate Saenko. RISE: randomized input sampling for explanation of black-box models. In British Machine Vision Conference 2018, BMVC 2018, Northumbria University, Newcastle, UK, September 3-6, 2018, page 151, 2018.
2. Quan Zheng, Ziwei Wang, Jie Zhou, and Jiwen Lu. 2022. Shap-CAM: Visual Explanations for Convolutional Neural Networks Based on Shapley Value. In Computer Vision–ECCV 2022: 17th European Conference. Springer, Tel Aviv, Israel, 459–474

### Questions
Could the authors provide an analysis of the running time or computational cost? This information is important, especially if the method is intended for large-scale or repeated experiments. 

Additionally, it would be helpful to specify how many images were used for evaluation in each dataset.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose an interpretable neural network that reduces the reliance on post-hoc saliency methods. The explanation scores are learned jointly with the model parameters. The approach first quantifies the importance of individual components, after which the model learns spatial and contextual dependencies while preserving the discovered importance structure. The quality of the explanations is evaluated in terms of semanticity, sparsity, reproducibility, and causality.

### Strengths
- I believe the idea of quantifying component importance and preserving it is valuable, as it reduces unnecessary complexity and encourages the model to align with human-interpretable concepts.
- The methodology is tested on several datasets, demonstrating good interpretability insights.

### Weaknesses
- **Importance mask:** The proposed “importance mask” appears conceptually similar to an **attention mechanism**; the paper does not clearly articulate how it differs from standard attention-based interpretability.
- **The use of patches:** The claim that image patches are “semantically meaningful” is questionable—**patches are not inherently semantic units**, and true semantic meaning would require segmentation or context modeling. Moreover, the **patch-based decomposition** risks losing coherence when important concepts span multiple patches, potentially diluting concept-level importance and reducing structural interpretability.
- **Clarity of the diagram:** Figure 1 is unclear - IA, EA, and PA modules are not visually distinguished, the text is too small, and the relationships among components are ambiguous.
- **Clarity of concepts:** the interaction between **EA and PA** (whether they are trained jointly or separately) is not well explained, and the connection between the EA and its interpretability claims remains unclear.
- **Semantics:** The terminology of *“semantic structure of embeddings”* is misleading since embeddings correspond to **patches**, not true semantic entities. Moreover, measuring “semanticity” by classification accuracy on non-segmented datasets only reflects **alignment with model predictions**, not genuine semantic alignment. With respect to the metric semanticity, the paper seems to define semanticity with full object segmentation. How is semanticity defined when only part of an object is relevant? Is the goal closer to segmentation? This may explain why Grad-CAM performs well, as it typically highlights larger regions.
- **Causality metrics:** The paper’s **causality analysis** relies on insertion/deletion metrics, which do not capture causal dependencies among correlated features and are **not novel and should be cited [1].**
- **Experiments and discussion: Figure 7** contradicts the claim in Section 4.2: only CUB-200 maintains accuracy, while other datasets show degradation. The **insertion/deletion experiments lack baseline comparisons**, making it difficult to assess the actual effectiveness of the proposed method. The paper includes masked insertion/deletion images in the main text, but quantitative results are only reported in the supplementary material, and no baseline comparisons are provided.

[1] Covert, I., Lundberg, S., & Lee, S. I. (2021). Explaining by  removing: A unified framework for model explanation. Journal of Machine  Learning Research, 22(209), 1-90.

### Questions
- I included some questions above.

### Soundness
2

### Presentation
2

### Contribution
2
