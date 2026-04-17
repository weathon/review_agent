# Learning Label-Efficient Interpretable Medical Image Diagnosis via Semi-supervised Hypergraph Concept Bottleneck Model

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Deep learning has revolutionized medical image analysis, delivering exceptional diagnostic accuracy across diverse applications. 
Yet, the lack of interpretability in its decision-making hinders clinical adoption, particularly in high-stakes medical contexts where transparency is paramount for trustworthiness. 
For example, in Placenta Accreta Spectrum (PAS), subtle cues in ultrasound imaging challenge reliable diagnosis, rendering black-box models untrustworthy for accurate scoring.
To address this, Concept Bottleneck Models (CBMs) offer a promising avenue by embedding clinically meaningful intermediate concepts into the diagnosis pipeline, enabling clinicians to scrutinize and refine model outputs. 
However, conventional CBMs falter in capturing complex inter-concept dependencies and demand costly, expert-driven concept annotations, limiting their scalability. 
This study introduces a novel semi-supervised CBM framework designed for medical imaging, which leverages dual-level hypergraph learning to model high-order concept dependencies and generate domain-adaptive pseudo-labels.
Our approach achieves superior interpretability and performance by integrating a concept-level hypergraph for enhanced reasoning and an image-level hypergraph for robust pseudo-label generation. 
Experiments on a newly annotated PAS ultrasound dataset and a breast ultrasound public dataset demonstrate the effectiveness of the proposed concept label-efficient interpretable framework.
Its universality is further validated on the dermoscopic image dataset SkinCon.
The core code is available at the appendix.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the lacking of inter-concepts relationship modeling and low label efficiency in traditional Concept Bottleneck Models (CBMs) for the medical domain by introducing a Hypergraph-based Semi-Supervised Concept Bottleneck Model (HyperCBM). The framework integrates Hypergraph-Enhanced Concept Representation Learning (HECRL) to capture high-order inter-concept relationships through adaptive hypergraph propagation, and Hypergraph Image Dynamic Pseudo-labeling (HIDP) to generate reliable pseudo-labels from unlabeled data. The resulting model achieves improved classification accuracy and concept accuracy across multiple medical imaging datasets.

### Strengths
1. The paper presents a creative and well-motivated integration of hypergraph structures into the CBM paradigm. This removes the assumption in existing CBMs that concepts are independent
2. The introduction of HIDP provides a principled mechanism to leverage unlabeled medical images.
3. Empirically, the method achieves consistent and noticable improvement over baseline methods

### Weaknesses
1. For the Interpretability Visualization, it is not clear to the people outside of the medical domain what the correct concept heatmap should look like; thus, it is impossible to evaluate the quality of the interpretability. The heatmap also appears to be fairly consistent across concepts, which makes me skeptical about its accuracy.
2. Computation cost: Introducing graph construction for each image would cause significant computational overhead, thereby hinder the scalability of the method. The paper would benefit from a clearer discussion or quantitative analysis of computational cost and scalability.

### Questions
1. Could you provide more visualization of the concept saliency maps with the ground-truth heatmap it should be for reference?
2. Sparse Autoencoder (SAE) is also capable of locating concepts and requires no concept labeling. What's the advantage of your approach over SAE?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HyperCBM, a new semi-supervised CBM framework aimed at improving interpretability and label efficiency for medical image diagnosis. The model leverages a dual-level hypergraph architecture: a concept-level hypergraph for modeling high-order inter-concept dependencies (HECRL) and an image-level hypergraph for generating domain-adaptive pseudo-labels (HIDP). While the technical contribution is novel and the empirical results against the primary baseline (SSCBM) are strong, the comparative evaluation against a broader range of advanced CBM literature is insufficient. This omission prevents a full assessment of the value added by the hypergraph structure.

### Strengths
1. The paper targets two highly relevant and significant challenges in the CBM field, particularly in high-stakes medical applications: the high cost of expert concept annotation (label efficiency) and the necessity of modeling non-linear, high-order dependencies between complex clinical concepts. This problem formulation is timely and important for the CBM community.

2. The core technical contribution, HyperCBM, is innovative by integrating dual hypergraph components (HECRL for inter-concept modeling and HIDP for domain-adaptive pseudo-labeling). This is shown to provide an efficient way to generate pseudo labels under semi-supervised settings.

3. The proposed Placenta Accreta Spectrum (PAS) ultrasound dataset with explicit clinical concepts contributes to future research in medical CBMs.

### Weaknesses
The comparative evaluation is a major weakness. 

1. While the paper includes SSCBM and other SSL methods, it fails to benchmark against several advanced CBM variants that also explicitly address complex concept dependencies or label scarcity.
   1. High-order concept dependencies modeling: works including ProbCBM (ICML 2023), Relational CBM (NeurIPS 2024), Energy-based CBM (ICLR 2024), Graph CBM (arxiv 2025) already explore many different strategies to capture more complex concept dependencies. Although some of the work are discussed and cited in the related work, no more rigorous evaluations or comparisons are provide against these advanced architectures.
   2. Reducing labeling cost: although few approaches explore the semi-supervised setting for CBMs, there are many methods for constructing CBMs with only image-level labels, such as LF-CBM (ICLR 2023), Post-hoc CBM (ICLR 2023), LaBo (CVPR 2023),  V2C-CBM (AAAI 2025), DOT-CBM (CVPR 2025), .... And some of the core idea or framework are already adapted/improved for medical settings such as AdaCBM (MICCAI 2025), MVP-CBM (IJCAI 2025), and Multimodal Medical Concept Bottleneck Model (MMCBM, Nature Communications volume 16, Article number: 3504 (2025)). The efficiency of the proposed semi-supervised methods is supposed to compare against advanced concept label-free methods.
2. Incomplete concept settings: beyond labeling cost, another challenge for adopting CBMs in clinical settings is concept incompleteness, which is also the main motivation of CEM. An evaluation for the proposed framework under such setting is missing.
3. The test-time intervention ability is the core essentials of CBMs, a more standard concept intervention procedure as those done in CBM and CEM, instead of applying a series of confidence thresholds, are necessary. The current conclusion from Figure 4 seems to be universal because less activations may directly lead to worse results.

### Questions
First, please refer to the weaknesses section. Second, regarding my additional question: Are there visualization or quantitative results, beyond a simple comparison of accuracy and AUC, that demonstrate the proposed method's ability to better model the essential inter-concept relationships inherent in medical imaging and critical for holistic reasoning? Alternatively, are there examples that illustrate the specific requirements of these inherent relationships that are not present in natural image Concept-Based Model (CBM) benchmarks? If not, wouldn't it be more solid to evaluate the proposed mechanism within established natural image CBM benchmarks?"

### Soundness
2

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
3

### Summary
This paper proposes HyperCBM, a semi-supervised hypergraph-driven concept bottleneck model for interpretable medical image diagnosis.   The approach aims to reduce dependence on expensive expert-level concept annotations while maintaining both diagnostic accuracy and interpretability.  

HyperCBM introduces a dual-level hypergraph framework that captures semantic dependencies among medical concepts and contextual relations across images.  
Experiments on three datasets (a newly collected PAS ultrasound dataset, the public BrEaST dataset, and SkinCon) show that HyperCBM achieves competitive or superior performance compared to fully supervised CBM and CEM models.

Overall, the paper is technically interesting and relevant, but its experimental validation is insufficient to support some of its stronger claims. The core idea of hypergraph-driven semi-supervised CBM is promising, yet the presentation could be strengthened by clearer analysis of computational cost, hyperparameter stability, and statistical significance. The empirical rigor and clarity fall short of acceptance standards at this stage.

### Strengths
**1. Strong motivation and originality**  
The paper addresses a critical challenge in medical imaging—balancing interpretability and annotation cost.  
By introducing a semi-supervised framework to reduce reliance on expert concept labels, it provides a practical and meaningful direction for interpretable medical AI.

**2. Methodological innovation**  
The proposed dual-level hypergraph design is conceptually novel:  
- The concept-level hypergraph (HECRL) models high-order semantic dependencies among medical concepts.  
- The image-level hypergraph (HIDP) generates domain-adaptive pseudo-labels, enhancing representation consistency across labeled and unlabeled data.  

This design effectively bridges semantic and visual reasoning within a unified framework.

**3. Data contribution and reproducibility**  
The newly introduced PAS ultrasound dataset, constructed through a hybrid pipeline combining medical vision–language models with expert validation, contributes valuable resources to the community.

### Weaknesses
**1. High methodological complexity**  
The overall pipeline involves multiple interdependent modules—including two hypergraph constructions, attention-weighting mechanisms, pseudo-label generation, and weight-sharing strategies—which may hinder implementation and reproducibility.  
The number of hyperparameters (λ₁, λ₂, kₘᵢₙ, ϵ, initial_ratio) is large, and results in Table 5 suggest sensitivity, raising questions about generalizability across datasets.

**2. Computational overhead and scalability**  
The paper does not analyze the computational cost of hypergraph construction.   Some quantitative comparison of runtime or memory consumption against CBM/CEM baselines would help assess scalability and deployment feasibility.

**3. Limited performance gains and statistical validation**  
While HyperCBM performs well overall, its improvements on certain metrics (e.g., ACC and AUC on the SkinCon dataset, Table 1) are marginal.  
Reporting statistical tests (e.g., t-tests or confidence intervals) would strengthen claims of significance.

**4. Minor presentation issue**  
In Figure 2, the color legend for labeled vs. unlabeled data appears inconsistent with the data flow arrows (red/blue switched).  
Clarifying this would improve figure readability and prevent confusion.

### Questions
## **Major Concerns**

**1. Computational Cost and Scalability**  
The method introduces dual hypergraphs and additional training components, which may increase training time and memory usage relative to standard CBM/CEM.

> **Questions:**  
> - What is the relative training time or memory footprint compared to CBM or CEM under the same hardware and batch size?  Can this approach scale to larger datasets?

---

**2. Statistical Significance and Stability**  
Some reported gains appear modest (e.g., ~1–2% AUC improvement on SkinCon in Tables 1–2). Without significance testing or variance reporting, it is difficult to assess robustness.

> **Questions:**  
> - Have the authors conducted statistical tests (e.g., t-test, bootstrap confidence intervals) across multiple runs?  
> - If performance is highly sensitive to hyperparameters, does this imply weaker stability across datasets and seeds?

---

**3. Hyperparameter Sensitivity and Generalization**  
Table 5 suggests strong sensitivity to λ₂ (e.g., at 5% label ratio, increasing λ₂ from 0.05 to 0.1 raises Class ACC from 57.48% to 72.15%). Optimal (λ₁, λ₂) also varies notably across datasets (BrEaST: 0.5,0.1; PAS: 1.0,0.1; SkinCon: 2.0,1.0).

> **Questions:**  
> - Does the pronounced sensitivity to λ₂ indicate limited generalization?  
> - How can one efficiently select λ₁ and λ₂ on a new dataset?  
> - Is there a principled rationale for the dataset-specific optima, or any pattern linking data properties (label ratio, concept sparsity, class imbalance) to recommended ranges?

---

**4. Freeze–Sync Design Rationale**  
The paper states that the concept encoder and unlabeled image encoder share weights, while the unlabeled encoder remains frozen during training and is synchronized with the concept encoder at the end of each epoch (p.7).

> **Questions:**  
> - What is the theoretical or empirical justification for freezing the unlabeled encoder during the epoch instead of joint training?  
> - Under what conditions does freeze–sync improve stability or prevent collapse compared with end-to-end optimization?

### Soundness
3

### Presentation
3

### Contribution
2
