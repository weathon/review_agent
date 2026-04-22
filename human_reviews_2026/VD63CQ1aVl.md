# Towards Unified Dynamic Face Landmark Detection

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Although advancements in face landmark detection (FLD) methods continue to push performance boundaries, they overlook two major functional limitations: (1) different network parameters need to be trained independently for each "N-point" benchmark dataset, and (2) a model trained on an "N-point" dataset reliably outputs only the N landmarks. In our work, we first conceptualize Face Part-Anchored Landmark Positions (FPALPs), wherein each landmark is treated as a progression value between zero (start) and one (end) along a face part's contour. Every landmark can be expressed in the FPALP format, irrespective of its source dataset, hence unlocking the ability to unify all "N-point" datasets into a single dataset. Secondly, we represent each landmark with an FPALP-based query, refine it progressively with a cross-modality decoder, and predict its coordinates based on the final representation. Our approach, called Unified Dynamic FLD, embodies these two design choices and streamlines the landmark detection pipeline by enabling (1) a single model to learn on any number of "N-point" datasets, and (2) yield any number of specific landmark predictions by loading the designated landmark queries at runtime. Extensive experiments carried out on several benchmark datasets demonstrate that our approach can achieve the above benefits while performing competitively with, if not better than, existing SOTA methods on individual- and cross-dataset evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper addresses two fundamental limitations in face landmark detection (FLD): the need to train separate models for each "N-point" dataset and the inability of these models to output a flexible number of landmarks. It introduces a paradigm-shifting solution built on two core innovations. First, it proposes Face Part-Anchored Landmark Positions (FPALPs), a novel, universal representation that defines each landmark not by its coordinates, but by its normalized, semantic position along a face-part contour. Second, it presents a query-based detection framework that accepts FPALP-based queries to enable fully dynamic, on-demand landmark prediction.

This new paradigm successfully enables a single model to be trained on a unified fusion of heterogeneous datasets, and to dynamically predict any number of landmarks at inference time. Experiments demonstrate that this approach achieves these critical new capabilities while maintaining performance that is highly competitive with specialized, state-of-the-art methods.

Crucially, the paper implicitly surfaces a fundamental challenge in the FLD task itself: the subtle inconsistencies in how different datasets define the precise contour shapes of face parts. The model's need to learn an "average" representation from these slightly varied annotations likely explains the minor performance gap compared to models trained on a single, self-consistent dataset. This is not a flaw of the method, but rather an inherent property of the data that this unified approach is the first to successfully navigate.

### Strengths
1. This paper introduces a powerful conceptual shift for the field of face landmark detection (FLD). By proposing the Face Part-Anchored Landmark Positions (FPALPs) representation, it reframes landmark detection from a dataset-specific coordinate regression problem into a generalized, semantic-querying task. This is a highly elegant and impactful innovation that addresses long-standing issues of data fragmentation and model inflexibility.
2. The FPALP concept directly unlocks three highly valuable capabilities that were previously unattainable with a single model:
a) Unified Multi-Dataset Training: It provides a principled framework for combining heterogeneous "N-point" datasets, significantly increasing the volume and diversity of data available for training a single, more robust model.
b) Flexible, On-Demand Prediction: The query-based architecture allows for dynamic prediction of any number of landmarks at inference time, offering unprecedented flexibility for diverse downstream applications.
3. The paper demonstrates, through rigorous experiments, that its unified and dynamic framework achieves performance that is highly competitive with specialized, state-of-the-art models. The fact that it incurs only a negligible performance drop while providing enormous gains in data utilization and output flexibility is a remarkable achievement and a testament to the soundness of the proposed approach.

### Weaknesses
The unified training approach, by design, exposes the model to subtle but real inconsistencies in how different datasets define the exact contour shape of a given face part. This introduces a form of unavoidable "label noise." The model's need to learn a generalized, "average" representation of each contour to accommodate this variance is likely the primary reason its performance, while highly competitive, does not exceed that of a specialist model trained on a single, perfectly self-consistent dataset. This is a fundamental trade-off between specialization and generalization, and the paper makes a compelling case for the immense value gained on the generalization front.

### Questions
1. Your results show that the unified model is highly competitive, yet does not strictly surpass specialist SOTA models on their native datasets. We believe this is not a weakness. Would you agree that this highlights a fundamental trade-off between generalization (from diverse data) and specialization (on self-consistent data)? Framing your results in this light seems to strengthen, rather than weaken, your contribution, as it showcases the immense flexibility gained for a minimal cost in specialization.
2. Following up on that, you astutely identified an average intra-cluster distance of 2.22 pixels when unifying the templates. Could one interpret this value as a proxy for the inherent "label noise" or "ambiguity" that exists between datasets? If so, would it be fair to say that your model, by learning a robust average representation, is performing optimally under this noisy supervision, and this inherent ambiguity itself constitutes the performance ceiling compared to a specialist model?
3. Given that the core challenge you've surfaced is this inherent ambiguity in contour definitions, have you considered future work that models this uncertainty explicitly? For instance, instead of predicting a single coordinate (x, y) for a given FPALP query, could the framework be extended to predict a probability distribution or an uncertainty ellipse? This seems like a natural and exciting next step to create models that are not only unified but also aware of the data's intrinsic ambiguities.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified landmark representation, FPALP, and a corresponding dynamic landmark detection network built upon it. FPALP is defined based on the ratio of lengths along pre-defined facial part curves. The proposed detection framework consists of three components: (1) Image-Agnostic Landmark Encoding Generation, which produces embeddings for both facial parts and their ratio representations; (2) Landmark Query Initialization, which interacts with image features through an attention mechanism; and (3) Landmark Query Refinement, which decodes the extracted features into the final landmark coordinates. The framework can be trained on datasets with varying numbers of annotated landmarks rather than relying on datasets with a fixed number. Extensive experiments demonstrate that the method achieves competitive accuracy on individual datasets and superior performance in cross-dataset evaluations.

### Strengths
1.	The paper introduces FPALP, a unified landmark representation defined by evenly distributed points along facial part curves. This design enables the framework to be trained across multiple datasets (e.g., WFLW, 300W, AFLW-19) while producing outputs that are not constrained to a fixed number of landmarks.
2.	The paper proposes a dynamic landmark detection framework based on FPALP, which predicts landmark positions using a fusion strategy that combines facial part embeddings and image features through a Dot-Product Attention Map and Deformable Image Cross-Attention.
3.	Extensive experiments demonstrate strong cross-dataset performance, and the ablation study is thorough, carefully examining both the image encoder and text encoder components.

### Weaknesses
Major:
1. The paper lacks experiments evaluating the accuracy of facial part curve prediction compared to interpolation-based methods. Since the core idea of FPALP is to model and predict facial part curves, such comparisons are essential to demonstrate FPALP’s effectiveness. The absence of these results leaves the validity of FPALP’s key contribution somewhat uncertain.
2. The paper does not adequately address the misalignment issue across datasets. One of the underlying assumptions is that the landmarks in each dataset are evenly distributed along facial part curves; however, this is not strictly true for all datasets. Moreover, FPALP’s handling of the start and end points of each curve— which vary across datasets—remains unclear. Although the authors briefly discuss this limitation, the analysis lacks sufficient depth.
3. The qualitative results focus primarily on frontal faces, with limited examples involving large poses. This raises concerns about the stability and robustness of FPALP under challenging geometric variations.

Minor:
1. Some parts of the mathematical formulation in the method section are unclear. For instance, the meaning of H_T in R^((H_T×W_ei×L)) (line 253) is not explained—possibly a typo? The paper would benefit from more explicit symbol definitions or improved figure annotations for clarity.
2. The use of text descriptions for facial part phrases appears unnecessary given the small number of facial parts. Although the authors attempt to justify this choice in the ablation study by replacing the frozen text encoder with a learnable embedding, the mechanism of the learnable embedding is not sufficiently explained. If it is simply a non-frozen version of the text encoder, the benefit of textual representation becomes questionable. Using one-hot encodings for facial parts might be a simpler and equally effective alternative.

### Questions
See the weakness

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
3

### Summary
This paper proposes a unified and dynamic face landmark detection (FLD) framework based on Face Part-Anchored Landmark Positions (FPALPs), which represent landmarks as normalized progression values along semantic face part contours. The method enables training on multiple "N-point" datasets (e.g., AFLW, 300W, WFLW) simultaneously and supports dynamic, on-demand landmark prediction at inference. A cross-modality decoder refines landmark queries constructed from FPALPs and text embeddings. Experiments show competitive performance with state-of-the-art methods while offering enhanced flexibility and generalization.

### Strengths
1. Unified Representation: The FPALP formulation effectively aligns heterogeneous landmark annotations across datasets, enabling a single model to learn from multiple sources without manual interpolation or 3D priors.
2. Dynamic Inference: The framework supports arbitrary landmark queries at test time, allowing customizable output granularity and adaptability to diverse downstream tasks—a significant advantage over fixed-output models.

### Weaknesses
1. Template Alignment Sensitivity: The method relies on approximate alignment of face templates across datasets, which may limit scalability when integrating new datasets with highly divergent landmark definitions or severe annotation inconsistencies.
2. Performance Trade-offs: While competitive overall, the model exhibits a slight performance drop on certain subsets (e.g., WFLW68) when trained on fused datasets, suggesting sensitivity to dataset-specific challenges and sampling strategies.

### Questions
1. How does the framework handle face parts that are heavily occluded or not present in the training data? The paper mentions future extensions but does not evaluate robustness under such conditions.
2. Could the dependency on text encoders (e.g., SentenceBERT) introduce biases or limit performance in low-resource languages or domains where facial part semantics differ?

### Soundness
2

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
3

### Summary
This paper proposed Face Part-Anchored Landmark Positions as a generalized representation for landmark, w.r.t. face part's contour.
This enables test-time specification of landmarks, with a single model (backbone and head) trained on any N-point datasets.
Experiments prove this method generalizes while keep competitive result with existing SOTAs.

### Strengths
Generalization to arbitrary landmark definitions is an important problem for the community. FPALP is a sound solution, and the authors also conducted extensive experiments to validate its effectiveness. Also, the paper is well-written and easy to follow.

### Weaknesses
As you mentioned in Sec 4 and A.10, WFLW dataset consists of extreme poses and occlusions. In A.7 you said those two factors can be challenging for your FPALP. It will be good if you can show some of those cases in addition to Figure 7.

### Questions
1. Fig. 3 the meaning of "Face Contour" is unclear. 
2. Is FPALP limited to contour landmarks in theory? For example, can we extend this method to represent a landmark at the center of eyeball?
3. As you mentioned at Line 242, if you believe facial layout semantics within text encoder is important, it will be helpful to show results where FPALP is trained with text encoder that was already aligned with image encoder, e.g., CLIP.

### Soundness
3

### Presentation
3

### Contribution
3
