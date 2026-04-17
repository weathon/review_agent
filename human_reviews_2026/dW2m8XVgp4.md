# Taming the Forensic Singularity: A Regularized Hyperbolic Framework for Generalizable AI-Generated Image Detection

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Detecting AI-generated images is a critical task in multimedia forensics, yet the generalization of detectors to unseen generative models remains a persistent challenge. While pre-trained Vision Transformers (ViTs) have emerged as powerful feature extractors, existing forensic methods often default to using final-layer features, which may discard crucial forensic traces. We embark on a systematic probe into the latent representations of various ViTs and uncover a universal phenomenon we term the "Forensic Singularity": a narrow region within the ViTs' mid-level layers where forensic separability culminates before giving way to semantic abstraction. To harness the immense potential of this "singularity" layer while mitigating its high risk of overfitting to limited training data, we propose a Regularized Hyperbolic Framework. Our framework learns a polar representation in the Poincaré Ball by disentangling semantic content and forensic evidence: the final-layer semantic feature guides the direction, while the singularity-layer forensic feature determines the radius. This inherent geometric constraint regularizes the model, promoting a more generalizable decision boundary. Extensive experiments demonstrate that our approach not only establishes a new state of the art on multiple datasets, and exhibits superior generalization performance on unseen generative models. Our work provides both a powerful new tool for AI forensics and a deeper insight into how the hierarchical representations of ViTs can be effectively harnessed.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies detection of AI-generated images (AIGI). Prior work predominantly relies on the final, most semantic layers of ViT-style encoders. The authors challenge this convention and report a consistent mid-layer “forensic singularity”—a relatively narrow band of layers where real vs. fake separability peaks. Motivated by the observation that mid-layers carry strong but potentially overfitting-prone forensic cues, while final layers are semantically stable but less separable, the paper proposes a Regularized Hyperbolic Framework (RHF). Concretely, it fuses mid-layer “forensic” features and final-layer “semantic” features, maps them to the Poincaré ball via a polar construction (direction = semantics; radius = forensic strength), and uses a radial hyperbolic classifier for detection. Experiments on in-domain and challenging OOD settings show promising results.

### Strengths
1.	Clear figures and intuition. The layer-wise separability plots and geometric visualizations make the high-level story easy to follow.

2.	Comprehensive comparisons. The paper evaluates across multiple datasets/generators.

### Weaknesses
1.	Organization. Although Regularized Hyperbolic is as central as Forensic Singularity, it is barely introduced in the Introduction and only appears later in the dedicated section, making it hard for readers to grasp the paper’s key contributions early on.

2.	Mid-layer advantage is not a novel observation. As the authors acknowledge, leveraging intermediate representations for detection tasks is not new; similar observations exist in both vision and LLM hallucination detection, where mid-layer features can outperform the final layer for anomaly detection. Thus, the claimed Forensic Singularity reads primarily as an application and reconfirmation within AIGI forensics.

3.	On the PE-Core-B16 anomaly (line 226). Attributing the “anomalously poor performance” to knowledge distillation lacks supporting references and experiments, which makes the explanation appear overconfident and weakens the overall robustness of the paper.

4.	Missing ablations. The framework lacks key ablations on Multi-Layer Forensic Feature Fusion and Multi-Task Learning with Auxiliary Heads, making it difficult to judge the necessity and actual contribution of these components.

### Questions
1.	(Line 118) Regarding “where crucial non-semantic artifacts have not yet been discarded by semantic abstraction”: how do the authors operationally define “semantic” vs. “non-semantic” features? What evidence shows that non-semantic artifacts are discarded in later layers? This clarification is important to motivate the separation between semantic and non-semantic (forensic) features in the RHF.

2.	The proposed layer-wise separability score is classifier-agnostic, which is valuable. However, since final detection still uses a classifier, please also report the performance of a classifier trained per layer to cross-validate that the separability score indeed predicts practical classifier performance.

3.	Concerning In-Domain results in Table 1: Figure 1 shows higher separability for mid-layers than for the final layer, yet Layer 13 still does not surpass Semantic even In-Domain. How do the authors reconcile this inconsistency? How does this align with line 265’s statement that “the mid-level layers possess the highest potential signal”?

4.	The Polar Construction in the Poincaré Ball jointly uses semantic and forensic vectors as direction and radius and shows gains. But do these gains stem from the hyperbolic framework itself, or simply from providing more features to the classifier? Please include a Euclidean baseline: a simple Euclidean classifier that concatenates or linearly fuses the semantic and forensic vectors, to check whether the gains come from your new design or simply from giving the classifier more features.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenge of generalizable AI-generated image detection. The authors first identify a phenomenon they term the **"Forensic Singularity"**: a consistent peak in forensic signal found in the **mid-level layers** of Vision Transformers (ViTs), rather than the commonly used final layer. They observe that while these mid-level features are highly discriminative, they are also prone to overfitting.

To address this, the paper proposes a **Regularized Hyperbolic Framework (RHF)**. This novel approach uses hyperbolic geometry to disentangle features: it maps the final-layer semantic features to the *direction* and the mid-level forensic features to the *radius* of a point in the Poincaré Ball. This design anchors real images to the center and pushes diverse fake images to the periphery, acting as a powerful geometric regularizer to improve generalization.

The main contributions are:

1. **Identification of the "Forensic Singularity"**: Providing a principled, data-driven guideline for selecting the most potent forensic features from ViT backbones.
2. **A Novel Hyperbolic Framework (RHF)**: A new architecture that effectively harnesses the power of mid-level features while mitigating their risk of overfitting through geometric regularization.
3. **State-of-the-Art Generalization**: Demonstrating through extensive experiments that the proposed method significantly outperforms previous approaches in detecting images from unseen, advanced generative models and in simulated real-world scenarios.

### Strengths
* **Originality:** The paper’s originality lies in identifying and framing the *Forensic Singularity*, offering a data-driven rationale for focusing on mid-level ViT features. The proposed Regularized Hyperbolic Framework (RHF) creatively applies hyperbolic geometry to balance the strong discriminative power and overfitting risk of these features.

* **Clarity:** The paper is clearly written and logically structured. The core ideas—*Forensic Singularity* and the geometric intuition of RHF—are well explained.

* **Significance:** The work addresses a key challenge in AI-generated image detection: generalization to unseen models. The results show improvements over prior methods, and the concept of *Forensic Singularity* could serve as a useful guide for future research.

### Weaknesses
* The paper does not analyze the **computational overhead** of RHF. Extracting and processing multiple intermediate layers inevitably increases inference cost. A quantitative comparison of inference time and memory usage with standard ViT baselines would clarify its practicality.

* The **robustness** of RHF against post-processing and adaptive attacks is not examined. Detectors can be bypassed by adversarially tuned generators or simple image transformations, which warrants further testing and discussion.

### Questions
### **Computational Overhead and Practical Deployment**

* What is the increase in inference time and GPU memory compared to a standard ViT using only the final layer?
* How many additional parameters are introduced by the multi-layer fusion and hyperbolic mapping modules?
* How does this overhead affect real-time or large-scale deployment feasibility?

---

### **Robustness to Postprocessing and Adaptive Generators**

AI-generated images can be post-processed to evade detection (e.g., JPEG compression, rescaling, color jittering, or adversarially fine-tuned generators).

* Have the authors tested RHF under such transformations or adaptive attacks?

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
This paper proposes a novel AIGC detection method by analyzing which layers of features in the pre-trained ViT (Vision Transformer) are more conducive to forensics. Specifically, it first defines "Forensic Singularity" to characterize the inter-class distance between real and generated images in high-level features. Based on this, it finds that the ViT layers with a depth range of 30% to 60% exhibit the best forensic discriminability. In addition, the paper suggests using Polar Construction to characterize the forensic feature space while eliminating the influence of the semantic space. Comparative experiments are conducted on multiple ViT backbones and datasets, demonstrating certain advantages over existing works.

### Strengths
- It analyzes which layers of features in ViT are beneficial for AIGC detection.
- It adopts Polar Construction (instead of linear methods) to characterize the feature space of real and generated images.
- It conducts relatively comprehensive experiments.

### Weaknesses
- In Lines 131–135, the paper first assumes that a good feature representation should cluster real images into one class and generated images into "another distinct and distant cluster". Is this assumption rigorous? For instance, images of the same category (e.g., all "cats") may be affected by content, making them consistently closer in the ViT space (which is trained for category discrimination tasks) compared to images of other categories (e.g., "fish"). Therefore, would it be more appropriate to use semantic categories as a baseline before clustering real and fake images? It is recommended to conduct a more in-depth discussion on the assumptions of this problem.
- Regarding the "Natural Center-Periphery Structure" in Line 287, the paper states that "anchor the singular normal class of real images at the geometrically unique origin". Is it possible to anchor generated images at the origin instead? In other words, could real images be more widely distributed in the "forensic space", while generated images are closer to each other due to the artifacts inherent in GANs (Generative Adversarial Networks) and Diffusion Models? It is recommended to add a comparative experiment using generated images as the origin to verify this reversely.
- Weak technical innovation: Although the idea of using Polar Construction/Poincaré Ball is interesting, the adopted calculation processes (e.g., Equations 1–3 only simply define the in-class/out-class distances of two clusters, and Equation 4 for constructing the polar directly uses geometric regularization) are all based on existing works.
- Lack of ablation studies: Although the experimental section evaluates the method on multiple test sets/ViT backbones, it does not explore other separation methods for semantic/forensic vectors. For example, can simple linear classification or clustering (instead of Polar Construction) achieve good performance? Or, if global features or only the features of the last layer are used in the front-end multi-feature fusion, can this also bring performance improvements?
- It is recommended to unify the font for referring to the same content, such as "SD v1.4", which is sometimes in \texttt font and sometimes in regular font.

### Questions
Please see Weaknesses.

### Soundness
3

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
The authors propose to use features from multiple layers from a CLIP or PE network, and train a network with a cross-entropy-classification branch over the last layer, with added intermediate layers and logits obtained from hyperbolic distances between feature maps and some center. They perform experiments on several datasets with their approach.

### Strengths
- They perform experiments on newer datasets. 
- the idea well is understandable
- they have a good overview graphic
- a creative idea

### Weaknesses
Important inclarities about the approach, making it not reproducible:

- how is the learnable center updated ? This makes a big difference for the performance of the algorithm.

- which way was used to compute the hyperbolic distance? There are a few available over the unit ball. $C_{rad} (p)$ is not defined.
- logits from a distance ?
 [...] to compute logits $o_{main} =C_{rad} (p)$ based on the hyperbolic distance of $p$. 
Do you use the distance as logits ?

- Ablation study missing: in Table 1 a simple classifier achieves 93.8 % accuracy . In Table 2 their construction using multiple features gets 95.09 % . To what extent is it the hyperbolic distance, and to what extent one achieves it because of using multiple features ? It does not seem that it is the hyperbolic distance but rather the combination of multiple features. If it would be so, that should be reported  in an ablation study no matter whether hyperbolic distance gives an edge or maybe not.

  - What happens if one uses only the distance of $v_{for}$ from a learnable center ?
  - What happens if one just classifiers with the concat of  $v_{for}, v_{sem}$ ? 
  - What happens if one uses a similar modification with exponential push-away such as $exp ( \| p- c\| )$ with $p$ as defined by the paper ?
  - These kind of ablations are needed to understand the impact of the components.

- the separability score is not novel. Such constructions exist, they should cite papers with similar constructions, such as linear discriminant classifiers and the like.
- the separability score fails to produce low scores for data which is linearly separable but has high variance parallel to the separating hyperplane . In that sense it has limitations.

- z score in Figure 2 is undefined. 

- a minor issue but still bad wording
"To resolve this, we propose a novel,
theoretically-grounded Regularized Hyperbolic Framework that tames the singularity
features by leveraging the unique geometry of hyperbolic space."

What is the theoretical grounding ?
What is the regularization ?

"taming" is hyperbolic LLM wording here: It is colorful but lacks substance. There is no danger or something overwhelming or similar involved. Singularity features ?  We talk about feature maps. A mild peak also does not make for a singularity either.

### Questions
See the section weaknesses please. The most important missing things are:
- center update
- ablation study also with simpler distances but the same multiple feature maps. 
- code to compute the hyperbolic distance

### Soundness
2

### Presentation
2

### Contribution
2
