# From Attention to Prediction Maps: Per-Class Gradient-Free Transformer Explanations

- Decision: Reject
- Scores: 6, 3, 3, 6

## Abstract
The Vision Transformer (ViT) has become a standard model architecture in computer vision, especially for classification tasks. As such, explaining ViT predictions has attracted significant research efforts in recent years. Many methods rely on attention maps, which highlight \emph{where} in the image the network directs its attention. In this paper, we introduce Prediction~Maps -- a novel explanation method that complements attention maps by revealing \emph{what} the network sees. Prediction maps visualize how each patch token within a given layer is associated with each possible class. This is done by utilizing the classification head at the output of the network, originally trained to be fed with the class token at the last layer. Specifically, to obtain the prediction map of a particular layer, we apply the classification head to every patch token within that layer. We show that prediction maps provide complementary information to attention maps and illustrate that combining them leads to state-of-the-art explainability performance. Furthermore, since our proposed method is neither gradient- nor perturbation-based, it offers superior computational and memory efficiency compared to competing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
ViTs are commonly used in computer vision and their interpretation is important. This paper introduces Prediction Maps, which is a novel combination of two interpretability methods. The first method applies the classifier head trained on the CLS token to intermediate patch representations to elicit class information at each patch location and each layer — "the what." The second method leverages the learned attention maps to interpret how spatial information is being moved around the feature map — "the where." Predictions maps are built in two steps. First, class-specific attention maps are computed by taking a weighted sum of attention maps; weights are the similarities between the attention map of a layer-head with the class-specific prediction map. Second, the per-element product is computed between the class-specific weighted attention map and the class-specific prediction map. Prediction Maps can better predict patch importance (table 1), are more useful for segmentation (table 2), and are compute friendly (table 3).

### Strengths
- This is an important topic — ViTs are seeing widespread adoption and we need better interpretability methods
- Combining prediction maps and attention maps is novel, and how they are combined is interesting
- Both perturbation and segmentation experiments are convincing to me
- The ablations (table 4) are interesting and important
- I like the BERT and CLIP demonstrations in the appendix
- I'm interested in using this method and I suspect others will be

### Weaknesses
I believe the class-wise prediction map is equivalent to the "logit lens" method. Logit lens applies the classifier trained on the final CLS token to intermediate token representations to form token- and layer-wise class predictions. To the best of my knowledge, this was introduced in NLP in 2020 [1] and formally studied in 2023 [2]. The logit lens was used in a ViT interpretability paper [3], another ViT paper [4], and a ViT interpretability repo [5].

Please clarify if your class-wise prediction map is indeed equivalent to the logit lens. If they are equivalent, I still like this paper — it introduces a novel combination of existing interpretability techniques. If they are not equivalent, please clarify. 

[1] Blog post: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

[2] Belrose et al. "Eliciting Latent Predictions from Transformers with the Tuned Lens"

[3] Vilas et al. "Analyzing Vision Transformers for Image Classification in Class Embedding Space", NeurIPS 2023

[4] Fuller et al. "LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate", NeurIPS 2024

[5] Repo: https://github.com/soniajoseph/ViT-Prisma

### Questions
For the perturbation tests, patches are masked based on their order of importance. Is this importance estimated using each interpretability method — allowing you to compare how well the interpretability methods can predict the classification importance of patches?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces the concept of Prediction Maps, an explainability method for Vision Transformers (ViTs) that complements traditional attention-based approaches. While attention maps show where the network focuses, Prediction Maps reveal what the network perceives at each location by applying the classification head to individual patch tokens throughout the network's layers (layer probing).
The main contributions are : (1) The introduction of Prediction Maps: A gradient-free, class-specific explainability method that visualizes how each patch token contributes to the different class predictions. (2) PredicAtt: A technique that fuses Prediction Maps with Attention Maps using similarity-based weighting to create comprehensive explanations that capture both what the network sees and where it looks. (3) : State-of-the-art Results: The method achieves superior performance on perturbation and segmentation tests while being computationally more efficient than existing gradient-based approaches and attention-based approaches. (4) By separating "what" from "where" in network predictions, the method enables better diagnosis of failure cases and understanding of how the network processes visual information across layers.

### Strengths
The paper introduces a novel approach to explaining Vision Transformer predictions through "Prediction Maps," which repurposes the classification head to analyze individual patch tokens across the layers. The work shows reasonable originality in its gradient-free method for class-specific explanations and its combination of attention and prediction information. The significance lies primarily in providing a computationally efficient tool (gradient-free and class-specific) for model interpretation and debugging.

### Weaknesses
1: Line 266, this is not surprising, the core idea of applying a classifier to intermediate representations has been explored before, notably in "LogitLens" [1] and "TunedLens" [2] for NLP which similarly applies the classification head to intermediate tokens to analyze model behavior.


2: The paper should better acknowledge and differentiate from such prior work, especially since the technical approach shares similarities.


3: No theoretical analysis of why patch token predictions are meaningful despite not being trained for this purpose. While Fig. 3 shows some qualitative examples, the paper lacks systematic analysis of how patch token predictions evolve across layers. A quantitative study tracking prediction accuracy, confidence, and class distribution for patch tokens across layers would provide crucial insights into how the network builds its representations.


4: The method requires consistent token dimensions, limiting applicability to architectures like Swin Transformer.


5: Design choices (e.g., dot product similarity in Eq. 4) lack thorough justification (e.g., cosine similarity, which normalizes for magnitude, or KL divergence since we're comparing probability-like distributions).


6: Looking at Tables 1-2, the performance gap between the proposed method and TransAttr appears to shrink with smaller models (ViT-B vs ViT-L). The authors should evaluate their method on even smaller architectures (e.g., ViT-S, ViT-Ti) to verify if the advantages persist. This would help understand if the benefits are truly architectural or simply emerge from scale.


7: The BERT and CLIP extensions feel preliminary and lack quantitative evaluation (e.g .object localization tasks where ground truth bounding boxes are available, activation & pruning tasks [3]).


8: [3] and [4] are highly related related-works, which are not mentioned and compared against.


[1] https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens


[2] https://arxiv.org/abs/2303.08112


[3] https://arxiv.org/abs/2202.07304


[4] https://arxiv.org/abs/2402.05602

### Questions
1. typo - Line 25 : our -> ours.


2. Line 156 - I wouldn't use the word "null", instead a (baseline) image. This acknowledges that while zero/null images are common baseline choices, other choices exist and can be meaningful [1].


3. Line 197: The statement about the [CLS] token needs more precise language. I would suggest revising "It is believed that..." to "The [CLS] token is designed to aggregate information from all other tokens to enable classification based on that token alone." This better reflects the architectural intent.


4. Line 223: The claim about attention maps not being class-specific needs qualification. Add discussion of Multi-class Token Transformer [2] which demonstrates class-specific attention through multiple class tokens. This work shows an alternative approach to obtaining class-discriminative attention maps (while being also gradient-free and class-specific).


5. line 406 - replace "as follows." -> "as follows :"  for proper enumeration.

[1] https://arxiv.org/abs/2310.04821


[2] https://arxiv.org/pdf/2203.02891

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes Prediction Maps as an explainability method for vision transformers. Prediction Maps mainly apply the trained classification head of the network to each patch token at each layer, thus classifying each patch at each step in the pipeline.  The method is evaluated on positive and negative perturbation tests on ImageNet and for segmentation on ImageNet Segmentation.

### Strengths
The topic if and how diffusion models are able to capture visual representations is of great interest. The topic is also very timely.

The paper is easy to follow.

### Weaknesses
- 1. Is patch classification really explainability? 

I understand that the term explainability leaves some room for discussion what falls under it and what not, so using the ISO/IEC TR 29119-11:2020, Sec 3.1.31. here as a base for the following discussion. This considers explainability as the understanding of how the AI-based system came up with a given result. Following this definition I'm not convinced that patch-wise classification actually gives this information, as it is, at the end, patch-wise classification. 
Note that patch-wise classification can still do good in localization, and it also makes sense that when you remove patches that carry this class information, the overall classification accuracy drops. But overall, I don't understand how such classification gives us any information on how the system came up with this class. In this case, the conclusion would just be that classification == explanation, so technically, we would not need gradient or attention-based methods, as we could do as well with classification (which would indeed be faster).  Happy to hear what my fellow reviewers are thinking about this.

- 2. Layer-wise patch classification

I understand that the layer-wise patch classification can be used as information on which layer classes are, e.g., constructed (different from the question of how they are constructed). And, given skip-connections, I can understand that class information does not magically appear in the last layer but that some information is carried by the layers before. This might be worth a deeper analysis, but the ablation is a bit shallow in this respect.

### Questions
- Does TransAttr refer to (Chefer et al., 2021b)? Please indicate somewhere which publication this refers to.

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
2

### Summary
This paper introduces Prediction Maps, a novel explanability method for Vision Transformers (ViTs), specifically for classification tasks. Prediction Maps provide additional insight to attention maps, by providing per-class explanability of what the network "sees" or perceives within each region of an image. The authors achieve this by applying the classification head - originally trained for the class token at the last layer - to every patch token within a given layer. This innovation offers both computational and memory efficiency as it does not rely on gradients or perturbations. Through their empirical studies, they demonstrate that Prediction Maps complemented with attention maps offer impressive explainability performance. They also assert that their method is the first to provide class-specific, gradient-free explainability for ViTs.

### Strengths
The paper is quite innovative, presenting a well-explained and distinct approach to ViT prediction explanation. Prediction Maps as an alternative to attention maps are a significant contribution, offering a fresh perspective in examining network interpretations. The method being gradient-free implies it is uniquely efficient in resource consumption, a favorable characteristic compared to other methods requiring significant memory and computational time. Additionally, the method's class-specificity offers a more precise explanation scope compared to other gradient-free methods. The evaluation methods are robust, and the results indicate the method's superior performance in measure tests and resource efficiency. Furthermore, the concept that the technique could be extended to other models, such as BERT or CLIP, is intriguing.

### Weaknesses
A notable limitation of this method is that the classification head must accept tokens of a specific dimension, which won't hold for certain architectures like DINO or Swin. This could limit the application spectrum of the method. Also, there is less detail concerning the complete evaluation process and the steps taken to ensure robustness. Further benchmark comparisons with other explanation methods would also be beneficial for the comprehensiveness of the findings.

### Questions
1, How would the method be adjusted to handle architectures where the classification head accepts tokens of a varying dimension, such as DINO and Swin?
2. How does the method perform on larger or more complex datasets? Meaning on scale

### Soundness
3

### Presentation
3

### Contribution
4
