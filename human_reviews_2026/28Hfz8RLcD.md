# MICLIP: Learning to Interpret Representation in Vision Models

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
Vision models have demonstrated remarkable capabilities, yet their decision-making processes remain largely opaque. Mechanistic interpretability (MI) offers a promising avenue to decode these internal workings.  However, existing interpretation methods suffer from two key limitations. First, they rely on the flawed activation-magnitude assumption, assuming that the importance of a neuron is directly reflected by the magnitude of its activation, which ignores more nuanced causal roles. Second, they are predominantly input-centric, failing to capture the causal mechanisms that drive a model's output. These shortcomings lead to inaccurate and unreliable internal representation interpretations, especially in cases of incorrect predictions. We propose MICLIP (Mechanism-Interpretability via Contrastive Learning), a novel framework that extends CLIP’s contrastive learning to align internal mechanisms of vision models with general semantic concepts, enabling interpretable and controllable representations. Our approach circumvents previous limitations by performing multimodal alignment between a model's internal representations and both its input concepts and output semantics via contrastive learning. We demonstrate that MICLIP is a general framework applicable to diverse representation unit types, including individual neurons and sparse autoencoder (SAE) features.  By enabling precise, causal-aware interpretation, MICLIP not only reveals the semantic properties of a model's internals but also paves the way for effective and targeted manipulation of model behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper presents MICLIP, a framework that interprets internal units—either neurons or SAE-derived sparse features—by learning a lightweight encoder that maps their activations into CLIP’s semantic space. Training uses a symmetric InfoNCE objective that aligns each unit’s representation simultaneously with the input image embedding and the model’s predicted-label text embedding, replacing activation-magnitude heuristics with representation-level alignment. Once trained, the shared space enables concept→mechanism localization (finding units for a queried concept) and mechanism→concept description (annotating a chosen unit), and it naturally extends to SAE pseudo-activations to mitigate polysemanticity. The authors validate behavioral relevance through interventions that amplify or ablate selected units and consistently shift model outputs in the predicted direction, and they report strong results across ResNet/ViT/CLIP backbones with out-of-distribution generalization to texture concepts (DTD). Additional analyses, including layerwise error trajectories, demonstrate that MICLIP offers a practical, general recipe for unit-level interpretation without relying on unstable gradients or purely input-centric assumptions.

### Strengths
Gradient-free “causal” proxy via CLIP. Instead of using unstable gradients or handcrafted causal scores, the paper detours through CLIP to obtain a robust semantic target, then validates behaviorally via interventions. 

Truly dual-anchor alignment (novel). Aligning activation→text and activation→image (not just label text) is a clean, original design that reduces input-centric bias and avoids the “activation-magnitude == importance” heuristic.

Works on SAE pseudo-activations. The description/localization pipeline applies neatly when only a single SAE feature is “turned on” via the decoder, which is practically valuable for polysemantic layers.

Thorough, convincing experiments. The paper hits standard metrics, reports state-of-the-art, includes interventions (↑ with amplify, ↓ with ablate), spans multiple backbones, and shows the method also works at OOD textures.

### Weaknesses
Not causal by design. The loss aligns to CLIP (image/text) semantics rather than optimizing a constraint tied to model logits or a formal mediator test.(Although this may be unstable) - so the causality holds between CLIP output and layer output, not between model output and layer output. The method demonstrates empirical causal relevance via interventions, but the mechanism is not guaranteed by the objective.

### Questions
Low-level features. Can you provide qualitative low-layer examples (e.g., “white horizontal line,” “45° diagonal line”) showing agreement between SAE top-activations and mechanism→concept retrieval? It would be a great aid to show MICLIP can effectively explain SAE features.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MiCLIP, a method that trains a CLIP neuron encoder to align with the visual and textual encoders of CLIP such that a neuron can be converted into an interpretable embeddings space (CLIP's embedding space) that can be queried via text. It aligns intermediate neuron activations of a target model (e.g., ResNet50) to the input image, and also adds the alignment of the activation values the predicted class labels for improved faithfulness. The method performs well compared to baselines.

### Strengths
- The paper is clearly written and easy to follow, it was an interesting read. I would like to thank the authors for the comprehensive appendix also. 
- The authors tackle the "input-centric" paradigm (usually unfaithful to the mechanism of the model's decision-making process), and which often fail in cases of incorrect predictions (because they are explicitly trained on human-desired concepts). They do this by considering the model’s output decision. 
- The proposed method can work with SAE features without actually training on those features (this means the SAE features are basis vectors of the dense features from the residual stream)
- The idea of representing a single neuron in the CLIP embedding space (via the Neuron encoder) is very interesting. 
- The analysis is Figure 5 is interesting as it allows us to see the layer where the model fails.

### Weaknesses
- The paper assumes that the target layer they operate on, actually contains concepts they align to (here, the concepts are the predicted class labels). This does not really make sense to me. Different layers have different functions. For example, early layers represent low-level features such as textures, blobs, edges...etc. But the authors align them with high-level concepts (class labels). In other words, we never know the actual concepts that a target layer uses, so we cannot assume that these concepts are the predicted class labels. For me, this is a major weakness. 
- The work of [R2] is very similar, it uses CLIP to learn interpretable embeddings (also sparse) which is exactly what the authors do. Although this work investigates CLIP, the general idea of learning a CLIP model to represent interpretable embeddings is the same.  Furthermore, [R3] also investigates the same idea of interpreting intermediate representations of models (including fine-grained neurons) with text by generating text in an autoregressive manner, and is trained on MILANNOTATIONS dataset [R4], a human-curated dataset for neuron annotation
- Why dont the authors include MILAN [R4] as a baseline? It is a popular baseline and should be reported. 
- I am not convinced with the motivation of the paper. The claim in L52-53: "An increase in activation value does not necessarily imply the occurrence of the corresponding concept during inference. Conversely, even negative activations can positively influence the model’s
prediction of certain concepts." Is there any reference for this? How do the authors come up with such a claim? It has been well-established since 2013 that high activation values correspond to concepts (or parts of concepts). Potentially, they could be biases (e.g., dumbells and hand occur together so the "hand" concept will activate on dumbell images). However, high activation values still *drive* the propagation to other future layers and eventually the prediction. The authors repeatedly mention this problem all over the paper, but to me this problem is not well justified. 
- Since the method aligns the neuron representation in the CLIP space, it therefore inherits CLIP's biases. For example, it may inherit the text-spotting bias in CLIP, even though this bias is not present in the target model.  
- There is an evaluation method proposed for neuron annotation works [R1], have the authors considered this evaluation method? Right now, most of the evaluation involves interventions. 
- In Table 1, why aren't the other baselines (Act-Values, Network Dissection and V-Interp) reported? 
- In Table 2 I dont understand why dont authors report the overall accuracy of the model with enhancement? For example, ResNet-50 imagenet accuracy is around 76%. The the 5% boost, does that mean it is 81% ? 
- The method seems to work less well for transformer models, according to Table 2. 
- The description of Figure 5 is wrong. It mentions a CD player but it is actually a sea anemone and feather boa. 
- Minor: In Table 1, what is the need of "Sig." ? Is it it fill space? 

[R1] CoSy: Evaluating Textual Explanations of Neurons
[R2] STAIR: Learning Sparse Text and Image Representation in Grounded Tokens
[R3] DeViL: Decoding Vision features into Language
[R4] Natural Language Descriptions of Deep Visual Features

### Questions
At the current stage, there are many problems in the paper (see weaknesses). Therefore my decision will be borderline reject for now. My decision could potentially be adjusted in the rebuttal.

### Soundness
3

### Presentation
4

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
This paper proposes a mechanistic interpretability approach for vision models, by projecting model internal representations to CLIP embedding space. The authors train an encoder to project model activations to CLIP embedding space via two contrastive losses: (1) between the projection and the CLIP embedding of the model output and (2) between the projection and the CLIP embedding of the image itself. The idea is to then project one-hot activations (resembling individual neurons) to CLIP space and thus learn which concept a unit represents. One can then conduct causal interventions on models, by modulating the units relevant for a concept. The authors validate their method by applying it to the final classification layer of various models and showing that the projections are close to CLIP / Mpnet embeddings of the ground-truth labels.

### Strengths
- The motivation of the work is very good, both the field's reliance on the activation-magnitude assumption and the fact that existing approaches focus on input-centric explanations too much is valid.
- The work is well-placed in the related literature.
- The proposed method is applied to multiple models of different architectures, and compared against multiple reasonable baselines.

### Weaknesses
- The implicit assumption behind this work, which is never addressed in the text, is that CLIP embedding space is somehow interpretable. To what extent this is true is hard to say, but my intuition is that while the method may work for the kinds of object-level representations one finds in late layers, it likely won't transfer to earlier layers which detect more primitive features.
- The clarity of the writing could be improved. For example, from figure 2 one might think that the target model is trained or fine-tuned (because while the clip-encoders are show to be frozen, the target model does not) and the paper generally spends more time making claims than backing them up (eg, line 198). The "Common-Nk" datasets in table 1 are never explained, I have to assume they are subsets of COCO?
- One technical problem that I see is that $E_n(a_i \cdot e^{(i)})$ is probably OOD for the encoder. The $a_i$ would have to be post-relu activations of very sparse layers for training on real activation vectors to generalize to these base vectors. Investigating vectors of unit length is also questionable because this assumes that the scaling of neuron activations is somewhat standardized, which might not be the case.
- The validation of the approach in table 1 is in my opinion insufficient, because the final classification layer is special w.r.t. the two issues outlined above: It represents high-level semantic concepts, and is likely much sparser than other layers.
- The reporting of significance in table 1 is superfluous: At 100k test images, even very small differences are significant. I would save the space and write in the caption that all differences are significant.
- In principle, I like the analysis in table 2, but I would have done it differently: You are currently not evaluating the specificity of interventions. When you are removing units responsible for a concept, you are probably not only hurting performance on the target class, but also on other classes. The relevant performance metric should not just be the absolute delta in classification accuracy for the target class, but the relative drop in accuracy for the target class relative to other classes. For example, it's possible that the -17% average accuracy delta for the target class you find in ResNet-50 is not different from all other classes, which could also be dropping by 17%. Higher specificity would mean that the target class accuracy drops by 17% while the average accuracy of all other classes drops by only 1%, for example.
- I thus don't think findings 2 and 3 are sufficiently supported by evidence (in Table 3 the CIs overlap and the absolute deltas are very small).
- Minor point, but in section 4.6 the example in the text doesn't match the example in figure 5.
- At the very end of section 4.6 you claim that your method points out "where the model's view shifts and why it fails". The first part is reasonable, the latter should be removed: We have no idea why the similarity to the wrong label surpasses that to GT, we can just observe that it happens. 

Overall, I am quite unimpressed with the paper. I initially liked the pitch of overcoming the activation-magnitude assumption, but I don't think the approach convincingly shows that it achieves any of its objectives. It is a general issue with the field of interpretability that evaluation is hard and usually not properly done, but I find these evaluations particularly insufficient.

### Questions
- I assume you keep the target model frozen and only train the encoder, but just to make sure: Which parts of the training pipeline are frozen?

### Soundness
2

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
This paper introduces MiCLIP, a novel framework for mechanistic interpretability of vision models that aligns internal representation units (neurons or sparse features) with human-understandable concepts in CLIP's semantic space. The key innovation is using contrastive learning to create a shared embedding space that connects model internals with both input images and output predictions, moving beyond the traditional "activation-magnitude assumption." MiCLIP enables bidirectional interpretation (concept-to-mechanism localization and mechanism-to-concept description) and supports model steering through targeted interventions. Extensive experiments on ResNet-50, ViT-B/16, and CLIP/ViT-B-16 demonstrate superior performance in interpretation accuracy, intervention effectiveness, and generalization to unseen concepts compared to established baselines.

### Strengths
1. **Novel and Well-Motivated Approach:** The dual input-output grounding through contrastive learning represents a significant advancement over existing input-centric interpretability methods. Moving beyond the activation-magnitude assumption addresses a fundamental limitation in current mechanistic interpretability research.

2. **Comprehensive Technical Framework:** The method provides a unified approach that works with both neurons and sparse autoencoder features, offering flexibility across different granularities of model internals. The integration with k-SAE for feature disentanglement is particularly valuable for addressing polysemanticity.

3. **Thorough and Multi-faceted Evaluation:** The paper presents an extensive experimental evaluation covering:
   - Quantitative interpretation accuracy across multiple models and concept sets
   - Intervention studies demonstrating precise model control
   - Generalization to unseen concepts and datasets
   - Semantic geometry analysis and attention map visualizations
   - Failure diagnosis through layer-wise semantic trajectory analysis

4. **Practical Utility:** The framework enables meaningful model steering and provides insights into flawed reasoning processes, making it valuable for both understanding and improving vision models.

### Weaknesses
1. **Dependence on CLIP's Semantic Space:** The interpretability power is inherently limited by CLIP's semantic coverage and potential biases. A more detailed discussion of these limitations and how they might affect interpretation fidelity would strengthen the paper.
2. **Architectural Scope:** While the method is evaluated on several discriminative vision architectures, its applicability to generative models or larger vision-language models remains unexplored. Some discussion of potential challenges in scaling to these domains would be valuable.
3.**Computational Cost Analysis:** Although Table 5 provides FLOPs comparison, a more comprehensive analysis of training time, memory requirements, and scalability to larger models would help practitioners assess practical deployment.
4. **Lack of Human Evaluation for Interpretability:** While the paper provides extensive automated evaluation, it lacks human studies to validate whether the discovered concept-mechanism alignments are actually meaningful and useful to human users. Automated metrics like CLIP score don't necessarily correlate with human-judged interpretability quality. The approaches of ACE (Towards Automatic Concept-based Explanations) and CRAFT(CRAFT: Concept Recursive Activation FacTorization for Explainability) to measure the interpretability quality could be considered

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
