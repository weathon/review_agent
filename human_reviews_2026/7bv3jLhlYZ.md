# LaTo:  Landmark-tokenized Diffusion Transformer for Fine-grained Human Face Editing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Recent multimodal models for instruction-based face editing enable semantic manipulation but still struggle with precise attribute control and identity preservation. Structural facial representations such as landmarks are effective for intermediate supervision, yet most existing methods treat them as rigid geometric constraints, which can degrade identity when conditional landmarks deviate significantly from the source (e.g., large expression or pose changes, inaccurate landmark estimates). To address these limitations, we propose LaTo, a landmark-tokenized diffusion transformer for fine-grained, identity-preserving face editing. Our key innovations include: (1) a landmark tokenizer that directly quantizes raw landmark coordinates into discrete facial tokens, obviating the need for dense pixel-wise correspondence; (2) a location-mapped positional encoding and a landmark-aware classifier-free guidance that jointly facilitate flexible yet decoupled interactions among instruction, geometry, and appearance, enabling strong identity preservation; and (3) a landmark predictor that leverages vision–language models to infer target landmarks from instructions and source images, whose structured chain-of-thought improves estimation accuracy and interactive control. To mitigate data scarcity, we curate HFL-150K, to our knowledge the largest benchmark for this task, containing over 150K real face pairs with fine-grained instructions. Extensive experiments show that LaTo outperforms state-of-the-art methods by 7.8% in identity preservation and 4.6% in semantic consistency. Code is available at https://github.com/alibaba/landmark-tokenized-dit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces LaTo, a novel face-editing framework that leverages landmark-tokenized diffusion transformers (DiTs) for identity-preserving and fine-grained facial attribute control. The model quantizes facial landmark coordinates into discrete tokens and uses a landmark tokenizer along with a location-mapping positional encoding. LaTo aims to overcome challenges faced by existing methods by improving computational efficiency, identity preservation, and semantic consistency. The paper further introduces a new dataset, HFL-150K, containing over 150,000 face pairs with fine-grained editing instructions, which is used to train and benchmark LaTo.

### Strengths
- Landmark tokenization is an innovative approach that moves beyond traditional pixel-based conditioning in face editing tasks. This method helps prevent identity drift when landmarks differ significantly from the source image.
- The HFL-150K dataset is a valuable contribution, offering a large collection of high-quality real-world face pairs with fine-grained instructions essential for training advanced face editing models.
- Comprehensive ablation studies and both qualitative and quantitative evaluations demonstrate LaTo’s effectiveness across multiple metrics, supporting its proposed innovations.

### Weaknesses
- The paper mainly focuses on facial expression and head pose editing but provides limited discussion on how well LaTo generalizes to other image domains or tasks (e.g., full-body or complex scene editing). Assessing its broader applicability would strengthen the claims.
- Although HFL-150K is extensive, there is little information about its diversity regarding age, ethnicity, or other demographics—factors crucial for ensuring robust performance across different populations. More details on how these aspects were considered during dataset construction would enhance transparency.
- There is an inconsistency: cfg is mentioned in the Methods section but not in the Abstract.
- The main contribution is not included in the framework, making it confusing and difficult to understand.

### Questions
- Can the model be optimized for real-time use in applications like live avatar creation or video conferencing? What specific optimizations are needed to enable efficient on-the-fly processing?
- How well does LaTo perform across various demographic groups such as age, ethnicity, or gender? Could you provide an in-depth analysis of its results on more diverse faces to ensure fair representation?
- While the paper notes limitations of pixel-wise methods when landmarks deviate from the source image, how does LaTo handle extreme cases where landmark positions are highly inaccurate due to large pose changes, occlusions, or exaggerated expressions? Is there any noticeable degradation in output quality under these conditions?

### Soundness
3

### Presentation
3

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
This paper achieves better facial control while maintaining high identity fidelity by tokenizing key points, as opposed to other methods that condition on rendered landmark images. Furthermore, the integration of a large model supporting text input further enriches the application scenarios. Additionally, the authors propose a new face editing dataset. I believe this is a complete and highly practical paper.

### Strengths
1. By tokenizing key points, this method achieves better facial control while maintaining high identity fidelity, in contrast to other approaches that condition on rendered landmark images.
2. The support for text input further expands the application scenarios.
3. The data and experiments are sufficient, and the writing is clear and easy to understand.

### Weaknesses
1. The DIT structure results in significant resource consumption, and it is necessary to evaluate the inference speed.
2. Keypoints have limited capability in modeling geometric information and are not fully decoupled from identity, which may pose challenges in cross-identity tasks.

### Questions
1. In cross-identity scenarios, could there be an issue with inadequate decoupling between geometry and identity?
2. What is the inference speed?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose LaTo, a landmark-tokenized diffusion transformer for fine-grained and identity-preserving face editing, and curate a face editing dataset HFL-150K. The core designs of LaTo are (i) landmark tokenization of raw coordinates, (ii) location-mapping positional encoding, and (iii) landmark-aware classifier-free guidance. Experiments demonstrate that LaTo achieves state-of-the-art performance.

### Strengths
- The edited images presented in this work demonstrate effective fine-grained control, with experiments showing improved performance over previous state-of-the-art methods. It enables more precise and adaptable face editing, holding significant practical value for relevant applications.
- The overall pipeline is logical and intuitive, and the writing is well-structured.
- Dataset contribution: This work develops an automated synthesis-and-curation pipeline and constructs a face-editing dataset comprising over 150K face pairs annotated with fine-grained instructions.

### Weaknesses
1. Weaknesses in Ablation Studies: Although the paper includes some ablation studies, they are relatively limited. Please provide the rationale for not using traditional landmark extractors as the landmark predictor, along with supporting ablation studies on its utility. Furthermore, it would be beneficial to include ablation analyses on the codebook size and the design of the multi-modal token fuser.
2. Lack of user study. For the fine-grained face editing task, the quantitative objective metrics are insufficient to fully validate the effectiveness of the editing results.

### Questions
1. Why did you choose to use a VLM for landmark prediction instead of traditional, well-established landmark detectors? Were any experiments conducted to demonstrate the effectiveness of your chosen approach?
2. Was the accuracy of the head rotation estimation evaluated qualitatively? (Alternatively, if emphasizing the lack of such evaluation: The experimental evaluation appears incomplete. Specifically, is there any qualitative assessment of the accuracy for the estimated head rotation angles?
3. Was a user study conducted?
4. Were there any ablation studies on the proposed token fusing mechanism? How does it compare to other fusion strategies?
5. How was the size of the codebook determined? Were ablation studies performed to validate the chosen size?

### Soundness
3

### Presentation
3

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
This paper introduces LaTo, a landmark-tokenized diffusion transformer for fine-grained, identity-preserving human face editing. 
LaTo quantizes landmark coordinates into discrete facial tokens, integrating them into diffusion transformers using location-mapped positional encoding. 
It also introduces a landmark predictor based on a fine-tuned vision–language model with structured chain-of-thought reasoning, allowing intuitive user control without explicit landmark input. 
The authors further curate HFL-150K, a large-scale dataset containing 150k real and synthetic face editing pairs with precise annotations. Extensive experiments demonstrate that LaTo achieves superior identity preservation and semantic consistency compared to prior state-of-the-art models.

### Strengths
1. Direct landmark tokenization provides an efficient geometric prior for diffusion transformers.
2. LaTo achieves consistent improvement across benchmarks (HFL-150K, ICE-Bench, GEdit-Bench).
3. The effect of positional encoding and classifier-free guidance is well quantified.
4. HFL-150K is a valuable, large-scale dataset that provides both real and synthetic samples with fine-grained instructions — a useful resource for the community.

### Weaknesses
1. The landmark predictor tends to produce nearly identical landmark configurations for the same expression instruction (e.g., “make him happy strongly”) across subjects, for example, the first four columns of Figure 1.
This over-templating reduces individual expressiveness and may lead to uniform, less personalized facial deformations.
The paper lacks analysis on inter-identity landmark variance.
2. Figure 1 (second to last column) includes a cartoon-style input image as an example. In such cases, landmark prediction becomes substantially more difficult because the facial geometry deviates from human anatomy — e.g., mouths drawn as single lines, exaggerated eyes, missing noses, or highly stylized jawlines. 
These features make standard 68-point facial landmark detection ill-defined.
However, the paper does not clarify whether the proposed landmark predictor is trained or evaluated on such stylized data. 
Given that HFL-150K is described as primarily real-human-face-based (Section 3.1), it is doubtful that the model has seen sufficient cartoon or synthetic faces during training.
3. Although the paper promises public release “upon acceptance,” both LaTo’s code and the HFL-150K dataset are currently unavailable.
For a paper emphasizing both model and dataset contributions, this limits reproducibility.

### Questions
1. How are identity-dependent facial variations (e.g., different muscle or bone structures) handled when generating landmarks for the same expression?
2. How are identity preservation metrics validated against human perceptual ratings? Is there a correlation analysis?
3. What is the sensitivity of the model to noisy or inaccurate landmark predictions from the VLM?
4. Can the proposed landmark predictor generate accurate and stable landmark coordinates for non-realistic, stylized cartoon faces? Does the full LaTo pipeline (landmark tokenization + DiT fusion) maintain editing quality when the input lies outside the natural-face manifold?
5. Can the landmark tokenizer or location-mapping positional encoding be applied to other structured domains, such as full-body or hand motion editing?
6. How does the landmark predictor perform under noisy or ambiguous instructions?
7. Are there any plans to release HFL-150K in subsets (e.g., synthetic only)?

### Soundness
3

### Presentation
3

### Contribution
2
