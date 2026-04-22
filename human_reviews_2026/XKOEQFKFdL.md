# DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Deep generative models have advanced text-to-online handwriting generation (TOHG), which aims to synthesize realistic pen trajectories conditioned on textual input and style references. However, most existing methods still primarily focus on character- or word-level generation, resulting in inefficiency and a lack of holistic structural modeling when applied to full text lines. To address these issues, we propose DiffInk, the first latent diffusion Transformer framework for full-line handwriting generation. We first introduce InkVAE, a novel sequential variational autoencoder enhanced with two complementary latent-space regularization losses: (1) an OCR-based loss enforcing glyph-level accuracy, and (2) a style-classification loss preserving writing style. This dual regularization yields a semantically structured latent space where character content and writer styles are effectively disentangled. We then introduce InkDiT, a novel latent diffusion Transformer that integrates target text and reference styles to generate coherent pen trajectories. Experimental results demonstrate that DiffInk outperforms existing state-of-the-art (SOTA) methods in both glyph accuracy and style fidelity, while significantly improving generation efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DiffInk, a framework for text-to-online handwriting generation (TOHG) that generates full-line pen trajectories conditioned on input text and a reference handwriting style. DiffInk is based on latent diffusion, consists of two core components: InkVAE and InkDIT.

### Strengths
1.  The application task is rare yet interesting, yielding strong results in both visual and quantitative evaluations. 

2. The method is clearly presented and easy to follow.

3. Each sub-module’s motivation is well justified, and ablation studies are provided.

### Weaknesses
1.  Although commendable performance has been achieved on the specific task, the overall framework more closely resembles an assemblage of existing techniques, limiting its degree of novelty. Specifically, reference [1] already employs a latent diffusion model to generate handwritten text lines, the present work differs mainly in the data format (offline or online) and the way how textual content is encoded. As for the content-encoding strategy, reference [2] likewise adopts a character embedding combined with a lightweight content encoder.

2. There are relatively few baselines, partly because prior work on full-line text generation is scarce. However, given the abundance of baselines for single-character generation, I believe this task can be viewed as a special case of line-level generation and constitutes a valid evaluation setting for assessing fine-grained handwriting style imitation.

3. Several implementation details in the experiment section can be stated more clearly, please refer to the questions (as well as some issues).



----------

[1]. Template-guided cascaded diffusion for stylized handwritten Chinese text-line generation.

[2]. Content and style aware generation of text-line images for handwriting recognition.

### Questions
1. Regarding the evaluation metrics, the handwritten-line recognition model and the writer-identification model both employ the encoder of the proposed InkVAE as their backbone. Is this the exact network obtained during the training of InkVAE, or was an additional network trained separately? 

2. What loss function is used for the writer-identification model? Moreover, given that the training and test sets correspond to disjoint writer classes, why is it valid to perform inference on the test set after training solely on the training set?

3. The method employs data augmentation to enlarge the training set by selecting individual characters from a single character dataset and inserting them into the existing line layouts. Since the synthesized characters appear disconnected, this disrupts the connections, seemingly contradicting the argument made in Section 4.5. What would the performance be if this form of data augmentation were removed? 

4. The comparison of generation speeds needs further clarification. For each method, what batch size was used? Moreover, for the prior diffusion based approaches, does the observed acceleration stem mainly from DDIM’s faster sampling schedule or from the lower dimensionality of the latent space? In addition, what is the relationship between the number of DDIM inference steps and the resulting generation quality?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a well-executed study with significant contributions to the task of end-to-end handwritten text-line generation, demonstrating both theoretical and practical advancements. While the empirical results are strong and the methodology is clearly justified, there are certain limitations in the experimental setup, data synthesis approach, and the exploration of real-world implications. Addressing these concerns would strengthen the work and provide a more comprehensive understanding of the method's impact and limitations.

If the authors can address my doubts or point out areas I may have overlooked, I would be willing to reconsider my evaluation and potentially improve my score.

### Strengths
1. **Significant Empirical Advancement in a Challenging Task**  
   The paper demonstrates a notable and convincing empirical improvement on the relatively underexplored and complex task of end-to-end online handwritten text-line generation. The proposed DiffInk framework achieves superior performance across both quantitative metrics (Content AR/CR, Style Accuracy, DTW) and qualitative visual comparisons, thereby setting a new state-of-the-art benchmark in this domain.

2. **Clear, Well-Structured, and Justified Methodology**  
   The overall framework is presented with exceptional clarity. The design of the two core components, InkVAE and InkDiT, is both logically sound and well-motivated. The introduction of task-specific regularization losses (OCR, style) to structure the latent space offers a direct and effective solution to a clearly identified problem, making the method both easy to follow and reproducible.

3. **Comprehensive and Convincing Ablation Studies**  
   The paper is notably supported by thorough ablation studies that systematically validate the contribution of each key component (e.g., InkVAE regularization losses, InkDiT's content encoder). These experiments provide strong evidence that the observed performance gains are directly attributable to the proposed architectural choices.

4. **Noteworthy Practical Efficiency**  
   In addition to its quality, the method achieves a dramatic speedup in generation (e.g., >800x faster than OLHWG), making it not only more effective but also significantly more practical for real-world applications.

### Weaknesses
1. **Incremental Nature of the Core Technical Contribution**  
   While the empirical results are robust, the core technical innovation can be characterized more as a skillful integration and refinement of existing components (VAE, DiT) rather than a groundbreaking algorithmic advancement.

2. **Fundamental Limitation in Training Data Synthesis**  
   The reliance on a data augmentation strategy that stitches isolated characters into text lines presents a significant methodological limitation. This approach inherently fails to capture the cursive connections and natural co-articulation found in real handwritten text, which likely imposes a ceiling on the model’s ability to generate fully authentic, fluid text lines and raises questions about the underlying sources of its performance gains.

3. **Potential Bias in the Experimental Setup**  
   The decision to compare the method against character-level models extended by a single, shared layout prediction module (from OLHWG) is a practical but potentially unbalanced choice for the baselines. This setup may unfairly attribute the weaknesses of this specific layout module to the baseline generation methods, thus exaggerating DiffInk's advantages in terms of coherence and layout.

4. **Limited Exploration of Practical Impact and Downstream Utility**  
   The paper would benefit from a more comprehensive discussion of its real-world implications. While the OCR data augmentation experiment is a step in the right direction, it uses a relatively weak baseline model, which diminishes the argument that the synthetic data is of sufficiently high quality to improve modern, high-performance systems.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a latent diffusion Transformer for online handwriting generation. Authors cite speed and the ability of conditioning on the  full line as the main problems addressed.

Two main components are:
* InkVAE: VAE trained with additional losses for recognition (forcing generated samples to be recognizable) and writer-identity-style classification (enforcing style-space disentanglement).
* InkDiT: diffusion Transformer (DiT) in the latent space created by InkVAE, conditioned on the embedding of a target text and style reference passed through InkVAE encoder.

Experiments are done on CASIA-OLHWDB (Chinese) dataset, with evaluation in terms of accuracy, style consistency, and generation speed.

### Strengths
* InkVAE design is interesting, novel, and well ablated (ex. Table 2, Figure 2 ablate the auxiliary losses for InkVAE construction). The approach seems generalizable (ex could be similarly applied to cursive handwriting, sketch generation, etc.) 

* Quantitative results are quite strong (most impressively in the generation speed, but also on recognition accuracy and style classification). While two-step approach (first generate bounding boxes, then fill the letters) could also allow for high generation speed (as individual characters could be generated in parallel, at least in the case of Chinese), the proposed one-step approach with InkDiT would be superior in cases like cursive and sketches.
 
* Additional results & ablations (generalizability to English handwriting,use as data augmentation tool for OCR, detailed comparison to two-stage approach) further strengthen the work (in my opinion, they should be more prominently highlighted in the main body - before reading them in the appendix, I assumed the model would generalize poorly beyond Chinese handwriting.

* Assuming the code will be released beside the paper, I believe this will be a significant contribution on which the community can further build.

* The writing is generally very clear, and the set of ablations is comprehensive.

### Weaknesses
Limited evaluation: Main evaluation is performed on a single Chinese dataset. Paper could be significantly strengthened by performing evaluation on other domain, such as other languages (BRUSH, IAM), documents (IAMOnDB), sketches (QuickDraw), math (MathWriting), etc.

Style Conditioning: The model is conditioned on a single style vector $x_{ref}$. It is unclear how it would handle more complex or ambiguous style inputs, such as multiple, varied reference samples from the same writer, or a single reference that contains two different styles. Paper could be strengthened by the evaluation of style mixing, etc.

Text conditioning: The content encoder relies on a learnable codebook for a fixed set of characters. This is a standard approach but presents a hard limitation for out-of-vocabulary (OOV) characters, which is a significant issue for languages like Chinese (esp given ~2.6k characters while the alphabet could expand to 30k). Paper could be strengthened by the discussion around this topic, or by evaluation of image-based conditioning for text.

### Questions
- Model implicitly learns layout from the style reference. Have you considered a more explicit form of control, such as conditioning the InkDiT on a target baseline curve, bounding boxes, or other spatial parameters?

- The paper claims 800x speedup over the baseline. Does 58.47 char/s figure for DiffInk include the conditioning, the 5-step DDIM sampling and the final VAE decoding step?

- Can the model currently handle OOV characters in any way?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of text-to-online handwriting generation (TOHG), focusing on full-line handwritten Chinese text. It introduces a latent diffusion Transformer framework for full-line handwriting synthesis, featuring InkVAE for latent-space modeling and InkDiT for precise pen trajectory generation. Experiments show that DiffInk surpasses current SoTA methods in both glyph accuracy and style fidelity for TOHG.

### Strengths
1. This paper proposes the glyph- & style-aware regularization to preserve character-level structural information and enhance the discriminability of writing styles. The introduced InkVAE can learn more discriminative and representative latent features.
2. The proposed InkDiT performs the diffusion process on the latent space, which is more efficient and effective than previous Diffusion models for TOHG.
3. Experiments show that the proposed DiffInk outperforms the previous SoTA models for TOHG with higher glyph accuracy, style fidelity, and generation efficiency.

### Weaknesses
1. The competing models in Table 1 are slightly few, with only three different models. More competing models for OHG should be included, such as SketchRNN.
2. Although TOHG is less investigated, the generation paradigm of DiffInk is very similar to the existing latent generation model StableDiffusion.

### Questions
1. InkDiT mainly performs diffusion on latent representations, and thus DiffInK heavily relies on the performance of VAEDecoder in InkVAE. The question is whether InkVAE can generalize to unseen writers and text corpora, especially for full-line text generation?

2. Although InkVAE can learn glyph- & style-aware features, the style and glyph information of each handwriting sample are mixed or entangled in latent presentations (extracted by InkVAE). In other words, the style features and textual contents are entangled, and only paired styles & texts can successfully reconstruct the input handwriting sample. Therefore, the concern is whether the X_ref and Content Z in Fig. 3 must attain the overlapped textual contents? Whether the model capable of generating accurate handwriting samples if the textual contents of X_ref and Content Z are completely disjointed?

3. Different from the auto-regressive model that explicitly produces the end-of-sequence, how does the DIffInk know the lengths of generated full-line handwriting trajectories?

### Soundness
3

### Presentation
4

### Contribution
3
