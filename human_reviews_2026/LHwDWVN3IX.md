# EDITOR: Effective and Interpretable Prompt Inversion for Text-to-Image Diffusion Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Text-to-image generation models (e.g., Stable Diffusion) have achieved significant advancements, enabling the creation of high-quality and realistic images based on textual descriptions. Prompt inversion, the task of identifying the textual prompt used to generate a specific artifact, holds significant potential for applications including data attribution, model provenance, and watermarking validation. Recent studies introduced a delayed projection scheme to optimize for prompts representative of the vocabulary space, though challenges in semantic fluency and efficiency remain. Advanced image captioning models or visual language models can generate highly interpretable prompts, but they often lack in image similarity. In this paper, we propose a prompt inversion technique called EDITOR for text-to-image diffusion models, which includes initializing embeddings using a pre-trained image captioning model, refining them through reverse-engineering in the latent space, and converting them to texts using an embedding-to-text model. Our experiments on the widely-used datasets, such as MS COCO, LAION, and Flickr, show that our method outperforms existing methods in terms of image similarity, textual alignment, prompt interpretability and generalizability. We further illustrate the application of our generated prompts in tasks such as cross-concept image synthesis, concept manipulation, evolutionary multi-concept generation and unsupervised segmentation. Code: https://anonymous.4open.science/r/EDITOR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EDITOER for inverting prompts in text-to-image diffusion models. Specifically, EDITOR first initializes a prompt using image captioning model, then optimize the text embedding of the initialized prompt based on the image reconstruction, and finally applies a pretrained embedding-to-text model to retrieve the prompt. Experiments show that the proposed method improves image similarity, textual alignment, and prompt interpretability, and can be effectively applied to various applications.

### Strengths
1.	The paper clearly points out the challenge of current prompt inversion and proposes an effective solution.
2.	Extensive experiments validate the advancement of the proposed method. The application study is interesting and further shows the potential of the method.

### Weaknesses
1.	Reverse-engineering for latent text embedding has been explored in previous study [1], but is not discussed in this work. Also, it would also be interesting to see the effect of initialization with the DDIM null text inversion mentioned in [1], which is widely used in the text-to-image model for maintaining the image content.
2.	The key technical innovation locates in the embedding-to-text model, but it is based on a preliminary work, not newly proposed here. So, the technical innovation of this paper is marginal. 
3.	The description of the embedding inversion is unclear. (1) It needs to train two E2T models, $M_{corr}$ and $M_{zero}$?  (2) The training details of embedding-to-text model are missing, e.g., the number of training text-representation pairs. (3) The details of beam search.
4.	The paper lacks an ablation study comparing the use of $M_{zero}$ only with using $M_{zero}$ and $M_{corr}$.
5.	Line 200 mentions “attacker”, but this term is neither explained nor referenced elsewhere in the paper.
6.	There are errors/typos in the paper. (1) Line 284 $T(y)$ (2) Line 836, wrong description for Figure 9.

### Questions
See the weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces EDITOR, a prompt‑inversion pipeline for text‑to‑image diffusion models that aims to recover readable prompts that reliably reproduce a target image. EDITOR initializes from an image‑captioning prompt, optimizes the text encoder’s contextual embedding in continuous space to match the target image under a fixed seed, and converts the optimized embedding back to text with an embedding‑to‑text (E2T) model plus a small correction module.

### Strengths
1. Optimizing contextual embeddings after the transformer and deferring discrete text decoding to an E2T model is novel

2. EDITOR improves image similarity and prompt interpretability/text alignment.

### Weaknesses
1. EDITOR depends on a trained E2T module;  this adds implementation and computation costs.

2. Mapping embedding to text to embedding may not be strict,  paraphrases can drift semantics. The extent to which this harms re-generation fidelity and editability is under-measured. Authors are suggested to give more details.

3. Experiments focus on COCO/LAION/Flickr subsets.The scale of the dataset is relatively limited.

### Questions
1. The pipeline introduces a trained embedding-to-text (E2T) model and a correction module, increasing complexity. The paper gives limited profiling of training time, memory, and sample efficiency.

2. What about the performance on more datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces EDITOR, a prompt-inversion pipeline for text-to-image diffusion models that aims to produce prompts that are interpretable and effective at re-generating the target image. It (i) initializes from a caption, (ii) optimizes the contextual text embedding directly in the encoder’s continuous space to reconstruct the given image, and (iii) maps that optimized embedding back to fluent text via an embedding-to-text (E2T) model with a small beam-search correction. Experiments on COCO/LAION/Flickr and transfer tests (e.g., SDXL-Turbo, SD3.5-Medium) report consistent gains vs. PEZ/PH2P and caption-only baselines.

### Strengths
- The paper provides a clean, modular pipeline that others can readily reuse.
- Better similarity and more fluent prompts than PEZ/PH2P and captioners across multiple datasets and model variants.
- Produces prompts that are human-interpretable, aiding provenance/attribution and even downstream editing.

### Weaknesses
- The method composes established components; the main idea (optimize contextual embeddings, then decode to text) is a practical tweak rather than a new paradigm.
- Only 100 images per dataset is used for evaluation; it's unclear how stable gains are across broader distributions.
- Protocol choices (initialization, token/step budgets) could affect PEZ/PH2P competitiveness; a standardized compute budget table would be good to have.

### Questions
1. Add confidence intervals/paired tests and per-image scatter plots for key tables to show variance/outliers.
2. Provide per-stage cost (init, inversion by iteration, etc.), plus scaling with prompt length and denoising steps.
3. Provide sensitivity analysis to noise seeds and to the choice/mixture of caption initializers.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a method for generating fluent, human-readable prompts. The approach begins by initializing with prompts produced by an image captioning model, which serve as a starting point for optimizing embeddings within the continuous latent space of diffusion models. These optimized embeddings are then transformed back into natural language using an embedding-to-text model. The effectiveness of the proposed technique is demonstrated through experimental comparisons with PEZ and PH2P across the MS COCO, LAION, and Flickr datasets.

### Strengths
1. The work addresses an interesting problem of reverse engineering diffusion models. 

2.Comprehensive evaluations and ablations show the effectiveness of the approach in comparison to prior work.

### Weaknesses
1.The work has limited novelty in the sense that it combines the gradient based optimization of prior work with the latent space of an existing model. 

2.The notations and equations are incorrect. The cross-entropy loss and the MLE loss are not correctly defined in equation 4 and 6. 

3. The approach does not consider recent multimodal architectures such as SD3.

4. Comparison to recent prompt inversion/search techniques such as [1].
[1] STEPS: Sequential Probability Tensor Estimation for Text-to-Image Hard Prompt  Search. CVPR 2025.

### Questions
1. Can the authors revisit and clarify the notations and equations. What is the difference between text decoder D and M.textdecoder in the algorithm. Also image is denoted by \mathbf{x} or $x$ at places. 

2. How does the approach extend to embedding spaces of the more recent architectures like SD3? Can this be mapped to the embedding of the captioning model? 

3. How does it compare to the recent work on prompt search which also generates human readable prompts?

### Soundness
3

### Presentation
3

### Contribution
2
