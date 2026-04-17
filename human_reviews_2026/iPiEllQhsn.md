# InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Current learning-based subject customization approaches, predominantly relying on U-Net architectures, suffer from limited generalization ability and compromised image quality. Meanwhile, optimization-based methods require subject-specific fine-tuning, which inevitably degrades textual controllability. To address these challenges, we propose InstantCharacter—a scalable framework for character customization built upon a foundation diffusion transformer. InstantCharacter demonstrates three fundamental advantages: first, it achieves open-domain personalization across diverse character appearances, poses, and styles while maintaining high-fidelity results. Second, we introduce a scalable dual-adapter architecture with stacked transformer encoders, which effectively processes open-domain character features and seamlessly interacts with the latent space of modern diffusion transformers. Third, to effectively train the framework, we construct a large-scale character dataset containing 10-million-level samples. The dataset is systematically organized into paired (multi-view character) and unpaired (text-image combinations) subsets. Our dual-adapter structure addresses the challenge of generating multi-character images by enhancing subject consistency through the image adapter and improving layout control of multiple subjects through the text adapter. Qualitative experiments demonstrate the advanced capabilities of InstantCharacter in generating high-fidelity, text-controllable, and character-consistent images, setting a new benchmark for character-driven image generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors of the paper propose a solution for customized text-to-image generation. 
They leverage adapter training on custom data and leave the diffusion network untouched.
There are multiple adapters for reference image and text processing, which come from preselected image and text feature extraction models. Only adapters get trained, and everything else remains untouched. They propose 3 stages of the training process, which have different purposes. The authors claim that their method can generate multiple characters as well.

### Strengths
1. Constructing a custom dataset with 10 million examples for character customization
2. Adapter-based text-to-image customization approach, which is flexible and avoids touching the diffusion model or causing knowledge loss.

### Weaknesses
1. Poor results on almost all quantitative comparisons. The superiority of the proposed model is not justified with quantitative comparisons. The results are mostly worse than other competitive solutions. This also questions the fairness of the example selection of qualitative comparisons.
    
   Specifically, in the "Quantitative Results" section, the claim about UNO performance is noticeably poor. To justify UNO's better performance in Tab 1 and Tab 2, the authors refer to a qualitative comparison. For a fair comparison, the claim and the justification should be in the same field: either both should be compared in qualitative results or in quantitative results. 

2. Limited scientific novelty. Even though the authors have done an extensive job, there is nothing unique or new.

3. The paper is poorly written.

    a) In expression 1, there is F and F^{Q}, but in the description, F is explained as a concatenation result of F^{siglip}, F^{dino}; meanwhile, F is the output of the attention.

    b) (small note) In expressions 2 and 3, H is noted as the hidden features of DiT. However, in expression 4, H is for text embeddings.

    c) In section 3.1.1 (208-211), the authors mention that the output features of the image encoder, F^{siglip}_{l} and F^{Dino}_{r}, go through separate encoders (multiple encoders) for further processing. What encoders are they? It can't be the image adapter, cause the authors mention separate and multiple encoders. There are no details about them.

### Questions
1. Would you give more details about "learnable queries" you mentioned in multiple places (e.g., 236)?
2. In the ablation study, how are reference image features used when the Transformer Encoder is removed?
3. In Figure 2, how are the hidden features of DiT described as image and textual features, in 2 separate groups? Aren't the Text Adapter outputs and textual embeddings supposed to have an attention before being injected into DiT?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a scalable framework, called InstantCharacter, for character customization built upon a DiT-based diffusion model: Flux. InstantCharacter consists of three key components: a scalable dual-adapter architecture that parses character features and interacts with DiTs latent space, a progressive three-stage training strategy that separates training for character consistency, text editability, and visual fidelity, and a new pipeline for constructing training data pairs for multi-character customization.

### Strengths
1. The images generated by the proposed method for character customization are plausible and impressive.
2. This paper is well-written and well-organized.
3. This paper provides a versatile 10-million-level character dataset, which contains paired (multi-view character) and unpaired (text-image combinations) subsets.
4. Extensive experiments are conducted to evaluate the performance of the proposed method.

### Weaknesses
1. What are the objective loss functions used in the three training stages of this paper? The first stage involves the reconstruction of the input image, which is presumably achieved using standard diffusion loss. However, both the second and third stages involve transformations of the original image; what loss functions are used in these stages?

2. As shown in Tables 1 and 2, the quantitative results of the proposed method do not seem very satisfactory, as it fails to demonstrate a clear advantage over existing baseline methods.

3. The ablation studies conducted in this paper are not sufficient, and the effects of many important components have not been evaluated. For example, what would be the difference in performance with and without the text adapter? What would be the difference in performance with and without the dual-stream feature fusion strategy (Section 3.1.1)?

4. What is the time efficiency of different methods? Some evaluations regarding this should be conducted.

### Questions
Please see **Weaknesses**.

Others:

Does the proposed method in this paper support 3-character(or more) personalization? Currently, many existing methods do not limit the number of concepts when performing multi-concept personalization.

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
This paper presents InstantCharacter, a novel and scalable framework for character customization built upon a foundation Diffusion Transformer (DiT). This work effectively addresses the critical gap left by previous U-Net and optimization-based methods, which suffered from limited generalization and compromised textual controllability. The core technical contributions include a scalable dual-adapter architecture for injecting character-specific features and enhancing multi-subject layout control , complemented by an effective three-stage progressive training strategy. Furthermore, the authors construct a large-scale (10-million-level) character dataset for training the framework. Experimental results demonstrate superior performance in generating high-fidelity, text-controllable, and character-consistent images across diverse appearances and styles.

### Strengths
* The work is the first to develop a DiT-based framework specifically optimized for character customization, introducing a novel dual-adapter design (Image Adapter and Text Adapter) that seamlessly interacts with the DiT's latent space to maintain high-fidelity results.
* The proposed three-stage progressive training strategy is highly effective in accommodating the heterogeneous 10M dataset, successfully decoupling the training for character consistency, textual controllability, and image fidelity.
* The comparative experiments demonstrate the method's superior capabilities in consistently preserving character identity and high facial fidelity while maintaining precise text controllability, showing excellent potential for real-world applications.
* Introducing the new Character350 evaluation benchmark.

### Weaknesses
The paper repeatedly emphasizes the “scalability” of its framework, yet only briefly mentions its Transformer-based adapter design. 
However, the paper lacks a rigorous technical argument or experimental evidence to convincingly justify why this DiT-based dual-adapter approach holds a tangible advantage over U-Net-based adapters or other micro-tuning techniques specifically when scaling to significantly larger DiT models. This central claim requires more thorough substantiation

### Questions
*  The 10-million-level dataset is a fundamental component of the model's success. It is highly recommended that the authors provide a more detailed and comprehensive explanation in the main text or supplementary material regarding the dataset's construction, cleaning, and filtering standards, as this information is crucial for reproducibility and understanding the model's performance.
* Please unify the formatting of the first column across all tables in the paper (e.g., consistently bold or consistently non-bold) for improved visual consistency.

### Soundness
2

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
4

### Summary
This paper proposes "InstantCharacter," a framework for character customization built on the FLUX.1 diffusion transformer (DiT) backbone. It introduces a dual-adapter architecture to address the limitations of prior U-Net and tuning-based methods. An Image Adapter injects multi-level character features (from SigLIP and DINOv2) into the DiT's image tokens to ensure character consistency. Concurrently, a Text Adapter injects character features into the text tokens, which is claimed to improve layout control, especially for multi-character scenarios. The method is trained on a massive 10-million sample dataset using a progressive three-stage strategy to balance consistency, controllability, and image fidelity.

### Strengths
1. Significant Engineering Effort: The authors demonstrate a substantial engineering effort, including the curation of a massive 10M-sample dataset, the implementation of a complex multi-stage/multi-resolution training pipeline, and a novel synthetic data-generation loop for multi-character images.

1. Thoughtful Architecture for DiT: The design of the Image Adapter, which uses stacked transformers to process multi-level features (low-level, region-level, and semantic) from multiple encoders (SigLIP + DINOv2), is a thoughtful approach to capturing a robust character representation suitable for the DiT.

1. Multi-Character Handling: The explicit inclusion of a Text Adapter to manage multi-character generation is a valuable design choice. This design choice directly addresses a common failure point in personalization models.

### Weaknesses
1. Outdated Premise and Inaccurate Framing: The paper's primary motivation, posing itself as a superior alternative to U-Net-based adapters, is largely outdated. The SOTA research frontier has decisively shifted to DiT-based methods for some time. This inaccurate framing extends to its claims of being the "first DiT-based framework", which is factually contradicted by the paper's own citations and comparisons to other concurrent DiT-based methods .

2. Incremental Contribution: Viewed in the correct context (as one of many DiT-adapter methods), the methodological novelty is limited. The dual-adapter approach (Image + Text adapters) is a logical, but not highly innovative, recombination of existing concepts (e.g., IP-Adapter's cross-attention injection, PhotoMaker's fused embeddings) applied to a DiT.

3. Critically Incomplete Baseline Comparisons: The comparisons do not reflect the true SOTA and are missing the actual competitors.

- It omits the dominant U-Net/SDXL-based SOTA methods that set the community benchmark, namely InstantID and PhotoMaker. A SOTA claim is impossible without comparing to them.

- More importantly, it fails to compare against or even acknowledge other advanced DiT-native personalization methods (e.g., FLUX-Kontext), which represent the true state-of-the-art for this backbone. This makes the paper's performance unevaluated against its true peers.

4. Incomplete and Ambiguous Ablation Study: The paper's core contribution is its "dual-adapter" architecture, but the ablation study fails to scientifically validate this specific design choice.

- The Text Adapter's contribution is unproven. The paper claims the Text Adapter is crucial for multi-character layout and separation . To prove this, an ablation study w/o Text Adapter should have been run. 

5. Unsatisfactory Qualitative Results & Model Bias: Despite claims of high fidelity, the qualitative results in the appendix (Figure S9) reveal significant failures in ID consistency. The model shows a strong stylistic bias. For example, when given 2D cartoon characters (e.g., columns 1 and 3) and the prompt "a {character} wearing sunglasses, rain", the generated outputs are rendered as 3D realistic characters, and the clothing is noticeably changed. This demonstrates a failure to preserve the core style and details of the reference character, undermining the paper's central claims.

6. Dependence on Proprietary Data: The method's performance is inextricably linked to a massive, 10-million-sample proprietary dataset. This makes the results non-reproducible and makes it impossible to disentangle the contribution of the architecture from the contribution of the data.

### Questions
Please refer to Weakness section. I will consider raise my score if all my concerns are well addressed.

### Soundness
2

### Presentation
2

### Contribution
2
