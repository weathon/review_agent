# Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
In this work, we propose aligning pretrained visual encoders to serve as tokenizers for latent diffusion models in image generation. Unlike training a variational autoencoder (VAE) from scratch, which primarily emphasizes low-level details, our approach leverages the rich semantic structure of foundation encoders.
We introduce a three-stage alignment strategy called AlignTok: (1) freeze the encoder and train an adapter and a decoder to establish a semantic latent space; (2) jointly optimize all components with an additional semantic preservation loss, enabling the encoder to capture perceptual details while retaining high-level semantics; and (3) refine the decoder for improved reconstruction quality.
This alignment yields semantically rich image tokenizers that benefit diffusion models.
On ImageNet 256$\times$256, our tokenizer accelerates the convergence of diffusion models, reaching a gFID of 1.90 within just 64 epochs, and improves generation both with and without classifier-free guidance. Scaling to LAION, text-to-image models trained with our tokenizer consistently outperforms FLUX VAE and VA-VAE under the same training steps. Overall, our method is simple, scalable, and establishes a semantically grounded paradigm for continuous tokenizer design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes replacing the tokenizer VAE in LDM with pretrained visual encoders, specifically DINOv2, and designs a three-stage training framework. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed approach is simple yet effective which leverages a pretrained DINO model as the encoder and introduces a three-stage training scheme that balances semantic representation and reconstruction quality.
2. Comprehensive experiments verify the effectiveness of the method.

### Weaknesses
1. The motivation, *“Our intuition is that learning semantics is inherently more difficult than learning reconstruction”*,  requires further clarification and supporting evidence.

2. My main concern is that all training and evaluation are conducted on ImageNet without any OOD testing. It is possible that the low gFID results stem from overfitting rather than genuine improvements in generalization, since introducing semantic latent distributions may inherently simplify the latent structure compared to pixel-level ones [1,2].

3. Moreover, the authors use DINOv2 as a frozen encoder, which is pretrained on hundreds of millions of images[3]. This model may already include exposure to datasets similar to ImageNet, making it unclear whether the observed gFID improvement truly arises from the proposed architecture itself, or simply from the pretrained encoder’s prior knowledge of similar data.

4. Although the proposed three-stage training improves reconstruction performance as much as possible, the reconstruction FID still degrades (see Table 5), which limits the applicability of the method as a general-purpose encoder for tasks requiring accurate reconstruction, such as image editing and inpainting.

5. The code and evaluation scripts should be released to ensure reproducibility.

[1] Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models

[2] Masked Autoencoders Are Effective Tokenizers for Diffusion Models

[3] DINOv2: Learning Robust Visual Features without Supervision

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- Generation models (e.g., Diffusion models, flow matching model) employ an autoencoder to reduce the dimension of the image dataset for computational efficiency and generation ability.
- The authors claim that the latent space of the encoder should be semantically meaningful.
- By introducing DINO-v2 initialized encoder and semantic preservation regularization, they achieve faster convergence while training and better fid with the same NFE while inference.

### Strengths
- The author first proposed to directly employ the pretrained vision foundation model, followed by an adaptor as an encoder for training a generative model.
- The proposed architecture is simple but effective.
- The experiments in Table 1 support the effects of each configuration. Specifically, in the Semantic preservation loss table, there is a trade-off relationship between the reconstruction and generation performance
- Compared to the VA-VAE, the proposed method achieves better gFID along various sampling steps and CFG scale. In addition, the proposed latent space makes the generative models converge faster than the latent space of VA-VAE
- Compared to FLUX VAE, the proposed methods achieve better generation performance qualitatively and quantitatively.

### Weaknesses
- The overall training strategy seems heuristic, falling short of scalability. Combining multiple stages into a simple pipeline or providing principled methods to measure when to stop each stage would be helpful for applicability.
- Why DINO-v2 works better than other foundation models? An analysis or theoretical explanation would be helpful
- Lack of analysis of the semantics of the latent space.
- [REPA-E] also incorporates the vision foundation model for training VAE. What is the advantage of the proposed method over the [REPA-E]? 

[REPA-E]: REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers

### Questions
- What is the author's thought about the “generative-friendly latent space”? According to the claim of the paper, it seems that preservation of the latent space of DINOv2 is helpful for training generative models. If so, would it be an optimal and promising direction of constructing latent space to only get closer to the DINO space? Are there any other components that should be considered?
- What  ‘linear probing’ exactly did you use for evaluation, and why? Do all the measurements for linear probing give the same correlation and conclusion?
- The vision foundation model could be huge and computationally expensive. Can you compare the size of the proposed tokneizer with the others?

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
4

### Summary
The authors propose a three stage framework to transform a pre-trained semantic encoder into an image tokenizer. In the first stage, they employ an adaptor that maps features from the frozen encoder into a lower-dimensional latent space (32 dimensions in their experiments). The adaptor and decoder are jointly trained using a reconstruction loss. In the second stage, the encoder is fine-tuned to improve reconstruction quality. However, since this process can degrade semantic capacity, an L2 regularization term is applied on the low-dimensional latent to keep the updated encoder close to its previous representation. Experiments conducted mainly on ImageNet 256 show a 0.46 improvement in FID when using the LightningDiT framework.

### Strengths
The presentation, though sometimes not self-contained, is overall clear.

The empirical performance, evaluated on ImageNet, shows 3.4% improvements on linear probing, on ImageNet 256.

### Weaknesses
1. It's not clear what the overall computational cost is for all stages, when compared to other methods.

2. I am not sure how sensitive the dimensionality of the adaptor is. If we want to generalize to ImageNet 512, does it mean that we only need to double the latent?

3. The use of an L2 loss to preserve semantic consistency may not be robust, especially when the latent space lacks proper normalization.

4. While semantic preservation appears improved, the drop in linear probing accuracy during training raises concerns about the trade-off between reconstruction fidelity and semantic capacity.

5. The decrease with rFID shows that the constraint on the adaptor and the stage-wise framework might impact the reconstruction accuracy.

### Questions
What's the performance before CFG?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new way of image tokenizer training for image generation. It targets the weak semantics of the latent space of traditional VAE tokenizers, which can be caused by the reconstruction-only training supervision. This method builds an image tokenizer based on pre-trained discriminative vision foundation models such as DINO. It introduces a progressive three-stage training method that trains an adaptor and a decoder to reconstruct the images, while maintaining the semantically rich latent space. Experiments on ImageNet and text-to-image generation prove the effectiveness. Extensive ablations justify the design of each training stage.

### Strengths
- This paper targets an important problem of semantic-rich image tokenizer traning. The solution shown is simple and effective. The idea is well presented and illustrated. 

- The extensive ablation and detailed reconstruction / semantic capacity tracking along the training process are very helpful and insightful. 

- The final quantivative evaluation results are strong.

### Weaknesses
- Not technically a weakness, but the final linear probing performance is only ~35%, dropping by a lot compared to the original DINO latent space. What could be the potential ways of further bridging the gap? 

- In the stage 2 training, applying the regularization to the output of $E_p$ instead of $A$ seems to be a more intuitive way and can better preserve the semantics. Could the authors provide some insights on why that does not work well?

- In summary, my main concern is that, although the idea is motivated by leveraging a semantically rich encoder to obtain a more diffusable latent space, the substantial linear-probing performance drop and the observed trade-offs between reconstruction fidelity and semantic preservation make it unclear whether the improved results stem from a genuinely richer semantic space or merely from initializing the tokenizer with a well-pretrained encoder that can also be some other models.

### Questions
- I didn't find detailed comparisons for the costs. Will using the pretrained vision encoder introduce increased memory consumption and encoding latency? 

- Could the authors provide GenEval score comparisons in Table 5? It will provide more comprehensive evaluations on alignment and better justify the core contribution of this paper. 

- Figure 4 right. The baseline VA-VAE fails to train in full training epochs in both setting due to numerical issues. Would it be possible to add another baseline for full-scale comparisons? This is optional as it might require prohibitive time during the rebuttal period.

### Soundness
3

### Presentation
3

### Contribution
3
