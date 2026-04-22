# Diffusion Transformers with Representation Autoencoders

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Latent generative modeling has become the standard strategy for Diffusion Transformers (DiTs), but the autoencoder has barely evolved. Most DiTs still use the legacy VAE encoder, which introduces several limitations: large convolutional backbones that compromise architectural simplicity, low-dimensional latent spaces that restrict information capacity, and weak representations resulting from purely reconstruction-based training. In this work, we investigate replacing the VAE encoder–decoder with pretrained representation encoders (e.g., DINO, SigLIP, MAE) combined with trained decoders, forming what we call \emph{Representation Autoencoders} (RAEs). These models provide both high-quality reconstructions and semantically rich latent spaces, while allowing for a scalable transformer-based architecture. A key challenge arises in enabling diffusion transformers to operate effectively within these high-dimensional representations. We analyze the sources of this difficulty, propose theoretically motivated solutions, and validate them empirically. Our approach achieves faster convergence without auxiliary representation alignment losses. Using a DiT variant with a lightweight wide DDT-head, we demonstrate state-of-the-art image generation performance, reaching FIDs of 1.18 @256 resolution and 1.13 @512 on ImageNet.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduced Represenation Autoencoders, a new paradigm for choosing the latent encoder and decoder while training diffusion transformers. Rather than the conventionally used autoencoder SD-VAE, the authors claim that utilizing a preexisting foundation model as the encoder is more suited for training diffusion models since they contain semantic representation as well as portray excellent reconstruction ability. With the newly proposed autoencoder and additional design choices, specifically, (a higher dimensinonal latent space for the transfomer, utilizing a DDT head at the end of the diffusion transformer, a revised noise schedule to suit for Dinov2 latent space) the authors achieve state-of-the-art results in 256x256 and 512x512 generation in ImageNet 1M dataset. With the proposed approach additionally, the authors obtain a representation that seems to be cfg robust.

### Strengths
1. The introduced methods, along with the design choices, significantly reduce the number of convergence epochs of training diffusion transformers in the ImageNet 1M dataset.
2. The proposed method seems to be robust with repect to cfg and achieves similar performance with and without cfg. This may enable faster generation with less compute overhead. I would advise the authors to consider this angle while presenting this work as a model that is robust to different cfg values as well as portray good performance without cfg would be useful to the community.
3. The authors have tried extensive experiments to make the DiT model work with Dino-v2  based representation encoder on ImageNet dataset. 
4. Finally the paper is well written and each design choice is clearly explained.

### Weaknesses
1. The authors have done a good job presenting the rFID numbers and extensive experiments for Dinov2, however, Dinov2 is known to discard information relevant for perfect reconstruction during the encoding process. (Ex:- In Fig 9 , on the bus, all the text-based information is lost, moreover, the headlight bulbs fused into one, and the lines in the bus got joined or bent ) Given that such information cannot be captured by the encoder, how can a generation process be guided by such features? Moreover, looking at Fig 13, 14, 15, there are multiple cases like this where the generated images lose semantic details.
2. How easy is it for RAE to generalize for text-to-image generation? Dinov2 is known to not preserve colours, textual information and straight lines. Would this impact the performance of RAE and its generalizability towards text-to-image generation?
3. Dinov2 features are not discriminative towards instances of the same object class, for example:  if two people are present in the scene, Dinov2 would give similar features for the hands of the two people. Assuming the text prompt is "raise the left hand for the person on the left" would this cause a problem towards the model being able  to identify this with text since all four hands in this case would be given similar features? Some examples can be seen in Figure 11/13 where multiple parrots/ dogs fuse to one.
4. The SDVAE was originally trained for LAION dataset, and the size of the encoder and decoder was made to obtain the best performance in LAION. How does RAE autoencoder perform in LAION? Is a bigger encoder-decoder needed for RAE to work with the LAOIN dataset? As an actionable item, I would like the author's clarification of the performance RAE encoder-decoder on LAION, specifically on the PSNR, SSIM and rFID 
5.  There is a performance oscillation for  DiT-DH-XL with different versions of Dinov2 encoders as portrayed in Table 6, and they don't scale linearly with parameter sizes. Why is this? Would this mean that the models may not scale with larger foundation model encoders in the future?
6. In Ln 448-450, the authors claim that "the diffusion models trained for 256x256 could be generalized to 512x512" is this claim valid for other datasets than ImageNet where more number of semantic objects are involved in the same scene?
7. I personally applaud the authors efforts in trying to make the model work with Dino-v2 latent space, however a concern I have is , is such kind of extensive encoder-dependent tuning necessary if we are to swap in a different encoder? Wouldn't this reduce the utility of the approach?

### Questions
1. What is the compute in GFLOPS due to the extra 163 M parameters introduced in RAE from Table 6 in RAE? Does this exceed the SDVAE compute cost?
2. Figure 8a and b are a bit confusing; the convergence scores at 10^11 training iterations is different for Dit-DH-XL, Could the authors please clarify this?
3.  Figure 2 portrays that the encoder of RAE has a compute cost of 22GFlops. However, the authors have not clarified which Dinov2 version this refers to. Could the authors please clarify this? 
4. Could the authors provide the performance for Table 7 DiT-DH-XL at 20 and 80 epochs in the presence of cfg?
5. Can a smaller encoder-decoder be trained for SDVAE so that it is tuned for ImageNet 1M. Would such an encoder present better reconstruction performance with smaller number of parameters?
6. By the design choices, would RAE need a big diffusion model and a large dino v2 encoder to train diffusion models on smaller datasets like FFHQ? 
7. In Table 9, Upsample is spelled wrong.
8. While training  diffusion transformers for smaller datasets like FFHQ wouldn't the limitation of large latent space hold and wouln't the model be severely overparametrized? Additionally, if the method is not generalizable towards text to image generation and infeasible towards smaller datasets, wouldn't this limit the scope of the work?

### Soundness
2

### Presentation
4

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
This paper discusses how to replace the VAE encoder in LDMs with pretrained representation encoders (e.g., DINOv2). The authors conduct a thorough analysis around this topic, and the paper is well-written, clear, and provides meaningful insights.

### Strengths
1. Introducing semantic tokenizers into generative diffusion models is a valuable and timely exploration. This idea may also be extended to unified MLLMs to alleviate the representational conflict between generation and understanding tasks caused by different semantic and pixel-level granularity requirements.

2. The paper is highly readable and the proposed method is elegant. The analyses on model width, noise schedule, and the proposed DDT head provide clear insights and further improve the effectiveness of semantic tokenizers for generation tasks.

3. Extensive experiments and analyses demonstrate that the proposed method achieves SOTA performance.

### Weaknesses
1. The results in Table 1 require further clarification.  RAEs with frozen encoders achieve consistently better reconstruction quality (rFID) than SD-VAE (lines 156–159). However, concurrent works [1, 2] reported that semantic tokenizers tend to decrease reconstruction fidelity.  Additional explanations for why rFID improves, along with more experiments, could help further clarify this point.

2. DINOv2 was trained on hundreds of millions of images [3], which may include ImageNet-like, object-centric samples. This raises potential concerns about evaluation fairness. It would be helpful if the authors could clarify how to avoid the data leakage risk.

3. The notably low gFID on ImageNet may raise concerns about potential overfitting. Since incorporating semantic latents could simplify the latent structure [4], a (large) pretrained tokenizer might more easily memorize simplified semantic distributions. Further OOD evaluations would help confirm the generality of the improvement.

4. It would be interesting to explore whether a smaller-width DiT, combined with a simple learnable linear module, could achieve comparable performance of the d>n setting. For example, consider the model $s_{\theta}(x_t) = Wx_t + Ug(V^Tx_t)$ where $V \in \mathbb{R}^{n \times r}$ and $r \ll n$. According to Theorem 4.2 in [5], the oracle score can be expressed as a linear term with a low-rank correction. This could potentially help alleviate the need for a large model width.

5. It would be interesting to discuss whether such semantic tokenizers could be incorporated into unified MLLMs (e.g., for joint generation, understanding, and editing) to mitigate representational conflicts between different tasks [6].

[1] Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Generation.

[2] Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models.

[3] DINOv2: Learning Robust Visual Features without Supervision.

[4] Masked Autoencoders Are Effective Tokenizers for Diffusion Models.

[5] Can Diffusion Models Learn Hidden Inter-Feature Rules Behind Images?

[6] Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new replacement to VAE autoencoder (used for downstream Diffusion Transformer models) with Representation Autoencoder (RAE). The authors argue that the traditional UNET based VAEs are computationally heavy and provide a weak latent space that leads to lower performance. Their RAE takes a pretrained representation encoder (like DINOv2) and lightweight ViT-based decoder to solve this issue. This new Autoencoder is not directly applicable to the DiT architectures 1) The image reconstruction is limited, hence a wide DiT is used 2) noise schedule shift is employed to account for increased dimensions 3) The latents are augmented with noise to inject stochasticity. To scale however, a shallow DDT head is attached to the DiT backbone which achieves FID scores of 1.18 on ImageNet-256

### Strengths
1)  The proposed shallow DiT + RAE method achieves an FID of 1.18 on ImageNet 256x256 and 1.13 on 512x512, outperforming all prior diffusion models.

2) The method converges faster than VAE-based baselines  for example 10x speedup over VA-VAE while maintaining the FID scores.

3) The RAE also employs an efficient decoder. This part also contributes to make the model faster.

4) The paper provides a theoretical proof and empirical evidence, that the DiT width should be at least equal to the dimensionality of the tokens.

### Weaknesses
1) The paper does not necessarily prove that the better reconstruction quality equals to better generation quality. This still raises questions on the diffusibility of the the latent space proposed in the paper.

2) The model is suitable for the high computation models, for the highly compressed models in VAE which can have very high compression ratios, this model might not be comparable in the computation cost do to the requirement of dimensionality and DiT width constraints.

3) Again Shallow DiT only benefits high dimension latents , for example it can perform worse that DiT-XL trained on low dim SD-VAE.

4) The new training scheme is unstable as compared to  the standard DiT which is stable with a constant learning rate.

### Questions
1) The fundamental question is what property beyond reconstruction makes a representation diffusable?  and why does DINOv2 succeed where MAE fails?

2) What is the qualitative trade-off of noise-augmented decoding, if the FID improves but is the reconstruction compromised?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the training paradigm of image tokenizers for image generation models. It addresses the limitations of the standard SD-VAE–style tokenizer, whose training primarily depends on reconstruction losses and therefore often yields latent spaces with limited semantic structure. The authors propose an unconventional approach: replacing the learnable encoder with a frozen, pretrained vision encoder and training only the decoder for image reconstruction. To mitigate the resulting challenges—such as a substantially higher latent dimensionality and a more fragile decoder—the paper introduces several architectural and training refinements to the DiT models. The resulting tokenizer demonstrates improved image generation quality and faster convergence.

### Strengths
- This paper targets a very important problem of rich semantics in image tokenization. The proposed tokenizer design is simple yet very effective. 

- The practical and theoretical justifications for the architectural design are very convincing. 

- The empirical results are strong on ImageNet-level evaluations.

### Weaknesses
**Presentation**

My major concern with this paper is that it seems to have been submitted in a huge rush. A throughout proofreading and reformatting is very necessary. Examples: 

- I failed to find Figure 3 and Figure 4 in the main manuscript. 

- Line 175: the two sentences beginning with “Because RAE…” are repetitive.

- Line 345: missing space before 'To'.

- Figure 8 appears earlier than Figure 7 in the manuscript. 


**Technical**

- In Line 013 of the abstract. I personally believe it is not appropriate to classify traditional VAEs as U-Nets. U-Nets typically include skip connections between the encoder and decoder, which are not applicable in image tokenization models.

- In Line 086 of Introduction. Calling the latent space of RAE 'discrete' can be very misleading. Readers might confuse it with typical latent space such as VQ. If the authors wish to retain this term, a clear explanation is required early in the paper; otherwise, readers will remain unclear about its meaning until Section 4.3.

- In line 306, the statement “noise smooths the latent distribution and mitigates OOD issues for the decoder, but also removes fine details” is unconvincing. Since the additional noise is introduced only during training, and the decoder receives clean latents during inference, the argument that noise removes fine details does not hold.

- In Table 9, compared to direct scaling tokens, the upsample method actually increases the gFID by 42%, which can hardly be considered as 'comparable' as stated by the authors. This raises concerns about the scalability of the proposed method.

### Questions
- I understand that the authors aim to ensure fair comparisons by controlling the model size in terms of parameter count. However, having the same number of parameters does not necessarily imply comparable efficiency across all aspects. Could the authors also report the wall-clock training and inference speeds, as well as the FLOPs, of the proposed model relative to the baselines?

- I understand that the rebuttal period is limited, and I'm not requesting significant experiments here. However, for future submissions, including moderate-scale text-to-image experiments would be valuable to more convincingly demonstrate the practical applicability of the proposed method.

### Soundness
3

### Presentation
2

### Contribution
3
