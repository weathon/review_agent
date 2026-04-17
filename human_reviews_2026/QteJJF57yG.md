# WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Visual tokenizer is a critical component for vision generation. However, the existing tokenizers often face unsatisfactory trade-off between compression ratios and reconstruction fidelity. To fill this gap, we introduce a powerful and concise WeTok tokenizer, which surpasses the previous leading tokenizers via two core innovations. (1) Group-wise lookup-free Quantization (GQ). We partition the latent features into groups, and perform lookup-free quantization for each group. As a result, GQ can efficiently overcome memory and computation limitations of prior tokenizers, while achieving a reconstruction breakthrough with more scalable codebooks. (2) Generative Decoding (GD). Different from prior tokenizers, we introduce a generative decoder with a prior of extra noise variable. In this case, GD can probabilistically model the distribution of visual data conditioned on discrete tokens, allowing WeTok to reconstruct visual details, especially at high compression ratios. On the ImageNet 50k validation set, at a high-fidelity setting, WeTok achieves a record-low zero-shot rFID of 0.12, outperforming leading continuous tokenizers like FLUX-VAE (0.18) and SD-VAE 3.5 (0.19) with 400% compression ratio. Furthermore, in a high-compression regime, WeTok achieves a zero-shot rFID of 3.49 at a 768× compression ratio, substantially surpassing Cosmos, which scores 4.57 at only 50% our compression ratio.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose Group-wise Lookup-Free Quantization (GQ) to reduce computational and memory costs while maintaining high-fidelity image reconstruction. This design enables flexible scaling of the codebook to an effectively unlimited size. Furthermore, WeTok introduces a generative decoder that integrates adversarial training, using Gaussian noise as input and quantized tokens as conditional guidance, to mitigate the reconstruction loss typically caused by high compression ratios. Experimental results demonstrate that WeTok achieves superior reconstruction and generation performance even under extremely high compression settings.

### Strengths
1. The authors propose Group-wise Lookup-Free Quantization (GQ) to overcome the computational bottleneck, thereby achieving higher image reconstruction fidelity.

2. The experiments yield promising results, and the ablation studies are comprehensive.

### Weaknesses
1. I am mainly concerned about the compression ratio. As shown in Table 4, when the compression ratio reaches 768×, the number of image tokens is only 8 × 8 = 64. However, the resulting rFID of 8.94 is considerably higher than that reported by TiTok [1], indicating a notable degradation in reconstruction quality under such high compression.

2. The main contribution, Group-wise Quantization (GQ), appears to primarily alleviate computational overhead. However, the paper does not provide experiments demonstrating how this design directly facilitates higher compression ratios. For instance, conducting experiments with a downsampling ratio of 32×32 and a hidden channel size of 64 (8×8) could better illustrate how GQ contributes to improving compression efficiency.

3. The Generative Decoder (GD) introduces Gaussian noise as the generator’s input, which may lead to training instability. To address this issue, the authors propose a two-stage training scheme; however, this approach appears overly complex and potentially difficult to reproduce. It would be helpful to report the performance when training all losses jointly from scratch, to evaluate whether the two-stage procedure is truly necessary.

[1] An image is worth 32 tokens for reconstruction and generation. NIPS 2024

### Questions
1. There is a minor concern regarding GQ. The approximation of the entropy loss is theoretically lower-bounded by BSQ, which performs worse than LFQ. However, Figure 4 in the ablation study shows a consistent improvement as the number of groups increases. It remains unclear where the performance begins to degrade — what is the optimal balance point for the number of groups?

2. There are several existing 1D tokenizer models, such as TiTok [1] and SweetTok [2]. The authors are encouraged to include comparisons with these methods to strengthen the experimental validation and make the results more comprehensive.

[1] An image is worth 32 tokens for reconstruction and generation. NIPS 2024

[2] SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization. ICCV 2025

### Soundness
2

### Presentation
3

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
The paper presents WeTok, which is a discrete tokenizer for improving the tradeoff between compression ratio and reconstruction fidelity. Discrete tokenizer has the benefit of having a higher compression ratio, but long has the problem of low reconstruction fidelity.  

WeTok proposes two key techniques, Group-wise lookup-free Quantization (GQ) and Generative Decoder (GD). 

GQ tries to address the memory and computational cost of Lookup-free-quantization (LFP). It partitions the codebook into groups and performs quantization on each group independently to eliminate token entropy loss as the memory bottleneck.

GD is a generative decoder, instead of the traditional one-step deterministic decoder used in prior methods. 

Qualitative results in Table 3 and Table 4 are relatively strong comparing to the prior state-of-the-art methods

### Strengths
(+) The results presented in the tables are relatively strong, achieving a much larger codebook size and good rFID and PSNR

(+) The proposed two methods (GQ and GD) make sense and are well motivated. GQ seems to be a practical and effective solution for solving the bottleneck of the CE loss

(+) The evaluation compared against many methods in Tables 3 and 4

### Weaknesses
(-) the GD method is not very novel. It is known to the community such diffusion decoder can work, dating back to OpenAI's "Consistency Decoder". In addition, such generative decoder does not come with no cost. First, the decoding time increases, which can limit some of the real-time or latency-sensitive applications. Second, as it is a generative model, the decoder could also hallucinate 

(-) lack of comparison with more state-of-the-art autoencoders. For example, infinity tokenizer (https://cvpr.thecvf.com/virtual/2025/poster/34414). Both work claim to increase codebook size and claim to be SotA in the field, though taking pretty different approaches. Therefore a comparison is worthy for the community. However the paper primarily focuses on older VAEs such as VQGAN and SD-VAE

(-) For generation tasks, such as the ones presented in Supp Mat Table 6, shows the proposed method is only marginally improving against SotA Open-MAGVIT2-AR-XL (Luo et al., 2024) 2.33 vs WeTok-AR-XL (Ours) 2.31 in FID

(-) The model is trained in two stages: "In first stage, we train our WeTok with the reconstruction loss, i.e., Eq. 2, 6 and 8. In the second stage, we adapt the model for generative tasks". This is atypically and lacks of explanation on why this is needed. Traditional VAE are trained with a single stage and its generation task quality is on-par with the proposed method (see the point above)

(-) seems datasets play a role in terms of the model quality (which is expected), as evidenced in Figure 6. However, in results section, all models are trained with different datasets, which complicates the analysis of whether the proposed method is effective or the 400M GD data is more suitable for Coco and ImageNet

### Questions
- "We surprisingly find that while reconstructions from leading models like FLUX-VAE and SD-VAE 3.5 collapse after iterations, WeTok’s
outputs are remarkably robust and converge to a fixed value." Is there any explanation on why?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents WeTok, a novel tokenizer that effectively balances high compression efficiency with high-fidelity image reconstruction, achieving state-of-the-art reconstruction performance.

### Strengths
The paper is well written.

The reconstruction results achieve SOTA performance among existing discrete tokenizer, demonstrating the effectiveness of the proposed framework.

### Weaknesses
## Fairness of Comparison

The comparison in Table 3 appears unfair.  The strong baseline MGVQ is a VQ-based tokenizer, whereas WeTok adopts LSQ, which has already been shown to be more efficient than VQ.  To ensure a fair evaluation, the authors should compare WeTok with an LSQ-based version of MGVQ. 
Furthermore, the MGVQ codebook size is only 8192 × 4, but its effective capacity is actually $2^{52}$, not limited by the nominal codebook size.

## Lack of Novelty
The proposed method shows limited novelty.  Overall, the approach seems like a combination of existing components rather than a fundamentally new idea.  Specifically:
- LSQ has been widely explored in prior works.  
- Group-wise quantization is not new.  
- The Generative Decoder design has already been introduced in previous studies.

The authors should better clarify what unique contribution or new insight WeTok introduces beyond these known elements.

## Missing Discussion on Semantic Tokenizers

The paper lacks discussion and comparison with **semantic tokenizers**, which have proven to be powerful for visual understanding and generation.  Several recent works are relevant and should be considered:

[1] ImageFolder: Autoregressive Image Generation with Folded Tokens. https://arxiv.org/pdf/2410.01756

[2] Factorized Visual Tokenization and Generation. https://arxiv.org/pdf/2411.16681

[3] DualToken: Towards Unifying Visual Understanding and Generation with Dual Visual Vocabularies. https://arxiv.org/pdf/2503.14324

[4] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation. https://arxiv.org/pdf/2412.03069

## Missing High-Compression Evaluation

The authors claim that **WeTok** achieves a **768× compression ratio**, which is indeed a significant advantage. However, there is no **quantitative or qualitative evidence** showing the generation quality under this extreme compression rate. It is recommended that the authors provide **generation results** and **gFID metrics** at the claimed **768× compression level** to substantiate this important claim.

### Questions
NA

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
3

### Summary
This paper introduces WeTok, a new family of discrete visual tokenizers that aims to resolve the long-standing trade-off between compression ratio and reconstruction fidelity in latent-space visual generation. WeTok combines two main innovations: Group-wise Lookup-Free Quantization (GQ) and Generative Decoder (GD). Extensive experiments on ImageNet-50k and MS-COCO show that WeTok achieves state-of-the-art (SOTA) reconstruction metrics and strong performance in zero-shot and class-conditional image generation.

### Strengths
1. The proposed GQ formulation provides a mathematically grounded way to reduce the entropy-loss memory bottleneck in LFQ and BSQ, with a provably smaller approximation error.
2. The paper includes large-scale ablations (quantization types, group numbers, architectures, learning schedules) and comparisons across both high-fidelity and high-compression regimes.
3. The proposed method achieves strong performance on both image reconstruction and AR-based generation results, even surpassing continuous tokenizers at similar compression ratios.

### Weaknesses
1. Diffusion-based decoder for visual reconstruction has been studied in previous literatures[1][2], it would be better to cite these work and further discuss the differences with them.
2. In the ablation study section, it's interesting to see that after converting the decoder to a generative model, the reconstructed images are more realistic. It would be better to include some further discussion or analysis.

[1] Epsilon-VAE: Denoising as Visual Decoding
[2] Diffusion Autoencoders are Scalable Image Tokenizers

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
