# PQGAN: Product-Quantised Image Representation for High-Quality Image Synthesis

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Product quantisation (PQ) is a classical method for scalable vector encoding, yet it has seen limited usage for latent representations in high-fidelity image generation.
In this work, we introduce \textit{PQGAN}, a quantised image autoencoder that integrates PQ into the well-known vector quantisation (VQ) framework of VQGAN and adapts it to the regime of large-scale latent generative models.
PQGAN achieves a noticeable improvement over state-of-the-art methods in terms of reconstruction performance, including both quantisation methods and their continuous counterparts. We achieve a PSNR score of 37dB, where prior work achieves 27dB, 
and are able to reduce the FID, LPIPS, and CMMD score by up to 96\%. 
Our key to success is a thorough analysis of the interaction between codebook size, embedding dimensionality, and subspace factorisation, with vector and scalar quantisation as special cases. We obtain novel findings, such that the performance of VQ and PQ behaves in opposite ways when scaling the embedding dimension. Furthermore, our analysis shows performance trends for PQ that help guide optimal hyperparameter selection.
Finally, we demonstrate that PQGAN can be seamlessly integrated into pre-trained diffusion models. This enables either a significantly faster and more compute-efficient generation, or a doubling of the output resolution at no additional cost, positioning PQ as a strong extension for discrete latent representations in image synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces PQGAN, a novel quantised autoencoder that uses product quantisation (PQ) instead of the traditional vector quantisation (VQ) used in models like VQGAN. The simple yet effective idea is to splits the vector into S subspaces, each quantized separately. This allow a codebook which is exponentially bigger than the VQ codebook, depending on the dimension of the subspaces (with one subspace only, PQ reduces to VQ). 

The motivation behind PQGAN is that standard VQ suffers from training sparsity, codebook collapse, and redundancy: PQGAN addresses this by factorising each latent vector into subspaces, quantising each independently.

On ImageNet 256×256, PQGAN outperforms the  baselines and achieves high fidelity with small codebooks (K = 128–512).

The paper carries on a detailed analysis on codebook usage (in terms of Perplexity and Entropy) and metrics are evaluated against size of the codebook, number of subspaces and latent embedding size.

Finally, PQGAN was integrated into Stable Diffusion 2.1, in three variants improving generation results.

### Strengths
- Novelty: despite being simple, the idea is novel and effective.

- The paper conducts a systematic analysis of how embedding dimension, number of subspaces, and codebook size interact, covering the full spectrum between scalar and vector quantisation.

- Extensive experiments on ImageNet, FFHQ, and LSUN establish both quantitative and qualitative superiority. Reported improvements are substantial and consistent.

- Integration with Stable Diffusion is a useful addition to the method and is carefully validated, showing also computational benefit.

- I have appreciated the analysis of codebook utilisation, with entropy and perplexity evaulations. Codebook utilization is an issue in standard VQ-VAEs 

- The paper is clearly written and well-structured. The presentation of results is well-designed

### Weaknesses
I think the main weakness of the paper is the substantial lack of theoretical foundation: PQ seems to be a well-established technique in signal processing and vector encoding [1], and its application to autoencoders has appeared in other context contexts (e.g., El-Nouby et al., 2023; Mentzer et al., 2020) as highlighted in the paper itself.

A deeper theoretical argument for why PQ improves latent scalability and codebook utilization (rather than relying only on empirical evidence) may strengthen the paper.

The paper also claims that PQ and VQ behave “in opposite ways when scaling the embedding dimension,” but this is again presented empirically without theoretical insight.

[1] Product Quantization for Nearest Neighbor Search, Jegou et al. - TPAMI 2011

### Questions
Please refer to the "Weakness".

I think the paper would benefit from more insights (possibly theoretical) on the reasons why PO obtains such successful results.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces PQGAN, which incorporates product quantisation (PQ) into the VQ-VAE framework. PQGAN achieves state-of-the-art reconstruction performance, significantly improving metrics such as PSNR, FID, LPIPS, and CMMD compared to existing methods. Furthermore, the paper demonstrates that PQGAN can be seamlessly integrated with pre-trained diffusion models, resulting in faster or higher-resolution image generation.

### Strengths
- PQGAN demonstrates significant improvements in image reconstruction compared to state-of-the-art methods.

### Weaknesses
- The novelty of the overall approach is very limited. PQ has been used for some time in the field of VQ-VAE, e.g., UniTok [a]. This paper is more like a technical report than a research paper, and it proposes few new ideas or inspirations.

- The proposed latent adaptation is also very straightforward; reducing the spatial resolution of the VAE and increasing the number of channels are common techniques for reducing computation when training diffusion models. This application does not necessarily demonstrate that PQ brings any additional benefits.

- The comparison with previous works is unfair, as this method uses significantly more tokens than other VQ-VAEs.

[a] Unitok: A unified tokenizer for visual generation and understanding. C Ma, Y Jiang, J Wu, J Yang, X Yu, Z Yuan, B Peng, X Qi. arXiv preprint arXiv:2502.20321

### Questions
- The authors are suggested to highlight the contribution of this work.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PQGAN, a novel way to quantize the latents of the auto encoders using product quantized learning. This method factorizes the high dimensional latent channel into many smaller, independent subspaces, each quantized with their own codebook. This method improve the reconstruction quality of the VAE and achieves a state-of-the-art 37.4 dB PSNR. The authors argue that the spatial resolution is the main bottleneck and through their method, they can operate at a lower spatial resolution but with a much higher channel dimension. This enables the method to generate larger resolution images or get speed up upto 4x.

### Strengths
1)  PQGAN achieves very high reconstruction fidelity, 37.4 dB PSNR, which is higher than 25.3 dB of the standard Stable Diffusion VAE and other methods.

2) The paper demonstrates a novel finding the product quantization improves the reconstruction quality in VAEs

3) The method can either double the output image resolution or achieving a 4x generation speedup at the same resolution, This is achieved with the same cost.

### Weaknesses
1) The paper fails to provide  quantitative comparison (e.g., FID, CLIP Score) between the generations of its adapted PQSD model and the original Stable Diffusion.

2) The independent learning of codebooks might not learn complex correlations as the number of subspaces increase. In the limit it is as if sampling independently from each dimension. The tend in the paper also shows that.

3) Since multiple indices are associated with the same pixel, it is incompatible with the autoregressive models. Only Flow based models and diffusion models can benefit from this.

### Questions
1) The paper does not provide quantitative generative metrics like FID or CLIP score to compare their with other methods. This is the main limitation and it would be good to see these scores to validate the claims in the paper. Why are these evaluations not in the paper? 

2. Discuss more about the tradeoff in weakness 2). What is the cut-off?

3. 50% increase in the inference cost seems high. What is the increase in the training time?

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
2

### Summary
This paper introduces PQGAN, a novel quantised image autoencoder that utilizes Product Quantisation (PQ) to achieve state-of-the-art fidelity for latent image representations, especially for high-quality image synthesis. PQGAN is integrated into pre-trained diffusion models, like Stable Diffusionl, which provides significantly faster and more compute-efficient generation, and improve the output resolution. 
The paper also reports difference behavior of VQ and PQ with respect to embedding dimension.

### Strengths
PQGAN surpasses both existing quantisation methods and continuous autoencoders in reconstruction quality

### Weaknesses
This work is not the first to consider PQ (or RQ) for image compression, which is mentioned in the paper, yet the phrasing is still misleading in some places (abstract). That's being said, the method of this paper is much better. It would be worth ablating and analyzing more why. 

For the mundane reader, it would be worth citing the paper that has introduced/popularized product quantization ("Product quantization for nearest neighbor search"). 

typo L329: benifits

### Questions
From Table 1, I understand that the big improvement in PSNR comes from using a resolution of 32x32, while the 16x16 patch size is only offering a PSNR of 28.3. 
Have you tried larger patch sizes? 

Similarly, RVQ is not evaluated in the context of compression, and only reported with patch size of 8x8. So I wonder if the better performance of PQ in this context is simply due to a better hyper-parameter choice of the patch size. Have you tried replacing PQ by a RVQ?

### Soundness
3

### Presentation
3

### Contribution
3
