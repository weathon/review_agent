# Latent Denoising Makes Good Tokenizers

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 4

## Abstract
Despite their fundamental role, it remains unclear what properties could make tokenizers more effective for generative modeling. We observe that modern generative models share a conceptually similar training objective---reconstructing clean signals from corrupted inputs, such as signals degraded by Gaussian noise or masking---a process we term \emph{denoising}. Motivated by this insight, we propose aligning tokenizer embeddings directly with the downstream denoising objective, encouraging latent embeddings that remain reconstructable even under significant corruption. To achieve this, we introduce the Latent Denoising Tokenizer (\method), a simple yet highly effective tokenizer trained to reconstruct clean images from latent embeddings corrupted via interpolative noise or random masking. Extensive experiments on class-conditioned (ImageNet $256\times256$ and $512\times512$) and text-conditioned (MSCOCO) image generation benchmarks demonstrate that our \method consistently improves generation quality across \textit{six} representative generative models compared to prior tokenizers. Our findings highlight denoising as a fundamental design principle for tokenizer development, and we hope it could motivate new perspectives for future tokenizer design. Code is available at: https://github.com/Jiawei-Yang/DeTok

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on training more effective visual tokenizers for subsequent generative modeling. Motivated by the shared denoising nature of current generative models, it incorporates the denoising objective into tokenizer training to align its embedding with downstream generative models. Specifically, this work introduces Latent Denoising Tokenizer (l-DeTok), a simple yet highly effective tokenizer trained to reconstruct clean images from latent embeddings corrupted via interpolative noise or random masking. Comprehensive experiments on various settings and types of generative models validate the effectiveness and generalization of the trained tokenizers.

### Strengths
1. This paper is well-motivated and easy to follow. It aligns tokenizer embeddings with the downstream denoising objective in generative modeling.
2. The writing of this paper is clear, and the expressions are polished and elegant. I appreciate this point.
3. The proposed method and spirit are general and robust, which is applicable to a wide spectrum of generative models.
4. The paper provides comprehensive experiments and ablations to validate the superiority of the proposed method.

### Weaknesses
1. How much additional computational overhead does the proposed denoising mechanism introduce compared with conventional tokenizer training? Detailed comparisons should be presented.
2. How do the performance gains vary across various tokenizer and generative model sizes? (results with different sizes of both tokenizer and generators)

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Latent Denoising Tokenizer (l-DeTok), a visual tokenizer trained to reconstruct clean images from corrupted latent embeddings using interpolative Gaussian noise and random masking during tokenizer training. The authors demonstrate that it leads to consistent gains for both non-autoregressive and autoregressive models. Extensive experiments on ImageNet and MS-COCO across two resolutions support the claim that a simple, intuition-driven denoising objective can yield broadly useful tokenizer embeddings that transfer across generative paradigms.

### Strengths
1. Generalizes across AR and non-AR generators.
- The same tokenizer improves both diffusion-based (non-AR) and autoregressive models without architectural changes, indicating that the denoising-aligned latent space is broadly compatible with diverse generation mechanisms.

2. Simple and practical method such that no external encoder alignment or semantic distillation required.
- While recent approaches emphasize semantics distillation from powerful pretrained vision models, l-DeTok shows that a lightweight, corruption-robust training objective can match (or surpass) such methods without explicitly aligning to external encoders.

3. Drop-in usability and clear training signal.
- The method is easy to implement, and the paper provides a clear heuristic that practitioners can readily apply.

### Weaknesses
Regarding with Related work, please add the following references.
- Zhao et al., ε-VAE: Denoising as Visual Decoding.
- Tschannen et al., Generative Infinite-Vocabulary Transformers.
- Kim et al., Efficient Generative Modeling with Residual Vector Quantization-Based Tokens.

### Questions
1. Discrete extension of the method.
- Given that the corruption scheme includes random masking, which is naturally defined in discrete token spaces (masking or replacing code indices), how well would the approach extend to discrete tokenizers such as VQ or RVQ? A brief discussion of potential pitfalls, such as codebook collapse, corruption schedule design and any small-scale evidence may extend the scope of the contribution.


2. Orthogonality to aggressive compression and compact regimes.
- Recent work indicates that aggressive latent compression, such as approximately 32 tokens or adaptive-length tokenization, can improve both quality and efficiency. Is the proposed denoising-aligned training orthogonal to these compression strategies, and does combining them yield further gains? In particular, beyond the reported configuration with patch size 16 and latent dimension 16, are there results at more compact representation?

References
- Yu et al., 2024. An Image is Worth 32 Tokens for Reconstruction and Generation
- Duggal et al., 2025. Adaptive Length Image Tokenization via Recurrent Allocation

### Soundness
3

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
This paper proposes a continuous VAE tokenizer that incorporates a denoising prior. The core idea is to corrupt the latent representations by injecting Gaussian noise or applying masking, and then training the tokenizer to reconstruct the original, uncorrupted latents. This design aims to address the discrepancy between the objectives of tokenizer training and subsequent image generative modeling, thereby improving the "denoisability" of the VAE latent space. The authors provide extensive experiments that effectively validate the proposed denoising tokenizer. Additionally, the paper presents several interesting empirical findings, including comparisons between random and fixed masking, as well as interpolative versus additive noise.

### Strengths
- The paper is well-motivated, addressing the critical challenge of aligning the training objectives of visual tokenizers and generative models.
- The proposed method is simple yet effective. The strategy of injecting interpolative or masking noise is conceptually sound and well-justified.
- The methodology and implementation details are presented with clarity, making the work easy to understand and reproduce.
- The experimental evaluation is extensive and well-structured, providing strong empirical support for the paper's claims.

### Weaknesses
- Convergence and scalability: A potential concern is the training convergence. While the denoising objective complements the pixel-reconstruction loss, it is plausible that learning to reconstruct from corrupted latents could slow down convergence compared to a vanilla baseline. It would be beneficial for the authors to provide an analysis of the training speed and computational overhead. Furthermore, a discussion on the scalability of the proposed method to larger models and datasets would strengthen the paper.

- Distribution of noise level: The paper appears to use a uniform distribution for the noise level factor \tau. Drawing inspiration from recent diffusion-based methods (e.g., SD3), which have shown that non-uniform timestep sampling can improve performance, it would be interesting to investigate whether a non-uniform sampling strategy for \tau could offer similar benefits for the proposed tokenizer.

### Questions
see weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the Latent Denoising Tokenizer, a visual tokenizer framework designed to explicitly align with the denoising objectives common to modern generative models. The authors propose training tokenizers to reconstruct images from latent representations subjected to strong interpolative noise and/or random masking, departing from traditional pixel-reconstruction VAE objectives. By doing so, $l$-DeTok aims to produce latent embeddings that are robust and reconstructable under significant corruption, which theoretically matches the requirements of downstream denoising-centric generative models. Empirical evaluation covers six prominent generative models—both autoregressive and non-autoregressive—across ImageNet and MS-COCO text-to-image settings, demonstrating that $l$-DeTok yields consistent improvements over standard, semantics-distilled, and convolutional tokenizers.

### Strengths
- The paper is motivated by an accurate and under-discussed observation: modern generative models, regardless of architecture, are fundamentally denoising systems. Training tokenizers with explicit latent corruption (interpolative noise, masking) is a clean conceptual shift that breaks with the tradition of mere pixel-wise autoencoding. This alignment is theoretically meaningful and empirically justified.

- The authors benchmark $l$-DeTok across a broad spectrum of generative models, on both class-conditional (ImageNet) and text-conditional (COCO) tasks. Baselines include state-of-the-art semantics-distilled tokenizers (e.g., VA-VAE, MAETok), standard VAE-style tokenizers, and convolutional tokenizers. These comparisons are fair, with well-matched training recipes and strong ablations.

- The authors provide clear, actionable analysis of the key components of $l$-DeTok. For instance, Figure 2 and Figure 3 dissect the importance of interpolative versus additive noise and the effects of constant versus randomized masking ratio on FID/IS. This isolates the impact of each design choice, giving both insight and reproducibility.

- Strong Generalization and Architectural-Agnosticism: The improvements hold for both Transformer-based and CNN-based tokenizers, as shown in Section A.5, and do not depend on external semantics distillation resources—important for domains lacking large vision encoders.

### Weaknesses
- Lack of Theoretical Analysis Regarding Optimality or Limitations: The empirical link between denoising-aligned tokenizers and improved downstream performance is clear, but the theoretical rationale is underdeveloped. For example, there is no formal analysis or proof of why interpolative over additive noise leads to strictly more robust or generative-friendly latents (as claimed in Section 5.1.1). While Figure 2 empirically demonstrates this, a mathematical discussion (e.g., in terms of mutual information or denoising risk) would strengthen the scientific value.

- Training/Test Distribution Discrepancy and Impact: Section A.3 (in APPENDIX) raises the training/inference mismatch issue: since the decoder is trained primarily on heavily corrupted inputs but, in deployment, must reconstruct from nearly clean latents, there is a risk of distribution shift. While decoder fine-tuning helps (see Figure A.1, Figure A.2), the paper might understate the downside: performance gains appear to hinge in part on decoder adaptability, not just the quality of the encoder's latents. The extent to which this is a fundamental limitation (versus an optimization artifact) is not fully explored.

- Potential Over-Claims on "Semantics Distillation Independence": The abstract and body at times appear to overemphasize $l$-DeTok’s superiority or independence compared to semantics distillation. However, Section A.4 shows that adding semantics distillation to $l$-DeTok further improves performance—especially for non-AR models—sometimes even surpassing pure denoising. This suggests the two are complementary rather than exclusive. The paper should be more circumspect in presenting $l$-DeTok as a replacement, rather than a supplement, to semantics-based approaches.

- The evaluation should be benchmarked against more powerful and contemporary VAEs, such as the SD3-VAE or Flux-VAE. The currently used SD-VAE is an outdated and underperforming baseline due to its 4-channel latent space. A more meaningful and fair comparison would be against a modern 16-channel VAE. This is a crucial point, as we have observed a concerning trend in recent tokenizer research where comparisons are made against the old SD-VAE to inflate perceived performance gains. Such a comparison is not a fair assessment of the proposed method's true capabilities.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
