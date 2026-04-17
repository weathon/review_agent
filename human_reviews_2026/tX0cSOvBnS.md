# NeRV-Diffusion: Diffuse Implicit Neural Representation for Video Synthesis

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
We present NeRV-Diffusion, an implicit latent video diffusion model that synthesizes videos via generating neural network weights.
The generated weights can be arranged as the parameters of a convolutional network, forming an implicit neural representation (INR) and decoding into videos with frame indices as the input. Our framework consists of two stages: First, a hypernetwork-based tokenizer that encodes raw videos from pixel space to neural parameter space, and the bottleneck latent serves as INR weights to decode; Second, an implicit diffusion transformer that denoises on the latent INR weights. In contrast to traditional video tokenizers that compress videos into frame-wise feature maps, NeRV-Diffusion generates a video as a compact dedicated neural network. This continuous holistic video representation obviates temporal cross-frame attentions while preserving flexible temporal interpolability. The INR decoder and weight latent feature sublinear complexity overhead regarding video resolution and length increase with additional upsampling layers. To enable Gaussian-distributed neural weights with high expressiveness, we reuse the bottleneck latent across all INR layers, as well as reform its weight modulation, upsampling connection and input coordinates. We also introduce SNR-adaptive loss weighting and scheduled sampling for effective training of the implicit diffusion model. NeRV-Diffusion reaches superior video synthesis quality over previous INR-based models and comparable performance to most recent state-of-the-art non-implicit models on real-world video benchmarks including UCF-101 and Kinetics-600. It also achieves outstanding decoding and generation efficiency when scaling up to high-resolution and long videos.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **NeRV-Diffusion**, a novel implicit video diffusion model that synthesizes videos by generating the weights of an INR. The framework consists of two stages:  

- **Tokenization stage**: A hypernetwork-based encoder compresses raw videos into a compact set of neural weights (i.e., INR parameters). These weights directly instantiate a NeRV  decoder, which reconstructs the video given frame indices as input.  
- **Generation stage**: An DiT is trained in this implicit weight space to denoise from Gaussian noise and generate new INR weights, thereby synthesizing novel videos.  

Experiments show that NeRV-Diffusion matches or surpasses current state-of-the-art non-implicit video generation models (e.g., MAGVITv2-AR, LARP) on UCF-101 and Kinetics-600, while using significantly smaller model and latent sizes. It also supports high-quality **temporal interpolation** and **INR weight interpolation**.

### Strengths
- **Highly innovative**: First work to combine diffusion models with video INR weight generation, establishing a new paradigm of “holistic video synthesis via neural weights.”  
- **Technically solid**: Key architectural improvements to NeRV (e.g., channel-wise parameterization, multi-head affine modulation) are well-motivated and validated via ablation studies.  
- **Comprehensive experiments**: Evaluated on UCF-101 and Kinetics-600 against GANs, autoregressive (AR), and diffusion-based models, with multi-dimensional assessments including reconstruction, generation, frame prediction, and interpolation.

### Weaknesses
- **Lack of quantitative efficiency analysis**: Although “fast decoding” is claimed, the paper does not report inference speed (e.g., FPS), memory footprint, or direct comparisons with LDM/VQ-VAE.  
- **Scalability and resolution limitations**: All experiments are conducted only on 128×128×16 videos. Performance at higher resolutions (e.g., 256×256) or longer sequences (>16 frames) is not evaluated.  
- **Limited analysis of long-video interpolation**: While long videos are generated via time interpolation, it remains unclear whether semantic consistency (e.g., object identity, action logic) is preserved.  
- **Insufficient comparison with recent tokenization methods**, such as LTX Video VAE or WAN VAE.

### Questions
**On Efficiency**:  
- Could you provide quantitative comparisons between NeRV-Diffusion and MAGVITv2 or Latte in terms of **inference latency** and **GPU memory usage**?  
- When generating 128-frame videos, does the per-frame forward pass of the NeRV decoder become a bottleneck?  
- What is the encoding/decoding efficiency when handling 720p video reconstruction and generation?  
- How does the **training efficiency** (e.g., GPU-hours, convergence speed) compare to standard video VAEs?


**On Model Architecture**:  
- What exactly does **Multi-head Affine Modulation** do? How does it enable the reconstruction of a full video from a single set of latent tokens?  

**On Latent Dimensionality**:  
- Each token has a dimension of 128, which seems large compared to recent DiT-based models (e.g., Sora/WAN), whose VAEs use only 16 or 48 dimensions. Why not adopt a lower latent dimension?  
- With only **128 tokens**, how is high-fidelity reconstruction still achievable? What factors determine this efficiency?  
- Could you report **PSNR, SSIM, and LPIPS** scores for reconstruction?  
- Even if training data differs, can you compare **compression rate** and reconstruction quality against LTX Video VAE or WAN 2.2 VAE?

**On Performance Bottlenecks**:  
- Why was the token count fixed at 128? What are the reconstruction and generation results with 512 or 1024 tokens?  
- If one prioritizes **maximum performance over efficiency**, does this method have a relatively low ceiling compared to conventional latent diffusion approaches?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a two-stage, INR-based generative pipeline: (1) a hypernetwork tokenizer encodes a video into a compact latent vector whose entries are used directly as the convolution kernels of a small NeRV decoder; (2) a Diffusion Transformer (DiT) denoises in this weight-token space to generate new videos. The authors have taken measures including KL bottleneck after the encoder, Gaussian bottleneck, channel-wise parameterization to ensure the weights follow (multivariate) normal distribution and facilitate smooth interpolations. On UCF-101 class-conditioned generation and Kinetics-600 frame prediction, the method reports competitive (g)FVD at 128×128, 16 frames, with fast per-frame decoding thanks to the tiny instance-specific decoder. It also shows smooth time and weight-space interpolations.

### Strengths
- The idea of generating leveraging the lightweightness and interpolation ability of INRs to produce videos with good spatiotemporal coherence (without having to rely on cross-attention) is interesting.
- The model is compact and associated with fast decoding. Requiring only 128 tokens, a very small decoder, and a reasonable-sized DiT generator, the model still delivers pretty impressive performance.
- The engineering efforts that make the weight-space diffusion work all make intuitive sense and well supported by the ablation results.

### Weaknesses
- **Narrow scale & task scope.** Though the use of UCF-101 and K600 is standard, all experiments are conducted only at 128x128 resolution with 16 frames, which is acceptable for early proofs-of-concept but could be disadvantageous for those larger baselines, as this setup is too simple to expose scalability at higher resolutions & longer videos.
- **Heavy training.** The two-stage system requires longer training times and increases engineering burden, and it is expected that scaling up this framework might be even heavier; further, the DiT is still sizeable so it is hard to determine the actual runtime speed up saved.
- **Gaussianity is not verified.** The paper doesn't show diagnostics (e.g., per-layer marginal tests, normality plots, etc.) to verify how Gaussian the INR weights actually are, and under what conditions would the Gaussianity assumption be counter-effective?
- **Limited metrics.** The authors should consider reporting more metrics (e.g., tLPIPS, or some other more well-rounded metrics, like those from VBench or WorldScore), especially consider the performance is not reported for the K600 dataset.

I think the paper has good novelty and the combination of diffusion (or generative models in general) with INRs for video generation/synthesis could be of great interests to the community. However, I am slightly concerned with how extendable & scalable this method is and how well it performs after more comprehensive evaluations, but I am willing to raise my score if these concerns are addressed by the authors' response.

### Questions
- What might be the failure modes for the Gaussianity-diffusion trade-off case? Would this be harder to guarantee for larger decoder/videos or videos with more complicated spatiotemporal dynamics?
- To scale up to higher resolutions, which one of NeRV capacity, weight latent size, DiT training, would be the first to break? Did you try increasing token size or decoder width beyond the ablation settings?

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
3

### Summary
NeRV-Diffusion is a novel two-stage generative framework for video synthesis that operates by generating the parameters of an Implicit Neural Representation (INR). Instead of synthesizing videos in pixel space or a traditional latent feature space, it first uses a hypernetwork-based autoencoder to "tokenize" videos, compressing them into a compact set of latent INR weights. This latent space is explicitly regularized to follow a Gaussian distribution. In the second stage, an Implicit Diffusion Transformer (DiT) is trained to model this latent space. To generate a new video, the DiT denoises a random noise vector into a new set of valid INR weights. These weights are then used to parameterize a convolutional INR (a "Generative NeRV"), which can decode the full-length video by taking continuous frame indices as input. The model achieves performance comparable to state-of-the-art non-implicit models on benchmarks like UCF-101 and Kinetics-600, while offering significant efficiency advantages and smooth interpolation properties.

### Strengths
1. By representing the video holistically as a single neural network, the model obviates the need for computationally expensive cross-frame attentions in the denoising network. This directly addresses a major computational bottleneck present in traditional frame-wise video diffusion models.
2. The framework generates a compact, instance-specific decoder (the INR) rather than relying on a large, shared, universal decoder (like a VAE). INRs are known for being fast at the decoding (rendering) stage, making the visualization of generated videos computationally efficient.
3. The model can be trained on sparse frames but generate smooth, high-frame-rate videos simply by sampling more frame indices at inference time (e.g., generating 128 frames from a 16-frame training setup).
4. Traditional tokenizers that create frame-wise feature maps lead to redundant representations by ignoring inter-frame coherence. NeRV-Diffusion's approach of encoding the entire video into a single set of parameters is a much more compact and holistic representation.
5. Despite being a new paradigm, the model demonstrates high-quality results. It significantly outperforms previous INR-based generative models (like DIGAN) and achieves comparable performance to non-implicit SOTA models (like MAGVITV2-AR) on standard gFVD benchmarks.

### Weaknesses
1. The framework is not end-to-end. It relies on a two-stage pipeline: (1) training the NeRV autoencoder, and (2) training the diffusion model on the frozen latent space. The final generation quality is fundamentally bottlenecked by the fidelity and "Gaussian-ness" of the latent space produced by the first stage.
2. The latent space consists of neural network weights, which are highly abstract and have no intuitive spatiotemporal or semantic meaning. This makes granular, controllable editing extremely difficult, if not impossible, without retraining.
3. The core challenge is forcing the INR weight space to be both highly expressive (to reconstruct complex videos) and simple enough to be modeled by a Gaussian distribution (for diffusion). The paper's numerous architectural modifications (multi-head affine, channel-wise parameterization et. al.) highlight the difficulty of balancing this trade-off. 
4. The experiments are conducted on low-resolution (128) and short (16-frame) videos. It is unclear how this framework scales to high-definition (HD/4K) or long-duration videos. A higher-resolution INR would require vastly more parameters, creating a much larger token sequence for the DiT to generate, which is a significant challenge.

### Questions
1. How does the performance (training stability, gFVD) and computational cost scale with respect to (a) video resolution (e.g. 512) and (b) video duration (e.g., 256 frames)? What is the growth rate of the INR parameter count (the token sequence length for the DiT)?
2. Given the abstract nature of the weight space, what are the prospects for conditional generation beyond simple class labels or frame-gaps? Could this framework be guided by text, depth maps, or semantic layouts?
3. How sensitive is the final generation quality to the fidelity of the first-stage autoencoder? How important is the D_KL loss weight in balancing reconstruction fidelity vs. a smooth, generative latent space?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes NeRV-Diffusion, which synthesizes videos by generating implicit neural representation (INR) weights through diffusion models. The method first uses a hypernetwork-based tokenizer to encode videos into neural network parameters, then uses a diffusion transformer to denoise and generate in the weight space. The core idea is to represent videos holistically as a set of INR weights, rather than frame-by-frame feature maps.

### Strengths
The idea of shifting from generating pixels to generating neural network weights is unique in the video generation field. Although there is precedent for INR generation (DIGAN 2022), applying diffusion models to video INR weight space still has exploratory significance.

The combined design of KL bottleneck, multi-head affine modulation, and channel-level parameterization to some extent solves the contradiction between weight distribution regularization and expressive capability.

### Weaknesses
Only compared decoder parameter count (3.5M-55M) and token count (128), missing actual inference speed.

Evaluation metrics are too singular, only using FVD as one metric.

Video generation resolution is low and length is short, all experiments only at 128×128, unable to prove the method can scale to higher resolutions.

The dataset scale used is small and old, unable to verify the method's performance in real large-scale scenarios.

Comparison methods and baselines are outdated, missing comparisons with latest methods such as wan2.1, Self-Forcing, etc., and method metrics are too few, moreover not SOTA, advantages are not obvious.

Lacks qualitative comparison results display and analysis with other methods.

Interpolation experiments lack quantitative metric verification, only on class-conditioned and frame prediction two tasks, unable to prove the method's generalizability, such as text-to-video, image-to-video, video editing, unconditional generation.

The core paradigm (generating INR weights) comes from DIGAN, improvements are mainly engineering tricks (KL constraint, multi-head modulation, scheduled sampling, etc.).

### Questions
Please address the problems in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
1
