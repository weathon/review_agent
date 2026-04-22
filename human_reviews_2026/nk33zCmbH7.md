# Scale-DiT: Ultra-High-Resolution Image Generation with Hierarchical Local Attention

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Ultra-high-resolution text-to-image generation demands both fine-grained texture synthesis and globally coherent structure, yet current diffusion models remain constrained to sub-$1K \times 1K$ resolutions due to the prohibitive quadratic complexity of attention and the scarcity of native $4K$ training data. We present **Scale-DiT**, a new diffusion framework that introduces hierarchical local attention with low-resolution global guidance, enabling efficient, scalable, and semantically coherent image synthesis at ultra-high resolutions. Specifically, high-resolution latents are divided into fixed-size local windows to reduce attention complexity from quadratic to near-linear, while a low-resolution latent equipped with scaled positional anchors injects global semantics. A lightweight LoRA adaptation bridges global and local pathways during denoising, ensuring consistency across structure and detail. To maximize inference efficiency, we repermute token sequence in Hilbert curve order and implement a fused-kernel for skipping masked operations, resulting in a GPU-friendly design. Extensive experiments demonstrate that **Scale-DiT** achieves more than $2\times$ faster inference and lower memory usage compared to dense attention baselines, while reliably scaling to $4K \times 4K$ resolution without requiring additional high-resolution training data. On both quantitative benchmarks (FID, IS, CLIP Score) and qualitative comparisons, **Scale-DiT** delivers superior global coherence and sharper local detail, matching or outperforming state-of-the-art methods that rely on native 4K training. Taken together, these results highlight hierarchical local attention with guided low-resolution anchors as a promising and effective approach for advancing ultra-high-resolution image generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents an efficient framework for ultra-high-resolution image generation. Specifically, it uses a standard-resolution image as structure guidance and trains LoRA-based attention layers to attend high-resolution tokens to their counterparts in the low-resolution image. To accelerate practical inference, the paper masks most irrelevant tokens and permutes the token order using Hilbert scanning. Experiments demonstrate the effectiveness and efficiency of the proposed method.

### Strengths
1. The proposed method is straightforward and somewhat novel.
2. The experiments demonstrate the effectiveness of the proposed method

### Weaknesses
1. The figures are not well organized, and the text in nearly all of them is too small to read clearly. This significantly hinders the readability and overall presentation quality.

2. Some related works are missing. For example, HiDiffusion [1] and FreCaS [2] are two recent methods specifically designed for efficient ultra-high-resolution image generation, and a comparison or discussion with these approaches would strengthen the paper’s positioning and context.

3. The training settings for LoRA are not provided. It would be beneficial to report key details such as the rank, layers involved, and training schedule, to ensure reproducibility and clarity.

[1] HiDiffusion: Unlocking High-Resolution Creativity and Efficiency in Low-Resolution Trained Diffusion Models, ECCV 2024.
[2] FreCaS: Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling, ICLR 2024.

### Questions
see weakness

### Soundness
3

### Presentation
2

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
This paper proposes Scale-DiT, a new diffusion framework designed for efficient and effective generation of ultra-high-resolution images. The core ideas are hierarchical local attention and low-resolution global guidance to generate images without needing native high-resolution training data. The approach involves fine-tuning only the multi-modal attention layers using LoRA, significantly reducing training costs. Experiments confirm that this approach achieves notable efficiency, including reduced memory usage and faster inference speeds compared to baselines.

### Strengths
- The method effectively generates ultra-high-resolution images, which is impressive.
- Mixing low-res and high-res data for training is an interesting idea and appears to be very effective.
- The training cost is very low due to the efficient LoRA fine-tuning, requiring tuning only on the mm-attention modules.
- Experimental results clearly demonstrate the method's efficiency advantages, showing significantly lower memory usage and higher generation speeds compared to baseline methods.

### Weaknesses
- The quantitative evaluation could be more comprehensive. Currently, the paper only includes FID and IS metrics, which alone aren't sufficient to fully capture the image quality. Including additional Image Quality Assessment (IQA) metrics such as PSNR or CLIP-IQA would provide a more complete evaluation.
- In terms of visual comparisons (e.g., Fig. 3 on page 6), Scale-DiT does not show clear visual superiority, especially when compared against the baseline HiFlow. The generated images from Scale-DiT are good, but not obviously better than those from existing approaches.

### Questions
- The authors claim that they do not use native 4K data for training. Have the authors tested whether incorporating native 4K data during training can lead to better results?

### Soundness
3

### Presentation
2

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
This paper proposes a framework to accelerate diffusion transformers (DiTs) at high resolution scales like 2K and 4K. The key insight here is to use local window attention and low-resolution global attention together, making them handle both local details and global semantics well. For implementation, the paper introduces a Hilbert curve order, making tokens within the same local window gathered in a memory-contiguous format, which benefits efficiency. Experiments are conducted on FLUX. The results indicate that the proposed method achieves better results than existing baselines like SANA, URAE, I-Max, HiFlow, etc.

### Strengths
1. The Hibert order is an interesting and effective method for implementation to accelerate practical efficiency.
2. The observation that allowing tokens within a local window to attend to tokens in the adjacent windows can resolve the boundary problem is useful.
3. The writing is overall coherent and easy to follow.

### Weaknesses
1. One major concern is the originality and novelty of the key insights. Using local attention to accelerate high-resolution generation has been explored in [a,b,c]. There are also models combining local attention and global low-resolution attention in the released models of [a]. The difference between [a] and this paper is to use neighborhood attention or window attention. And the difference between [b,c] and this paper is whether to use global low-resolution attention. These are not substantial difference.
2. The practical gain of the proposed Hibert order attention is not studied in the experiments.


[a] CLEAR: Conv-Like Linearization Revs Pre-Trained Diffusion Transformers Up, Liu et al., NeurIPS 2025.

[b] Fast Video Generation with Sliding Tile Attention, Zhang et al., ICML 2025.

[c] Grouping First, Attending Smartly: Training-Free Acceleration for Diffusion Transformers, Ren et al., arXiv 2025.

### Questions
1. From my experience, it is hard for FLUX itself to generate 4K images directly from noise. Why can merely changing the attention formulation and involving low-resolution attention resolve this issue without any 4K training data? More studies would be beneficial here.
2. It seems that the method achieves sub-optimal results in 2K resolution. Is it possible to use the proposed method together with previous training-free ones like I-Max and HiFlow to obtain stronger results?

### Soundness
3

### Presentation
3

### Contribution
2
