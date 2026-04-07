## Summary
This paper introduces ARSS, the first decoder-only autoregressive transformer for novel view synthesis from a single image with explicit camera trajectory control. It employs a video tokenizer for temporal consistency, a camera autoencoder for 3D positional guidance, and a spatial permutation strategy to adapt causal modeling to visual data. The method achieves competitive performance against diffusion-based baselines, particularly in maintaining quality over long camera trajectories.

## Strengths
- **Novel paradigm**: ARSS is the first to rigorously adapt a GPT-style, causal autoregressive model to camera-controlled novel view synthesis, opening a new direction for sequential 3D-aware generation.
- **Comprehensive evaluation**: Experiments on RealEstate10K, ACID, and zero-shot DL3DV benchmarks are thorough, with an insightful error accumulation analysis (Fig. 6) demonstrating robust long-horizon performance.
- **Technically sound design**: The integration of a video tokenizer (VidTok) to preserve temporal coherence, a camera autoencoder with explicit geometric constraints (Eq. 5), and a spatial permutation strategy effectively address key challenges in autoregressive visual generation for 3D tasks.

## Weaknesses
- **Performance trade-offs**: While ARSS excels in pixel-level and perceptual metrics (PSNR, LPIPS), it shows slightly lower geometric consistency (SSIM, FID) compared to the best diffusion-based baseline (SEVA) on some datasets (Table 1). The comparison is partially confounded by SEVA's training on larger-scale data, which the paper acknowledges but does not fully equalize.
- **Incomplete ablation study**: The paper lacks an ablation on the camera autoencoder's contribution (e.g., removing camera tokens or using a simpler pose encoding). This is necessary to substantiate the claim that learned camera tokens provide essential 3D guidance.
- **Limited evidence of 3D awareness**: The method is claimed to have "3D spatial awareness," but no direct analysis (e.g., depth accuracy, geometric consistency measures beyond image metrics) is provided to verify how well it models underlying geometry.
- **Methodological clarity gaps**: The conditioning mechanism of camera tokens—how they precisely guide visual token prediction in the interleaved sequence (Eq. 6, 8)—and the handling of causal attention masks after spatial permutation could be more explicitly detailed for full reproducibility.

## Nice-to-Haves
- Comparison to autoregressive video-generation baselines (e.g., adapted from Pang et al. 2025) to isolate the benefits of the proposed architecture over generic AR video models.
- Testing on longer trajectories beyond the trained sequence length (17 frames) to further validate scalability claims for "large environments."
- Analysis of failure modes (e.g., under large view changes or textureless regions) and visualization of full sequences as videos to better assess temporal consistency.
- Discussion of computational efficiency (training/inference cost) relative to diffusion baselines.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Formatting nitpicks**: Concerns about Equation 7 formatting are parser artifacts, not paper errors.
- **Scope creep**: Demanding comparison to autoregressive video-generation baselines is outside the paper's focused contribution to novel view synthesis with camera control.
- **Overly specific demands**: Requests for attention maps or synthetic 3D dataset evaluation are insightful but not standard requirements for this paper's community; they are moved to nice-to-haves.

## Novel Insights
None beyond the paper's own contributions. The paper's core insight is demonstrating that a decoder-only autoregressive model, when augmented with video tokenization, camera conditioning, and spatial permutation, can effectively perform 3D-aware novel view synthesis with causal trajectory generation—a novel and promising direction.

## Suggestions
- Conduct an ablation study to evaluate the necessity of the camera autoencoder, e.g., by comparing against a baseline that uses raw Plücker coordinates or simple embeddings for camera conditioning.
- Provide a more detailed, step-by-step explanation of how camera tokens condition visual token prediction during training and inference, including the attention masking scheme after permutation.
- Incorporate a direct measure of 3D consistency (e.g., depth estimation accuracy or novel-view metrics on synthetic data) to strengthen claims about geometric awareness.