# Towards Sequence Modeling Alignment between Tokenizer and Autoregressive Model

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Autoregressive image generation aims to predict the next token based on previous ones. However, this process is challenged by the bidirectional dependencies inherent in conventional image tokenizations, which creates a fundamental misalignment with the unidirectional nature of autoregressive models. To resolve this, we introduce AliTok, a novel Aligned Tokenizer that alters the dependency structure of the token sequence. AliTok employs a bidirectional encoder constrained by a causal decoder, a design that compels the encoder to produce a token sequence with both semantic richness and forward-dependency. Furthermore, by incorporating prefix tokens and employing a two-stage tokenizer training process to enhance reconstruction performance, AliTok achieves high fidelity and predictability simultaneously. Building upon AliTok, a standard decoder-only autoregressive model with just 177M parameters achieves a gFID of 1.44 and an IS of 319.5 on ImageNet-256. Scaling to 662M, our model reaches a gFID of 1.28, surpassing the SOTA diffusion method with 10x faster sampling. On ImageNet-512, our 318M model also achieves a SOTA gFID of 1.39. Code and weights at https://github.com/ali-vilab/alitok.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles an fundamental issue in visual autoregressive modeling: the mismatch between the bidirectional structure of visual tokenizers and the unidirectional nature of autoregressive models. The authors propose AliTok, a new tokenizer that introduces a causal decoder during training, enforcing the encoder to produce tokens with forward-dependency patterns more aligned with autoregressive modeling. A two-stage training strategy and prefix tokens are used to balance generation-friendliness and reconstruction fidelity.

### Strengths
1. The motivation of the paper is clearly articulated and the proposed approach is effective.
2. The paper is well-organized and clearly written. The framework is described in detail and the authors provide sufficient information about it.
3. Instead of modifying the AR model, the authors redesign the tokenizer training objective to produce causally-aligned tokens. This “tokenizer alignment” perspective is novel and could generalize to other modalities.

### Weaknesses
1. AliTok forces tokens to be easier to predict, but that very constraint reduces representational richness.
2. It seems that there is no guarantee that the learned codebook remains causally coherent after AR fine-tuning.

### Questions
1. Can you visualize or quantify how the learned codebook differs from standard VQGAN/ViT-VQ tokens?
2. How does AliTok affect token entropy compared to standard bidirectional tokenizers? A measurable drop would suggest over-regularization, the model becomes easier to train but less expressive.

### Soundness
3

### Presentation
3

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
This paper tackles the misalignment between conventional bidirectional image tokenizers and unidirectional autoregressive (AR) models in image generation by introducing AliTok, an aligned tokenizer. AliTok uses a bidirectional encoder constrained by a causal decoder to produce semantically rich, forward-dependent token sequences, plus prefix tokens and a two-stage training process to boost reconstruction and generation performance. On ImageNet-256, AR models with AliTok excel: a 177M-parameter model achieves 1.44 gFID and 319.5 IS, while the 662M-parameter AliTok-XL hits 1.28 gFID  with 10× faster sampling.

### Strengths
1. The paper is clearly written and well-structured, making it easy to read and understand.
2. The work provides a compelling analysis of the fundamental conflict between conventional bidirectional tokenizers and unidirectional autoregressive (AR) models, and addresses it through a novel design that combines a bidirectional encoder with a causally constrained decoder to facilitate AR image generation.
3. The proposed framework is thoroughly validated through extensive experiments, ablation studies, and well-motivated design choices, including the use of a causal decoder, prefix tokens with an auxiliary loss, and a two-stage tokenizer training strategy.
4. The proposed AliTok approach achieves state-of-the-art performance on the ImageNet-256 benchmark, surpassing leading diffusion-based methods while offering significantly faster sampling speed.

### Weaknesses
1. The experimental evaluation is limited to ImageNet-256; results on higher-resolution datasets such as ImageNet-512 are missing, which leaves uncertainty about the scalability and robustness of the proposed method to larger image resolutions.
2. The paper does not provide a detailed analysis or explanation for the reported 10× sampling speedup. My understanding is that the method introduces additional buffer tokens and prefix tokens, which could potentially incur higher computational cost compared to the baseline. A clear breakdown of the factors contributing to the acceleration would strengthen the claims.
3. The approach may face challenges when scaling to higher resolutions, as the number of prefix tokens is likely to grow with image size (e.g., for ImageNet-512), which might impact efficiency and memory usage.

### Questions
See weakness.

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
The paper proposes AliTok, an image tokenizer aimed at reducing the misalignment between conventional bidirectional visual tokenization and the unidirectional nature of autoregressive (AR) models.
AliTok trains a bidirectional encoder under a causal decoder constraint, encouraging the latent tokens to exhibit forward-dependency so that an AR decoder can better model them. The authors further introduce prefix tokens for first-row reconstruction and a two-stage training scheme to restore image fidelity.
On ImageNet-256, AliTok enables a standard AR transformer to outperform diffusion models (gFID 1.28 vs 1.35) and sample about 10× faster.

### Strengths
- Clear motivation and writing. The authors communicate the misalignment problem and causal-decoder idea clearly.

- Comprehensive ablations. The study isolates each design component’s effect on AR accuracy and gFID.

- Interpretability. Attention-map visualization nikens learn a causal bias.

- Empirical competence. Experiments are reproducible, metrics are standard, and odel sizes.

### Weaknesses
- **Limited conceptual novelty:**  
  The core idea—adding a causal constraint to the tokenizer—is elegant but incremental, extending existing causal-structure ideas (e.g., masked or random-order AR) rather than offering a fundamentally new paradigm.

- **Order-specific and brittle design:**  
  The method hard-codes raster order and needs ad-hoc fixes (prefix tokens, auxiliary loss) for top-row reconstruction. It is unclear if the alignment holds under different scan orders, resolutions, or tasks.

- **Two-stage compromise undermines the claim:**  
  The encoder is trained causally but later paired with a bidirectional decoder, partially reversing the intended unidirectional alignment.

- **Narrow experiments:**  
  Results are limited to ImageNet-256 class-conditional generation. No tests on higher resolutions, text-conditional, or multimodal setups—so generality is unproven.

- **Marginal quantitative gains and unclear fairness:**  
  The gFID improvement over diffusion (1.35 → 1.28) is small and possibly within evaluation noise; speedups rely on different kernel and caching settings without matched baselines.

- **Tokenizer bottleneck:**  
  The discrete tokenizer itself caps achievable FID (~1.15), limiting scalability unless codebook size or representation capacity increases.

### Questions
While raster-scan decoding is the traditional form of autoregressive image generation, it is widely recognized as suboptimal for visual data due to its rigid ordering and limited global context.  
Recent works such as **MaskGIT**, **MAR** have successfully relaxed this constraint using masked or random-order prediction, enabling bidirectional context while maintaining autoregressive structure.

Given this progress, why does your method **intentionally return to a strict raster-scan decoding scheme** instead of adopting or extending these more flexible paradigms?  
Doesn’t enforcing raster causality reintroduce the very inefficiencies and spatial myopia that modern AR methods have been trying to overcome?  
How can you justify this design choice as a step forward rather than a regression in autoregressive modeling?

### Soundness
2

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
5

### Summary
This paper proposes AliTok, a discret image tokenizer designed to match the unidirectional nature of autoregressive models. AliTok is trained in a two-stage manner, the first stage uses causal decoder to regularize the encoder, and the second stage uses bidirectional decoder to improve reconstruction. AliTok also employs prefix tokens to fix the "first row" issue in autoregressive models. AliTok achieves comparable or better performance on ImageNet dataset with a faster sampling speed.

### Strengths
1. The intuition of tailoring the tokenizer to suit the AR generation is well-established. Image as 2-d grid does not really have a naturally direction. Several existing method uses random order AR. AliTok solves this with 1d tokenizer and a causal decoder.
2. The fix for "first row" using prefix tokens is smart. As stated, image does not really have direction, this makes the first row hard to predict. Using prefix row to encode global semantic information provides guidance for this with minimal cost (256 + 17 prefix).
3. The experiments results demonstrate the superior generation performance with extensive ablations and visualization.

### Weaknesses
1. No system-level comparison of reconstruction performance. Despite the final goal for tokenizer is to enable better generation, a detailed comparison and analysis of reconstruction performance cannot be neglected. This is a huge missing in a paper focusing on tokenizer.
2. Several experimental details are missing, especially for reconstruction. For example, what is the batch size / epochs used in two-stage training? What are the size of the tokenizer compared to other model? Without those details, it is hard for apples-to-apples comparison.
3. Sampling speed is not really a vital contribution of AliTok. It naturally fits in autoregressive models which originates from LlamaGen.

### Questions
1. A system-level comparison of reconstruction performance is a must for this paper. Especially with codebook size / codebook dim / training epochs / number of parameters.
2. In Tab. 2, the paper uses original codebases for all methods. Does this mean other methods do not use KV-Cache?
3. In Tab. 2, the performance of GigaTok is attributed to its codebook size. How does AliTok perform with different vocabulary size?

### Soundness
3

### Presentation
3

### Contribution
3
