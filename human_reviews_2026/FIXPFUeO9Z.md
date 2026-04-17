# MANZANO: A Simple and Scalable Unified Multimodal Model with a Hybrid Vision Tokenizer

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Unified multimodal Large Language Models (LLMs) that can both understand and generate visual content hold immense potential. However, existing open-source models often suffer from a performance trade-off between these capabilities. We present Manzano, a simple and scalable unified framework that substantially reduces this tension by coupling a hybrid image tokenizer with a well-curated training recipe. A single shared vision encoder feeds two lightweight adapters that produce continuous embeddings for image-to-text understanding and discrete tokens for text-to-image generation within a common semantic space. A unified autoregressive LLM predicts high-level semantics in the form of text and image tokens, with an auxiliary diffusion decoder subsequently translating the image tokens into pixels. The architecture, together with a unified training recipe over understanding and generation data, enables scalable joint learning of both capabilities. Manzano achieves state-of-the-art results among unified models, and is competitive with specialist models, particularly on text-rich evaluation. Our studies show minimal task conflicts and consistent gains from scaling model size, validating our design choice of a hybrid tokenizer.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents Manzano, a method for unified understanding and generation of images, via fusion of respectively continuous and discrete tokens. This leads to improved or competitive performance on various tasks.

### Strengths
The fusion of continuous and discrete tokens, as each of them is better suited for generation and understanding, is interesting. 

The results are good, structurally favorable or similar to other methods.

### Weaknesses
A discussion of why discrete tokens are better than the proposed method for generation - DPG is lacking. 

The results are mostly about discussing that the quantitative performance is good. I would be interested in a discussion about the learned representations and a discussion on where the unified framework benefits from the unification and where it hurts, including qualitative examples and/or empirical findings that go deeper than the numbers. 

It is not clear to me that the generated images are better than other methods.

### Questions
Can you discuss how the learned representations, where the unified framework benefits from the unification and where it hurts, including qualitative examples and/or empirical findings that go deeper than the numbers?

Bagel is very performant too. Are there particular advantages of the proposed method compared to Bagel?

How are the generated images better than those of other methods, what should the reader take from the illustrations?

### Soundness
3

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
This paper proposes a hybrid vision tokenizer and a unified MLLM for both image understanding and generation. The vision tokenizer produce both continuous and discrete latent for different tasks: continuous for multimodal understanding and discrete for autoregressive image generation. Experiments on benchmarks, model scaling and tokenizer comparisons validate its effectiveness.

### Strengths
- Clear motivation. One tokenzier to handle both understanding and generation is an improtant step to unified models. Compared to many works with two seperate vision tokenizers, this paper proposes a simple and scalable vision tokenizer, leading to higher degree of unification.
- Detailed illustration of model architecture, training recipe and evaluation.
- Sufficient experiments on model scaling and model comparisons

### Weaknesses
- Lack of comparisons with related works like ILLUME [1], which also leverages continuous features for image understanding and discrete feature for image generation. Although ILLUME and MANZANO implement the vision tokenizer in a different way, they share similar motivation and architecture in tokenizer and MLLM. This makes the paper with limited techinical novelty.
- Hybrid tokenizer training. Why to use LLM decoder for tokenizer training instead of reconstructing a pretrained vision tokenizer (e.g. CLIP/SigLIP)? I believe it do good to the image understanding task but may not optimal for the image generation task.
- Although the two types of visual representation are similar, but the embedding for image generation cannot be reused for image generation like discrete tokenizers. It makes the model hard to handle interleaved image-text tasks.


[1] ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance. arXiv 2412.06673

### Questions
- Why not to use open-sourced LLMs?
- Tab.1. Why Dual Encoder performs much worse on generation tasks?
- Sec A.2.1. Why the CLIP model is trainable during vision tokenizer training? In ILLUME, the CLIP model is frozen and benefits from faster convergence.
- How about the performance of image editing on benchmarks? Which representation is used for encoding the input image, discrete or continuous?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Manzano, a unified and scalable multimodal large language model designed to jointly handle visual understanding and generation. The framework employs a hybrid image tokenizer and a shared vision encoder with dual lightweight adapters that produce both continuous and discrete visual representations within a common semantic space. A single autoregressive LLM models text and image tokens jointly, while an optional diffusion decoder reconstructs pixels from image tokens. Through a unified training recipe, Manzano effectively balances understanding and generation without significant task interference, achieving state-of-the-art performance among unified models and strong competitiveness with task-specific systems.

### Strengths
1. The paper presents a very clear and well-motivated narrative, with the logical progression from problem statement to solution design articulated in a highly readable manner.

2. The experimental evaluation is thorough, convincingly demonstrating the effectiveness of the hybrid tokenizer and the unified autoregressive architecture across both visual understanding and generation.

3. The unified framework tackles a challenging trade-off (between understanding and generation) and delivers empirical evidence that minimal task conflict occurs—this strengthens the claim of scalability and generality.

### Weaknesses
1. The scaling analysis of LLM size is incomplete — the paper jumps directly from 3B to 30B without intermediate settings such as 7B or 14B, leaving a significant gap in understanding how performance scales with model capacity.

2. While the experiments and insights are clearly presented, the overall methodological novelty is rather limited, as the work mainly integrates existing design elements in a well-engineered manner.

3. The claim “We first pre-train the hybrid tokenizer with a small LLM decoder to pre-align the image features with the LLM feature space” would benefit from additional references to similar prior works, such as X-omni[1] and ETT[2], to better contextualize its contribution.

[1] Geng Z, Wang Y, Ma Y, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again[J]. arXiv preprint arXiv:2507.22058, 2025.
[2] Wang W, Zhang F, Cui Y, et al. End-to-end vision tokenizer tuning[J]. arXiv preprint arXiv:2505.10562, 2025.

4. There is a minor typo on the first page — “Manzanoemploys” — which should be corrected for polish.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3
