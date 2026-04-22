# Multimodal Latent Language Modeling with Next-Token Diffusion

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Multimodal generative models require a unified approach to handle both discrete data (e.g., text and code) and continuous data (e.g., image, audio, video). In this work, we propose Latent Language Modeling (LatentLM), which seamlessly integrates continuous and discrete data using causal Transformers. Specifically, we employ a variational autoencoder (VAE) to represent continuous data as latent vectors and introduce next-token diffusion for autoregressive generation of these vectors. Additionally, we develop 
-VAE to address the challenges of variance collapse, which is crucial for autoregressive modeling. Extensive experiments demonstrate the effectiveness of LatentLM across various modalities. In image generation, LatentLM surpasses Diffusion Transformers in both performance and scalability. When integrated into multimodal large language models, LatentLM provides a general-purpose interface that unifies multimodal generation and understanding. Experimental results show that LatentLM achieves favorable performance compared to Transfusion and vector quantized models in the setting of scaling up training tokens. In text-to-speech synthesis, LatentLM outperforms the state-of-the-art VALL-E 2 model in speaker similarity and robustness, while requiring 10
 fewer decoding steps. The results establish LatentLM as a highly effective and scalable approach to advance large multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on autoregressive generation of continuous tokens, to enable multimodal models that work with continuous modalities without quantizing them.
It is influenced by [39] that suggested a diffusion head to predict continuous features.
Their contributions are (as I see them):
- analysis of VAE variance and its influence on exposure bias in autoregressive models with continuous features (mismatch between teacher-forcing and generation)
- suggestion of an improved VAE version, that does not predict the variance, but rather uses a randomly sampled one.
- scaling the approach for multimodal image-text LLMs, and TTS

### Strengths
Extensive experiments across different tasks and modalities. 
Interesting and important task of modeling continuous and discrete featutes in the same model.

### Weaknesses
Section 1:
You stated that your continuous latent vectors are lossless. 

Section 4.3: sigma VAE:
z is a multiplication of two Gaussians, it does NOT follow the Gaussian distribution.
sigma is not a variance - it can be zero or negative. (line 229)


Section 5.3.2 - compression rate:
Compression rate measures the number of bits required to store the compressed sequence.
In your case, which considers continuous features, each float requires 16 bits for saving (assuming bfloat16). 
If you have a 32-dimensional vector, it requires 16*32=512 bits.
In contrast a discrete single 1024 codebook requires just 10 bits. 
You should rename this column to be “sequence/token reduction rate” - using compression here is wrong and misleading.
I also want to mention that your contribution of managing to reduce the token rate is important, and increasing the latent dimension is a good idea. 

Final comment on compression/frame rates: works that use K codebooks, usually use multi-codebook prediction heads (MusicGen/Soundstorm). 
So I would treat the 2-codebook DAC like the wav-tokenizer when measuring “sequence/token reduction rate”.

### Questions
- How does the <EOD> work? What does the discrete head predict when there's a continuous token (if it ignores the output, the discrete head might predict EOD in there by mistake, and truncate the generation)

- Figure 5: can you elaborate on the procedure done in detail? What changes in the inference when using different variance values?

- How do you measure the condition sigma_gen < sigma_train?

- Do you input mu or z to LatentLM? 

- I understand how sigma-VAE can make a more robust decoder, but does it reduce the accumulation of noise during the autoregressive generation?

- Can you elaborate how did you isolate the source for the sigma-VAE sucess? 

- I did not find what values of C_sigma did you use

### Soundness
2

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
3

### Summary
LatentLM is a causal-Transformer framework that treats continuous modalities by encoding them into latent vectors with a σ-VAE and generating them via a next-token diffusion head, while handling discrete tokens with standard next-token prediction. The σ-VAE fixes latent variance per example to mitigate exposure-bias drift and stabilize long-sequence autoregression. Trained jointly across text, images, and audio, the single backbone provides a unified interface for multimodal understanding and generation with competitive results and favorable scaling across tasks, including T2I, vision-language modeling, and TTS. The design reuses LLM infrastructure for efficient causal decoding and leverages fast diffusion solvers for practical inference.

### Strengths
LatentLM is a unified multimodal generator that keeps the causal, next-token structure of language modeling while handling continuous signals as latent sequences. Its next-token diffusion head enables autoregressive generation of these latents, and the σ-VAE stabilizes long-horizon decoding by fixing latent variance per example. In practice, the model delivers competitive image and speech generation under a compact representation, and its single backbone supports multitask training and inference across text, images, and audio, pointing toward a genuinely multimodal language model.

### Weaknesses
- Originality
    - While the approach unifies modalities, many components are adaptations of existing ideas: autoregressive diffusion, and low‑rate latent coding from neural codecs. The paper claims image generation is “competitive” with diffusion, but Table 1 shows diffusion models still achieve lower FID (e.g., DiT, LDM) than LatentLM, suggesting that the method is not strictly superior and might be constrained by the compact latent representation.
    - Related work lacks discussion of speech‑generation approaches (e.g. VALL‑E 2, VoiceBox), and recent multimodal LLMs such as Qwen‑Omni, Gemini. A broader survey would clarify LatentLM’s positioning.
- Quality
    - The authors do not report how chunk size (in next‑token diffusion) affects latency versus quality. A systematic study of chunk length vs. FID/perplexity and decoding speed would strengthen the argument for compact latents.
    - Eq. 2 uses ε (noise) prediction; it would be instructive to compare x‑ or v‑prediction alternatives, as done in diffusion literature.
    - In σ‑VAE, noise variance could be scheduled based on sequence length or time (e.g., increasing σ as the sequence grows). It is unclear whether the authors tried time‑dependent scheduling.
- Clarity
    - The paper should explain whether next‑token diffusion is less effective than full diffusion because of the compact representation.
    - A more explicit derivation of next‑token diffusion and its relation to existing diffusion transformers would aid understanding.
- Significance
    - The efficiency gains of σ‑VAE and next‑token diffusion are compelling, yet the comparison to diffusion transformers is restricted to FID/IS; more diverse tasks (e.g. high-resolution image synthesis, long-form speech) would demonstrate generality.

### Questions
- The model’s ability to handle long-range dependencies (e.g., long sequences or large images) is not fully explored.
- Have the authors experimented with different latent chunk sizes in next‑token diffusion to study the quality vs. latency trade-off? Results on FID/perplexity vs. chunk length would be valuable.
- Did the authors try predicting x or v and compare performance (in Eq. 2)?
- In σ‑VAE, did the authors explore noise schedules that scale with the sequence’s time domain (e.g., increasing σ over time) and if so, how does that affect stability and quality?
- For clarity, could the authors bold the best scores within each evaluation in Table 1?

### Soundness
3

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
This paper proposes Latent Language Modeling (LatentLM) to integrate continuous and discrete data in causal transformers, where distributed training and KV cache are applicable. The authors develop a new VAE model to better capture data variance and address the collapsed variance issue in AR modeling.

### Strengths
The proposed $\sigma$-VAE is applicable to different types of continuous data such as image and audio, which helps unify the understanding and generation in AR framework.

### Weaknesses
Limited results are presented in Table 3 and Table 4, as only three models are involved in the comparison, which makes the results less convincing.

### Questions
1. For Table 3, it would be better if more multimodal LLMs could be included in the comparison (such as Show-o, SEED-X, LWM, etc). Even models that can only perform a specific task can be included (say, models for Text-to-Image generation only, or models for visual understanding only). BEiT-3 can also be included since the encoder of LatentLM is initialized from this model, as mentioned in the paper. One additional suggestion is to include more image understanding benchmarking datasets, such as MME and MMMU. 

2. Still, for Table 3, since there’s also no explicit reference cited in the paper, I’m confused by what “VQ-MLLM” is referring to. Does it mean all "models using vector quantized image tokenizers" (line 372)? 

3. Can the authors also provide an ablation study regarding the inference sampling steps for image generation (similar to Fig. 15 in speech synthesis)?

4. The qualitative results for image generation are shown in the paper. Can the authors also provide some qualitative results for text (i.e. image-to-text) generation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a method for unifying text and image generation into a single model by combining auto-regressive models with diffusion based models. They do so by having a transformer that processes both modalities and contains a language model head as well as a diffusion head, the diffusion head serves the denoising job, while the language head outputs the logits over the next token.

### Strengths
- The paper presents a method that consolidates diffusion with AR which gives the best of both worlds
- The paper presents large scale experiments to demonstrate their methods robustness
- The paper notices a problem in existing VAEs so that  diffusion + AR models can squeeze the most performance and present a simple solution

### Weaknesses
- The presentation is a little hard to follow and it is not completely clear how the diffusion part is being done. Do we append the whole sequence of denoising steps as inputs to the transformer or just the current step? How is the time embedding taking into consideration?
- The authors mention a discrepancy in the variance of training vs inference time for the VAEs. However they don't present any evidence that this is the case
- It is unclear how the new VAE training technique improves upon existing results and how it should be used during testing time. Specifically, $\sigma \sim \mathcal{N}(0, C_\sigma)$ is a random variable across the training. Is this fixed at the beginning or does it get varied on every sampling step? Furthermore, $\sigma$ should also be a positive value if it is meant to represent a standard deviation, but this is not enforced by sampling from a normal; is something else being done in addition to this? Could the authors also mention how $C_\sigma$ was selected?
- In figure 5, could the authors compare their result to using a VAE trained in a standard way or using some standard VAE?
- Minor: The related works is missing several important works that aim to combine language and image generation capabilities for example OmniFlow [1], Diffuse-Everything [2], Janus-Flow [3]. Other methods like Show-O, Transfusion and JetFormer are mentioned, but there are no comparisons to their work.



[1] Li, Shufan, et al. "Omniflow: Any-to-any generation with multi-modal rectified flows." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Rojas, Kevin, et al. "Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces." arXiv preprint arXiv:2506.07903 (2025).

[3] Ma, Yiyang, et al. "Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
- Could the authors include the details of how many GPUs it was required to train their model
- Could the authors talk about how their proposed $\sigma-$VAE is different from  [1]? It seems like both approaches aim to improve the ability of the VAE to provide useful features, both are approached in different ways and understanding their different use cases could be important

[1] Skorokhodov, Ivan, et al. "Improving the diffusability of autoencoders." arXiv preprint arXiv:2502.14831 (2025).

### Soundness
3

### Presentation
1

### Contribution
2
