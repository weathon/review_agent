# Sequential Diffusion Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Diffusion language models (DLMs) have strong theoretical efficiency but are limited by fixed-length decoding and incompatibility with key-value (KV) caches. Block diffusion mitigates these issues, yet still enforces a fixed block size and requires expensive training. We introduce Next Sequence Prediction (NSP), which unifies next-token and next-block prediction, enabling the model to adaptively determine the generation length at each step. When the length is fixed to 1, NSP reduces to standard next-token prediction. Building on NSP, we propose Sequential Diffusion Language Model (SDLM), which can retrofit pre-trained autoregressive language models (ALMs) at minimal cost. Specifically, SDLM performs diffusion inference within fixed-size mask blocks, but dynamically decodes consecutive subsequences based on model confidence, thereby preserving KV-cache compatibility and improving robustness to varying uncertainty and semantics across the sequence. Experiments show that SDLM matches or surpasses strong autoregressive baselines using only 3.5M training samples, while achieving 2.1× higher throughput than Qwen-2.5. Notably, the SDLM-32B model delivers even more pronounced efficiency gains, demonstrating the strong scalability potential of our modeling paradigm. Code and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces sequential diffusion language models that (1) reformulate the training objective from block diffusion that all the inputs in the block are now masked tokens, and (2) enable parallel training through a specially designed attention mask. During inference, the authors employ greedy decoding and self-speculative decoding to generate and sample tokens in parallel. The results show that it can have strong performance compared to its base model and faster inference speed.

### Strengths
1. The paper proposes two novel language models that enable parallel decoding while achieving superior performance compared to previous diffusion language models.
2. The proposed approach is methodologically sound, and the resulting models demonstrate strong empirical performance.

### Weaknesses
1. I was wondering if this model can still be called a diffusion model anymore. The only thing left in the current DLM that is related to the term `diffusion` is that they randomly mask different tokens at different timesteps. However, according to the training objective proposed in this paper in equation 3, it omits all the timestep-related design and adds several mask tokens together. It has no relationship with the theory of diffusion models. Can the author clarify why this model can still be called a diffusion model?
2. Important comparisons and ablations are missing. The most straightforward baseline to compare with is the block diffusion. The training objective and baseline idea are derived from that paper, but no comparison is shown. What benefits can we get from this new training objective? What would be the effect if we removed the timestep-dependent random masking?
3. Missing comparison with multi-token prediction and speculative baselines. This paper can be very similar to the idea of extending an AR model to predict multi-tokens (Medusa series) and then using/not using speculative decoding (the self-speculative decoding part). Do you have any comparisons with those baselines?

### Questions
1. How much extra training overhead would be introduced by applying this special attention mask? Since the token length would be longer than the original sequence.
2. Why would the performance of Math be higher than LLaDA/Dream by a large margin? I guess it's not the benefits from the new training recipe, but maybe some different tricks in the instruction tuning stage

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
3

### Summary
This paper proposes sequential diffusion models. SDLMs perform inference by predicting a block of $D$ tokens ahead in parallel, all at once, from masked tokens. This gives more FLOPs per prediction than eg multi-token prediction. Predictions are then truncated based on sequence confidence (Longest Prefix Decoding). A couple options are explored for confidence functions: product of probabilities, product of entropy-related quantities, and a self-consistency check drawn from Blockwise Parallel Decoding. Training is the same as Block Diffusion, relying on fixed block sizes.

Results demonstrate that SDLMs offer speedups over the original autoregressive model. No comparisons are made to speculative decoding or multi-token prediction methods.

### Strengths
The method leads to speedups over sequential decoding, but so will any speculative decoding method with a decent acceptance rate. The writing was clear.

### Weaknesses
The main weakness is a lack of comparison to spec decoding methods. I assume this weakness has been anticipated by the authors. Since speculative decoding has been optimized by the community, I would not expect SDLMs to be faster on wall-clock time. A reasonable win would be to show an improved accept length per prediction. Faster wall-clock time as well would of course be great. Eagle-3 claims to have speedups of around 3.5-4.4x and accept lengths around 5-6 tokens on Llama 8b. The results in Table 3 are not far off in terms of accept lengths, but are not an apples-to-apples comparison.

One main benefit of diffusion models is any-order inference, e.g. infilling. It's possible that if pure spec-decoding cannot be beat in the sequential setting, infilling may be the main benefit of blending diffusion models + speculative inference.

### Questions
The main discussion point is comparison to speculative decoding methods. In the weaknesses section, I suggested comparing to Eagle 3.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes Sequential Diffusion Language Models (SDLM), a framework that aims to unify autoregressive decoding and diffusion decoding, through a next sequence prediction (NSP) objective. SDLM/NSP adopts block-wise token generation (e.g., generating 4 token at a time). It further uses a confidence-based or consistency-based length function to identify a usable, longest prefix of the generation -- making the generation block-wise but variable length as well. SDLM is continued trained from regular autoregressive LLMs (e.g., Qwen-2.5). The experiments compare SDLM with autoregressive and diffusion LMs, showing inference speedup with reasonable performance as well.

### Strengths
This paper addresses an important problem in block-wise token generation in LLMs, making the usually fixed block size variable-length. The proposed SDLM can be continued to be trained from regular LLMs. The experiments show reasonable speed-ups for inference. The writing is clear and well-organized.

### Weaknesses
(1) Relevance to diffusion models. Though this work frames SDLM as a diffusion model, it doesn't seem to have a "diffusion" procedure (e.g., with a noising procedure, interpolating between a clean and noise distribution, etc.). Instead, each prediction block is always fully masked with a masking probability of 1. This makes the connection to diffusion language models questionable. The prediction length in each block is also much shorter (D=4). 

(2) Baselines in the multi-token prediction domain. This work compares with regular language models (Qwen) and also two diffusion language models (LLaDA, Dream). However, this work should have a stronger connection with the multi-token prediction works. Direct comparisons with Medusa [1] and work like [2] should be most relevant.

(3) The performance drop. The performance drop of SDLM at the same model size of regular LMs is non-trivial. Table 1 shows that when D=4, in average there's a 2.2 to 7.3 points performance drop. Though it offers 1.89 to 2.39x speed-ups, this gap needs more justification. Also, does average tokens per pass directly translate to the same amount of acceleration in reality? (e.g., need to factor in advanced inference frameworks like vLLM, speculative decoding, etc?)

[1] Cai et al., 2024. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.

[2] Gloeckle et al., 2024. Better & Faster Large Language Models via Multi-token Prediction.

### Questions
Does the method work for D > > 8? It could be hard for general tasks I assume, but would it work for tasks that require large chunks of copying?

### Soundness
2

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
The authors propose to cheaply finetune autoregressive LMs for diffusion decoding of contiguous token blocks. They extend standard block diffusion inference for variable-length block sizes, which adapts to confidence. The authors show 5x speedup over prior dLLM LLaDA with improved benchmark performance and 2x speedup over Qwen, scaling their models up to 32B parameters.

### Strengths
- **Technical novelty**: The authors overcome the limitations in existing block diffusion models by proposing adaptive block sizes at inference, while introducing entropy-normalized confidence and self-speculative decoding for dLLMs
- **Strong experimental results**: The authors show 5x speedup over LLaDA and 2x speedup over Qwen with limited SFT examples, scaling up to 32B parameters. Further, the benchmark performance improves substantially over prior dLLMs LLaDA and Dream. The released models will positively impact the research community
- **Thorough benchmarking and ablations**: The authors provide thorough analysis across benchmarks, showing comparable performance to AR SFT with better throughput. The authors also carefully assess the importance of block size, confidence metric, logit shifting, attention mechanism, and speculative decoding and show that their proposed recipe improves downstream performance

### Weaknesses
- **Unclear whether proposed Next Sequence Prediction modeling objective corresponds to a valid NELBO**
- **Ambiguous training algorithm**: It is unclear how the sequence is split into blocks at random positions (line 213), or why noised blocks are randomly inserted in the training input sequence (line 250). Training pseudocode is not provided.
- **Contiguous block decoding prevents long-range token interactions**: The proposed decoding strategy is restricted to generate contiguous blocks, preventing long-range token interactions and non-causal token conditioning, which are advantages of diffusion.
- **Missing ablation on the importance of contiguous block decoding**: It is unclear whether contiguous block decoding is necessary, which is a core contribution over [1]
- **Missing experimental comparison to multi-token prediction**: The proposed decoding strategy generates the longest contiguous block of tokens possible in each decoding step, which is very similar to multi-token prediction (MTP) approaches [2,3] as noted in Appendix B. Given their close similarities, the authors should show experimental comparison to MTP methods to better motivate their approach
- **Overstated contribution**: The authors claim that standard block diffusion training [1] is not scalable (lines 100-101), while the authors propose to overcome this by initializing from an AR checkpoint. However, the proposed training algorithm (Section 3.3) has comparable efficiency to [1]. Further, [1] could easily be directly extended to initialize from an AR checkpoint
- **The proposed parallel training is very similar to the vectorized training in [1]**, which is not cited or compared against in the text (lines 245-258). [1] also proposes parallel training to predict all blocks using FlexAttention kernels
- **Presentation/clarity**: Various parts of the paper feel rushed with numerous grammatical errors/missing definitions
  - line 96, 108: "with only 3.5M training data"
  - line 40: "benefit from the diffuse nature in efficiency"
  - line 194: "shifted-prediction objective" undefined in the section
  - line 200: $\hat{X}$ is undefined
  - line 169: $\alpha_t$ undefined and the forward noise process $q$ defined informally
  - line 214: confusing notation: $Y_T^i$ corresponds to a block of $D$ tokens, but the superscript refers to indexing at position $i$?

[1] Arriola et al. Interpolating between autoregressive and diffusion language models. ICLR 2025

[2] Cai et al. Medusa: Simple llm inference acceleration framework with multiple decoding heads. ICML 2024

[3] Gloeckle et al. Better & Faster Large Language Models via Multi-token Prediction. arXiv 2024

### Questions
- Can the authors provide pseudocode for the training and inference algorithms?
- Does the proposed next sequence prediction objective (Eq 3) correspond to a valid NELBO?
- Can the authors show the importance of contiguous block decoding via ablation?
- Can the authors show the effectiveness of their approach relative to multi-token prediction [2,3]?
- Can the authors detail the differences between the proposed parallel block diffusion training and the vectorized training from [1]?
- In addition, the choice of $\tau \in \[ 0.82, 0.98 \]$  in Table 1 seems heuristic; can the authors clarify how $\tau$ was picked?

[1] Arriola et al. Interpolating between autoregressive and diffusion language models. ICLR 2025

[2] Cai et al. Medusa: Simple llm inference acceleration framework with multiple decoding heads. ICML 2024

[3] Gloeckle et al. Better & Faster Large Language Models via Multi-token Prediction. arXiv 2024

### Soundness
2

### Presentation
1

### Contribution
2
