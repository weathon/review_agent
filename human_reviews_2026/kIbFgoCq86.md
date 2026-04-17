# Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed

- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
The token-by-token decoding nature of autoregressive (AR) language models limits their generation throughput, especially in common memory-constrained scenarios. To address this, diffusion language models (dLMs) have emerged as a promising paradigm to enable parallel, non-autoregressive generation for higher throughput. However, existing dLMs have either failed to deliver faster speeds than AR models or have been restricted to small model scales due to high training costs, resulting in limited capability. To this end, we build on pretrained AR models and develop a training framework to convert them into dLMs that excel in speed. First, we introduce a continuous pretraining scheme with a block-wise attention pattern that remains causal across blocks while enabling bidirectional modeling within each block, which we find to better preserve pretrained models' abilities than the fully bidirectional modeling used in prior work such as Dream. Second, to mitigate the training–test gap in mask token distributions, we propose a position-dependent token masking strategy that assigns higher masking probabilities to later tokens. Leveraging this framework, we conduct extensive studies of dLMs’ attention patterns, training dynamics, and other design choices, providing actionable insights into scalable AR-to-dLM conversion. We also deliver the Efficient-DLM model family, which outperforms state-of-the-art AR models and dLMs with better accuracy–throughput trade-offs, e.g., Efficient-DLM 4B achieves +1.88% higher accuracy with 4.63x throughput compared to Dream 7B, and +7.79% accuracy with 1.82x throughput compared to Qwen3 1.7B.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Efficient-DLM, a Diffusion Language Model (dLMs) designed for high generation throughput and accuracy, achieved by continuously training pretrained Autoregressive (AR) models. The core contributions are a block-wise attention pattern that preserves AR model capabilities and enables KV caching, and a position-dependent token masking strategy to mitigate the training-test distribution gap. Efficient-DLM demonstrates superior accuracy-throughput trade-offs compared to state-of-the-art AR models and existing dLMs (LLaDA, Dream).

### Strengths
- The paper provides comprehensive empirical studies of the training recipes for discrete diffusion models, providing insights into factors that influence performance, efficiency and stability. These findings can serve as a practical reference for future research and development in discrete diffusion-based language modelling.
- The final model, Efficient-DLM, achieves better performance among a wide range of evaluations compared to other DLMs.

### Weaknesses
- My main concern is the actual contribution of this paper. To me, it appears to be a block-diffusion approach [1] that uses pretrained auto-regressive models to initialise the model weights. While the paper emphasises engineering discussion, the overall method seems to be a straightforward combination of previous ideas:
    - The block-diffusion concept is derived from [1].
    - The auto-regressive initialisation comes from [2].
    - The efficiency and acceleration result from leveraging KV-cache with block diffusion, as in [3].
- The paper reads more like an empirical study on the training of block-diffusion models. However, most of the observations are already well-known within the community, which limits the novelty and impact of the work.

[1] Arriola, Marianne, et al. "Block diffusion: Interpolating between autoregressive and diffusion language models." *arXiv preprint arXiv:2503.09573* (2025).

[2] Gong, Shansan, et al. "Scaling diffusion language models via adaptation from autoregressive models." *arXiv preprint arXiv:2410.17891* (2024).

[3] Liu, Zhiyuan, et al. "dllm-cache: Accelerating diffusion large language models with adaptive caching." *arXiv preprint arXiv:2506.06295* (2025).

### Questions
- It is encouraging to see that the proposed Efficient-DLM achieves the best average performance among discrete language models. However, it is unclear which specific components contribute most to this improvement. Adding more ablation studies, such as evaluating the effects of position-dependent masking, …, would help clarify the contributions of each component.
- It is surprising to see in Table 3 that Efficient-DLM achieves significantly better token-per-second (TPS) performance. Why is this the case? While block-diffusion is generally more efficient at test time due to the KV-cache, during training, I would expect it to be less efficient than standard discrete diffusion. This is because block-diffusion requires concatenating duplicated text to enable parallel training, which should typically reduce TPS.
- In Table 3, what is the difference between the two Efficient-DLM rows?

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
3

### Summary
This paper conducts conversion of AR models into diffusion LMs at scale, and offers several practical insights. Using these insights, paper introduces Efficient-DLM 4B, that achieves similar accuracy and higher throughput compared to AR baselines.

### Strengths
- conducts lots of experiments to provide plethora of takeaways for adapting autoregressive models to diffusion
- demonstrates speed ups compared to AR model with minimal accuracy drop
- due to the careful study, choosing the "right" block size can mean choosing varying block sizes at inference, showing accuracy-efficiency trade-offs

### Weaknesses
- the writing can be improved, right now the main paper is very dense and there are a lot of details to keep track of; when starting section 5, perhaps what changes are actually incorporated in the final efficient-DLM could be listed once again. Let me know what changes you plan to do to make the main paper more accessible and clear :)

### Questions
- how would the latency change when the context length increases?
- is there any KV caching within a block as well?

Since i'm not an expert in this field, i'll be looking at feedback of other reviewers too to gain a better understanding.
That being said, i'm definitely willing to improve my score during our discussion :)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a systematic study on converting pretrained autoregressive (AR) language models into efficient diffusion language models (dLMs) that enable parallel generation while preserving accuracy. The authors address two key challenges:

1. Incompatibility between AR initialization and standard bidirectional dLM training: Solved via a *block-wise attention pattern* that maintains causality across blocks while permitting bidirectional modeling within blocks.
2. Training-test gap in token masking distributions: Tackled through a novel *position-dependent token masking strategy* that assigns higher masking probabilities to later tokens.

Their analysis shows that dLMs retain a left-to-right generation tendency despite parallel capabilities and that appropriate attention patterns greatly reduce weight drift from pretrained models. The Efficient-DLM family achieves strong accuracy-throughput trade-offs—e.g., Efficient-DLM 4B attains +1.65% higher accuracy with 4.77× throughput over Dream 7B, and +7.56% accuracy with 1.87× throughput over Qwen3 1.7B. The work offers actionable guidance for AR-to-dLM conversion and meaningfully bridges theoretical parallelism with practical generation speedups, especially for memory-constrained deployments.

### Strengths
- **Good recipe for AR→dLM conversion.** Combines block-wise attention (causal across blocks, bidirectional within), clean context conditioning, and position-dependent masking $w_i(t) = \exp[\beta(1-t)i]$ with a nice half-life parameterization ($\lambda = \ln 2/(\beta L')$). The design choices make sense and address real training-inference gaps.

- **Very thorough experiments.** Covers attention ablations, block sizes, long training runs (200B tokens), LoRA-based conversion, plus detailed diagnostics (weight drift, position-wise loss). The accuracy-throughput analysis across coding/math/knowledge tasks is pretty comprehensive.

- **Paper is clear.** Eq. (1) is well-stated, figures show the attention patterns nicely, and the half-life semantics make the masking strategy easy to understand.

### Weaknesses
**Eq. (1) doesn't match the implementation description:**

Eq. (1) sums over all blocks ($b = 1..B$), but §2.1 and Fig. 1(d) say each term conditions on a clean prefix ($\mathbf{x}^{<b}$) plus a corrupted current block ($\tilde{\mathbf{x}}_t^b$). If every block is corrupted simultaneously, you can't keep all the prefixes clean unless you replicate the sequence $B$ times.

If training corrupts exactly one block per sample (keeping prefix clean), the objective should be:

$$
\mathbb{E}_{b \sim \mathrm{Unif}[B]}\mathbb{E}_{t,\tilde{\mathbf{x}}_t^b} \Big[-\frac{1}{\alpha_t} \log p_{\theta}(\mathbf{x}^b \mid \tilde{\mathbf{x}}_t^b,\mathbf{x}^{<b})\Big]
$$

not a sum over all blocks. As written, Eq. (1) suggests either (a) multiple forward passes per sequence (one per block), or (b) an impossible conditioning setup.

This also conflicts with Row (e) in Tab. 1 ("doubled token budget to account for the increased sequence length caused by concatenating noisy and clean tokens"). If the sum is truly over all $b$, compute is already multiplied; if selecting one block per sample, Eq. (1) shouldn't have a sum.

---

**Inconsistent mask count:** The paper defines per-block positional weights $w_i(t)$ for $i \in [1..L']$, but then says "The set of mask tokens is drawn … with $k = \lfloor t L \rfloor$." If sampling is per block, shouldn't $k$ be $\lfloor t L' \rfloor$ instead of $\lfloor t L \rfloor$? Using total sequence length ($L$) seems inconsistent with the per-block approach and would ask for more masks than fit in a block.

---

**Best dLM still lags behind the AR source.** In Tab. 1, Row (a) (AR Qwen2.5-1.5B) "Avg" is 41.79, but the best dLM setting (Row (g): block-wise + clean context + no shift) only hits 38.41 after 50B tokens. That's a −3.38 point drop (≈−8.1% relative). The text says block-wise attention "better preserves pretrained AR models' abilities"—which is true compared to *other* dLMs—but at the same model size the proposed approach doesn't recover the AR baseline accuracy.

---

**Missing definition:** $\alpha_t$ is used but never defined.

---

**Novelty concerns:** The core recipe (block‑wise attention with clean prefix + confidence‑based parallel decoding + KV caching) is quite close to Block Diffusion (which also does AR across blocks, diffusion within blocks, with native KV caching). The "continuous pretraining from AR" idea was originally in DiffuLLaMa, which isn't cited here.

### Questions
- The embedding evaluation doesn't include LLM2Vec (cited tho), which also does continuous training from pretrained AR models and supports bi‑mask + contrastive training, why?

Please also see above section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies how to turn autoregressive (AR) models to diffusion language models (dLMs) that are both fast and accurate. The authors identify key ingredients in turning AR models into diffusion language models: block attention to preserve local bidirectional modeling, appropriate block sizes, sampling preferences leaning toward left-to-right, conditioning masking on clean past data, and avoiding token shift. Together, these changes yields efficient and performative dLMs, with upto 2x throughput while also improving accuracy.

### Strengths
- The paper tackles an interesting question of distilling AR models into dLMs.
- Many choices in this conversion are elicited and evaluated.
- The resulting method of continued pertaining with block-wise bidirectional attention with clean past context provides significant improvements in efficiency because it allows using many existing methods to speed up inference like the KV-cache.
- Adaptively efficient decoding by changing the confidence threshold in decoding tokens shows near Pareto-optimal trade-offs.

### Weaknesses
See questions.

### Questions
- what is the role of confidence threshold in decoding for different choices of the half-life? It seems that forcing models to decode later tokens could worsen the trade-off.
- Is the set of "good choices' you discovered unique to pre-trained AR models? It seems true because you made choices to not let the weight go too far. What would happen if I trained with your procedure? That seems like an important comparison, showing whether language priors inside the AR model help learn diffusion models or whether it's really just the proposed set of choices.
- Is there any repeated training? In that Qwen models train on the same data you do?

### Soundness
3

### Presentation
4

### Contribution
4
