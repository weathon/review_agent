# Self-Speculative Decoding Accelerates Lossless Inference in Any-Order and Any-Subset Autoregressive Models

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 4, 8

## Abstract
In arbitrary-order language models, it is an open question how to sample tokens
in parallel from the correct joint distribution. With discrete diffusion models, the
more tokens they generate in parallel, the less their predicted distributions adhere
to the originally learned data distribution, as they rely on a conditional independence 
assumption that only works with infinitesimally small timesteps. We find
that a different class of models, any-subset autoregressive models (AS-ARMs),
holds the solution. As implied by the name, AS-ARMs can generate tokens in any
order, and in parallel. Moreover, AS-ARMs support parallelized joint probability
density estimation, allowing them to correct their own parallel-generated token
distributions, via our Any-Subset Speculative Decoding (ASSD) algorithm. ASSD
provably enables generation of tokens from the correct joint distribution, with the
number of neural network calls upper bounded by the number of tokens predicted
– notably, previous speculative decoding algorithms lack our efficiency guarantee. 
We empirically verify that ASSD speeds up language generation, without
sacrificing quality. Furthermore, we provide a mathematically justified scheme for
training AS-ARMs for generation, and show that AS-ARMs achieve state-of-the-art
performance among sub-200M parameter models on infilling benchmark tasks,
and nearly match the performance of models 50X larger on code generation. Our
theoretical and empirical results indicate that the once-forgotten AS-ARMs are a
promising direction of language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ASSD, a speculative decoding method for any-subset autoregressive models (AS-ARMs). ASSD yields about a 10% speedup over standard AS-ARM sampling and, thanks to its sound joint probabilistic modeling, outperforms much larger discrete diffusion models on code generation and infilling benchmarks.

### Strengths
1. The paper is very well written and easy to follow.

2. Unlike speculative decoding with drafting for ARMs, ASSD is guaranteed to *not* increase the number of calls, and does not need a separate drafting model.

3. Generation statistics for sequential decoding and ASSD are similar (Table 1), consistent with the theoretical guarantees (Lemma 1, Theorems 1 and 2).

4. I appreciate that the authors found and applied a fix to the DiffuLlama evaluation code and chose to compare against DiffuLlama with this correction, making the baseline stronger, instead of ignoring it.

### Weaknesses
1. A direct comparison of the speed and generation perplexity of ASSD against standard AR generation (e.g. GPT-2 with and without KV caching) and masked diffusion models (e.g. MDLM) is missing. I expect ASSD to be slower, but reporting these metrics is important for a complete evaluation, and does *not* diminish the value of the contributions, even as the observed ~10% speedup is notably lower than the 2-3x gains commonly reported for speculative decoding with a drafting model (e.g. in [1]).


2. Important details about hyperparameter choices are too often deferred to the appendix (e.g., lines 308-309 and 350-351). Since there is space available, a brief summary in the main text would be justified.

3. The motivation for the chosen training hyperparameters in code generation (appendix E7) is unclear. In particular, why was masking rate warmup used? Were there experiments comparing with and without it? Including results or rationale for this decision would strengthen the work and guide future research. Currenly, the choice appears somewhat ad-hoc.

4. Line 422: The "k" hyperparameter could be more clearly denoted as the number of speculated tokens.


[1] Fast Inference from Transformers via Speculative Decoding, Leviathan et al.

### Questions
1. On KV caching and causal masking: Lines 117-118 state that all prompt tokens attend to each other. If this is the case, KV caching should not be possible, as each new token would require recomputing previous activations due to bidirectional attention. Could the authors clarify how KV caching is handled in their experiments (appendix)? Is the prompt kept fixed during sampling, with new tokens using causal attention? Additionally, from line 213 ("density estimation via causal-like masking"), are the authors actually applying causal masking in the prompt?

2. The results in Table 3 show that performance on infilling multiple sequences in ROCStories is better than infilling a single sequence, which is counter-intuitive. I would expect the single-sequence case to be easier. Do the authors have an explanation for this? Additionally, after fine-tuning, the XLNet model performs worse at single-sentence infilling. Could this be related to the proportion of prompt tokens during the fine-tuning or something similar? This seems important to clarify.

3. Still, the speedup is much smaller than regular speculative decoding, do the authors have a sense of how to maybe push this further than 10% ?

4. I understand that no loss is applied to the prompt tokens during fine-tuning. Is that correct?

5. Have the authors considered training from scratch, compared to fine-tuning from an XLNet, even on some small synthetic datasets (if costs are prohibitive)? In any case, do you see some advantages or disadvantages of pre-training from scratch in terms of capabilities / inductive-biases of the final model?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new algorithm called Any-Subset Speculative Decoding (ASSD) to significantly speed up text generation for a special class of models called Any-Subset Autoregressive Models (AS-ARMs). This method allows for parallel generation of multiple tokens at once without any loss in output quality, effectively solving a major bottleneck in generating text for tasks like infilling (filling in missing parts of a text).

### Strengths
Lossless Generation: It is mathematically proven to produce text from the exact same distribution as slow, one-by-one sequential decoding. There is no quality degradation.
No Slowdown: The paper proves that the number of model calls will never exceed the number of tokens being generated. This is a critical guarantee that standard speculative decoding lacks, as a poor draft model can actually slow things down.
Free Draft Model: Since the AS-ARM acts as its own drafter and verifier, there is no need to train or maintain a separate, smaller draft model, saving on resources.
Enhanced Versatility: The method is designed for "any-subset" tasks, meaning it can handle an exponential number of infilling patterns (O(2^N)) compared to the linear number of prefix-based patterns (O(N)) handled by standard autoregressive models.

### Weaknesses
1. Scope beyond infilling use cases is not defined or compared.
2. Detailed analysis of accuracy of these models is not mentioned.

### Questions
1. How do AO-ARMs compare with regular ARMs and do you think modeling the joint probability distribution in AO-ARMs facilitate the speculation intrinsically while compromising on model quality as compared to regular ARMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes to combine any-order models with speculative decoding. The method approaches any-order modeling by sampling a mask then using an encoder-decoder architecture to encode all previous tokens and produces remaining tokens autoregressively from left-to-right. The generation process is sped-up via self-spec-decoding, which uses parallel sampling (as in masked diffusion models) to draft and autoregressive mode to verify.

Empirical results on infilling in WikiText, coding, and ROCstories demonstrate a reasonable accuracy-speed tradeoff.

### Strengths
The method is reasonable, and exploring speculative decoding for infilling is novel. The writing was also easy to understand.

### Weaknesses
The main weakness is that the speedups are relatively small, and the acceptance length of 2.24 tokens per draft (Section 7.1) seems quite small. It's possible this is an artifact of model size. The evaluations are constrained by computational resources, so I will not let this affect my review too strongly. Additionally, the evaluations are a bit outdated as well. Again, this seems to be a limitation of computational resources.

### Questions
The token comparison in Table 3 does not seem to be fair, as I believe those are MDLM pretraining tokens.

### Soundness
3

### Presentation
3

### Contribution
2
