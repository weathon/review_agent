# Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
Discrete diffusion models offer a promising alternative to autoregressive generation through parallel decoding, but they suffer from a sampling wall: once categorical sampling occurs, rich distributional information collapses into one-hot vectors and cannot be propagated across steps, forcing subsequent steps to operate with limited information. To mitigate this problem, we introduce Loopholing, a novel and simple mechanism that preserves this information via a deterministic latent pathway, leading to Loopholing Discrete Diffusion Models (LDDMs). Trained efficiently with a self-conditioning strategy that avoids unrolling the full denoising trajectory, LDDMs achieve substantial gains—reducing generative perplexity by up to 61\% over prior baselines, thereby closing (and in some cases surpassing) the gap with autoregressive models, and producing more coherent text. Applied to reasoning tasks, LDDMs also improve performance on arithmetic benchmarks such as Countdown and Game of 24. These results also indicate that loopholing mitigates idle steps and oscillations, providing a general and effective path toward high-quality non-autoregressive text generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies a key limitation in discrete diffusion models, which the authors term the sampling wall. This refers to the information collapse that occurs when the rich, continuous categorical distributions predicted by the model are sampled, reducing them to one-hot vectors for the next denoising step. The authors hypothesize that this loss of distributional information is a root cause of known inefficiencies in discrete diffusion, such as "idle steps" and "excessive oscillation" .To address this, the paper introduces Loopholing, a mechanism that creates a deterministic latent pathway to complement the standard stochastic sampling path. The resulting models, Loopholing Discrete Diffusion Models (LDDMs), are evaluated on language modeling and arithmetic reasoning. Experiments show that LDDMs significantly improve upon baseline models, achieving up to a 61% reduction in PPL. It also improves performance on reasoning tasks like Countdown and Game of 24.

### Strengths
a. The proposed Loopholing mechanism is a simple and logical solution to the stated problem. Propagating the continuous latent state $h_s$ is a direct way to preserve information, and the architectural modification is relatively minor but highly effective.

b. The authors successfully apply Loopholing to two different families of diffusion models (Masked Diffusion and Uniform Diffusion) and two different domains (open-ended language modeling and structured arithmetic reasoning ). This suggests that Loopholing is a general and widely applicable technique.

c. The analysis of Temporal KL (TKL) and Token-Prediction Entropy (TPE) in Figure 5 compellingly supports the hypothesis that Loopholing mitigates oscillations and idle steps.

d. Loopholling mechanism surpass of its discrete diffusion backbone even under the same computing budget.

### Weaknesses
a. While the results are excellent at this scale, it remains an open question whether these substantial relative gains will persist with much larger models.

b. The paper notes that an attempt to apply Loopholing only during fine-tuning was unsuccessful. This limits the capability to improve existing diffusion models. Is it possible to continue pre-train with an existing diffusion model with the loopholing mechanism?

c. Loopholing yields strong improvements for MDLM but marginal gains for UDLM (Table 2), except on PTB. The authors attribute this to domain shift sensitivity in UDLM perplexity, but this undermines the generality of the proposed mechanism. Are there any other metrics can be used to evaluate the performance?

### Questions
The paper modified the TopK decoding of MGDM to only apply to masked tokens. However, Table 4 in the appendix shows that the original MGDM decoding actually performs better for both the baseline and LDDM-G. Why does this approach that ‘conflicts with the training objective’ achieve higher performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper shows strong improvements in generation and reasoning accuracy of discrete diffusion models by allowing each diffusion step to condition on the pre-logit hidden states of the previous step.  This avoids the problem of losing information in the discrete sampling step, despite the fact that these hidden states are not trained for this purpose.

This problem is fundamentally the same as the problem with backprop across a discrete sampling step, and the intuition for solving it is essentially the same as the intuition behind Gumbel softmax (i.e. mixing the complete predicted distribution with the sampled one).  But the authors identify the forward direction of this problem as a separate issue independent of the backward direction, and demonstrate its importance for diffusion models.  They also propose a practical architecture which implements this intuition in an efficient way, by conditioning on the pre-logit hidden states (mixing these pre-sampling embeddings with the embeddings from the sampled tokens) and not backproping error across this conditioning.

### Strengths
The model and its motivation are well explained.  The empirical results are strong, thorough and convincing, and include the obvious ablations.

Using diffusion for non-autoregressive generation of text is an important open problem, and this paper demonstrates a method for closing the gap in accuracy with auto-regressive models.  It also shows how iterative methods such as this can improve reasoning ability.

### Weaknesses
Identifying the problem of passing information forward across discrete sampling steps is not as novel as they claim, since it just the forward perspective on the well-known problems with backprop across a discrete sampling step.  But the method for addressing this intuition is novel, and it is novel to show that it applies even when the model has no backprop across this step.  It would have been interesting to see how this compares to a model which does backprop across this step, but this is mentioned as future work.

Equation (2) is confusing because it uses the same variables to sometimes refer to the entire sequence and sometimes refer to individual tokens within the sequence.  I think it would be clearer to add explicit token indexes where appropriate.

I wasn't convinced by the discussion of Idle Steps, since the graphs are averages and thus do not show whether individual paths have idle steps.

For the claim in lines 360-361, please mention which graph backs up this claim.

### Questions
Please explain if I have misunderstood the argument about Idle Steps and Excessive Oscillation.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Loopholing Discrete Diffusion Models (LDDMs), which modify discrete diffusion LMs by adding a deterministic latent pathway (h_t) that is passed between denoising steps, in addition to the standard sampled token outputs. The idea is to “bypass the sampling wall,” defined as loss of distributional information once you collapse logits into a one-hot token during sampling. The paper also proposes a two-pass self-conditioning scheme to train this recurrent latent path without having to unroll time.
The authors claim:

lower perplexity than prior discrete diffusion baselines such as MDLM and UDLM on LM1B and OpenWebText,

lower “generative perplexity” (measured by GPT-2) and higher GPT-4.1 “consistency”/“naturalness” scores for unconditional generations,

and improved success rates on small arithmetic reasoning tasks like Countdown and Game of 24 when applied to MGDM.

### Strengths
* The paper articulates a clear failure mode of discrete diffusion LMs: once you sample a discrete token, you throw away the richer distributional state and force the next step to start from a one-hot. They call this the “sampling wall.” 

* The proposed “loopholing” mechanism - passing a continuous latent h_t forward alongside the sampled token sequence — is architecturally straightforward and appealing for implementation. 

* LDDM-M beats MDLM on perplexity on LM1B and OWT (e.g. ≤25.95 vs ≤27.60 on LM1B; ≤21.90 vs ≤23.05 on OWT). 
* On unconditional generation, “Gen PPL” improves a lot (e.g. 49.13 vs 108.94) and GPT-4.1 gives higher fluency/consistency scores. 
* On arithmetic reasoning puzzles, LDDM-G improves success rates vs MGDM (e.g. Countdown4: 86.5→94.4% at 85M params).

### Weaknesses
* Evaluation is narrow and relies only on perplexity metric.
The paper repeatedly claims “closing the autoregressive gap,” but this is based entirely on approximate perplexity (NELBO upper bounds) rather than true log-likelihood or downstream task performance.
There is no evaluation on actual generative tasks such as summarization, dialogue, open-ended story writing, or reasoning. Consequently, we cannot conclude that the method improves generation quality. It is important to evaluate the models on actual dowsntream performances, some of the current widely used benchmakrs for evaluation of the large scale language models are summarization tasks, MMLU, MMLU-Pro, Math-500, GSM8k, and reasoning benchmarks, and code benchmarks like Humaneval, MBPP, ... 
 
* Limited novelty beyond standard self-conditioning.
The proposed “loopholing” mechanism - passing a continuous latent state between denoising steps trained via a two-pass stop-gradient trick — is almost identical to self-conditioning already used in diffusion and consistency models (e.g. Jabri et al., 2022; Chen et al., 2022).
The only real difference is that LDDM explicitly names and propagates a latent vector rather than the previous clean prediction. This is a minor architectural variant, not a new principle.

* The “sampling wall” is largely a renamed version of known problems.
What the paper calls the “sampling wall” (loss of information when discretizing predictions) is conceptually the same as the “idle steps” and “oscillation” problems already identified in earlier discrete diffusion work (e.g., MDLM, SSD-LM).
The paper renames these issues and overstates their novelty, without providing direct causal evidence that loopholing specifically resolves them.

* Causal explanation lacks experimental proof.
The claim that the propagated latent  “bypasses” the sampling wall is asserted, not demonstrated.
There is no ablation isolating: the effect of the latent path from the effect of self-conditioning, or the effect of feeding forward x_{theta, t} versus h_t
	​
* The observed differences could easily stem from generic stabilization due to the two-pass stop-gradient process.

* Weak baseline fairness and inherited limitations from MDLM.
LDDM inherits MDLM’s fairness issues:
Perplexity is computed as an upper bound, which is not directly comparable to autoregressive likelihoods.


* Trivial or overly simple benchmarks.
Language modeling experiments are limited to LM1B and OpenWebText at relatively small scale.
The “reasoning” benchmarks (Countdown, Game of 24) involve symbolic arithmetic and tiny vocabularies.
These tasks do not capture the complexities of realistic generation (e.g., global coherence, factual reasoning, long context).
The method is not tested on any challenging downstream generation benchmarks (summarization, QA, GSM8K, MATH, etc.).

* Efficiency and compute cost not substantiated.
The paper claims fewer idle steps and higher efficiency, but provides no wall-clock, FLOP, or token-throughput comparisons to MDLM or UDLM.
Training is actually ~30% slower due to doubled embeddings and latent propagation overhead.
There is no evidence that LDDM improves quality per unit of compute.

### Questions
* Novelty vs self-conditioning: How does loopholing differ from existing self-conditioning techniques beyond storing an explicit latent ?
* Please include an ablation where standard self-conditioning is applied to MDLM without your recurrent latent path, to show what the new component adds.

* Isolation of latent path effect:
Can you run controlled ablations for:
(a) no recurrent latent but with self-conditioning;
(b) recurrent latent but single-pass training;
(c) caching and reusing previous x_{\theta, t} instead of h_t? 
	​This would clarify which mechanism drives the improvements.

* Task breadth: 
The current benchmarks (LM1B, OWT, Countdown, Game of 24) are short and low-complexity.
Can you evaluate on actual generation tasks such as summarization (CNN/DailyMail, XSum), long-form reasoning (GSM8K, MATH), or instruction-following datasets?
Without these, it’s unclear if LDDM meaningfully improves text generation quality.

* Perplexity vs real performance:
Since the evaluation focuses on approximate NELBO perplexity and GPT-based metrics, can you show concrete qualitative examples or human evaluations demonstrating that loopholing produces better text (fluency, coherence, factuality) than MDLM or AR baselines?

* Efficiency claim:
Can you provide real compute metrics (tokens/sec, FLOPs/sample, latency for 1k tokens) to support the “higher efficiency” claim?
Does LDDM actually require fewer diffusion steps at equivalent output quality?

* Baseline fairness:
Are the MDLM and AR baselines trained with equal data, compute, and precision settings?
How sensitive are the reported improvements to these factors?

* Robustness of evaluation modifications:
For MGDM, your evaluation differs from Ye et al. (2024). Can you report results under the original evaluation scheme for transparency?

### Soundness
2

### Presentation
3

### Contribution
1
