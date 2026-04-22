# Esoteric Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Diffusion-based language models offer a compelling alternative to autoregressive (AR) models by enabling parallel and controllable generation. Among this family of models, Masked Diffusion Models (MDMs) achieve the strongest performance but still underperform AR models in perplexity and lack key inference-time efficiency features—most notably, KV caching. In this work, we introduce Eso-LMs, a new family of models that fuses AR and MDM paradigms, enabling smooth interpolation between their perplexities while overcoming their respective limitations. Crucially, we introduce KV caching for MDMs while preserving parallel generation, significantly improving inference efficiency. Combined with an optimized sampling schedule, our method achieves a new state of the art on the speed-quality Pareto frontier for unconditional generation. On long contexts, our method achieves 14−65× faster inference than standard MDMs and 3−4× faster inference than prior semi-autoregressive approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The most important contribution of this paper is a novel hybrid architecture that combines the AR and MDM paradigms for text generation. This method can be viewed as a smooth interpolation between AR and MDM, and shows the potential for a trade-off of their pros and cons.
Another interesting contribution is the KV-caching scheme for the AR component during inference, which can improve sampling efficiency.

### Strengths
- The presentation of this paper is clear and easy to follow.
- The proposed hybrid architecture that combines the AR and MDM paradigms introduced a flexible dimension for model design space, i.e., $\alpha_0$. The experiment in Section 5.2 suggests that a non-trivial choice of $\alpha_0$ makes sense.
- The proposed KV-caching scheme for the AR component during inference can improve sampling efficiency.
- The

### Weaknesses
- The authors put their emphasis on the trade-off between efficiency and speed. However, many studies also pay attention to the global understanding ability and better generalization on downstream tasks of diffusion models. I think the authors should provide more evidence on these aspects.
- I think the sentence "This is the first time for IW bounds to be obtained for discrete diffusion" in Line 393 overclaims their contribution. 
- In Figure 2, the curve exhibits a sudden decrease when $\alpha=0.125$. Does this imply that this method is not robust enough?
- MDM generally samples slower than AR, as KV caching is not enabled. Also, AR tends to give a lower perplexity. Then how can we expect a trade-off between efficiency and speed?

### Questions
- In line 425, the authors state that settign $\alpha^{train]=1$ performs the best. Doesn't this seem contradictory with the motivation of interpolating between AR and MDM?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Eso-LMs, a hybrid language modeling framework that interpolates between masked diffusion models (MDMs) and autoregressive (AR) models. The core ideas are: (i) a two-phase sampling procedure where some tokens are denoised in parallel (diffusion) and the rest left-to-right (AR), and (ii) an attention-bias scheme that enables causal attention and unified KV caching even during diffusion, improving inference efficiency. A variational bound decomposes training into an AR loss over masked positions plus an MDM loss, with a hyperparameter controlling the interpolation. Empirically, Eso-LMs achieve competitive perplexities versus MDMs and a better speed–quality Pareto frontier, with large speedups at long context via KV caching.

### Strengths
- Clear hybrid objective with principled ELBO derivation and interpolation knob $\alpha_0$ (Sec. 3.1, Eq. 7).
- Practical sampling schedule plus attention bias enabling KV caching in both phases; addresses a key MDM bottleneck (Sec. 3.2, 4.2).
- Strong speed–quality trade-offs and long-context latency improvements vs. MDLM/BD3-LM (Figs. 3–4; Table 9).
- Useful analysis of importance-weighted bounds for discrete diffusion to approximate true PPL (Sec. 3.3; Table 2, Table 6)

### Weaknesses
- Sampling schedule clarity: Sec. 3.2 introduces a “Denoising Schedule” by name before defining how it is computed; the detailed procedure is deferred to Appx. B.3 and could be surfaced earlier for readability [p4, lines 212–215].

- Efficiency restriction explanation: The claim “restrict the forward pass at step k to only the previously denoised tokens and the current mask tokens” would benefit from a precise equation/attention mask description on the main text page [p4, lines 223–229].

- Diffusion-phase attention assumption: Sec. 4.1.1 states “mask tokens are denoised using only clean tokens but clean tokens do not attend to mask tokens,” which deviates from common MDM training with bidirectional attention; the rationale relies on the proposed causal-bias construction and AO-ARM connections, but the text could more explicitly contrast with standard MDM [p5, lines 262–267].

- Train–test mismatch concerns: The sequential-phase uses a concatenation $z_0\oplus x$ for training (Sec. 4.1.2), but this concatenation is “unnecessary” at sampling; please justify that this does not introduce a distribution shift or extra leakage [p5–6, line 287].

### Questions
- Sec. 3.2, line 212: How exactly do you pre-compute the order in which tokens are denoised? Please make it clear in the main text with a concise algorithm.

- Sec. 3.2, line 223: “Restricting the forward pass at step k to only the previously denoised tokens” — could you provide a compact equation for the attention bias or a figure on the main page that shows Q/K/V subsets and KV reuse at step k?

- Sec. 4.1.1, line 262: The statement “mask tokens are denoised using only clean tokens but clean tokens do not attend to mask tokens” is unclear. Please reconcile this with standard MDM training where mask and clean tokens interact bidirectionally, as in Large Language Diffusion Models (Nie et al., 2025). What assumption or theoretical result justifies your restriction?

- Sec. 4.1.2, line 287: The training concatenation $z_0\oplus x$  seems different from inference. Is there a gap between training and inference? Please specify conditions under which this is unbiased and show an ablation that removes concatenation.

### Soundness
3

### Presentation
2

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
This paper introduces a hybrid LLM paradigm that aims to unify autoregressive (AR) and masked diffusion model (MDM) for language modeling. The key idea is to interpolate between AR and MDM objectives. Correspondingly, the transformer architecture is modified to accommodate both causal (AR) and bidirectional (MDM) attention, and — crucially — enable KV caching during diffusion. 
This design allows Eso-LMs to achieve parallel generation while retaining AR-level inference speed.
Experiments on LM1B and OpenWebText show that Eso-LMs interpolate between diffusion and AR perplexities, establish better results on the speed–quality Pareto frontier.

### Strengths
* Timely and relevant topic: Bridging the gap between AR and diffusion models addresses a central limitation in current LLMs, shedding light to how to better balance quality and efficiency in language modeling. 

* Clear technical contributions: The proposed hybrid formulation and unified attention mechanism are clearly presented and novel to me. The derivations seem technically sound.

* Strong empirical results: The model demonstrates consistent improvements over both MDM and block-diffusion baselines in perplexity and inference efficiency on benchmarks. Even though the scale is relatively small, the results look promising.

### Weaknesses
### Presentation: 
Even though the high-level idea is clearly presented in Sec 3.1, there are a lot of technical details in the implementation process and I feel these subsequent designs are not well-justified and discussed. The ablation analysis is somewhat limited given the number of components involved.

### Scientific value
I appreciate this work on the technical level very much. But I think the scientific values can be further improved. 
The combination of AR and diffusion objectives feels somewhat mechanical—focused on unifying two training losses rather than addressing a specific modeling limitation or data property. 
It remains unclear what type of data/scenario actually benefits from this interpolation, e.g., general, coding, math, agent, etc. 
Given that actually implementing the proposed model requires significant changes to the training/inference pipelines and transformer architectures, and that setting alpha_0 to 1 or 0 performs significantly worse than standard MDM and GPT, I feel this paper could provide more intuition/justification for why this interpolation helps beyond the observed empirical trade-offs in limited benchmarks.

### Questions
* The decomposition in Eqn (5) seems a bit artificial. Could you provide more intuitive motivation for why this hybrid formulation could yield a better generative model, and under what data regimes (some sufficient conditions) it is expected to help? 

* In table 5 of appendix C.1, what is the reason for the divergence? 

* How does this work relate to the any-order GPT line of work [1,2], which also kind of combines GPT and MDM? 

* If my understanding is correct, for different values of \alpha_0, we have to re-train the model from scratch. Is there some potential methods that can accommodate multiple \alpha_0's in one training run? 

[1] σ-GPTs: A New Approach to Autoregressive Models, arxiv 2404.09562

[2] Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture, arxiv 2506.19935

### Soundness
3

### Presentation
2

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
This paper proposes Esoteric Language Models (Eso-LMs), a new model family that fuses autoregressive (AR) and Masked Diffusion (MDM) models, aiming to get the best of both: the quality and efficiency of AR models and the parallel generation capabilities of MDMs. Specifically, it introduces a novel training procedure and attention mechanism that enables KV caching during the diffusion phase. This is achieved by replacing the standard bidirectional attention in MDMs with a causal attention mechanism on a shuffled sequence, allowing a single, unified KV cache to be shared across both parallel (diffusion) and sequential (AR) generation phases. The model is trained using a hybrid objective that combines a standard autoregressive (AR) loss with a Masked Diffusion Model (MDM) loss , allowing the model to interpolate between the two paradigms. The experiments show that Eso-LMs achieve a new state-of-the-art on the speed-quality Pareto frontier among MDMs, and on long contexts, they are 14-65x faster than standard MDMs and 3-4x faster than prior semi-autoregressive approaches.

### Strengths
1. KV caching is a key problem in MDM generation. Eso-LMs achieve massive inference speedups, especially on long contexts.
2. The authors identify key shortcomings in the previous hybrid model, BD3-LM, such as "degraded samples at low sampling steps" and "incomplete caching". Eso-LMs are shown to overcome these limitations and outperform BD3-LMs. Especially, Eso-LMs remain competitive with MDMs in the low-NFE regime and with AR models in the high-NFE regime. 
3. The model uses a new training and sampling procedure that replaces bidirectional attention with causal attention on a shuffled sequence. This is the key innovation that enables KV caching during the diffusion phase.
4. Computing importance-weighted bounds for discrete diffusion models is also an interesting contribution.

### Weaknesses
1. Perplexity still lags behind AR models. The hybrid training objective pushes the model closer to AR models, but there is still a gap (24.51 vs 22.38 on LM1B, 20.86 vs 17.90 on OWT). 
2. Replacing bidirectional attention with causal attention to enable KV caching comes at a cost. When Eso-LM is trained as a pure diffusion model ($\alpha_0=1$), its perplexity is noticeably worse than the standard MDLM baseline (which uses bidirectional attention). 
3. When evaluated on unseen datasets (zero-shot likelihood evaluation in Appx E2), the Eso-LMs performed consistently worse than all other baselines, including the standard AR model, MDLM, and even BD3-LMs.

### Questions
Why is the zero-shot generalization worse than baselines? Is such overfitting expected? What are the possible reason for this? 

Why is the training time with $\alpha_0<1$ 1.37x of the pure MDLM ($\alpha_0=1$)?

### Soundness
3

### Presentation
3

### Contribution
2
