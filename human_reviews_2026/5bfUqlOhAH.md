# Efficient Autoregressive Inference for Transformer Probabilistic Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
Set-based transformer models for amortized probabilistic inference and meta-learning, such as neural processes, prior-fitted networks, and tabular foundation models, excel at single-pass _marginal_ prediction. However, many applications require _joint distributions_ over multiple predictions. Purely autoregressive architectures generate
these efficiently but sacrifice flexible set-conditioning. Obtaining joint distributions from set-based models requires re-encoding the entire context at each autoregressive step, which scales poorly. We introduce a _causal autoregressive buffer_ that combines the strengths of both paradigms. The model encodes the context once and caches it; a lightweight causal buffer captures dependencies among generated targets, with each new prediction attending to both the cached context and all previously predicted targets added to the buffer. This enables efficient batched autoregressive sampling and joint predictive density evaluation. Training integrates set-based and autoregressive modes through masked attention at minimal overhead. Across synthetic functions, EEG time series, a Bayesian model comparison task, and tabular regression, our method closely matches the performance of full context re-encoding while delivering up to $20\times$ faster joint sampling and density evaluation, and up to $7\times$ lower memory usage.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a causal autoregressive buffer for transformer-based amortized probabilistic models (NPs, PFNs, tabular FMs). The key idea is to encode the context once and keep it in a static cache; generated targets enter a causal buffer that attends to the context cache and prior buffer tokens, enabling joint dependencies without re-encoding the augmented set each step. Complexity drops from (O(K(N!+!K)^2)) to (O(N^2 + KN + K^2)), supporting batched AR sampling and single-pass joint likelihood evaluation. Experiments across synthetic, EEG, cognitive, and tabular tasks match standard AR accuracy while achieving speedups.

### Strengths
This is a practical architectural improvement that preserves set-conditioning benefits while unlocking efficient joint predictions typical of AR models. The idea of a role-aware attention mask (immutable context, causal buffer, no writes back to context) feels simple yet powerful, and widely applicable to TNPs/PFNs/tabular FMs. The potential for large efficiency gains in meta-learning scenarios with repeated sampling is significant.

- The factorization and masking constraints (R1–R4) are clearly stated; buffering semantics are sound. The complexity analysis matches the masking structure.
- Training uses a buffer-size curriculum with structured masks that let targets attend to context or a variable prefix of the buffer, aligning training and inference. The link to minimizing KL to posterior predictive under varying conditioning sets is consistent with PFN theory.
- The claims of accuracy comparison with standard AR and speedups are supported on small/medium contexts and (K< 32) (as per reported settings). The authors also acknowledge a degradation risk when target counts exceed the trained buffer size.

### Weaknesses
In the experiments, they are rather limited to moderate sizes of K and small/medium-sized contexts. I have a feeling that this may be limiting the characterization of the approach. Also, residual order dependence for likelihood evaluation requires averaging; I think an analysis on the order sensitivity is missing.

### Questions
- Is it possible to have tests on larger (K) regimes and very large contexts to map out failure boundaries, plus ablation on positional embeddings inside the buffer and ordering effects when approximating permutation invariance via order-averaging?
 - Is it possible to quantify order-averaging required to stabilize joint likelihoods with a metric measuring their cost/benefit?
- How do the authors decide on the size of (N)s? Is it there a experiment for emprically showing for large values of N, proposed method has compounding advantages?

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
2

### Summary
The paper addresses a conflict between two types of models: set-based probabilistic models and autoregressive models. When models (like Neural Processes) try to generate a sequence of predictions (e.g., filling in a signal), they have to re-process all the original data and all the new predictions at every single step. The core idea is to decouple the static context from the dynamic predictions by introducing a new architectural component called the "causal autoregressive buffer". The advantage is (i) on the computational advantage: The expensive $\mathcal{O}(N^2)$ cost (encoding the context) is paid only once at the beginning and (ii) preserved accuracy as shown in the experiments section.

### Strengths
1. The paper is well-written and novel. The problem of repeated, expensive re-encoding is visualized in Figure 1, which provides an intuitive understanding of the entire paper's premise. The proposed solution is an elegant architectural fix that decouples the static context from the dynamic predictions, supported by a clear complexity analysis ($\mathcal{O}(N^{2}+NK+K^{2})$). 

2. The empirical evidence is convincing. It delivers a speedup (up to 20x) with small additional training cost and without sacrificing predictive accuracy.

### Weaknesses
I'm not an expert in this field (neural processes, probabilistic meta-learning, amortized inference), so I'll leave my comments in the following Questions sections.

### Questions
1. What happens when the target sequence length M is larger than the buffer size? I saw a discussion on Appendix E.2 and also the limitation in the final discussion section, which suggests that to achieve better performance, we need more computational time. I'm wondering what the final complexity is if including the operation "we evaluate every $K$ targets once and perform AR for $M/K$ steps". Will this still lead to the 20 times speed up? How often does the situation "target count exceeds training bounds of the buffer" happen in real-world analysis?

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
3

### Summary
The paper proposes a “causal autoregressive buffer” to accelerate joint inference in transformer-style probabilistic models such as Transformer Neural Processes (TNPs) and Prior-Fitted Networks (PFNs). The idea is: instead of repeatedly re-encoding the entire context set at every autoregressive step, the model encodes the context once, caches it, and then maintains a lightweight causal buffer of generated targets. This is claimed to both (i) avoid recomputing attention over the full growing set, and (ii) still model target-to-target dependencies during sequential rollout. The authors argue this reduces complexity from  
$O(K (N+K)^2) \) to \( O(N^2 + NK + K^2) $ 
and yields up to 20× faster joint sampling while matching predictive quality of stronger autoregressive baselines, across synthetic regression, EEG forecasting, and tabular regression tasks.

### Strengths
### 1 Practical inference speedup for probabilistic transformers  
The work directly targets a real bottleneck: autoregressive joint inference in set-conditioned probabilistic models is extremely expensive if you have to repeatedly re-encode the entire (context ∪ generated targets) set at every step. The proposed causal autoregressive buffer reduces the need to rerun the full encoder every time by caching the context representation once and then only extending a lightweight causal buffer of generated targets. This reduces the stated complexity from \(O(K (N+K)^2)\) to \(O(N^2 + NK + K^2)\) for K rollout steps, which the authors claim translates into 3×–20× wall-clock speedups in their experiments.
This is a meaningful systems win for anyone who wants to deploy TNP/PFN-style models in real-time or interactive settings.

### 2 Keeps target-to-target dependency modeling (vs. fully independent heads)  
A common cheap fallback in meta-regression / neural process style models is to just predict each target independently conditioned on the same context encoding. That is fast but discards correlations across targets.  
Here, the buffer still allows each new target to attend causally to previously generated targets in the rollout, so you capture sequential statistical structure across predicted outputs without paying the full re-encode cost. This is especially relevant for tasks like trajectory / time-series forecasting (e.g. EEG forecasting) where temporal consistency matters.  
In other words, you get something closer to coherent joint samples instead of just pointwise fits.

### 3 Unification: one model that can act both “independently” and “autoregressively”  
The paper proposes a training curriculum in which, during training, some targets are forced to condition only on the static context block (independent prediction mode), and others are allowed to attend to a prefix of previously generated targets through the buffer (autoregressive mode). The same set of parameters is trained to handle both regimes.  
If this actually works robustly, it’s attractive from an engineering perspective: you don’t need to maintain two separate heads (one batchable + independent, one fully autoregressive). You get a single model that can cheaply do fast batched prediction or slower-but-more-coherent sequential rollout.

### 4 Conceptual bridge between permutation-invariant context encoders and standard AR decoding  
Architecturally, the paper is trying to marry two worlds:
- permutation-invariant / set-encoder style context processing (as in Neural Processes, PFNs, etc.), and
- standard causal decoding with KV caching (as in language models).

### Weaknesses
## 2. Major Concerns

### 2.1 Claimed novelty vs. standard KV caching  
The proposed “causal autoregressive buffer” is essentially a block-structured attention mask plus caching of keys/values:

- A *static* context block (permutation-invariant encoder over the observed context set) is computed once and cached.
- A *growing* autoregressive buffer stores keys/values for previously generated targets and is updated step by step with a causal mask.
- During generation, new target tokens attend to (a) the frozen context block and (b) previous targets in the buffer, but the buffer cannot “write back” into the context block.

This is very close in spirit to ordinary autoregressive Transformer inference with KV caching: you prefill a prefix once, then incrementally extend the cache with each new token under a causal mask. 

The paper repeatedly describes this as a general architectural mechanism that “decouples context encoding from sequential prediction,” but standard decoder-only Transformers already do this: prefix context is cached, new tokens causally attend to the prefix and prior tokens. The only real twist here is that the “prefix” came from a permutation-invariant encoder instead of from plain tokens, and that the mask enforces a hard separation between context and generated targets (“R1–R4”).
Right now, the paper oversells this as conceptually new. It reads more like applying known KV-caching style inference to set-conditioned meta-learners. The authors need to justify why this is more than “cache the context encodings and use a causal mask for the generated block.”

---

### 2.2 Theory is bookkeeping, not analysis of approximation error  
The main “theoretical” contribution is a complexity argument: naive autoregressive deployment of a TNP-like model requires re-running attention over the enlarged context-plus-buffer at every step, costing \( O(K (N+K)^2) \), while the buffer method pays \( O(N^2) \) once for the context and then \( O(NK + K^2) \) to roll out K targets.

This is essentially computational accounting, not a theoretical guarantee about inference quality. The paper makes a strong claim that the buffered rollout “preserves model quality,” because each new prediction can still attend to all prior targets via the buffer and to the cached context. But there is no formal analysis of when this approximation matches the behavior of the “true” fully-updated model that would have re-encoded the augmented context set after each new target is added.

In fact, once you stop re-encoding, you’ve thrown away strict permutation invariance over the *union* of context and generated targets. After generation begins, the model is no longer allowed to revisit and re-symmetrize the combined set. The paper acknowledges degradation when the number of autoregressive targets exceeds the buffer size K used during training.
This is a core limitation, but it’s treated as an implementation detail rather than a fundamental modeling gap.

---

### 2.3 Empirical evaluation seems arranged to flatter the method  
Most of the storytelling is about runtime: joint sampling speed, likelihood evaluation speed, etc. The new buffer method is reported as 3–20× faster than autoregressive baselines like TNP-A or TNP-D-AR, which repeatedly re-encode context at each step, especially at large N.

Concerns:

- The paper admits their implementation uses heavy engineering (FlashAttention-2, Triton kernels, KV caching), and also says baselines were “optimized beyond their public versions.” But we never see an ablation that isolates the architectural contribution from pure systems tuning. We can’t tell if the 20× number is “new idea” vs. “better CUDA.”
- There is no memory comparison. The buffer maintains per-layer KV caches for both the frozen context block and the autoregressive buffer tokens, which can blow up for long rollouts and large batch sizes. Only wall-clock time is emphasized, which is convenient but incomplete.
- When you actually look at predictive quality (e.g. EEG forecasting), the buffered model with K=16 can underperform the slow full autoregressive baseline (TNP-D-AR), and sometimes is even worse than extremely small-buffer variants (effectively K=1).
  So the tradeoff is not “same accuracy, way faster.” It’s “sometimes noticeably worse, but faster.”

Despite that, the abstract and intro still assert that the method “matches predictive accuracy … while delivering up to 20× faster joint sampling,” which is too strong given these cases.

In short, the evaluation is narrated as “little to no quality loss, huge speedup,” but the actual numbers show a real quality vs. speed tradeoff.

---

### 2.4 Generality claims are speculative and weakly supported  
The paper claims broad applicability: Perceiver-like architectures with pseudo-tokens, probabilistic neuroscience modeling, and “tabular foundation models” such as PFN-style inference.

But:

- The Perceiver extension is hypothetical; there is no experiment demonstrating the buffer with learned pseudo-tokens.
- The “tabular foundation model” experiment is actually quite small. The model is trained from scratch on synthetic structural causal model data, then evaluated on a few UCI-like tabular tasks with N=128 context / M=32 targets.
  That’s nowhere near the scale implied by the phrase “foundation model.”
- In those tabular results, “AR w/ buffer (K=32)” performs roughly on par with a standard AR baseline (K=1) and modestly better than fully independent predictions.
  That mostly shows that causal conditioning across targets helps, which is expected. It does not prove that the specific buffering trick is uniquely enabling.

So the claim that this mechanism is a general, broadly applicable inference upgrade feels speculative. The evidence is narrow, and in some cases purely suggestive.

---

### 2.5 Missing baselines / missing ablations  
The baselines are primarily TNP variants (TNP-D, TNP-D-AR, TNP-A, etc.).
However, to really argue “our method is necessary,” I would expect at least:

1. **A simple two-tower baseline**  
   - Tower A encodes the context set once (frozen).
   - Tower B is a causal decoder over targets that cross-attends into Tower A, with standard KV caching.
   - This is, in spirit, what the proposed model is.  
   Without ablations on masking structure and curriculum, we don’t know if the fancy buffer rules (R1–R4) are actually critical or if a trivial cross-attend-decoder would do the same.

2. **A plain autoregressive Transformer treating the context as a prefix**  
   - Just linearize the context into a token sequence, feed it as a prefix prompt, and then autoregress over targets exactly like a language model, ignoring permutation invariance entirely.
   - The paper insists that permutation invariance of the context block is crucial, but it never quantifies how much you lose if you drop that constraint and just go full decoder-style.

### Questions
see in weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Neural processes are finding applications across a range of areas, including in tabular foundation models like TabPFN and in weather modelling e.g. in Aardvark Weather. In many of these applications, it’s important to be able to model joint predictive distributions that capture the dependencies between target variables at multiple locations e.g. when imputing multiple missing values into a row of a tabular dataset or when predicting precipitation across a region to assess flood risk in weather forecasting.  Autoregressive transformer neural processes are arguably the go-to method for solving such tasks.

This paper makes a neat contribution: it addresses one of the central limitations of autoregressive transformer neural processes and sequence models -- their high computational cost at inference time. It introduces a lightweight causal buffering mechanism that preserves the expressive power and calibration benefits of full autoregression while dramatically reducing computation and latency. This makes autoregressive probabilistic models viable for real-time and large-scale applications, from time-series forecasting to neural operator learning. In doing so, the paper bridges the gap between expressive but slow autoregressive methods and fast but less flexible parallel inference, providing a practical path toward scalable, uncertainty-aware inference in scientific and foundation-model settings.

I liked the paper and would vote for acceptance.

### Strengths
This paper has at its core a simple but neat idea that works well. I liked the fact that it had a wealth of experimental results including

1. Synthetic function modelling (Gaussian Process (GP) functions and sawtooth synthetic functions.)
2. EEG forecasting and interpolation including real EEG recordings from 20 human subjects
3. Multisensory causal inference modelling involving simulated and real human behavioural data from a multisensory integration study (Liu et al., 2025).
4. Tabular foundation model experiments
5. Further Efficiency and ablation studies

The results were strong -- the method really works.

The presentation and the writing were very clear and well polished. Figures were very well presented. Generally, I thought it was a well executed piece of work.

### Weaknesses
The idea is quite simple which could be viewed as a weakness, but I actually view this as a strength for multiple reasons: it's easy to understand, it's simple to implement, and because of this it could be deployed widely. 

The idea might also seem niche as AR-TNPs are not super well-known, but this is also not correct. TNPs in the sim2real setting were rebranded by the impactful TabPFN line of work and the contributions here are directly aligned with this breed of tabular foundation model as the final experiment shows. Moreover, NPs have been deployed in another Nature paper in Aardvark weather, which has also been impactful and which could leverage these results to produce scalable weather forecasts. I expect more and more examples of applied NPs will emerge over the coming years.

### Questions
Small point, but I’m not a huge fan of the phrase “joint likelihood” since in many contexts in the paper the primary focus is really producing a "joint predictive density” rather than the likelihood of the parameters need for e.g. learning. Indeed, if you trained a model only using likelihoods derived from univariate predictives, you could then immediately use this to produce joint predictive densities (c.f. the AR CNP paper). So I'd place the emphasis on joint distributions most of the time, rather than on the likelihood functions.

line 49: "However, this breaks the set-based structure”  I think that, at this stage in the paper, this is a bit ambiguous. E.g. in standard mode, adding the generated targets back into the context and recomputing everything is arguably as good as it gets in terms of permutation invariance. Perhaps say “However, this involves significant computational overhead…”

line 161: "In practice, Eq. (2) is not exact for likelihood evaluation as it breaks permutation invariance of the model.” I don’t agree with this. This is really a modelling choice, or something that is determined by the task itself, rather than being wrong or right. You can choose to model the order as latent (and therefore something that you need to average over) or known and fixed (in which case you don’t). For time-series, for example, there is often an implied order over the targets. For spatial data, there might not be. 

line 291: you could point out here that discrete diffusion is really an any order AR model in disguise and is very closely related to a NP with a discrete input set (arguably it is an NP) and as such could leverage your ideas.

### Soundness
4

### Presentation
4

### Contribution
3
