# Logit‑KL Flow Matching: Non‑Autoregressive Text Generation via Sampling‑Hybrid Inference

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Non-autoregressive (NAR) language models offer notable efficiency in text generation by circumventing the sequential bottleneck of autoregressive decoding. However, accurately modeling dependencies in discrete sequences remains challenging in this paradigm. In this work, we advance the field of NAR generation by applying conditional flow matching (CFM) methods grounded in geometrically principled interpolation, specifically leveraging Kullback-Leibler (KL) divergence geodesics, which correspond to linear interpolation in logit space. We rigorously establish that maximizing conditional likelihood in this setting precisely recovers the flow matching velocity field, supplying the theoretical justification for this approach in sequence modeling. To address practical performance gaps of \emph{basic} inference, we propose a novel empirical \emph{sampling} strategy that iteratively denoises and re-noises, along with a \emph{hybrid} scheme that integrates our \emph{sampling} method with \emph{basic} procedure. Across unconditional and conditional text and code infilling, the approach improves perplexity and downstream metrics over prior NAR baselines under matched settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Logit-KL Flow Matching (KL-Flow), a non-autoregressive (NAR) text generation framework that performs flow matching in logit space rather than probability space.
By interpreting the KL divergence geodesic as a linear path in logits, the authors provide a theoretically grounded interpolation scheme for discrete sequence modeling. They further introduce an iterative sampling–hybrid inference procedure combining deterministic ODE integration and stochastic denoising steps.
Empirically, KL-Flow shows consistent gains over prior discrete flow and diffusion baselines (DFM, Dirichlet Flow, Fisher Flow, SEDD) on unconditional, conditional, and code infilling benchmarks.

### Strengths
1. The paper rigorously connects conditional likelihood maximization to flow velocity recovery in logit space.
2. The token-wise conditional likelihood formulation is both simple and tractable, providing a bridge between discrete diffusion and flow models.
3. The hybrid inference scheme demonstrates consistent improvements across several datasets (TinyStories, FineWeb, WMT14, MBPP), showing competitive results for both language and code tasks.

### Weaknesses
1. While the paper covers some discrete flow and diffusion models, it does not include strong diffusion-based text generation baselines such as MDLM (Masked Diffusion Language Model) and conditional NAR transformers such as Tracformer (https://arxiv.org/pdf/2502.07616?)
2. Most experiments focus on small to mid-scale models (≤1.5B parameters); Would scaling the model or using stronger backbones (e.g., Llama-2 or Mistral) improve performance beyond the 1.5B-parameter setting?
3. Although empirically effective, the hybrid scheme’s transition point t* is largely heuristic. Could the authors provide intuition or ablation results for how the choice of the deterministic–sampling switch point t* influences diversity and perplexity across datasets?

### Questions
See Weaknesses.

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
3

### Summary
The paper proposes a non-autoregressive (NAR) text generator built on conditional flow matching (CFM) in logit space. Instead of interpolating token probabilities linearly on the simplex or along Fisher–Rao geodesics, it interpolates logits between a simple Dirichlet-like start and the target one-hot token, which the authors argue is the KL geodesic; they show that, under this path, maximizing the conditional likelihood ( $\log p_\theta(x_1 \mid x_t, t)$ ) exactly recovers the desired flow velocity field. On top of that, they introduce a hybrid inference scheme that runs a deterministic ODE-style update in the early time steps and switches to an iterative sampling / re-noising procedure in later steps to fix token-level errors. On several text, conditional, and code-infilling benchmarks, the method outperforms earlier discrete / Dirichlet / Fisher flow-matching baselines and comes close to, but does not fully match, similarly sized AR models.

### Strengths
1. Clear geometric diagnosis + concrete fix. The paper identifies a real failure mode of earlier probability-space paths — linear on the simplex, Fisher-Rao sphere, even some Dirichlet settings — namely that $KL((x_{\text{data}}|x_t))$ collapses too quickly so mid-time supervision vanishes on large vocabularies. The logit-space (KL-geodesic) interpolation directly targets this and shows improved calibrations against those baselines. This is well aligned with prior observations in Dirichlet Flow Matching that “naïve linear FM on the simplex is pathological.”
2. Bridging “train a denoiser” and “learn the flow field” for sequences, not just single tokens. Earlier CFM/DFM papers had versions of “conditional likelihood recovers the field,” but mostly in single-site or weaker sequence assumptions; this paper pushes the argument specifically for logit-KL paths and uses it to justify a very practical objective (just NLL on corrupted sequences). That reduces the gap between elegant flow theory and what people actually train.
3. Inference is engineered rather than hand-waved. Many discrete flow papers stop at “we have the field, integrate it”; here they run a 3-way comparison (deterministic / stochastic / hybrid) and show that pure ODE is insufficient and pure sampling collapses entropy, while a staged hybrid fixes both. That’s a useful empirical lesson for the whole discrete-flow community.

### Weaknesses
1. Novelty margin over very close contemporaries is thin. At least two 2024–2025 papers already investigated conditional text generation via KL-geodesic / logit-space flow matching and even proposed almost the same empirical “sampling + noise re-injection + hybrid” recipe to fix the underperforming basic sampler. The descriptions in Sevriugov & Oseledets (2024) and its 2025 extensions match this work’s geometric choice and sampling intuition almost line-for-line. If the contribution here is meant to be “we prove the conditional-likelihood = exact field under this path and scale it to larger datasets,” the paper needs to separate itself much more crisply from those concurrent KL-geodesic efforts, especially since they also claim better results over discrete FM. Right now the delta looks incremental.
2. Key equivalence rests on a factorized / per-position view that is not obviously valid in early timesteps. The derivation leans on the idea that the optimal vector field at time (t) can be written as an expectation of target logits under ($p(x_1 \mid x_t)$) token-wise. But in NAR text, ($p(x_1 \mid x_t)$) is usually not well factorized when (t) is small: tense, agreement, long-range topic constraints all couple positions. The paper’s fix is “do deterministic updates early, sampling later,” which is an empirical workaround, not a proof that the factorization is OK. So a central theoretical selling point (“likelihood = flow”) is relying on a data-distribution property that’s weakest exactly where the model needs guidance most. That’s a structural, not cosmetic, gap.
3. Evaluation uses LM-perplexity proxies and medium-scale AR baselines, so the true competitiveness is unclear. Measuring perplexity by scoring NAR outputs with an external LM is standard for discrete flows, but it is a proxy; it is known to favor models that mimic the scorer’s style rather than models that are truly diverse or controllable. And the main AR point of comparison is GPT-2-class models, not the 2025-era instruction-tuned or code-tuned LLMs that NAR methods would actually have to replace. In other words, the paper shows “better than prior discrete flows” but not “this can plausibly replace competitive AR models under identical training budgets.” That’s a material, not rhetorical, limitation.
4. Hybrid inference is admitted to be heuristic and under-analyzed. The whole motivation of the paper is “basic ODE flow isn’t good enough on text,” which is fair; but the proposed fix (early deterministic, late sampling + noise) is only justified by curves. There is no stability analysis, no guarantee of staying close to the learned KL-geodesic, and no complexity comparison with recent few-step DFM / consistency-trained flows that do target a fixed number of steps directly. A reviewer can reasonably ask why we shouldn’t just adopt FS-DFM-style step-budget-aware training or consistency distillation to get the same effect with a cleaner theory.

### Questions
1. How does this behave under true few-step regimes (e.g. 8–16 NFE) against step-consistent discrete flows like FS-DFM or consistency-trained DFM? Right now the advantage is shown largely when you afford hundreds of steps or a tailored hybrid schedule; but the practical motivation for NAR is low latency. A head-to-head with step-budget-aware models is missing. Can the KL-geodesic path still deliver better gradients than Dirichlet / Fisher when you compress the time discretization that aggressively?
2. What exactly is the uniqueness claim over existing KL-geodesic / logit-FM papers? Several public works from late 2024 onward already stated (i) “logit-space interpolation is the KL geodesic,” (ii) “maximizing ($p_\theta(x_1 \mid x_t,t)$) gives you the right field,” and (iii) “basic deterministic inference is weak; add an iterative sampling-and-noise scheme; hybridize.” If this paper’s contribution is a stronger sequence-level derivation or better scaling to open-domain data, please spell out the technical gap (e.g. a specific lemma about token-wise optimality under KL-paths, or a complexity advantage in hybrid inference) that is not present in Sevriugov & Oseledets (2024) or the concurrent KL-geodesic variants. Right now it reads more like a consolidation than a breakthrough.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a novel non-autoregressive text generation framework that uses a Kullback-Leibler (KL) divergence geodesic for interpolation, which is shown to be equivalent to linear interpolation in logit space. The objective function is to minimize the negative log-likelihood of the target distribution at the sequence level. Experiments demonstrate that the proposed method outperforms other non-autoregressive (NAR) baselines such as DFM and SEDD.

### Strengths
The proposed method achieves strong performance against other NAR baselines across various tasks.

The discussion of the inference process is insightful.

### Weaknesses
It is unclear why the deterministic inference process performs poorly, given that the "Logit-KL Flow Matching" objective recovers the velocity field.

The efficiency of the proposed method is not thoroughly discussed, particularly in comparison to methods that are trained with an MSE loss (in training) and solve ODEs using numerical techniques (in inference).

Perplexity is measured using samples from GPT-2, GPT-3, and Llama-2, which may introduce bias from these reference models.

### Questions
In the experiments, the authors state that all models use a bidirectional transformer backbone. Was the GPT-2 baseline trained in an autoregressive manner, and was its causal attention mechanism replaced with bidirectional attention?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a non-autoregressive text-generation framework that performs conditional flow matching on the probability simplex using KL-divergence geodesics. The authors show (in their way) that maximizing the token-level conditional likelihood exactly recovers the flow-matching velocity field, providing a principled training objective.

### Strengths
1. The mathematical descriptions in this paper are relatively accurate.
2. The paper is well structured.

### Weaknesses
1. The used base model in the experiments seems obsolete, which makes it unclear whether the proposed method still works for the SOTA models nowadays.
2. Lines 226 ~ 228 seem confusing. A bidirectional attention is used to model sequence-level NLL, but the condition variable $x_t$ represents a single token, not a total sequence. I am not convinced of this modelling approximation.
3. Line 309 says all models used in the experiments are bidirectional backbones. How do you apply this to text generation, where you do not have access to any future token information?
4. I see some autoregressive models have been used in the experiments (GPT-2), so you replaced the causal attention in the model with a bidirectional attention? If so, how do you initialize your model weights? Is it the same with the pre-trained weights?
5. The experiment setting does not include baseline introductions, which makes the reader very hard to get familiar with the relevant work.
6. The proposed method in this paper has two inference methods, namely, basic inference and sampling inference. However, there is no formal section in the paper to systematically compare these two inference methods and discuss the corresponding advantages and disadvantages. The experiment section also missed this.

### Questions
No, see weakness.

### Soundness
2

### Presentation
3

### Contribution
2
