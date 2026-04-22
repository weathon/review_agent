# Optimal Stopping vs Best-Of-$N$ for Inference Time Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large language model (LLM) generation often requires balancing output quality against inference cost, especially when using multiple generations. We introduce a new framework for inference-time optimization based on the classical Pandora’s Box problem. Viewing each generation as opening a costly “box” with random reward, we develop algorithms that decide when to stop generating without knowing the underlying reward distribution. Our first contribution is a UCB-style Pandora’s Box algorithm, which achieves performance that is provably close Weitzman’s algorithm, the optimal strategy when the distribution is known. We further adapt this method to practical LLM settings by addressing reward scaling across prompts via a Bradley–Terry inspired transformation. This leads to an adaptive inference-time optimization method that normalizes rewards and learns stopping thresholds on the fly. Experiments on the AlpacaFarm and HH-RLHF datasets, using multiple LLM–reward model pairs, show that our adaptive strategy can obtain the same performance as non-adaptive Best-of-$N$ sampling while requiring 15-35\% fewer generations on average. Our results establish a principled bridge between optimal stopping theory and inference-time scaling, providing both theoretical performance bounds and practical efficiency gains for LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reframes Best-of-N as an optimal-stopping problem. The goal is to learn a fair-cap threshold online for a given prompt and stop once the running best reward crosses it. They estimate this threshold with a UCB-style rule and use a Bradley–Terry–style transform centered at a per-prompt quantile to normalize rewards across prompts. This adaptive policy is shown to achieve the same performance as a non-adaptive Best-of-N policy while requiring 15-35% fewer generations on average.

### Strengths
- Casting BoN as a fair-cap stopping problem is intuitive and principled. It introduces a formal decision-theoretic framework for inference-time alignment by providing  a concrete “when to stop” rule instead of grid-searching for N.
- I appreciate the thoughtfulness in considering practical challenges like cross-prompt reward miscalibration, as well as setting an acceptance-rate target.
- The empirical experiments are extensive, illustrative, and useful.

### Weaknesses
- My main concern is that best of $N$ is not run sequentially in practice. Modern inference systems are heavily optimized for parallel, batched decoding to maximize hardware utilization. An adaptive, sequential approach trades total compute for potentially higher latency and lower system throughput. This is why BoN is very attractive as an inference-time method, aside from its simplicity and effectiveness. The optimal stopping method adds more complexity because of the uncertainty in $N$, potentially making inference much less efficient.

- The sample counts (the $N$ in BoN) are large and unrealistic (> 50 in Figures 1 and 2). This may suggest (if I understand correctly) that the cost the authors fix to show the efficiency of their algoirthm may be unrealistic.

- There is a significant gap between the theory presented and the practical version of the algorithm. I appreciate the author's discussion of their algorithm that uses an explicit cost c and max utility B, which are acknowledged to be difficult to specify. However, what is actually deployable in practice is the target-acceptance variant. I believe this aspect was not discussed sufficiently, as it underpins the performance of their algorithm. This disconnect weakens the claim that the practical algorithm is principled and provably near-optimal.

- The practical algorithm's performance hinges on a strong, unmotivated parametric assumption: that the right tail of the (exponentiated) reward distribution can be modeled by a shifted exponential. The paper does not provide an empirical validation (e.g., goodness-of-fit tests) for this assumption. If there is a model mismatch, the UCB estimates for the fair-cap value could be invalid.

- The paper's novelty claims should be framed more precisely. I do not believe that the first contribution (claiming that you present the "the first stopping strategy that adapts to unknown reward distributions") is factual. The novelty is better framed as first application to LLM inference-time alignment, not the learning-theoretic idea itself.

### Questions
Q1: Qualitatively or quantitatively, can you discuss the tradeoffs between parallel BoN and sequential inference?

Q2: using the data collected for Figure 4, can you show how well the shifted exponential fit is on average per prompt? 

Q3: You set κ to the prompt-wise α-percentile and then map via AR_κ(v). This ties the utility to a quantile that itself is estimated online. This is confusing to me. This seems to say that the utility function is itself non-stationary. How does this reconcile with the theoretical analysis in Section 3, which assumes a fixed, albeit unknown, reward distribution?

Suggestion 1: Aside from fixed N, you could consider a baseline that stops after the reward has not improved by a meaningful amount for k consecutive steps, or one that stops after crossing a fixed reward threshold. Including such baselines would better isolate the specific benefits of the optimal stopping framework, which I do not exactly see. I would also test the impact of uncalibrated rewards to convince the readers of the usefulness of your BT-normalization compared to other standard techniques.

Suggestion 2: The current experiments evaluate the algorithm's end-to-end performance, but it is difficult to assess the accuracy of the underlying estimation of the fair-cap value τ, as the ground truth is unknown. Have you considered a setup with a verifiable reward (such as math)? I believe the usefulness of optimal stopping might be more apparent there.

Suggestion 3: I believe the paper is missing a few important references. One is AdaBoN [1] which is similar in spirit to your paper (posted on arxiv on May 2025). They also estimate the reward distribution online and then adaptively allocate the budget. Also, [2] train a model that predicts the expected marginal gain in reward from allocating additional responses to a prompt. I appreciate if you would discuss how your work compares to them. Moreover, [3] give a way to find the an optimal $N$ offline for BoN (posted on arxiv on June 2025)..However, they require access to a true reward calibration dataset.

[1] Raman, V., Asi, H., & Kale, S. (2025). AdaBoN: Adaptive Best-of-N Alignment. arXiv preprint arXiv:2505.12050.

[2] Damani, M., Shenfeld, I., Peng, A., Bobu, A., & Andreas, J. (2024). Learning how hard to think: Input-adaptive allocation of lm computation. COLM 2025.

[3] Khalaf, H., Verdun, C. M., Oesterling, A., Lakkaraju, H., & du Pin Calmon, F. (2025). Inference-Time Reward Hacking in Large Language Models. NeurIPS 2025.

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
2

### Summary
In this work, the authors model inference time sampling with early stopping as a "Pandora's Box Algorithm", where the goal is to develop an algorithm that decides when to stop generating inference samples and select the best one out of the existing set. The goal of this approach is to get efficiency speedups over having a fixed number of generations N a priori which might be wasteful if the reward-maximizing generation occurs early. 

They propose a UCB-Style Pandoras Box algorithm to handle settings where the underlying reward distribution is unknown and use this to create an on-the-fly inference time algorithm that determines when to stop adaptively.

### Strengths
* The framing of the problem is novel and highlights an area of improvement for the design of inference time algorithms
* The authors provide theoretical insights into the optimality gap of their approach.
* Through experiments, the method is demonstrated to have significant savings in terms of number of samples compared to Best-of-N.

### Weaknesses
* I'm not convinced that the Bradley-Terry Transformation is an important contribution beyond what allows you to prove about your algorithm. Prior works have used the Probability Integral Transform to normalize rewards, both theoretically and empirically, with success [1]. However, this is framed (in the abstract especially) as a core contribution of this work. I would like to see more commentary on the importance of this framework for future research.
* Additionally, it may be worth addressing [1] because they propose a randomized stopping method, which seems (slightly) relevant to your discussion on the limitations of stopping at a fixed N.


Small Notation Comments:
* Line 128: "with N fixed in advance. Moreover, in practice, N is typically fixed" is repetitive
* In the Algorithm, line 8, should the $\mu$s be $\hat{\mu}$s?

[1] Khalaf, H., Verdun, C. M., Oesterling, A., Lakkaraju, H., & Calmon, F. D. P. (2025). Inference-Time Reward Hacking in Large Language Models. NeurIPS 2025

### Questions
* Can you elaborate on 1) if the Bradley-Terry Transformation is uniquely necessary to your results (and conversely, whether the PIT could be used to get the same insights) and 2) the tradeoffs between the two? One benefit I see of this transformation is that you don't need to estimate a CDF.
* Can you also clarify the choice of $\mathcal{F}$? What happens if the true distribution of rewards is vastly different than $\mathcal{F}$? Is there some sort of approximation error or tradeoff to choosing a specific (e.g., Exponential) distribution? All of your results begin with assuming a distributional family and then demonstrate that the anytime UCB policy has a low optimality gap on Weitzman's optimal policy. Can you say anything about using the wrong choice of distributional family in your Algorithm?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce an inference‑time optimization framework grounded in the classical Pandora’s Box optimal‑stopping problem, treating each generation as opening a costly “box” with a random reward, and show that an adaptive stopping strategy can match the performance of non‑adaptive Best‑of‑N sampling while using roughly 15–35% fewer generations on average.

### Strengths
The paper is original in recasting prompt‑level sampling as a Pandora’s Box optimal‑stopping problem, importing fair‑cap thresholds and Weitzman‑style reasoning to yield an adaptive stopping rule that subsumes fixed-N heuristics and turns test‑time compute into a principled decision problem. It then formalizes the fair‑cap objective, an anytime‑valid UCB construction for the unknown threshold, a general additive‑gap template (Theorem 5) with an explicit exponential‑tail instantiation (Theorem 6, Corollary 7), and a practical algorithm with closed‑form components and small overhead (Algorithm 1). The empirical significance is demonstrated empirically across ~1,600 profiles (100 prompts × 2 datasets × 4 LLMs × 2 reward models): the adaptive policy matches non‑adaptive Best‑of‑(N) while saving roughly 15–35% of generations on average and winning under matched budgets, indicating a relevant method for practitioners seeking an inference‑time alternative to best-of-N.

### Weaknesses
**Corollary 7 and "vanishing regret" need reconciliation**

Corollary 7 upper-bounds the sub-optimality by $O_\delta(1/\lambda)$, i.e., a distribution-dependent constant, not a term that vanishes with samples $n$ or confidence $\delta$. Please clarify whether any "vanishing" behavior is intended (e.g., in a different asymptotic or under additional assumptions), and align the abstract/claims ("guarantees vanishing regret relative to Weitzman's optimal policy") with the explicit bound you prove.

**Heavy reliance on modeling assumptions; robustness is unclear**

Guarantees assume (i) i.i.d. draws per prompt, (ii) known per-sample cost $c$, and (iii) a well-specified distribution family $\mathcal{F}$; the worked-out theory and the practical estimator both hinge on a (shifted) exponential tail fit. Could you provide any robustness results (or diagnostics) under tail mis-specification, e.g., when the right tail is heavier/lighter than exponential or multi-modal? Even a lemma or experiment quantifying degradation under controlled mis-fit would help.

**Discussion/conclusion are thin**

I understand the size limitation but the discussion section is not great. The paper doesn't have a proper conclusion and the discussion section reads as brief future-work notes; the paper lacks a proper Conclusion that synthesizes contributions, caveats, and takeaways for practitioners. Adding a short, explicit conclusion would improve readability.

**Related-work coverage is incomplete (see list below)**

Important recent work on inference-time alignment and adaptive compute is missing. Please add and discuss the listed papers; position your method theoretically and empirically relative to them.

- **Adaptive inference-time compute strategies:** (e.g., arXiv:2410.02725, arXiv:2503.01422, arXiv:2412.15287). Please compare assumptions, decision rules, and metrics. If possible, add a small head-to-head on a shared setup.

- **Inference-time alignment variants of Best-of-N:** soft-Best-of-N, Best-of-Poisson (e.g., arXiv:2506.19248, arXiv:2505.03156, arXiv:2507.05913). Ask: Where does your adaptive stopping dominate these stochastic Best-of-N variants at equal compute? Could your UCB fair-cap be combined with soft-aggregation?

**[L127]** "with N fixed in advance. Moreover, in practice, N is typically fixed."

Please rephrase to avoid repetition.

### Questions
**[L106]** "We note, however, that Weitzman's algorithm applies to multiple box types."

Could you please spell out the multiple-box setting (heterogeneous boxes) explicitly: assume different reward distributions ($D_1,\dots,D_k$) and opening costs ($c_1,\dots,c_k$)? Are draws independent across boxes?

**[L153]** "Weitzman's celebrated algorithm provides the optimal stopping strategy using fair-cap values when distributions are known."

Please add 2-3 sentences summarizing Weitzman's policy and its assumptions (known $D_i$ and $c_i$, independence, objective). Include a citation to Weitzman (1978) already in your references, plus a general optimal-stopping reference.

**[L173]** "no hope for designing a single, minimax optimal stopping policy S whose additive sub-optimality gap is uniformly bounded across all distributions."

Please define minimax precisely: is this $\inf_S \sup_{D} \mathbb{E}_D[R_W-R_S]$? If so, say so, and clarify that the impossibility is distribution-free over all $D$, motivating restriction to a family $\mathcal{F}$.

**[L174]** "we will assume that we have a known distribution family $\mathcal{F}$ such that the unknown $D\in\mathcal{F}$."

Please list the concrete assumptions on $\mathcal{F}$: identifiability, i.i.d. draws, existence/monotonicity of the fair-cap map $\tau(D,c)$, and that $\mathcal{F}$ admits an anytime-valid confidence sequence for $\tau$ (Def. 4). If you assume parametric (e.g., exponential tail), say so here, not only in §3.2/Alg. 1.

**[L198]** "the confidence parameter is a hyperparameter that influences the exploration-exploitation balance."

Could you please be explicit here? Something like: smaller $\delta$ ⇒ wider UCB on $\tau$ ⇒ more samples (conservative stopping). Is that what you meant? A short sentence quantifying how $\sigma_{\delta,\tau}(n)$ scales with $\delta$ (via $r_\delta(n)$ in Thm. 6) would also be nice.

**[L200–205]** "the fair-cap value often admits a simple monotonic dependence on the distribution's parameters…constructing a confidence sequence for $\tau$ can be reduced to…(1) CB for parameters…(2) propagate through the monotonic mapping."

Please clarify which parameters and the direction of monotonicity. Does monotonicity hold for every parameter or only for a specific scalar (e.g., mean/rate in the exponential)? How about the multi-parametric case?

**[L205 and L243]** The "confidence bounds for parameters → monotone map → confidence bounds for $\tau$" idea appears twice (around L205 and again near the statement/proof of Thm. 5).

Please keep it once and forward-reference the other to reduce redundancy.

**[L229]** You use "family" and later "Exponential distribution family."

Add a remark that *"exponential" here means the Exponential distribution (with rate/scale), not the general exponential family of distributions*. A footnote with a link to a standard reference should be fine.

**[L280]** "practitioners may adjust this parameter based on quality requirements."

What is your take on that? What is your opinion on how this is usually done? Please offer guidance.

**[L300]** "This transformation maps rewards into $[0,1]$."

Can $\mathrm{AR}_\kappa$ ever be exactly 0? Only in the limit $v\to-\infty$; for finite $v$ it's $>0$. Also, please explain the factor 2: it makes $\mathrm{AR}=1$ since the Bradley–Terry term equals 1/2 at $v=\kappa$.

**[L302]** "acceptance rate approximates the probability that an end-user accepts the response"

Confirm this refers to the particular sample with reward $v$. A small rephrase like "a response with reward $v$" would remove ambiguity.

**[L324]** "we exponentiate the rewards and fit a shifted exponential to the right-tail (above median)."

How many tail samples do you require before fitting? Please specify the minimum $t$ (Alg. 1), report sensitivity to $t$, and add a sanity check like a QQ-plot. Could you specify a rule-of-thumb to help practitioners?

**[L362–363]** "Streaming updates eliminate redundant computation"

Could be clarified with one sentence on what you cache.

**[L364]** "$\alpha$-percentile … computed in $O(1)$ from an analytical formula."

Please add the formula for the percentile.

**[L366]** "Riemann sum with $\sim 5000$ intervals."

Please justify accuracy (e.g., relative error $<1\%$ across all runs) and note the integrand's regularity. Did you try an adaptive quadrature (e.g., Simpson/Gauss–Legendre)?

**[L390]** "This formulation is useful in settings where quality requirements are clear but utilities are hard to quantify, for instance, when "good enough" responses are well-defined but the value of marginal improvements is ambiguous."

I find this hard to parse. Could you give an example?

**[L401]** "we use … FsfairX-LLaMA3-RM-v0.1 and RM-Mistral-7B."

Why these two reward models? Is there any reason for that?

**[L405]** "We always fix the max utility $B=1$."

Add one line that explains it. Something like: "by linearity, scaling utilities by $s>0$ and costs by $s$ leaves the threshold unchanged (since $\mathbb{E}[(U-\tau)_+]=c$ scales on both sides), so $B$ can be set to 1 w.l.o.g."

**Secretary problem mention**

Briefly mention the secretary problem as a classical optimal-stopping foil (unknown distributions, no recall), to position Pandora/Weitzman relative to other classics.

**Add general optimal-stopping references**

Like "Chow, Robbins, Siegmund (1971) *Great Expectations: The Theory of Optimal Stopping*" and maybe a modern set of lecture notes such as https://www.math.ucla.edu/~tom/Stopping/Contents.html. These will help readers new to the area.

**Fair-cap intuition**

Consider adding a one-line intuition to the fair-cap definition, something like: "$\tau$ is the reward level at which I'm indifferent between stopping now and paying cost $c$ for one more draw; i.e., the expected improvement from one more sample equals $c$."

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
* Problem: Large language model (LLM) generation requires balancing output quality against inference-time cost. Current strategies like Best-of-N sampling fix the number of generations in advance, leading to inefficiency.
* Solution: The paper reframes LLM inference as a Pandora’s Box optimal stopping problem. It introduces (1) a UCB-style Pandora’s Box algorithm that learns when to stop sampling without knowing the underlying reward distribution, and (2) a Bradley–Terry–based reward normalization to address cross-prompt scaling.
* Evaluation: Experiments on AlpacaFarm and HH-RLHF datasets across four LLMs and two reward models demonstrate that the adaptive method achieves comparable reward to Best-of-N sampling while using 15–35% fewer generations, supported by theoretical regret bounds relative to Weitzman’s optimal policy.

### Strengths
1. Novel framing: Establishes a principled connection between inference-time optimization and the Pandora’s Box optimal stopping problem which is an original theoretical perspective.
2. Theoretical grounding: Provides regret bounds relative to Weitzman’s optimal stopping policy and extends these results to exponential families.
3. Empirical consistency: Results are coherent across models, showing that adaptive inference can yield similar reward quality with reduced sampling.

### Weaknesses
1. A single basic baseline: The paper only compares against basic non-adaptive Best-of-N, omitting adaptive or GenRM/self-evaluation Best-of-N methods (e.g., speculative rejection, self-consistency, GenRM, self-evaluation, and many other approaches) that have surfaced in the past year or two. This limits the empirical significance of the reported 15–35% improvement which is relatively modest compared to what other works claim.
2. Lack of real compute and latency measurement: Efficiency is measured purely in terms of sample count, ignoring the additional cost of reward model inference or the sequential sampling latency. Parallel Best-of-N can often amortize cost, whereas the proposed adaptive approach incurs sequential latency. FLOPs or wall-clock time should be measured for fairness.

### Questions
1. Could the authors compare against recent adaptive test-time compute allocation methods (e.g., Wang et al. 2025, Sun et al. 2024, Manvi et al. 2024, etc.)?
2. Does the adaptive stopping decision depend on sequential generation, or could rewards be computed in parallel after sampling a batch?
3. How sensitive is the algorithm to the chosen percentile α (e.g., 0.99)? Would smaller values degrade performance significantly?

### Soundness
2

### Presentation
3

### Contribution
2
