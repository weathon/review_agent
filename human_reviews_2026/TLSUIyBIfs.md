# Quantitative Bounds for Length Generalization in Transformers

- Decision: Accept (Oral)
- Scores: 6, 8, 6, 8

## Abstract
We study the problem of length generalization (LG) in transformers: the ability of a model trained on shorter sequences to maintain performance when evaluated on much longer, previously unseen inputs. Prior work by Huang et al. (2024) established that transformers eventually achieve length generalization once the training sequence length exceeds some finite threshold, but left open the question of how large it must be. In this work, we provide the first quantitative bounds on the required training length for length generalization to occur.
Motivated by previous empirical and theoretical work, we analyze LG in several distinct problem settings: $\ell_\infty$ error control vs. average error control over an input distribution, infinite-precision softmax attention vs. finite-precision attention (which reduces to an argmax) in the transformer, as well as for one- or two-layer transformers. In all scenarios, we prove that LG occurs when the internal behavior of the transformer on longer sequences can be ``simulated'' by its behavior on shorter sequences seen during training. Our bounds give qualitative estimates for the required length of training data required for a transformer to generalize, and we verify these insights empirically. These results sharpen our theoretical understanding of the mechanisms underlying extrapolation in transformers, and formalize the intuition that richer training data is required for generalization on more complex tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper provides quantitative bounds on the required training sequence length for transformer models to achieve length generalization by proving that this occurs when the model's behavior on longer sequences can be "simulated" by its behavior on the shorter sequences it was trained on.

### Strengths
1. The paper tackles an interesting and important aspect of length generalization, namely the minimum training length to enable it, given a task.
2. The paper takes care of the finite-precision arithmetic that is used in practice, something often overlooked in theoretical work.
3. Overall the paper is generally well-written and explained. Intuition and remarks are provided after most mathematical definitions and results which makes it easy to follow the reasoning and the motivation.

### Weaknesses
1. I am missing a comprehensive discussion on the limitations of the paper’s analysis, especially on all the technical assumptions and simplifications on both the class of tasks that is considered and on the architectural modifications. 
2. The introduction of the $\phi_{l,h}(j,i)$ terms in the definition of the limit transformer appears to be a significant difference from standard attention. And contrary to what is said on line 133, it seems that they can only capture dependencies between the positions, not the content. At any rate, the paper would benefit from explaining why allowing for more expressive dependencies than the bilinear interaction of classic attention is negligible for the properties of the hypothesis class.
3. The definition of translation invariance is also a bit strange as nothing prevents positional information to also flow through the $\mathbf y_j \mathbf K^\top \mathbf Q \mathbf y_i$ term.  Perhaps it would be neater if you separate the input embeddings and the positional encodings as two separate components rather than adding them together.
4. The experiments and the corresponding plots can be explained more clearly and in more detail. The subplots are missing captions which could clarify what is being plotted and the first subplot is missing indication on the legend as to what the different lines are showing.

### Questions
Typos:
- Line 133: “denote allow”.
- Line 441: “value the test loss” -> “value of the test loss”

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors provide the first non-asymptotic quantitative theory of length generalization (LG) in transformers, establishing explicit bounds on the minimum training sequence length $N$ required for a model trained on short sequences to generalize to arbitrarily long ones. In the finite-precision (hardmax) regime, Theorem 4.1 shows that for one-layer $\tau$-local, $\Delta$-periodic limit transformers, if two models agree on inputs up to length $N = O(\max{2^{p/\gamma}, L^{2}\Delta^{7}|\Sigma|^{6}\tau^{2}\varepsilon^{-2}})$, then they agree within $O(\varepsilon)$ on all longer sequences, with $L$ being a Lipshitz style upper bound, $\gamma$ the attention logit margin, and $|\Sigma|$ the vocabulary size. Theorem 4.2 extends this to average-case error control, showing that under regular input distributions, $|f-g|{T,P} = O(\varepsilon^{1/2})$ for all $T \ge N$, with $N$ scaling polynomially in $L, \Delta, \tau, |\Sigma|$. In the infinite-precision (softmax) setting, Theorem 5.2 introduces a complexity measure $C(f)$ containing exp and poly factors in the norms, and and provides the equivalent characterization of the length $N$ for LG in two-layer transformers, implying exponential dependence on the inverse positional margin, which although will be huge in general. All proofs rely on a unified simulation argument constructing a short sequence $z$ that preserves the sufficient statistics of a long sequence $x$ so that $f(x)\approx f(z)$. Empirical studies on synthetic tasks (SimpleTask, ModPTask, and in-context $k$-gram) validate these scalings: test loss plateaus at finite lengths, decreasing with training length. Collectively, the results establish that the capacity for LG in terms of the model parameters and the usual error tolerance, providing a rigorous quantitative framework linking transformer parameters to their extrapolation behavior.

Note the questions and concerns I have raised below.

### Strengths
1. The Dirichlet assumption in Theorem 4.2 is a good choice, as classical NLP literature has worked with it for decades in many practical applications. This makes the contribution of this work more practical

2. The authors' efforts in studying both finite and infinite precision and also providing bounds for L_{inf} and average case for finite setting is a good contribution to this line of research.

3. I found Lemma 5.3 to be a useful addition. While I did not read the proof end to end, this is a good supporting lemma of independent interest - assuming that it does not exist in the literature in a similar form yet.

### Weaknesses
1. I have a concern about the value of N - $2^{\frac{p}{\min\{\gamma(f), \gamma(g)\}}}$ . While I understand the requirements for the proofs, I am curious if the authors have some thoughts on adapting equation 2 on page 13 – for say cases where the logit differences can be binned and perhaps some covering number arguments can be used? Please note this is just a rough guess and I have not put proper thought behind this statement, and I am just curious about the ideas from the authors’ end. 

2. In line 1024 of page 19, while the triviality of - |Σ|^3 + τ ≤ |Σ| ^ X τ , why was it even necessary to loosen the bound by such an extent. Furthermore, for a good number of practical datasets where |Σ| = O(τ), this unnecessarily loosens the bound. Again, curious about this situation.

### Questions
1. What exact upper bound for the Lipshitz constant are the authors using to support the argument - “This is unavoidable, as the Lipschitz constant of the first layer softmax scales exponentially” - line 366 on page 7. To my understanding, there have been various proposals for the upper bound such as [1-3], some of which not scaling exponentially. 

2. Can the authors clarify the argument - “One observes that f ∗ is expressible by a one-layer limit transformer with no positional embeddings and L = Θ(ω).” from line 418? More specifically, how was this bound achieved for L and similarly in line 424 for the ModPTask? 

3. The left subfigure in Figure 1 on page 8 is interesting – the smaller values of ω exhibit larger test loss across all lengths. I am curious to understand the authors’ discussion on this, because either using the bounds from this paper, or using standard generalization bounds, for smaller ω, a smaller test loss across all lengths is expected. Looking forward to this response in case I made a mistake in my understanding. 

 

[1] Kim, H., Papamakarios, G., & Mnih, A. (2021). The Lipschitz Constant of Self-Attention. In Proc. ICML 2021, PMLR vol. 139 

[2] Castin, V., Ablin, P., & Peyré, G. (2024). How Smooth Is Attention? (ICML 2024). Proceedings of Machine Learning Research vol. 235 

[3] Wang, Y., Chauhan, J., Wang, W., & Hsieh, C.-J. (2023). Universality and Limitations of Prompt Tuning. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper builds upon the theoretical constructions (Limit Transformer) and the conclusion in  Huang et al. (2024)  - that Limit Transformers can achieve Length Generalization if trained under sufficient length for at least (if not at most) class of problems that can be expressed by a C-RASP program. This paper attempts to go beyond that conclusion and also provide quantitative bounds for the required lengthts under different settings - such as infinite-precision vs finite-precision attention or different error controls, single-layer and 2-layer transformers. 

The key level idea used for the proof is to construct a simulation map from strings of arbitrary length to strings of bounded length.

### Strengths
* Generally well written and well connected with the literatur. 
* Provides theoretical proof to support quantitative bounds for required training length
* Provides some empirical support that error rate at which test loss plateus decreases with increasing training length.

### Weaknesses
Not any significant blocker to acceptance to my awareness. One could question the practical impact or limited scope, but it is targeted as a theoretical paper under learning theory - and seems to sufficiently fulfill its targeted goal. There are precedents of papers with similar scope getting accepted. 

Besides this, one limitation is that the study is mainly tied to (periodic) absolute positional encodings (as far as I understand) - but the authors keep studies of other positional schemes for future work - which is fair enough.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper focuses on the problem of length generalization (LG) of Transformers - i.e., the ability of a model to maintain its performance on unseen long sequences after being trained on short sequences - and for the first time gives quantitative bounds on the length of the training sequences required to achieve LG. The study covers a variety of scenarios, with the core logic that LG can be achieved when the internal behavior of long sequences can be “simulated” by short sequences in the training set. The paper clarifies the correlation between training length and model parameter paradigms, period Δ, locality T, etc. through theoretical derivation, and verifies the conclusions through synthetic tasks such as SimpleTask, ModPTask, etc., which provide theoretical references for scaling the length of training contexts of Large Language Models (LLMs).

### Strengths
- The length of the training sequence required for LG is quantified for the first time, addressing the key issue that previous studies have only demonstrated the “existence of a threshold” but have not clarified the “size of the threshold”.
- The conclusions are more broadly applicable by also considering a variety of variables such as the type of accuracy, the number of model layers, and the way the error is controlled. Covers many different types of theoretical scenarios and explores the quantitative issues regarding generalizability in a comprehensive and detailed manner.
- The boundaries are established by rigorous mathematical derivation and then verified by a synthetic task with high confidence in the conclusions. The theory is paid to the experiment, and the correctness of the theory is effectively verified through the results.

### Weaknesses
- Very interesting paper that gives a quantitative analysis of length generalizability, based on the theoretical framework of the “limit transformer”, and therefore lacks an analysis of the transformer scheme with relative positional coding, which is currently more used in various methods.
- The paper verifies the correctness of its theory on a small-scale transformer structure, and it is hoped that the paper will give further analysis on whether there are limitations in the analysis and experimentation of larger models.
- Are the complex model parameters on which quantitative analysis relies also greater in more complex models? Does this create bottlenecks in analysis for larger models? Hopefully the paper will give at least a qualitative judgment.
- Although I have the above doubts, I nevertheless think that the work in this paper is solid, informative, and importantly groundbreaking for the study of length generalizability, and I therefore hope that the authors will be able to give at least a qualitative judgment on the above doubts.

### Questions
see Weakness

### Soundness
3

### Presentation
3

### Contribution
2
