# A Bias–Variance Tradeoff Perspective for Improving Test-Time Scaling

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Parallel test-time scaling (PTTS) improves the reasoning performance of large language models (LLMs) by aggregating multiple candidate solutions at inference time. However, existing methods remains largely heuristic, lacking a principled framework that explains their behavior, clarifies their limitations, and guides systematic improvement. To bridge this notable gap, we introduce the first general framework for PTTS through a unifying probabilistic inference formulation, seamlessly encompassing prior disparate methods as special cases. This framework enables a novel bias-variance tradeoff perspective to reveal the intrinsic limitations of existing methods and serves as a principled foundation for developing new ones. Specifically, our framework reveals that existing verifier-based methods act as *high-variance* importance sampling (IS) estimators, yielding marginal gains under small scaling budgets, whereas generator-based methods act as *biased* variational inference (VI) estimators, yielding suppressed scalability even under large budgets. To formally characterize the tradeoff, we derive the first theoretical bias–variance formulation for PTTS, revealing that the relative variance upper bound is jointly governed by the generator and the verifier via their respective optimality gaps. To mitigate this tradeoff, we build upon our general framework and derive a theory-driven PTTS method named TSMC-TTS. Specifically, it instantiates our framework with twisted sequential Monte Carlo (TSMC) and performs EM-like optimization of the generator and verifier guided by the theoretical variance bound, thus achieving monotonic variance reduction without
compromising unbiasedness. Furthermore, we introduce a new self-evolving TSMC
mechanism, effectively alleviating both the reward sparsity and the computational overhead
issues inherent in vanilla TSMC. Rigorous theoretical analysis and comprehensive
empirical results demonstrate the efficacy of our proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to view test-time scaling (TTS) of language models through the lens of bias-variance tradeoff, by viewing "verifier-based methods" (i.e. TTS using outcome reward models (ORMs) or process reward models (PRMs) to score and choose from multiple proposed outputs, without training) as unbiased, high-variance; and "generator-based methods" (i.e. finetuning a model in some way to generate better responses, usually guided by a reward function/model) as biased, low-variance. Then, the authors propose a new method for TTS they call TSMC-TTS (twisted sequential Monte Carlo TTS). They provide experiments on GSM8K and MATH500.

As outlined in "Weaknesses" below, I believe the papers suffers from multiple weaknesses: Large parts of the theory are either trivial or not proven, notation and assumptions are unclear, and the experiments should include TSMC as a baseline and are somewhat limited (only two datasets). Hence, **I recommend rejection of the paper in its current form, and encourage the authors to work on improving on these points.**

### Strengths
The bias-variance viewpoint of training-free TTS (viewed as importance sampling) as low bias (unbiased) and high variance, vs finetuning methods as biased, low variance is interesting (yet not new, as the authors themselves say this is "well established", line 191).
The idea of combining TSMC with finetuning could be interesting, if it was better supported and evaluated.
The experimental results seem to show that TSMC-TTS outperforms some baselines, but the results are limited (see "Weaknesses").

### Weaknesses
In my view, the paper suffers from many extensive weaknesses.

### Notation
The notation is confusing and seems at least partially wrong. For example, the authors define the reward function as $R(x_{1:T})$; however, reward functions also take as input the prompt $x_0$, which is omitted in this notation; same for $\phi$. It is not clear what is meant with notations like $q(x_{1:T})$ (is the prompt fixed here?). The authors switch between $q$, $p_0$, and $p$, without making it clear how these relate and whether these are the same distributions (I assume $p$ is supposed to be the base distribution and $q$ then one we sample from, possibly finetuned from $p$, but this is never made explicit). E.g., in Equation (1) they use $p$, but when referring to Equation (1) in Proposition 4.1, they replace it by $q$. Various other confusions in the notation arise, e.g.: Is $R$ the same as $W$? How do they relate? Is $R$ supposed to be an unknown ground truth reward? Much of the paper -- e.g. Equation (4) -- only seems to make sense if $R=W$ or they are at least assumed to be close, but this is never made specific. The authors omit clear definitions of various concepts used throughout the paper, for example "verifier", which is used e.g. in Proposition 4.1 without a clear definition; from the Proposition, it seems like "verifier" refers to a PRM, but this is never made explicit. If this is the case, then similar to $R$ and $\phi$, the verifier should also depend on $x_0$.

### Section 4
All claims in Section 4 seems largely trivial or unsupported. Neither Proposition 4.1 nor 4.2 have any real proofs, only "extended versions" in the appendix, that don't contain proofs of any claims either.

In particular, the extended version of Proposition 4.1, i.e. Proposition C.1, seems to rely on an assumption that $R$ and $W$ are close in some sense without making it explicit. Equation (23)/(5) is trivial -- it states two terms that only differ by a constant normalizing factor have the same maximizer. The only interesting part in Proposition 4.1 in my opinion is the "$\approx$" identity, which relates how well the true answer frequency $f_a$ can be approximated by $\hat{f}^{IS}_a$. However, the authors simply state this is approximately equal, without giving any justification for why, or any quantification of the actual approximation (e.g., what would be nice to see is an asymptotic result that shows that as $N$ increases, the approximation becomes arbitrarily good).

Proposition 4.2 has no proof either. Furthermore, much of the notation and claims are unclear to me: How is $q_{VI}$ defined? I don't see how this has anything to do with VI. In VI, a distance/divergence is minimized over a certain class of distributions (e.g. factorized distributions). However, it seems here the authors simply minimize $KL(q ||\sigma)$ without any restrictions, which is trivially minimized by $\sigma$. In particular, this is not VI in the usual sense.
The equivalence $q_{RFT}\Leftrightarrow q_{VI}$ which the authors prove in Appendix D.6 is the well-known result that the optimizer of the KL-regularized objective is the tilted distribution $\sigma\propto p_0 e^{R/\beta}$, which is not novel and does not require a proof. Consequently, Equation (8) (presumably the main proposed contribution of Proposition 4.2) also trivially follows from this relationship. Furthermore, Proposition 4.2 defines $q_{RFT}$ as "the generator trained via RFT [...] by minimizing [the KL regularized objective". However, consequently the authors treat $q_{RFT}$ as the *minimizer* of this objective, which is clearly not the same as the policy one would get from fine-tuning in practice.

To summarize, I do not see any novelty in either Proposition 4.1 or 4.2.

### Experiments
The proposed method combines TSMC with a fine-tuning objective, where the policy and a verifier (parametrized as a head on top of the base policy) are fine-tuned, before running TSMC at inference using the fine-tuned model and verifier (if I understand the algorithm correctly). TSMC is a fairly new concept I was not familiar with, but I had to look it up (https://arxiv.org/abs/2404.17546) since it's not well-explained in the paper. In particular, the authors do not compare vs. vanilla TSMC as a baseline, which would be the most obvious and straightforward baseline to compare against. As a consequence, it remains unclear if the proposed method achieves any improvement over TSMC.

Furthermore, the experimental results are somewhat limited (only two datasets), and do not contain any error bars. The paper would benefit from more extensive evaluation.

### Questions
To the best of my knowledge, I have carefully read and evaluated the paper's weaknesses as stated above; however, if the authors believe I gravely misunderstood certain parts of the paper, please point them out to me.
Some questions that I have already stated in the previous section include:

- could you clarify the notation? E.g. why does $R$ not depend on the prompt (same for $\phi$ etc); what's the relationship between $q$, $p_0$, and $p$?
- how do $R$ and $W$ relate?
- could you clarify the contributions of Propositions 4.1 and 4.2? In particular, Equation (5) seems trivial, and the relationship $q_{RFT}\Leftrightarrow q_{VI}$, in your notation, is already well-known, which means Equation (8) does not provide any novel contribution either as far as I can see.
- in what way do you believe VI plays a role? To me it seems you are simply minimizing the objective $KL(q||\sigma)$ without restrictions which is trivially minimized by $\sigma$.
- how does your proposed method compare to TSMC, both conceptually (inference time algorithm) and empirically (experimental accuracy)?

### Soundness
1

### Presentation
2

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
This paper proposes an improvement to Test-time Scaling (TTS) from a bias-variance perspective. The authors first categorize existing TTS improvement methods into two groups: verifier-based approaches, which aggregate answers using external verifiers, and generator-based approaches, which enhance the generation model to produce better candidates.

The authors then introduce a probabilistic inference framework that unifies these two TTS-augmentation paradigms from the perspective of bias and variance. Through theoretical derivation, they show that the verifier-based method corresponds to importance sampling, which has low bias but high variance, while the generator-based method corresponds to variational inference, which has high bias but low variance.

Building on this perspective, the authors propose TSMC-TTS, a method that employs twisted sequential Monte Carlo as the inference algorithm to reduce variance while maintaining unbiasedness. Furthermore, they use a theoretically derived upper bound to jointly optimize the generator and the verifier, thereby reducing variance without sacrificing unbiasedness.

The experimental results on gsm8k and MATH show that the proposed method outperform the verifier-based method and the generator-based method.

### Strengths
1. The bias-variance perspective proposed by the authors is highly novel, offering a new way to understand and analyze inference-time scaling methods.
2. The authors conducted extensive theoretical derivations to support their conclusions, such as establishing the correspondence between their framework and methods like importance sampling and variational inference.
3. The authors designed a targeted optimization method based on their analysis and supported it with rigorous theoretical justification.

### Weaknesses
1. The excessive use of derivations and mathematical notations makes the paper difficult to read, especially since even simple concepts (like majority voting) and methods are expressed with an overabundance of symbols and formulas.
2. The experimental evaluation is insufficient, as experiments were only conducted on Llemma-7B and DeepSeek-7B, neither of which are state-of-the-art models. Moreover, the evaluation was limited to just two datasets, GSM8K and MATH, lacking validation across a broader range of domains and benchmarks.

### Questions
Is the experimental comparison fair? TSMC-TTS is a method that combines both verifier-based and generator-based approaches, yet it is only compared against individual methods. It remains unclear what the results would be if a naïve combination of the verifier-based and generator-based approaches were implemented for comparison.

Why is the Generator-based approach categorized as a method for improving TTS? For example, methods like RFT are primarily training-time enhancement techniques rather than inference-time ones.

### Soundness
3

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
3

### Summary
This paper introduces a probabilistic inference framework for Parallel Test-Time Scaling (PTTS), casting existing methods into a bias-variance tradeoff perspective. It frames verifier-based methods as high-variance Importance Sampling (IS) estimators and generator-based methods as biased Variational Inference (VI) estimators. To resolve this tradeoff, the authors propose TSMC-TTS, a novel algorithm based on Twisted Sequential Monte Carlo designed to reduce variance while aiming to preserve unbiasedness. The method is validated empirically, achieving state-of-the-art results on mathematical reasoning benchmarks.

### Strengths
- Elegant Theoretical Framework: The paper's primary strength is its novel and insightful unification of verifier-based (IS) and generator-based (VI) methods under the bias-variance tradeoff. This provides a much-needed theoretical lens for a field largely driven by heuristics.

- Principled and Effective Algorithm: The proposed TSMC-TTS is a principled instantiation of the framework. The joint generator-verifier optimization, guided by a variance bound, is a sophisticated and well-motivated approach that directly addresses the identified limitations of prior work.

- Clever Handling of Practical Challenges: The adoption of a self-evolving target distribution σ(x) ∝ q(x)R(x) is a significant strength. It effectively mitigates the reward sparsity and high computational costs associated with relying on a fixed base model p0.

- Strong Empirical Validation: The experiments are thorough and the results are compelling. The method achieves state-of-the-art performance, and the scaling curves (Fig 1, 2) provide strong empirical evidence for the theoretical claims regarding the limitations of IS- and VI-based approaches.

### Weaknesses
- Fragile "Unbiasedness" Claim: The paper's central claim of "preserving unbiasedness" is misleading as it relies on the unrealistic assumption that the learned verifier is perfect (ψ = R). In practice, any learned verifier has its own estimation error, which introduces a verifier-induced bias. The paper's analysis completely overlooks this crucial factor and lacks a discussion of how this bias propagates and potentially dominates the variance term, thereby weakening the main argument for adopting a low-variance estimator like TSMC.

- Theory-Practice Gap in Reward Signal: The theoretical analysis of generator-based methods (VI) assumes access to a true reward signal R(x). However, all practical RFT/RLHF implementations rely on a learned, and therefore biased, reward model RM(x). This disconnect means the theory analyzes an ideal scenario, while the experiments operate in a practical (biased) one. Consequently, the paper conflates two different sources of error: the intrinsic mode-seeking bias of VI and the external bias from the imperfect reward model. The framework, in its current form, cannot disentangle these two effects.

### Questions
1. The claim of unbiasedness for TSMC-TTS hinges on ψ = R. Since this is never true in practice, how does the verifier error, say ||ψ - R||, propagate through the TSMC estimator? Can you provide any analysis on how this verifier-induced bias compares in magnitude to the variance reduction gained from using TSMC?

2. Your VI analysis is based on a true reward R(x), but practical RFT uses a learned reward model RM(x). What does the theoretical equivalence between RFT and VI hold when considering a biased RM(x)? Is it possible that the observed "VI bias" discussed in the paper is largely dominated by this external "reward model bias"?

3. For practical applications, the trade-off between performance and computational cost is critical. Could you provide a more detailed cost-benefit analysis (e.g., final accuracy vs. total FLOPs or wall-clock time) comparing TSMC-TTS against simpler but strong baselines, such as a well-fine-tuned generator using only majority voting?

### Soundness
3

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
4

### Summary
This paper addresses the problem of improving Parallel Test-time Scaling (PTTS) for mathematical reasoning in Large Language Models (LLMs). The authors propose a novel and general probabilistic inference framework that recasts existing PTTS methods through the lens of the bias-variance tradeoff. Within this framework, the paper unifies two common approaches: Verifier-based methods are identified as Importance Sampling (IS) estimators, which are unbiased but suffer from high variance. Generator-based methods (using reinforcement fine-tuning) are identified as Variational Inference (VI) estimators, which achieve low variance but introduce systematic bias. To mitigate this tradeoff and achieve an estimator that is both unbiased and has low variance, the paper proposes TSMC-TTS. This method uses Twisted Sequential Monte Carlo (TSMC) as the inference algorithm. The key technical novelty is a principled joint optimization of the generator and the verifier (i.e., the twist function)5. This optimization is guided by a theoretically derived upper bound on the relative variance and employs an EM-like algorithm to ensure monotonic variance reduction. This approach also uses a "self-evolving" target distribution, which iteratively refines the generator. Experiments on GSM8K and MATH500 show that TSMC-TTS outperforms the verifier-based and generator-based baselines included in the main tables.

### Strengths
- The paper is well-written, and the core idea is presented clearly.

- The primary strength is the novel conceptual framework that unifies PTTS approaches under the bias-variance tradeoff. Casting verifier-based methods as IS and generator-based methods as VI is an elegant and insightful contribution. This perspective provides a principled and systematic way to understand the intrinsic limitations (high variance vs. systematic bias) of these popular methods.

- The proposed TSMC-TTS method is theoretically well-grounded. The derivation of the relative variance upper bound (Theorem 5.2) and the formulation of the EM-like joint optimization algorithm (Property 5.5) provide a solid theoretical foundation for the proposed approach.

- Empirically, the method achieves strong results, outperforming the baseline methods (Base+MV, Base+ORM, RFT+MV) presented in the main comparison tables.

### Weaknesses
- The main weakness of this paper lies in its comparison to very closely related prior work, **specifically Feng et al. (2025)**. The paper's core technical contribution over Feng et al. is the *joint optimization* of the generator and verifier (using a "self-evolving" target distribution), whereas Feng et al. used a *fixed* generator and a fixed target distribution.

- However, the empirical gains from this new joint optimization appear to be **marginal**. For example, on DeepSeek-7B/MATH500, TSMC-TTS achieves 61.8%, while the fixed-generator TSMC from Feng et al. (2025) reported 60.8%. On Llemma-7B/GSM8K, the gap is similarly small at 82.2% vs. 80.4%. This minimal improvement (0.8-1.8%) calls into question the practical value of the paper's main technical novelty, especially given the significant additional training compute required to jointly optimize the generator.

- For transparency, the results from Feng et al. (2025) should be included directly in the main results (Table 2), as this is the most relevant baseline for the TSMC-based approach. Relegating this comparison to a discussion in the appendix makes the paper's contribution seem larger than it is.

- Additionally, the proposed EM-like optimization (Alg. 1) relies on estimating gradients from samples. The stability of this "online updating" process is a potential concern. If the batch size $B$ is small, the gradient estimates (Eq. 19, 20) could have high variance, potentially leading to unstable training.

### Questions
Related to the stability concern in the weakness section: How sensitive is the joint optimization algorithm (Alg. 1) to the batch size $B$ used for gradient estimation? Given that the algorithm's design already takes steps to avoid "noisy gradients", could a small $B$ lead to high-variance loss estimates that make the training unstable or prevent the joint optimization from converging effectively?

### Soundness
3

### Presentation
3

### Contribution
2
