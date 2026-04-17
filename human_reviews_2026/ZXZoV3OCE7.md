# On the Complexity Theory of Masked Discrete Diffusion: From $\mathrm{poly}(1/\epsilon)$ to Nearly $\epsilon$-Free

- Decision: Reject
- Scores: 2, 2, 8, 6

## Abstract
We study *masked discrete diffusion*---a flexible paradigm for text generation in which 
tokens are progressively corrupted by special mask symbols before being denoised.
Although this approach has demonstrated strong empirical performance, its theoretical complexity in high-dimensional settings remains insufficiently understood. 
Existing analyses largely focus on *uniform* discrete diffusion, and more recent attempts addressing masked diffusion either (1) overlook widely used Euler samplers, (2) impose restrictive bounded-score assumptions, or (3) fail to showcase the advantages of masked discrete diffusion over its uniform counterpart.
To address this gap, we show that Euler samplers can achieve \(\epsilon\)-accuracy in total variation (TV) with $\tilde{O}(d^{2}\epsilon^{-3/2})$ discrete score evaluations, thereby providing the first rigorous analysis of typical Euler sampler in masked discrete diffusion.
We then propose a *Mask-Aware Truncated Uniformization* (MATU) approach that both removes bounded-score assumptions and preserves unbiased discrete score approximation. 
By exploiting the property that each token can be unmasked at most once, MATU attains a nearly $\epsilon$-free complexity of $O(d\ln d\cdot (1-\epsilon^2))$. 
This result surpasses existing uniformization 
methods under uniform discrete diffusion, eliminating the $\ln(1/\epsilon)$ factor and substantially speeding up convergence.
Our findings not only provide a rigorous theoretical foundation for masked discrete diffusion, showcasing its practical advantages over uniform diffusion for text generation, but also 
pave the way for future efforts to analyze diffusion-based language models developed under masking paradigm.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the sampling complexity of masked diffusion models. The authors prove complexity bounds to achieve $\epsilon$-error in total variation, for Euler and Mask-Aware Truncated Uniformization (MATU) samplers, where the latter is an algorithm introduced in this paper to exploit specfic properties of masked diffusion. The primary contribution is that the sampling complexity of MATU is (nearly) $\epsilon$-free, improving upon the $\ln(1/\epsilon)$ dependence in the case of discrete diffusion with uniform transition.

### Strengths
It gives a mathematical proof that masked diffusion models with their proposed sampler (MATU) actually enjoy an $\epsilon$-free sampling complexity. The theory explains the difference between uniform and masked diffusions in terms of sampling complexity. It also removed some assumptions from the previous work on bounding discrete diffusion complexities, thereby widening the applicability of the theory to more general situations.

### Weaknesses
While there are several weaknesses, the main point ([W1]) is that it does not seem to be worthwhile considering more than $d$ sampling steps in masked diffusion, since there are only $d$ transition steps in forward (and thus backward) processes. Details are as follows:
- [W1] Masked diffusion (with the same and independent forward transition in each dimension) is time-agnostic; namely, the distributions conditioned on the number of masks is independent of time [1, Proposition 3.2]. We can also analytically sample the transition time and position/dimension [1, Proposition 4.1], so we just need $d$ function calls, each unmasking one position at a time, with errors only coming from the estimation error. While [1] uses a denoising parameterization (directly parameterizing $p(x_0^i|x_t)$ at each position $i$), it is easily convertible from/to the score parameterization [2, Proposition 3]. **Therefore, having** ***>d*** **sampling steps is not very meaningful in practice for masked diffusion.** In other words, an $\epsilon$-free sampling complexity is already known in the case of masked diffusion.
- [W2] While the authors introduce $\tilde{q}_t$ in Eq. 9 as an initial distribution for masked diffusion, people do not use such "partially masked and partially uniform" distributions for generative purposes. I think it comes from the time-homogeneous assumption of the forward diffusion noising, which is never used in the actual training of masked diffusion.
- [W3] Some citations should be clarified.
    - [W3-1] In Table 1, the rate of Ren et al. [3] is different from the original rate in KL divergence. Similarly, it is written that the "higher-order" method gives $O(\epsilon^{-1})$, which is different from the original paper, again in KL divergence. I assume that the authors implicitly use Pinsker's inequality to deduce those bounds, but it is not a fair comparison/citation, since the original work do not aim to give a tight bound in terms of TV distance.
    - [W3-2] In L46, the authors cite Nie et al. [4] to state "masked discrete diffusion has empirically outperformed uniform discrete diffusion" but the cited work does not mention uniform diffusion. I know masked diffusion models are more investigated, but there is also a uniform diffusion language model competitive with a masked model of similar scale [5].
- [W4] I feel that the order notation is used in an ambiguous way.
    - [W4-1] For example, in L53, the authors write "at least $O(\epsilon^{-1})$ steps" as if $\epsilon^{-1}$ would give a lower bound of the complexity. However, the $O$-notation only gives an upper bound and none of the cited work in L52 gives such a lower bound. The only lower bound I know in this context is the later half of [6, Theorem 1].
    - [W4-2] Whether the parameters $d$ or $K$ are regarded as constants (i.e., whether they are inside $O(\cdot)$) frequently changes throughout the manuscript. In addition, why does $K$ disappear only in the second term in L94, compared with Eq. 20?

Minor points:
- The definition of $\text{Ham}(y, y')$ should be $d-$ (current value).
- The value in Eq. 20 can be upper bounded by $2Kd+12Kd\ln d$, which is $\epsilon$-free, while the sampling algorithm can depend on $\epsilon$. Calling it "nearly" $\epsilon$-free is misleading.

References:
- [1] Zheng, K., Chen, Y., Mao, H., Liu, M. Y., Zhu, J., & Zhang, Q. (2025). Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. ICLR 2025.
- [2] Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Deligiannidis, G., & Doucet, A. (2022). A continuous time framework for discrete denoising models. NeurIPS 2022.
- [3] Ren, Y., Chen, H., Rotskoff, G. M., & Ying, L. (2025). How discrete and continuous diffusion meet: Comprehensive analysis of discrete diffusion models via a stochastic integral framework. ICLR 2025.
- [4] Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J. R., & Li, C. (2025). Large language diffusion models. arXiv:2502.09992.
- [5] Sahoo, S. S., Deschenaux, J., Gokaslan, A., Wang, G., Chiu, J., & Kuleshov, V. (2025). The diffusion duality. ICML 2025.
- [6] Hayakawa, S., Takida, Y., Imaizumi, M., Wakaki, H., & Mitsufuji, Y. (2025). Distillation of discrete diffusion through dimensional correlations. ICML 2025.

### Questions
- [Q1] "Girsanov theory" is only mentioned in Conclusion. Can the authors provide a brief explanation and motivation for avoiding it?
- [Q2] Related to [W3-1], am I correct in assuming that some of the bounds in Table 1 are given by Pinsker's inequality from KL-based bounds? In that case, can the TV-based bounds in this paper also be transformed to bounds in KL (since it seems that many of the analyses in the appendix are conducted through KL divergence).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a rigorous theoretical analysis of the computational complexity of masked discrete diffusion models for text generation. The authors make two primary contributions. First, they provide the first formal complexity analysis for the widely used Euler sampler in this context, showing that it requires $O(d^2\epsilon^{-3/2})$ score evaluations to achieve $\epsilon$-accuracy in total variation (TV) distance. Second, they introduce a sampling method called Mask-Aware Truncated Uniformization (MATU). The key advantage of MATU is that it removes the restrictive bounded-score assumptions required by prior theoretical work and leverages the property that tokens are unmasked at most once. The authors prove that MATU achieves a nearly $\epsilon$-free complexity of $O(d\ln d)$ that surpasses existing methods for both masked and uniform discrete diffusion.

### Strengths
The paper is properly written, with formal assumptions, lemmas, and theorems. The authors try to present rigorous theoretical analyses, which is appreciated.

### Weaknesses
However, given the following reasons, I believe the paper has major oversights on previous work and makes little contribution to the dLLM community.
- Missing discussion of important related work. The uniformization discussed in the paper is just an approximated version of the "first-hitting sampler" in [1]. [1] already proves the equivalence of MDMs to masked models in [2][3][4], suggesting that the time $t$ in MDMs can be fully removed, and the sampling of MDMs is **exact** by $d$-step token-by-token decoding. Given this fact, all the discussions on complexity in this paper appear unnecessary and outdated to me because [1] has proposed the ultimate solution to MDMs. The paper did not even mention [1], and did not discuss the relationship between [1] and the proposed MATU.
- Lack of experiments. The proposed algorithm and complexity analyses are not verified by experiments, even on the most basic setting of SEED/MDLM, which measures generative perplexity. Not to mention the applicability to more practical dLLMs.
- Useless for application. As far as I know, practical dLLMs like LLaDA already remove time $t$ in the network, and adopt the MaskGIT-style token-by-token decoding, even with more heuristic confidence-based decoding orders. Therefore, all the analyses concerning $t$ or Euler sampler in this paper are completely useless to practical dLLMs.

[1] Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling (ICLR 2025)

[2] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)

[3] MaskGIT: Masked Generative Image Transformer (2022)

[4] Mask-Predict: Parallel Decoding of Conditional Masked Language Models (2019)

### Questions
- How do your findings reconcile with [1], which demonstrates that Masked Diffusion Models are theoretically equivalent to time-agnostic masked models? Does your complexity analysis, which is tied to the continuous-time diffusion process, still provide practical motivation for using the diffusion framework over simpler, equivalent masked model formulations?
- [1] identified a critical numerical precision issue in Gumbel-based sampling that led to unfairly optimistic evaluations of MDMs. Does your theoretical analysis account for such numerical instabilities? More practically, how does your proposed MATU sampler perform empirically when benchmarked using 64-bit precision, which has been shown to be necessary for fair evaluation?
- Could you provide an empirical comparison of MATU against the Euler sampler and the "first-hitting sampler" from [1]? A comparison of wall-clock time, generative perplexity, and sentence entropy on a standard text dataset would be crucial to validate that the theoretical gains in complexity translate to empirical benefits.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a theoretical analysis of masked discrete diffusion (MDD), deriving convergence bounds for the Euler sampler and proposing a Mask-Aware Truncated Uniformization (MATU) method with nearly ε-free complexity. The work aims to explain why masked diffusion achieves faster convergence than uniform discrete diffusion.

### Strengths
* The paper establishes a clear theoretical foundation for **masked** discrete diffusion, a setting widely used in discrete and language diffusion models but previously lacking formal complexity analysis.
* Provides the first rigorous proof for the Euler sampler in discrete diffusion, bridging empirical implementations and theoretical understanding.
* They offer a near $\epsilon$-free convergence guarantee, with MATU algorithm representing a genuine improvement over prior uniformization approaches.

### Weaknesses
* **Presentation: missing intuitive root for the difference.**
  As a reader, I wanted to clearly see *what makes the $\tau$-leaping algorithm different from Euler sampling*, and *how that difference theoretically leads to different error bounds.*
  Currently, the paper concludes that Euler achieves a better complexity rate, but the *intuitive mechanism* behind this improvement remains unclear.

* **(Minor) Lack of connection between theoretical and empirical behaviors.**
  While the paper focuses on theoretical convergence rates, it would help to include an explicit discussion on how these rates correspond to practical behavior observed in masked discrete diffusion models. Without that bridge, the theoretical results feel somewhat detached from the empirical phenomenon the paper aims to explain. A qualitative, Intuitive explanation would be enough.

### Questions
* What is the precise difference between $\tau$-leaping and Euler sampling algorithms?
  In the context of masked discrete diffusion, could the authors concretely explain how these two update rules differ, and *what drives the improvement* from $\tau$-leaping’s $O(d\epsilon^{-2})$ to Euler’s $\tilde{O}(d^2\epsilon^{-3/2})$ in Theorem 1?
  Is the improvement mainly due to variance reduction from removing Poisson randomness, or does it come from the independence assumption between token coordinates?

* In addition, [1] proposes **$\tau$-leaping–Tweedie**. How does that method differ from Euler sampling under the masked discrete diffusion setup?
  What complexity rate would you expect if τ-leaping–Tweedie were analyzed under your theoretical framework?

* This may be tangential, but what is the effective **complexity of autoregressive models** in comparison?
  Conceptually, how does that relate to the $O(d \ln d)$ result of Theorem 2?

[1] *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, [https://arxiv.org/abs/2310.16834](https://arxiv.org/abs/2310.16834)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the theoretical complexity of masked discrete diffusion models, i.e. discrete diffusion models based on absorbing/masking continuous-time Markov chains. Current studies on the mathematical theory of discrete diffusion models mainly focused the case of the uniform case, while this paper is one of the first studies on the masking/absorbing case. Compared to relatively few prior work on the theory of masked discrete diffusion models like [1], this paper provides the first theoretical analysis of the commonly used Euler sampler and a modified version of the uniformization-style sampler called Mask-Aware Truncated Uniformization (MATU) for the case of masked discrete diffusion models. Specifically for the latter uniformization-style sampler, the modification proposed in this paper is mainly motivated from the truncation technique adopted in [2], which makes full use of the masking property, removes the assumption on the boundedness of the learnt score function, and achieves a nearly $\epsilon$-free convergence rate with respect to the TV distance (Here $\epsilon$ denotes the score approximation error). Such theoretical results partially explain why the masking case surpasses the uniform case in practice under certain circumstances.

### Strengths
To the best of the reviewer's knowledge, there has been relatively few work studies the theoretical complexity of masked discrete diffusion model - probably [1] is the only related work. Hence, the reviewer thinks that this paper studies an important question with interesting results that improves the dependency on the score approximation error $\epsilon$. Also, the results are presented in a rigorous and detailed way (For instance, Table 1 clearly exhibits how the results presented in this work compared to existing work on the theory of uniform/masked discrete diffusion models).

### Weaknesses
(1) It seems to the reviewer that the assumption [A2] has never been adopted in existing work [1] on the theory of masked discrete diffusion models. Hence, the authors probably need to add a short paragraph to further comment on whether this is a strong assumption and how it impacts the proof strategy. Does it somehow relate to Assumption 4 in [1]?

(2) Even though the reviewer fully understands that this is a theoretical paper, the authors are encouraged to include a few small-scale experiments (maybe just one on a toy model) to illustrate how the theoretical studies might partially explain why masked discrete diffusion model surpasses the uniform discrete diffusion model under certain circumstances. This would greatly improve the quality of the manuscript. 

(3) Moreover, it seems to the reviewer that the authors might be missing a few related work on the theory of discrete diffusion models. For instance, an incomplete of concurrent work include but not limit to [3,4,5]. Given that the theoretical studies of discrete diffusion models is a relatively new area in the machine learning theory community, including a complete list of related work can potentially help enhance the quality of the paper.

### Questions
Would it be possible for the authors to comment on whether the technique developed in this paper can be applied to analyze the Tweedie $\tau$-leaping proposed in [6] for the case of masked discrete diffusion models? Even some brief intuition on how complexity might change would be appreciated!  

Overall, I think the paper might be considered for top ML venues like ICLR, but the authors should probably address all questions above, add papers listed below as extra references and discuss them appropriately.

References:

[1] Liang, Yuchen, Renxiang Huang, Lifeng Lai, Ness Shroff, and Yingbin Liang. "Absorb and Converge: Provable Convergence Guarantee for Absorbing Discrete Diffusion Models." arXiv preprint arXiv:2506.02318 (2025).

[2] Huang, Xunpeng, Yingyu Lin, Nikki Lijing Kuang, Hanze Dong, Difan Zou, Yian Ma, and Tong Zhang. "Almost Linear Convergence under Minimal Score Assumptions: Quantized Transition Diffusion." arXiv preprint arXiv:2505.21892 (2025).

[3] Srikanth, Aadithya, Mudit Gaur, and Vaneet Aggarwal. "Discrete State Diffusion Models: A Sample Complexity Perspective." arXiv preprint arXiv:2510.10854 (2025).

[4] Wan, Zhengyan, Yidong Ouyang, Qiang Yao, Liyan Xie, Fang Fang, Hongyuan Zha, and Guang Cheng. "Error Analysis of Discrete Flow with Generator Matching." arXiv preprint arXiv:2509.21906 (2025).

[5] Liang, Yuchen, Yingbin Liang, Lifeng Lai, and Ness Shroff. "Discrete Diffusion Models: Novel Analysis and New Sampler Guarantees." arXiv preprint arXiv:2509.16756 (2025).

[6] Lou, Aaron, Chenlin Meng, and Stefano Ermon. "Discrete diffusion modeling by estimating the ratios of the data distribution." arXiv preprint arXiv:2310.16834 (2023).

### Soundness
3

### Presentation
3

### Contribution
3
