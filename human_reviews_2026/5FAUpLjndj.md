# SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding

- Decision: Reject
- Scores: 2, 2, 6, 8

## Abstract
Autoregressive language models achieve state-of-the-art performance across a wide range of natural language processing tasks, but suffer from high inference latency due to their sequential decoding nature. Speculative decoding (SD) mitigates this by employing a lightweight draft model to propose candidate tokens, which are selectively verified by a larger target model. While existing methods either adopt multi-draft strategies to increase acceptance rates or block verification techniques to jointly verify multiple tokens, they remain limited by treating these improvements in isolation. In this work, we propose SpecTr-GBV, a novel SD method that unifies multi-draft and greedy block verification (GBV) into a single framework. By formulating the verification step as an optimal transport problem over draft and target token blocks, SpecTr-GBV improves both theoretical efficiency and empirical performance. We theoretically prove that SpecTr-GBV achieves the optimal expected number of accepted tokens for any fixed number of draft sequences, and this bound improves as the number of drafts increases. Empirically, we evaluate SpecTr-GBV across five datasets and four baselines. Our method achieves superior speedup and significantly higher block efficiency while preserving output quality. In addition, we perform comprehensive ablation studies to evaluate the impact of various components in the model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes SpecTr-GBV, a speculative decoding (SD) method that combines multi-draft generation (à la SpecTr) with greedy block verification (GBV). The core idea is to formulate token verification as an optimal transport (OT) problem over entire draft blocks from multiple sequences, rather than position-by-position. The authors provide a theoretical proof that their method achieves the optimal expected number of accepted tokens for any fixed number of drafts, and this bound improves as the number of drafts increases. Extensive experiments across five datasets and four strong baselines demonstrate that SpecTr-GBV consistently outperforms prior art in both block efficiency and end-to-end speedup while preserving output quality.

### Strengths
1. Strong Theoretical Foundation: The paper provides a rigorous theoretical analysis, proving that SpecTr-GBV achieves the optimal expected acceptance length. This is a significant contribution that elevates the work beyond purely empirical improvements.
2. Comprehensive and Convincing Experiments: The empirical evaluation is thorough, covering multiple model families (DeepSeek, CodeLlama, Vicuna), diverse datasets (code, math, language modeling, instruction following), and extensive ablation studies on key hyperparameters (K, L, T).
3. Clear Practical Impact: The consistent speedup gains (e.g., +29.3% over SD, +8.2% over SpecTr in the 33B setting) demonstrate tangible benefits for real-world LLM inference.

### Weaknesses
1. The additional verification time cost caused by GBV+SpecTr needs to be theoretically analyzed and experimentally discussed.
2. The article is a combination of two methods in speculative decoding. This form of combination lacks significance.
3. All comparisons are against vanilla SD, SpecTr, and GBV—but miss recent strong contenders like Hydra, Medusa, or EAGLE, which use tree-structured or multi-head drafts. These often outperform i.i.d. multi-draft methods in practice. Without comparing to them, the claimed “state-of-the-art” status is unconvincing.

### Questions
1. Can you provide a breakdown of wall-clock time spent in each phase for SpecTr-GBV vs. SpecTr? It is an interesting question where the acceleration benefits come from.
2. Why not compare against tree-based speculative decoding methods? Your i.i.d. draft assumption may be fundamentally less efficient than structured candidate generation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper revisits speculative decoding (SD) by unifying multi-draft verification (SpecTr) with block-level verification (GBV). The method, SpecTr-GBV, formulates verification as an OT problem over draft/target blocks, gives closed-form acceptance/residual rules (Eqs. (2)–(4)), adds a distribution-modification step to preserve fidelity across iterations (Alg. 2, Thm. 3.2), and claims to achieve the maximum expected acceptance length for any fixed number of drafts K (Lemma 4.1,Thm. 4.3).

### Strengths
Re-tackles an important bottleneck. Accelerating multi‑draft verification is a natural and valuable direction.

Clear paper and algorithm presentation.

### Weaknesses
My main concern is theoretical soundness. Theory appears not rigorous to me.

1. The quantity $h_{ik}$ from Eq. (2) is used as a probability but can exceed 1 for $K=1$.

This paper assume $h_{ik}$ to be a probability. In page 5, it saids "Each token sub-block ... is accepted with probability $h_{ik}$". The same “(h) as a probability” usage happens in Appendix Eq. (11) in the proof. So for the proof to be valid, $h_{i,k}$ need to be no larger than 1.

However, setting K=1 in Eq. (2) yields

$$h\_{ik}=\dfrac{A+\Delta\_+}{A-\Delta},A=\sum\_x (q(x_i,x)-p(x_i,x))\_+,\Delta = q(x_i)-p(x_i)$$

If $\Delta>0$ (i.e., $q(x_i)>p(x_i)$), then $h_{ik}>1$.

Thus $h_{ik}$ is not a valid probability under the paper’s own acceptance test and proof steps.


2. Lemma 4.1 relies on a conditional‑independence step that is not guaranteed by the coupling.
See proof in Eq. (13).

### Questions
SpecTr optimality vs. K‑SEQ. My understanding is that SpecTr’s exact optimality follows from solving an OT, which is generally intractable, hence K‑SEQ is an approximation. In your method, Algorithm 1 does not solve an LP. In the (L=1) case, is SpecTr‑GBV guaranteed to match SpecTr’s optimal acceptance rate? If yes, why it could avoid LP which is necessary for SpecTr? If not, how does Thm. 4.3 still claim optimality?

### Soundness
1

### Presentation
3

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
This paper proposes SpecTr-GBV, a speculative decoding (SD) algorithm that unifies multi-draft decoding and greedy block verification (GBV) under a single framework. It formulates the verification process as an optimal transport (OT) problem between multiple draft token blocks and the target token block, and proves that this formulation achieves the optimal expected acceptance length for any fixed number of drafts. The method theoretically extends prior works—SpecTr (Sun et al., 2023) and GBV (Sun et al., 2024b)—and empirically shows consistent speed-ups across five datasets (HumanEval, GSM8K, MGSM, LM1B, Alpaca) using various target/draft model pairs (DeepSeek-33B/1.3B, 6.7B/1.3B, etc.).

### Strengths
The paper is very clearly written; The paper unifies two influential lines of speculative decoding: multi-draft optimal transport (SpecTr) and block-level greedy verification (GBV). The evaluation is very comprehensive involving multiple models. The paper also provides in-depth analysis in terms of temperature, draft lengths.

### Weaknesses
The paper is technically correct and experimentally comprehensive, but the overall contribution is incremental. It mainly combines two existing speculative decoding mechanisms (SpecTr and GBV) under one unified framework, without introducing new conceptual insights or architectural innovation.

### Questions
N/A

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
The paper introduces a new method for speculative decoding, SpecTr-GBV. The paper conducts a fairly thorough theoretical analysis of their method and prove that it 1) faithfully samples from the desired distribution and 2) achieves expected acceptance length that is (in some sense, compared to a class of methods) optimal. The paper then presents several sets of experiments demonstrating the advantage of their method compared to other methods.

### Strengths
I believe the paper has an interesting and compelling mix of theory and empirics. It is particularly compelling that the proposed method SpecTr-GBV achieves the upper bound for expected acceptance length, and moreover that this performance gain is reflected in experiments. Specifically there is a major performance gain in Block Efficiency (BE), which (if my understanding is correct) corresponds to improvement in $\mathbb{E}[\tau]$ as suggested by the theoretical guarantee Theorem 4.3. The theoretical bound also subsumes that of GBV when $K=1$, which makes the extension to larger $K$ and the general approach convincing.

Overall, I think the method makes sense, has relevant theoretical guarantees, and has practical significance. The paper is also overall well written and I found it easy to follow.

### Weaknesses
I am relatively unfamiliar with the prior literature on speculative decoding, my background is on the theory side, in high dimensional sampling algorithms. As such it is not clear to me how novel the proposed algorithm is, and how much of a contribution it represents vs the literature. It seems like it is built off of similar ideas as KSEQ and GBV, albeit with several modifications / combining ideas (this is even stated explicitly by the authors in line 197 at start of subsection 3.1). That is fine, but it's unclear to me how much of a development this paper comprises in the literature.

Another weakness is about theoretical guarantees of efficiency (some of these concerns apply also to previously developed algorithms). The method requires generating K draft sequences, does this make the method slower (in theory) than GBV? Also eq (2) requires summing over all possible tokens x, similarly (4) requires sampling from a distribution over tokens x with no formal guarantees of `niceness' of a discrete univariate distribution such as unimodality (note that these issues also apply to methods like GBV). Does the proposed method require many more such inefficient computations than previously developed methods? Such concerns are certainly of practical importance as well, e.g. as noted in lines 418-422 that the generation of the token from the draft model cannot be ignored in practice.

Finally, a few writing suggestions: section 6 on related work could be moved earlier. Also Algorithm 1 is long and very hard to read.

### Questions
-Could the authors please provide some intuition on how the attained bound in Theorem 4.3 seems to behave in terms of L, K? Of course this depends on p and q but some idea on how this might behave (e.g. linear in L, $\sqrt{L}$, dependence on $K$ etc for some particular but representative cases of p, q) would be much appreciated. Some properties (Monotonicity, Convergence, Consistency) are written but a better idea of the quantitative behavior would be useful.

-Is there an idea of the theoretical guarantees for SpecTr-GBV's computational efficiency vs pre-existing methods in the literature?

-Theorem 4.3 shows that SpecTr-GBV satisfies the upper bound in Lemma 4.1, i.e. Lemma 4.1 is tight for SpecTr-GBV. Some intuition on why this is the case would be great, could the authors please provide some?

### Soundness
3

### Presentation
3

### Contribution
3
