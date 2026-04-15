# Efficient Inference for Large Language Model-based Generative Recommendation

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Large Language Model (LLM)-based generative recommendation has achieved notable success, yet its practical deployment is costly particularly due to excessive inference latency caused by autoregressive decoding. For lossless LLM decoding acceleration, Speculative Decoding (SD) has emerged as a promising solution. However, applying SD to generative recommendation presents unique challenges due to the requirement of generating top-K items (i.e., K distinct token sequences) as a recommendation list by beam search. This leads to more stringent verification in SD, where all the top-K sequences from the target LLM must be successfully drafted by the draft model at each decoding step. To alleviate this, we consider 1) boosting top-K sequence alignment between the draft model and the target LLM, and 2) relaxing the verification strategy to reduce trivial LLM calls. To this end, we propose an alignment framework named AtSpeed, which presents the AtSpeed-S optimization objective for top-K alignment under the strict top-K verification. Moreover, we introduce a relaxed sampling verification strategy that allows high-probability non-top-K drafted sequences to be accepted, significantly reducing LLM calls. Correspondingly, we propose AtSpeed-R for top-K alignment under this relaxed sampling verification. Empirical results on two real-world datasets demonstrate that AtSpeed significantly accelerates LLM-based generative recommendation, e.g., near 2x speedup under strict top-K verification and up to 2.5x speedup under relaxed sampling verification. The codes and datasets are available at~\url{https://github.com/Linxyhaha/AtSpeed}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a speculative decoding approach for generative recommendation. In traditional speculative decoding, a cheap draft LLM is given a prefix and at each decoding step it suggests $N$ tokens among which the token predicted by the target LLM should be. This enables the target model to verify all the drafted predictions in parallel through a single call, conditioning on the draft model's previous predictions. The sequence of drafted predictions is then accepted up to the decoding step where the draft model and the target model disagree in their predictions.

In top-$K$ generative recommendation, a beam search is performed to generate the $K$ item identifiers. Applying speculative decoding to this problem then requires the $N$ tokens suggested by the draft model to contain the $K$ tokens predicted by the target model to accept the current decoding step. This condition is hard to fulfill in practice. To address this, the paper defines a framework named AtSpeed, which trains the draft model to be more aligned with the target model and optionally relaxes the acceptance condition to allow the $N$ drafted tokens to contain tokens that are similar (rather than identical) to the $K$ target tokens. AtSpeed was validated through experiments on the Amazon Beauty and Games datasets.

### Strengths
- The approach proposed for applying speculative decoding to generative recommendation is both novel and intuitive.
- The experiments are comprehensive and provide convincing evidence of the benefits of AtSpeed from an empirical standpoint.
- The source code was made available.

### Weaknesses
- The paper suffers from an overall lack of mathematical rigor, both in the proofs and notations, which raises concerns about the theoretical soundness of the approach. In particular, some key formulas, such as the proposed alignment objective, appear to be either incorrect or lack sufficient derivation to validate their correctness (see specific points in the 'Questions' section).
- The paper is overall lacking polish and contains multiple typos in mathematical formulas, which makes it hard for the reader to follow the details of the approach. Thorough proofreading would be needed to improve the paper's readability. Certain parts of the paper could also benefit from being further clarified, as pointed out in the 'Questions' section.

### Questions
- In Section 3.1, paragraph "Acceptance Rate", the condition given for having $\beta = 1$, namely that $p(\mathbf{y}) \geq p(\mathbf{y}_K)$ for all $\mathbf{y}$ in $\mathcal{Y}_q$, doesn't seem to be fulfillable. By definition of $\mathbf{y}_K$, there exists only $K-1$ sequences with greater probability according to $p$, yet the condition requires that all $N$ ($\geq K$) sequences in $\mathcal{Y}_q$ have a probability greater than $\mathbf{y}_K$. The only possibility for this condition to be realized is if $\mathcal{Y}_q$ contains duplicates, which should not happen in the case of beam search.
- Derivation of Eq 15 in App A.2 (which leads to the definition of the alignment objective in Eq 3) seems incorrect or at least misses important steps to be understandable for the reader. More steps or explanations are needed.
- Eq 3 introduces $\mathcal{D'}$ in which $\mathcal{Y}$ is the top-$K$ of a mixture of $q$ and $p$. Following Gu et al (2024) is mentioned as the motivation for this, but it would be appreciated to have more intuitions on this choice. Moreover, the $\mathcal{D'}$ used later in the relaxed alignment objective (Eq 8) is different. What is the rationale to have different $\mathcal{D'}$ in the strict and relaxed objectives?
- What loss is used in practice for $\mathcal{L}_{Rec}$? It is only mentioned to be a recommendation loss but additional details would be appreciated (at least in the appendix).
- The relaxation sampling verification strategy misses some intuition: why is $p(\mathbf{y}) \geq q(\mathbf{y})$ the right criterion for accepting a sequence $\mathbf{y}$? Why are such $\mathbf{y}$'s good candidates? Moreover, in the case of rejection, $\mathbf{y}$ is drawn from $p' = norm(max(0, p(\mathbf{y}) - q(\mathbf{y})))$ but it is unclear whether this is an actual distribution and what the norm operator exactly consists of. Is $p'$ denoting the uniform distribution over the $\mathbf{y}$'s such that $p(\mathbf{y}) \geq q(\mathbf{y})$? If so, how does one sample from this in practice? The definition of this distribution would deserve more clarifications and details.
- The "with-replacement sampling approximation" paragraph in App A.2 is difficult to follow so it would benefit from being reworked.
- The proof of Lemma 1 in App A.2, which corresponds to Lemma 3.3 from Leviathan et al (2023), is missing the step with the term $1 - \sum_{\mathbf{y}} \frac{p(\mathbf{y})+q(\mathbf{y})-|p(\mathbf{y})-q(\mathbf{y})|}{2}$.
- In Eq 7, what does $\sum_K$ denote? It seems like there is an index variable missing there.
- The strategy relying on tree-based attention to speed up inference would benefit from being described in more details (at least in Appendix). Section 3.3 only refers to Figure 2(c) to describe the strategy, but this figure is not self-explanatory.
-  The WS metric represents the walltime speedup, but with respect to which baseline? I assume this is in comparison to directly running the target model without speculative decoding, but it would be helpful for the reader to mention this when defining the metric.
- Table 1 reports the results for AtSpeed-S and AtSpeed-R on both the strict and the relaxed settings. How can AtSpeed-S be applied to the relaxed setting and AtSpeed-R to the strict setting?
- In Figure 5 of the Appendix, it seems that a larger value for $\alpha$ is always beneficial for AtSpeed-R, whereas Figure 3 (c) showed that $\alpha$ should neither be too small nor too large for AtSpeed-S. Are there any intuitions on these different behaviors between AtSpeed-R and AtSpeed-S?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AtSpeed, a framework to accelerate LLM-based generative recommendation systems through speculative decoding. The main challenge addressed is the inherent inefficiency of autoregressive beam search in generating top-K recommendations. The authors identify that traditional SD methods, which focus on N-to-1 verification, are insufficient for recommendation tasks that require N-to-K verification to output a ranked list of K items. To tackle this, the paper introduces two methods: AtSpeed-S for strict top-K alignment and AtSpeed-R for relaxed sampling verification, which aims to reduce the number of LLM calls while preserving accuracy. Experimental results demonstrate that AtSpeed achieves significant speedups (up to 2.5×) with minimal degradation in recommendation accuracy.

### Strengths
- The paper extends the N-to-1 verification to the N-to-K verification in the context of generative recommendation, which fits the setting of real-world recommender systems.
- The authors provide a solid theoretical foundation for AtSpeed to align the draft and target models. The optimization objectives are clearly motivated and mathematically sound.
- Empirical results are compelling, showing that AtSpeed can achieve up to a 2.5× speedup while maintaining competitive recommendation accuracy. The results are well presented and highlight the practical utility of the proposed method.

### Weaknesses
- The paper may overclaim their contribution to be the first to propose the speculative decoding task for LLM-based recommender acceleration. There already exists prior work on speculative decoding for LLM-based recommendation: *A Decoding Acceleration Framework for Industrial Deployable LLM-based Recommender Systems (Xi, Yunjia, et al)*, which needs to be compared and discussed.
- The experiments are somewhat limited to two datasets (Amazon Beauty and Games). While these datasets are commonly used, the paper would benefit from broader validation across additional domains or larger-scale datasets.

### Questions
- [Q1] The paper primarily compares AtSpeed with KD-based baselines. Existing SD-based baselines should also be compared including the paper I mentioned before (A Decoding Acceleration Framework for Industrial Deployable LLM-based Recommender Systems). 
- [Q2] Have the authors tested their framework on any larger and more diverse datasets, such as MovieLens or more complex datasets like Goodreads? If not, can the authors comment on the expected performance and generalization ability of AtSpeed in these contexts?
- [Q3] The paper mentions the use of LLaMA-7B, but it would be interesting to know how AtSpeed scales with even larger models (e.g., LLaMA-13B or GPT-3). Do the authors anticipate any bottlenecks or limitations when scaling to larger models, especially concerning memory usage and GPU efficiency?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on accelerating inference in LLM-based generative recommendation using speculative decoding. It highlights the challenges of applying speculative decoding directly to the generative recommendation, due to the N-to-K issue. The authors introduce two improvements in both the drafting and verification stages. Experiments on two public datasets demonstrate the efficiency of the proposed method.

### Strengths
1. A timely study addressing inference efficiency in generative recommendation.
2. The ideas of (1) fine-tuning the draft model to align with top-K items of the target model and (2) adding a probability to accept rejected items in a more flexible verification setting are both novel and interesting.
3. The paper is well-written and easy to follow.
4. Experiments on two public datasets demonstrate the efficiency of the proposed method.
5. Code is available during the review phase, enhancing reproducibility.

### Weaknesses
1. Performance metrics (e.g., NDCG and Recall) are not well-presented. Only Table 2 includes some ranking metrics (also only Recall, without NDCG), which may raise doubts about the proposed method's ranking performance. Including these metrics in Table 1 and Figure 3 would strengthen the results. The limited metrics reported make it challenging to fully assess AtSpeed's ranking performance.
2. Presentation issues. Reporting WS@K and AS@K for all values of K in {1, 3, 5, 10, 20} in Table 1 seems unnecessary, especially since the discussion focuses on average WS and AS. I suggest presenting only representative Ks alongside the averages, freeing space for additional ranking metrics.

### Questions
Please refer to "Weaknesses" for more details.

Lastly, I want to share a related paper, "Inductive Generative Recommendation via Retrieval-based Speculation". This paper was released after the ICLR submission deadline and is currently available only as a preprint. While this is not a critique or question, it’s relevant as it discusses speculative decoding in generative recommendation. It proposes a dynamic N for the draft model and introduces a technique to perform limited beam search, using prefixes generated by the first few steps of the target model to guide the draft model. I believe this work shares some conceptual similarities with this paper, so I thought it worth mentioning.

### Soundness
3

### Presentation
4

### Contribution
3
