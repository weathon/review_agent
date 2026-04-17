# TARG: Training-Free Adaptive Retrieval Gating for Efficient RAG

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Retrieval-Augmented Generation (RAG) improves factuality but retrieving for every query often hurts quality while inflating tokens and latency. We propose Training-free Adaptive Retrieval Gating (TARG), a single-shot policy that decides when to retrieve using only a short, no-context draft from the base model. From the draft’s prefix logits, TARG computes lightweight uncertainty scores—mean token entropy, a margin signal derived from the top-1/top-2 logit gap via a monotone link, or small-$N$ variance across a handful of stochastic prefixes—and triggers retrieval only when the score exceeds a threshold. The gate is model-agnostic, adds only tens to hundreds of draft tokens, and requires no additional training or auxiliary heads. On NQ-Open, TriviaQA, and PopQA, TARG consistently shifts the accuracy–efficiency frontier: relative to Always-RAG it matches or improves EM/F1 while reducing retrieval by 70–90\% and cutting end-to-end latency, and it remains close to Never-RAG in overhead. A central empirical finding is that under modern instruction-tuned LLMs the margin signal is a robust default (entropy compresses as backbones sharpen), with small-$N$ variance offering a conservative, budget-first alternative. We provide ablations over gate type and prefix length and use a $\Delta$-latency view to make budget trade-offs explicit.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TARG (Training-free Adaptive Retrieval Gating), a single-shot policy designed to enhance the efficiency and reliability of RAG systems by moving beyond the costly and error-prone Always-RAG approach. This approach enables the LLM to conduct a cost-effective self-assessment before retrieval by generating a brief, context-free prefix (around 20 tokens). It then calculates a lightweight Uncertainty Score based on the prefix's raw logits, such as the Margin (the difference between the top-1 and top-2 logits), and only initiates retrieval if the score exceeds a set threshold. The proposed method consistently enhances the accuracy-efficiency frontier compared to Always-RAG, achieving similar or better EM/F1 scores while greatly reducing the retrieval rate and significantly decreasing end-to-end latency, demonstrating its utility as a practical and cost-effective solution.

### Strengths
1. This paper addresses the three most urgent challenges for RAG deployment: high cost, increased latency, and accuracy decline caused by noisy context from unconditional retrieval. By solving when to retrieve, the paper offers a foundational solution for making RAG economically feasible and reliable in real-world applications.

2. The authors introduce a novel training-free method for conditional retrieval. It avoids the complexity and cost of training extra models or control heads by relying only on inherent uncertainty signals (Margin, Variance) from the base LLM's raw prefix logits.

3. The method addresses RAG's main cost issue by significantly lowering the retrieval rate compared to Always-RAG. This results in substantial token savings and nearly matches the minimal latency of the Never-RAG baseline.

### Weaknesses
1. Questions on Experimental Rigor and Integrity
> The paper's validation requires greater statistical rigor and transparency regarding the experimental baselines. First, the reporting uses single-point estimates (EM/F1) without confidence intervals or standard deviations. The notable performance shifts observed at minimal retrieval rates (e.g., 0.001 Retrieval Rate) necessitate a comprehensive statistical significance test (such as a t-test) across multiple independent runs to confirm the reliability and stability of these minimal-budget operating points. Second, the experimental design raises methodological questions regarding the Always-RAG baseline, which consistently and significantly underperforms Never-RAG across all datasets. This result suggests that the retrieval context is largely noisy or distracting in the current setup, meaning the reported gains primarily demonstrate TARG's ability to filter suboptimal retrieval rather than its capacity to maximize the benefit from high-quality external evidence.

2. Limited Scope of Comparative Baselines
> The comparison is restricted to only the unconditional baselines (Always-RAG and Never-RAG). However, since the main advantage of TARG is its ability to reduce latency through context filtering, the comparison remains incomplete. The authors should evaluate TARG's accuracy and efficiency against established methods for context compression and summarization. Comparing TARG's gating approach with these alternative methods for reducing context length and latency is essential to demonstrate its full competitive advantage.

3. Non-Trivial Calibration Cost Challenges the "Training-Free" Claim
> The simplicity of TARG is fundamentally challenged by the high real-world overhead of threshold calibration. Since performance and latency are highly sensitive to the decision threshold, finding the optimal threshold requires a development-set sweep for every new domain or model. We question whether this non-trivial, domain-specific optimization process is, in practice, as demanding to maintain as the auxiliary training that TARG is designed to replace.

### Questions
1. Statistical Reliability: Given the use of single-point estimates (EM/F1) and the dramatic shifts at minimal retrieval rates (e.g., 0.001), can the authors confirm the stability of these results by reporting the statistical significance (e.g., t-test) across multiple independent runs?

2. Baseline Quality: Since Always-RAG significantly underperforms Never-RAG across all datasets, the retrieval context appears to be consistently noisy. Can the authors discuss whether the reported gains primarily demonstrate filtering suboptimal retrieval rather than maximizing the benefit from high-quality external evidence?

3. Missing Comparative Baselines: Since TARG's main advantage is latency reduction via context filtering, can the authors compare its accuracy-efficiency frontier against established context compression or summarization methodologies to demonstrate TARG's full competitive advantage?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Predictions of the retriever could be noisy and in-turn lead to degradation in the performance, as well as efficiency due to increased context length. This paper focuses on when to trigger the retriever in the RAG setup. Retriever adds additional context to disambiguate the query, it could be that the parameters of LLM already has enough information that the query can be answered correctly without the need for context enrichment. The internal knowledge of LLM can measured using heuristic-metric based on the output or the internal state of the LLM. To efficiently use the retriever, it is triggered only when the internal knowledge is not sufficient. This paper comes up with three such metrics that measure uncertainty in the LLM generations (measure of internal knowledge) -- 1) mean token entropy, 2) margin score from top-1 versus top-2 logit gap and 3) a small-N variance from handfull of stochastic prefixes. Among these metrics top-1 vs top-2 margin is found to be more robust. On NQ-open, Trivia-QA and PopQA, their approach, TRAG, shows improvements in both accuracy and efficiency over systems that always triggers retriever and that does not use one at all.

### Strengths
1. Proposes three signal to measure the uncertainity in the LLM generations -- 1) entropy, 2) margin score and 3) small-N variance.
2. Gating decision is just based on a single scalar threhold, and for long generations an optional single re-checking is done after every m token generations if retreiver is not yet used.
3. Proposes method to calibrate the threshold based on retrieval budget and to maximize accuracy.

### Weaknesses
1. Paper shows results on only three simple QA datasets containing short answers.
2. $u_t$ described in lines 153-154 is not used anywhere else.
3. Existence of $\tau_*$ (lines 223-225): There is no relation between the quality of retrieval and the proposed uncertainty measure so you cannot guarantee the existence of such a threshold. Take for example two cases; a) the retriever always gives the correct answer to the question as context, here Always-RAG should do better than TRAG and b) always give the same random text as context, then here Zero-RAG is better than TRAG. So the in-equality (lines 223-225) does not hold.
4. Paper only considers two LLMs: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
5. Paper contains lots of repeated text like the discussion section, pointed mentioned in the section is already coverd before.

### Questions
1. The method assumes that the three signals proposed are good proxy for parametric knowledge of the LLM. Could you show an analysis of this correlation? That is when the signal is low base generator generates correct answers and when signal is high the answer is wrong.
2. Paper presents results on simple QA that requires short generations: NQ, TriviaQA and PopQA. Please show numbers on other datasets: a) Complex QA requiring mult-hop reasoning to answer the question -- 2WikiMultiHopQA, HotpotQA, b) requiring long form generations --Biography, ALCE-ASQA c) PubHealth covering true-false questions d) Arc-Challenge consisting multiple choice questions.
3. There are other signals proposed in the papers mentioned in the related works, which are based on the output or the internal state of the LLM. How does your approach fare against them? Like a) "semantic entropy" in SUGAR, b) "Self-aware Uncertainty Estimator" in SEAKR which uses determinant of Gram matrix of hidden representation to measure uncertainity in generations. Create a comparison table something like Table 1 and 2 in SUGAR comparing different adaptive RAG methods.
4. Show performance on more backbone LLMs, Gemma, Phi, Mistral series in both the 1-3 billion and 6-9 billion parameter range.
5. Perform a more comprehensive hyperparameter sweeps, for both k and the threhold. Paper considers only 3 different values for k = {10, 20, 30}. Pick more values for both k and threshold, and create plots to show how performance varies with k and threshold -- separately using line plots and combined using heat-maps.

### Soundness
2

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
4

### Summary
This paper addresses the inefficiencies inherent in standard Retrieval-Augmented Generation (RAG) pipelines. While RAG improves factuality, the common practice of retrieving for every query ("Always-RAG") significantly increases latency and token consumption. Furthermore, it can degrade performance if the retrieved context is noisy or irrelevant.

The authors propose TARG (Training-free Adaptive Retrieval Gating), a lightweight, model-agnostic, single-shot policy to decide when to retrieve. TARG operates by first generating a short, no-context draft (prefix) using the base LLM. It then computes an uncertainty score based on the logits of this prefix. If the uncertainty exceeds a calibrated threshold $\tau$, retrieval is triggered; otherwise, the model proceeds using only its parametric memory.

The paper investigates three training-free uncertainty signals:
- Entropy: Mean token entropy of the prefix.
- Margin: Derived from the gap between the top-1 and top-2 logits (a smaller gap indicates higher uncertainty).
- Variance: Measured by disagreement across a small number (N=3) of stochastic prefixes.

A key empirical finding is the interaction between uncertainty signals and the "sharpness" of the underlying LLM. As modern instruction-tuned models become more peaked (e.g., Llama-3.1-8B), prefix entropy compresses and loses discriminative power. In contrast, the Margin and Variance signals retain their dynamic range and correlate better with the necessity of retrieval.

Evaluated on NQ-Open, TriviaQA, and PopQA, TARG consistently shifts the accuracy-efficiency frontier. It often matches or exceeds the accuracy of Always-RAG while reducing retrieval frequency by 70-90%, keeping latency close to the Never-RAG baseline.

### Strengths
- Simplicity: the "plug-and-play" nature allows for easy integration into existing RAG systems with minimal overhead (limited to the generation of a short k-token prefix).
- Analysis is insightful: The analysis regarding the behavior of different uncertainty metrics under modern, sharp instruction-tuned LLMs (Section 6) is a valuable contribution. The observation that entropy compresses as backbones improve, while the top-1/top-2 logit gap (Margin) and disagreement (Variance) retain dynamic range, provides actionable guidance for implementing uncertainty estimation.
- Strong empirical results: The results convincingly demonstrate that TARG improves the accuracy-efficiency trade-off. It significantly reduces retrieval rates while often improving accuracy over the Always-RAG baseline. The authors' use of "Δ latency" (incremental overhead vs. Never-RAG) provides a clear and practical framing of the computational cost.

### Weaknesses
- 'Usefulness calibration' assumption may be strong: the theoretical underpinning of TARG (Section 3.3) relies on the assumption that the uncertainty score $U(q)$ correlates strongly with the expected benefit of retrieval ($\Delta(q)$). This assumption may not always hold. Scenarios where the model is confidently wrong (low U, high potential $\Delta$) or uncertain but the retriever consistently fails (high U, negative $\Delta$) could violate this assumption. The paper would benefit from a deeper error analysis focused on these quadrants.
- Re-check is not evaluated: while Algorithm 1 describes an optional re-check every $m$ tokens, its effectiveness is not evaluated.

### Questions
- The paper evaluates the three gates (Entropy, Margin, Variance) independently and concludes that Margin is the best default. Did the authors consider aggregating these signals?
- Could the authors provide an analysis of the cases where TARG decides not to retrieve (low U) but the resulting answer is incorrect? How frequently do these errors occur, and do they represent "unknown unknowns" (the model was confidently wrong), or cases where the knowledge was absent from the corpus anyway?
- Section 3.2 briefly mentions an optional "single re-check" applied every $m$ tokens. Were experiments conducted using this dynamic approach? How does the accuracy-efficiency trade-off compare to the single-shot TARG?

### Soundness
3

### Presentation
3

### Contribution
3
