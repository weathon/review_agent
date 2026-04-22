# When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Ensembling Large Language Models (LLMs) has gained attention as a promising approach to surpass the performance of individual models by leveraging their complementary strengths. In particular, aggregating models’ next-token probability distributions to select the next token has been shown to be effective in various tasks. However, while successful for short-form answers, its application to long-form generation remains underexplored. In this paper, we show that using existing ensemble methods in long-form generation requires a careful choice of ensembling positions, since the standard practice of ensembling at every token often degrades performance. We identify two key factors for determining the ensembling positions: tokenization mismatch across models and consensus in their next-token probability distributions. Based on this, we propose $\textbf{SAFE}$, ($\textbf{S}$table $\textbf{A}$nd $\textbf{F}$ast LLM $\textbf{E}$nsembling), a framework that selectively ensembles by jointly considering these factors.  To further improve stability, we apply a probability sharpening strategy when the ensemble distribution becomes overly smooth, enabling the selection of more confident tokens during ensembling. Our experiments on diverse benchmarks, including MATH500 and BBH, demonstrate that SAFE outperforms existing methods in both accuracy and efficiency, with gains achieved even when ensembling fewer than 1\% of tokens.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies ensemble methods in long-form generation tasks, demonstrating that existing ensemble methods fail to perform well in these tasks because of tokenization mismatch across models and divergence in their next-token probability distributions. To address this problem, they propose SAFE, a novel ensemble method. Experiments show that SAFE achieves better performance and efficiency.

### Strengths
1. The focus on long generation/CoT tasks is a novel perspective for model ensemble research.
2. The proposed approach leads to better performance across different benchmarks and ensemble strategies. It also address the huge inference overhead.
3. The writing is clear and easy to follow.

### Weaknesses
1. Some related works should be discussed and compared, which could impact the novelty and relative performance:
    a) Speculative decoding is already utilized to accelerate model ensemble in CoS [1].

    a) The problem of OOV can be addressed by ensembling at a larger granularity (e.g., word of span), as shown in SweetSpan [2]. I think the same idea is used in SAFE.

    b) EVA [3] (which is the first token-level ensembling approach) shows that the divergence between models could be addressed by dynamically adjusting the weight of different models according to their confidence. I would suggest that the authors try these approaches.

2. For the verification step, the proposed token is only accepted when it is the most possible one for all models. However, modern LLMs typically use random sampling with topk/topp renorm for diversity. This is crucial for generating long sequences without rotting and further test-time scaling. I think expanding the acceptance threshold would be a more reasonable approach.

[1] Fu, Jiale, et al. "Fast Large Language Model Collaborative Decoding via Speculation." 

[2] Xu, Yangyifan, et al. "Hit the sweet spot! span-level ensemble for large language models."

[3] Xu, Yangyifan, et al. "Bridging the Gap between Different Vocabularies for LLM Ensemble."

### Questions
1. What is the main contribution compared to existing approaches?
2. What would be the result if greedy decoding is replaced with common sampling techniques?

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
2

### Summary
This paper introduces a new framework, SAFE, to determine when to ensemble models' next-token generation probability. The framework allows ensemble happen only when (1) no tokenization mismatch occurs across models and (2) the models’ next-token distributions show low consensus. SAFE increase model inference efficiency and accuracy on long-form generation under ensemble.

### Strengths
Disclaimer: reviewer is not an expert on the topic.

1. The paper is well-presented.
2. I think the paper's idea of using token mismatching and token confidence to decide whether to ensemble is intuitive. SAFE can be applied to any existing next-token ensemble method.
3. The paper experiments clearly show gains for efficiency and accuracy for using SAFE.

### Weaknesses
Disclaimer: reviewer is not an expert on the topic.

I'd like to see an ablation study for how each of the two criteria in SAFE impact ensemble method performance.

I will check other reviewers' comment on the weakness part of this paper.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper argues that LLM ensembling only needs to occur for a subset of tokens determined by: (1) the mismatch in tokenization across different models and (2) the level of consensus in their next-token probability distributions. The authors introduce SAFE (Stable And Fast LLM Ensembling), a framework designed to find the best tokens for ensembling during long-sequence generation. SAFE uses a speculative strategy where one "drafter" model generates a lookahead sequence of tokens, and the other "verifier" models identifies the specific token-level points within that sequence that require ensembling.

### Strengths
* The paper is well-motivated and clearly written.

* The experiments are fairly rigorous. The method achieves better quality and efficiency over baseline ensembling methods.

* A notable systems contribution is the implementation of KV caching for ensembling, which the authors also apply to baselines.

* It is an interesting finding that ensembling at a small fraction of tokens (less than 20%) can significantly improve the overall generation quality.

### Weaknesses
* The paper introduces the term "OOV-like," but this appears to be identical to the existing concept of "non-canonical" tokenization found in prior work [1, 2, 3, 4]. The paper fails to cite or discuss this relevant literature on tokenization.

* The probability sharpening strategy seems arbitrary. The choice of the 0.1 threshold is not well-justified. There are other standard ways to sharpen distributions (e.g., using a geometric mean instead of an arithmetic mean) that are not explored or compared.

* The experiments relating to efficiency could be more complete to show where exactly the speedups are coming from (see questions below).

[1] Cao, Kris, and Laura Rimell. "You should evaluate your language model on marginal likelihood over tokenisations." arXiv preprint arXiv:2109.02550 (2021).

[2] Geh, Renato Lui, et al. "Where is the signal in tokenization space?." arXiv preprint arXiv:2408.08541 (2024).

[3] Vieira, Tim, et al. "Language Models over Canonical Byte-Pair Encodings." arXiv preprint arXiv:2506.07956 (2025).

[4] Chatzi, Ivi, et al. "Canonical Autoregressive Generation." arXiv preprint arXiv:2506.06446 (2025).

### Questions
1. To confirm my understanding, is it true that if you apply SAFE ensembling between a model and itself, it will give the same result as just sampling directly from the model?  I want to confirm that ensembling does not induce temperature-scaling-like effects that boost performance.

2. The authors repeatedly claim that vocabulary alignment is a primary expense, e.g., "the number of costly ensemble operations grows with sequence length. This expense arises primarily from vocabulary alignment..." and "high computational cost of repeated vocabulary alignment makes such approaches inefficient." Can you elaborate on this? My understanding is that the primary bottleneck for ensembling is the cost of forward passes through multiple models, not vocabulary alignment. Why is alignment expensive in this setup?

3. The latency is shown to be lower, but what is the maximum memory usage (peak memory) of SAFE compared to the baselines?

4. What is the computational overhead of repeatedly calling the verifier models' tokenizers during the verify step?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a LLM Ensemble method, specifically categorized as an ensemble-during-inference method.   
The proposed method uses a main model to generate fragment-level information, and then utilizes several other models as validators to check the "potential error-prone areas", and finally integrates the outputs of multiple models in these areas.

### Strengths
The proposed generate-verify-ensemble method, in terms of its overall approach and framework design (i.e., from a high-level perspective), is reasonable and intuitive.

### Weaknesses
**Weaknesses and Suggestions.**


1) The presentation of the method and experiments is not clear enough. This version of the paper requires significant revisions to make the presentation clearer and more accessible for readers.  
Specifically: The presentation in the method section (Section 3) is not clear enough. Regarding the experiment section, for example: (i) In Section 4.2, when introducing the set of LLMs considered, mentioning the prior work proposed by Yao et al. is unnecessary and may cause confusion, as the LLM set in this paper differs from that in the prior work. (ii) The meanings of "GaC + SAFTE" and "UniTE + SAFE" are not clearly explained in the Section 4.2.

2) For the related work section, it is recommended to include an introduction to some classic works on speculative decoding, as the research in this paper is closely related to speculative decoding.

3) The main experimental results (Table 2) presented in this paper seem to show marginal performance improvement. It is recommended to include an overall average result in Table 2 to show the average performance improvement across all cases. Further experiment tuning and the design of new tricks could help improve the performance of the proposed method.

### Questions
Please refer to the above "Weaknesses and Suggestions" and the following:  
1) The 5.72% performance improvement mentioned in the introduction—where is it reflected in the tables? The explanation of this in the experiment section seems unclear.  
2) What aspect or meaning does the term "stable" in the paper's title primarily refer to? It would be helpful to explain and address this question from both the methodological and experimental result perspectives. It is recommended that the authors consider this issue in the final version of the paper. 
3) If the authors intend to emphasize efficiency, it is recommended that they provide a comparison of the overall runtime of the proposed method and the baselines on these datasets.

### Soundness
3

### Presentation
2

### Contribution
3
