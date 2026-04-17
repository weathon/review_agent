# Paraphrase-Robust Conformal Prediction for Reliable LLM Uncertainty Quantification

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Uncertainty quantification (UQ) provides interpretable measures of predictive confidence and supports reliable decision-making with large language models (LLMs). However, existing UQ methods are often neither statistically rigorous nor robust to paraphrase variations. To address these limitations, we propose a new framework for paraphrase-robust UQ, which builds on conformal prediction to ensure valid coverage and introduces a paraphrase-aware nonconformity score to enhance robustness. The score is derived by generating independent semantic paraphrases of each query, training an ancillary model that both approximates and robustifies the predictive distribution, and aggregating variability across these paraphrases.  On five general multiple-choice Question Answering (MCQA) datasets and two medical MCQA datasets with $\texttt{Qwen2.5-7B}$, our method achieves nominal coverage with compact prediction sets and demonstrates improved robustness to paraphrase shifts across different rewording settings. The results also generalize to $\texttt{Llama-3.1-8B}$ and $\texttt{Phi-3-small}$, underscoring the reliability of the framework across model families. Code is available at https://anonymous.4open.science/r/paraphrase_uq-FDD8.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a paraphrase-robust scoring function and evaluates it using existing conformal prediction frameworks on multiple-choice questions for LLMs. The framework involves training a lightweight classifier to achieve better model calibration and introducing paraphrased variations of the inputs to enhance robustness.

### Strengths
- Well-written paper with clear experimental demonstrations and ablations.
- Very important and timely problem.
- Released Code.

### Weaknesses
- The proposed approach builds entirely on existing conformal prediction frameworks without offering new theoretical insights.
- To preserve the theoretical guarantees of CP frameworks, the (X,Y) pairs must satisfy the exchangeability assumption, even under adversarial paraphrasing. This limitation weakens the contribution, as the proposed method effectively remains an application of CP frameworks when the test and calibration distributions are identical. I would expect a framework designed to maintain robustness under adversarial distribution shifts, which are typically unknown at calibration time.
 - As a suggestion: the adversarial paraphrasing for UQ methods has been discussed in early works: https://aclanthology.org/2025.acl-long.1429
- Overall, my concern is about the contribution of the paper. This paper combines existing ideas for LLMs in a convincing way, but I struggle to understand the unique perspective of the paper.

### Questions
- Do you consider any parahprashing as "adversarial"?

### Soundness
2

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
5

### Summary
This paper aims to develop a paraphrase-robust uncertainty quantification framework based on split conformal prediction, which enhances the robustness of large language models under semantic variations by incorporating paraphrase-aware nonconformity scores. Experiments show that the method achieves smaller prediction sets and better stability across multiple MCQA datasets.

### Strengths
1. The paper presents the issue that when a query has multiple expressions, even though the semantics remain consistent, it can still affect the prediction set. 

2. A probability vector predictor is trained using the hidden state, encoding and aggregating semantically consistent questions.

### Weaknesses
1. I believe that, at its core, the contribution of this paper is essentially the discovery that rephrasing, even when semantically equivalent, can impact the final prediction set size. 

2. The reference to Figure 1 in the introduction is vague. From Figure 1 alone, I cannot discern the general work of the paper, nor can I see the comparison before and after paraphrasing. Additionally, the "two popular CP scores" are not clearly demonstrated.

3. I believe that since robustness is mentioned, the boundary of paraphrasing should be identified—specifically, when a problematic paraphrase occurs but still does not affect the prediction set.

Typo: 
At the end of page five, the content should be appropriately adjusted. The citation of LLM-Uncertainty-Benchmark should not be added at the bottom of the page.

### Questions
1. The idea of PA is great. For example, when performing uncertainty decomposition, we also aggregate by rephrasing the question. Have you considered methods other than using the hidden state to train?

2. Does QCCP rely too much on "Conformal prediction with conditional guarantees"? I feel that typically, aiming for a marginal guarantee is sufficient, and there's no need to apply the conditional framework just to emphasize practical significance 

3. Even if we don't rephrase the question, keeping calibration and test sets consistent, would there be a significant difference? For example, if a question in the test set is rephrased, would it break the exchangeability with the calibration data?

### Soundness
2

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
4

### Summary
Proposes a framework for conformal prediction in LLM MCQA under paraphrases by training a proxy MLP model on embeddings. The idea is that semantically similar phrases should give similar uncertainty estimates. Experiments on several QA datasets show tighter coverage and better robustness to adversarial paraphrasing under several LLM models.

### Strengths
* Clear problem and approach
* Well written and clear presentation
* Thorough evaluation on seven datasets and modern LLM with sufficient ablation experiments
* Identifies potential failure mode of CP in LLM 
*

### Weaknesses
* Limited to QA classification tasks and not free-form generation
* Relies on embeddings and proxy model that might be unstable 
* Dependency on paraphrase generator; a weak generator could lead of over-optimistic coverage
* The adversrial paraphrasing experiments could be more quantitative to validate robustness claims. 
* Lack of runtime/cost analysis 
* Proposed method is incremental
* Little theoretical analysis why paraphrase-aggregated scores preserved CP guarantees

### Questions
* How sensitive is performance to the quality or number of paraphrases?
* How to extend to other task such as summarization?
* What is overall overhead to inference in terms of wall time? 
* How to ensure the paraphrase generator produces sufficiently diverse paraphrases that are still semantically related?
* What about using other embedding layers or different pooling aggregation such as attention weighted instead of mean pooling? 
* Can you add confidence intervals or error bars to your plots? 
* How does your method guarantee valid coverage across paraphrases? What about other forms of semantic shifts? 
* How does the choice of paraphrase generator affect robustness?
* Why is the LLM projection head miscalibrated? Why does the proxy model fix this? Did you try calibrating the final layer with ECE loss?

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
This paper addresses uncertainty quantification (UQ) for large language models (LLMs) by proposing a paraphrase-robust conformal prediction framework. The key insight is that existing conformal prediction (CP) methods for LLMs are sensitive to paraphrase variations, which can lead to unstable prediction sets. To address this, the authors introduce paraphrase-aware (PA) nonconformity scores that aggregate uncertainty across semantically equivalent rephrasings of each query. Experiments on five general MCQA datasets and two medical MCQA datasets with Qwen2.5-7B, Llama-3.1-8B, and Phi-3-small show that the method achieves nominal coverage (≈90%) with 2–4× smaller prediction sets than baselines (LAC, APS) under adversarial paraphrasing.

### Strengths
1. The paper identifies a genuine gap in existing conformal prediction for LLMs, which is lack of robustness to paraphrase variations. This is practical and relevant, as natural language queries can be expressed in many equivalent ways.
2. The use of paraphrase aggregation to achieve semantic invariance is intuitive and well-motivated. The three variants (mean, weighted, worst-case) provide a useful spectrum of robustness-efficiency trade-offs.
3. The datasets, LLM families and ablations in the experiments are extensive.
4. The method consistently achieves target coverage with substantially smaller prediction sets (often 2–4× reduction) compared to baselines, demonstrating practical value.

### Weaknesses
1. Why does paraphrase aggregation preserve/improve coverage? The paper does not provide theoretical analysis of how averaging scores across paraphrases affects the coverage guarantee. Under what conditions does this aggregation preserve the validity of conformal prediction?
2. While empirically robust, there are no formal guarantees (e.g., bounds on prediction set size variation) under paraphrase perturbations.
3. The connection between semantic invariance and statistical coverage is assumed but not rigorously established.
4. The paper does not verify that generated paraphrases truly preserve semantics. Are they evaluated by humans or using semantic similarity metrics?
5. Paraphrases are generated by an LLM (Qwen2.5-7B) and used to evaluate the same or similar LLMs. This could introduce systematic biases.
6. Generating 6 paraphrases per query increases computational cost by ≈6×. The paper does not report inference time or discuss computational efficiency.
7. The paper only compares against LAC and APS, which are relatively simple scores. Recent LLM-specific UQ methods (semantic entropy, perturbation-based methods mentioned in related work) were not considered as baselines. 
8. Since paraphrases are generated from training/calibration samples and treated as additional samples, there may be data leakage or distribution shift issues not addressed.

### Questions
1. Can you provide theoretical analysis of coverage preservation under paraphrase aggregation? Specifically, under what conditions does averaging scores across paraphrases preserve the coverage guarantees of conformal prediction?
2. Can you prove or provide bounds on the coverage gap between S_mean(x,y) and S_prob(x,y)?
3. How do you ensure paraphrase quality and semantic preservation? Have you validated that generated paraphrases preserve semantics (human evaluation, semantic similarity scores)?
4. How does paraphrase quality affect the final results?
5. What is the computational cost and inference time? Can you report wall-clock time comparisons (with and without paraphrasing)?
6. Can you provide any preliminary results on open-ended generation, factuality verification, or other tasks mentioned in the related work?
7. Can you compare with more recent/relevant baselines mentioned above?
8. How does your method compare to simple ensembling or temperature-based uncertainty?
9. Since paraphrases of calibration samples are used in training the proxy, and paraphrases of test samples are used in evaluation, is there a risk of information leakage?
10. Have you tried generating paraphrases from a completely separate LLM to avoid circular dependencies?
11. Why does the "worst" score perform poorly (Figure 7)? Intuitively, the worst-case score should be most robust to adversarial paraphrasing, but it produces the largest sets and overshoots coverage. Can you explain this counterintuitive result?

### Soundness
3

### Presentation
3

### Contribution
3
