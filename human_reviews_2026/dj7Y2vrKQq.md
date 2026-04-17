# Exploring Minimum Bayes Risk Decoding for Text-to-SQL Ensemble

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
The task of translating natural language into SQL (NL2SQL or text-to-SQL) enables users to query relational databases without requiring SQL expertise. Although recent large language model (LLM) approaches have advanced the field, achieving robust performance continues to depend on ensemble methods. Existing heuristic-based ensembles such as Minimum Bayes Risk (MBR) and Model-Based MBR (MBMBR) either ignore model-predicted probabilities or allow low-probability candidates to dominate the selection process, and they suffer from prompt sensitivity when estimating candidate likelihoods.
We propose a novel heuristic-based ensemble method that directly incorporates each candidate’s own probability into its heuristic score while mitigating prompt sensitivity through marginal probability estimation across diverse prompts. This formulation both improves traditional MBR and stabilizes probability estimation, enabling more accurate and higher-performing candidate selection without the computational overhead of supervised or prompt-based ensembles.
Extensive experiments on the SPIDER and BIRD benchmarks demonstrate that our approach consistently outperforms state-of-the-art heuristic methods, achieving higher execution accuracy across fine-tuned and pretrained LLMs. Ablation studies confirm that both the probabilistic scoring function and the marginal probability estimation independently contribute to performance gains, with the full method delivering the strongest results.
Our findings establish a new state of the art for heuristic-based ensembles in NL2SQL and highlight the broader potential of probability-aware ensemble strategies for natural language generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
LLMs have advanced in the field of text to SQL and many methods have leveraged ensembled approach. Existing heuristic based ensembles such as Minimum Bayes Risk (MBR) and Model Based MBR (MBMBR) either ignore model-predicted probabilities or allow low probability candidates to dominate the selection process, and they suffer from prompt sensitivity when estimating candidate likelihoods. They propose a novel heuristic-based ensemble method that directly incorporates each candidate’s own probability into its heuristic score while mitigating prompt sensitivity through marginal probability estimation across diverse prompts.

### Strengths
The paper identifies the limitations in the existing MBR and MBMBR, that they either ignore model predicted probabilities or allow low probability candidates to dominate the selection process, and they suffer from prompt sensitivity when estimating candidate likelihoods. They have proposed a heuristic based (these methods are efficient in comparison to supervised ensemble models and Prompt-based ensemble methods) ensemble method.

### Weaknesses
The proposed solution presents a practical refinement to the existing methods. The performance gains are consistent but relatively small. A deeper theoretical analysis could have also been helpful.

### Questions
How performance and runtime scale as the number of prompt variants and the number of generated candidates increase?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a improved version of Minimum Bayes Risk (MBR) decoding algorithm to address the text-to-SQL problem. The authors claim that the original MBR algorithm is ignorant to the candidate’s own probability, which can potentially lead to a reward hacking scenario where the candidate with highest utility but lowest likelihood is selected. The proposed method directly incorporates each candidate's own probability into its score and uses marginal probability estimation across diverse prompts to ensure stability and accuracy. The evaluation shows performance improvement on BIRD and SPIDER compared to traditional MBR and MBMBR methods.

### Strengths
- The paper is targeted on candidate selection of LLMs, which is a high-impact problem to solve. The paper is well motivated given the clear limitations from previous work.
- The overall mythology design is technically sound and reasonable.

### Weaknesses
- The absolute performance improvement in BIRD and SPIDER appears to be small, which makes the effectiveness of the proposed method on text-to-SQL (and potentially other tasks) questionable.
- It is not totally clear to the readers where the performance improvement is from.
- While being seemingly applicable to other language tasks where model ensemble is used, only text-to-SQL results are shown in this paper. Therefore, the generalizability of the methodology, while it may be exist theoretically, is not well evaluated in this paper.
- How hyperparameter is selected and how is impacts the performance is not well discussed in the paper.

### Questions
- Is it the desired behavior of MBR decoding that a low-probability candidate can be selected over a high-probability one when it has a higher expected utility due to its similarity to the other candidates in the sample pool? This seems to be the core idea of MBR to correct the shortcomings of relying solely on the model's assigned probability by seeking consensus.
- Can authors give concrete examples that in text-to-SQL benchmarks, the reward hacking described in section 3.1 actually happens and it can be mitigated by the proposed method. This brings greater confidence to readers that the performance improvement presented in Table 1 is indeed a consequence of addressing the reward hacking issue.
- The example in Section 3.1 seems to be mathematically incorrect. P(A) and P(B) cannot be 0.9 at the same time.
- How does the proposed method quantitatively compare to fine-tuned sample selectors, e.g. the ones used in CHASE-SQL or MSc-SQL? Is there any quantitative results showing the in-distribution and out-of-distribution accuracy supporting the argument on low generalizability in Section 1?
- How does the proposed method compare to other MBR extensions for LLM decoding, like [1] and [2].
- It appears that the proposed method can also be applied to other language tasks. Can authors show performance results on other tasks, like Q&A, code generation, translation, etc., so that its generalizability can also be validated?
- Can authors clarify what the statistical significance means in table 1? Does it mean its performance superiority over other methods in the same block is statistically significant with p? If so, what about the results of 14B and 32B models?

[1]. Daheim, Nico, et al. "Uncertainty-aware decoding with minimum bayes risk." *arXiv preprint arXiv:2503.05318* (2025).

[2]. Heineman, David, Yao Dou, and Wei Xu. "Improving minimum bayes risk decoding with multi-prompt." *Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing*. Vol. 2024. 2024.

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
5

### Summary
This paper addresses the task of Text-to-SQL, focusing on improving ensemble methods for candidate selection. The authors identify two key problems in existing heuristic-based ensembles like Minimum Bayes Risk (MBR) and Model-Based MBR (MBMBR): (1) they are sensitive to prompt formatting ("prompt sensitivity bias") when estimating candidate probabilities, and (2) their scoring can be dominated by low-probability candidates that are similar to others, as they don't sufficiently value a candidate's own probability. The paper proposes a novel heuristic-based method with two components to solve this. First, it introduces a new probabilistic scoring function that explicitly incorporates each candidate's own probability $P(h)$ into its utility score. Second, it proposes a "marginal probability calculation" to mitigate prompt sensitivity by estimating a candidate's probability as an average over multiple, diverse few-shot demonstrations. Experiments on the SPIDER and BIRD benchmarks show that the proposed method consistently outperforms standard baselines like voting, MBR, and MBMBR.

### Strengths
- **S1.** The paper identifies a clear and relevant set of problems with existing heuristic-based ensemble methods, namely the "prompt sensitivity bias" and the failure of standard MBR/MBMBR to properly incorporate a candidate's own probability, $P(h)$.
- **S2.** The proposed two-part solution is technically sound and well-motivated. The derivation of the new scoring function (Eq. 4), which provides a principled way to integrate $P(h)$ into the MBR framework, is a solid contribution. Furthermore, the idea of using marginal probabilities (Sec 3.2.2) to mitigate prompt sensitivity is an effective approach.
- **S3.** The experimental validation is thorough. The authors demonstrate the method's effectiveness across two standard benchmarks (BIRD, SPIDER) and, importantly, across multiple model families. The ablation study in Table 2 clearly isolates the individual contributions of the two components, showing they are complementary.

### Weaknesses
- **W1. Hyperparameter Complexity and Sensitivity:** The proposed utility function in Section 3.1.1 (Eq. 5) introduces a significant number of hard-coded, non-obvious hyperparameters (e.g., $e^{-2}$, $e^{-1}$, $\epsilon$). This design feels brittle and raises concerns about its practical applicability, as it seems to require extensive tuning. While the experiments show consistent gains, there is a strong possibility that these are the result of an "overfitted" set of hyperparameters specific to the (model, dataset) pairs tested. The paper lacks a sensitivity analysis for these hyperparameters, making it difficult to assess the method's general robustness.
- **W2. Unaddressed Cost-Benefit Trade-off:** A significant weakness of the paper is its failure to address the massive computational cost of the proposed probability estimation method (Sec 3.2.2) and its trade-off with performance. The method requires running $n$ (e.g., 5) distinct LLM calls per candidate to estimate its marginal probability. For the paper's setting of 32 candidates, this amounts to 160 LLM probability estimations to select a single answer. This is an orders-of-magnitude increase in cost compared to simple Self-Consistency (SC) Voting (which has 0 additional calls). The experimental results (Table 1) show performance gains of ~0.5-2% over these much cheaper baselines. For a real-time application like Text-to-SQL, this marginal gain seems to come at an unacceptable cost. The paper must provide a detailed analysis of this trade-off (e.g., Latency vs. Accuracy, Cost vs. Accuracy).
- **W3. Lack of Qualitative Case Studies:** The paper's motivation in Section 3.1 hinges on a key flaw of MBMBR—that it can select low-probability candidates. While an abstract example ({A, B, C}) is provided to illustrate this, the paper lacks a concrete, qualitative case study from the actual SPIDER or BIRD datasets. Showing a real example where MBMBR fails by picking an implausible-but-similar SQL query, and how the proposed scoring function (Eq. 4) corrects this error, would significantly strengthen the paper's motivation.
- **W4. Minor Presentation Error:** The illustrative example in Section 3.1 has a distracting error. The probabilities given ($P(A)=0.9$, $P(B)=0.9$, $P(C)=0$) sum to 1.8, not 1. This is a minor but sloppy mistake that should be corrected.

### Questions
- **Q1. Cost-Performance Trade-off:** My main concern is the practicality of the method. Could the authors provide a detailed trade-off analysis, comparing Latency/Cost versus Execution Accuracy for the proposed method against the cheaper baselines (SC-Voting, MBR, MBMBR)? Given the high cost of marginal probability estimation (Sec 3.2.2), how do the authors justify this trade-off for a real-time task?
- **Q2. Utility Function Motivation:** What is the theoretical or empirical motivation for the specific piecewise exponential design of the utility function in Eq. 5? Why this particular form over a simpler, linear penalty or a standard, non-piecewise function?
- **Q3. Utility Function Motivation and Sensitivity:** The utility function in Eq. 5 appears complex and relies on several hard-coded values. What is the theoretical or empirical motivation for its specific piecewise exponential design? Why were the penalties for empty/error results and results with high None values set specifically to $e^{-2}$ and $e^{-1}$, respectively? Could the authors provide a sensitivity analysis for the tunable hyperparameters in this function, namely the None threshold $\epsilon$ and the Jaccard similarity scaling parameter $\lambda$? This analysis is crucial to alleviate concerns about "overfitting" these hyperparameters to the test sets.
- **Q4. Qualitative Analysis:** To make the motivation in Section 3.1 more concrete, could the authors provide a qualitative case study from the BIRD or SPIDER dataset? Specifically, can you show a real example where MBMBR selects an incorrect, low-probability query (as per the {A, B, C} example), and demonstrate how your proposed scoring function (Eq. 4) successfully identifies the correct, higher-probability candidate?

### Soundness
3

### Presentation
3

### Contribution
3
