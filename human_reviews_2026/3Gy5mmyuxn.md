# ROC-n-reroll: How verifier imperfection affects test-time scaling

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Test-time scaling aims to improve language model performance by leveraging additional compute during inference. 
Many works have empirically studied techniques such as Best-of-N (BoN) and Rejection Sampling (RS) that make use of a verifier to enable test-time scaling. However, to date there is little theoretical understanding of how verifier *imperfection* affects performance — a gap we address in this work. Specifically, we prove that the instance-level accuracy of these methods is precisely characterized by the geometry of the verifier’s ROC curve. Our theory has two important takeaways, confirmed by experiments with Qwen and LLama models on GSM8K and MATH500. First, RS outperforms BoN for fixed compute, while both methods converge to the same accuracy in the infinite-compute limit. Second, it is generally impossible to predict the high-compute performance of either method based on observations in the low-compute regime.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a theoretical and empirical analysis of how verifier imperfections influence test-time scaling methods such as Best-of-N (BoN) and Rejection Sampling (RS). It shows that the accuracy of these methods is determined by the geometry of the verifier’s ROC curve, rather than by specific implementation details. It proves that RS outperforms BoN for fixed compute budgets and that both converge to the same accuracy as compute increases. The authors validate these findings empirically using Qwen and LLaMA models on GSM8K and MATH500, confirming that high-compute performance cannot be extrapolated from low-compute behavior.

### Strengths
1. The paper provides a theoretical understanding for resampling-based inference-time compute scaling, a timely and important topic in the era.
2. The analysis connects verifier imperfection to scaling behavior via the ROC curve. The link between ROC geometry and compute-performance scaling provides a unifying lens to reason about imperfect verifiers.
3. The paper is well-motivated, and the structure is clear.

### Weaknesses
1. The experiments only focus on two mathematical reasoning datasets (GSM8K and MATH500), where numeric ground truths simplify verification. I wonder whether the theoretical insights generalize to other reasoning datasets (e.g., commonsense reasoning) or open-ended generation (e.g., summarization, dialogue) or code generation.
2. Although the paper briefly discusses connections to RL (Appendix C.1), it doesn’t empirically test and validate these hypotheses. A comparison to small-scale RL fine-tuning (e.g., DPO, PPO) would strengthen the generalization of this argument.
3. The empirical performance curves in Figure 1 are only shown for a hand-picked example (test question 58).
4. The paper provides no actionable method to improve verifiers or scaling strategies — it only diagnoses why scaling may fail. Especially, one of the paper's findings in the abstract is that it is "generally impossible to predict the high-compute performance" from low-compute observations, which is a negative result without constructive guidance on how to solve this problem.

### Questions
See in weaknesses.

### Soundness
2

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
This work investigates the performance of two LLM test-time scaling methods in scenarios with an imperfect verifier:  
- Rejection Sampling (RS): generates responses sequentially until one passes verification.  
- Best-of-N (BoN): generates N responses in parallel and selects the one with the highest verification score.  

Within a formal statistical framework, the authors rigorously show that the accuracy of either method for solving a single query under a fixed compute budget is determined precisely by the ROC curve corresponding to the generator and verifier.
  
High-level takeaways from the analysis include:  
1. RS outperforms BoN under fixed compute (assuming the ROC curve is concave), although both converge to the same accuracy when compute is unlimited.  
2. For either method, high-compute accuracy cannot be reliably predicted from low-compute performance.  

Theoretical findings are validated through experiments, lending further support to the results.

### Strengths
This paper presents a solid and well-executed contribution to the LLM research community, even though it does not introduce new algorithms. The theoretical analysis of RS and BoN is thorough and rigorous, providing fresh insights into their behaviors when given *imperfect verifiers*. The work offers practical takeaways for researchers and practitioners considering repeated sampling approaches.  

*(Disclaimer: Some technical content extends beyond my core expertise, and thus I have not verified the correctness of all proofs in detail.)*

### Weaknesses
I don't see major weaknesses in this work, assuming that the theoretical analyses are correct.


Some suggestions:

- The takeaway message "RS outperforms BoN under fixed compute" requires the assumption that the ROC curve (for the particular query under consideration) is concave, according to Proposition 5. I think this assumption should be stated more explicitly in the abstract and introduction, otherwise the takeaway message alone could be slightly misleading. 
In reality, there is no guarantee *a priori* that the ROC curve should be concave.

- The proof for BoN's limit accuracy in Theorem 1 (which equals that of RS in Proposition 2) is quite complicated.
On the other hand, we might think of (or approximate) BoN in the infinite-compute limit as (1) generating many samples, (2) keeping the samples whose scores exceed $1 - \delta$ (with $\delta \rightarrow 0$), (3) breaking ties and picking one of the high-score samples randomly.
In this case, BoN's accuracy is effectively the posterior probability that a generated sample is correct, conditioned on its score exceeding $1 - \delta$.
This probability can be easily derived by the Bayes rule, and doesn't fundamentally differ from the analysis for RS.
I wonder how far from rigor this analysis is, and whether it could offer some intuitions about Theorem 1 and the takeaway message that BoN and RS reach the same accuracy with infinite samples.

- Use a notation other than $\pi$ (Line 137) to avoid confusion with the mathematical constant $\pi \approx 3.14$.  

- Line 373, "Comparing with Proposition 1": should be "Proposition 2".

- Include and discuss more related works, which could situate this paper more clearly in the literature on LLM test-time scaling.
Examples include [1, 2, 3] that provide theoretical analyses for different test-time scaling methods, and [4] that also investigates the limits of LLM repeated sampling when given imperfect verifiers (albeit more empirically).

[1] Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems (NeurIPS 2024)

[2] Provable Scaling Laws for the Test-Time Compute of Large Language Models (NeurIPS 2025)

[3] When More is Less: Understanding Chain-of-Thought Length in LLMs (ICLR 2025 Workshop)

[4] Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers (arXiv 2024)

### Questions
See "weaknesses" above.

### Soundness
4

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
2

### Summary
This paper studies rejection sampling and best-of-N sampling for LLMs with imperfect verifiers. The core idea is that, for a fixed sampling budget, the performance of both methods is determined by the verifier's ROC curve and the generator's base accuracy. The authors show this theoretically and provide empirical validation on math datasets.

### Strengths
1. Interesting problem, clean formalization, and clear scope that focuses on RS and BoN.
2. The result that RS outperforms BoN with fixed average compute seems useful in practice.
3. Empirical experiments show consistent results with theory's prediction.

### Weaknesses
1. **Limited evaluation domain**: Can the authors also consider other benchmark categories such as coding or more general QA? This is critical for evaluation whether the conclusion generalizes.
2. **Compute metric mismatch and missing hybrid method**
	- The compute metric for RS is defined as an expectation, while the compute metric for BoN is defined as the deterministic N. The stopping time of  RS creates variance but the analysis optimize only the mean. Also, the experiments on MATH500 and GSM8K has a cap at 25 (line 448), which changes the RS distribution in a way not covered by the theory.
	- Taking a step further, it feels natural to consider a hybrid approach where BoN early-stops if a certain score threshold is reached (or RS early stops if a certain N is reached). Although the authors mention this as future work, I believe that this is a baseline much more realistic than RS without stopping.

### Questions
- Can you verify the results on more diverse benchmarks?
- How will a hybrid method change your conclusions? Will this hybrid method outperform both RS and BoN? Could you test it empirically?

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
This paper presents a timely study on how imperfections in a verifier affect the performance of test-time scaling, where the goal is to improve accuracy by spending more compute at inference time. The authors analyze two commonly used techniques: rejection sampling (RS) and Best-of-N (BoN). They show that the performance of both methods is fully determined by the geometry of the verifier’s ROC curve. For RS, they prove that performance in the low-compute and high-compute regimes depends on the local geometry of the ROC curve — specifically, how the ROC behaves near the operating points corresponding to large and small false-positive rates (FPR). In the high-compute regime, accuracy is governed by the slope of the ROC near FPR $\to$ 0. For BoN, performance depends on global ROC properties.

Their theoretical results, supported by experiments, show that RS is more compute-efficient than BoN in the finite compute regime, while both methods converge to the same asymptotic accuracy as compute goes to infinity. A key insight is that test-time scaling cannot be extrapolated: low-compute performance does not predict high-compute performance unless one knows the ROC geometry.

### Strengths
- One of the main strengths of this paper is that the authors present a simple and interpretable mathematical framework for analyzing how verifier imperfections impact the effectiveness of test-time scaling. Their formulation leverages classical concepts from statistical learning theory (e.g., ROC curves, true/false positive rates), providing a clean lens through which to reason about the verifier. This contribution is valuable because it connects modern LLM sampling practices with well-established theoretical tools, which may inspire clearer definitions and analysis in future research.

- Another strength is that the theoretical predictions align well with empirical results. For example, in Figure 3, the experiments on LLaMA models show the expected behavior: RS outperforms BoN under finite compute budgets, consistent with the theoretical claim that RS is more compute-efficient.

- Overall, the paper offers practical guidance on how to deploy test-time scaling methods when compute is constrained. It clarifies when additional sampling is worthwhile, and when verifier limitations make further compute ineffective.

### Weaknesses
- One weakness of this paper is its clarity / organization. I think the clarity can be significantly improved by including a table of notations in the main manuscript. On the first pass, it was difficult to follow the derivations because I had to look back to recall the notation.

- The theory models correctness as a binary variable (either correct or incorrect). This abstraction simplifies analysis, but it may limit applicability in settings where output quality is continuous or graded, e.g., in RL where partial correctness matters. It is unclear how the theoretical results extend to cases with non-binary reward structures or scalar score models, which are common in modern LLM evaluation and verification frameworks.

### Questions
Here are both questions and comments:

- This question might be a bit more abstract: the weakness stated above leads me to wonder whether compute could be defined differently in this setting. In the paper, compute is measured as the total number of generations. However, one could instead measure compute as the total number of tokens generated. If compute were defined in terms of token usage rather than completions, the relative advantage of RS could change. In particular, BoN may appear more favorable under a token-based compute budget, since it generates a fixed number of completions of comparable length, whereas RS may generate highly variable amounts of text before accepting a sample. Have the authors thought about these directions, if this is not sound I am happy to discuss.

- There is an iid assumption on the samples generated for BoN; is this pivotal in the proofs? I'm not sure this can be met in practice so it leaves me wondering if this is a drawback at all, or it doesn't matter and is just a definition.

-  I guess $\tau$ should be in [0, 1] in line 141

- It feels confusing to me to say "let the score $f$ to be fixed"; this sounds like the score is fixed and not the verifier. Perhaps calling it the verifier as done in line 301 is better.

- Does the theory depend at all if $h_F$ is not unique, i.e., there are multiple solutions to Equation (1)?

### Soundness
3

### Presentation
2

### Contribution
3
