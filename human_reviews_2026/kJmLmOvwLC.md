# SAFER: Risk-Constrained Sample-then-Filter in Large Language Models

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
As large language models (LLMs) are increasingly deployed in risk-sensitive applications such as real-world open-ended question answering (QA), ensuring the trustworthiness of their outputs has become critical. Existing selective conformal prediction (SCP) methods provide statistical guarantees by constructing prediction sets with a constrained miscoverage rate for correct answers. However, prior works unrealistically assume that admissible answers for all instances can be obtained via finite sampling, even for open-ended QA scenarios that lack a fixed and finite solution space. To address this, we introduce a two-stage risk control framework comprising abstention-aware **SA**mpling and conformalized **F**ilt**ER**ing (SAFER). Firstly, on a held-out calibration set, SAFER calibrates a sampling budget within the maximum sampling cap, using the Clopper–Pearson exact method at a user-desired risk level (i.e., the maximum allowable miscoverage rate of the sampling sets). If the risk level cannot be satisfied within the cap, we abstain; otherwise, the calibrated sampling budget becomes the minimum requirements at test time. Then, we employ calibration instances where correct answers are attainable under the calibrated budget and apply the conformal risk control method to determine a statistically valid uncertainty threshold, which filters unreliable distractors from the candidate set for each test data point. In this stage, SAFER introduces an additional risk level to guide the calculation of the threshold, thereby controlling the risk of correct answers being excluded. We evaluate SAFER on three free-form QA datasets utilizing five popular LLMs, and demonstrate that it rigorously constrains two-stage miscoverage risks at test time. 
Furthermore, we show that SAFER is compatible with various task-specific admission criteria and calibration-test split ratios, highlighting its robustness and high data efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposers Sampling and conformalized filtering  that use conformal risk control into two stage framework. The first stage calibrates sampling budget for responses candidate set to contain admissible answers at miscoverage rate $\alpha$. The second stage introduces second risk level $\beta$ to control probability of incorrectly filtering out of all admissible responses to produce a prediction set. The final prediction set fails to cover an admissible answer with probability $1-delta$ over the calibration set with bound $\alpha + \beta - \alpha\beta$. Experiments on three QA datasets (TriviaQA, CoQA, ScienceQA) using five different LLMs demonstrate that SAFER validly controls the empirical error rate below the specified bounds.

### Strengths
* The abstention aware sampling is mechanism is the main novelty. Should clarify and compare with Yadkori 2024. Is this just an application/extension of existing works?
* The bound on the sampling risk looks to be formally derived and empirically validated across several LLMs and datasets. 
* The evaluation of different correctness criteria (sentence similarity, Rouge-L, NLI-based bi-entailment, and LLM-based semantic evaluation) add empirical support

### Weaknesses
* The paper is incremental in terms of technical contribution, combining existing LTT and conformal ideas with abstention-aware calibration.
* The calibration for Stage 1 requires generating/evaluating $N\timesM$ samples and evaluating these responses against the ground truth could be very computationally expensive.
* The maximum sampling cap M is a key hyperparameter that is not explored in detail.
* Although many related CP works are cited, the experiments ignores them completely and only compares against single TRON baseline.
* Only QA datasets are used, no evaluation on reasoning, summarization, or multi-modal tasks

### Questions
* Is stage I just straightfoward application of LTT and stage II just application of Conformal Risk Control? Can you clarify core contributions beyond just direct application of existing frameworks?
* Is there bias in sequential calibration when stage II calibration is only performed on biased subset of admissible samples from stage I? This seems like form of conditional coverage and is not discussed. How do you justify that this biased selection step does not break exchangeability?
* How to choose $M$ and why are ablation experiments not conducted on such a critical hyperparameter?
* How to extend framework to black-box API-only models without logit access?
* How does the quality of the uncertainty heuristic $U(u)$ affect the efficiency of the final prediction set size? 
* Can the 3D plots be visualized in a better way? 
* Can the sampling budget be made adaptive to improve efficiency based on example difficulty or some other criteria?
* Can the two-stage  $\alpha + \beta - \alpha\beta$ be made tighter? 
* Could the calibration of $\alpha$ and $\beta$ be done jointly rather than separately? 
* How does Clopper-Pearson (classical statistical tool) compare to alternative calibration approaches to find confidence intervals?
* How does this framework extend to when admissibility is fuzzy or difficult to measure exactly? 
* The framework requires the user to specify two risk levels, how should a user practically choose $\alpha$ and $\beta$ to target a specific overall risk level $\epsilon$ between sampling and filtering?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SAFER, a two-stage framework for uncertainty quantification in open-ended question answering with LLMs. The framework consists of: (1) abstention-aware sampling that calibrates a minimum sampling budget using the Clopper-Pearson exact method, abstaining when risk constraints cannot be satisfied, and (2) conformalized filtering that removes unreliable candidates using conformal risk control. The authors provide theoretical guarantees that miscoverage probability is bounded by α + β - αβ and validate the approach on three QA datasets with five LLMs.

### Strengths
1. Unlike prior work (TRON, ConU), SAFER doesn't assume all instances can yield correct answers within finite sampling, incorporating a principled abstention mechanism that makes the framework more realistic for deployment.
2. The paper provides formal statistical guarantees with detailed proofs, properly applying the Clopper-Pearson exact method for finite-sample bounds and extending conformal risk control to the filtering stage.
3. The experimental validation is comprehensive
4. The filtering stage substantially reduces prediction set sizes (e.g., 7.9 to 5.5 for β=0.1 on TriviaQA) while maintaining statistical validity, improving practical usability.

### Weaknesses
1. The main components (Clopper-Pearson bounds, conformal risk control) are existing techniques. The contribution is primarily in their combination with abstention handling rather than fundamental methodological innovation.
2. Relies on exchangeability between calibration and test sets with no empirical evaluation of robustness to distribution shift
3. Requires users to specify both α (sampling risk) and β (filtering risk). The combined bound α + β - αβ is less interpretable than a single risk parameter
4. No principled guidance provided for parameter selection in practice
5. What fraction of instances require abstention? This is crucial for assessing practical utility
6. No wall-clock time comparisons or discussion of overhead
7. Only compares against TRON; missing comparisons with ConU, LofreeCP, SE-SCP, and other recent conformal prediction methods for open-ended QA.

### Questions
1.What are the abstention rates across different datasets, models, and risk levels? How does abstention correlate with model capability or question difficulty?
2. What is the wall-clock time cost compared to baseline sampling approaches? How does this scale with M and ŝ?
3. Can you provide practical guidance or heuristics for selecting α and β for different applications? What are reasonable default values?
4. How does SAFER perform when calibration and test distributions differ (e.g., temporal shift, domain shift)? This seems critical for real deployment.
5. How does SAFER compare quantitatively to ConU, LofreeCP, and other recent methods on the same datasets?
6. Have you evaluated alternative uncertainty measures? How much does performance depend on this choice?
7. How does the framework handle questions with multiple semantically distinct but correct answers?

### Soundness
3

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
3

### Summary
This paper introduces SAFER, a two-stage framework for controlling the risk of miscoverage in open-ended question answering (QA) with Large Language Models (LLMs).  The core contribution is a practical and theoretically-grounded method for providing rigorous coverage guarantees for LLM outputs in open-ended QA, particularly by formally integrating an abstention option.

### Strengths
One of the main strengths of this work is its sound technical approach to a challenging problem. The formulation of a two-stage process that independently controls risk at both the sampling and filtering stages is elegant. The introduction of an explicit, statistically-grounded abstention mechanism is a significant step forward from prior methods that often assume an admissible answer is always attainable. This makes the framework more practical for real-world, risk-sensitive applications. The authors validate their method with experiments across multiple datasets, LLMs, and several different correctness evaluation metrics, demonstrating the robustness and data-efficiency of their framework.

### Weaknesses
A primary concern is the clarity of the novelty. While the application of abstention to this specific two-stage QA framework is new, the paper could better situate its contribution within the broader literature on conformal prediction and abstention, where similar concepts have been explored. The presentation is quite dense and assumes a high level of familiarity with conformal prediction, which may limit its accessibility to a wider audience. Furthermore, while the experiments are quite broad, they could be more rigorous. The analysis focuses on demonstrating that the empirical error rate is below the theoretical bounds, but a deeper analysis of the abstention cases or a more direct comparison with a wider set of non-conformal baselines would strengthen the claims of practical utility.

### Questions
Questions for the authors:

The abstention mechanism is a key component. Could you provide more insight into the practical trade-offs involved in setting the maximum sampling cap (M)? How sensitive is the framework's performance to this hyperparameter, and what is the relationship between M, the risk level ω, and the resulting calibrated sample size?
The filtering stage relies on an uncertainty measure U, which is sentence entropy in this work. How dependent is SAFER's performance on the quality of this uncertainty measure? Have you explored how the two-stage guarantees hold up when using other uncertainty heuristics, especially those that might be less correlated with correctness?

### Soundness
3

### Presentation
2

### Contribution
2
